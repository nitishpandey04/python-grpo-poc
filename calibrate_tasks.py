"""Sample the original model on training tasks only; freeze mixed-outcome tasks."""
import argparse
import collections
import json
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TRACKIO_DIR',str(ROOT/'runs/trackio'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

from execution import Sandbox
from monitoring import Monitor
from task_selection import assert_disjoint,calibration_bucket,expanded_score,read_rows,sha


def main(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
    if args.output.exists() or args.dataset.exists():
        raise ValueError('Calibration output/dataset already exists; choose new paths')
    quality=json.loads((args.quality/'manifest.json').read_text())
    if args.protocol=='assert_suite' and quality.get('protocol')!='opencoder-screened-v1':
        raise ValueError('Assertion-suite calibration requires an OpenCoder screening manifest')
    if args.protocol=='mbpp_expanded' and quality.get('protocol')=='opencoder-screened-v1':
        raise ValueError('OpenCoder tasks require --protocol assert_suite')
    scorer=expanded_score
    if args.protocol=='assert_suite':
        from opencoder_data import assertion_suite_score
        scorer=assertion_suite_score
    rows=read_rows(args.quality/'train.jsonl')
    if not quality.get('counts_after') or not rows:
        raise ValueError('Reference verification is incomplete or no training tasks passed')
    for split,digest in quality['files_sha256'].items():
        if sha(args.quality/f'{split}.jsonl')!=digest:
            raise ValueError('Verified data changed')
    random.Random(args.seed).shuffle(rows)
    config={"purpose":"train-only difficulty calibration",**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},"quality_manifest_sha256":sha(args.quality/'manifest.json'),"selection_rule":"0 < full_pass_candidates < samples; no validation/test outcomes used"}
    args.output.mkdir(parents=True)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    monitor=Monitor(args.output,args.output.name,config)
    selected=[];records=[]
    try:
        set_seed(args.seed);torch.set_num_threads(4)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        monitor.phase='loading original model for calibration'
        tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,local_files_only=True)
        tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        with Sandbox() as sandbox:
            for i,row in enumerate(rows):
                monitor.phase=f'calibration generating {i+1}/{len(rows)}'
                text=tokenizer.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True)
                inputs=tokenizer(text,return_tensors='pt').to(model.device)
                with torch.inference_mode():
                    outputs=model.generate(**inputs,num_return_sequences=args.samples,max_new_tokens=args.max_tokens,do_sample=True,temperature=args.temperature,top_p=1.0,top_k=0,repetition_penalty=1.0,pad_token_id=tokenizer.pad_token_id)
                candidates=[]
                monitor.phase=f'calibration scoring {i+1}/{len(rows)}'
                for j,output in enumerate(outputs):
                    completion=tokenizer.decode(output[inputs.input_ids.shape[1]:],skip_special_tokens=True)
                    score=scorer(sandbox,completion,row['tests'],row['task_id'])
                    candidate={'task_id':row['task_id'],'candidate':j,'completion':completion,'reward':float(score['correct']),**score}
                    candidates.append(candidate);monitor.sample(candidate)
                correct=sum(c['correct'] for c in candidates)
                bucket=calibration_bucket(correct,args.samples)
                record={'task_id':row['task_id'],'correct':correct,'samples':args.samples,'bucket':bucket,'candidates':candidates}
                records.append(record)
                with (args.output/'tasks.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
                if bucket=='mixed':selected.append(row)
                counts=collections.Counter(r['bucket'] for r in records)
                monitor.log({'calibration/tasks_completed':i+1,'calibration/tasks_total':len(rows),'calibration/percent':100*(i+1)/len(rows),'calibration/selected_mixed':len(selected),'calibration/hard':counts['hard'],'calibration/easy':counts['easy'],'calibration/current_full_pass_fraction':correct/args.samples})
                print(f"[calibration] {i+1}/{len(rows)} task={row['task_id']} passes={correct}/{args.samples} {bucket}; selected={len(selected)}",flush=True)
        counts=dict(collections.Counter(r['bucket'] for r in records))
        (args.output/'summary.json').write_text(json.dumps({('screened_training_tasks' if args.protocol=='assert_suite' else 'verified_training_tasks'):len(rows),'selected_tasks':len(selected),'buckets':counts,'selected_quality_tiers':dict(collections.Counter(r.get('quality_tier','mbpp_expanded') for r in selected)),'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2},indent=2))
        if len(selected)<4 and args.protocol=='mbpp_expanded':raise RuntimeError('Fewer than four mixed-outcome tasks; do not start broad-batch training')
        validation=read_rows(args.quality/'validation.jsonl');confirmation=read_rows(args.quality/'confirmation.jsonl')
        assert_disjoint(selected,validation,confirmation)
        args.dataset.mkdir(parents=True)
        for split,items in [('train',selected),('validation',validation),('confirmation',confirmation)]:
            (args.dataset/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in items))
        manifest={**config,'protocol':('opencoder-selected-v1' if args.protocol=='assert_suite' else 'selected-grpo-v2'),'counts':{'train':len(selected),('training_screened' if args.protocol=='assert_suite' else 'training_verified'):len(rows),'validation':len(validation),'confirmation':len(confirmation)},'buckets':counts,'image_id':quality['image_id'],'files_sha256':{split:sha(args.dataset/f'{split}.jsonl') for split in ['train','validation','confirmation']},'task_ids':{'train':[r['task_id'] for r in selected],'validation':[r['task_id'] for r in validation],'confirmation':[r['task_id'] for r in confirmation]},'confirmation_predictions_generated':False}
        if args.protocol=='assert_suite':
            ready=[r for r in selected if r['quality_tier']=='independent_contract_checked']
            (args.dataset/'train-independent.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ready))
            manifest.update(selected_quality_tiers=dict(collections.Counter(r['quality_tier'] for r in selected)),independent_selected=len(ready),screened_only_requires_semantic_review=True,training_adapter_required=True,independent_file_sha256=sha(args.dataset/'train-independent.jsonl'))
        (args.dataset/'manifest.json').write_text(json.dumps(manifest,indent=2))
        monitor.phase='complete'
        print(json.dumps(manifest,indent=2),flush=True)
    except BaseException:
        monitor.phase='failed or interrupted';raise
    finally:
        monitor.close()
        (args.output/'status.json').write_text(json.dumps({'phase':monitor.phase,'updated_at':time.time(),'tasks_completed':len(records)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',choices=['mbpp_expanded','assert_suite'],default='mbpp_expanded')
    parser.add_argument('--quality',type=Path,default=Path('runs/task-quality-v2'))
    parser.add_argument('--output',type=Path,default=Path('runs/calibration-v2'))
    parser.add_argument('--dataset',type=Path,default=Path('data/selected-v2'))
    parser.add_argument('--model',default='Qwen/Qwen2.5-Coder-0.5B-Instruct')
    parser.add_argument('--revision',default='ea3f2471cf1b1f0db85067f1ef93848e38e88c25')
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--temperature',type=float,default=0.8)
    parser.add_argument('--max-tokens',type=int,default=256)
    parser.add_argument('--seed',type=int,default=42)
    main(parser.parse_args())
