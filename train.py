"""One local Python GRPO path, with baseline and sampled-reward checks."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('TRACKIO_DIR',str(ROOT/'runs/trackio'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/'.cache/triton'))
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'.cache/torchinductor'))
os.environ.setdefault('PYTORCH_ALLOC_CONF','expandable_segments:True')
headers=ROOT/'.cache/python-headers/usr/include'
if headers.exists():os.environ['C_INCLUDE_PATH']=os.pathsep.join([str(headers/'python3.12'),str(headers),os.environ.get('C_INCLUDE_PATH','')])

from evaluate import compare,evaluate
from execution import Sandbox,solution_score
from monitoring import Monitor
from evaluate import expanded_score,read_rows,sha
from report import build_report

CONCISE_CODE_INSTRUCTION='Do not include comments or reasoning. Keep the solution concise and complete.'


def concise_code_rows(rows):
    result=[]
    for row in rows:
        if row['prompt'][0]['role']!='system':raise ValueError('Concise-code override requires a system message')
        prompt=[dict(message) for message in row['prompt']]
        prompt[0]['content']+=' '+CONCISE_CODE_INSTRUCTION
        result.append({**row,'prompt':prompt})
    return result


def validate_generation_groups(ids, prompts, generations):
    if len(ids) != prompts * generations:
        raise ValueError("Incomplete generation batch")
    groups = [ids[i:i+generations] for i in range(0,len(ids),generations)]
    if any(len(set(group)) != 1 for group in groups) or len({group[0] for group in groups}) != prompts:
        raise ValueError("Generation batch must have distinct prompt groups with exactly G candidates each")
    return {"distinct_prompts": prompts, "candidates_per_prompt": generations, "candidates": len(ids)}


def batch_settings(prompts,generations,micro_batch):
    if min(prompts,micro_batch)<1 or generations<2:
        raise ValueError('Need positive prompt/micro-batch counts and at least two generations')
    total=prompts*generations
    if total % micro_batch:
        raise ValueError('Micro-batch must divide the generation batch exactly')
    return {'per_device_train_batch_size':micro_batch,'gradient_accumulation_steps':total//micro_batch,'generation_batch_size':total,'num_generations':generations}




def resolve_model(manifest,model=None,revision=None):
    if (model is None)!=(revision is None):raise ValueError('Model overrides require both --model and --model-revision')
    return (model,revision) if model is not None else (manifest['model'],manifest['revision'])


def select_probe_rows(rows,count,seed):
    """Fixed balanced sample from training only, without observing model answers."""
    if not 0<count<=len(rows):raise ValueError('Probe count must fit the training pool')
    groups={label:sorted([r for r in rows if r['difficulty']==label],key=lambda r:hashlib.sha256(f"{seed}/{r['task_id']}".encode()).hexdigest()) for label in sorted({r['difficulty'] for r in rows})}
    selected=[]
    for index in range(max(map(len,groups.values()))):
        for group in groups.values():
            if index<len(group):selected.append(group[index])
            if len(selected)==count:return selected


def summarize_probe(groups,generations):
    import collections
    if not groups or len({r['task_id'] for r in groups})!=len(groups):raise ValueError('Empty or duplicate probe groups')
    if any(len(r['rewards'])!=generations or any(v not in {0,1} for v in r['rewards']) for r in groups):raise ValueError('Incomplete or non-binary probe group')
    def metrics(rows):
        correct=sum(sum(r['rewards']) for r in rows);n=len(rows)
        return {'problems':n,'candidates':n*generations,'correct_candidates':correct,'full_pass_fraction':correct/(n*generations),
            'all_fail_groups':sum(not any(r['rewards']) for r in rows),'mixed_reward_groups':sum(0<sum(r['rewards'])<generations for r in rows),
            'all_pass_groups':sum(all(r['rewards']) for r in rows),'solved_at_least_once':sum(any(r['rewards']) for r in rows),
            'token_cap_hits':sum(r['token_cap_hits'] for r in rows),'failure_counts':dict(collections.Counter(status for r in rows for status in r['statuses']))}
    return {'overall':metrics(groups),'by_difficulty':{label:metrics([r for r in groups if r['difficulty']==label]) for label in sorted({r['difficulty'] for r in groups})}}


def probe_rewards(model,tokenizer,rows,sandbox,args,monitor):
    import torch
    from transformers import GenerationConfig
    # Match the installed GRPO trainer's EOS union and sampling defaults.
    eos_ids=model.generation_config.eos_token_id
    eos_ids=[] if eos_ids is None else [eos_ids] if isinstance(eos_ids,int) else list(eos_ids)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:eos_ids.insert(0,tokenizer.eos_token_id)
    generation=GenerationConfig(max_new_tokens=args.max_tokens,do_sample=True,temperature=args.temperature,top_p=1.0,top_k=0,min_p=None,repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.bos_token_id,eos_token_id=eos_ids or None,disable_compile=True)
    (args.output/'probe-generation.json').write_text(generation.to_json_string())
    model.eval();groups=[];started=time.monotonic()
    for index,row in enumerate(rows):
        monitor.phase=f"probe: generating {index+1}/{len(rows)} ({row['difficulty']})"
        text=tokenizer.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True)
        inputs=tokenizer(text,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            tokens=model.generate(**inputs,generation_config=generation,num_return_sequences=args.generations,use_cache=True)
        rewards=[];statuses=[];cap_hits=0
        for sample,sequence in enumerate(tokens):
            monitor.phase=f'probe: testing problem {index+1}/{len(rows)}, candidate {sample+1}/{args.generations}'
            generated=sequence[inputs.input_ids.shape[1]:].tolist()
            eos=next((i for i,t in enumerate(generated) if t in eos_ids),None)
            token_count=len(generated) if eos is None else eos+1
            cap_hit=eos is None and token_count>=args.max_tokens;cap_hits+=cap_hit
            completion=tokenizer.decode(generated,skip_special_tokens=True)
            score=solution_score(sandbox,completion,row['tests'],row['task_id'])
            reward=int(score['correct']);rewards.append(reward);statuses.append(score['status'])
            monitor.sample({'task_id':row['task_id'],'difficulty':row['difficulty'],'rating':row['rating'],'sample_index':sample,'completion':completion,'reward':reward,'tokens':token_count,'token_cap_hit':cap_hit,**score})
        group={'task_id':row['task_id'],'difficulty':row['difficulty'],'rating':row['rating'],'rewards':rewards,'statuses':statuses,'token_cap_hits':cap_hits}
        groups.append(group)
        with (args.output/'probe-groups.jsonl').open('a') as handle:handle.write(json.dumps(group)+'\n')
        summary=summarize_probe(groups,args.generations)
        eta=(time.monotonic()-started)/(index+1)*(len(rows)-index-1)
        monitor.log({f'probe/{k}':v for k,v in summary['overall'].items() if isinstance(v,(int,float))})
        monitor.log({'probe/completed_problems':index+1,'probe/total_problems':len(rows),'probe/percent':100*(index+1)/len(rows),'probe/eta_seconds':eta})
        for label,values in summary['by_difficulty'].items():monitor.log({f'probe/{label}/{k}':v for k,v in values.items() if isinstance(v,(int,float))})
        print(f"[probe] {index+1}/{len(rows)}; {sum(rewards)}/{args.generations} pass; mixed groups {summary['overall']['mixed_reward_groups']}; ETA {eta/60:.1f} min",flush=True)
    return {**summarize_probe(groups,args.generations),'elapsed_seconds':time.monotonic()-started}


def main(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM,AutoTokenizer,TrainerCallback,set_seed
    from trl import GRPOConfig,GRPOTrainer

    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('This pilot is configured for one GPU')
    if args.reward_probe and args.temperature<=0:raise ValueError('Sampling temperature must be positive')
    if args.steps<1 or args.eval_every<1:raise ValueError('Steps and evaluation interval must be positive')
    if args.output is None:args.output=ROOT/'runs'/args.name
    batches=batch_settings(args.prompts_per_update,args.generations,args.micro_batch)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    if manifest['protocol'] not in {'python-program-grpo-v1','python-grpo-v1'}:raise ValueError('Prepare native Python programs with prepare_data.py first')
    for split,digest in manifest['files_sha256'].items():
        if sha(args.dataset/f'{split}.jsonl')!=digest:raise ValueError('Frozen selected data changed')
    train_rows=read_rows(args.dataset/'train.jsonl');val_rows=read_rows(args.dataset/'validation.jsonl')
    model_id,model_revision=resolve_model(manifest,args.model,args.model_revision)
    stdio=manifest.get('execution_mode')=='stdio'
    if stdio and any(r.get('language')!='python' for r in train_rows+val_rows):raise ValueError('This run accepts Python-only data')
    if args.concise_code:
        if not stdio:raise ValueError('Concise-code comparison is for the Python standard-I/O dataset')
        train_rows=concise_code_rows(train_rows);val_rows=concise_code_rows(val_rows)
    if args.max_tokens is None:args.max_tokens=manifest.get('max_completion_tokens',256)
    evaluation_scorer=solution_score if stdio else expanded_score
    # Only IDs are used to guard the sealed confirmation pool. No confirmation prediction is generated.
    train_ids={r['task_id'] for r in train_rows};val_ids={r['task_id'] for r in val_rows};confirmation_ids=set(manifest['confirmation']['task_ids'])
    if train_ids&val_ids or train_ids&confirmation_ids or val_ids&confirmation_ids:raise ValueError('Data leakage')
    if len(train_rows)<args.prompts_per_update or not val_rows:raise ValueError('Insufficient training or development problems')
    if (args.output/'config.json').exists():raise ValueError('Run exists; choose a new name')
    if sha(Path(manifest['evaluation_manifest']))!=manifest['evaluation_manifest_sha256']:raise ValueError('Fixed evaluation metadata changed')
    args.output.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    set_seed(args.seed);torch.set_num_threads(4)
    selected=select_probe_rows(train_rows,args.probe_problems,args.seed) if args.reward_probe else None
    if selected is not None:
        selection={'split':'train','seed':args.seed,'rule':'Hash-order within difficulty, round-robin equal allocation, independent of model outcomes. Balanced probe is not a population-weighted score.','problems':[{k:r[k] for k in ['task_id','difficulty','rating']} for r in selected]}
        (args.output/'probe-selection.json').write_text(json.dumps(selection,indent=2))
    run_stage='short pipeline smoke test' if args.steps<=10 else 'bounded GRPO experiment'
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'model':model_id,'model_revision':model_revision,'precision':'nf4-bfloat16-compute' if args.load_in_4bit else 'bfloat16','cuda_total_memory_mb':torch.cuda.get_device_properties(0).total_memory/1024**2,'stage':'reward-probe' if args.reward_probe else 'baseline-only' if args.baseline_only else run_stage,'batch_settings':batches,'dataset_manifest_sha256':sha(args.dataset/'manifest.json'),'reward':manifest['training_reward'],'development_protocol':manifest['development_protocol'],'execution_image':manifest.get('execution_image','python-grpo-cpython:v3'),'execution_image_id':subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',manifest.get('execution_image','python-grpo-cpython:v3')],text=True).strip(),'source_sha256':{name:sha(ROOT/name) for name in ['train.py','prepare_data.py','evaluate.py','execution.py','cpython_worker.py','monitoring.py']},'confirmation_evaluated':False,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
    if selected is not None:config['probe_selection_sha256']=sha(args.output/'probe-selection.json')
    config['prompt_override_instruction']=CONCISE_CODE_INSTRUCTION if args.concise_code else None
    config['effective_system_prompts']=sorted({row['prompt'][0]['content'] for row in train_rows+val_rows})
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    monitor=Monitor(args.output,args.output.name,config)
    try:
        monitor.phase='loading original model'
        tokenizer=AutoTokenizer.from_pretrained(model_id,revision=model_revision,local_files_only=True)
        tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        load_kwargs={'revision':model_revision,'local_files_only':True,'dtype':torch.bfloat16,'attn_implementation':'sdpa'}
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0})
        model=AutoModelForCausalLM.from_pretrained(model_id,**load_kwargs)
        # GRPOTrainer creates/freezes PEFT adapters and enables checkpointing inputs.
        # Keep non-quantized weights in BF16: generic k-bit prep upcasts Qwen's KV cache.
        if not args.load_in_4bit:model=model.to('cuda')
        with Sandbox(image=manifest.get('execution_image'),memory=manifest.get('container_memory','1g')) as sandbox:
            if args.reward_probe:
                summary=probe_rewards(model,tokenizer,selected,sandbox,args,monitor)
                report={'stage':'reward-probe','steps':0,'train_problems':len(train_rows),'probe_problems':len(selected),'generations':args.generations,'temperature':args.temperature,'max_completion_tokens':args.max_tokens,
                    'development_evaluated':False,'confirmation_evaluated':False,'dataset_manifest_sha256':sha(args.dataset/'manifest.json'),'probe_selection_sha256':config['probe_selection_sha256'],
                    'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'peak_gpu_reserved_mb':torch.cuda.max_memory_reserved()/1024**2,**summary}
                (args.output/'report.json').write_text(json.dumps(report,indent=2));build_report(args.output)
                monitor.phase='complete';print(json.dumps(report,indent=2),flush=True)
                return
            before,baseline=evaluate(model,tokenizer,val_rows,sandbox,args.output/'baseline-validation.jsonl',args.max_tokens,monitor,'baseline_validation',score_fn=evaluation_scorer)
            if args.baseline_only:
                groups={}
                for label in sorted({r.get('difficulty','development') for r in val_rows}):
                    ids={r['task_id'] for r in val_rows if r.get('difficulty','development')==label}
                    outputs=[r for r in before if r['task_id'] in ids]
                    groups[label]={'correct':sum(r['correct'] for r in outputs),'problems':len(outputs),'mean_tokens':sum(r['tokens'] for r in outputs)/len(outputs),'token_cap_hits':sum(r['tokens']>=args.max_tokens for r in outputs)}
                    monitor.log({f'baseline/{label}_correctness':groups[label]['correct']/len(outputs)})
                report={'stage':'baseline-only','steps':0,'train_problems':len(train_rows),'development_problems':len(val_rows),'sealed_confirmation_problems':len(confirmation_ids),'confirmation_evaluated':False,'baseline_validation':baseline,'by_difficulty':groups,'max_completion_tokens':args.max_tokens,'dataset_manifest_sha256':sha(args.dataset/'manifest.json'),'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'peak_gpu_reserved_mb':torch.cuda.max_memory_reserved()/1024**2,'development_protocol':manifest['development_protocol']}
                (args.output/'report.json').write_text(json.dumps(report,indent=2));build_report(args.output)
                monitor.phase='complete';print(json.dumps(report,indent=2),flush=True)
                return
            best={'step':0,'correctness':baseline['correctness']}

            def reward(completions,tests,task_id,**kwargs):
                layout=validate_generation_groups(task_id,args.prompts_per_update,args.generations)
                rewards=[]
                monitor.phase='executing Python solutions against supplied tests'
                for completion,problem_tests,problem_id in zip(completions,tests,task_id,strict=True):
                    text=completion[0]['content'] if isinstance(completion,list) else completion
                    score=solution_score(sandbox,text,problem_tests,problem_id)
                    value=float(score['correct']);rewards.append(value)
                    monitor.sample({'task_id':problem_id,'completion':text,'reward':value,**score})
                grouped=[rewards[i:i+args.generations] for i in range(0,len(rewards),args.generations)]
                mixed=sum(len(set(group))>1 for group in grouped)
                monitor.log({'batch/'+key:value for key,value in layout.items()})
                monitor.log({'batch/mixed_reward_groups':mixed,'batch/zero_std_group_fraction':1-mixed/len(grouped),'batch/full_pass_fraction':sum(rewards)/len(rewards)})
                monitor.phase='updating weights'
                return rewards

            class Progress(TrainerCallback):
                def on_step_begin(self,config,state,control,**kwargs):
                    monitor.step=state.global_step;monitor.phase='generating training rollouts'
                def on_log(self,config,state,control,logs=None,**kwargs):
                    monitor.step=state.global_step
                    monitor.log({'train/'+key:value for key,value in (logs or {}).items()})
                    monitor.log({'progress/optimizer_step':state.global_step,'progress/total_steps':args.steps,'progress/percent':100*state.global_step/args.steps})
                def on_step_end(self,config,state,control,model=None,**kwargs):
                    monitor.step=state.global_step
                    monitor.log({'progress/optimizer_step':state.global_step,'progress/total_steps':args.steps,'progress/percent':100*state.global_step/args.steps})
                    if state.global_step%args.eval_every==0 or state.global_step==args.steps:
                        _,metrics=evaluate(model,tokenizer,val_rows,sandbox,args.output/f'validation-step-{state.global_step}.jsonl',args.max_tokens,monitor,'validation',score_fn=evaluation_scorer)
                        monitor.log({'validation/baseline_correctness':baseline['correctness'],'validation/change_from_baseline':metrics['correctness']-baseline['correctness']})
                        if metrics['correctness']>best['correctness']:
                            best.update(step=state.global_step,correctness=metrics['correctness'])
                            model.save_pretrained(args.output/'best-adapter')
                        monitor.phase='training'

            cfg=GRPOConfig(output_dir=str(args.output/'checkpoints'),max_steps=args.steps,**batches,
                max_completion_length=args.max_tokens,learning_rate=args.learning_rate,beta=0.02,temperature=args.temperature,
                bf16=True,gradient_checkpointing=True,optim='adamw_torch',logging_steps=1,logging_first_step=True,
                save_steps=args.eval_every,save_total_limit=2,report_to='none',seed=args.seed,
                remove_unused_columns=False,disable_dropout=True,use_vllm=False)
            trainer=GRPOTrainer(model=model,processing_class=tokenizer,args=cfg,
                peft_config=LoraConfig(r=16,lora_alpha=32,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'),
                train_dataset=Dataset.from_list([{key:row[key] for key in ['prompt','tests','task_id']} for row in train_rows]),
                reward_funcs=reward,callbacks=[Progress()])
            monitor.phase='training'
            trainer.train()
            trainer.save_model(str(args.output/'final-adapter'));tokenizer.save_pretrained(args.output/'final-adapter')
            after=read_rows(args.output/f'validation-step-{args.steps}.jsonl')
            report={'stage':run_stage,'steps':args.steps,'train_problems':len(train_rows),'development_problems':len(val_rows),'sealed_confirmation_problems':len(confirmation_ids),'dataset_manifest_sha256':sha(args.dataset/'manifest.json'),'confirmation_evaluated':False,'development_protocol':manifest['development_protocol'],'baseline_validation':baseline,'final_validation':json.loads((args.output/f'validation-step-{args.steps}.summary.json').read_text()),'selected_checkpoint':best,'development_comparison':compare(before,after,note='Development-only comparison in a bounded pipeline pilot; not independent confirmation.'),'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'peak_gpu_reserved_mb':torch.cuda.max_memory_reserved()/1024**2,'limitations':['A bounded development-only run does not establish general improvement.','Small development set and one training seed; no claim of general improvement.','Confirmation predictions are deliberately deferred.']}
            (args.output/'report.json').write_text(json.dumps(report,indent=2))
            metrics=[json.loads(line) for line in (args.output/'metrics.jsonl').read_text().splitlines()]
            layouts=[r for r in metrics if 'batch/distinct_prompts' in r]
            if len(layouts)!=args.steps or any(r['batch/distinct_prompts']!=args.prompts_per_update or r['batch/candidates']!=args.prompts_per_update*args.generations for r in layouts):
                raise RuntimeError('Observed optimizer batches did not match the declared layout')
            build_report(args.output)
            monitor.phase='complete'
            print(json.dumps(report,indent=2),flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        failure={'error_type':'cuda_oom','phase':monitor.phase,'optimizer_step':monitor.step,'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'peak_gpu_reserved_mb':torch.cuda.max_memory_reserved()/1024**2,'message':str(exc)}
        (args.output/'failure.json').write_text(json.dumps(failure,indent=2))
        monitor.phase='failed: CUDA out of memory';raise
    except BaseException:
        monitor.phase='failed or interrupted';raise
    finally:
        monitor.close()
        (args.output/'status.json').write_text(json.dumps({'phase':monitor.phase,'optimizer_step':monitor.step,'updated_at':time.time()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=Path('data/codeforces-python'))
    parser.add_argument('--name',default='simple-smoke')
    parser.add_argument('--model')
    parser.add_argument('--model-revision')
    parser.add_argument('--load-in-4bit',action='store_true')
    parser.add_argument('--concise-code',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-tokens',type=int)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--baseline-only',action='store_true')
    mode.add_argument('--reward-probe',action='store_true')
    parser.add_argument('--probe-problems',type=int,default=32)
    parser.add_argument('--temperature',type=float,default=0.8)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--eval-every',type=int,default=5)
    parser.add_argument('--prompts-per-update',type=int,default=4)
    parser.add_argument('--generations',type=int,default=8)
    parser.add_argument('--micro-batch',type=int,default=2)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--seed',type=int,default=42)
    main(parser.parse_args())
