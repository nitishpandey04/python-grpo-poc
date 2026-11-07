"""A bounded, validation-only pilot of verified task selection and broad GRPO batches."""
import argparse
import hashlib
import json
import os
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
from execution import Sandbox
from monitoring import Monitor
from task_selection import assert_disjoint,expanded_score,read_rows,sha,validate_generation_groups


def batch_settings(prompts,generations,micro_batch):
    if min(prompts,micro_batch)<1 or generations<2:
        raise ValueError('Need positive prompt/micro-batch counts and at least two generations')
    total=prompts*generations
    if total % micro_batch:
        raise ValueError('Micro-batch must divide the generation batch exactly')
    return {'per_device_train_batch_size':micro_batch,'gradient_accumulation_steps':total//micro_batch,'generation_batch_size':total,'num_generations':generations}


def main(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM,AutoTokenizer,TrainerCallback,set_seed
    from trl import GRPOConfig,GRPOTrainer

    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('This pilot is configured for one GPU')
    batches=batch_settings(args.prompts_per_update,args.generations,args.micro_batch)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    if manifest['protocol']!='selected-grpo-v2':raise ValueError('Expected a calibrated verified dataset')
    for split,digest in manifest['files_sha256'].items():
        if sha(args.dataset/f'{split}.jsonl')!=digest:raise ValueError('Frozen selected data changed')
    train_rows=read_rows(args.dataset/'train.jsonl');val_rows=read_rows(args.dataset/'validation.jsonl')
    # Only IDs are used to guard the sealed confirmation pool. No confirmation prediction is generated.
    train_ids={r['task_id'] for r in train_rows};val_ids={r['task_id'] for r in val_rows};confirmation_ids=set(manifest['task_ids']['confirmation'])
    if train_ids&val_ids or train_ids&confirmation_ids or val_ids&confirmation_ids:raise ValueError('Data leakage')
    if len(train_rows)<args.prompts_per_update or not val_rows:raise ValueError('Insufficient training or development problems')
    if args.output.exists():raise ValueError('Pilot output exists; choose a new run name')
    args.output.mkdir(parents=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    set_seed(args.seed);torch.set_num_threads(4)
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'stage':'validation-only pilot','batch_settings':batches,'dataset_manifest_sha256':sha(args.dataset/'manifest.json'),'reward':'binary original + official expanded suites','confirmation_evaluated':False,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    monitor=Monitor(args.output,args.output.name,config)
    try:
        monitor.phase='loading original model'
        tokenizer=AutoTokenizer.from_pretrained(manifest['model'],revision=manifest['revision'],local_files_only=True)
        tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['revision'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
        with Sandbox() as sandbox:
            before,baseline=evaluate(model,tokenizer,val_rows,sandbox,args.output/'baseline-validation.jsonl',manifest['max_tokens'],monitor,'baseline_validation',score_fn=expanded_score)
            best={'step':0,'correctness':baseline['correctness']}

            def reward(completions,tests,task_id,**kwargs):
                layout=validate_generation_groups(task_id,args.prompts_per_update,args.generations)
                rewards=[]
                monitor.phase='scoring expanded-test training rollouts'
                for completion,problem_tests,problem_id in zip(completions,tests,task_id,strict=True):
                    text=completion[0]['content'] if isinstance(completion,list) else completion
                    score=expanded_score(sandbox,text,problem_tests,problem_id)
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
                    if state.global_step%args.eval_every==0 or state.global_step==args.steps:
                        _,metrics=evaluate(model,tokenizer,val_rows,sandbox,args.output/f'validation-step-{state.global_step}.jsonl',manifest['max_tokens'],monitor,'validation',score_fn=expanded_score)
                        monitor.log({'validation/baseline_correctness':baseline['correctness'],'validation/change_from_baseline':metrics['correctness']-baseline['correctness']})
                        if metrics['correctness']>best['correctness']:
                            best.update(step=state.global_step,correctness=metrics['correctness'])
                            model.save_pretrained(args.output/'best-adapter')
                        monitor.phase='training'

            cfg=GRPOConfig(output_dir=str(args.output/'checkpoints'),max_steps=args.steps,**batches,
                max_completion_length=manifest['max_tokens'],learning_rate=args.learning_rate,beta=0.02,temperature=manifest['temperature'],
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
            report={'stage':'validation-only pilot','steps':args.steps,'train_problems':len(train_rows),'development_problems':len(val_rows),'sealed_confirmation_problems':len(confirmation_ids),'confirmation_evaluated':False,'baseline_validation':baseline,'final_validation':json.loads((args.output/f'validation-step-{args.steps}.summary.json').read_text()),'selected_checkpoint':best,'development_comparison':compare(before,after,note='Development-only comparison in a bounded pipeline pilot; not independent confirmation.'),'peak_gpu_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'limitations':['Selection, stronger rewards and batching changed together; this pilot does not isolate their causal effects.','Small development set and one training seed; no claim of general improvement.','Confirmation predictions are deliberately deferred.']}
            (args.output/'report.json').write_text(json.dumps(report,indent=2))
            metrics=[json.loads(line) for line in (args.output/'metrics.jsonl').read_text().splitlines()]
            layouts=[r for r in metrics if 'batch/distinct_prompts' in r]
            if len(layouts)!=args.steps or any(r['batch/distinct_prompts']!=args.prompts_per_update or r['batch/candidates']!=args.prompts_per_update*args.generations for r in layouts):
                raise RuntimeError('Observed optimizer batches did not match the declared layout')
            monitor.phase='complete'
            print(json.dumps(report,indent=2),flush=True)
    except BaseException:
        monitor.phase='failed or interrupted';raise
    finally:
        monitor.close()
        (args.output/'status.json').write_text(json.dumps({'phase':monitor.phase,'optimizer_step':monitor.step,'updated_at':time.time()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=Path('data/selected-v2'))
    parser.add_argument('--output',type=Path,default=Path('runs/pilot-v2'))
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--eval-every',type=int,default=5)
    parser.add_argument('--prompts-per-update',type=int,default=4)
    parser.add_argument('--generations',type=int,default=8)
    parser.add_argument('--micro-batch',type=int,default=2)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--seed',type=int,default=42)
    main(parser.parse_args())
