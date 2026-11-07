"""Archive the verified-data/calibration/batching pilot without publishing model weights."""
import argparse
import collections
import json
from pathlib import Path
import shutil

from task_selection import read_rows,sha


def build(args):
    quality=json.loads((args.quality/'manifest.json').read_text())
    calibration=json.loads((args.calibration/'summary.json').read_text())
    dataset=json.loads((args.dataset/'manifest.json').read_text())
    pilot=json.loads((args.pilot/'report.json').read_text())
    if pilot['confirmation_evaluated'] or dataset['confirmation_predictions_generated']:
        raise ValueError('Confirmation must remain sealed during this pipeline pilot')
    for split,digest in dataset['files_sha256'].items():
        if sha(args.dataset/f'{split}.jsonl')!=digest:raise ValueError('Frozen data changed during pilot')
    if args.output.exists():raise ValueError('Archive already exists')
    metrics=[json.loads(line) for line in (args.pilot/'metrics.jsonl').read_text().splitlines()]
    layout=[r for r in metrics if 'batch/distinct_prompts' in r]
    signal=[r for r in metrics if 'batch/mixed_reward_groups' in r]
    if len(layout)!=pilot['steps'] or any(r['batch/distinct_prompts']!=4 or r['batch/candidates_per_prompt']!=8 for r in layout):
        raise ValueError('Pilot did not demonstrate the promised four-by-eight batch layout')
    mixed=sum(r['batch/mixed_reward_groups'] for r in signal)
    total=4*len(signal)
    summary={'quality_counts':quality['counts_after'],'calibration':calibration,'pilot_steps':pilot['steps'],'observed_prompt_groups':total,'mixed_reward_groups':mixed,'mixed_group_fraction':mixed/total,'baseline_development_solved':round(pilot['baseline_validation']['correctness']*pilot['development_problems']),'final_development_solved':round(pilot['final_validation']['correctness']*pilot['development_problems']),'development_problems':pilot['development_problems'],'peak_gpu_allocated_mb':pilot['peak_gpu_allocated_mb'],'confirmation_problems':pilot['sealed_confirmation_problems'],'confirmation_evaluated':False}
    args.output.mkdir(parents=True)
    copies=[(args.quality/'manifest.json','quality-manifest.json'),(args.quality/'references.jsonl','reference-checks.jsonl'),(args.calibration/'summary.json','calibration-summary.json'),(args.calibration/'config.json','calibration-config.json'),(args.calibration/'tasks.jsonl','calibration-tasks.jsonl'),(args.dataset/'manifest.json','dataset-manifest.json'),(args.pilot/'report.json','pilot-report.json'),(args.pilot/'config.json','pilot-config.json'),(args.pilot/'metrics.jsonl','pilot-metrics.jsonl'),(args.pilot/'baseline-validation.jsonl','baseline-validation.jsonl')]
    for source,name in copies:shutil.copy2(source,args.output/name)
    for source in args.pilot.glob('validation-step-*.jsonl'):shutil.copy2(source,args.output/source.name)
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for name in ['task_selection.py','calibrate_tasks.py','train_v2.py','evaluate.py','execution.py','cpython_worker.py']:
        shutil.copy2(name,snapshot/name)
    (snapshot/'SHA256SUMS.json').write_text(json.dumps({p.name:sha(p) for p in snapshot.glob('*.py')},indent=2))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Verified task selection and batching pilot', '', 'This is a development-only pipeline pilot. No general model improvement is established, and no confirmation answers were generated. Historical experiment-001 results remain unchanged.', '', '| Stage | Result |','|---|---|',f"| References passing original and expanded checks | {quality['counts_after']['train']} training tasks |",f"| Eight-candidate calibration | {calibration['selected_tasks']} mixed, {calibration['buckets'].get('hard',0)} all failed, {calibration['buckets'].get('easy',0)} all passed |",f"| Observed batches | {len(layout)} updates, four distinct prompts × eight candidates = 32 rollouts/update |",f"| Groups with a correctness comparison signal | {mixed}/{total} ({100*mixed/total:.1f}%) |",f"| Development correctness: original / final | {summary['baseline_development_solved']}/{summary['development_problems']} / {summary['final_development_solved']}/{summary['development_problems']} |",f"| Peak PyTorch allocated memory | {summary['peak_gpu_allocated_mb']/1024:.2f} GiB |",f"| Sealed confirmation tasks | {summary['confirmation_problems']}; unscored |",'', '## Protocol', '', 'The original Qwen2.5-Coder-0.5B-Instruct model was sampled eight times per verified training task at temperature 0.8 with a 256-token completion cap and seed 42. A training task was selected only when 1–7 of its eight sampled answers passed all original and official expanded tests. Validation and confirmation model outcomes never influenced task selection.', '', 'Training uses a binary reward for passing all original and official expanded tests. Two rollouts per micro-batch and 16 gradient accumulation steps form one 32-rollout optimizer update. The installed TRL sampler and runtime reward callback both verify four distinct contiguous prompt groups of eight. The learning rate remains 1e-5; LoRA rank 16 and KL coefficient 0.02 remain unchanged.', '', 'Existing development validation is used only as development data. Confirmation candidates come from the 148 original test tasks never evaluated in experiment 001; reference/coverage checks retain 49. Both split IDs and file hashes are frozen before training. Neither calibration nor the validation-only pilot generates confirmation predictions.', '', '## Interpretation', '', 'This combined pilot changes task verification, selection, reward strength, and batching. It demonstrates that the new pipeline works and measures its signal; it does not isolate which change helps performance. The calibrated training pool is small, development validation has only 29 tasks, and only one seed was run. A longer run or a one-task validation change would not establish general improvement.', '', 'Before spending the confirmation set, expand the verified task pool and compare controlled variants with matched rollout budgets and additional seeds. Keep the original model as the reference baseline.']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--quality',type=Path,default=Path('runs/task-quality-v2'))
    parser.add_argument('--calibration',type=Path,default=Path('runs/calibration-v2'))
    parser.add_argument('--dataset',type=Path,default=Path('data/selected-v2'))
    parser.add_argument('--pilot',type=Path,default=Path('runs/pilot-v2'))
    parser.add_argument('--output',type=Path,default=Path('experiments/pipeline-v2'))
    build(parser.parse_args())
