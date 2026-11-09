"""Summarize a completed development-only run from its saved evidence."""
import argparse
import json
from pathlib import Path


def build_report(run):
    report=json.loads((run/'report.json').read_text())
    if report.get('stage')=='baseline-only':
        metrics=report['baseline_validation'];n=report['development_problems']
        lines=[f'# {run.name}: Python Codeforces baseline','',
            'Baseline only: no optimizer steps, adapter training, or confirmation predictions.','',
            f"Prepared training problems: {report['train_problems']}. Fixed development problems: {n}. Completion cap: {report['max_completion_tokens']} tokens.",'',
            f"Original-model correctness: {round(metrics['correctness']*n)}/{n}. Every solved problem passed all official I/O records.",'',
            '| Scope | Correct / problems | Outputs hitting token cap |','|---|---:|---:|']
        for label,group in report['by_difficulty'].items():lines.append(f"| {label} | {group['correct']}/{group['problems']} | {group['token_cap_hits']} |")
        lines += ['',report['development_protocol'],'',
            'This is a small, previously public contest development set. Pretraining overlap is possible. A greedy baseline does not determine whether a problem will produce mixed rewards under sampled GRPO rollouts. All prepared easy and medium tasks remain eligible regardless of these results.', '',
            'The grader runs fresh Python processes, applies the recorded per-case limits and normalizes whitespace and YES/NO case as in the upstream no-checker runner. It stops at the first failure; unchecked test cases are not counted as evaluated. This is not an exact reproduction of Codeforces language-specific timing.','']
        (run/'REPORT.md').write_text('\n'.join(lines));return
    before=report['baseline_validation'];after=report['final_validation'];n=report['development_problems']
    rollouts=[json.loads(l) for l in (run/'rollouts.jsonl').read_text().splitlines()]
    metrics=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
    groups=[r for r in metrics if 'batch/mixed_reward_groups' in r]
    mixed=sum(r['batch/mixed_reward_groups'] for r in groups)
    prompts=sum(r['batch/distinct_prompts'] for r in metrics if 'batch/distinct_prompts' in r)
    comparison=report['development_comparison'];best=report['selected_checkpoint']
    lines=[f'# {run.name}: native Python test-program GRPO','',
        'This is a short pipeline smoke test. It does not establish general model improvement.','',
        f"Available training problems: {report['train_problems']}. Optimizer steps: {report['steps']}. Distinct sampled training problems: {len({r['task_id'] for r in rollouts})}.",'',
        f"Fixed development correctness: original {round(before['correctness']*n)}/{n}; final adapter {round(after['correctness']*n)}/{n}. Selected step: {best['step']} (zero retains the original model).",'',
        f"Development change: {comparison['improvement_percentage_points']:+.2f} percentage points; paired bootstrap interval {comparison['ci95_percentage_points']}. Problems improved/regressed: {comparison['improved']}/{comparison['regressed']}.",'',
        f'Mixed-reward prompt groups: {mixed}/{prompts}. Training reward uses supplied test programs, whose coverage and expectations may be imperfect.', '',
        f"Peak PyTorch allocation: {report['peak_gpu_allocated_mb']/1024:.2f} GiB. Confirmation problems: {report['sealed_confirmation_problems']}; no confirmation predictions generated.",'',
        report.get('development_protocol','Fixed development evaluation.')+' Tests are finite; semantic/pretraining overlap remains possible. This short run does not establish general improvement.', '',
        'Configuration, source/data fingerprints, model outputs, scalar metrics, development comparisons, and the final adapter are saved alongside this report. Historical experiment files are unchanged.','']
    (run/'REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);build_report(p.parse_args().run)
