"""Archive data acquisition and train-only calibration, with quality limits explicit."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil

from task_selection import read_rows,sha


def build(args):
    quality=json.loads((args.quality/'manifest.json').read_text())
    selected=json.loads((args.dataset/'manifest.json').read_text())
    calibration=json.loads((args.calibration/'summary.json').read_text())
    rows={s:read_rows(args.quality/f'{s}.jsonl') for s in ['train','validation','confirmation']}
    records=read_rows(args.calibration/'tasks.jsonl')
    if args.output.exists():raise ValueError('Archive exists')
    if sha(args.quality/'manifest.json')!=selected['quality_manifest_sha256']:raise ValueError('Curation manifest changed after calibration')
    if any(len(r['candidates'])!=selected['samples'] or r['correct']!=sum(c['correct'] for c in r['candidates']) for r in records):raise ValueError('Incomplete or inconsistent calibration records')
    for split,items in rows.items():
        if sha(args.quality/f'{split}.jsonl')!=quality['files_sha256'][split]:raise ValueError('Quality data changed')
        if sha(args.dataset/f'{split}.jsonl')!=selected['files_sha256'][split]:raise ValueError('Selected data changed')
    if {r['task_id'] for r in records}!={r['task_id'] for r in rows['train']} or len(records)!=len(rows['train']):
        raise ValueError('Calibration must cover exactly the training tasks')
    for a,b in [('train','validation'),('train','confirmation'),('validation','confirmation')]:
        for key in ['task_id','group_id','contract_group']:
            if {r[key] for r in rows[a] if r[key]} & {r[key] for r in rows[b] if r[key]}:
                raise ValueError('Split overlap: '+key)
    if quality['confirmation_predictions_generated'] or selected['confirmation_predictions_generated']:raise ValueError('Confirmation was queried')
    args.output.mkdir(parents=True)
    checks=Path('runs/opencoder-tests.log')
    if checks.exists():shutil.copy2(checks,args.output/'verification.txt')
    for source,name in [(args.quality/'manifest.json','quality-manifest.json'),(args.quality/'row-exclusions.jsonl','row-exclusions.jsonl'),(args.quality/'group-checks.jsonl','group-checks.jsonl'),(args.calibration/'summary.json','calibration-summary.json'),(args.calibration/'config.json','calibration-config.json'),(args.calibration/'tasks.jsonl','calibration-tasks.jsonl'),(args.dataset/'manifest.json','selected-manifest.json')]:
        shutil.copy2(source,args.output/name)
    for split in ['train','validation','confirmation']:
        shutil.copy2(args.quality/f'{split}.jsonl',args.output/f'screened-{split}.jsonl')
    shutil.copy2(args.dataset/'train.jsonl',args.output/'mixed-train.jsonl')
    shutil.copy2(args.dataset/'train-independent.jsonl',args.output/'mixed-train-independent.jsonl')
    mixed=read_rows(args.dataset/'train.jsonl')
    by_id={r['task_id']:r for r in records}
    reviewed={r['row_sha256']:r for r in json.loads(Path('opencoder_contract_review.json').read_text())}
    reviews=[]
    for row in rows['train']:
        if row['quality_tier']!='independent_contract_checked':continue
        digest=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
        review=reviewed.get(digest,{'task_id':row['task_id'],'row_sha256':digest,'verdict':'pending_review','note':'No matching instruction/reference/oracle review.'})
        reviews.append({**review,'family':row['independent_family'],'calibration_bucket':by_id[row['task_id']]['bucket']})
    reviewed_ids={r['task_id'] for r in reviews if r['verdict']=='behavior_reviewed_on_tested_domain'}
    (args.output/'contract-review.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reviews))
    reviewed_mixed=[r for r in mixed if r['task_id'] in reviewed_ids]
    (args.output/'mixed-independent-reviewed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reviewed_mixed))
    queue=[{**r,'calibration_passes':by_id[r['task_id']]['correct'],'calibration_samples':by_id[r['task_id']]['samples'],'review_status':'pending'} for r in mixed if r['needs_semantic_review']]
    (args.output/'semantic-review-queue.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in queue))
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for name in ['import_opencoder.py','opencoder_data.py','educational_oracles.py','calibrate_tasks.py','opencoder_report.py','execution.py','cpython_worker.py','task_selection.py','monitoring.py']:
        shutil.copy2(name,snapshot/name)
    shutil.copy2('opencoder_contract_review.json',snapshot/'opencoder_contract_review.json')
    (snapshot/'SHA256SUMS.json').write_text(json.dumps({p.name:sha(p) for p in snapshot.iterdir()},indent=2))
    summary={'raw_candidates':quality['raw_candidates'],'problem_groups':quality['problem_groups'],'retained_groups':quality['retained_groups'],'split_counts':quality['counts_after'],'quality_tiers':quality['quality_tiers'],'calibration':calibration,'mixed_independent_behavior_reviewed':len(reviewed_mixed),'independent_contract_review_issues':sum(r['verdict']=='needs_instruction_or_domain_resolution' for r in reviews),'mixed_pending_semantic_review':len(queue),'mixed_independent_contract_checked':selected['independent_selected'],'confirmation_predictions_generated':False,'new_training_performed':False,'improvement_claim':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    n=quality['counts_after'];tiers=quality['quality_tiers'];b=calibration['buckets']
    lines=['# OpenCoder acquisition and screening — 2026-11-07','',
    'This is a data-curation and difficulty-calibration run. No new model was trained and no model improvement is claimed.','',
    '| Stage | Result |','|---|---|',
    f"| Pinned source sample | {quality['raw_candidates']} rows |",
    f"| AST / prompt / recognized-contract groups | {quality['problem_groups']} |",
    f"| Retained problems | {quality['retained_groups']} |",
    f"| Independent narrow contracts with generated cases | {tiers.get('independent_contract_checked',0)} |",
    f"| Supplied tests + nine constant controls only | {tiers.get('reference_and_constant_control_screened',0)}; semantic review pending |",
    f"| Frozen splits | {n['train']} training / {n['validation']} development / {n['confirmation']} confirmation |",
    f"| Eight-sample training-only calibration | {b.get('mixed',0)} mixed / {b.get('hard',0)} all fail / {b.get('easy',0)} all pass |",
    f"| Mixed tasks with independent contract checks | {selected['independent_selected']} |",
    f"| Mixed tasks requiring semantic review | {len(queue)} |",f"| Mixed independent tasks after instruction/domain review | {len(reviewed_mixed)} on their tested domains |",'',
    '## Source and reproducibility','',
    f"Source: [OpenCoder educational_instruct](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2/tree/{quality['source']['revision']}); license declared by the source is MIT. Attribution belongs to the OpenCoder dataset authors. Native dataset rows are sampled using seed 42, a 10,000-row streaming shuffle buffer, and a 3,000-row cap. Source revision, row IDs, raw SHA-256, protected-data hashes, Docker image ID, and frozen split hashes are in quality-manifest.json. Exact source snapshots and hashes accompany this report. The intermediate curation-v1 output was superseded by the final contract-family-aware curation-v2 pass before any model calibration.", '',
    'Normalization accepts short synchronous functions with direct literal-input equality assertions. Tests needing shared setup are excluded. References run only in resource-limited Docker. Recognized contracts receive deterministic edge/random cases from separately implemented bounded oracles. Every retained suite rejects nine constant-output controls; all assertions must pass for binary reward. Assertions are bundled into one fresh child process per candidate, with a two-second total suite limit. Unlike the MBPP evaluator, calls within this suite share candidate state. This protocol is recorded explicitly and does not use the MBPP+ scorer.', '',
    'Canonical AST clones, near-identical prompts, and recognized contract variants are grouped before choosing representatives or splitting. Known overlaps with all existing MBPP validation/test rows are excluded using reference ASTs, prompt similarity, and recognized contracts. This is a conservative filter, not a guarantee of semantic or pretraining decontamination. Representatives are chosen by test/reference quality before any model sampling. Hashes assign groups approximately 80/10/10; task IDs, group IDs, and recognized contracts are disjoint between splits.', '',
    'The original pinned Qwen2.5-Coder-0.5B-Instruct was sampled eight times on every training task, with temperature 0.8, a 256-token cap, and seed 42. A mixed task has 1–7 fully passing answers. All task-level answers and checks are archived. Development and confirmation receive no model predictions. The existing 49 MBPP confirmation tasks also remain untouched.', '',
    '## Limits and next decision','',
    'Reference/test agreement and rejection of trivial constants do not independently establish task correctness. The lower quality tier must not be called verified or fed into training before semantic review and stronger tests. Narrow independent contracts validate behavior on their generated domain; they do not prove correctness for all inputs or enforce algorithm restrictions such as recursion or a ban on built-ins. Those instruction constraints need review too.', '',
    'The recorded code-and-instruction review of the 13 independent-contract training rows found four instruction/domain issues: a no-built-ins reference itself calls range; recursion is not enforced by behavioral tests; the parentheses-only test domain needs to be explicit; and digit-sum tests missed negative values unsupported by the reference. contract-review.jsonl records the verdicts. mixed-independent-reviewed.jsonl excludes these four tasks from the independently checked mixed subset. This review is limited to the tested domains and does not waive the need for stronger plausible-error controls.', '',
    'The new development split contains no independently augmented contracts, so it is not a strong replacement for the existing expanded MBPP development evaluation. Keep the existing 29-task expanded development suite and 49-task untouched confirmation pool while strengthening this new data. Newly frozen OpenCoder confirmation rows are an additional pool whose tests still need review.', '',
    f"The initial target of hundreds of independently checked learnable tasks has not been reached: only {selected['independent_selected']} mixed tasks currently have independent narrow-contract checks. Review semantic-review-queue.jsonl, add independent edge cases and plausible non-constant faulty implementations, resolve instruction constraints, then freeze a new approved training manifest. The existing trainer rejects this new dataset protocol to prevent accidentally using the MBPP-specific scorer. No longer training run is justified by these screening counts alone.", '']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    license_path=Path('third_party/OPENCODER_LICENSE')
    shutil.copy2(license_path,args.output/'OPENCODER_LICENSE')
    (args.output/'ATTRIBUTION.md').write_text('Derived educational tasks and references originate from OpenCoder-LLM/opc-sft-stage2, educational_instruct. The upstream dataset declares the MIT license. Source: https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2 . Pinned revision: '+quality['source']['revision']+' . Project authors retain attribution to the upstream OpenCoder authors.\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quality',type=Path,default=Path('runs/opencoder-curation-v2'));p.add_argument('--calibration',type=Path,default=Path('runs/calibration-opencoder-v1'));p.add_argument('--dataset',type=Path,default=Path('data/opencoder-selected-v1'));p.add_argument('--output',type=Path,default=Path('experiments/opencoder-data-v1'))
    build(p.parse_args())
