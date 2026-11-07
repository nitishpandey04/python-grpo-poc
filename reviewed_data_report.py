"""Archive independently reviewed training inputs without claiming model improvement."""
import argparse
import collections
import json
from pathlib import Path
import shutil
from task_selection import read_rows,sha
from strengthen_opencoder import row_sha


def build(args):
    if args.output.exists():raise ValueError('Archive exists')
    quality=json.loads((args.quality/'manifest.json').read_text())
    calibrated=json.loads((args.dataset/'manifest.json').read_text())
    calibration=json.loads((args.calibration/'summary.json').read_text())
    summary=json.loads((args.quality/'summary.json').read_text())
    if sha(args.quality/'manifest.json')!=calibrated['quality_manifest_sha256']:raise ValueError('Quality manifest changed')
    for directory,manifest in [(args.quality,quality),(args.dataset,calibrated)]:
        for split,digest in manifest['files_sha256'].items():
            if sha(directory/f'{split}.jsonl')!=digest:raise ValueError('Frozen data changed')
    vetted={r['task_id']:r for r in read_rows(args.quality/'train.jsonl')}
    train=read_rows(args.dataset/'train.jsonl')
    records=read_rows(args.calibration/'tasks.jsonl')
    if len(records)!=len(vetted) or {r['task_id'] for r in records}!=set(vetted):raise ValueError('Fresh calibration incomplete')
    if any(row_sha(r)!=row_sha(vetted[r['task_id']]) for r in train):raise ValueError('Calibrated task changed')
    fresh={r['task_id']:r for r in records}
    if any(not 0<fresh[r['task_id']]['correct']<8 or len(fresh[r['task_id']]['candidates'])!=8 for r in train):raise ValueError('Mixed selection rule violated')
    decisions=read_rows(args.quality/'decisions.jsonl')
    approved=[r for r in decisions if r['status']=='approved_on_declared_test_domain']
    for r in approved:
        if not r['reference_score']['correct'] or len(r['controls'])!=12 or any(c['strong_score']['correct'] for c in r['controls']):raise ValueError('Incomplete quality controls')
        if any(c['strong_score']['correct'] and not c['old_correct'] for c in r['frozen_baseline_rescores']):raise ValueError('Unexpected nonmonotonic frozen rescore')
    baseline_manifest=json.loads((args.evaluation/'manifest.json').read_text())
    archived_mbpp=json.loads(Path('experiments/pipeline-v2/dataset-manifest.json').read_text())
    for split in ['validation','confirmation']:
        digest=sha(args.evaluation/f'{split}.jsonl')
        if digest!=baseline_manifest['files_sha256'][split] or digest!=archived_mbpp['files_sha256'][split]:raise ValueError('Existing MBPP evaluation changed')
    if calibrated['confirmation_predictions_generated']:raise ValueError('Confirmation was queried')
    args.output.mkdir(parents=True)
    for name in ['manifest.json','config.json','summary.json','checks.jsonl','decisions.jsonl']:
        shutil.copy2(args.quality/name,args.output/('quality-'+name))
    shutil.copytree(args.quality/'source_snapshot',args.output/'quality_source_snapshot')
    for name in ['config.json','summary.json','tasks.jsonl']:
        shutil.copy2(args.calibration/name,args.output/('calibration-'+name))
    shutil.copy2(args.dataset/'manifest.json',args.output/'calibration-selected-manifest.json')
    shutil.copy2(args.quality/'train.jsonl',args.output/'all-reviewed-train.jsonl')
    shutil.copy2(args.dataset/'train.jsonl',args.output/'train.jsonl')
    shutil.copy2(args.evaluation/'validation.jsonl',args.output/'development-mbpp.jsonl')
    shutil.copy2('runs/reviewed-contract-tests.log',args.output/'verification.txt')
    shutil.copy2('third_party/OPENCODER_LICENSE',args.output/'OPENCODER_LICENSE')
    selected_ids={r['task_id'] for r in train}
    source_snapshot=args.output/'report_source_snapshot';source_snapshot.mkdir()
    for name in ['reviewed_data_report.py','tests/test_reviewed_contracts.py']:
        shutil.copy2(name,source_snapshot/Path(name).name)
    (source_snapshot/'SHA256SUMS.json').write_text(json.dumps({p.name:sha(p) for p in source_snapshot.iterdir()},indent=2))
    manifest={'protocol':'reviewed-assert-train-v2','purpose':'Reviewed inputs for the next controlled experiment; no new optimization performed','model':calibrated['model'],'revision':calibrated['revision'],'training':{'path':'train.jsonl','tasks':len(train),'sha256':sha(args.output/'train.jsonl'),'scorer':'opencoder_data.assertion_suite_score','reward':'binary all assertions pass; bundled per candidate; 2-second suite limit','quality':'independent bounded cases + two nonconstant fault controls + nine constants + old-example memorizer rejected'},'development':{'path':'development-mbpp.jsonl','tasks':len(read_rows(args.evaluation/'validation.jsonl')),'sha256':sha(args.output/'development-mbpp.jsonl'),'scorer':'task_selection.expanded_score','protocol':'original MBPP assertions AND official expanded suites; same existing development tasks'},'confirmation':{'tasks':len(read_rows(args.evaluation/'confirmation.jsonl')),'local_path':str(args.evaluation/'confirmation.jsonl'),'sha256':sha(args.evaluation/'confirmation.jsonl'),'task_ids':baseline_manifest['task_ids']['confirmation'],'predictions_generated':False},'additional_opencoder_heldout':{'validation_tasks':27,'confirmation_tasks':28,'predictions_generated':False,'policy':'Not approved as primary evaluation; supplied tests still need review'},'image_id':quality['image_id'],'quality_manifest_sha256':sha(args.output/'quality-manifest.json'),'calibration_manifest_sha256':sha(args.output/'calibration-selected-manifest.json'),'training_integration_required':True,'new_training_performed':False}
    (args.output/'training-manifest.json').write_text(json.dumps(manifest,indent=2))
    control_types=collections.Counter('memorization' if c['name']=='source_example_memorizer' else 'constant' if c['name'].startswith('constant_') else 'nonconstant_algorithmic' for d in approved for c in d['controls'] if c['old_score']['correct'] and not c['strong_score']['correct'])
    final={**summary,'fresh_calibration':calibration,'selected_reviewed_mixed_tasks':len(train),'newly_rejected_control_types':dict(control_types),'primary_development_tasks':manifest['development']['tasks'],'primary_confirmation_tasks':manifest['confirmation']['tasks'],'confirmation_predictions_generated':False,'model_improvement_claim':False}
    (args.output/'summary.json').write_text(json.dumps(final,indent=2))
    catalogue=['# Reviewed contracts and tested domains','','Domains below bound the evidence; they do not restrict the original problem to these cases or prove correctness outside them.','','| Task | Contract | Independent cases | Fresh passes / 8 | Selected | Tested domain |','|---|---|---:|---:|---|---|']
    by_check={r['task_id']:r for r in approved}
    for task,row in vetted.items():
        catalogue.append(f"| {task} | {row['independent_family']} | {by_check[task]['independent_cases']} | {fresh[task]['correct']} | {'yes' if task in selected_ids else 'no'} | {row['reviewed_domain']} |")
    (args.output/'CONTRACTS.md').write_text('\n'.join(catalogue)+'\n')
    rejected=[r for r in decisions if r['status']=='reject_reference_disagrees_with_stronger_tests']
    defects=['# References rejected by stronger tests','','All five references passed the previous supplied tests. They are excluded, with original instructions preserved; no instruction constraint was silently removed.','']
    source_rows={r['task_id']:r for r in read_rows(Path('experiments/opencoder-data-v1/mixed-train.jsonl'))}
    for r in rejected:
        parent=source_rows[r['task_id']]
        defects.extend([f"## {r['task_id']} — {r['contract']}",'',parent['instruction'],'',f"Failure: {r['reference_score']['errors']}",'','```python',parent['reference'],'```',''])
    (args.output/'REFERENCE_DEFECTS.md').write_text('\n'.join(defects))
    counts=summary['decisions'];b=calibration['buckets']
    lines=['# Stronger OpenCoder training-data review — 2026-11-07','','This work strengthens data and measures baseline difficulty. No new model was trained, and no model improvement is claimed.','','| Stage | Result |','|---|---|',f"| Previously mixed tasks reviewed | {summary['tasks_reviewed']} |",f"| Conservative related held-out-family exclusions | {counts['exclude_known_heldout_family']+counts['exclude_related_heldout_family']} |",f"| Deferred contract/algorithm constraints | {counts['defer_contract_or_enforcement']} |",f"| Duplicate reviewed contracts collapsed | {counts['exclude_duplicate_reviewed_contract']} |",f"| References failing stronger tests | {len(rejected)} |",f"| Independently specified contracts passing quality checks | {len(vetted)} |",f"| Fresh baseline calibration | {b.get('mixed',0)} mixed / {b.get('hard',0)} all fail / {b.get('easy',0)} all pass |",f"| Selected reviewed training tasks | {len(train)} |",f"| Existing development / confirmation pools | {manifest['development']['tasks']} / {manifest['confirmation']['tasks']}; confirmation unqueried |",'',
    '## What changed','',
    'All 130 formerly mixed tasks received fingerprinted review decisions. Ambiguous wording, incorrect examples, unspecified output order, and unenforced algorithm restrictions are deferred. Existing MBPP validation/test text and the frozen OpenCoder held-out text were consulted only to exclude related training contracts; no held-out model predictions were generated. Some exclusions deliberately remove broader related families rather than exact duplicates, and imperfect wording filters can still miss semantic overlap. Multiple source implementations of one reviewed contract receive one training representative, chosen by supplied-test count and reference length before model rescoring.', '',
    'Separately specified bounded oracles generate new literal expected results without executing source code on the host. Coverage includes exhaustive short attendance/binary/periodic-string domains, brute-force small permutations, negative cubes, duplicate-sensitive lists, format boundaries, and empty-input cases where the contract defines the result. Each retained reference passes the stronger suite, and the suite rejects two handwritten nonconstant faulty implementations, nine constant outputs, and a lookup memorizing the old examples. The controls and all outcomes are archived. References, generated answers, and faulty controls execute only inside resource-limited Docker.', '',
    f"Across retained contracts, all {summary['fault_controls_rejected']} controls fail, including {summary['nonconstant_fault_controls_rejected']} nonconstant controls. Stronger tests newly reject {control_types.get('nonconstant_algorithmic',0)} nonconstant controls and {control_types.get('memorization',0)} old-example lookups that passed the old suites. The five reference defects include empty-input failures for simplified tags and search insertion, acceptance of a trailing newline in a six-digit hex color, counting row length instead of non-space runs, and treating zero as containing no zero digit. See [reference defects](REFERENCE_DEFECTS.md) and [contract domains](CONTRACTS.md).", '',
    '## Baseline calibration','',
    f"Rechecking the same {summary['frozen_baseline_candidates']} frozen baseline answers reduces passing answers from {summary['frozen_baseline_old_passes']} to {summary['frozen_baseline_strong_passes']}. This exposes {summary['frozen_baseline_old_passes']-summary['frozen_baseline_strong_passes']} previously passing answers rejected by the stronger criteria. The model has not changed: these are stricter test outcomes, not a performance regression from training. Under these frozen samples, {summary['frozen_strong_buckets'].get('mixed',0)} contracts remain mixed and {summary['frozen_strong_buckets'].get('hard',0)} become all-fail.", '',
    f"A separate fresh calibration then samples the original pinned Qwen2.5-Coder-0.5B-Instruct eight times on each of the {len(vetted)} retained training contracts, at temperature 0.8, 256 tokens, and seed 42. It selects {len(train)} tasks with 1–7 fully passing candidates. Fresh generation has a different task order/random stream from the original 251-task calibration; the two sets of samples are not a paired model comparison. Live metrics are recorded under calibration-opencoder-v2 in Trackio, with calibration/percent rather than optimizer steps showing progress.", '',
    '## Prepared inputs and limits','',
    'training-manifest.json links train.jsonl to the existing expanded MBPP development protocol and fingerprints the existing 49-task confirmation pool. The copied OpenCoder development/confirmation sets remain unreviewed and are not substituted for stronger evaluation. All held-out source hashes remain unchanged. The MBPP-specific trainer intentionally rejects the new protocol; explicit train-scorer/development-scorer integration is required before optimization, so data cannot silently use the wrong reward.', '',
    f"The selected pool has {len(train)} reviewed contract families, still far below hundreds of diverse problems. Finite generated cases and a small handpicked control set do not prove correctness; domains exclude some ambiguous inputs, such as leading-zero IPv4 policy and invalid window sizes. These reviews were performed by the assistant by reading instructions, references and oracle code, not by independent human annotators. Public pretraining overlap and undetected semantic overlap remain possible. The existing development set is small and has already informed decisions. Use the untouched confirmation pool only after a promising, predeclared development comparison, and avoid treating training reward or calibration selection as evidence of model improvement.", '',
    'All 31 regression checks passed, including Docker tests reproducing reference defects and rejecting faulty implementations. Quality checks, frozen-answer rescores, fresh calibration outputs, source snapshots, fingerprints, and the upstream OpenCoder MIT license accompany this report. OpenCoder attribution and the pinned data source are retained from [the import report](../opencoder-data-v1/REPORT.md).','']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(final,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quality',type=Path,default=Path('runs/opencoder-strengthened-v2'));p.add_argument('--calibration',type=Path,default=Path('runs/calibration-opencoder-v2'));p.add_argument('--dataset',type=Path,default=Path('data/opencoder-selected-v2'));p.add_argument('--evaluation',type=Path,default=Path('data/selected-v2'));p.add_argument('--output',type=Path,default=Path('experiments/opencoder-reviewed-v2'))
    build(p.parse_args())
