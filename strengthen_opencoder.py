"""Review training tasks only, verify stronger suites and audit frozen baseline answers."""
import argparse
import ast
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

from execution import Sandbox
from opencoder_data import assertion_suite_score,literal_examples
from reviewed_contracts import cases_for,family,faulty_controls,protected_hits
from task_selection import calibration_bucket,read_rows,sha


def row_sha(row):return hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()


def load_protected(source,mbpp):
    result=[]
    for path in [mbpp/'validation.jsonl',mbpp/'test.jsonl',source/'screened-validation.jsonl',source/'screened-confirmation.jsonl']:
        for row in read_rows(path):
            result.append({'task_id':row['task_id'],'instruction':row.get('instruction') or row['prompt'][-1]['content'].split('\n\nRequired signature:')[0]})
    return result


def make_suite(row,kind,seed):
    pairs,domain=cases_for(kind,row,seed)
    extra=[f"assert {row['entry_point']}({', '.join(repr(x) for x in inputs)}) == {expected!r}, 'independent_{i:04d}'" for i,(inputs,expected) in enumerate(pairs)]
    return list(dict.fromkeys(row['tests']+extra)),pairs,domain


def controls_for(row,kind):
    result=faulty_controls(kind,row['entry_point'])
    result += [{'name':'constant_'+str(i),'code':f"def {row['entry_point']}(*args,**kwargs):\n return {value}"} for i,value in enumerate(['None','False','True','0','1','-1',"''",'[]','{}'])]
    examples=literal_examples(row['tests'],row['entry_point'])
    lookup={repr((args,kwargs)):expected for args,kwargs,expected in examples}
    result.append({'name':'source_example_memorizer','code':f"def {row['entry_point']}(*args,**kwargs):\n table={lookup!r}\n return table.get(repr((args,kwargs)))"})
    for control in result:ast.parse(control['code'])
    return result


def run(args):
    if args.output.exists():raise ValueError('Review output exists; choose a new path')
    previous=json.loads((args.source/'quality-manifest.json').read_text())
    for split,digest in previous['protected_source_sha256'].items():
        if sha(args.mbpp/f'{split}.jsonl')!=digest:raise ValueError('Protected MBPP source changed')
    for split,digest in previous['files_sha256'].items():
        if sha(args.source/f'screened-{split}.jsonl')!=digest:raise ValueError('Prior frozen source changed')
    rows=read_rows(args.source/'mixed-train.jsonl')
    reviews={r['task_id']:r for r in json.loads(args.reviews.read_text())}
    if set(reviews)!={r['task_id'] for r in rows}:raise ValueError('Review IDs must match all prior mixed training tasks')
    old={r['task_id']:r for r in read_rows(args.source/'calibration-tasks.jsonl')}
    protected=load_protected(args.source,args.mbpp)
    decisions=[];groups=collections.defaultdict(list)
    for row in rows:
        review=reviews[row['task_id']]
        if row_sha(row)!=review['row_sha256']:raise ValueError('Review fingerprint mismatch: '+row['task_id'])
        if review['verdict']!='candidate_independent_contract':
            decisions.append({**review,'status':review['verdict']});continue
        kind=review['contract'];hits=protected_hits(kind,protected)
        if hits:
            decisions.append({**review,'status':'exclude_related_heldout_family','protected_task_ids':hits});continue
        groups[family(kind)].append(row)
    image=subprocess.run(['docker','image','inspect','--format','{{.Id}}','python-grpo-cpython:v2'],capture_output=True,text=True,check=True).stdout.strip()
    if image!=previous['image_id']:raise ValueError('Docker image changed')
    args.output.mkdir(parents=True)
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for name in ['strengthen_opencoder.py','reviewed_contracts.py','opencoder_data.py','educational_oracles.py','execution.py','cpython_worker.py','task_selection.py','calibrate_tasks.py','monitoring.py']:
        shutil.copy2(name,snapshot/name)
    shutil.copy2(args.reviews,snapshot/args.reviews.name)
    source_hashes={p.name:sha(p) for p in snapshot.iterdir()}
    (snapshot/'SHA256SUMS.json').write_text(json.dumps(source_hashes,indent=2))
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'image_id':image,'source_sha256':source_hashes,'prior_quality_manifest_sha256':sha(args.source/'quality-manifest.json'),'prior_mixed_train_sha256':sha(args.source/'mixed-train.jsonl'),'old_calibration_sha256':sha(args.source/'calibration-tasks.jsonl'),'selection_rule':'reference passes stronger suite; all 2 nonconstant + 9 constant + memorization controls fail; one representative per reviewed contract; then fresh train-only 1..7/8 calibration'}
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    local=threading.local();connections=[];lock=threading.Lock()
    def init():
        local.sandbox=Sandbox();local.sandbox.start()
        with lock:connections.append(local.sandbox)
    def check_group(group):
        audits=[]
        ordered=sorted(group,key=lambda r:(-r['provided_assertions'],len(r['reference']),r['task_id']))
        for row in ordered:
            kind=reviews[row['task_id']]['contract']
            tests,pairs,domain=make_suite(row,kind,args.seed)
            score=assertion_suite_score(local.sandbox,row['reference'],tests)
            audit={'task_id':row['task_id'],'contract':kind,'parent_row_sha256':row_sha(row),'domain':domain,'independent_cases':len(pairs),'reference_score':score}
            if not score['correct']:
                audits.append({**audit,'status':'reject_reference_disagrees_with_stronger_tests','oracle_cases':pairs});continue
            controls=[]
            for control in controls_for(row,kind):
                before=assertion_suite_score(local.sandbox,control['code'],row['tests'])
                after=assertion_suite_score(local.sandbox,control['code'],tests)
                if before['status']=='syntax_error' or after['status']=='syntax_error':raise RuntimeError('Faulty control itself has invalid syntax')
                controls.append({**control,'old_score':before,'strong_score':after})
            audit['controls']=controls
            if any(c['strong_score']['correct'] for c in controls):
                audits.append({**audit,'status':'reject_surviving_faulty_control','oracle_cases':pairs});continue
            approved={**row,'tests':tests,'quality_tier':'independent_contract_checked','independent_family':kind,'reviewed_contract_family':family(kind),'reviewed_domain':domain,'review_status':'behavior_and_fault_controls_checked_on_declared_domain','needs_semantic_review':False,'additional_assertions':len(tests)-row['provided_assertions'],'nonconstant_fault_controls_rejected':2,'constant_controls_rejected':9,'memorization_control_rejected':True,'parent_row_sha256':row_sha(row),'strong_suite_seed':args.seed}
            # Outcomes below cannot affect semantic/reference approval or representative choice.
            rescored=[]
            for candidate in old[row['task_id']]['candidates']:
                outcome=assertion_suite_score(local.sandbox,candidate['completion'],tests)
                rescored.append({'candidate':candidate['candidate'],'completion':candidate['completion'],'old_correct':candidate['correct'],'strong_score':outcome})
            audits.append({**audit,'status':'approved_on_declared_test_domain','oracle_cases':pairs,'frozen_baseline_rescores':rescored,'frozen_old_passes':sum(c['old_correct'] for c in rescored),'frozen_strong_passes':sum(c['strong_score']['correct'] for c in rescored)})
            chosen=row['task_id']
            for duplicate in ordered:
                if duplicate['task_id']!=chosen and not any(a['task_id']==duplicate['task_id'] for a in audits):
                    audits.append({'task_id':duplicate['task_id'],'status':'exclude_duplicate_reviewed_contract','representative_task_id':chosen,'contract':reviews[duplicate['task_id']]['contract']})
            return approved,audits
        return None,audits
    approved=[]
    print(f'[review] {len(rows)} tasks triaged; {len(decisions)} excluded/deferred before execution; {len(groups)} candidate contract groups',flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers,initializer=init) as pool:
            for i,(row,audits) in enumerate(pool.map(check_group,groups.values())):
                if row:approved.append(row)
                decisions.extend(audits)
                with (args.output/'checks.jsonl').open('a') as f:
                    for record in audits:f.write(json.dumps(record)+'\n')
                status={'phase':'reviewing','groups_completed':i+1,'groups_total':len(groups),'approved':len(approved),'updated_at':time.time()}
                (args.output/'status.json').write_text(json.dumps(status,indent=2))
                print(f'[review] {i+1}/{len(groups)} groups; approved={len(approved)}',flush=True)
    finally:
        for sandbox in connections:sandbox.close()
    if len({r['task_id'] for r in decisions})!=len(rows) or len(decisions)!=len(rows):raise ValueError('Every task must have exactly one final review record')
    (args.output/'decisions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in decisions))
    (args.output/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in approved))
    for split in ['validation','confirmation']:shutil.copy2(args.source/f'screened-{split}.jsonl',args.output/f'{split}.jsonl')
    positives=[d for d in decisions if d['status']=='approved_on_declared_test_domain']
    summary={'tasks_reviewed':len(rows),'approved_contracts_before_recalibration':len(approved),'decisions':dict(collections.Counter(r['status'] for r in decisions)),'fault_controls_rejected':12*len(approved),'nonconstant_fault_controls_rejected':2*len(approved),'controls_newly_rejected':sum(c['old_score']['correct'] and not c['strong_score']['correct'] for d in positives for c in d['controls']),'frozen_baseline_old_passes':sum(d['frozen_old_passes'] for d in positives),'frozen_baseline_strong_passes':sum(d['frozen_strong_passes'] for d in positives),'frozen_baseline_candidates':8*len(approved),'frozen_strong_buckets':dict(collections.Counter(calibration_bucket(d['frozen_strong_passes'],8) for d in positives)),'heldout_predictions_generated':False,'new_training_performed':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    manifest={'protocol':'opencoder-screened-v1','stage':'independent-reviewed-fault-controls-v2','counts_after':{'train':len(approved),'validation':27,'confirmation':28},'image_id':image,'source':previous['source'],'parent_manifest_sha256':sha(args.source/'quality-manifest.json'),'reviews_sha256':sha(args.reviews),'files_sha256':{s:sha(args.output/f'{s}.jsonl') for s in ['train','validation','confirmation']},'quality_tiers':dict(collections.Counter(r['quality_tier'] for r in approved)),'primary_evaluation':'Existing 29 expanded MBPP development tasks; existing 49 MBPP confirmation tasks remain unqueried. Copied OpenCoder validation/confirmation files remain unreviewed and are not approved primary evaluation.','confirmation_predictions_generated':False,'caveats':['Tests and fault controls are finite, domain-limited evidence, not a correctness proof.','Only previously mixed training tasks were reviewed; no held-out outcomes informed review.','Handwritten family filters can miss semantic overlap and can conservatively exclude distinct tasks.','No previously deferred instruction/algorithm constraint has been silently removed.'],'source_sha256':source_hashes}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'status.json').write_text(json.dumps({'phase':'complete','approved':len(approved),'updated_at':time.time()},indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('experiments/opencoder-data-v1'));p.add_argument('--reviews',type=Path,default=Path('opencoder_review_v2.json'));p.add_argument('--mbpp',type=Path,default=Path('data/prepared'));p.add_argument('--output',type=Path,default=Path('runs/opencoder-strengthened-v2'));p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20261108)
    run(p.parse_args())
