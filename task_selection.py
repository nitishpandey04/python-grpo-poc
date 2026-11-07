"""Train-only reference verification, leakage guards, and calibration selection rules."""
import argparse
import ast
import collections
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

from execution import Sandbox


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task IDs in {path}")
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_disjoint(*groups):
    seen = set()
    for group in groups:
        ids = {row["task_id"] for row in group}
        if len(ids) != len(group) or seen & ids:
            raise ValueError("Duplicate or overlapping task IDs across data splits")
        seen.update(ids)


def calibration_bucket(correct, samples):
    if samples < 2 or not 0 <= correct <= samples:
        raise ValueError("Invalid calibration counts")
    return "hard" if correct == 0 else "easy" if correct == samples else "mixed"


def validate_generation_groups(ids, prompts, generations):
    if len(ids) != prompts * generations:
        raise ValueError("Incomplete generation batch")
    groups = [ids[i:i+generations] for i in range(0,len(ids),generations)]
    if any(len(set(group)) != 1 for group in groups) or len({group[0] for group in groups}) != prompts:
        raise ValueError("Generation batch must have distinct prompt groups with exactly G candidates each")
    return {"distinct_prompts": prompts, "candidates_per_prompt": generations, "candidates": len(ids)}


def expanded_score(sandbox, completion, tests, task_id):
    original = sandbox.score(completion, tests)
    result = {**original, "correct": False}
    if original["passed"] != original["total"]:
        result["expanded"] = {"status":"not_run", "reason":"original_tests_failed"}
        return result
    plus = sandbox.score_plus(completion,task_id)
    if plus.get("status") in {"oracle_error", "oracle_resource_limit"}:
        raise RuntimeError(f"Verified task oracle failed: {task_id}: {plus}")
    result["expanded"] = plus
    result["correct"] = bool(plus.get("correct",False))
    if not result["correct"]:
        result["status"] = "timeout" if plus.get("status") == "timeout" or any(s.get("status")=="timeout" for s in plus.get("suites",{}).values()) else "expanded_test_failure"
    return result


def verify(args):
    if args.output.exists() and not args.resume:
        raise ValueError("Output already exists; choose a new quality run directory or --resume")
    args.output.mkdir(parents=True,exist_ok=True)
    pools = {split:read_rows(args.data / f"{split}.jsonl") for split in ["train","validation","test"]}
    # Only historical task IDs are read; prior model outcomes never guide selection.
    prior_ids = {row["task_id"] for row in read_rows(args.previous / "baseline-test.jsonl")}
    pools["confirmation"] = [row for row in pools.pop("test") if row["task_id"] not in prior_ids]
    assert_disjoint(pools["train"],pools["validation"],pools["confirmation"])
    with gzip.open(args.plus_data,"rt") as handle:
        plus={int(row["task_id"].split("/")[-1]):row for row in map(json.loads,handle)}
    image_id=subprocess.run(["docker","image","inspect","--format","{{.Id}}","python-grpo-cpython:v2"],capture_output=True,text=True,check=True).stdout.strip()
    manifest={"protocol":"verified-task-pools-v2","image_id":image_id,"created_at":time.time(),"source_sha256":{split:sha(args.data/f"{split}.jsonl") for split in ["train","validation","test"]},"previous_test_ids_sha256":hashlib.sha256(json.dumps(sorted(prior_ids)).encode()).hexdigest(),"plus_data_sha256":sha(args.plus_data),"selection_uses":"training reference quality only; model difficulty calibration is a subsequent train-only stage","reward":"binary: all original + official base/extra tests pass","counts_before":{split:len(rows) for split,rows in pools.items()}}
    if args.resume:
        previous=json.loads((args.output/'manifest.json').read_text())
        for key in ['protocol','image_id','source_sha256','previous_test_ids_sha256','plus_data_sha256','counts_before']:
            if previous.get(key)!=manifest[key]:raise ValueError('Cannot resume with changed quality inputs/image')
        audit=[json.loads(line) for line in (args.output/'references.jsonl').read_text().splitlines()]
    else:
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));audit=[]
    done={(r['split'],r['task_id']) for r in audit}
    accepted={split:[row for row in rows if any(r['split']==split and r['task_id']==row['task_id'] and r['reason']=='verified' for r in audit)] for split,rows in pools.items()}
    with Sandbox() as sandbox:
        for split,rows in pools.items():
            for i,row in enumerate(rows):
                if (split,row['task_id']) in done:continue
                record={"split":split,"task_id":row["task_id"]}
                if row["task_id"] not in plus:
                    record['reason']='not_in_mbppplus'
                elif not plus[row['task_id']]['base_input'] or not plus[row['task_id']]['plus_input']:
                    record['reason']='empty_expanded_suite'
                elif ast.parse(row['signature']+'\n pass').body[0].name != plus[row['task_id']]['entry_point']:
                    record['reason']='entry_point_mismatch'
                else:
                    original=sandbox.score(row['reference'],row['tests']);record['original']=original
                    if original['passed'] != original['total']:
                        record['reason']='original_reference_failed'
                    else:
                        expanded=sandbox.score_plus(row['reference'],row['task_id']);record['expanded']=expanded
                        if expanded.get('status')=='oracle_error':
                            raise RuntimeError(str(expanded))
                        if expanded.get('status') in {'oracle_resource_limit','memory_error','timeout'}:
                            record['reason']='canonical_oracle_resource_limit'
                        elif not expanded.get('correct',False):
                            record['reason']='reference_failed_expanded_tests'
                        else:
                            record['reason']='verified'
                            accepted[split].append(row)
                audit.append(record)
                with (args.output/'references.jsonl').open('a') as handle:
                    handle.write(json.dumps(record)+'\n')
                print(f"[{split}] {i+1}/{len(rows)} task={row['task_id']} {record['reason']}",flush=True)
                (args.output/'status.json').write_text(json.dumps({"phase":"verifying_references","split":split,"completed":i+1,"total":len(rows),"updated_at":time.time()}))
    assert_disjoint(*accepted.values())
    for split,rows in accepted.items():
        (args.output/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest['counts_after']={split:len(rows) for split,rows in accepted.items()}
    manifest['exclusions']={split:dict(collections.Counter(r['reason'] for r in audit if r['split']==split)) for split in pools}
    manifest['files_sha256']={split:sha(args.output/f'{split}.jsonl') for split in accepted}
    manifest['confirmation_task_ids']=[r['task_id'] for r in accepted['confirmation']]
    manifest['confirmation_predictions_generated']=False
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'status.json').write_text(json.dumps({"phase":"complete","counts":manifest['counts_after'],"updated_at":time.time()}))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data/prepared'))
    parser.add_argument('--previous',type=Path,default=Path('runs/experiment-001'))
    parser.add_argument('--plus-data',type=Path,default=Path('evaluation_assets/MbppPlus-v0.2.0.jsonl.gz'))
    parser.add_argument('--output',type=Path,default=Path('runs/task-quality-v2'))
    parser.add_argument('--resume',action='store_true')
    verify(parser.parse_args())
