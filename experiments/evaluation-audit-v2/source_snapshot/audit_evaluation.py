"""Re-score frozen experiment outputs; never regenerate answers or alter historical results."""
import argparse
import ast
import collections
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

from evaluate import compare
from execution import Sandbox


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["task_id"] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate task IDs: {path}")
    return {row["task_id"]: row for row in rows}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records):
    summary = {}
    for split in sorted({row["split"] for row in records}):
        group = [row for row in records if row["split"] == split]
        for suite in ["original_cpython", "expanded_compatible"]:
            eligible = [row for row in group if row["reference_valid"] and (suite == "original_cpython" or row["expanded"].get("eligible"))]
            pairs = {name: [] for name in ["baseline", "final"]}
            statuses = {}
            for name in pairs:
                for row in eligible:
                    score = row["models"][name] if suite == "original_cpython" else row["expanded"][name]
                    pairs[name].append({"task_id": row["task_id"], "correct": score["correct"]})
                statuses[name] = dict(collections.Counter((row["models"][name] if suite == "original_cpython" else row["expanded"][name])["status"] for row in eligible))
            metrics = {"problems": len(eligible), "statuses": statuses}
            for name in pairs:
                metrics[name + "_solved"] = sum(row["correct"] for row in pairs[name])
                metrics[name + "_correctness"] = metrics[name + "_solved"] / len(eligible) if eligible else None
            if eligible:
                metrics["paired_comparison"] = compare(pairs["baseline"], pairs["final"], note="Frozen experiment-001 answers; paired problem-level bootstrap; evaluator audit, not a fresh benchmark.")
                if suite == "original_cpython":
                    metrics["historical_baseline_solved_on_same_tasks"] = sum(row["legacy"]["baseline_correct"] for row in eligible)
                    metrics["historical_final_solved_on_same_tasks"] = sum(row["legacy"]["final_correct"] for row in eligible)
            summary[split + "/" + suite] = metrics
        summary[split + "/coverage"] = {"saved_problems": len(group), "invalid_original_references": [row["task_id"] for row in group if not row["reference_valid"]], "expanded_exclusions": dict(collections.Counter(row["expanded"].get("reason", "eligible") for row in group))}
    return summary


def write_report(output, records, manifest):
    summary = summarize(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Evaluation audit: experiment 001", "", "The exact saved baseline and final-adapter answers were re-scored. No model generation or training was performed. Historical experiment-001 files remain unchanged.", "", "| Split and evaluator | Problems | Original model solved | Final adapter solved |", "|---|---:|---:|---:|"]
    for key, row in summary.items():
        if "problems" in row:
            lines.append(f"| {key} | {row['problems']} | {row['baseline_solved']} | {row['final_solved']} |")
    lines += ["", "## Interpretation and coverage", "", "Original-test results isolate the change from the restricted interpreter to CPython. Expanded results use EvalPlus 0.3.1 with MBPP+ v0.2.0, on the matched subset whose original reference passes both official base and extra tests. Eligibility is checked independently of either model's answers. Test inputs, types, floating-point tolerances, and special oracles use upstream EvalPlus code.", "", "This is a retrospective audit of old-prompt answers on a compatible subset, not a standard full MBPP+ benchmark or fresh evidence of general improvement. Uncovered tasks and reference/contract incompatibilities are explicitly reported. This audit must not be used to choose training settings against the inspected test set. New training needs a newly frozen evaluation protocol and fresh confirmation data.", "", "Original tests run in a fresh bounded CPython child per test; EvalPlus uses its official within-task state semantics. Results may differ because of interpreter behavior, oracle semantics, stronger tests, or reference incompatibilities. Passing three original tests is not proof of a correct oracle.", "", "## Coverage details", "", "```json", json.dumps({k:v for k,v in summary.items() if k.endswith('/coverage')}, indent=2), "```", "", "Reproduction metadata, source SHA-256 fingerprints, Docker image ID, dataset fingerprint, task-level scores, error types, and coverage exclusions are retained in `manifest.json`, `results.jsonl`, and `summary.json`."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main(args):
    source = args.source
    files = {"baseline_validation": source / "baseline-validation.jsonl", "final_validation": source / "validation-step-200.jsonl", "baseline_test": source / "baseline-test.jsonl", "final_test": source / "final-test.jsonl"}
    inputs = {name: read_rows(path) for name, path in files.items()}
    problems = {split: read_rows(args.data / f"{split}.jsonl") for split in ["validation", "test"]}
    with gzip.open(args.plus_data, "rt") as handle:
        plus = {int(row["task_id"].split("/")[-1]): row for row in (json.loads(line) for line in handle)}
    image_id = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", "python-grpo-cpython:v2"], check=True, capture_output=True, text=True).stdout.strip()
    manifest = {"protocol": "evaluation-audit-v2", "created_at": time.time(), "image_id": image_id, "source_files": {name: fingerprint(path) for name,path in files.items()}, "prepared_files": {split:fingerprint(args.data / f"{split}.jsonl") for split in problems}, "plus_data_sha256": fingerprint(args.plus_data), "evaluator_source_sha256": {name:fingerprint(Path(name)) for name in ["cpython_worker.py", "execution.py", "audit_evaluation.py"]}, "plus_version": "v0.2.0", "evalplus_version": "0.3.1", "limit_per_split": args.limit, "seed": "saved outputs, no new randomness", "original_address_space_limit_bytes": 536870912, "expanded_address_space_limit_bytes": 2147483648, "container_physical_memory_limit_bytes": 1073741824, "original_test_wall_seconds": 2, "plus_min_case_seconds": 0.2, "plus_ref_time_factor": 4.0}
    if args.output.exists() and not args.resume:
        raise ValueError("Audit output already exists; use a new directory or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_file = args.output / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_file.read_text())
        if {k:v for k,v in manifest.items() if k!='created_at'} != {k:v for k,v in previous.items() if k!='created_at'}:
            raise ValueError("Cannot resume with changed evaluator, image, inputs, or protocol")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    result_file = args.output / "results.jsonl"
    records = [json.loads(line) for line in result_file.read_text().splitlines()] if result_file.exists() else []
    done = {(row["split"], row["task_id"]) for row in records}
    with Sandbox() as sandbox:
        for split in ["validation", "test"]:
            baseline, final = inputs["baseline_" + split], inputs["final_" + split]
            if baseline.keys() != final.keys() or not baseline.keys() <= problems[split].keys():
                raise ValueError("Saved task sets are not identical or references are missing")
            ids = list(baseline)[:args.limit] if args.limit else list(baseline)
            for i, task_id in enumerate(ids):
                if (split,task_id) in done:
                    continue
                problem = problems[split][task_id]
                ref = sandbox.score(problem["reference"], problem["tests"])
                valid = ref["passed"] == ref["total"]
                row = {"split":split, "task_id":task_id, "reference_valid":valid, "reference":ref, "legacy":{"baseline_correct":baseline[task_id]["correct"], "final_correct":final[task_id]["correct"]}, "models":{}, "expanded":{"eligible":False}}
                if valid:
                    for name, saved in [("baseline", baseline[task_id]), ("final", final[task_id])]:
                        result = sandbox.score(saved["completion"], problem["tests"])
                        row["models"][name] = {**result, "correct":result["passed"] == result["total"]}
                expanded = row["expanded"]
                if not valid:
                    expanded["reason"] = "invalid_original_reference"
                elif task_id not in plus:
                    expanded["reason"] = "not_in_mbppplus"
                elif ast.parse(problem["signature"] + "\n pass").body[0].name != plus[task_id]["entry_point"]:
                    expanded["reason"] = "entry_point_mismatch"
                else:
                    plus_ref = sandbox.score_plus(problem["reference"], task_id)
                    expanded["reference"] = plus_ref
                    if plus_ref.get("status") == "oracle_error":
                        raise RuntimeError("EvalPlus infrastructure/oracle failed: " + str(plus_ref))
                    if plus_ref.get("status") in {"oracle_resource_limit", "memory_error", "timeout"}:
                        expanded["reason"] = "canonical_oracle_resource_limit"
                    elif not plus_ref.get("correct", False):
                        expanded["reason"] = "original_reference_failed_expanded_tests"
                    else:
                        expanded.update(eligible=True, reason="eligible", base_tests=len(plus[task_id]["base_input"]), extra_tests=len(plus[task_id]["plus_input"]))
                        for name, saved in [("baseline", baseline[task_id]), ("final", final[task_id])]:
                            result = sandbox.score_plus(saved["completion"], task_id)
                            if result.get("status") == "oracle_error":
                                raise RuntimeError("EvalPlus oracle failed: " + str(result))
                            expanded[name] = {**result, "correct":result.get("correct",False)}
                with result_file.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                records.append(row)
                status = {"phase":"evaluating", "split":split, "completed_in_split":i+1, "total_in_split":len(ids), "total_records":len(records), "updated_at":time.time()}
                (args.output / "status.json").write_text(json.dumps(status,indent=2))
                print(f"[{split}] {i+1}/{len(ids)} task={task_id} reference={valid} expanded={expanded['reason']}",flush=True)
    write_report(args.output, records, manifest)
    (args.output / "status.json").write_text(json.dumps({"phase":"complete","records":len(records),"updated_at":time.time()},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=Path("runs/experiment-001"))
    parser.add_argument("--data",type=Path,default=Path("data/prepared"))
    parser.add_argument("--plus-data",type=Path,default=Path("evaluation_assets/MbppPlus-v0.2.0.jsonl.gz"))
    parser.add_argument("--output",type=Path,default=Path("runs/evaluation-audit-v2"))
    parser.add_argument("--limit",type=int)
    parser.add_argument("--resume",action="store_true")
    main(parser.parse_args())
