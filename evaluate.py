"""Fixed greedy decoding and paired, problem-level bootstrap comparison."""
import json
import hashlib
import collections
import random
from pathlib import Path


def evaluate(model, tokenizer, rows, sandbox, output, max_new_tokens=256, monitor=None, prefix="eval", score_fn=None):
    if not rows or any(not row.get("tests") for row in rows):
        raise ValueError("Evaluation requires nonempty problems and test suites")
    import torch
    was_training = model.training
    model.eval()
    results = []
    try:
        for i, row in enumerate(rows):
            if monitor:
                monitor.phase = f"{prefix}: generating {i + 1}/{len(rows)}"
            text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, use_cache=True)
            completion = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if monitor:
                monitor.phase = f"{prefix}: testing {i + 1}/{len(rows)}"
            score = score_fn(sandbox, completion, row["tests"], row["task_id"]) if score_fn else sandbox.score(completion, row["tests"])
            results.append({"task_id": row["task_id"], "completion": completion, "correct": score["passed"] == score["total"], "tokens": tokens.shape[1] - inputs.input_ids.shape[1], **score})
            if monitor:monitor.log({f'{prefix}/completed':i+1,f'{prefix}/total':len(rows),f'{prefix}/percent':100*(i+1)/len(rows)})
            print(f"[{prefix}] {i + 1}/{len(rows)}: {sum(r['correct'] for r in results)}/{len(results)} correct", flush=True)
        metrics = {"correctness": sum(r["correct"] for r in results) / len(results), "mean_tokens": sum(r["tokens"] for r in results) / len(results), "problems": len(results)}
        if all(r.get('checked',r['total'])==r['total'] for r in results):
            metrics['mean_test_fraction']=sum(r['passed']/r['total'] for r in results)/len(results)
        metrics['mean_checked_tests']=sum(r.get('checked',r['total']) for r in results)/len(results)
        metrics.update(evaluator=sandbox.backend, correctness_protocol=score_fn.__name__ if score_fn else "original_tests", failure_counts=dict(collections.Counter(r["status"] for r in results)))
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in results))
        path.with_suffix(".summary.json").write_text(json.dumps(metrics, indent=2))
        if monitor:
            monitor.log({f"{prefix}/{k}": v for k, v in metrics.items()})
        return results, metrics
    finally:
        model.train(was_training)


def compare(before, after, note="Greedy single-attempt evaluation; paired bootstrap over tasks; filtered MBPP original tests."):
    before_map = {r["task_id"]: r for r in before}
    after_map = {r["task_id"]: r for r in after}
    if len(before_map) != len(before) or len(after_map) != len(after):
        raise ValueError("Duplicate task IDs in paired comparison")
    if not before_map or before_map.keys() != after_map.keys():
        raise ValueError("Comparison requires identical nonempty task sets")
    deltas = [int(after_map[k]["correct"]) - int(before_map[k]["correct"]) for k in before_map]
    rng = random.Random(42)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(5000))
    return {"problems": len(deltas), "improvement_percentage_points": 100 * sum(deltas) / len(deltas), "ci95_percentage_points": [100 * samples[125], 100 * samples[4874]], "improved": deltas.count(1), "regressed": deltas.count(-1), "note": note}


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task IDs in {path}")
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
