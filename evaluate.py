"""Fixed greedy decoding and paired, problem-level bootstrap comparison."""
import json
import random
from pathlib import Path


def evaluate(model, tokenizer, rows, sandbox, output, max_new_tokens=256, monitor=None, prefix="eval"):
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
            score = sandbox.score(completion, row["tests"])
            results.append({"task_id": row["task_id"], "completion": completion, "correct": score["passed"] == score["total"], "tokens": tokens.shape[1] - inputs.input_ids.shape[1], **score})
            print(f"[{prefix}] {i + 1}/{len(rows)}: {sum(r['correct'] for r in results)}/{len(results)} correct", flush=True)
        metrics = {"correctness": sum(r["correct"] for r in results) / len(results), "mean_test_fraction": sum(r["passed"] / r["total"] for r in results) / len(results), "mean_tokens": sum(r["tokens"] for r in results) / len(results), "problems": len(results)}
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in results))
        path.with_suffix(".summary.json").write_text(json.dumps(metrics, indent=2))
        if monitor:
            monitor.log({f"{prefix}/{k}": v for k, v in metrics.items()})
        return results, metrics
    finally:
        model.train(was_training)


def compare(before, after):
    before_map = {r["task_id"]: r for r in before}
    after_map = {r["task_id"]: r for r in after}
    if not before_map or before_map.keys() != after_map.keys():
        raise ValueError("Comparison requires identical nonempty task sets")
    deltas = [int(after_map[k]["correct"]) - int(before_map[k]["correct"]) for k in before_map]
    rng = random.Random(42)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(5000))
    return {"problems": len(deltas), "improvement_percentage_points": 100 * sum(deltas) / len(deltas), "ci95_percentage_points": [100 * samples[125], 100 * samples[4874]], "improved": deltas.count(1), "regressed": deltas.count(-1), "note": "Greedy single-attempt evaluation; paired bootstrap over tasks; filtered MBPP original tests."}
