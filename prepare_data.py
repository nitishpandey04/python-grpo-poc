"""Prepare a reference-verified MBPP subset; original split boundaries are preserved."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from execution import Sandbox


def prepare(source, output, backend="cpython"):
    if output.exists() and any(output.glob("*.jsonl")):
        raise ValueError("Prepared data already exists; choose a new --output directory to preserve frozen data")
    output.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "validation": [], "test": []}
    seen = set()
    rejected = []
    with Sandbox(timeout=8, backend=backend) as sandbox:
        for line in source.read_text().splitlines():
            row = json.loads(line)
            task_id = row["task_id"]
            split = "train" if task_id >= 601 else "validation" if task_id >= 511 else "test" if task_id >= 11 else None
            if split is None:
                continue
            try:
                tree = ast.parse(row["code"])
                funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                if not funcs or row.get("test_setup_code", "").strip():
                    raise ValueError("Unsupported setup")
                # Obtain the required entry point from tests, rather than an arbitrary helper.
                names = {n.name: n for n in funcs}
                calls = [n.func.id for n in ast.walk(ast.parse(row["test_list"][0])) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names]
                if not calls:
                    raise ValueError("No simple function entry point")
                function = names[calls[0]]
                signature = f"def {function.name}({ast.unparse(function.args)}):"
                if len(row["code"]) > 2500:
                    raise ValueError("Reference too long for first experiment")
                key = hashlib.sha256(" ".join(row["text"].lower().split()).encode()).hexdigest()
                if key in seen:
                    raise ValueError("Duplicate problem statement")
                seen.add(key)
                score = sandbox.score(row["code"], row["test_list"])
                if score["passed"] != score["total"]:
                    raise ValueError("Reference failed: " + score["status"])
                prompt = [
                    {"role": "system", "content": "Write correct Python functions. Return only Python code in a single ```python code block. Include the requested function and any helper functions. Do not include example calls, print statements, or explanations."},
                    {"role": "user", "content": row["text"] + "\n\nRequired signature:\n" + signature},
                ]
                splits[split].append({"task_id": task_id, "prompt": prompt, "tests": row["test_list"], "reference": row["code"], "signature": signature})
                print(f"accepted {task_id}: " + str({k: len(v) for k, v in splits.items()}), flush=True)
            except (SyntaxError, ValueError) as exc:
                rejected.append({"task_id": task_id, "reason": str(exc)})
    for split, rows in splits.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"evaluator_backend": backend, "source": "https://github.com/google-research/google-research/tree/master/mbpp", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "counts": {k: len(v) for k, v in splits.items()}, "rejected": rejected, "scope": "Reference-verified subset; original MBPP splits; three original tests per task, not expanded EvalPlus tests."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/mbpp.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--execution-backend", choices=["cpython", "openenv"], default="cpython")
    args = parser.parse_args()
    prepare(args.source, args.output, args.execution_backend)
