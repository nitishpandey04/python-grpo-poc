"""Create a readable before/after report from a completed experiment's saved outputs."""
import argparse
import json
from pathlib import Path

from evaluate import compare


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def build_report(run):
    before = read_rows(run / "baseline-test.jsonl")
    final = read_rows(run / "final-test.jsonl")
    selected = read_rows(run / "selected-test.jsonl")
    existing = json.loads((run / "report.json").read_text())
    metrics = read_rows(run / "metrics.jsonl")
    training = [row for row in metrics if "train/reward" in row]
    baseline_validation = json.loads((run / "baseline-validation.summary.json").read_text())
    selected_step = existing["selected_checkpoint"]["optimizer_step"]

    def rate(rows):
        return 100 * sum(row["correct"] for row in rows) / len(rows)

    lines = [
        f"# {run.name}: Python GRPO results", "",
        "| Test metric | Original model | Final trained adapter | Validation-selected model |",
        "|---|---:|---:|---:|",
        f"| Correct problems | {sum(r['correct'] for r in before)}/{len(before)} | {sum(r['correct'] for r in final)}/{len(final)} | {sum(r['correct'] for r in selected)}/{len(selected)} |",
        f"| Single-attempt correctness | {rate(before):.1f}% | {rate(final):.1f}% | {rate(selected):.1f}% |",
        f"| Mean generated tokens | {sum(r['tokens'] for r in before)/len(before):.1f} | {sum(r['tokens'] for r in final)/len(final):.1f} | {sum(r['tokens'] for r in selected)/len(selected):.1f} |", "",
        f"The baseline scored {baseline_validation['correctness']*100:.1f}% on {baseline_validation['problems']} validation problems. Validation selected step {selected_step} (step 0 means the original model was retained).", "",
    ]
    for name, rows in [("Final trained adapter", final), ("Validation-selected model", selected)]:
        result = compare(before, rows)
        lo, hi = result["ci95_percentage_points"]
        lines.append(f"{name}: {result['improvement_percentage_points']:+.1f} percentage points versus baseline; paired bootstrap 95% interval [{lo:+.1f}, {hi:+.1f}]. {result['improved']} problems improved and {result['regressed']} regressed.")
        lines.append("")
    if training:
        mixed = sum(row["train/frac_reward_zero_std"] == 0 for row in training)
        lines += [f"{len(training)} optimizer steps were logged; {mixed} had candidates with different correctness rewards. Peak PyTorch allocated memory: {existing['peak_gpu_allocated_mb']/1024:.2f} GiB.", ""]
    lines += ["## Scope", "", "Greedy decoding with the same prompts and token budget before and after training. Final test data was not used to select checkpoints. Results use a reference-verified subset of MBPP and its original small test suites. Pretraining contamination and semantic overlap remain possible; this is POC evidence, not an EvalPlus benchmark result.", "", "Training reward is averaged over changing sampled problems, so compare held-out correctness when assessing learning. A zero-width interval for a retained baseline simply reflects comparing identical model outputs.", "", "Detailed code and execution results are in `baseline-test.jsonl`, `final-test.jsonl`, and `selected-test.jsonl`. Configuration, dataset fingerprints, metrics, and resumable checkpoints are retained alongside this report."]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    build_report(parser.parse_args().run)
