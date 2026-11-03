# experiment-001: Python GRPO results

| Test metric | Original model | Final trained adapter | Validation-selected model |
|---|---:|---:|---:|
| Correct problems | 97/300 | 106/300 | 97/300 |
| Single-attempt correctness | 32.3% | 35.3% | 32.3% |
| Mean generated tokens | 87.9 | 76.3 | 87.9 |

The baseline scored 36.7% on 79 validation problems. Validation selected step 0 (step 0 means the original model was retained).

Final trained adapter: +3.0 percentage points versus baseline; paired bootstrap 95% interval [+0.3, +5.7]. 13 problems improved and 4 regressed.

Validation-selected model: +0.0 percentage points versus baseline; paired bootstrap 95% interval [+0.0, +0.0]. 0 problems improved and 0 regressed.

200 optimizer steps were logged; 91 had candidates with different correctness rewards. Peak PyTorch allocated memory: 1.12 GiB.

## Scope

Greedy decoding with the same prompts and token budget before and after training. Final test data was not used to select checkpoints. Results use a reference-verified subset of MBPP and its original small test suites. Pretraining contamination and semantic overlap remain possible; this is POC evidence, not an EvalPlus benchmark result.

Training reward is averaged over changing sampled problems, so compare held-out correctness when assessing learning. A zero-width interval for a retained baseline simply reflects comparing identical model outputs.

Detailed code and execution results are in `baseline-test.jsonl`, `final-test.jsonl`, and `selected-test.jsonl`. Configuration, dataset fingerprints, metrics, and resumable checkpoints are retained alongside this report.
