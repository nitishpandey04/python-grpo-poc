# Python coding GRPO experiment ledger

We are tracking attempts to improve a tiny model's Python coding performance, including unsuccessful and inconclusive results. Model selection uses validation; test results are reported separately.

| Run | Date | Starting model | Change | Validation outcome | Final-adapter test result | Conclusion |
|---|---|---|---|---|---|---|
| [001](experiments/experiment-001/REPORT.md) | 2026-11-03 | Qwen2.5-Coder-0.5B-Instruct | 200 GRPO optimizer steps, LoRA rank 16, four candidates per prompt, test-fraction reward | No evaluated trained checkpoint beat baseline 29/79; baseline retained | 106/300 vs baseline 97/300; +3.0 percentage points | No validation-supported improvement; test-only gain warrants follow-up |

Code: https://github.com/nitishpandey04/python-grpo-poc

Final experimental adapter: https://huggingface.co/nitishpandey04/qwen2.5-coder-0.5b-python-grpo-poc

## Record the next experiment

Before training, record the hypothesis and the concrete change from experiment 001. Keep the starting model, dataset version, reward, training budget, random seed, decoding settings, hardware, and checkpoint-selection rule explicit. Use a new run name and retain its configuration, curves, and task-level evaluation outputs.

Tune using training and validation data. The experiment-001 test results have now been inspected, so repeated tuning against that set would weaken its role as an independent test. Confirm a promising approach on fresh held-out problems and additional training seeds.

Suggested record fields:

- Hypothesis and change under test.
- Run ID, code commit, model revision, data fingerprints, and seed.
- Training settings and reward definition.
- Baseline and checkpoint validation results.
- Selected checkpoint and final test comparison.
- Failure modes, limitations, and the next decision.
