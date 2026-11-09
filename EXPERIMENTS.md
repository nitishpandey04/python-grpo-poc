# Python coding GRPO experiment ledger

Historical sections describe the code at their recorded tags. The current active workflow is in [README.md](README.md); retired scripts remain recoverable through Git and frozen experiment snapshots.

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

## Evaluation audit — 2026-11-07

[Report and frozen source snapshot](experiments/evaluation-audit-v2/REPORT.md). Existing answers were re-scored without training or generation. The corrected CPython evaluator accepts ordinary Python constructs and records actual assertion/runtime failures.

| Protocol | Validation: original / final | Test: original / final |
|---|---:|---:|
| Original tests under CPython | 30/79 / 25/79 | 99/300 / 106/300 |
| Expanded compatible subset | 16/29 / 14/29 | 40/96 / 43/96 |

Expanded results use EvalPlus 0.3.1 with MBPP+ v0.2.0. Seven validation and 35 test references failed expanded checks; uncovered tasks, one entry-point mismatch, and one canonical-oracle memory limit are also excluded explicitly. This is an audited subset of old-prompt answers, not a standard full MBPP+ score. No validation-supported improvement was established. Historical experiment-001 records and its published weights remain unchanged.

Next training work should review reference/test quality, freeze a suitable evaluation protocol, then calibrate training-task difficulty and broaden batches. No second training run had started at the time of this evaluation audit.

## Verified selection and batching pilot — 2026-11-07

[Full pipeline report](experiments/pipeline-v2/REPORT.md).

- Stronger reference checks retained 66 training tasks, 29 development tasks, and 49 untouched confirmation tasks.
- Eight baseline samples per training task selected 27 mixed-outcome tasks; 37 had no passing samples and two had all samples pass.
- The 10-step GPU pilot used four distinct prompts and eight candidates per update, with binary original-plus-expanded correctness rewards.
- Runtime checks confirmed 32 candidates per optimizer update. Mixed rewards occurred in 31/40 prompt groups (77.5%).
- Development validation was 16/29 before and after; one problem improved and one regressed. The original model remains selected.
- Peak PyTorch allocation was 1.25 GiB. Confirmation predictions were not generated.

This combined pipeline pilot demonstrates verified selection, useful reward comparisons, and working batch composition. It does not demonstrate model improvement or isolate the effects of individual changes. The next bottleneck is the small verified, learnable training pool; expand it before a longer run, and use matched-budget comparisons and additional seeds before spending confirmation data.

## OpenCoder data acquisition and calibration — 2026-11-07

[Report, frozen data, review queue, and source snapshot](experiments/opencoder-data-v1/REPORT.md).

- Pinned 3,000 educational rows; grouping and screening retained 306 problems: 251 training, 27 development, and 28 additional confirmation tasks.
- Fourteen retained tasks received independently computed narrow-contract cases; 292 passed only supplied-test and nine-constant-control screening and still need semantic review.
- Eight original-model samples on each training task produced 130 mixed, 86 all-fail, and 35 all-pass tasks.
- The 130 mixed tasks comprise 124 pending semantic review and six with independent contract checks.
- Code-and-instruction review flagged four of the 13 independent-contract training rows for instruction/domain issues. 5 mixed tasks remain after those exclusions, with correctness claims limited to their tested domains.
- No model predictions were generated on the new development/confirmation splits or the existing 49-task MBPP confirmation pool. No new model was trained.

This first data import did not achieve hundreds of independently verified learnable tasks. The larger mixed pool is a review queue, not approved training data. Keep the existing expanded MBPP evaluation, strengthen tests and semantic checks, and integrate the separate assertion-suite reward protocol only after freezing approved data.

## Stronger task review and recalibration — 2026-11-07

[Full review report and frozen evidence](experiments/opencoder-reviewed-v2/REPORT.md).

- Reviewed all 130 formerly mixed tasks: 60 related held-out-family exclusions, 33 deferred instruction/algorithm contracts, three collapsed duplicates, five reference failures, and 29 retained contracts.
- Each retained contract passes independently computed bounded cases and rejects two nonconstant faulty implementations, nine constant outputs, and a lookup memorizing old examples.
- On the same 232 frozen baseline answers, stronger criteria reduce passing answers from 102 to 91. This measures stricter testing, not model training or regression.
- Fresh eight-answer baseline calibration selects 22 mixed tasks; six are all-fail and one all-pass.
- All 31 regression checks passed. No additional model was trained. Existing expanded MBPP development data and the 49-task unqueried confirmation pool remain unchanged.

The resulting 22-task manifest is a stronger, small input pool for a controlled run, with separate assertion-suite training and expanded-MBPP development scorers required. It does not establish a model improvement, guarantee full semantic decontamination, or reach the target of hundreds of diverse verified problems.

## Simplified native-program pipeline — 2026-11-09

[Smoke report](experiments/simple-pipeline-v1/REPORT.md) · [Data accounting and simplification](experiments/simple-pipeline-v1/DATA.md).

- Retired 19 superseded root files and kept one active preparation/training/evaluation path. Historical source and results remain available.
- Reprocessed the original 3,000 rows without reference-pass, assertion-count, code-length, task-family or difficulty gates. Retained 2,613 source-test-eligible training problems.
- Preserved native test setup and statement order. All 23 active regression checks passed.
- The 10-step smoke run sampled 40 distinct problems; 21/40 prompt groups had mixed rewards. Peak PyTorch allocation was 1.46 GiB.
- Fixed expanded-MBPP development correctness: 16/29 original, 15/29 final. No checkpoint beat the original model, which remains selected.
- No confirmation model predictions were generated. The source tests remain imperfect; corpus eligibility is not independent correctness certification.

This completes the simplification and an end-to-end smoke check. It does not establish model improvement or justify selecting the smoke adapter. Full-budget training is a separate experiment.

## Codeforces source exploration — 2026-11-09

[Dataset inspection and boundary audit](experiments/codeforces-exploration/REPORT.md).

Scanned the pinned Open-R1 `verifiable/train` metadata: 8,338 tasks, 1,654 complete official suites, 1,247 with complete suites and a simple standard-I/O/no-custom-checker interface, and 344 in that scope rated 800–1000. Those 344 average 45.16 official I/O records (median 38); no minimum test-count filter was used.

Downloaded one 26 MB shard and inspected 27 easier problems with 985 official records. Four simple tasks were checked independently against all 142 supplied outputs, with full agreement. Beautiful Matrix's 25 tests cover all 25 valid board states. Complete official suites still need not be exhaustive for larger domains.

No model predictions, training changes or test-split reads were performed. The next step is a small standard-I/O execution path and a baseline-only check on the cached sample before deciding to train on this source.
