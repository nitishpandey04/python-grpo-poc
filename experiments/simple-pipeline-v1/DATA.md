# Simplified data preparation — 2026-11-09

The original pinned 3,000-row OpenCoder download is unchanged. Preparation yields **2,613 eligible training problems**, not independently certified gold problems.

| Exclusion | Rows |
|---|---:|
| Exact/near-identical statement duplicates | 302 |
| Existing reserved OpenCoder held-out rows | 55 |
| Statement overlap with protected evaluation | 22 |
| Syntax errors in supplied test programs | 2 |
| Setup-only programs without checks | 2 |
| Documented setup using root before assignment | 3 |
| Documented incorrect example in statement | 1 |

Native test statement order and setup are preserved. Reference solutions are optional interface/debug metadata, never reward targets or a pass/fail gate. All five problems previously excluded because references failed stronger tests are eligible in this source-test corpus. This does not repair or certify their supplied tests: expectations and coverage may still be imperfect. There is no minimum assertion count, code-length gate, automatic task-family exclusion or baseline-difficulty selection.

`data-exclusions.jsonl` records every excluded task ID and duplicate/overlap representative. Data source revision, sampling seed, raw SHA-256 and frozen train/development hashes are in `data-manifest.json`. The prepared train data is reproducible using the pinned downloader, the current prepare_data.py and the versioned evaluation inputs; it is not duplicated in this archive.

The active flow is prepare_data.py → train.py → report.py with execution.py/cpython_worker.py and evaluate.py. Nineteen superseded root scripts, metadata files and feature-specific tests were retired. Their original versions and historical experiments remain available in Git and existing experiment snapshots. The current executable source is versioned by tag simple-pipeline-v1, with exact source hashes captured in config.json; no additional source-copy pipeline is introduced.

The smoke run completes ten optimizer steps, samples 40 distinct training problems and records 21/40 mixed-reward groups. Fixed expanded-MBPP development correctness changes from 16/29 to 15/29; the original model remains selected. Peak PyTorch allocation is 1.46 GiB. This verifies the simplified path and does not establish a model improvement. The 49-task MBPP confirmation pool and the 55 reserved OpenCoder tasks receive no model predictions.

All 23 active regression checks passed. No new model weights are uploaded; the smoke adapter remains local. The original published adapter is unchanged.
