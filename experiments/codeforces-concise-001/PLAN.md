# Concise-code prompt check and 50-update GRPO experiment

Started 2026-11-10. Results are pending; live records are stored locally under `runs/` and in Trackio.

Use Qwen3-4B-Instruct-2507 at pinned revision `cdbee75f17c01a7cc42f958dc650907174af0554` with NF4 QLoRA. Append only this instruction to the existing system message:

> Do not include comments or reasoning. Keep the solution concise and complete.

The Python code-fence format, problem statements, official tests, 1,024-token cap, seed and sampling configuration stay fixed. Compare against the previously completed NF4 probe on the identical 32 training tasks × eight samples: 137/256 passes, 16 mixed groups, 78 capped outputs. This is a practical one-seed training-data prompt comparison, not a claim of statistical significance or generalization.

The decision rule was frozen before new answers: use the concise prompt if passing candidates do not decrease and cap hits decrease; otherwise retain the original prompt. Mixed groups and difficulty breakdowns are reported. No task is removed and development is not consulted for prompt selection.

Then start 50 optimizer updates from the original pinned NF4 base, with a freshly computed 67-task greedy baseline under the selected prompt, and evaluate every ten updates. Keep four distinct prompts × eight candidates per update, micro batch 2, learning rate 1e-5 and rank-16 adapters. All 628 training tasks, including 324 medium tasks, remain eligible. A checkpoint is selected only for strict development improvement. Confirmation remains untouched; no model upload is scheduled.

The detached local controller performs both stages through the existing `train.py`. It stops on a failed stage, changed source/data or insufficient checkpoint disk space. Trackio remains on `0.0.0.0:7860`; run names are `codeforces-4b-concise-probe-001` and `codeforces-4b-grpo50-001`. All 32 regression checks passed before launch.

The exact plan, controller, source hashes and verification output accompany this file. Completed evidence will be archived after the run is reviewed.
