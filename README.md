# Python GRPO on a 16 GB GPU

Local Python coding experiments with Qwen2.5-Coder-0.5B-Instruct, LoRA, TRL GRPO, bounded CPython execution in Docker, and a local Trackio dashboard. Experiment 001 used the preserved legacy OpenEnv interpreter. No cloud training or hosted logging is required.

## Experiment status

**Experiment 001 is complete. It did not establish a validation-supported improvement.** None of the evaluated trained checkpoints beat the original model on validation, so the selection rule retained the original model.

The final step-200 adapter separately scored 106/300 (35.3%) on the held-out test subset versus 97/300 (32.3%) for the original model. This test-set gain is exploratory evidence, not a claim that the trained model is generally better. Original MBPP tests are limited, contamination is possible, and only one training seed was run.

**Evaluation audit (2026-11-07):** Corrected CPython evaluation gives 30/79 versus 25/79 on validation and 99/300 versus 106/300 on test (original versus final adapter). Expanded EvalPlus checks on compatible subsets give 16/29 versus 14/29 on validation and 40/96 versus 43/96 on test. These are retrospective scores on frozen answers, with coverage exclusions reported; they do not establish improvement. [Audit report](experiments/evaluation-audit-v2/REPORT.md).

- [Experiment ledger](EXPERIMENTS.md)
- [Full experiment report](experiments/experiment-001/REPORT.md)
- [Experimental LoRA adapter on Hugging Face](https://huggingface.co/nitishpandey04/qwen2.5-coder-0.5b-python-grpo-poc)

The published model is the final trained adapter for reproducibility and future comparison. It is not the model selected by validation. It requires the original Qwen base model; loading instructions are in the model card.

## What this version measures

Single-submission Python function correctness. Each training problem gets four sampled candidates. Reward is the fraction of original MBPP tests passed. The separate greedy evaluator counts a problem as correct only if **all** its tests pass. Generated code never runs on the host.

Evaluation v2 uses normal CPython, with one fresh child process per original test, a fixed hash seed, precise failure labels, and memory/time limits inside Docker. Function defaults, classes, printing, and ordinary Python imports are supported. Experiment 001's legacy OpenEnv API remains available through `--execution-backend openenv`; its historical results are unchanged. Interactive execution/repair is a later experiment.

## Setup

Python 3.12, NVIDIA driver, `uv`, and Docker are required. The RTX 5060 Ti requires a Blackwell-compatible PyTorch build; this project uses CUDA 12.8 wheels.

```bash
uv --cache-dir /tmp/python-grpo-uv venv --python /usr/bin/python3 .venv
uv --cache-dir /tmp/python-grpo-uv pip install --python .venv/bin/python torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv --cache-dir /tmp/python-grpo-uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
mkdir -p evaluation_assets
curl -L --fail https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz -o evaluation_assets/MbppPlus-v0.2.0.jsonl.gz
docker build -t python-grpo-cpython:v2 -f Dockerfile.evaluation .
mkdir -p data
curl -L --fail https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl -o data/mbpp.jsonl
.venv/bin/python prepare_data.py
```

TRL's fused GPU kernels also require a C compiler and Python development headers. If `Python.h` is missing on Ubuntu, download and extract the headers locally (no system installation):

```bash
mkdir -p .cache/python-headers
cd .cache/python-headers
apt download libpython3.12-dev
dpkg-deb -x libpython3.12-dev_*.deb .
cd ../..
```

`train.py` automatically adds these local headers to the compiler's include path.

The final `requirements.lock.txt` records the exact host packages used in the verified run. The source SHA-256 and accepted/rejected tasks are recorded in `data/prepared/manifest.json`.

## Run and watch live

Run the dashboard in another terminal using the same Trackio directory:

```bash
TRACKIO_DIR="$PWD/runs/trackio" .venv/bin/trackio show --project python-grpo
```

Open the local URL printed by Trackio. Metrics include training reward, zero-variance reward groups, entropy, validation correctness, output lengths, error fractions, GPU memory/utilization/power, and elapsed time. Sampled solutions appear every five optimizer steps; all training rollouts are also saved locally. Select the appropriate run in the dashboard.

Smoke test:

```bash
.venv/bin/python -u train.py --name smoke --steps 10 --eval-every 5 --val-size 8 --test-size 12
```

First experiment:

```bash
.venv/bin/python -u train.py --name experiment-001 --steps 200 --eval-every 25
```

To leave the dashboard or training running after the terminal closes, use the launcher instead:

```bash
.venv/bin/python launch.py dashboard
.venv/bin/python launch.py train --name experiment-001 --steps 200 --eval-every 25
```

It saves the PID and `console.log` under `runs/dashboard/` or `runs/<name>/`. The launcher binds the dashboard to `0.0.0.0:7860`, so it is available locally and through this computer's Tailscale IP at `http://<tailscale-ip>:7860/`.

Default evaluation caps are 100 validation and 300 test tasks, or the entire accepted split if smaller. Each evaluation uses identical greedy decoding and a 256-token budget. Baseline validation/test outputs are saved before training. Validation chooses the best adapter; if no checkpoint strictly improves validation, the baseline is retained. The final trained adapter is evaluated and reported separately even if it is not selected.

Each run writes to `runs/<name>/`:

- `status.json`: phase and heartbeat, updated every 15 seconds.
- `metrics.jsonl`: scalar measurements, including actual optimizer step.
- `rollouts.jsonl`: generated training code, score, failures, and task ID.
- `baseline-*.jsonl`, `validation-step-*.jsonl`, `selected-test.jsonl`, `final-test.jsonl`: evaluation evidence.
- `checkpoints/`: two most recent resumable checkpoints.
- `final-adapter/` and, if validation improves, `best-adapter/`.
- `report.json`: paired changes, improved/regressed tasks, bootstrap confidence intervals, peak allocated GPU memory.
- `REPORT.md`: a readable before/after summary. Generate it for an existing completed run with `.venv/bin/python report.py runs/<name>`.

To resume an interrupted run, preserve the name and settings:

```bash
.venv/bin/python -u train.py --name experiment-001 --steps 200 --eval-every 25 --execution-backend openenv --resume runs/experiment-001/checkpoints/checkpoint-50
```

Trackio's event index includes heartbeats and scalar events. Use `progress/optimizer_step` to distinguish optimization progress from telemetry. A long generation can take time between training updates; the heartbeat shows that the process is alive.

## Validation and limits

```bash
.venv/bin/python -m unittest discover -s tests -p test_evaluation.py
.venv/bin/python -m unittest discover -s tests -p test_sandbox.py
```

The Docker worker runs without network, mounted host files, root privileges, or Linux capabilities. Its root filesystem is read-only, with a writable bounded temporary directory, CPU/PID limits, and a 1 GiB physical-memory limit. Original tests each run in a new CPython child with a 512 MiB virtual-address limit and up to two seconds wall time. EvalPlus needs a 2 GiB virtual-address allowance for its shared-library mappings while remaining inside the same 1 GiB Docker physical-memory cap.

Failures distinguish `assertion_error`, `syntax_error`, `runtime_error`, `timeout`, and `memory_error`. Missing/malformed child responses and early exits cannot count as passes. Empty test suites, invalid trusted tests, and infrastructure errors fail closed. Candidate output is captured and bounded. EvalPlus's official checker/oracles are invoked only inside Docker; neither candidate code nor canonical reference code runs on the training host.

Data preparation preserves MBPP's original split boundaries, rejects duplicate problem text, and keeps only tasks whose reference solutions pass the selected executor. **Semantic overlap between splits and pretraining contamination remain possible.** This is a filtered MBPP POC with the original small test suites, not an EvalPlus result or evidence of broad coding ability. The audit below adds expanded tests on compatible tasks. Fresh confirmation data, task-family deduplication, multiple training seeds, and multi-turn debugging are follow-up work.

The 200-step budget is a cap, not a promised improvement. Training loss/reward alone does not establish success. Read validation and final-test correctness, confidence intervals, and token usage together.

MBPP source and licensing: https://github.com/google-research/google-research/tree/master/mbpp

OpenEnv source/license is retained under `vendor/coding_env/`.

## Audit the frozen experiment-001 answers

```bash
python3 -u audit_evaluation.py --output runs/evaluation-audit-v2
```

This performs no generation or training. It rechecks the saved baseline/final validation and test outputs with CPython on the original tests, and separately applies the official EvalPlus 0.3.1 checker with MBPP+ v0.2.0 to compatible task IDs. First it verifies the original reference against the original tests and both expanded suites. Tasks absent from MBPP+, entry-point mismatches, canonical-oracle resource limits, and references failing expanded tests are reported and excluded from both models' expanded denominator independently of model performance.

Expanded result objects use `correct` and upstream per-suite `pass` / `fail` / `timeout` fields for model outcomes; their outer `status` describes whether the checker itself completed.

The expanded score is a retrospective compatible-subset audit of old-prompt answers, not a full standard MBPP+ benchmark. It does not create a fresh test set. Original experiment files are never overwritten. Every run writes `manifest.json`, `results.jsonl`, `summary.json`, `REPORT.md`, and `status.json` to a new output directory. Check `status.json` or stream standard output to follow progress. `--resume` requires matching input/evaluator fingerprints and the same Docker image ID.

To reproduce the old execution backend, clone the pinned OpenEnv source, build its image, and opt in explicitly:

```bash
git clone https://huggingface.co/spaces/openenv/coding_env vendor/coding_env
git -C vendor/coding_env checkout 8b9b08339e64c14efef959c12efb18bdaf991fbd
docker build -t python-grpo-sandbox:local -f Dockerfile.sandbox .
python3 prepare_data.py --execution-backend openenv --output data/prepared-legacy
```

New data preparation and training default to CPython; changing backend changes the protocol. Existing prepared data cannot be overwritten, and an old training run cannot resume with a different backend or changed data fingerprints. Use a new run name and re-measure the baseline. The published experiment-001 dataset fingerprints and scores apply only to its recorded legacy protocol.

## Verified task selection and broader batches

[Completed pilot report](experiments/pipeline-v2/REPORT.md): 27 calibrated training tasks; four prompts × eight candidates per update; 31/40 groups with mixed rewards; development validation unchanged at 16/29; 49 confirmation tasks remain unscored.

The v2 pipeline verifies training references with original and official expanded tests, samples the original model on training tasks only, and selects tasks with mixed outcomes. Development and confirmation outcomes are excluded from selection.

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/hf download Qwen/Qwen2.5-Coder-0.5B-Instruct --revision ea3f2471cf1b1f0db85067f1ef93848e38e88c25
python3 -u task_selection.py --previous experiments/experiment-001 --output runs/task-quality-v2
.venv/bin/python -u calibrate_tasks.py --quality runs/task-quality-v2 --output runs/calibration-v2 --dataset data/selected-v2
.venv/bin/python -u train_v2.py --dataset data/selected-v2 --output runs/pilot-v2 --steps 10 --eval-every 5
python3 pipeline_report.py
```

The recorded pilot used the frozen legacy-prepared data from experiment 001. To reproduce its exact task counts in a fresh checkout, use the legacy preparation commands above, pass that directory with `task_selection.py --data data/prepared-legacy`, and verify its JSONL hashes against `experiments/experiment-001/config.json`. Fresh data prepared under CPython can produce different task pools and must be treated as a new experiment. The public `experiments/experiment-001` directory supplies the prior evaluated task IDs; private `runs/` files are not required.

Use new output paths for subsequent runs. Reference verification supports `--resume` with matching inputs and image; calibration and pilot outputs are immutable and cannot be overwritten. The calibration rule is predeclared: select a task if 1–7 of eight candidates pass every original and expanded test. Default generation uses the pinned original model, temperature 0.8, 256 tokens, and seed 42.

The pilot uses **four distinct prompts × eight candidates = 32 rollouts per optimizer update**, realized as micro-batches of two with 16 accumulation steps. The runtime checks group membership and logs `batch/distinct_prompts`, `batch/candidates_per_prompt`, `batch/mixed_reward_groups`, and `batch/zero_std_group_fraction`. Calibration progress is `calibration/percent`; actual optimizer progress is `progress/percent`. Primary correctness requires all original and expanded tests to pass; `mean_test_fraction` remains a diagnostic for the original tests only.

The pilot is validation-only. A separate confirmation pool is drawn from test task IDs never evaluated in experiment 001 and is reference-checked and frozen before training. No confirmation predictions are generated by these commands. Treat the small calibrated pool as a pipeline pilot; stronger rewards, selection, and batching change together, so a score change is not a causal ablation or proof of general improvement.
