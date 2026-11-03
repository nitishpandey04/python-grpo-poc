# Python GRPO on a 16 GB GPU

A local POC with Qwen2.5-Coder-0.5B-Instruct, LoRA, TRL GRPO, actual OpenEnv Coding execution in Docker, and a local Trackio dashboard. No cloud training or hosted logging is required.

## Experiment status

**Experiment 001 is complete. It did not establish a validation-supported improvement.** None of the evaluated trained checkpoints beat the original model on validation, so the selection rule retained the original model.

The final step-200 adapter separately scored 106/300 (35.3%) on the held-out test subset versus 97/300 (32.3%) for the original model. This test-set gain is exploratory evidence, not a claim that the trained model is generally better. Original MBPP tests are limited, contamination is possible, and only one training seed was run.

- [Experiment ledger](EXPERIMENTS.md)
- [Full experiment report](experiments/experiment-001/REPORT.md)
- [Experimental LoRA adapter on Hugging Face](https://huggingface.co/nitishpandey04/qwen2.5-coder-0.5b-python-grpo-poc)

The published model is the final trained adapter for reproducibility and future comparison. It is not the model selected by validation. It requires the original Qwen base model; loading instructions are in the model card.

## What this version measures

Single-submission Python function correctness. Each training problem gets four sampled candidates. Reward is the fraction of original MBPP tests passed. The separate greedy evaluator counts a problem as correct only if **all** its tests pass. Generated code never runs on the host.

This uses OpenEnv's PythonCodeActEnv library API (`reset` and `step`) inside a resource-limited persistent Docker worker. It does not use TRL's multi-turn `environment_factory`: interactive execution/repair is a later experiment. OpenEnv's built-in style/safety rewards are ignored.

## Setup

Python 3.12, NVIDIA driver, `uv`, and Docker are required. The RTX 5060 Ti requires a Blackwell-compatible PyTorch build; this project uses CUDA 12.8 wheels.

```bash
uv --cache-dir /tmp/python-grpo-uv venv --python /usr/bin/python3 .venv
uv --cache-dir /tmp/python-grpo-uv pip install --python .venv/bin/python torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv --cache-dir /tmp/python-grpo-uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
git clone https://huggingface.co/spaces/openenv/coding_env vendor/coding_env
git -C vendor/coding_env checkout 8b9b08339e64c14efef959c12efb18bdaf991fbd
docker build -t python-grpo-sandbox:local -f Dockerfile.sandbox .
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
.venv/bin/python -u train.py --name experiment-001 --steps 200 --eval-every 25 --resume runs/experiment-001/checkpoints/checkpoint-50
```

Trackio's event index includes heartbeats and scalar events. Use `progress/optimizer_step` to distinguish optimization progress from telemetry. A long generation can take time between training updates; the heartbeat shows that the process is alive.

## Validation and limits

```bash
.venv/bin/python -m unittest discover -s tests -p test_evaluation.py
.venv/bin/python -m unittest discover -s tests -p test_sandbox.py
```

The Docker worker runs without network, mounted host files, root privileges, or Linux capabilities, with CPU/memory/PID limits and a watchdog. Restricted syntax/imports are intentional. Host infrastructure failures stop the run rather than silently awarding zero reward.

Generated code that exceeds the sandbox memory budget receives zero reward and a `memory_error` status. The host verifies Docker's `OOMKilled` flag and starts a fresh worker for the next rollout. Timeouts similarly restart the worker; unrelated infrastructure failures still stop training.

Data preparation preserves MBPP's original split boundaries, rejects duplicate problem text, and keeps only tasks whose reference solutions pass the same restricted executor. **Semantic overlap between splits and pretraining contamination remain possible.** This is a filtered MBPP POC with the original small test suites, not an EvalPlus result or evidence of broad coding ability. Expanded hidden tests, task-family deduplication, multiple training seeds, and multi-turn debugging are follow-up work.

The 200-step budget is a cap, not a promised improvement. Training loss/reward alone does not establish success. Read validation and final-test correctness, confidence intervals, and token usage together.

MBPP source and licensing: https://github.com/google-research/google-research/tree/master/mbpp

OpenEnv source/license is retained under `vendor/coding_env/`.
