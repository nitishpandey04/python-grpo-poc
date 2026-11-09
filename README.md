# Python coding GRPO on a 16 GB GPU

One active path: **prepare Codeforces problems → generate Python solutions → execute official tests in Docker → train with GRPO and compare on fixed development data**. Qwen2.5-Coder-0.5B-Instruct, LoRA, TRL and local Trackio logging.

Codeforces problems are language-neutral. Every prepared row requests **Python 3**, and the runner executes Python only. No C++ or other-language solutions are used as training targets.

## Prepared data

| Experiment scope | Training | Development |
|---|---:|---:|
| Easy: ratings 800–1000 | 304 | 30 |
| Medium: ratings 1100–1600 | 324 | 37 |
| **Total** | **628** | **67** |

These labels and the rating-1600 cap are choices for this experiment. Higher-rated and unrated tasks are outside the requested scope. Medium problems remain in the dataset regardless of baseline performance.

All included problems have complete official I/O suites, a usable statement, standard I/O and no supplied custom checker or explicit unsupported alternative-output contract. The training set contains **32,983 official I/O records**; development contains 3,396. There is no minimum test-count, reference-pass or sampled-success gate. Eight otherwise in-scope source entries lacked a statement. Twenty others explicitly allowed alternative/tolerance-based answers despite empty checker metadata; those require a semantic checker before use. Source aliases and near-identical statements are grouped before a deterministic roughly 10% development split; protected statements are checked for overlap.

Source: [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces), revision `fbe3f6e903ee854eec2e69e9d96d0306cde59baf`, **training split only**. Preparation downloads about 2.4 GB of training shards into the cache. It does not download the separate 110 GB generated-test corpus or the official test split. Incomplete suites, custom-checker and file/interactive tasks are deferred rather than silently graded incorrectly.

## Run

With the local environment already installed:

```bash
docker build -t python-grpo-cpython:v3 -f Dockerfile.evaluation .
.venv/bin/python prepare_data.py
.venv/bin/python launch.py dashboard
.venv/bin/python launch.py train --name cf-baseline --baseline-only
```

Preparation and runs refuse to overwrite existing outputs. Prepared data is under `data/codeforces-python/`; skip preparation when reusing it. Use `prepare_data.py --output data/my-data` and `train.py --dataset data/my-data --name my-run` for another dataset. `--max-rating` defaults to 1600. To run in the foreground, call `train.py` directly.

When starting a training smoke check, omit `--baseline-only` and set a short budget, for example `--steps 10 --eval-every 5`. The existing runtime checks enforce four distinct prompts × eight candidates per optimizer update. The default Codeforces completion cap is 1,024 tokens; `--max-tokens` can override it and is recorded in the run configuration.

**Baseline result:** the unchanged original model solves 1/67 development problems greedily (easy 0/30, medium 1/37). This is a different benchmark from earlier MBPP results; no model training or regression occurred. A single greedy attempt does not establish sampled GRPO reward availability. Check that signal before committing to a long training budget. [Readiness and baseline evidence](experiments/codeforces-python-ready/REPORT.md).

## Reward and evaluation

Reward is **1 only when every official I/O record passes; otherwise 0**. Each test launches a fresh Python process and working directory, with real stdin/stdout and the recorded time/memory limits. Expected outputs stay in the grader. Output comparison follows the upstream no-checker policy: normalize whitespace and YES/NO case; do not coerce arbitrary numbers or accept other answer variations.

Execution stops at the first failure. `checked` records how many cases ran; unexecuted cases are never counted as passed or tested. Full-suite correctness is the primary metric. A partial-test fraction is omitted when fail-fast leaves unchecked cases. This runner is not an exact reproduction of Codeforces language-specific timing multipliers.

Training and development use the same official I/O grading semantics. The 67-task development split was frozen before any model predictions. The original 29-task MBPP development files remain unchanged as historical/secondary inputs. The existing 49-task MBPP confirmation pool, 55 reserved OpenCoder tasks, and official Codeforces test split remain unqueried.

Official completeness means all recorded official tests are present, not exhaustive input coverage. Source bugs, converted mathematical notation and semantic/pretraining overlap remain possible. Old public contest problems are not a clean generalization benchmark.

## Live progress and saved outputs

The dashboard binds to **0.0.0.0:7860**. From your tablet open `http://<computer-tailscale-ip>:7860/?project=python-grpo` and select the run name. `baseline_validation/percent` shows baseline progress; `progress/optimizer_step` shows actual training steps. A baseline-only run has zero optimizer steps.

`runs/<name>/` stores configuration and source/data/image fingerprints, status heartbeats, metrics, generated code, development results and a readable report. Training runs also save checkpoints and adapters. Validation retains the original model unless a checkpoint strictly improves development correctness. No new model weights were published during data preparation or baseline checking.

## Setup on a fresh machine

Python 3.12, Docker, `uv`, an NVIDIA driver and a compatible PyTorch build are required. Exact host versions are in `requirements.lock.txt`; the RTX 5060 Ti run uses CUDA 12.8 wheels.

```bash
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
mkdir -p evaluation_assets
curl -L --fail https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz -o evaluation_assets/MbppPlus-v0.2.0.jsonl.gz
HF_HOME="$PWD/.cache/huggingface" .venv/bin/hf download Qwen/Qwen2.5-Coder-0.5B-Instruct --revision ea3f2471cf1b1f0db85067f1ef93848e38e88c25
```

TRL kernels require a compiler and matching Python development headers. The current local headers are under `.cache/python-headers/usr/include`; `train.py` adds that location automatically. Docker v3 retains native-test and expanded-MBPP compatibility for historical inputs.

Candidate code executes only inside Docker: no network or host mounts, non-root user, read-only filesystem, capability restrictions and PID/CPU limits. Codeforces runs use a 2 GiB container budget so the recorded candidate limits, up to 1 GiB, fit alongside the grader. Output is file-size-limited and previews are bounded.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
```

## History and attribution

[EXPERIMENTS.md](EXPERIMENTS.md) records prior inconclusive attempts, [the pipeline simplification](experiments/simple-pipeline-v1/DATA.md), and [the source exploration](experiments/codeforces-exploration/REPORT.md). Earlier code remains in Git tags and historical snapshots. The old OpenCoder downloader is retired; its raw data and all previous results remain intact. The original published adapter remains unchanged.

Attribution: Open-R1/Hugging Face dataset authors, Guilherme Penedo and colleagues, and the original Codeforces problem authors. See the [pinned dataset card](https://huggingface.co/datasets/open-r1/codeforces/blob/fbe3f6e903ee854eec2e69e9d96d0306cde59baf/README.md). Its metadata/prose licence-label discrepancy is recorded in the exploration report. Existing MBPP/OpenCoder attribution is retained under `evaluation/` and `third_party/`.
