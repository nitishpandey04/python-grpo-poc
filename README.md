# Python coding GRPO on a 16 GB GPU

One active path: **prepare native Python test programs → train with GRPO → compare against a fixed development set**. Qwen2.5-Coder-0.5B-Instruct, LoRA, TRL, Docker, and local Trackio logging.

The simplified importer retains **2,613 training problems from the pinned 3,000-row OpenCoder sample**. This is an eligible source-test corpus, not an independently verified gold dataset. Supplied tests can be weak or wrong. Reference solutions are optional metadata and are never required to pass, supplied to the model, or used to calculate training rewards.

## Run

With the existing local environment and cached data:

```bash
# Skip the download when data/opencoder-candidates already exists.
.venv/bin/python import_opencoder.py --revision 7d28f40d579edd7c24402d17d0c7639f991e6f8d --rows 3000
.venv/bin/python prepare_data.py
.venv/bin/python launch.py dashboard
.venv/bin/python launch.py train --name my-run --steps 10 --eval-every 5
```

Preparation and training refuse to overwrite completed outputs. For another prepared dataset, use `prepare_data.py --output data/my-data`, then `train.py --dataset data/my-data --name my-run`. To train in the foreground, call `train.py` directly.

The dashboard binds to **0.0.0.0:7860**. Open `http://<computer-tailscale-ip>:7860/?project=python-grpo` from your tablet. Select the run name. `progress/optimizer_step` and `progress/percent` show training progress; GPU heartbeats and other telemetry events are not optimizer steps.

[Completed simplification and smoke results](experiments/simple-pipeline-v1/DATA.md): 2,613 training problems; ten steps; development 16/29 → 15/29, so the original model remains selected.

The default ten steps check the pipeline. They are not a meaningful budget for claiming broad model improvement.

## What is kept and excluded

`prepare_data.py` preserves the supplied test program in order, including setup, imports, helpers, loops, non-equality assertions and explicit `raise` checks. There is no minimum assertion count, reference-pass requirement, code-length threshold, task-family gate, or model-difficulty filter.

The pinned sample's exclusions are:

| Reason | Rows |
|---|---:|
| Exact or near-identical statement duplicates | 302 |
| Previously reserved OpenCoder development/confirmation rows | 55 |
| Statement overlap with protected evaluation problems | 22 |
| Test syntax errors | 2 |
| Setup-only programs with no checks | 2 |
| Documented test setup using a variable before assignment | 3 |
| Documented contradictory example in a statement | 1 |
| **Remaining training problems** | **2,613** |

All exclusions and matching task IDs are saved in `data/python-programs/exclusions.jsonl`. Original raw rows remain unchanged. Statement matching preserves punctuation and checks numeric/negation differences, but does not guarantee semantic or pretraining decontamination. Difficulty is logged during training; all-pass and all-fail tasks remain in the corpus.

The execution interface is a plain Python test program that signals failure through assertions or exceptions. It does not implement pytest/unittest discovery. Syntax checks and explicit-check presence do not establish that a program's expectations are correct or its checks necessarily execute. Unsupported formats should receive adapters when actually needed, rather than being confused with invalid problems.

## Training and evaluation

Training reward is **1 when all supplied test-program checks pass, otherwise 0**. Tests share one fresh child process per candidate so setup and call order are preserved. The default batch is four distinct prompts × eight candidates, realized as micro-batches of two with 16 gradient accumulation steps. The runtime checks this layout and logs mixed/zero-variance reward groups.

Development evaluation remains the existing **29-problem MBPP subset**, using original assertions plus official EvalPlus expanded checks and identical greedy decoding before and after training. It is a small development set already used in prior decisions, not a fresh independent benchmark. The **49-task confirmation pool remains unqueried**. Its IDs/fingerprint are recorded in `evaluation/manifest.json`; training does not load its questions. The 55 previously reserved OpenCoder tasks remain excluded from training and unqueried.

`runs/<name>/` holds configuration and source/data hashes, metrics, generated solutions, baseline/checkpoint development results, checkpoints, the final adapter, `report.json`, `REPORT.md`, a PID and a status heartbeat. Validation retains the original model unless a checkpoint strictly beats its development correctness. Short-run reward changes do not establish model improvement.

## Setup on a fresh machine

Python 3.12, Docker, `uv`, an NVIDIA driver and CUDA-compatible PyTorch are required. The RTX 5060 Ti run uses PyTorch CUDA 12.8 wheels. Exact host versions are in `requirements.lock.txt`.

```bash
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
mkdir -p evaluation_assets
curl -L --fail https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz -o evaluation_assets/MbppPlus-v0.2.0.jsonl.gz
docker build -t python-grpo-cpython:v2 -f Dockerfile.evaluation .
HF_HOME="$PWD/.cache/huggingface" .venv/bin/hf download Qwen/Qwen2.5-Coder-0.5B-Instruct --revision ea3f2471cf1b1f0db85067f1ef93848e38e88c25
```

TRL's GPU kernels need a compiler and Python development headers. The local environment already has them under `.cache/python-headers/usr/include`; `train.py` adds that location automatically. On a fresh Ubuntu machine install the matching development headers, or extract the `libpython3.12-dev` package locally as in the historical setup instructions.

Candidate, reference and test-control execution happens only inside Docker. The worker has no network or mounted host files, uses a non-root user, a read-only filesystem, CPU/PID limits and a 1 GiB container-memory cap. Native test programs get a two-second total suite budget. Expanded evaluation uses its recorded separate budget.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v
```

## History and attribution

[Experiment ledger](EXPERIMENTS.md) preserves unsuccessful and inconclusive results. The original [published adapter](https://huggingface.co/nitishpandey04/qwen2.5-coder-0.5b-python-grpo-poc) remains unchanged; experiment 001 did not establish validation-supported improvement.

The earlier selection/oracle/review pipelines have been retired from the root. Their exact source and reports remain under `experiments/` and Git tags `experiment-001`, `pipeline-v2-pilot`, `data-opencoder-v1`, and `data-reviewed-v2`. Check out the corresponding tag to reproduce a historical workflow; old report commands describe that version.

OpenCoder source: [educational_instruct](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2), revision `7d28f40d579edd7c24402d17d0c7639f991e6f8d`. Upstream MIT copyright/license: [third_party/OPENCODER_LICENSE](third_party/OPENCODER_LICENSE). MBPP/evaluation attribution and original source fingerprints are under [evaluation/](evaluation/README.md).
