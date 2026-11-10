# Local model capacity comparison — 2026-11-10

Qwen3-4B-Instruct-2507 with NF4 QLoRA is the strongest tested practical starting point on this RTX 5060 Ti 16 GB. It completed ten GRPO updates and improved fixed development correctness from 29/67 to 31/67. This small development-only result does not establish general improvement.

## Matched sampled-reward probe

All models used the identical frozen 32 training problems (16 easy, 16 medium), eight candidates each, seed 42, temperature 0.8, top-p 1, top-k 0, repetition penalty 1 and a 1,024-token completion cap. A pass requires every official test to pass. Mixed groups contain both passing and failing candidates and therefore supply a relative correctness signal for GRPO.

| Original model / precision | Passing / 256 | Mixed groups / 32 | Medium passes / 128 |
|---|---:|---:|---:|
| Qwen2.5-Coder-0.5B BF16, historical | 3 | 2 | 0 |
| Qwen2.5-Coder-1.5B BF16, historical | 9 | 4 | 1 |
| Qwen2.5-Coder-3B BF16 | 29 | 9 | 8 |
| Qwen3-4B-Instruct-2507 BF16 | 153 | 14 | 68 |
| Qwen3-4B-Instruct-2507 NF4 | 137 | 16 | 57 |

The 4B model supplies substantially more useful reward signal, including on medium problems. These are practical comparisons across model families and precisions, not a controlled estimate of parameter-count effects. BF16 versus NF4 sample differences use one seed and do not isolate a general quantization penalty.

## Actual training capacity and before/after evaluation

Each successful smoke run used 10 optimizer updates, four distinct prompts × eight candidates per update, micro batch 2, accumulation 16, rank-16 LoRA, learning rate 1e-5, beta 0.02, gradient checkpointing and BF16 compute. Each sampled 40 distinct training tasks and generated 320 candidates. Development used one greedy answer on each of the same 67 fixed problems before and after training.

| Model / precision | Updates | Development before → after | Peak allocated GiB | Peak reserved GiB | Peak sampled device GiB |
|---|---:|---:|---:|---:|---:|
| 1.5B BF16, historical | 10 | 4 → 4 | 6.87 | See prior report | 7.58 |
| 3B BF16 | 10 | 7 → 7 | 10.82 | 11.18 | 11.56 |
| 4B BF16 | 0; CUDA OOM | 27 → unavailable | 14.68 | 15.09 | 15.44 |
| 4B NF4 QLoRA | 10 | 29 → 31 | 11.69 | 12.71 | 13.09 |

PyTorch allocation, allocator reservation and sampled whole-device usage measure different things. Device sampling can miss instantaneous peaks. The 4B BF16 failure occurred during the first 32-candidate rollout generation, before any update; lowering only the training micro batch would leave that generation batch unchanged, so that redundant retry was skipped. NF4 completed without reducing prompts, candidates, context cap or official test coverage.

Four billion parameters is the largest model tested here, not an absolute GPU size ceiling. Fit depends on architecture, precision, prompt/completion length and rollout batch size. This establishes adapter training capacity for this configuration; it does not measure full-parameter training capacity.

## Training result and selection

The 3B run produced 47/320 passing training candidates and 17/40 mixed groups; development remained 7/67 with no newly solved or regressed problems. Its original base remains selected.

The 4B NF4 run produced 157/320 passing training candidates and 17/40 mixed groups. Greedy development increased from 43.28% to 46.27%: four newly solved problems, two regressions, net two additional solves (+2.99 percentage points). The paired bootstrap 95% interval is -4.48 to +10.45 percentage points. Step 10 is selected under the existing development-selection rule. Both best and final adapters remain local at `runs/codeforces-4b-smoke-nf4-micro2-001/`; their source identities and final adapter hash are archived. No new model weights were uploaded.

The 27/67 BF16 and 29/67 NF4 original-model results are different precision baselines. The change from 27 to 29 is not a GRPO gain. Only 29 → 31 compares the quantized model before and after training under the same evaluation settings. Ten updates, one seed and 67 development problems are insufficient to claim a reliable general improvement. The existing 49-task MBPP confirmation pool, 55 reserved OpenCoder tasks and official Codeforces test split remain unqueried. Public contest problems may overlap pretraining.

## Remaining bottleneck

The 4B BF16 probe had 52 capped outputs and NF4 had 78, all classified as syntax failures by the frozen extractor. Inspected examples spent the token budget on reasoning written as Python comments inside an unfinished fenced block. These failures can include incomplete output/extraction rather than intrinsically invalid Python. No outputs were repaired or regraded for the primary comparison. A separate concise-code prompt experiment is a useful next check before committing to a long training run; freeze any revised protocol and recompute its baseline.

Elapsed time includes executing official suites. For example, six passing candidates on training problem 236/B spent about 237 seconds in grading its 102 records during update 4. More correct solutions can increase grading time because fail-fast no longer stops early. Test coverage was preserved throughout.

## Reproducibility and implementation

Data remains 628 training problems (304 easy, 324 medium), 67 development problems, 32,983 training official records and 3,396 development records. No task was removed based on model success. Manifest SHA-256: `68dc5df9f80ae4ca15c1fba54b5f1d3868da00b1d53ebb8494ecf583fb7d6f50`. Selection SHA-256: `f574311016b663737778a75389504e687881299fbf56f7745418b7bf851aa9e3`. The grader, fail-fast binary reward and source limits were unchanged.

Pinned models:

- Qwen/Qwen2.5-Coder-3B-Instruct: `488639f1ff808d1d3d0ba301aef8c11461451ec5`, Qwen Research license, noncommercial research/evaluation scope.
- Qwen/Qwen3-4B-Instruct-2507: `cdbee75f17c01a7cc42f958dc650907174af0554`, Apache 2.0.

All downloaded weight shards were checked against pinned LFS SHA-256 identities before use. The earlier 1.5B and completed 3B raw weight blobs were evicted to fit the 4B download on disk; results, data, metadata and local adapters were preserved. The selected 4B base remains cached.

`train.py --load-in-4bit` adds NF4 double quantization with BF16 compute and GPU-only loading; the ordinary BF16 path remains available. bitsandbytes is pinned to 0.50.2. TRL creates/freezes the PEFT adapters and enables checkpointing inputs. A generic k-bit preparation call was removed after a small Qwen3 diagnostic showed that its blanket upcast made the KV cache FP32. The retained native PEFT path demonstrated BF16 KV state, frozen base parameters and finite nonzero adapter gradients, then completed actual training. These diagnostics and both source snapshots are archived. Structured CUDA OOM reports and completed-step progress were added to the existing training path.

All 31 regression checks passed before the runs, including after the NF4 preparation adjustment. The archive additionally verifies frozen data/selection/source identities, ten completed updates, 320 rollouts in 40 groups of eight, paired 67-task development outputs and absence of confirmation predictions. `comparison.json`, per-run records, verification output and source snapshots retain the evidence. Raw consoles and model weights are excluded.

Use the cached pinned model for another explicitly named run:

```bash
.venv/bin/python launch.py train --name cf-4b-nf4-probe-next --reward-probe --model Qwen/Qwen3-4B-Instruct-2507 --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 --load-in-4bit
.venv/bin/python launch.py train --name cf-4b-nf4-smoke-next --steps 10 --eval-every 10 --model Qwen/Qwen3-4B-Instruct-2507 --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 --load-in-4bit
```

These commands start new runs; they do not resume or overwrite this completed experiment. Live progress continues through the existing Trackio dashboard.
