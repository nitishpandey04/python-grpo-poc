# Qwen2.5-Coder-1.5B comparison and GRPO smoke test

Completed comparison and a bounded training experiment. **The 1.5B model is a stronger starting point on these checks, and BF16 + LoRA GRPO fits the local 16 GB RTX 5060 Ti. Ten updates did not improve development correctness.**

## Same-problem comparison

| Measure | Original 0.5B | Original 1.5B |
|---|---:|---:|
| Sampled Python solutions passing all official tests | 3/256 (1.17%) | 9/256 (3.52%) |
| Training problems solved at least once | 2/32 | 4/32 |
| Groups with both correct and incorrect solutions | 2/32 | 4/32 |
| Fixed greedy development correctness | 1/67 | 4/67 |
| Probe peak PyTorch allocation | 1.21 GiB | 3.43 GiB |

The 1.5B probe passed 8/128 easy candidates and 1/128 medium candidates; its mixed groups were three easy and one medium. It completed in 185.5 seconds after loading. The 32 IDs and their hash-selected 16-easy/16-medium allocation are byte-identical to the earlier probe. Prompts, official Python I/O grader, eight samples per problem, temperature 0.8, top-p 1, disabled top-k, repetition penalty 1, seed 42, BF16 weights and the 1,024-token cap were kept fixed. These are small, single-seed diagnostics, not evidence of general superiority or a controlled statistical estimate from 256 independent tasks. The larger model still failed all eight attempts on 28/32 problems.

## Actual GRPO training

| Item | Result |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| Pinned revision | 2e1fd397ee46e1388853d2af2c993145b0f1098a |
| Method | BF16 frozen base + LoRA rank 16, TRL GRPO, beta 0.02, learning rate 1e-5 |
| Batch | Four distinct prompts × eight candidates; micro-batch 2, accumulation 16 |
| Budget | Ten optimizer updates; greedy development before and after |
| Sampled training data | 40 distinct problems, 320 Python solutions |
| Passing training candidates | 16/320 |
| Mixed-reward training groups | 7/40 |
| Development before → after | **4/67 → 4/67** |
| Problems newly solved / regressed | **0 / 0** |
| Selected checkpoint | Step 0: retain original 1.5B model |
| Peak PyTorch allocation | **6.87 GiB** |
| Peak periodically sampled total device use | 7.58 GiB |

Training completed with nonzero gradients on rewarded groups, no OOM and no quantization. All 31 regression checks passed. The 7/40 mixed-group rate is from a different training sample while weights were changing; it should not be compared as an improvement over the frozen 32-problem probe. The development bootstrap interval [0, 0] is degenerate because the observed binary before/after correctness vectors are identical; it does not prove the true population effect is zero.

All 628 tasks remain eligible, including 324 medium problems; no task or difficulty group was removed based on predictions. Training group counts by difficulty are recorded in comparison.json. The 67-task development set remains frozen. Existing confirmation pools and the official Codeforces test split were not queried.

## Reproduce

The installed system `hf` CLI downloaded the exact weights after an Xet transfer stalled. The cached weight SHA-256 was verified as `c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8`. The model is Apache-2.0 according to the pinned Hub metadata. The source dataset, grading assumptions and attribution remain as documented in [the prepared-data report](../codeforces-python-ready/REPORT.md).

```bash
HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen2.5-Coder-1.5B-Instruct model.safetensors config.json generation_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a --cache-dir "$PWD/.cache/huggingface/hub"
.venv/bin/python launch.py train --name my-15b-probe --reward-probe --model Qwen/Qwen2.5-Coder-1.5B-Instruct --model-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a
.venv/bin/python launch.py train --name my-15b-smoke --steps 10 --eval-every 10 --model Qwen/Qwen2.5-Coder-1.5B-Instruct --model-revision 2e1fd397ee46e1388853d2af2c993145b0f1098a
```

Use fresh run names. The default dataset and exact execution image must already be prepared. Probe settings are in probe/config.json and probe/probe-generation.json; the final training settings are in smoke/config.json.

Live Trackio runs: `codeforces-1.5b-reward-probe-001` and `codeforces-1.5b-smoke-001`. The local final adapter is `runs/codeforces-1.5b-smoke-001/final-adapter`; it is retained as experimental evidence and is not selected or uploaded. The previously published 0.5B adapter is unchanged. A local adapter fingerprint is in comparison.json; model weights and optimizer checkpoints are not duplicated into GitHub.

The practical outcome is a working 1.5B GRPO configuration with more initial reward signal, alongside an honest no-improvement result from ten updates. This smoke run checks feasibility and bookkeeping; it neither establishes learning gains nor shows that a longer run would help. A next experiment can use this 1.5B configuration while measuring its still-sparse rewards, with one documented change at a time.
