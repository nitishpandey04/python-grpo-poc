# Verified task selection and batching pilot

This is a development-only pipeline pilot. No general model improvement is established, and no confirmation answers were generated. Historical experiment-001 results remain unchanged.

| Stage | Result |
|---|---|
| References passing original and expanded checks | 66 training tasks |
| Eight-candidate calibration | 27 mixed, 37 all failed, 2 all passed |
| Observed batches | 10 updates, four distinct prompts × eight candidates = 32 rollouts/update |
| Groups with a correctness comparison signal | 31/40 (77.5%) |
| Development correctness: original / final | 16/29 / 16/29 |
| Peak PyTorch allocated memory | 1.25 GiB |
| Sealed confirmation tasks | 49; unscored |

## Protocol

The original Qwen2.5-Coder-0.5B-Instruct model was sampled eight times per verified training task at temperature 0.8 with a 256-token completion cap and seed 42. A training task was selected only when 1–7 of its eight sampled answers passed all original and official expanded tests. Validation and confirmation model outcomes never influenced task selection.

Training uses a binary reward for passing all original and official expanded tests. Two rollouts per micro-batch and 16 gradient accumulation steps form one 32-rollout optimizer update. The installed TRL sampler and runtime reward callback both verify four distinct contiguous prompt groups of eight. The learning rate remains 1e-5; LoRA rank 16 and KL coefficient 0.02 remain unchanged.

Existing development validation is used only as development data. Confirmation candidates come from the 148 original test tasks never evaluated in experiment 001; reference/coverage checks retain 49. Both split IDs and file hashes are frozen before training. Neither calibration nor the validation-only pilot generates confirmation predictions.

## Interpretation

This combined pilot changes task verification, selection, reward strength, and batching. It demonstrates that the new pipeline works and measures its signal; it does not isolate which change helps performance. The calibrated training pool is small, development validation has only 29 tasks, and only one seed was run. A longer run or a one-task validation change would not establish general improvement.

Before spending the confirmation set, expand the verified task pool and compare controlled variants with matched rollout budgets and additional seeds. Keep the original model as the reference baseline.
