---
language:
- en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
base_model_relation: adapter
pipeline_tag: text-generation
datasets:
- google-research-datasets/mbpp
tags:
- grpo
- trl
- lora
- python
- code
- experimental
---

# Qwen2.5-Coder-0.5B Python GRPO POC — experiment 001

**Experimental final-step LoRA adapter. No validation-supported improvement was established.** This repository preserves the trained artifact, including its inconclusive result, so future experiments can be compared honestly.

This is an adapter, not standalone full model weights. It requires [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct). The starting model was already specialized for coding and instruction-tuned.

Code and experiment ledger: https://github.com/nitishpandey04/python-grpo-poc

Detailed evidence: https://github.com/nitishpandey04/python-grpo-poc/tree/main/experiments/experiment-001

## Results and interpretation

| Metric | Original model | Final step-200 adapter |
|---|---:|---:|
| Held-out test problems solved | 97/300 | 106/300 |
| Test single-attempt correctness | 32.3% | 35.3% |
| Mean generated test tokens | 87.9 | 76.3 |
| Validation problems solved | 29/79 | 24/79 |

The best evaluated trained checkpoint solved 27/79 validation problems; none beat the original model's 29/79. The predefined validation-selection rule therefore retained the original model. **The adapter published here is the final trained artifact, not a validation-selected winner.**

The final adapter's test-only gain was +3.0 percentage points: 13 tasks improved and 4 regressed. A paired bootstrap over the 300 test tasks gave a 95% interval of approximately [+0.3, +5.7] percentage points. That interval does not capture training-seed variation, pretraining contamination, weak tests, or disagreement with validation. It does not establish general improvement.

The test set has now been inspected and should not be reused as a tuning target. Further conclusions require stronger or fresh held-out tests and repeated training runs.

## Training

- Date: 2026-11-03; run: `experiment-001`; random seed: 42.
- Base revision: `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`.
- Method: TRL GRPO with LoRA, 200 optimizer steps.
- Four sampled candidates per prompt; temperature 0.8; maximum completion length 256 tokens.
- Reward: fraction of the original MBPP tests passed, measured through the OpenEnv Coding executor inside a restricted Docker container. Built-in style/length rewards were ignored.
- LoRA rank 16, alpha 32, dropout 0; attention and MLP projection layers.
- Learning rate 1e-5, KL coefficient 0.02, per-device batch size 1, gradient accumulation 4.
- BF16; AdamW; gradient checkpointing; no vLLM.
- Hardware: one NVIDIA RTX 5060 Ti 16 GB. Peak PyTorch allocation was about 1.12 GiB.
- Reference-verified MBPP subset: 334 training tasks, 79 validation tasks, and 448 available test tasks; 300 test tasks were evaluated after a seed-42 shuffle.
- The run resumed from checkpoint 50 after an executor failure. Recovery from Docker-enforced memory exhaustion was fixed; memory-limit failures receive zero reward and restart the worker.

Exact configuration, dataset fingerprints, and package versions are included here and in the code repository.

## Evaluation scope

Evaluation used identical prompts, greedy decoding, a 256-token completion budget, one submission per problem, and all original MBPP tests for that problem. No execution feedback was given to the model during evaluation. This is a filtered MBPP POC, **not an EvalPlus benchmark result**. Original MBPP test suites are small, semantic overlap and pretraining contamination are possible, and only one training seed was used. No interactive debugging or repository-level ability was measured.

## Load the adapter

Tested with PyTorch 2.11.0+cu128, Transformers 5.22.1, PEFT 0.21.2, and Accelerate 1.15.0. See `requirements.lock.txt` for the full environment.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
adapter_id = "nitishpandey04/qwen2.5-coder-0.5b-python-grpo-poc"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    revision="ea3f2471cf1b1f0db85067f1ef93848e38e88c25",
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id).eval()
messages = [
    {"role": "system", "content": "Write correct Python functions. Return only Python code in a single ```python code block. Include the requested function and any helper functions. Do not include example calls, print statements, or explanations."},
    {"role": "user", "content": "Write a function that returns the sum of even numbers in a list.\n\nRequired signature:\ndef sum_even(numbers):"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Generated code may be incorrect. Execute it only in an appropriate sandbox. The reference evaluation's restricted syntax and imports are documented in the code repository.

## Attribution

Base model: Qwen team, Apache-2.0. Dataset: [Google Research MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp), CC BY 4.0 according to its dataset card; Austin et al., *Program Synthesis with Large Language Models* (2021), https://arxiv.org/abs/2108.07732. This experiment filtered the dataset and used its tests as rewards. Execution used OpenEnv Coding; training used TRL, Transformers, PEFT, and PyTorch. Upstream components retain their own licenses.
