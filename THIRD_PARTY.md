# Upstream components and attribution

- Base model: [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct), Apache-2.0. The published LoRA adapter requires this model. The experiment used cached revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`.
- Dataset: [Mostly Basic Python Problems (MBPP), Google Research](https://huggingface.co/datasets/google-research-datasets/mbpp), CC BY 4.0 according to its dataset card. Original source: [Google Research MBPP](https://github.com/google-research/google-research/tree/master/mbpp), accompanying Austin et al., *Program Synthesis with Large Language Models* (2021), https://arxiv.org/abs/2108.07732. This project filters tasks and records task IDs, dataset fingerprints, generated model outputs, and measurements. The raw source dataset is downloaded separately rather than bundled.
- Execution environment: [OpenEnv Coding](https://huggingface.co/spaces/openenv/coding_env), source revision `8b9b08339e64c14efef959c12efb18bdaf991fbd`; upstream source/license is retained in the separately downloaded `vendor/coding_env` checkout. OpenEnv's source files retain their upstream BSD-style notices. No upstream source is copied into this repository.
- Training and logging use TRL, Transformers, PEFT, PyTorch, and Trackio. Exact host package versions are in `requirements.lock.txt`; upstream packages retain their own licenses.

The project license applies to the original code in this repository. It does not replace third-party licenses.

- Expanded evaluation: [EvalPlus](https://github.com/evalplus/evalplus), pinned package version 0.3.1, and its MBPP+ v0.2.0 release. Evaluation runs upstream input conversions, tolerances, and special oracles; package and dataset licenses remain upstream. The expanded dataset is downloaded separately into ignored `evaluation_assets/`.
