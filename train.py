"""Local GRPO experiment: baseline, checkpoint validation, final paired evaluation."""
import argparse
import inspect
import json
import os
from pathlib import Path
import random
import time
import hashlib

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TRACKIO_DIR", str(ROOT / "runs/trackio"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache/triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / ".cache/torchinductor"))
headers = ROOT / ".cache/python-headers/usr/include"
if headers.exists():
    os.environ["C_INCLUDE_PATH"] = os.pathsep.join([str(headers / "python3.12"), str(headers), os.environ.get("C_INCLUDE_PATH", "")])

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from evaluate import compare, evaluate
from execution import Sandbox
from monitoring import Monitor


def load_rows(split):
    rows = [json.loads(line) for line in (ROOT / f"data/prepared/{split}.jsonl").read_text().splitlines()]
    random.Random(42).shuffle(rows)
    return rows


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run outside the app sandbox with the CUDA-enabled project environment.")
    set_seed(42)
    torch.set_num_threads(4)
    output = ROOT / "runs" / args.name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists() and not args.resume:
        raise ValueError("Run already exists. Choose --name or explicitly use --resume CHECKPOINT.")
    config = {**vars(args), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "seed": 42, "decode": "greedy, one attempt", "train_reward": "fraction of original MBPP tests passed"}
    config["prepared_data_sha256"] = {split: hashlib.sha256((ROOT / f"data/prepared/{split}.jsonl").read_bytes()).hexdigest() for split in ("train", "validation", "test")}
    if args.resume:
        previous = json.loads((output / "config.json").read_text())
        previous_backend = previous.get("execution_backend", "openenv")
        if previous_backend != args.execution_backend:
            raise ValueError("Cannot switch execution backend while resuming; use the recorded backend or a new run name")
        for key in ("model", "max_tokens", "prepared_data_sha256"):
            if previous.get(key) != config.get(key):
                raise ValueError(f"Cannot resume with changed evaluation protocol: {key}")
    (output / "config.json").write_text(json.dumps(config, indent=2))
    monitor = Monitor(output, args.name, config)
    sandbox = Sandbox(timeout=8, backend=args.execution_backend)
    try:
        sandbox.start()
        monitor.phase = "loading model"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
        config["model_commit"] = getattr(model.config, "_commit_hash", None)
        (output / "config.json").write_text(json.dumps(config, indent=2))
        train_rows = load_rows("train")
        val_rows = load_rows("validation")[:args.val_size]
        test_rows = load_rows("test")[:args.test_size]
        if args.resume:
            baseline_val = [json.loads(line) for line in (output / "baseline-validation.jsonl").read_text().splitlines()]
            baseline_val_metric = sum(r["correct"] for r in baseline_val) / len(baseline_val)
        else:
            _, baseline_metrics = evaluate(model, tokenizer, val_rows, sandbox, output / "baseline-validation.jsonl", args.max_tokens, monitor, "baseline_validation")
            baseline_val_metric = baseline_metrics["correctness"]
            # Save baseline final-test outputs before training; never use them for checkpoint selection.
            evaluate(model, tokenizer, test_rows, sandbox, output / "baseline-test.jsonl", args.max_tokens, monitor, "baseline_test")

        def correctness_reward(completions, tests, task_id, **kwargs):
            monitor.phase = "scoring training rollouts"
            rewards, statuses = [], []
            for completion, problem_tests, problem_id in zip(completions, tests, task_id):
                text = completion[0]["content"] if isinstance(completion, list) else completion
                score = sandbox.score(text, problem_tests)
                rewards.append(score["passed"] / score["total"])
                statuses.append(score["status"])
                monitor.sample({"task_id": problem_id, "completion": text, "reward": rewards[-1], **score})
            monitor.log({"execution/syntax_error_fraction": statuses.count("syntax_error") / len(statuses), "execution/assertion_error_fraction": statuses.count("assertion_error") / len(statuses), "execution/runtime_error_fraction": statuses.count("runtime_error") / len(statuses), "execution/timeout_fraction": statuses.count("timeout") / len(statuses), "execution/memory_error_fraction": statuses.count("memory_error") / len(statuses), "execution/unsupported_fraction": statuses.count("unsupported") / len(statuses), "execution/all_tests_passed_fraction": sum(r == 1 for r in rewards) / len(rewards)})
            monitor.phase = "updating weights / generating next group"
            return rewards

        class Progress(TrainerCallback):
            def __init__(self):
                self.best = baseline_val_metric
                if (output / "selection.json").exists():
                    self.best = json.loads((output / "selection.json").read_text())["validation_correctness"]

            def on_log(self, args, state, control, logs=None, **kwargs):
                monitor.step = state.global_step
                monitor.log({"train/" + k: v for k, v in (logs or {}).items()})
                monitor.log({"progress/optimizer_step": state.global_step, "progress/total_steps": cli.steps, "progress/percent": 100 * state.global_step / cli.steps})

            def on_step_begin(self, args, state, control, **kwargs):
                monitor.phase = "generating training rollouts / updating weights"

            def on_step_end(self, args, state, control, model=None, **kwargs):
                monitor.step = state.global_step
                if state.global_step % cli.eval_every == 0 or state.global_step == cli.steps:
                    _, metrics = evaluate(model, tokenizer, val_rows, sandbox, output / f"validation-step-{state.global_step}.jsonl", cli.max_tokens, monitor, "validation")
                    monitor.log({"validation/baseline_correctness": baseline_val_metric, "validation/change_from_baseline": metrics["correctness"] - baseline_val_metric})
                    if metrics["correctness"] > self.best:
                        self.best = metrics["correctness"]
                        model.save_pretrained(output / "best-adapter")
                        tokenizer.save_pretrained(output / "best-adapter")
                        (output / "selection.json").write_text(json.dumps({"optimizer_step": state.global_step, "validation_correctness": self.best}, indent=2))
                monitor.phase = "training"

        cli = args
        kwargs = dict(output_dir=str(output / "checkpoints"), max_steps=args.steps,
                      per_device_train_batch_size=1, gradient_accumulation_steps=4,
                      num_generations=4, max_completion_length=args.max_tokens,
                      learning_rate=1e-5, beta=0.02, temperature=0.8,
                      bf16=True, gradient_checkpointing=True, optim="adamw_torch",
                      logging_steps=1, logging_first_step=True, save_steps=args.eval_every,
                      save_total_limit=2, report_to="none", seed=42,
                      remove_unused_columns=False, disable_dropout=True,
                      use_vllm=False)
        # Pin and record installed versions; tolerate max_prompt_length's removal in recent TRL.
        if "max_prompt_length" in inspect.signature(GRPOConfig).parameters:
            kwargs["max_prompt_length"] = 512
        trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                              args=GRPOConfig(**kwargs),
                              peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"),
                              train_dataset=Dataset.from_list([{k: r[k] for k in ("prompt", "tests", "task_id")} for r in train_rows]),
                              reward_funcs=correctness_reward, callbacks=[Progress()])
        monitor.phase = "training"
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_model(str(output / "final-adapter"))
        tokenizer.save_pretrained(output / "final-adapter")
        monitor.log({"gpu/peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2})
        # Select using validation only. If no checkpoint improves, preserve the original model.
        if (output / "best-adapter").exists():
            trainer.model.load_adapter(str(output / "best-adapter"), adapter_name="selected")
            trainer.model.set_adapter("selected")
            selection = json.loads((output / "selection.json").read_text())
            after, _ = evaluate(trainer.model, tokenizer, test_rows, sandbox, output / "selected-test.jsonl", args.max_tokens, monitor, "selected_test")
        else:
            selection = {"optimizer_step": 0, "validation_correctness": baseline_val_metric, "note": "No strict validation improvement; baseline retained."}
            with trainer.model.disable_adapter():
                after, _ = evaluate(trainer.model, tokenizer, test_rows, sandbox, output / "selected-test.jsonl", args.max_tokens, monitor, "selected_test")
        # Also report the final trained checkpoint, with no further selection using its test score.
        trainer.model.set_adapter("default")
        final, _ = evaluate(trainer.model, tokenizer, test_rows, sandbox, output / "final-test.jsonl", args.max_tokens, monitor, "final_test")
        before = [json.loads(line) for line in (output / "baseline-test.jsonl").read_text().splitlines()]
        report = {"selected_checkpoint": selection, "selected_vs_original": compare(before, after), "final_vs_original": compare(before, final), "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2, "limitations": ["Original MBPP tests only; expanded hidden tests are future work.", "MBPP pretraining contamination and semantic overlaps cannot be ruled out.", "Single training seed; small filtered subset; POC evidence only."]}
        (output / "report.json").write_text(json.dumps(report, indent=2))
        from report import build_report
        build_report(output)
        monitor.phase = "complete"
        print(json.dumps(report, indent=2), flush=True)
    except BaseException:
        monitor.phase = "failed or interrupted"
        raise
    finally:
        sandbox.close()
        monitor.close()
        (output / "status.json").write_text(json.dumps({"phase": monitor.phase, "optimizer_step": monitor.step, "updated_at": time.time()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="poc-" + time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--execution-backend", choices=["cpython", "openenv"], default="cpython")
    parser.add_argument("--resume")
    main(parser.parse_args())
