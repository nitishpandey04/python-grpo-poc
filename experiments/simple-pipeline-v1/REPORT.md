# simple-smoke-001: native Python test-program GRPO

This is a short pipeline smoke test. It does not establish general model improvement.

Available training problems: 2613. Optimizer steps: 10. Distinct sampled training problems: 40.

Fixed development correctness: original 16/29; final adapter 15/29. Selected step: 0 (zero retains the original model).

Development change: -3.45 percentage points; paired bootstrap interval [-10.344827586206897, 0.0]. Problems improved/regressed: 0/1.

Mixed-reward prompt groups: 21/40. Training reward uses supplied test programs, whose coverage and expectations may be imperfect.

Peak PyTorch allocation: 1.46 GiB. Confirmation problems: 49; no confirmation predictions generated.

Training and evaluation use different, explicit test sources: native supplied programs for training, and unchanged original-plus-expanded MBPP checks for development. Source tests are not independently certified. Development is small and has informed prior decisions. Pretraining/semantic overlap remains possible.

Configuration, source/data fingerprints, model outputs, scalar metrics, development comparisons, and the final adapter are saved alongside this report. Historical experiment files are unchanged.
