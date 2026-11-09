# codeforces-baseline-002: Python Codeforces baseline

Baseline only: no optimizer steps, adapter training, or confirmation predictions.

Prepared training problems: 628. Fixed development problems: 67. Completion cap: 1024 tokens.

Original-model correctness: 1/67. Every solved problem passed all official I/O records.

| Scope | Correct / problems | Outputs hitting token cap |
|---|---:|---:|
| easy | 0/30 | 1 |
| medium | 1/37 | 1 |

Fixed Codeforces train-derived development split, same official Python I/O grader before/after. Existing MBPP development/confirmation inputs remain unchanged.

This is a small, previously public contest development set. Pretraining overlap is possible. A greedy baseline does not determine whether a problem will produce mixed rewards under sampled GRPO rollouts. All prepared easy and medium tasks remain eligible regardless of these results.

The grader runs fresh Python processes, applies the recorded per-case limits and normalizes whitespace and YES/NO case as in the upstream no-checker runner. It stops at the first failure; unchecked test cases are not counted as evaluated. This is not an exact reproduction of Codeforces language-specific timing.
