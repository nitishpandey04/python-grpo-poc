# codeforces-1.5b-reward-probe-001: sampled training rewards

Unchanged original model; zero optimizer steps. 32 training problems, 8 sampled Python solutions per problem; temperature 0.8, completion cap 1024 tokens.

| Scope | Problems | Passing candidates | Mixed groups | All fail | All pass |
|---|---:|---:|---:|---:|---:|
| easy | 16 | 8/128 | 3 | 13 | 0 |
| medium | 16 | 1/128 | 1 | 15 | 0 |
| Total | 32 | 9/256 | 4 | 28 | 0 |

Mixed means at least one correct and one incorrect candidate for the same problem. With binary all-pass rewards, all-fail and all-pass groups have zero within-group reward variation.

Selection used only the training pool, balancing difficulty in a fixed hash order before any sampled answers. These are training-reward diagnostics, not validation scores or evidence of model improvement. The balanced aggregate is not weighted to corpus proportions. Development and confirmation were not queried; the training pool remains unchanged.

Token-cap hits: 2/256. Wall time after loading: 3.1 minutes. Peak PyTorch allocation: 3.43 GiB.

Official I/O grading uses the same Python Docker worker and fail-fast binary all-pass reward as training. A small probe and eight failures cannot establish that a problem is unsolvable. No tasks are removed based on this probe.
