# codeforces-4b-probe-bf16-001: sampled training rewards

Unchanged original model; zero optimizer steps. 32 training problems, 8 sampled Python solutions per problem; temperature 0.8, completion cap 1024 tokens.

| Scope | Problems | Passing candidates | Mixed groups | All fail | All pass |
|---|---:|---:|---:|---:|---:|
| easy | 16 | 85/128 | 4 | 4 | 8 |
| medium | 16 | 68/128 | 10 | 3 | 3 |
| Total | 32 | 153/256 | 14 | 7 | 11 |

Mixed means at least one correct and one incorrect candidate for the same problem. With binary all-pass rewards, all-fail and all-pass groups have zero within-group reward variation.

Selection used only the training pool, balancing difficulty in a fixed hash order before any sampled answers. These are training-reward diagnostics, not validation scores or evidence of model improvement. The balanced aggregate is not weighted to corpus proportions. Development and confirmation were not queried; the training pool remains unchanged.

Token-cap hits: 52/256. Wall time after loading: 12.5 minutes. Peak PyTorch allocation: 9.30 GiB.

Official I/O grading uses the same Python Docker worker and fail-fast binary all-pass reward as training. A small probe and eight failures cannot establish that a problem is unsolvable. No tasks are removed based on this probe.
