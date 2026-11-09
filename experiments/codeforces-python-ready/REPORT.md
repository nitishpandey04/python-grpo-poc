# Python-only Codeforces preparation — 2026-11-09

Status: data, Python execution and the training entry point are prepared. A baseline-only check completed; no optimizer updates, adapters or model uploads were produced.

Codeforces problems are language-neutral. Both prompts and runtime explicitly require **Python 3**. No C++ or other-language source solutions are used as training targets. Official I/O labels are language-independent; upstream accepted-submission validation is evidence about the source suites, not a restriction on our solution language.

## Frozen data

| Scope | Training | Development |
|---|---:|---:|
| Easy, rating 800–1000 | 304 | 30 |
| Medium, rating 1100–1600 | 324 | 37 |
| Total | 628 | 67 |

The experiment cap is 1600; it is a practical definition of the requested medium scope, not a guarantee of model ability. Source rating 1700+ and unrated rows are outside this run. All prepared medium tasks remain included regardless of model performance.

Source: [Open-R1 Codeforces](https://huggingface.co/datasets/open-r1/codeforces), revision `fbe3f6e903ee854eec2e69e9d96d0306cde59baf`, `verifiable/train`. Ten pinned training shards were cached (about 2.4 GB). The official test split and separate generated-test corpus were not downloaded. Files and SHA-256 hashes are in dataset-manifest.json.

Of 8,338 verifiable training rows, 6,684 need separately stored generated tests, 399 carry custom checkers, eight use file/interactive interfaces, and 524 are outside the rating scope or unrated. Of the 723 initial rating/interface candidates, eight have no problem statement and 20 explicitly allow alternative/tolerance-based outputs without a usable checker. The remaining 695 are split into 628 training and 67 development problems. No protected statement overlaps or alias/near-statement duplicate groups were found in this scope. Text matching does not guarantee absence of semantic or pretraining overlap.

Training contains **32,983 official I/O records** and development 3,396. No minimum test-count, reference-pass or sampled-success filter is applied. Complete official suites need not be exhaustive. Harder, generated-test and checker-dependent tasks are deferred, not asserted to be bad data.

Published aliases and near-identical statements are grouped before a fixed hash split (seed 42, roughly 10% development). This split is frozen for the next experiment. Existing MBPP development/confirmation files and the 55 reserved OpenCoder tasks remain unchanged and unqueried during this work.

## Grader compatibility correction

Empty checker metadata does not establish that an answer is unique. Final source inspection found explicit “print any” contracts despite empty checker fields, including 1051/B Relatively Prime Pairs. String comparison would reject legitimate alternative pairings. These 20 contracts are deferred until an appropriate semantic checker is available; no per-problem checker framework was introduced. deferred-contracts.csv records the 20 output contracts and eight missing statements.

A provisional 647-training/68-development dataset and baseline were retained locally as codeforces-python-provisional and codeforces-baseline-001. The final data and baseline here supersede them. This correction came from the source output contract of a training task, not selection on which model answers passed. The remaining tasks retain their previous hash assignment. The final 67-task development run was regenerated after the compatibility rule and final manifest were frozen.

The compatibility rule catches explicit alternative-answer and tolerance wording; it is not a proof that every remaining task has unique output. Source grading assumptions and converted mathematical notation can still be imperfect.

## Python execution and verification

The existing Docker worker now supports complete Python stdin/stdout programs. Each I/O record launches a fresh Python process and temporary working directory. The code receives only its input; expected outputs remain in the parent grader. Per-case time and address-space limits are taken from the source, inside a 2 GiB container with no network or host mounts, non-root execution, a read-only root, PID/capability restrictions and bounded output previews.

No-checker comparison matches the documented upstream policy: whitespace normalization and case normalization for YES/NO. Arbitrary numeric coercion and semantic-equivalence guesses are not applied. Python uses the recorded source limits; this is not an exact replica of Codeforces language-specific timing multipliers.

The reward is binary all-pass. Execution stops on the first failure, reporting both total and checked counts. Unchecked cases are never counted as tested. The evaluator omits a full-test-fraction metric when fail-fast leaves unchecked cases.

All **28 active regression checks passed**, including Python-only syntax, real buffered stdin, normal process exits, output normalization, time/memory limits, fresh working directories, medium retention and explicit unsupported-checker detection. Two known-correct Python solutions also passed all 45 real official records for Beautiful Matrix and Watermelon; grader-preflight.json records those checks. These checks validate the execution path, not the correctness of every upstream test.

## Baseline and next step

The unchanged Qwen2.5-Coder-0.5B-Instruct solves **1/67** development problems greedily: easy **0/30**, medium **1/37**, with a 1,024-token completion cap. The baseline is not a comparison against prior MBPP percentages, because the tasks and I/O contract differ. No weights changed. Spot checks of several easy failures show incorrect algorithms; the low solve rate is not evidence that medium problems should be silently dropped. A single greedy attempt also does not establish whether eight sampled rollouts will produce mixed GRPO rewards.

The dataset and one training entry point are ready. Before a long optimization run, measure sampled reward availability on training tasks while retaining both requested difficulty groups. No full-budget run has started. The frozen development split is for progress tracking; existing confirmation pools and the official Codeforces test split remain unqueried.

Commands:

```bash
docker build -t python-grpo-cpython:v3 -f Dockerfile.evaluation .
.venv/bin/python prepare_data.py
.venv/bin/python launch.py train --name cf-baseline --baseline-only
# A short training check uses the same entry point without --baseline-only.
```

Use fresh output names/directories. Existing prepared data is under data/codeforces-python; skip preparation when reusing it. The old OpenCoder importer has been retired from the root, with its raw data, code history and experiment evidence preserved. No new training framework was added.

Attribution: Open-R1/Hugging Face dataset authors (Guilherme Penedo and colleagues) and original Codeforces problem authors; see the pinned dataset card for citation and the [earlier exploration](../codeforces-exploration/REPORT.md) for the source licence-label discrepancy. Full statement/test payloads remain in the local pinned cache; this archive records metadata, fingerprints, outputs and checks rather than duplicating the corpus.
