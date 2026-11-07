# Stronger OpenCoder training-data review — 2026-11-07

This work strengthens data and measures baseline difficulty. No new model was trained, and no model improvement is claimed.

| Stage | Result |
|---|---|
| Previously mixed tasks reviewed | 130 |
| Conservative related held-out-family exclusions | 60 |
| Deferred contract/algorithm constraints | 33 |
| Duplicate reviewed contracts collapsed | 3 |
| References failing stronger tests | 5 |
| Independently specified contracts passing quality checks | 29 |
| Fresh baseline calibration | 22 mixed / 6 all fail / 1 all pass |
| Selected reviewed training tasks | 22 |
| Existing development / confirmation pools | 29 / 49; confirmation unqueried |

## What changed

All 130 formerly mixed tasks received fingerprinted review decisions. Ambiguous wording, incorrect examples, unspecified output order, and unenforced algorithm restrictions are deferred. Existing MBPP validation/test text and the frozen OpenCoder held-out text were consulted only to exclude related training contracts; no held-out model predictions were generated. Some exclusions deliberately remove broader related families rather than exact duplicates, and imperfect wording filters can still miss semantic overlap. Multiple source implementations of one reviewed contract receive one training representative, chosen by supplied-test count and reference length before model rescoring.

Separately specified bounded oracles generate new literal expected results without executing source code on the host. Coverage includes exhaustive short attendance/binary/periodic-string domains, brute-force small permutations, negative cubes, duplicate-sensitive lists, format boundaries, and empty-input cases where the contract defines the result. Each retained reference passes the stronger suite, and the suite rejects two handwritten nonconstant faulty implementations, nine constant outputs, and a lookup memorizing the old examples. The controls and all outcomes are archived. References, generated answers, and faulty controls execute only inside resource-limited Docker.

Across retained contracts, all 348 controls fail, including 58 nonconstant controls. Stronger tests newly reject 11 nonconstant controls and 29 old-example lookups that passed the old suites. The five reference defects include empty-input failures for simplified tags and search insertion, acceptance of a trailing newline in a six-digit hex color, counting row length instead of non-space runs, and treating zero as containing no zero digit. See [reference defects](REFERENCE_DEFECTS.md) and [contract domains](CONTRACTS.md).

## Baseline calibration

Rechecking the same 232 frozen baseline answers reduces passing answers from 102 to 91. This exposes 11 previously passing answers rejected by the stronger criteria. The model has not changed: these are stricter test outcomes, not a performance regression from training. Under these frozen samples, 26 contracts remain mixed and 3 become all-fail.

A separate fresh calibration then samples the original pinned Qwen2.5-Coder-0.5B-Instruct eight times on each of the 29 retained training contracts, at temperature 0.8, 256 tokens, and seed 42. It selects 22 tasks with 1–7 fully passing candidates. Fresh generation has a different task order/random stream from the original 251-task calibration; the two sets of samples are not a paired model comparison. Live metrics are recorded under calibration-opencoder-v2 in Trackio, with calibration/percent rather than optimizer steps showing progress.

## Prepared inputs and limits

training-manifest.json links train.jsonl to the existing expanded MBPP development protocol and fingerprints the existing 49-task confirmation pool. The copied OpenCoder development/confirmation sets remain unreviewed and are not substituted for stronger evaluation. All held-out source hashes remain unchanged. The MBPP-specific trainer intentionally rejects the new protocol; explicit train-scorer/development-scorer integration is required before optimization, so data cannot silently use the wrong reward.

The selected pool has 22 reviewed contract families, still far below hundreds of diverse problems. Finite generated cases and a small handpicked control set do not prove correctness; domains exclude some ambiguous inputs, such as leading-zero IPv4 policy and invalid window sizes. These reviews were performed by the assistant by reading instructions, references and oracle code, not by independent human annotators. Public pretraining overlap and undetected semantic overlap remain possible. The existing development set is small and has already informed decisions. Use the untouched confirmation pool only after a promising, predeclared development comparison, and avoid treating training reward or calibration selection as evidence of model improvement.

All 31 regression checks passed, including Docker tests reproducing reference defects and rejecting faulty implementations. Quality checks, frozen-answer rescores, fresh calibration outputs, source snapshots, fingerprints, and the upstream OpenCoder MIT license accompany this report. OpenCoder attribution and the pinned data source are retained from [the import report](../opencoder-data-v1/REPORT.md).
