# OpenCoder acquisition and screening — 2026-11-07

This is a data-curation and difficulty-calibration run. No new model was trained and no model improvement is claimed.

| Stage | Result |
|---|---|
| Pinned source sample | 3000 rows |
| AST / prompt / recognized-contract groups | 1546 |
| Retained problems | 306 |
| Independent narrow contracts with generated cases | 14 |
| Supplied tests + nine constant controls only | 292; semantic review pending |
| Frozen splits | 251 training / 27 development / 28 confirmation |
| Eight-sample training-only calibration | 130 mixed / 86 all fail / 35 all pass |
| Mixed tasks with independent contract checks | 6 |
| Mixed tasks requiring semantic review | 124 |
| Mixed independent tasks after instruction/domain review | 5 on their tested domains |

## Source and reproducibility

Source: [OpenCoder educational_instruct](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2/tree/7d28f40d579edd7c24402d17d0c7639f991e6f8d); license declared by the source is MIT. Attribution belongs to the OpenCoder dataset authors. Native dataset rows are sampled using seed 42, a 10,000-row streaming shuffle buffer, and a 3,000-row cap. Source revision, row IDs, raw SHA-256, protected-data hashes, Docker image ID, and frozen split hashes are in quality-manifest.json. Exact source snapshots and hashes accompany this report. The intermediate curation-v1 output was superseded by the final contract-family-aware curation-v2 pass before any model calibration.

Normalization accepts short synchronous functions with direct literal-input equality assertions. Tests needing shared setup are excluded. References run only in resource-limited Docker. Recognized contracts receive deterministic edge/random cases from separately implemented bounded oracles. Every retained suite rejects nine constant-output controls; all assertions must pass for binary reward. Assertions are bundled into one fresh child process per candidate, with a two-second total suite limit. Unlike the MBPP evaluator, calls within this suite share candidate state. This protocol is recorded explicitly and does not use the MBPP+ scorer.

Canonical AST clones, near-identical prompts, and recognized contract variants are grouped before choosing representatives or splitting. Known overlaps with all existing MBPP validation/test rows are excluded using reference ASTs, prompt similarity, and recognized contracts. This is a conservative filter, not a guarantee of semantic or pretraining decontamination. Representatives are chosen by test/reference quality before any model sampling. Hashes assign groups approximately 80/10/10; task IDs, group IDs, and recognized contracts are disjoint between splits.

The original pinned Qwen2.5-Coder-0.5B-Instruct was sampled eight times on every training task, with temperature 0.8, a 256-token cap, and seed 42. A mixed task has 1–7 fully passing answers. All task-level answers and checks are archived. Development and confirmation receive no model predictions. The existing 49 MBPP confirmation tasks also remain untouched.

## Limits and next decision

Reference/test agreement and rejection of trivial constants do not independently establish task correctness. The lower quality tier must not be called verified or fed into training before semantic review and stronger tests. Narrow independent contracts validate behavior on their generated domain; they do not prove correctness for all inputs or enforce algorithm restrictions such as recursion or a ban on built-ins. Those instruction constraints need review too.

The recorded code-and-instruction review of the 13 independent-contract training rows found four instruction/domain issues: a no-built-ins reference itself calls range; recursion is not enforced by behavioral tests; the parentheses-only test domain needs to be explicit; and digit-sum tests missed negative values unsupported by the reference. contract-review.jsonl records the verdicts. mixed-independent-reviewed.jsonl excludes these four tasks from the independently checked mixed subset. This review is limited to the tested domains and does not waive the need for stronger plausible-error controls.

The new development split contains no independently augmented contracts, so it is not a strong replacement for the existing expanded MBPP development evaluation. Keep the existing 29-task expanded development suite and 49-task untouched confirmation pool while strengthening this new data. Newly frozen OpenCoder confirmation rows are an additional pool whose tests still need review.

The initial target of hundreds of independently checked learnable tasks has not been reached: only 6 mixed tasks currently have independent narrow-contract checks. Review semantic-review-queue.jsonl, add independent edge cases and plausible non-constant faulty implementations, resolve instruction constraints, then freeze a new approved training manifest. The existing trainer rejects this new dataset protocol to prevent accidentally using the MBPP-specific scorer. No longer training run is justified by these screening counts alone.
