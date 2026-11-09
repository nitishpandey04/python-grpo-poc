# Codeforces dataset exploration — 2026-11-09

Recommendation: investigate Open-R1 Codeforces as a better-supported training source before generating more tests ourselves. This is data exploration, not a training run or a claim of model improvement. The existing training code and dataset remain unchanged.

## What was measured

Source: [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces), pinned revision `fbe3f6e903ee854eec2e69e9d96d0306cde59baf`, `verifiable/train`. Only the training split was read. The publisher's test split and our existing confirmation pools remain unqueried.

| Scope | Problems | Mean official I/O records | Median |
|---|---:|---:|---:|
| Verifiable training split | 8,338 | — | — |
| Complete official suites | 1,654 | 54.21 | 46 |
| Complete, standard I/O, no custom checker, non-interactive | 1,247 | 55.37 | 46 |
| Same simple interface, rating 800–1000 | 344 | 45.16 | 38 |

The 344 first-scope candidates include 177 rated 800, 73 rated 900 and 94 rated 1000. Their official suites range from 2 to 271 I/O records. No minimum test-count filter was applied. No alias points to another candidate row. Exact/numeric/semantic overlap with the project's existing development data has not yet been audited for this new source; that remains necessary before training.

These are I/O records, not assertion counts: one input file can contain several logical subcases. For all 1,654 complete suites, the number of supplied official input records matches `testset_size`. The `executable` flag is true throughout this subset; according to the publisher this means at least three accepted human solutions passed the supplied official tests. We did not independently rerun all those solutions.

The other 6,684 verifiable training problems have incomplete official suites but report validated generated tests. They are not intrinsically bad problems. Their corresponding generated tests must be loaded before treating those suites as complete. The publisher documents a separate roughly 110 GB generated-test corpus; selected contest files can be fetched individually. This exploration did not download it.

## Constraints and grading information

The data includes separate description, input format, output format, time limit, memory limit, rating, example tests, official tests, completeness flag and optional custom checker. Constraints are generally expressed in the input-format prose, not as machine-validated numeric schemas. Some superscripts are flattened (for example `109` in extracted text intended to denote a power of ten), so automatic boundary generation still requires care.

The publisher explicitly addresses truncated official inputs and outputs with multiple valid answers. Generated tests are labelled using accepted human submissions and cross-checked against multiple accepted solutions; incorrect submissions are used when available to help identify discriminating tests. Custom checkers handle non-unique answers. A problem requiring a checker must not be graded with simple string equality.

All training rows are 8,301 standard-I/O and 37 file-I/O tasks. Across the split, 1,643 rows carry custom checkers. Deferring file, interactive and custom-checker tasks is an initial implementation scope, not a general data-quality rejection.

For the no-checker path, the publisher's run script normalizes whitespace and the case of YES/NO before comparing output. Our current native-function test path is not a drop-in Codeforces grader: it needs a standard-input/output execution path with explicit limits and matching output semantics. The existing Docker isolation can be reused; a new training framework is unnecessary.

## Concrete test inspection

The metadata was read from small Parquet columns using range requests, leaving large official output columns remote. The dataset preview API timed out and returned 502 errors when fetching individual full rows. Instead, we downloaded the smallest pinned training shard (26,286,337 bytes), which contains 27 first-scope candidate problems. Their complete official suites contain 985 I/O records. This is a convenience sample from one shard, not a representative coverage study.

Four simple problems received independent arithmetic checks against all 142 of their supplied official outputs. All expectations agreed:

| Problem | Official records | Coverage observed | Example faulty alternative rejected |
|---|---:|---|---|
| 263/A Beautiful Matrix | 25 | All 25 valid positions of the single 1; exhaustive for the stated 5x5 board domain | Maximum displacement instead of sum of row/column displacements; fails 16 cases |
| 228/A Horseshoes | 34 | All four distinct-color counts: 1, 2, 3 and 4; large integer colors | Counts adjacent duplicates only; fails 11 cases |
| 266/A Stones on the Table | 34 | Input lengths 1 and 50, intermediate lengths and uniform-color strings | Removes non-adjacent repeated colors too; fails 28 cases |
| 996/A Hit the Lottery | 49 | Amounts 1 and 10^9, several denomination boundaries | Uses only 100 and 1 denominations; fails 43 cases |

Arithmetic used: Manhattan distance to the center, four minus distinct colors, equal-adjacent-pair count, and change using denominations 100/20/10/5/1. Only our trusted calculations executed locally; no source submission or model-generated program was executed. This limited check does not certify all 344 candidates.

There are still gaps. For example, the lottery suite includes 5, 6, 10 and 20, but not 99, 100 or 101. Complete official tests mean all recorded official tests are present, not exhaustive coverage of every valid input. The exhaustive matrix example is possible because that problem has only 25 valid input states.

## Next decision

Use the cached 27-problem sample for a baseline-only model-fit check after adding a small standard-I/O path to the existing sandbox. Keep the official tests as rewards and use no editorials or solutions in the model prompt. Inspect truncation, error types, solve rates and execution cost before selecting a training budget. Rating 800–1000 is a starting hypothesis for our 0.5B model, not evidence that it can solve these tasks. No model predictions or optimization were performed in this exploration.

Once model fit and grading are confirmed, the 344 first-scope candidates provide a manageable initial pool. Keep training and evaluation splits separate, check benchmark overlap/aliases, and expand to generated-test or custom-checker tasks only when needed. Old public Codeforces tasks may have appeared in pretraining; performance here must not be presented as clean generalization evidence.

## Provenance and attribution

`source.json` pins the repository and shard sizes. `summary.json` contains full measured counts. `candidate-problems.csv` lists the 344 candidate IDs, source URLs and metadata. `sample-provenance.json` fingerprints the downloaded shard and 27-row local sample. `boundary-audit.json` records the four detailed checks. Full problem/test payloads remain in the local cache rather than being duplicated in this report.

Attribution: the Open-R1/Hugging Face dataset authors (Guilherme Penedo and colleagues) and the original Codeforces problem authors. See the [dataset card and citation](https://huggingface.co/datasets/open-r1/codeforces/blob/fbe3f6e903ee854eec2e69e9d96d0306cde59baf/README.md). The card metadata declares CC-BY-4.0, while its prose license section uses a different label; that discrepancy is recorded here rather than interpreted as a settled downstream licence determination.
