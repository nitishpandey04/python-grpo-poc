# Evaluation audit: experiment 001

The exact saved baseline and final-adapter answers were re-scored. No model generation or training was performed. Historical experiment-001 files remain unchanged.

| Split and evaluator | Problems | Original model solved | Final adapter solved |
|---|---:|---:|---:|
| test/original_cpython | 300 | 99 | 106 |
| test/expanded_compatible | 96 | 40 | 43 |
| validation/original_cpython | 79 | 30 | 25 |
| validation/expanded_compatible | 29 | 16 | 14 |

## Interpretation and coverage

Original-test results isolate the change from the restricted interpreter to CPython. Expanded results use EvalPlus 0.3.1 with MBPP+ v0.2.0, on the matched subset whose original reference passes both official base and extra tests. Eligibility is checked independently of either model's answers. Test inputs, types, floating-point tolerances, and special oracles use upstream EvalPlus code.

This is a retrospective audit of old-prompt answers on a compatible subset, not a standard full MBPP+ benchmark or fresh evidence of general improvement. Uncovered tasks and reference/contract incompatibilities are explicitly reported. This audit must not be used to choose training settings against the inspected test set. New training needs a newly frozen evaluation protocol and fresh confirmation data.

Original tests run in a fresh bounded CPython child per test; EvalPlus uses its official within-task state semantics. Results may differ because of interpreter behavior, oracle semantics, stronger tests, or reference incompatibilities. Passing three original tests is not proof of a correct oracle.

## Coverage details

```json
{
  "test/coverage": {
    "saved_problems": 300,
    "invalid_original_references": [],
    "expanded_exclusions": {
      "not_in_mbppplus": 167,
      "eligible": 96,
      "original_reference_failed_expanded_tests": 35,
      "entry_point_mismatch": 1,
      "canonical_oracle_resource_limit": 1
    }
  },
  "validation/coverage": {
    "saved_problems": 79,
    "invalid_original_references": [],
    "expanded_exclusions": {
      "eligible": 29,
      "not_in_mbppplus": 43,
      "original_reference_failed_expanded_tests": 7
    }
  }
}
```

Reproduction metadata, source SHA-256 fingerprints, Docker image ID, dataset fingerprint, task-level scores, error types, and coverage exclusions are retained in `manifest.json`, `results.jsonl`, and `summary.json`.
