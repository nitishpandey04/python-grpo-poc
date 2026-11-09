# Fixed evaluation inputs

`development.jsonl` is the unchanged 29-problem development set from the earlier expanded-MBPP pilot. `protected-statements.jsonl` contains statement text and IDs used only to exclude training overlaps: 527 prior MBPP validation/test tasks and the 55 previously reserved OpenCoder tasks. No reference solution or model outcome is used by the overlap check.

The manifest fingerprints these files and the existing 49-task MBPP confirmation pool. Training reads confirmation IDs only and never generates confirmation predictions. The confirmation questions are not required for a development-only run.

MBPP text and development examples originate from Google Research's Mostly Basic Python Problems dataset: https://github.com/google-research/google-research/tree/master/mbpp . OpenCoder statements originate from OpenCoder-LLM/opc-sft-stage2, pinned revision 7d28f40d579edd7c24402d17d0c7639f991e6f8d; the upstream MIT notice is retained under third_party/OPENCODER_LICENSE. Original source-file SHA-256 hashes are recorded in manifest.json. Statement matching does not guarantee semantic or pretraining decontamination.
