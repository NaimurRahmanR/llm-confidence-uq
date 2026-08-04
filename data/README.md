# Data reconstruction

Raw BoolQ records are intentionally excluded from version control.

## Frozen source

- Dataset: `google/boolq`
- Revision: `35b264d03638db9f4ce671b711558bf7ff0f80d5`
- Licence: CC BY-SA 3.0
- Selection: `sha256-stratified-v1`
- Seed: `20260803`
- Prompt-token eligibility ceiling: 768

## Completed deterministic allocation

| Research split | Official source | Rows | False | True |
|---|---|---:|---:|---:|
| Training | train | 800 | 301 | 499 |
| Calibration | train, disjoint from training | 200 | 75 | 125 |
| Test | validation | 400 | 151 | 249 |

Manifests contain source coordinates, labels, prompt-token lengths, and
cryptographic identities, but no raw questions or passages. Reconstruction
refuses dataset-revision or source-hash drift.

Test labels were unavailable during model inference and were not used for
temperature fitting, threshold selection, prompt selection, or
hyperparameter tuning. Evidence transformations were label blind.
