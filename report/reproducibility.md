# Reproducibility

## Frozen identities

- Dataset: `google/boolq` at `35b264d03638db9f4ce671b711558bf7ff0f80d5`.
- Model/tokenizer: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Split selection: `sha256-stratified-v1` with seed `20260803`.
- LoRA seeds: `20260811`, `20260812`, and `20260813`.
- Laplace posterior sampling seed: `20260821`.

## Environment

The verified full runs used:

- Python `3.12.13`
- PyTorch `2.11.0+cu128`
- Transformers `5.13.1`
- PEFT `0.19.1`
- CUDA build `12.8`
- GPU `NVIDIA A100-SXM4-80GB` with 79.25 GiB reported VRAM
- `torchao` excluded after an incompatible optional installation was diagnosed.

Pinned direct Python dependencies are listed in `requirements.txt` and `pyproject.toml`.

## Data isolation

- Training uses only the frozen 800-example training manifest.
- Temperature fitting uses only 200 calibration examples under original evidence.
- The 400 test inputs originate from the official BoolQ validation split.
- Test labels are joined only after inference and are never used for fitting or tuning.
- Raw BoolQ passages are reconstructed from pinned source coordinates and are not committed.

## Artifact integrity

Each completed stage publishes payload hashes and a completion marker last. Existing differing artifacts are rejected. Failed-run manifests are preserved. Gate 8 binds each figure to its machine-readable source table and records that no manual result values were used.

Historical experiment manifests contain `git_head: null` because the runs occurred before the repository's first commit. Their exact configuration, input, checkpoint, prediction, and output hashes remain recorded. The eventual release commit identifies the published source snapshot but does not retroactively change those manifests.

## Full command order

Use the commands in the root README in order, retaining smoke gates before full runs. GPU stages are baseline inference, LoRA training/inference, calibration inference, and frozen-representation extraction. Temperature fitting, evaluation, ensemble aggregation, result construction, and report generation support CPU execution.

## Files intentionally excluded

- Hugging Face caches and reconstructed raw data.
- LoRA adapter/model checkpoint payloads under `outputs/checkpoints/`.
- Secrets, `.env` files, notebook state, and temporary files.

Checkpoint and prediction identities remain available through SHA-256 values in configurations, ledgers, and run manifests.

## Documentation lineage

The following verified inputs generate the README, report, citation, data note, and claim boundaries:

| Input | SHA-256 |
|---|---|
| `outputs/results/gate8/summary.json` | `b3ad0f23112ea836b01456f12df4dcd9c725c7a584c01dcc1c6cb6a474b1a9ba` |
| `outputs/results/gate8/method_metrics.jsonl` | `4c695007811fd131cbae2689b81d827e491ef39b6be9e79f617df91c773bcd39` |
| `outputs/results/gate8/paired_changes.jsonl` | `9b9d40a1378b7232d4bd0882cf8d60b60697eb5848fafb19601bb4e41fb91ef8` |
| `outputs/results/gate8/expressed_summary.jsonl` | `2208397df19e810665ca711e71ebf35e2b7806f5dce80ed8e954e3ed2676f854` |
| `outputs/results/gate8/figure_manifest.json` | `9fe8792c94f3c3cce4e70f14bb80e48bc0222659eb3007c84d1625faa4045291` |
| `data/manifests/summary.json` | `5f45de15e667f18289174ad387f05b18c9355a8ac53d9b0e4f05c87fff6f1a74` |
| `data/degradations/summary.json` | `5fbdd462ca40cf6db9cdfc846e9d347b6d0d394e80e531cf3aa0b6d08f86f2dd` |
| `configs/data.yaml` | `412fd6da74212e071159463e104329efefbc8fbe6b853c29c825e3ab37cb64ac` |
| `configs/degradations.yaml` | `303238ae02025c9e5cb59bbd061949b6798f5fde8936a1f0604d15f0552cf2e3` |
| `configs/lora.yaml` | `90e1f7b14c97603caa5d2d50620afbba7bb56f5aa17a0a5bc8f23f7b6d513c69` |
| `configs/ensemble.yaml` | `085c26a387f649aa667911209af5b52bc4eecda1e52b423b65b7e224481d931b` |
| `configs/laplace_head.yaml` | `7feb85f16088a05e2882bb5dd6db5dfa4aac49b16873b26b989eeda9d7dae48c` |
| `outputs/calibration/temperature/baseline/summary.json` | `cc1d0c8e77e21ad782b6c1287d892c03b02f0f83ccf838ce68210a23801aa90f` |
| `outputs/calibration/temperature/full_seed_1/summary.json` | `6782660495a9b91ba86ca90a839efc5c45cf71367544649257923e30a3e08c83` |
| `outputs/calibration/temperature/full_seed_2/summary.json` | `d6d6b7744dea660f5af61614269e2198136a964dba5cec151d43f42904657660` |
| `outputs/calibration/temperature/full_seed_3/summary.json` | `33a86878294f132c212eab4b4d5108cdc4b800a89cefb337199cf70b34ae4fcc` |
| `outputs/manifests/lora/full_seed_1-success-2026-08-04T13-38-29.751818_00-00.json` | `d4b1014a9bce56867ea705225d0ec9cf2cda9144f1422a5eac3eff4559e3045a` |
| `outputs/manifests/lora/full_seed_2-success-2026-08-04T13-47-03.972891_00-00.json` | `b260b7e1cbbed223df4a704d75259e7fd54aa4da0175047321710f08e5f5f5b3` |
| `outputs/manifests/lora/full_seed_3-success-2026-08-04T13-54-03.460582_00-00.json` | `059514d058336cfe453b811c35ede98269783c4287c25df9637f5fe45aee684b` |
| `outputs/ensemble/three_lora/metrics.json` | `98c28866d7ee77522ed926186799d3b62e950301bb2f899516a41dee4b29a74a` |
| `outputs/laplace_head/full/metrics.json` | `1ccf074a0886a96a373b440a085d0ab84be447a5638054b6e84422ba899a4b53` |
