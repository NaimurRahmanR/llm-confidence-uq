# Confidence Alignment and Uncertainty Quantification in LLMs Under Evidence Degradation

A reproducible PyTorch study of expressed, token-derived, calibrated,
ensemble, and approximate Bayesian-head uncertainty under controlled
evidence perturbations.

> **Status:** Completed application-stage research artefact. All eight
> validation gates passed. This work is not peer reviewed.

## Headline findings

- The strongest single adapter, LoRA seed 2 after temperature scaling,
  reached **73.88%** overall
  accuracy, an absolute **6.50-percentage-point** increase
  over the calibrated frozen baseline (67.38%).
- The calibrated three-LoRA ensemble produced the highest observed
  error-detection AUROC (**0.7102**).
- The Laplace-approximated linear head produced the lowest aggregate
  10-bin ECE (**0.0230**), but not the
  best accuracy or error ranking; low ECE alone is not evidence of superior
  reliability.
- Expressed-confidence parsing remained valid for
  85.46%
  of baseline rows, but only 19.21%
  to 47.00%
  across the three adapters. Invalid values were retained as missing and
  never imputed.

## Study design

- **Task:** BoolQ binary question answering.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` at revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- **Splits:** 800 training, 200 calibration, and 400 held-out test inputs.
- **Conditions:** original evidence plus lexical evidence removal, 50%
  prefix truncation, an irrelevant-distractor proxy, a lexical-contradiction
  proxy, and no passage.
- **Methods:** frozen baseline, three independently seeded LoRA adapters,
  temperature scaling, a three-adapter ensemble, and a
  Laplace-approximated Bayesian binary linear head over frozen transformer
  representations.
- **Runtime:** Python 3.12.13, PyTorch 2.11.0+cu128, NVIDIA A100-SXM4-80GB.

## Overall results

| Method | Accuracy | NLL ↓ | Brier ↓ | ECE (10 bins) ↓ | Error AUROC ↑ |
|---|---:|---:|---:|---:|---:|
| Baseline raw | 67.38% | 0.9824 | 0.2530 | 0.2138 | 0.6735 |
| Baseline calibrated | 67.38% | 0.6035 | 0.2072 | 0.0686 | 0.6735 |
| LoRA seed 1 calibrated | 73.46% | 0.5492 | 0.1821 | 0.0771 | 0.7088 |
| LoRA seed 2 calibrated | 73.88% | 0.5422 | 0.1800 | 0.0756 | 0.7073 |
| LoRA seed 3 calibrated | 72.00% | 0.5723 | 0.1927 | 0.0866 | 0.6879 |
| Three-LoRA ensemble calibrated | 72.96% | 0.5482 | 0.1829 | 0.0723 | 0.7102 |
| Laplace head | 67.67% | 0.6067 | 0.2088 | 0.0230 | 0.6356 |

Metrics are descriptive across all 2,400 condition-level test rows. No
test label was used for training, temperature fitting, prompt selection,
or hyperparameter tuning.

![Method comparison](outputs/results/gate8/figures/08_method_comparison.png)

## Reproduction

Run from the repository root in the pinned Colab environment:

```bash
python scripts/prepare_data.py --config configs/data.yaml --output-dir data/manifests
python scripts/prepare_degradations.py --data-config configs/data.yaml --degradation-config configs/degradations.yaml --output-dir data/degradations
python scripts/run_baseline.py --stage full --config configs/baseline.yaml
python scripts/train_lora.py --stage full_seed_1 --config configs/lora.yaml
python scripts/train_lora.py --stage full_seed_2 --config configs/lora.yaml
python scripts/train_lora.py --stage full_seed_3 --config configs/lora.yaml
python scripts/run_lora_inference.py --adapter full_seed_1 --stage full --config configs/lora_inference.yaml
python scripts/run_lora_inference.py --adapter full_seed_2 --stage full --config configs/lora_inference.yaml
python scripts/run_lora_inference.py --adapter full_seed_3 --stage full --config configs/lora_inference.yaml
python scripts/run_calibration_inference.py --method baseline --stage full --config configs/calibration_inference.yaml
python scripts/run_calibration_inference.py --method full_seed_1 --stage full --config configs/calibration_inference.yaml
python scripts/run_calibration_inference.py --method full_seed_2 --stage full --config configs/calibration_inference.yaml
python scripts/run_calibration_inference.py --method full_seed_3 --stage full --config configs/calibration_inference.yaml
python scripts/fit_temperature.py --method baseline --config configs/temperature.yaml
python scripts/fit_temperature.py --method full_seed_1 --config configs/temperature.yaml
python scripts/fit_temperature.py --method full_seed_2 --config configs/temperature.yaml
python scripts/fit_temperature.py --method full_seed_3 --config configs/temperature.yaml
python scripts/evaluate_predictions.py --method baseline --config configs/evaluation.yaml
python scripts/evaluate_predictions.py --method full_seed_1 --config configs/evaluation.yaml
python scripts/evaluate_predictions.py --method full_seed_2 --config configs/evaluation.yaml
python scripts/evaluate_predictions.py --method full_seed_3 --config configs/evaluation.yaml
python scripts/evaluate_ensemble.py --config configs/ensemble.yaml
python scripts/run_laplace_head.py --stage full --config configs/laplace_head.yaml
python scripts/build_results.py --config configs/results.yaml
python scripts/build_report.py --config configs/report.json
python -m unittest discover -s tests -p 'test_*.py' -v
```

The pipeline refuses incompatible existing artifacts and records both
successful and failed runs. Raw BoolQ passages and LoRA checkpoints are not
committed. See [the reproducibility guide](report/reproducibility.md).

## Repository map

- `configs/`: pinned protocols, revisions, seeds, and hashes.
- `src/llm_confidence_uq/`: data, training, inference, UQ, evaluation, and reporting code.
- `scripts/`: explicit command-line experiment stages.
- `tests/`: unit and contract tests.
- `data/`: deterministic manifests without raw passages.
- `outputs/`: manifests, machine-readable predictions/results, and figures.
- `report/technical_report.md`: complete methods and results report.
- `report/claim_boundaries.md`: supported and prohibited claims.

## Claim boundary

This repository demonstrates parameter-efficient LoRA adaptation, not
full-model training. Only the binary linear prediction head receives a
Laplace approximation; the Qwen transformer is not Bayesian. The three
adapters are independently seeded ensemble members, not posterior samples.
The perturbations are categorical stress tests, not an ordered severity
scale. See [claim boundaries](report/claim_boundaries.md).

## Licences

- Repository code: MIT.
- BoolQ data: CC BY-SA 3.0.
- Qwen2.5-1.5B-Instruct: Apache 2.0.

Third-party materials retain their original licences.
