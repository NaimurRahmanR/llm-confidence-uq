# Confidence Alignment and Uncertainty Quantification in LLMs Under Evidence Degradation

**LoRA adaptation improved binary QA performance, but generated confidence
formatting became substantially less reliable; temperature scaling improved
probabilistic scores, while ensemble and Laplace methods showed different
calibration and error-ranking trade-offs.**

A reproducible PyTorch study of expressed, token-derived, calibrated,
ensemble, and approximate Bayesian-head uncertainty under controlled
evidence perturbations, with paired bootstrap intervals clustered by input.

> **Status:** Completed application-stage research artefact. All original
> eight validation gates and the post-release statistical-analysis gate
> passed. This work is not peer reviewed.

## Headline findings

- Across all three calibrated LoRA seeds, overall accuracy was
  **73.11% mean ± 0.98 percentage points**
  (range 72.00%–73.88%).
  The mean paired improvement over the calibrated baseline was
  **5.74 points**
  (95% input-cluster bootstrap interval
  2.96 to
  8.33).
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

### Three-seed LoRA summary

| Metric | Three-seed mean | Sample SD | Range |
|---|---:|---:|---:|
| Accuracy | 73.11% | 0.98 pp | 72.00%–73.88% |
| NLL | 0.5545 | 0.0158 | 0.5422–0.5723 |
| Brier | 0.1849 | 0.0068 | 0.1800–0.1927 |
| ECE (10 bins) | 0.0797 | 0.0060 | 0.0756–0.0866 |
| Error AUROC | 0.7013 | 0.0117 | 0.6879–0.7088 |

### Paired uncertainty and direct UQ diagnostics

| Method and score | Error AUROC | Error AUPRC | Any-degraded AUROC | Any-degraded AUPRC |
|---|---:|---:|---:|---:|
| Calibrated baseline predictive entropy | 0.6735 | 0.4639 | 0.6441 | 0.8892 |
| Calibrated LoRA predictive entropy (three-seed mean ± SD) | 0.7013 ± 0.0117 | 0.4085 ± 0.0118 | 0.6213 ± 0.0007 | 0.8781 ± 0.0008 |
| Calibrated ensemble predictive entropy | 0.7101 | 0.4334 | 0.6232 | 0.8798 |
| Calibrated ensemble member-probability variance | 0.6459 | 0.3657 | 0.5956 | 0.8663 |
| Calibrated ensemble MI-style disagreement | 0.5949 | 0.3336 | 0.5683 | 0.8577 |
| Laplace predictive entropy | 0.6356 | 0.4373 | 0.5631 | 0.8609 |
| Laplace posterior-predictive variance | 0.6200 | 0.4204 | 0.5712 | 0.8719 |
| Laplace mutual information | 0.6083 | 0.4091 | 0.5725 | 0.8731 |

**Any-degraded AUPRC baseline prevalence = 0.8333.** Degraded evidence is
the positive class for five of the six condition rows per input.
Error AUPRC should not be compared naively across models because each model has a different error prevalence.

Intervals for accuracy, NLL, Brier, ECE, and error AUROC differences use
2,000 paired percentile bootstrap replicates over the 400 input IDs; all six
conditions travel with each sampled input. Full comparison and condition
tables are in `outputs/results/statistical_analysis/`.

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
python scripts/build_statistical_analysis.py --config configs/statistical_analysis.yaml
python scripts/build_report.py --config configs/report.json
python -m unittest discover -s tests -p 'test_*.py' -v
```

For a fresh Colab runtime, run `bash colab/setup.sh` first. The pipeline refuses incompatible existing artifacts and records both
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
