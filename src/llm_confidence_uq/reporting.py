"""Evidence-derived release documentation for the completed BoolQ study."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-release-documentation-v1"
EXPECTED_CONDITIONS = (
    "original",
    "lexical_evidence_removal",
    "prefix_truncation_50",
    "irrelevant_distractor",
    "lexical_contradiction",
    "no_passage",
)
PRIMARY_VARIANTS = (
    "baseline_raw",
    "baseline_calibrated",
    "full_seed_1_calibrated",
    "full_seed_2_calibrated",
    "full_seed_3_calibrated",
    "three_lora_ensemble_calibrated",
    "laplace_head",
)
DISPLAY = {
    "baseline_raw": "Baseline raw",
    "baseline_calibrated": "Baseline calibrated",
    "full_seed_1_calibrated": "LoRA seed 1 calibrated",
    "full_seed_2_calibrated": "LoRA seed 2 calibrated",
    "full_seed_3_calibrated": "LoRA seed 3 calibrated",
    "three_lora_ensemble_calibrated": "Three-LoRA ensemble calibrated",
    "laplace_head": "Laplace head",
}
CONDITION_DISPLAY = {
    "original": "Original",
    "lexical_evidence_removal": "Lexical evidence removal",
    "prefix_truncation_50": "Prefix truncation (50%)",
    "irrelevant_distractor": "Irrelevant distractor",
    "lexical_contradiction": "Lexical contradiction",
    "no_passage": "No passage",
}
FIGURES = (
    "01_reliability_diagram.png",
    "02_risk_coverage.png",
    "03_accuracy_by_condition.png",
    "04_ece_by_condition.png",
    "05_expressed_vs_token_confidence.png",
    "06_confidence_change.png",
    "07_uncertainty_change.png",
    "08_method_comparison.png",
)


class ReportingError(RuntimeError):
    """Raised when evidence or documentation violates the release contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportingError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _read_json(payload: bytes, name: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as error:
        raise ReportingError(f"invalid JSON input: {name}") from error


def _read_jsonl(payload: bytes, name: str) -> list[dict[str, Any]]:
    require(payload.endswith(b"\n"), f"JSONL input lacks final newline: {name}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        require(bool(line), f"blank JSONL row: {name}:{index}")
        value = json.loads(line)
        require(isinstance(value, dict), f"non-object JSONL row: {name}:{index}")
        rows.append(value)
    return rows


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "report configuration is not a mapping")
    require(config.get("schema_version") == SCHEMA_VERSION, "report schema drift")
    require(config.get("protocol_version") == PROTOCOL_VERSION, "report protocol drift")
    return config


def load_inputs(repo: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for name, specification in config["inputs"].items():
        path = repo / str(specification["path"])
        require(path.is_file(), f"missing report input: {specification['path']}")
        payload = path.read_bytes()
        require(
            sha256_bytes(payload) == specification["sha256"],
            f"report input hash drift: {specification['path']}",
        )
        if path.suffix == ".json":
            outputs[name] = _read_json(payload, name)
        elif path.suffix == ".jsonl":
            outputs[name] = _read_jsonl(payload, name)
        else:
            outputs[name] = payload.decode("utf-8")
    validate_inputs(outputs)
    return outputs


def _finite_unit(value: Any, label: str) -> float:
    number = float(value)
    require(math.isfinite(number) and 0.0 <= number <= 1.0, f"invalid {label}")
    return number


def validate_inputs(inputs: Mapping[str, Any]) -> None:
    summary = inputs["results_summary"]
    require(summary["schema_version"] == 1, "result schema drift")
    require(summary["rows_per_variant"] == 2400, "result row count drift")
    require(summary["inputs"] == 400, "result input count drift")
    require(tuple(summary["conditions"]) == EXPECTED_CONDITIONS, "condition drift")
    require(tuple(summary["primary_variants"]) == PRIMARY_VARIANTS, "primary variant drift")
    boundaries = summary["claim_boundaries"]
    require(boundaries["test_labels_used_for_fitting_or_tuning"] is False, "test-label leakage claim")
    require(boundaries["full_transformer_is_bayesian"] is False, "Bayesian transformer overclaim")
    require(boundaries["ensemble_members_are_posterior_samples"] is False, "ensemble posterior overclaim")

    primary = summary["primary_overall"]
    require(set(primary) == set(PRIMARY_VARIANTS), "primary result keys drift")
    for variant, row in primary.items():
        require(row["rows"] == 2400 and row["scope"] == "overall", f"invalid result scope: {variant}")
        for metric in ("accuracy", "brier", "ece_10_bin", "error_detection_auroc"):
            _finite_unit(row[metric], f"{variant}/{metric}")
        require(math.isfinite(float(row["nll"])) and row["nll"] >= 0.0, f"invalid NLL: {variant}")

    metrics = inputs["method_metrics"]
    require(len(metrics) == 77, "method-metric cardinality drift")
    require(Counter(row["variant"] for row in metrics) == Counter({name: 7 for name in summary["all_variants"]}), "method-metric variant drift")
    require(len(inputs["paired_changes"]) == 66, "paired-change cardinality drift")
    require(len(inputs["expressed_summary"]) == 28, "expressed-summary cardinality drift")

    figure_manifest = inputs["figure_manifest"]
    require(figure_manifest["manual_result_values_used"] is False, "manual figure values detected")
    require(tuple(figure_manifest["figures"]) == FIGURES, "figure set drift")

    data = inputs["data_summary"]
    require(data["selected_sizes"] == {"train": 800, "calibration": 200, "test": 400}, "data split drift")
    degradation = inputs["degradation_summary"]
    require(degradation["condition_count"] == 6, "degradation count drift")
    require(tuple(degradation["condition_order"]) == EXPECTED_CONDITIONS, "degradation order drift")

    for key in ("training_seed_1", "training_seed_2", "training_seed_3"):
        manifest = inputs[key]
        report = manifest["report"]
        require(manifest["status"] == "success", f"training did not succeed: {key}")
        require(report["all_losses_finite"] is True, f"non-finite training loss: {key}")
        require(report["parameters_changed"] is True, f"unchanged adapter: {key}")
        require(report["all_trainable_parameters_received_gradients"] is True, f"missing gradients: {key}")
        require(report["optimizer_steps"] == 75, f"optimizer-step drift: {key}")

    for key in ("temperature_baseline", "temperature_seed_1", "temperature_seed_2", "temperature_seed_3"):
        temperature = inputs[key]
        require(float(temperature["temperature"]) > 0.0, f"non-positive temperature: {key}")
        require(temperature["test_labels_used"] is False, f"test labels used for temperature: {key}")
        require(temperature["predictions_unchanged"] is True, f"temperature changed classes: {key}")

    ensemble = inputs["ensemble_metrics"]
    require(ensemble["test_labels_used_for_fitting_or_tuning"] is False, "ensemble test tuning")
    require(ensemble["rows"] == 2400 and len(ensemble["members"]) == 3, "ensemble contract drift")
    laplace = inputs["laplace_metrics"]
    require(laplace["full_transformer_is_bayesian"] is False, "Laplace scope overclaim")
    require(laplace["test_labels_used_for_fitting_or_tuning"] is False, "Laplace test tuning")
    require(laplace["rows"] == 2400, "Laplace row drift")

    statistical = inputs["statistical_summary"]
    require(statistical["protocol_version"] == "boolq-clustered-statistical-analysis-v1", "statistical protocol drift")
    require(statistical["clusters"] == 400 and statistical["conditions_per_cluster"] == 6, "statistical cluster drift")
    require(statistical["bootstrap_repetitions"] == 2000, "bootstrap repetition drift")
    require(len(inputs["bootstrap_differences"]) == 49, "bootstrap comparison cardinality drift")
    require(len(inputs["lora_seed_summary"]) == 7, "LoRA seed summary cardinality drift")
    require(len(inputs["uq_error_detection"]) == 11, "UQ error-signal cardinality drift")
    require(len(inputs["uq_degradation_detection"]) == 66, "UQ degradation-signal cardinality drift")
    require(len(inputs["lora_seed_uq_summary"]) == 1, "LoRA seed UQ summary cardinality drift")


def pct(value: Any, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def dec(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def signed(value: Any, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def _metric_index(inputs: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(row["variant"], row["scope"]): row for row in inputs["method_metrics"]}


def _paired_index(inputs: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(row["variant"], row["condition"]): row for row in inputs["paired_changes"]}


def _expressed_index(inputs: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(row["method"], row["scope"]): row for row in inputs["expressed_summary"]}


def _overall_table(inputs: Mapping[str, Any]) -> str:
    rows = inputs["results_summary"]["primary_overall"]
    lines = [
        "| Method | Accuracy | NLL ↓ | Brier ↓ | ECE (10 bins) ↓ | Error AUROC ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in PRIMARY_VARIANTS:
        row = rows[variant]
        lines.append(
            f"| {DISPLAY[variant]} | {pct(row['accuracy'])} | {dec(row['nll'])} | "
            f"{dec(row['brier'])} | {dec(row['ece_10_bin'])} | {dec(row['error_detection_auroc'])} |"
        )
    return "\n".join(lines)


def _condition_table(inputs: Mapping[str, Any]) -> str:
    metrics = _metric_index(inputs)
    variants = (
        "baseline_calibrated",
        "full_seed_2_calibrated",
        "three_lora_ensemble_calibrated",
        "laplace_head",
    )
    lines = [
        "| Evidence condition | Baseline calibrated | LoRA seed 2 calibrated | Calibrated ensemble | Laplace head |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in EXPECTED_CONDITIONS:
        values = [pct(metrics[(variant, condition)]["accuracy"]) for variant in variants]
        lines.append(f"| {CONDITION_DISPLAY[condition]} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _temperature_table(inputs: Mapping[str, Any]) -> str:
    rows = (
        ("Baseline", inputs["temperature_baseline"]),
        ("LoRA seed 1", inputs["temperature_seed_1"]),
        ("LoRA seed 2", inputs["temperature_seed_2"]),
        ("LoRA seed 3", inputs["temperature_seed_3"]),
    )
    lines = [
        "| Method | Temperature | Calibration NLL before | Calibration NLL after |",
        "|---|---:|---:|---:|",
    ]
    for label, row in rows:
        lines.append(
            f"| {label} | {dec(row['temperature'])} | {dec(row['initial_nll'])} | {dec(row['final_nll'])} |"
        )
    return "\n".join(lines)


def _training_table(inputs: Mapping[str, Any]) -> str:
    lines = [
        "| Adapter | Seed | Epoch mean losses | Training time | Peak allocated GPU memory |",
        "|---|---:|---|---:|---:|",
    ]
    for index, key in enumerate(("training_seed_1", "training_seed_2", "training_seed_3"), start=1):
        manifest = inputs[key]
        report = manifest["report"]
        losses = ", ".join(dec(item["mean_microbatch_loss"]) for item in report["epoch_reports"])
        lines.append(
            f"| LoRA seed {index} | {report['seed']} | {losses} | "
            f"{float(manifest['training_seconds']):.1f} s | {float(manifest['peak_allocated_gpu_gib']):.2f} GiB |"
        )
    return "\n".join(lines)


def _expressed_table(inputs: Mapping[str, Any]) -> str:
    expressed = _expressed_index(inputs)
    methods = (
        ("baseline", "Baseline"),
        ("full_seed_1", "LoRA seed 1"),
        ("full_seed_2", "LoRA seed 2"),
        ("full_seed_3", "LoRA seed 3"),
    )
    lines = [
        "| Method | Parser-valid rows | Valid rate | Mean expressed confidence (valid only) | Mean calibrated absolute divergence (valid only) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in methods:
        row = expressed[(method, "overall")]
        lines.append(
            f"| {label} | {row['valid_rows']}/2400 | {pct(row['valid_rate'])} | "
            f"{pct(row['mean_expressed_confidence_valid_only'])} | "
            f"{dec(row['mean_calibrated_absolute_divergence_valid_only'])} |"
        )
    return "\n".join(lines)


def _lora_seed_summary_table(inputs: Mapping[str, Any]) -> str:
    row = next(item for item in inputs["lora_seed_summary"] if item["scope"] == "overall")
    labels = (
        ("accuracy", "Accuracy", True),
        ("nll", "NLL", False),
        ("brier", "Brier", False),
        ("ece_10_bin", "ECE (10 bins)", False),
        ("error_detection_auroc", "Error AUROC", False),
    )
    lines = [
        "| Metric | Three-seed mean | Sample SD | Range |",
        "|---|---:|---:|---:|",
    ]
    for key, label, percentage in labels:
        values = row["metrics"][key]
        if percentage:
            lines.append(
                f"| {label} | {pct(values['mean'])} | {100.0 * values['sample_standard_deviation']:.2f} pp | "
                f"{pct(values['minimum'])}–{pct(values['maximum'])} |"
            )
        else:
            lines.append(
                f"| {label} | {dec(values['mean'])} | {dec(values['sample_standard_deviation'])} | "
                f"{dec(values['minimum'])}–{dec(values['maximum'])} |"
            )
    return "\n".join(lines)


def _interval(value: Mapping[str, Any], *, percentage: bool = False) -> str:
    difference = float(value["difference_candidate_minus_reference"])
    lower = float(value["percentile_interval_lower"])
    upper = float(value["percentile_interval_upper"])
    if percentage:
        return f"{100.0 * difference:+.2f} pp [{100.0 * lower:+.2f}, {100.0 * upper:+.2f}]"
    return f"{difference:+.4f} [{lower:+.4f}, {upper:+.4f}]"


def _bootstrap_table(inputs: Mapping[str, Any]) -> str:
    rows = [row for row in inputs["bootstrap_differences"] if row["scope"] == "overall"]
    labels = {
        "baseline_calibrated_vs_raw": "Baseline calibrated − raw",
        "lora_seed_1_vs_baseline_calibrated": "LoRA seed 1 − calibrated baseline",
        "lora_seed_2_vs_baseline_calibrated": "LoRA seed 2 − calibrated baseline",
        "lora_seed_3_vs_baseline_calibrated": "LoRA seed 3 − calibrated baseline",
        "lora_seed_mean_vs_baseline_calibrated": "LoRA three-seed mean − calibrated baseline",
        "ensemble_calibrated_vs_baseline_calibrated": "Calibrated ensemble − calibrated baseline",
        "laplace_head_vs_baseline_calibrated": "Laplace head − calibrated baseline",
    }
    lines = [
        "| Paired comparison | Δ accuracy [95% CI] | Δ NLL [95% CI] | Δ Brier [95% CI] | Δ ECE [95% CI] | Δ error AUROC [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {labels[row['comparison']]} | {_interval(metrics['accuracy'], percentage=True)} | "
            f"{_interval(metrics['nll'])} | {_interval(metrics['brier'])} | "
            f"{_interval(metrics['ece_10_bin'])} | {_interval(metrics['error_detection_auroc'])} |"
        )
    return "\n".join(lines)


def _uq_signal_table(inputs: Mapping[str, Any]) -> str:
    error = {(row["variant"], row["signal"]): row for row in inputs["uq_error_detection"]}
    degraded = {
        (row["variant"], row["signal"]): row
        for row in inputs["uq_degradation_detection"]
        if row["target"] == "any_degraded"
    }
    seed = inputs["lora_seed_uq_summary"][0]
    lines = [
        "| Method and score | Error AUROC | Error AUPRC | Any-degraded AUROC | Any-degraded AUPRC |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline = error[("baseline_calibrated", "predictive_entropy")]
    baseline_degraded = degraded[("baseline_calibrated", "predictive_entropy")]
    lines.append(
        f"| Calibrated baseline predictive entropy | {dec(baseline['auroc'])} | {dec(baseline['average_precision'])} | "
        f"{dec(baseline_degraded['auroc'])} | {dec(baseline_degraded['average_precision'])} |"
    )
    seed_error = seed["error_detection"]
    seed_degraded = seed["degradation_detection"]["any_degraded"]
    lines.append(
        "| Calibrated LoRA predictive entropy (three-seed mean ± SD) | "
        f"{dec(seed_error['auroc']['mean'])} ± {dec(seed_error['auroc']['sample_standard_deviation'])} | "
        f"{dec(seed_error['average_precision']['mean'])} ± {dec(seed_error['average_precision']['sample_standard_deviation'])} | "
        f"{dec(seed_degraded['auroc']['mean'])} ± {dec(seed_degraded['auroc']['sample_standard_deviation'])} | "
        f"{dec(seed_degraded['average_precision']['mean'])} ± {dec(seed_degraded['average_precision']['sample_standard_deviation'])} |"
    )
    selections = (
        ("three_lora_ensemble_calibrated", "predictive_entropy", "Calibrated ensemble predictive entropy"),
        ("three_lora_ensemble_calibrated", "ensemble_member_probability_variance", "Calibrated ensemble member-probability variance"),
        ("three_lora_ensemble_calibrated", "ensemble_mi_style_disagreement", "Calibrated ensemble MI-style disagreement"),
        ("laplace_head", "predictive_entropy", "Laplace predictive entropy"),
        ("laplace_head", "laplace_posterior_predictive_variance", "Laplace posterior-predictive variance"),
        ("laplace_head", "laplace_mutual_information", "Laplace mutual information"),
    )
    for variant, signal, label in selections:
        error_row = error[(variant, signal)]
        degraded_row = degraded[(variant, signal)]
        lines.append(
            f"| {label} | {dec(error_row['auroc'])} | {dec(error_row['average_precision'])} | "
            f"{dec(degraded_row['auroc'])} | {dec(degraded_row['average_precision'])} |"
        )
    return "\n".join(lines)


def render_readme(inputs: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    primary = inputs["results_summary"]["primary_overall"]
    seed_summary = next(row for row in inputs["lora_seed_summary"] if row["scope"] == "overall")
    seed_accuracy = seed_summary["metrics"]["accuracy"]
    seed_comparison = next(
        row for row in inputs["bootstrap_differences"]
        if row["comparison"] == "lora_seed_mean_vs_baseline_calibrated" and row["scope"] == "overall"
    )
    accuracy_interval = seed_comparison["metrics"]["accuracy"]
    return f"""# {config['release']['title']}

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
  **{pct(seed_accuracy['mean'])} mean ± {100.0 * seed_accuracy['sample_standard_deviation']:.2f} percentage points**
  (range {pct(seed_accuracy['minimum'])}–{pct(seed_accuracy['maximum'])}).
  The mean paired improvement over the calibrated baseline was
  **{100.0 * accuracy_interval['difference_candidate_minus_reference']:.2f} points**
  (95% input-cluster bootstrap interval
  {100.0 * accuracy_interval['percentile_interval_lower']:.2f} to
  {100.0 * accuracy_interval['percentile_interval_upper']:.2f}).
- The calibrated three-LoRA ensemble produced the highest observed
  error-detection AUROC (**{dec(primary['three_lora_ensemble_calibrated']['error_detection_auroc'])}**).
- The Laplace-approximated linear head produced the lowest aggregate
  10-bin ECE (**{dec(primary['laplace_head']['ece_10_bin'])}**), but not the
  best accuracy or error ranking; low ECE alone is not evidence of superior
  reliability.
- Expressed-confidence parsing remained valid for
  {pct(inputs['results_summary']['expressed_overall']['baseline']['valid_rate'])}
  of baseline rows, but only {pct(inputs['results_summary']['expressed_overall']['full_seed_1']['valid_rate'])}
  to {pct(inputs['results_summary']['expressed_overall']['full_seed_2']['valid_rate'])}
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

{_overall_table(inputs)}

### Three-seed LoRA summary

{_lora_seed_summary_table(inputs)}

### Paired uncertainty and direct UQ diagnostics

{_uq_signal_table(inputs)}

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
"""


def render_technical_report(inputs: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    summary = inputs["results_summary"]
    primary = summary["primary_overall"]
    paired = _paired_index(inputs)
    baseline = primary["baseline_calibrated"]
    ensemble = primary["three_lora_ensemble_calibrated"]
    laplace = primary["laplace_head"]
    seed_summary = next(row for row in inputs["lora_seed_summary"] if row["scope"] == "overall")
    seed_accuracy = seed_summary["metrics"]["accuracy"]
    seed_comparison = next(
        row for row in inputs["bootstrap_differences"]
        if row["comparison"] == "lora_seed_mean_vs_baseline_calibrated" and row["scope"] == "overall"
    )
    seed_accuracy_interval = seed_comparison["metrics"]["accuracy"]
    method = config["method_contract"]
    data = inputs["data_summary"]
    no_passage = paired[("three_lora_ensemble_calibrated", "no_passage")]
    return f"""# {config['release']['title']}

**Technical report — version {config['release']['version']} ({config['release']['release_date']})**

## Abstract

This study evaluates whether confidence estimates from a compact
instruction-tuned language model respond appropriately when evidence is
removed, truncated, distracted, contradicted, or withheld. A frozen
Qwen2.5-1.5B-Instruct baseline was compared with three independently seeded
LoRA adapters, temperature-scaled probabilities, a three-adapter ensemble,
and a Laplace-approximated Bayesian binary linear head over frozen model
representations. Experiments used deterministic 800/200/400 BoolQ
train/calibration/test subsets and 2,400 condition-level test rows. Across
the three adapters, calibrated overall accuracy was {pct(seed_accuracy['mean'])}
mean with a {100.0 * seed_accuracy['sample_standard_deviation']:.2f}-point
sample standard deviation and a {pct(seed_accuracy['minimum'])}–{pct(seed_accuracy['maximum'])}
range. The mean improvement over the {pct(baseline['accuracy'])} calibrated
baseline was {100.0 * seed_accuracy_interval['difference_candidate_minus_reference']:.2f}
points (95% paired input-cluster bootstrap interval
{100.0 * seed_accuracy_interval['percentile_interval_lower']:.2f} to
{100.0 * seed_accuracy_interval['percentile_interval_upper']:.2f}). The ensemble gave
the strongest observed error-detection AUROC ({dec(ensemble['error_detection_auroc'])}),
whereas the Laplace head gave the lowest aggregate 10-bin ECE
({dec(laplace['ece_10_bin'])}) without leading on accuracy or error ranking.
Evidence removal generally reduced confidence and raised entropy, but the
response depended on the method and perturbation. Generated numerical
confidence was frequently malformed after LoRA adaptation, limiting direct
expressed-versus-statistical confidence comparisons. Results are descriptive
for one model, one bounded dataset sample, and predefined perturbations.

## 1. Research question

How do expressed confidence, token-derived statistical confidence,
calibrated confidence, ensemble uncertainty, and approximate Bayesian-head
uncertainty behave when an LLM receives incomplete, distracting,
contradictory, or absent supporting evidence?

## 2. Connection to Evidence-State Reliability

The project extends the separate Evidence-State Reliability study from
pipeline-level evidence handling to model-level confidence. The earlier
work motivated a narrower question: a system may become easier to parse
without becoming more epistemically reliable. Here, generated confidence
format validity and probability-based uncertainty are measured separately,
so syntactic compliance cannot substitute for uncertainty quality.

## 3. Dataset and evidence conditions

The source dataset is `google/boolq` at revision
`{data['dataset_revision']}`. Deterministic
SHA-256-stratified selection with seed {data['selection_seed']} produced
{data['selected_sizes']['train']} training, {data['selected_sizes']['calibration']}
calibration, and {data['selected_sizes']['test']} test inputs. The official
BoolQ training split supplies the disjoint training and calibration subsets;
the official validation split supplies the test subset. Raw passages are not
committed.

Six categorical conditions were fixed before evaluation:

1. Original evidence.
2. Removal of the sentence span with greatest unique-token Jaccard overlap
   with the question.
3. Retention of the first half of passage tokens, rounded upward.
4. Insertion of a fragment from another selected input with minimum lexical
   overlap under a deterministic hash tie-break.
5. Insertion of a templated negation of a high-overlap sentence fragment.
6. The question without a passage.

All 400 inputs were transformed successfully under all conditions. These
conditions are not an ordered degradation scale. Lexical evidence removal
is a proxy, minimum lexical overlap does not prove semantic irrelevance, and
the templated statement is not guaranteed to contradict the labelled answer.

## 4. Model and fine-tuning method

The base model is `Qwen/Qwen2.5-1.5B-Instruct` at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Contextual continuations
` Yes` and ` No` were verified as single tokens with IDs 7414 and 2308.

LoRA adapters were attached to `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj` in each transformer layer. Rank was
{method['lora_rank']}, alpha {method['lora_alpha']}, and dropout
{method['lora_dropout']}. The adapters contained
{method['trainable_parameters']:,} trainable parameters
({100.0 * method['trainable_parameters'] / method['total_parameters_with_adapter']:.3f}%
of {method['total_parameters_with_adapter']:,} total
parameters). Training used an explicit PyTorch loop, answer-token
cross-entropy, AdamW, gradient accumulation to an effective batch size of
{method['effective_batch_size']}, gradient clipping at
{method['gradient_clip']}, learning rate
{method['learning_rate']}, and three epochs. No numerical
confidence target was used.

{_training_table(inputs)}

All losses remained finite, all trainable parameters received gradients,
parameters changed, and each saved adapter reproduced validation logits
after reload. Declining training loss is an optimization diagnostic, not a
generalization claim.

## 5. Confidence definitions

- **Expressed confidence:** the parsed integer from the generated
  `Confidence: N` line, retained only when format validation succeeds.
- **Token-derived confidence:** the two-class softmax probability of the
  selected contextual Yes/No answer token.
- **Calibrated confidence:** token probability after division of logits by
  a positive temperature fitted on calibration data.
- **Predictive entropy:** binary entropy of the predictive mean.
- **Ensemble uncertainty:** variation and entropy decomposition across three
  independently seeded LoRA members.
- **Bayesian-head uncertainty:** posterior predictive variation from sampled
  weights of a diagonal Laplace approximation to a binary linear head.

Generated confidence is not treated as a statistical probability, and
malformed values are not assigned a fallback.

## 6. Calibration method

One positive scalar temperature per baseline/adapter method was fitted by
minimizing calibration-split negative log-likelihood on original evidence.
Class ordering remained `[Yes, No]`; test labels were unavailable during
fitting; and temperature scaling did not alter predicted classes.

{_temperature_table(inputs)}

Calibration-split improvement does not guarantee improvement for every test
condition. ECE is bin-dependent and can appear favourable for an inaccurate
or underconfident model.

## 7. Ensemble method

The ensemble averages binary probabilities from LoRA adapters trained with
seeds 20260811, 20260812, and 20260813. Member weights, initializations, and
prediction files were verified to differ. Predictive entropy, expected
member entropy, population probability variance, vote disagreement, and the
finite-ensemble entropy Jensen gap were recorded. The members are not
posterior samples, so the Jensen gap is described as mutual-information-style
rather than exact Bayesian mutual information.

## 8. Bayesian prediction-head method

The transformer was frozen and supplied 1,536-dimensional final hidden
states at the last attended answer-prefix token. A binary logistic linear
head was fitted by MAP estimation with a zero-mean isotropic Gaussian prior
of precision 1.0. The diagonal of the exact logistic negative-log-posterior
curvature was inverted to form a local diagonal Gaussian approximation.
The full evaluation used 256 sampled head-weight vectors to estimate
posterior predictive means, variances, expected entropy, and mutual
information.

This is a **Laplace-approximated Bayesian prediction head over frozen LLM
representations**. It does not make the transformer Bayesian. The diagonal
approximation ignores parameter correlations and is local to the MAP mode.

## 9. Experimental protocol

Labels were unavailable to baseline, adapter, ensemble, and Laplace test
inference. Calibration used only 200 calibration labels under original
evidence. Test labels were joined only after predictions were frozen. Each
method was evaluated on the same 400 inputs under six conditions. Source
coordinates, transformations, predictions, metrics, configurations,
checkpoints, and figures are bound by SHA-256 ledgers. Failed runs remain in
the manifest history.

## 10. Metrics

- **Accuracy** measures discrete answer correctness but ignores confidence.
- **Negative log-likelihood (NLL)** scores the probability assigned to the
  observed class and strongly penalizes confident errors.
- **Brier score** is squared error of the Yes probability; it combines
  calibration and discrimination.
- **10-bin ECE** compares mean confidence and empirical accuracy within
  fixed bins; it is sensitive to binning and sample size.
- **Predictive entropy** measures uncertainty of a binary predictive mean.
- **Error-detection AUROC** measures whether `1-confidence` ranks errors
  above correct predictions; it does not select an operating threshold.
- **Error-detection AUPRC** summarizes precision–recall ranking with errors
  as the positive class and must be interpreted against the error prevalence.
- **UQ-signal discrimination** evaluates predictive entropy, ensemble
  member variance and MI-style disagreement, and Laplace predictive variance
  and mutual information for both error ranking and original-versus-degraded
  evidence ranking.
- **Risk–coverage** orders examples by confidence and reports selective risk
  as lower-confidence predictions are withheld.
- **Paired confidence/entropy change** compares each degraded input with its
  original counterpart.
- **Expressed/token divergence** is absolute difference on parser-valid rows
  only; coverage must be reported beside it.

## 11. Results

### 11.1 Overall comparison

{_overall_table(inputs)}

The three calibrated LoRA seeds are summarized together rather than selecting
the numerically strongest seed:

{_lora_seed_summary_table(inputs)}

All reported differences below are candidate minus reference. Intervals are
95% paired percentile intervals from 2,000 bootstrap samples of the 400 input
IDs; every sampled input carries all six evidence conditions.

{_bootstrap_table(inputs)}

The calibrated ensemble had {pct(ensemble['accuracy'])} accuracy and the strongest
error-detection AUROC ({dec(ensemble['error_detection_auroc'])}). The
Laplace head's low ECE ({dec(laplace['ece_10_bin'])}) coexisted with
{pct(laplace['accuracy'])} accuracy and AUROC
{dec(laplace['error_detection_auroc'])}; this illustrates why calibration
error cannot be interpreted alone.

![Overall method comparison](../outputs/results/gate8/figures/08_method_comparison.png)

### 11.2 Evidence-condition accuracy

{_condition_table(inputs)}

Original-evidence accuracy was highest for the calibrated three-LoRA
ensemble ({pct(_metric_index(inputs)[('three_lora_ensemble_calibrated', 'original')]['accuracy'])}).
For that ensemble, removing all passage evidence reduced accuracy to
{pct(_metric_index(inputs)[('three_lora_ensemble_calibrated', 'no_passage')]['accuracy'])},
mean confidence by {signed(no_passage['mean_confidence_change_from_original'])},
and increased predictive entropy by
{signed(no_passage['mean_predictive_entropy_change_from_original'])} nats.
Lexical contradiction was also damaging, but results do not define a
monotonic perturbation severity order.

![Accuracy by condition](../outputs/results/gate8/figures/03_accuracy_by_condition.png)

### 11.3 Expressed-confidence alignment

{_expressed_table(inputs)}

The adapters often produced the correct answer token while failing the
required two-line generated format. Consequently, divergence means for
adapters describe a selected parser-valid subset and must not be generalized
to all rows. The absence of imputation is deliberate.

![Expressed versus token confidence](../outputs/results/gate8/figures/05_expressed_vs_token_confidence.png)

### 11.4 Selective prediction and uncertainty shifts

The saved risk–coverage table contains one deterministic ordering for every
primary variant, with ties broken by analysis ID. Confidence and entropy
heatmaps show that the no-passage condition generally produces the largest
confidence reductions and entropy increases for token-derived methods. The
Laplace head changes less across some perturbations, which should not be read
as automatically better shift detection because its aggregate error AUROC
is lower.

![Risk–coverage](../outputs/results/gate8/figures/02_risk_coverage.png)

### 11.5 Direct evaluation of uncertainty signals

{_uq_signal_table(inputs)}

For binary predictions, predictive entropy is monotone in `1-confidence`,
so it gives the same AUROC ordering as the earlier confidence-derived error
score. Ensemble member-probability variance and MI-style disagreement, and
Laplace posterior-predictive variance and mutual information, were weaker
error rankers in this experiment. Original-versus-any-degraded AUROCs were
modest rather than decisive. The any-degraded AUPRC rows have a 5/6 positive
prevalence by construction, so their high numerical values must be compared
with that 0.8333 prevalence baseline. Per-condition diagnostics are retained
in `outputs/results/statistical_analysis/uq_degradation_detection.jsonl`.

## 12. Failure analysis

The strongest operational failure was generated-format instability after
answer-token-only LoRA training. This is consistent with the training
objective: it directly supervises only the answer token and does not train
the numerical confidence line. The lexical-removal procedure removed the
whole passage for 34 inputs, while three selected spans had zero lexical
overlap. Distractor fragments were shorter than requested for 205 inputs;
contradiction fragments were shorter for 386. These flags are retained in
the transformation metadata instead of being hidden.

## 13. Limitations

1. Results use one 1.5B-parameter model and a bounded 400-input test subset.
2. Bootstrap intervals are post-hoc descriptive intervals clustered by the
   400 selected inputs; they are not preregistered hypothesis tests and do
   not capture model-family or dataset-sampling uncertainty.
3. The perturbations use lexical proxies and do not establish semantic
   irrelevance, answer contradiction, or ordered severity.
4. Only three LoRA members were trained; ensemble estimates are coarse and
   are not Bayesian posterior quantities.
5. The Laplace approximation covers only a linear head, uses diagonal
   curvature, and depends on frozen representation quality.
6. ECE depends on ten fixed bins and can obscure within-bin behaviour.
7. Expressed-confidence analysis has severe method-dependent missingness.
8. Deterministic settings improve within-environment reproducibility but do
   not guarantee bitwise identity across hardware or library versions.
9. Test results were inspected only after protocols were frozen; they are
   not a basis for retrospective method selection or tuning.

## 14. Reproducibility statement

The repository records exact model and dataset revisions, source hashes,
split IDs, configurations, package versions, seeds, hardware, checkpoint
hashes, prediction hashes, success/failure manifests, derived tables, and
figure-source mappings. The full runs used Python 3.12.13, PyTorch
2.11.0+cu128, Transformers 5.13.1, PEFT 0.19.1, and an NVIDIA
A100-SXM4-80GB. Raw passages and model weights are excluded. Historical run
manifests have `git_head: null` because experiments preceded the first
repository commit; immutable input/output hashes provide the execution
lineage, while the release commit identifies the published code snapshot.
The Colab setup contract, preserved successful CPU test log, and GitHub
Actions CPU workflow make environment reconstruction and contract testing
explicit. See `report/reproducibility.md`.

## 15. Claim boundaries

Supported claims include deterministic data preparation, explicit PyTorch
LoRA training, probability extraction, calibration-only temperature fitting,
three-member ensemble analysis, and a diagonal Laplace approximation for a
binary linear head. Unsupported claims include full-model training, a fully
Bayesian transformer, posterior-sample interpretation of LoRA members,
semantic guarantees for lexical perturbations, statistical significance,
state-of-the-art performance, and peer review.

## 16. Future work

Future work should repeat the protocol across model families and larger
independent samples, introduce semantically validated perturbations,
supervise or separately model expressed confidence, compare richer covariance
approximations, and predefine formal statistical tests before collecting new
test results.

## 17. References

1. Clark et al. (2019), *BoolQ: Exploring the Surprising Difficulty of
   Natural Yes/No Questions*, NAACL.
2. Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*,
   ICLR.
3. Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML.
4. Lakshminarayanan, Pritzel, and Blundell (2017), *Simple and Scalable
   Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS.
5. Daxberger et al. (2021), *Laplace Redux—Effortless Bayesian Deep
   Learning*, NeurIPS.
6. Qwen Team, `Qwen/Qwen2.5-1.5B-Instruct` model card, pinned revision used
   in this repository.
"""


def render_claim_boundaries(inputs: Mapping[str, Any]) -> str:
    primary = inputs["results_summary"]["primary_overall"]
    return f"""# Claim boundaries

## Supported by verified artifacts

- Deterministic, disjoint BoolQ subsets contain 800 training, 200
  calibration, and 400 test inputs.
- Six label-blind evidence conditions were generated for every test input.
- Qwen2.5-1.5B-Instruct baseline inference completed for 2,400 rows.
- Three independently seeded LoRA adapters completed explicit PyTorch
  training; losses were finite, gradients reached all trainable parameters,
  weights changed, and checkpoints reloaded within tolerance.
- Temperature scaling used only calibration labels and preserved predicted
  classes.
- A three-adapter ensemble and a Laplace-approximated Bayesian binary linear
  head over frozen representations completed evaluation.
- Saved predictions produced traceable metrics, risk–coverage tables, paired
  changes, expressed-confidence summaries, and eight verified figures.
- The highest observed overall accuracy was
  {pct(primary['full_seed_2_calibrated']['accuracy'])} for calibrated LoRA
  seed 2; this is a descriptive result for the fixed study protocol.

## Required qualifications

- LoRA is parameter-efficient adaptation, not full-model training.
- Only the linear prediction head is Laplace approximated; the transformer
  is not Bayesian.
- LoRA ensemble members are independently seeded models, not posterior
  samples. Their entropy Jensen gap is mutual-information-style.
- Generated numerical confidence is verbalized output, not a statistical
  probability.
- Expressed-confidence divergence is reported only for parser-valid rows;
  invalid rows remain missing.
- Lexical evidence removal is a deterministic overlap proxy, minimum lexical
  overlap does not prove irrelevance, and templated negation does not prove
  contradiction of the labelled answer.
- Evidence conditions are categorical and must not be interpreted as a
  monotonic severity scale.
- Lower ECE alone does not establish better overall reliability.
- Test results are post-hoc descriptive evidence, not tuning or model
  selection data.

## Unsupported claims

- State-of-the-art or large-scale LLM training.
- A fully Bayesian LLM or exact Bayesian inference.
- Statistically significant improvements or population-level generalization.
- Robustness to arbitrary distribution shift or adversarial attacks.
- Semantically guaranteed evidence removal, distraction, or contradiction.
- Publication, acceptance, or peer review.

## Permitted description

This work may be described as a **reproducible application-stage empirical
study of confidence alignment and uncertainty quantification under controlled
evidence perturbations**, implemented with PyTorch, Transformers, PEFT,
temperature scaling, a three-member LoRA ensemble, and a
Laplace-approximated Bayesian prediction head over frozen LLM
representations.
"""


def render_reproducibility(inputs: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    method = config["method_contract"]
    lines = [
        "# Reproducibility",
        "",
        "## Frozen identities",
        "",
        f"- Dataset: `google/boolq` at `{inputs['data_summary']['dataset_revision']}`.",
        "- Model/tokenizer: `Qwen/Qwen2.5-1.5B-Instruct` at "
        "`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.",
        f"- Split selection: `{inputs['data_summary']['selection_algorithm']}` with seed `{inputs['data_summary']['selection_seed']}`.",
        "- LoRA seeds: `20260811`, `20260812`, and `20260813`.",
        "- Laplace posterior sampling seed: `20260821`.",
        "",
        "## Environment",
        "",
        "The verified full runs used:",
        "",
        f"- Python `{method['python']}`",
        f"- PyTorch `{method['torch']}`",
        f"- Transformers `{method['transformers']}`",
        f"- PEFT `{method['peft']}`",
        f"- CUDA build `{method['cuda_build']}`",
        f"- GPU `{method['gpu']}` with {method['gpu_vram_gib']:.2f} GiB reported VRAM",
        "- `torchao` excluded after an incompatible optional installation was diagnosed.",
        "",
        "Pinned direct Python dependencies are listed in `requirements.txt` and `pyproject.toml`. A fresh GPU Colab runtime can be reconstructed and checked with `bash colab/setup.sh`; `colab/verify_environment.py` fails closed on package, CUDA, or TorchAO drift.",
        "",
        "CPU-compatible unit and contract tests run in `.github/workflows/cpu-tests.yml`. The successful local output is preserved under `artifacts/test-results/` with its SHA-256 digest.",
        "",
        "## Data isolation",
        "",
        "- Training uses only the frozen 800-example training manifest.",
        "- Temperature fitting uses only 200 calibration examples under original evidence.",
        "- The 400 test inputs originate from the official BoolQ validation split.",
        "- Test labels are joined only after inference and are never used for fitting or tuning.",
        "- Raw BoolQ passages are reconstructed from pinned source coordinates and are not committed.",
        "",
        "## Artifact integrity",
        "",
        "Each completed stage publishes payload hashes and a completion marker last. Existing differing artifacts are rejected. Failed-run manifests are preserved. Gate 8 binds each figure to its machine-readable source table and records that no manual result values were used. The statistical-analysis stage adds 2,000 paired bootstrap samples clustered by all 400 input IDs and direct UQ-signal ranking tables without rerunning model inference.",
        "",
        "Historical experiment manifests contain `git_head: null` because the runs occurred before the repository's first commit. Their exact configuration, input, checkpoint, prediction, and output hashes remain recorded. The eventual release commit identifies the published source snapshot but does not retroactively change those manifests.",
        "",
        "## Full command order",
        "",
        "Use the commands in the root README in order, retaining smoke gates before full runs. GPU stages are baseline inference, LoRA training/inference, calibration inference, and frozen-representation extraction. Temperature fitting, evaluation, ensemble aggregation, clustered statistical analysis, result construction, and report generation support CPU execution.",
        "",
        "## Files intentionally excluded",
        "",
        "- Hugging Face caches and reconstructed raw data.",
        "- LoRA adapter/model checkpoint payloads under `outputs/checkpoints/`.",
        "- Secrets, `.env` files, notebook state, and temporary files.",
        "",
        "Checkpoint and prediction identities remain available through SHA-256 values in configurations, ledgers, and run manifests.",
        "",
        "## Documentation lineage",
        "",
        "The following verified inputs generate the README, report, citation, data note, and claim boundaries:",
        "",
        "| Input | SHA-256 |",
        "|---|---|",
    ]
    for specification in config["inputs"].values():
        lines.append(f"| `{specification['path']}` | `{specification['sha256']}` |")
    return "\n".join(lines) + "\n"


def render_citation(config: Mapping[str, Any]) -> str:
    release = config["release"]
    author = release["author"]
    return f"""cff-version: 1.2.0
message: "If you use this research artefact, please cite it using the metadata below."
title: "{release['title']}"
type: software
version: "{release['version']}"
date-released: "{release['release_date']}"
repository-code: "{release['repository']}"
license: MIT
authors:
  - family-names: "{author['family_names']}"
    given-names: "{author['given_names']}"
    alias: "{author['alias']}"
preferred-citation:
  type: report
  title: "{release['title']}"
  year: 2026
  authors:
    - family-names: "{author['family_names']}"
      given-names: "{author['given_names']}"
      alias: "{author['alias']}"
"""


def render_data_readme(inputs: Mapping[str, Any]) -> str:
    data = inputs["data_summary"]
    counts = data["selected_class_counts"]
    return f"""# Data reconstruction

Raw BoolQ records are intentionally excluded from version control.

## Frozen source

- Dataset: `google/boolq`
- Revision: `{data['dataset_revision']}`
- Licence: CC BY-SA 3.0
- Selection: `{data['selection_algorithm']}`
- Seed: `{data['selection_seed']}`
- Prompt-token eligibility ceiling: {data['eligibility_max_prompt_tokens']}

## Completed deterministic allocation

| Research split | Official source | Rows | False | True |
|---|---|---:|---:|---:|
| Training | train | {data['selected_sizes']['train']} | {counts['train']['false']} | {counts['train']['true']} |
| Calibration | train, disjoint from training | {data['selected_sizes']['calibration']} | {counts['calibration']['false']} | {counts['calibration']['true']} |
| Test | validation | {data['selected_sizes']['test']} | {counts['test']['false']} | {counts['test']['true']} |

Manifests contain source coordinates, labels, prompt-token lengths, and
cryptographic identities, but no raw questions or passages. Reconstruction
refuses dataset-revision or source-hash drift.

Test labels were unavailable during model inference and were not used for
temperature fitting, threshold selection, prompt selection, or
hyperparameter tuning. Evidence transformations were label blind.
"""


def render_requirements(config: Mapping[str, Any]) -> str:
    dependencies = list(config["method_contract"]["dependencies"])
    require(len(dependencies) >= 10, "requirements dependency loss")
    return (
        "# Direct dependencies pinned for the verified Colab experiment environment.\n"
        "# The observed PyTorch build was 2.11.0+cu128 on an NVIDIA A100-SXM4-80GB.\n"
        + "\n".join(dependencies)
        + "\n"
    )


def build_documents(inputs: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, bytes]:
    documents = {
        "README.md": render_readme(inputs, config),
        "report/technical_report.md": render_technical_report(inputs, config),
        "report/claim_boundaries.md": render_claim_boundaries(inputs),
        "report/reproducibility.md": render_reproducibility(inputs, config),
        "CITATION.cff": render_citation(config),
        "data/README.md": render_data_readme(inputs),
        "requirements.txt": render_requirements(config),
    }
    require(set(documents) == set(config["outputs"]), "documentation output set drift")
    outputs: dict[str, bytes] = {}
    for name, text in documents.items():
        require(text.endswith("\n"), f"documentation lacks final newline: {name}")
        require("\r" not in text, f"documentation contains CR line endings: {name}")
        outputs[name] = text.encode("utf-8")
    return outputs


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_documents(
    repo: Path,
    config: Mapping[str, Any],
    documents: Mapping[str, bytes],
    *,
    allow_initial_update: bool,
) -> dict[str, Any]:
    statuses: dict[str, str] = {}
    for relative_name, payload in documents.items():
        path = repo / relative_name
        expected_hash = sha256_bytes(payload)
        if path.exists() and path.read_bytes() == payload:
            statuses[relative_name] = "verified_existing"
            continue
        initial_hash = config["outputs"][relative_name]["initial_sha256"]
        if not allow_initial_update:
            raise ReportingError(f"refusing differing documentation output: {relative_name}")
        if path.exists():
            require(initial_hash is not None, f"unexpected existing output: {relative_name}")
            require(sha256_bytes(path.read_bytes()) == initial_hash, f"unapproved existing output: {relative_name}")
            status = "updated"
        else:
            require(initial_hash is None, f"expected pre-existing output is missing: {relative_name}")
            status = "created"
        _atomic_write(path, payload)
        require(sha256_bytes(path.read_bytes()) == expected_hash, f"post-write hash mismatch: {relative_name}")
        statuses[relative_name] = status

    ledger = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(documents.items())
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "input_sha256": {name: value["sha256"] for name, value in config["inputs"].items()},
        "document_sha256": {name: value["sha256"] for name, value in ledger.items()},
        "manual_result_values_used": False,
        "test_labels_used_for_fitting_or_tuning": False,
        "full_transformer_is_bayesian": False,
        "peer_reviewed": False,
    }
    release_directory = repo / config["publication"]["output_directory"]
    release_payloads = {
        "artifact_hashes.json": canonical_bytes(ledger),
        "provenance.json": canonical_bytes(provenance),
    }
    for name, payload in release_payloads.items():
        path = release_directory / name
        if path.exists():
            require(path.read_bytes() == payload, f"release ledger drift: {name}")
            statuses[f"{config['publication']['output_directory']}/{name}"] = "verified_existing"
        else:
            _atomic_write(path, payload)
            statuses[f"{config['publication']['output_directory']}/{name}"] = "created"
    completion = canonical_bytes({
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "complete": True,
        "artifact_hashes_sha256": sha256_bytes(release_payloads["artifact_hashes.json"]),
    })
    completion_path = release_directory / "COMPLETE.json"
    if completion_path.exists():
        require(completion_path.read_bytes() == completion, "release completion marker drift")
        statuses[f"{config['publication']['output_directory']}/COMPLETE.json"] = "verified_existing"
    else:
        _atomic_write(completion_path, completion)
        statuses[f"{config['publication']['output_directory']}/COMPLETE.json"] = "created"
    return {"statuses": statuses, "ledger": ledger, "provenance": provenance}


def preflight(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(repo / config_path)
    inputs = load_inputs(repo, config)
    documents = build_documents(inputs, config)
    return {
        "documents": len(documents),
        "document_sha256": {name: sha256_bytes(payload) for name, payload in sorted(documents.items())},
        "reported_variants": len(inputs["results_summary"]["all_variants"]),
        "primary_variants": len(inputs["results_summary"]["primary_variants"]),
        "figures": len(inputs["figure_manifest"]["figures"]),
        "files_modified": False,
    }


def run(repo: Path, config_path: Path, *, allow_initial_update: bool = False) -> dict[str, Any]:
    config = load_config(repo / config_path)
    inputs = load_inputs(repo, config)
    documents = build_documents(inputs, config)
    published = publish_documents(repo, config, documents, allow_initial_update=allow_initial_update)
    return {
        "documents": len(documents),
        "artifact_statuses": published["statuses"],
        "artifact_hashes": published["ledger"],
        "output_directory": config["publication"]["output_directory"],
        "manual_result_values_used": False,
        "test_labels_used_for_fitting_or_tuning": False,
        "full_transformer_is_bayesian": False,
        "peer_reviewed": False,
    }
