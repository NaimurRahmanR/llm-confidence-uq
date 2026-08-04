"""Unified, label-audited analysis over verified BoolQ result artifacts."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-unified-results-v1"
SOURCE_EVALUATION_PROTOCOL = "boolq-calibrated-test-evaluation-v1"
SOURCE_ENSEMBLE_PROTOCOL = "boolq-three-lora-ensemble-v1"
SOURCE_LAPLACE_PROTOCOL = "boolq-frozen-qwen-diagonal-laplace-head-v1"
METHODS = ("baseline", "full_seed_1", "full_seed_2", "full_seed_3")
CONDITIONS = (
    "original",
    "lexical_evidence_removal",
    "prefix_truncation_50",
    "irrelevant_distractor",
    "lexical_contradiction",
    "no_passage",
)
ALL_VARIANTS = (
    "baseline_raw",
    "baseline_calibrated",
    "full_seed_1_raw",
    "full_seed_1_calibrated",
    "full_seed_2_raw",
    "full_seed_2_calibrated",
    "full_seed_3_raw",
    "full_seed_3_calibrated",
    "three_lora_ensemble_raw",
    "three_lora_ensemble_calibrated",
    "laplace_head",
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
DISPLAY_LABELS = {
    "baseline_raw": "Baseline raw",
    "baseline_calibrated": "Baseline calibrated",
    "full_seed_1_raw": "LoRA seed 1 raw",
    "full_seed_1_calibrated": "LoRA seed 1 calibrated",
    "full_seed_2_raw": "LoRA seed 2 raw",
    "full_seed_2_calibrated": "LoRA seed 2 calibrated",
    "full_seed_3_raw": "LoRA seed 3 raw",
    "full_seed_3_calibrated": "LoRA seed 3 calibrated",
    "three_lora_ensemble_raw": "Three-LoRA ensemble raw",
    "three_lora_ensemble_calibrated": "Three-LoRA ensemble calibrated",
    "laplace_head": "Laplace head",
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


class AnalysisError(RuntimeError):
    """Raised when source artifacts or derived analysis violate the protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _finite_probability(value: Any, name: str) -> float:
    number = float(value)
    require(math.isfinite(number) and 0.0 <= number <= 1.0, f"invalid {name}")
    return number


def entropy(p_yes: float) -> float:
    p_yes = _finite_probability(p_yes, "entropy probability")
    p_no = 1.0 - p_yes
    result = 0.0
    for probability in (p_yes, p_no):
        if probability > 0.0:
            result -= probability * math.log(probability)
    return result


def _normalized_row(
    *,
    variant: str,
    input_id: str,
    source_index: int,
    condition: str,
    condition_index: int,
    ground_truth: str,
    p_yes: float,
    predictive_entropy: float,
    source_row_sha256: str,
    expected_entropy: float | None = None,
    mutual_information: float | None = None,
) -> dict[str, Any]:
    require(variant in ALL_VARIANTS, "unknown normalized variant")
    require(condition in CONDITIONS and condition_index == CONDITIONS.index(condition), "condition drift")
    require(ground_truth in ("Yes", "No"), "invalid ground truth")
    p_yes = _finite_probability(p_yes, "normalized Yes probability")
    require(isinstance(input_id, str) and input_id, "invalid input ID")
    require(type(source_index) is int, "invalid source index")
    require(isinstance(source_row_sha256, str) and len(source_row_sha256) == 64, "invalid source-row hash")
    prediction = "Yes" if p_yes >= 0.5 else "No"
    confidence = max(p_yes, 1.0 - p_yes)
    predictive_entropy = float(predictive_entropy)
    require(math.isfinite(predictive_entropy) and predictive_entropy >= 0.0, "invalid predictive entropy")
    if expected_entropy is not None:
        expected_entropy = float(expected_entropy)
        require(math.isfinite(expected_entropy) and expected_entropy >= 0.0, "invalid expected entropy")
        require(predictive_entropy + 1e-12 >= expected_entropy, "expected entropy exceeds predictive entropy")
    if mutual_information is not None:
        mutual_information = float(mutual_information)
        require(math.isfinite(mutual_information) and mutual_information >= 0.0, "invalid mutual information")
        require(expected_entropy is not None, "mutual information lacks expected entropy")
        require(abs(mutual_information - max(0.0, predictive_entropy - expected_entropy)) <= 1e-10, "entropy decomposition drift")
    identity = {
        "variant": variant,
        "input_id": input_id,
        "source_index": source_index,
        "condition": condition,
        "source_row_sha256": source_row_sha256,
        "protocol_version": PROTOCOL_VERSION,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "analysis_id": "boolqanalysis-" + sha256_text(canonical_json(identity)),
        "variant": variant,
        "display_label": DISPLAY_LABELS[variant],
        "input_id": input_id,
        "source_index": source_index,
        "condition": condition,
        "condition_index": condition_index,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "correct": prediction == ground_truth,
        "p_yes": p_yes,
        "p_no": 1.0 - p_yes,
        "confidence": confidence,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "source_row_sha256": source_row_sha256,
    }


def _validate_source_hash(row: Mapping[str, Any]) -> None:
    unhashed = dict(row)
    observed = unhashed.pop("row_sha256", None)
    require(isinstance(observed, str) and len(observed) == 64, "missing source row hash")
    require(sha256_text(canonical_json(unhashed)) == observed, "source row hash drift")


def normalize_evaluation_rows(method: str, rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(method in METHODS and len(rows) == 2400, "evaluation input contract drift")
    normalized: list[dict[str, Any]] = []
    expressed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        require(row.get("protocol_version") == SOURCE_EVALUATION_PROTOCOL, "evaluation protocol drift")
        require(row.get("method") == method, "evaluation method drift")
        _validate_source_hash(row)
        condition = str(row["condition"])
        condition_index = int(row["condition_index"])
        key = (str(row["input_id"]), condition)
        require(key not in seen, "duplicate evaluation key")
        seen.add(key)
        raw_p_yes = _finite_probability(row["raw_p_yes"], "raw Yes probability")
        calibrated_p_yes = _finite_probability(row["calibrated_p_yes"], "calibrated Yes probability")
        for family, p_yes in (("raw", raw_p_yes), ("calibrated", calibrated_p_yes)):
            variant = f"{method}_{family}"
            normalized.append(_normalized_row(
                variant=variant,
                input_id=str(row["input_id"]),
                source_index=int(row["source_index"]),
                condition=condition,
                condition_index=condition_index,
                ground_truth=str(row["ground_truth"]),
                p_yes=p_yes,
                predictive_entropy=float(row[f"{family}_entropy"]),
                source_row_sha256=str(row["row_sha256"]),
            ))
        valid = bool(row["expressed_parser_valid"])
        value = row["expressed_confidence"]
        require((valid and type(value) is int and 0 <= value <= 100) or (not valid and value is None), "expressed confidence policy drift")
        expressed.append({
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "method": method,
            "input_id": str(row["input_id"]),
            "source_index": int(row["source_index"]),
            "condition": condition,
            "condition_index": condition_index,
            "ground_truth": str(row["ground_truth"]),
            "correct": bool(row["correct"]),
            "parser_valid": valid,
            "parser_reason_code": str(row["expressed_parser_reason_code"]),
            "expressed_confidence": value / 100.0 if valid else None,
            "raw_token_confidence": float(row["raw_confidence"]),
            "calibrated_token_confidence": float(row["calibrated_confidence"]),
            "raw_absolute_divergence": float(row["expressed_raw_absolute_divergence"]) if valid else None,
            "calibrated_absolute_divergence": float(row["expressed_calibrated_absolute_divergence"]) if valid else None,
            "source_row_sha256": str(row["row_sha256"]),
        })
    require(len(seen) == 2400, "evaluation key cardinality drift")
    return normalized, expressed


def normalize_ensemble_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(rows) == 2400, "ensemble row cardinality drift")
    outputs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        require(row.get("protocol_version") == SOURCE_ENSEMBLE_PROTOCOL, "ensemble protocol drift")
        _validate_source_hash(row)
        key = (str(row["input_id"]), str(row["condition"]))
        require(key not in seen, "duplicate ensemble key")
        seen.add(key)
        require(len(row["members"]) == 3, "ensemble member cardinality drift")
        for family in ("raw", "calibrated"):
            outputs.append(_normalized_row(
                variant=f"three_lora_ensemble_{family}",
                input_id=str(row["input_id"]),
                source_index=int(row["source_index"]),
                condition=str(row["condition"]),
                condition_index=int(row["condition_index"]),
                ground_truth=str(row["ground_truth"]),
                p_yes=float(row[f"{family}_ensemble_p_yes"]),
                predictive_entropy=float(row[f"{family}_predictive_entropy"]),
                expected_entropy=float(row[f"{family}_expected_entropy"]),
                mutual_information=float(row[f"{family}_mutual_information_style"]),
                source_row_sha256=str(row["row_sha256"]),
            ))
    require(len(seen) == 2400, "ensemble key cardinality drift")
    return outputs


def normalize_laplace_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(rows) == 2400, "Laplace row cardinality drift")
    outputs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        require(row.get("protocol_version") == SOURCE_LAPLACE_PROTOCOL, "Laplace protocol drift")
        require(row.get("full_transformer_is_bayesian") is False, "Bayesian transformer overclaim")
        _validate_source_hash(row)
        key = (str(row["example_id"]), str(row["condition"]))
        require(key not in seen, "duplicate Laplace key")
        seen.add(key)
        outputs.append(_normalized_row(
            variant="laplace_head",
            input_id=str(row["example_id"]),
            source_index=int(row["source_index"]),
            condition=str(row["condition"]),
            condition_index=int(row["condition_index"]),
            ground_truth=str(row["ground_truth"]),
            p_yes=float(row["posterior_mean_p_yes"]),
            predictive_entropy=float(row["predictive_entropy"]),
            expected_entropy=float(row["expected_entropy"]),
            mutual_information=float(row["mutual_information"]),
            source_row_sha256=str(row["row_sha256"]),
        ))
    require(len(seen) == 2400, "Laplace key cardinality drift")
    return outputs


def build_normalized_rows(
    evaluation_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    ensemble_rows: Sequence[Mapping[str, Any]],
    laplace_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(tuple(evaluation_rows) == METHODS, "evaluation methods absent or out of order")
    normalized: list[dict[str, Any]] = []
    expressed: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows, method_expressed = normalize_evaluation_rows(method, evaluation_rows[method])
        normalized.extend(method_rows)
        expressed.extend(method_expressed)
    normalized.extend(normalize_ensemble_rows(ensemble_rows))
    normalized.extend(normalize_laplace_rows(laplace_rows))
    require(len(normalized) == len(ALL_VARIANTS) * 2400, "normalized row cardinality drift")
    require(len(expressed) == len(METHODS) * 2400, "expressed row cardinality drift")
    by_variant = {variant: [row for row in normalized if row["variant"] == variant] for variant in ALL_VARIANTS}
    anchor = {(row["input_id"], row["source_index"], row["condition"]): row["ground_truth"] for row in by_variant["baseline_raw"]}
    require(len(anchor) == 2400, "anchor alignment cardinality drift")
    for variant, rows in by_variant.items():
        projection = {(row["input_id"], row["source_index"], row["condition"]): row["ground_truth"] for row in rows}
        require(projection == anchor, f"cross-method alignment drift: {variant}")
    require(len({row["analysis_id"] for row in normalized}) == len(normalized), "duplicate analysis ID")
    return normalized, expressed


def reliability_bins(rows: Sequence[Mapping[str, Any]], bins: int = 10) -> list[dict[str, Any]]:
    require(rows and bins >= 2, "invalid reliability input")
    outputs = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [row for row in rows if lower < float(row["confidence"]) <= upper]
        outputs.append({
            "bin_index": index,
            "lower_exclusive": lower,
            "upper_inclusive": upper,
            "count": len(members),
            "mean_confidence": sum(float(row["confidence"]) for row in members) / len(members) if members else None,
            "accuracy": sum(bool(row["correct"]) for row in members) / len(members) if members else None,
        })
    require(sum(row["count"] for row in outputs) == len(rows), "reliability bins omit rows")
    return outputs


def error_detection_auroc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    require(rows, "empty AUROC input")
    values = sorted((1.0 - float(row["confidence"]), not bool(row["correct"])) for row in rows)
    positives = sum(label for _, label in values)
    negatives = len(values) - positives
    if positives == 0 or negatives == 0:
        return None
    positive_rank_sum = 0.0
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in values[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, nll_epsilon: float = 1e-12, bins: int = 10) -> dict[str, Any]:
    require(rows and 0.0 < nll_epsilon < 0.5, "invalid metric input")
    count = len(rows)
    nll = 0.0
    brier = 0.0
    for row in rows:
        target = 1.0 if row["ground_truth"] == "Yes" else 0.0
        p_yes = min(max(float(row["p_yes"]), nll_epsilon), 1.0 - nll_epsilon)
        nll -= target * math.log(p_yes) + (1.0 - target) * math.log(1.0 - p_yes)
        brier += (p_yes - target) ** 2
    reliability = reliability_bins(rows, bins)
    ece = sum(item["count"] / count * abs(item["accuracy"] - item["mean_confidence"]) for item in reliability if item["count"])
    expected = [float(row["expected_entropy"]) for row in rows if row["expected_entropy"] is not None]
    information = [float(row["mutual_information"]) for row in rows if row["mutual_information"] is not None]
    return {
        "rows": count,
        "accuracy": sum(bool(row["correct"]) for row in rows) / count,
        "errors": sum(not bool(row["correct"]) for row in rows),
        "nll": nll / count,
        "brier": brier / count,
        "ece_10_bin": ece,
        "mean_confidence": sum(float(row["confidence"]) for row in rows) / count,
        "mean_predictive_entropy": sum(float(row["predictive_entropy"]) for row in rows) / count,
        "mean_expected_entropy": sum(expected) / len(expected) if expected else None,
        "mean_mutual_information": sum(information) / len(information) if information else None,
        "error_detection_auroc": error_detection_auroc(rows),
        "prediction_counts": dict(sorted(Counter(row["prediction"] for row in rows).items())),
    }


def build_method_metrics(normalized: Sequence[Mapping[str, Any]], *, nll_epsilon: float = 1e-12, bins: int = 10) -> list[dict[str, Any]]:
    outputs = []
    for variant in ALL_VARIANTS:
        members = [row for row in normalized if row["variant"] == variant]
        require(len(members) == 2400, f"variant cardinality drift: {variant}")
        for scope in ("overall", *CONDITIONS):
            scoped = members if scope == "overall" else [row for row in members if row["condition"] == scope]
            expected = 2400 if scope == "overall" else 400
            require(len(scoped) == expected, f"scope cardinality drift: {variant}/{scope}")
            outputs.append({
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "variant": variant,
                "display_label": DISPLAY_LABELS[variant],
                "scope": scope,
                **summarize_rows(scoped, nll_epsilon=nll_epsilon, bins=bins),
            })
    return outputs


def build_reliability_table(normalized: Sequence[Mapping[str, Any]], bins: int = 10) -> list[dict[str, Any]]:
    outputs = []
    for variant in PRIMARY_VARIANTS:
        members = [row for row in normalized if row["variant"] == variant]
        for item in reliability_bins(members, bins):
            outputs.append({
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "variant": variant,
                "display_label": DISPLAY_LABELS[variant],
                "scope": "overall",
                **item,
            })
    return outputs


def build_risk_coverage(normalized: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    outputs = []
    for variant in PRIMARY_VARIANTS:
        members = sorted(
            (row for row in normalized if row["variant"] == variant),
            key=lambda row: (-float(row["confidence"]), str(row["analysis_id"])),
        )
        errors = 0
        for rank, row in enumerate(members, start=1):
            errors += not bool(row["correct"])
            outputs.append({
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "variant": variant,
                "display_label": DISPLAY_LABELS[variant],
                "retained": rank,
                "total": len(members),
                "coverage": rank / len(members),
                "risk": errors / rank,
                "selective_accuracy": 1.0 - errors / rank,
                "confidence_threshold": float(row["confidence"]),
            })
    require(len(outputs) == len(PRIMARY_VARIANTS) * 2400, "risk-coverage cardinality drift")
    return outputs


def build_paired_changes(normalized: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    outputs = []
    for variant in ALL_VARIANTS:
        members = [row for row in normalized if row["variant"] == variant]
        original = {row["input_id"]: row for row in members if row["condition"] == "original"}
        require(len(original) == 400, f"original alignment drift: {variant}")
        for condition in CONDITIONS:
            degraded = [row for row in members if row["condition"] == condition]
            pairs = [(original[row["input_id"]], row) for row in degraded]
            require(len(pairs) == 400, f"paired-change cardinality drift: {variant}/{condition}")
            information_pairs = [(a, b) for a, b in pairs if a["mutual_information"] is not None and b["mutual_information"] is not None]
            outputs.append({
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "variant": variant,
                "display_label": DISPLAY_LABELS[variant],
                "condition": condition,
                "mean_accuracy_change_from_original": sum(float(b["correct"]) - float(a["correct"]) for a, b in pairs) / len(pairs),
                "mean_confidence_change_from_original": sum(float(b["confidence"]) - float(a["confidence"]) for a, b in pairs) / len(pairs),
                "mean_predictive_entropy_change_from_original": sum(float(b["predictive_entropy"]) - float(a["predictive_entropy"]) for a, b in pairs) / len(pairs),
                "mean_mutual_information_change_from_original": sum(float(b["mutual_information"]) - float(a["mutual_information"]) for a, b in information_pairs) / len(information_pairs) if information_pairs else None,
            })
    return outputs


def build_expressed_summary(expressed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(expressed) == len(METHODS) * 2400, "expressed summary input drift")
    outputs = []
    for method in METHODS:
        method_rows = [row for row in expressed if row["method"] == method]
        for scope in ("overall", *CONDITIONS):
            scoped = method_rows if scope == "overall" else [row for row in method_rows if row["condition"] == scope]
            valid = [row for row in scoped if row["parser_valid"]]
            outputs.append({
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "method": method,
                "scope": scope,
                "rows": len(scoped),
                "valid_rows": len(valid),
                "valid_rate": len(valid) / len(scoped),
                "mean_expressed_confidence_valid_only": sum(float(row["expressed_confidence"]) for row in valid) / len(valid) if valid else None,
                "mean_raw_token_confidence_valid_only": sum(float(row["raw_token_confidence"]) for row in valid) / len(valid) if valid else None,
                "mean_calibrated_token_confidence_valid_only": sum(float(row["calibrated_token_confidence"]) for row in valid) / len(valid) if valid else None,
                "mean_raw_absolute_divergence_valid_only": sum(float(row["raw_absolute_divergence"]) for row in valid) / len(valid) if valid else None,
                "mean_calibrated_absolute_divergence_valid_only": sum(float(row["calibrated_absolute_divergence"]) for row in valid) / len(valid) if valid else None,
                "invalid_rows_retained_without_imputation": len(scoped) - len(valid),
            })
    return outputs


def _metric_lookup(metrics: Sequence[Mapping[str, Any]], variant: str, scope: str) -> Mapping[str, Any]:
    matches = [row for row in metrics if row["variant"] == variant and row["scope"] == scope]
    require(len(matches) == 1, f"metric lookup drift: {variant}/{scope}")
    return matches[0]


def render_figures(
    *,
    normalized: Sequence[Mapping[str, Any]],
    expressed: Sequence[Mapping[str, Any]],
    method_metrics: Sequence[Mapping[str, Any]],
    reliability: Sequence[Mapping[str, Any]],
    risk_coverage: Sequence[Mapping[str, Any]],
    paired_changes: Sequence[Mapping[str, Any]],
    directory: Path,
) -> list[Path]:
    """Render the eight required figures from derived tables and normalized rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
    })
    colors = dict(zip(PRIMARY_VARIANTS, plt.get_cmap("tab10").colors[: len(PRIMARY_VARIANTS)]))
    paths: list[Path] = []

    def save(fig: Any, name: str) -> None:
        require(name in FIGURES, "unknown figure output")
        path = directory / name
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", metadata={"Software": "llm-confidence-uq"})
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharex=True, sharey=True)
    panels = (
        (axes[0], ("baseline_raw", "baseline_calibrated"), "Baseline"),
        (axes[1], ("full_seed_1_calibrated", "full_seed_2_calibrated", "full_seed_3_calibrated"), "Calibrated LoRA seeds"),
        (axes[2], ("three_lora_ensemble_calibrated", "laplace_head"), "Ensemble and Bayesian head"),
    )
    for axis, variants, title in panels:
        axis.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="black", linewidth=1, label="Ideal")
        for variant in variants:
            rows = [row for row in reliability if row["variant"] == variant and row["count"] > 0]
            axis.plot([row["mean_confidence"] for row in rows], [row["accuracy"] for row in rows], marker="o", linewidth=1.5, color=colors[variant], label=DISPLAY_LABELS[variant])
        axis.set_title(title)
        axis.set_xlim(0.48, 1.01)
        axis.set_ylim(0.0, 1.01)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[0].set_ylabel("Empirical accuracy")
    for axis in axes:
        axis.set_xlabel("Mean confidence")
    fig.suptitle("Reliability diagrams (10 equal-width bins)")
    save(fig, "01_reliability_diagram.png")

    fig, axis = plt.subplots(figsize=(8.6, 5.2))
    for variant in PRIMARY_VARIANTS:
        rows = [row for row in risk_coverage if row["variant"] == variant and (row["retained"] == 1 or row["retained"] % 24 == 0)]
        axis.plot([row["coverage"] for row in rows], [row["risk"] for row in rows], linewidth=1.5, color=colors[variant], label=DISPLAY_LABELS[variant])
    axis.set_xlabel("Coverage retained")
    axis.set_ylabel("Selective risk")
    axis.set_title("Risk–coverage comparison")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    save(fig, "02_risk_coverage.png")

    selected_conditions = list(CONDITIONS)
    x = np.arange(len(selected_conditions))
    width = 0.8 / len(PRIMARY_VARIANTS)
    fig, axis = plt.subplots(figsize=(13.2, 5.3))
    for offset, variant in enumerate(PRIMARY_VARIANTS):
        values = [_metric_lookup(method_metrics, variant, condition)["accuracy"] for condition in selected_conditions]
        axis.bar(x - 0.4 + width / 2 + offset * width, values, width, label=DISPLAY_LABELS[variant], color=colors[variant])
    axis.set_xticks(x, [condition.replace("_", "\n") for condition in selected_conditions])
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Accuracy by evidence condition")
    axis.legend(ncol=2)
    axis.grid(axis="y", alpha=0.25)
    save(fig, "03_accuracy_by_condition.png")

    fig, axis = plt.subplots(figsize=(13.2, 5.3))
    for offset, variant in enumerate(PRIMARY_VARIANTS):
        values = [_metric_lookup(method_metrics, variant, condition)["ece_10_bin"] for condition in selected_conditions]
        axis.bar(x - 0.4 + width / 2 + offset * width, values, width, label=DISPLAY_LABELS[variant], color=colors[variant])
    axis.set_xticks(x, [condition.replace("_", "\n") for condition in selected_conditions])
    axis.set_ylabel("ECE (10 bins; lower is better)")
    axis.set_title("Calibration error by evidence condition")
    axis.legend(ncol=2)
    axis.grid(axis="y", alpha=0.25)
    save(fig, "04_ece_by_condition.png")

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 8.2), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, METHODS):
        valid = [row for row in expressed if row["method"] == method and row["parser_valid"]]
        axis.scatter([row["calibrated_token_confidence"] for row in valid], [row["expressed_confidence"] for row in valid], s=7, alpha=0.18, edgecolors="none")
        axis.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="black", linewidth=1)
        axis.set_title(f"{method.replace('_', ' ')}: {len(valid)}/2400 valid")
        axis.set_xlim(0.48, 1.01)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Calibrated token confidence")
    for axis in axes[:, 0]:
        axis.set_ylabel("Expressed confidence")
    fig.suptitle("Expressed versus token-derived confidence (valid parser rows only)")
    save(fig, "05_expressed_vs_token_confidence.png")

    degradation_conditions = list(CONDITIONS[1:])
    matrix_variants = list(PRIMARY_VARIANTS)
    confidence_matrix = np.array([[next(row["mean_confidence_change_from_original"] for row in paired_changes if row["variant"] == variant and row["condition"] == condition) for condition in degradation_conditions] for variant in matrix_variants])
    entropy_matrix = np.array([[next(row["mean_predictive_entropy_change_from_original"] for row in paired_changes if row["variant"] == variant and row["condition"] == condition) for condition in degradation_conditions] for variant in matrix_variants])

    def heatmap(matrix: Any, title: str, label: str, name: str) -> None:
        limit = max(float(np.max(np.abs(matrix))), 1e-9)
        fig, axis = plt.subplots(figsize=(10.5, 5.1))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(range(len(degradation_conditions)), [condition.replace("_", "\n") for condition in degradation_conditions])
        axis.set_yticks(range(len(matrix_variants)), [DISPLAY_LABELS[variant] for variant in matrix_variants])
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:+.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, label=label)
        axis.set_title(title)
        save(fig, name)

    heatmap(confidence_matrix, "Mean confidence change from original evidence", "Δ confidence", "06_confidence_change.png")
    heatmap(entropy_matrix, "Mean predictive-entropy change from original evidence", "Δ entropy (nats)", "07_uncertainty_change.png")

    overall = [_metric_lookup(method_metrics, variant, "overall") for variant in PRIMARY_VARIANTS]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    comparisons = (
        (axes[0, 0], "accuracy", "Accuracy", True),
        (axes[0, 1], "nll", "Negative log-likelihood", False),
        (axes[1, 0], "brier", "Brier score", False),
        (axes[1, 1], "error_detection_auroc", "Error-detection AUROC", True),
    )
    labels = [DISPLAY_LABELS[variant] for variant in PRIMARY_VARIANTS]
    for axis, key, title, higher in comparisons:
        raw_values = [row[key] for row in overall]
        values = [float("nan") if value is None else float(value) for value in raw_values]
        bars = axis.bar(range(len(values)), values, color=[colors[variant] for variant in PRIMARY_VARIANTS])
        axis.set_xticks(range(len(values)), labels, rotation=35, ha="right")
        axis.set_title(f"{title} ({'higher' if higher else 'lower'} is better)")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, raw_values):
            if value is None:
                axis.text(bar.get_x() + bar.get_width() / 2, 0.0, "NA", ha="center", va="bottom", fontsize=7)
            else:
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{float(value):.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Overall method comparison")
    save(fig, "08_method_comparison.png")

    require(tuple(path.name for path in paths) == FIGURES, "figure order or cardinality drift")
    return paths
