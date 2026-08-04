"""Clustered uncertainty intervals and direct UQ-signal diagnostics.

All bootstrap resampling is performed over the 400 aligned BoolQ input IDs.
Each sampled cluster carries every evidence condition, which preserves the
paired structure used for method differences.
"""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from .analysis import (
    CONDITIONS,
    PRIMARY_VARIANTS,
    canonical_json,
    require,
    summarize_rows,
)


PROTOCOL_VERSION = "boolq-clustered-statistical-analysis-v1"
BOOTSTRAP_METRICS = (
    "accuracy",
    "nll",
    "brier",
    "ece_10_bin",
    "error_detection_auroc",
)
LORA_SEED_VARIANTS = (
    "full_seed_1_calibrated",
    "full_seed_2_calibrated",
    "full_seed_3_calibrated",
)
COMPARISONS = (
    ("baseline_calibrated_vs_raw", ("baseline_calibrated",), "baseline_raw"),
    ("lora_seed_1_vs_baseline_calibrated", ("full_seed_1_calibrated",), "baseline_calibrated"),
    ("lora_seed_2_vs_baseline_calibrated", ("full_seed_2_calibrated",), "baseline_calibrated"),
    ("lora_seed_3_vs_baseline_calibrated", ("full_seed_3_calibrated",), "baseline_calibrated"),
    ("lora_seed_mean_vs_baseline_calibrated", LORA_SEED_VARIANTS, "baseline_calibrated"),
    ("ensemble_calibrated_vs_baseline_calibrated", ("three_lora_ensemble_calibrated",), "baseline_calibrated"),
    ("laplace_head_vs_baseline_calibrated", ("laplace_head",), "baseline_calibrated"),
)


def binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Tie-aware AUROC for a score whose larger values indicate positive."""
    require(len(labels) == len(scores) and len(labels) > 0, "invalid AUROC input")
    values = sorted((float(score), bool(label)) for score, label in zip(scores, labels))
    require(all(math.isfinite(score) for score, _ in values), "non-finite AUROC score")
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


def binary_average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Threshold-grouped average precision with deterministic tie handling."""
    require(len(labels) == len(scores) and len(labels) > 0, "invalid AUPRC input")
    values = sorted(
        ((float(score), bool(label)) for score, label in zip(scores, labels)),
        key=lambda item: item[0],
        reverse=True,
    )
    require(all(math.isfinite(score) for score, _ in values), "non-finite AUPRC score")
    positives = sum(label for _, label in values)
    if positives == 0:
        return None
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end][0] == values[index][0]:
            end += 1
        true_positives += sum(label for _, label in values[index:end])
        false_positives += sum(not label for _, label in values[index:end])
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def _scope_rows(rows: Sequence[Mapping[str, Any]], scope: str) -> list[Mapping[str, Any]]:
    require(scope == "overall" or scope in CONDITIONS, "unknown analysis scope")
    return list(rows) if scope == "overall" else [row for row in rows if row["condition"] == scope]


def build_lora_seed_summary(
    normalized: Sequence[Mapping[str, Any]],
    *,
    nll_epsilon: float = 1e-12,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """Summarize the three calibrated LoRA seeds without selecting a winner."""
    by_variant = {
        variant: [row for row in normalized if row["variant"] == variant]
        for variant in LORA_SEED_VARIANTS
    }
    require(all(len(rows) == 2400 for rows in by_variant.values()), "LoRA seed cardinality drift")
    outputs: list[dict[str, Any]] = []
    for scope in ("overall", *CONDITIONS):
        seed_metrics = {
            variant: summarize_rows(
                _scope_rows(rows, scope), nll_epsilon=nll_epsilon, bins=bins
            )
            for variant, rows in by_variant.items()
        }
        summaries: dict[str, Any] = {}
        for metric in BOOTSTRAP_METRICS:
            values = [float(seed_metrics[variant][metric]) for variant in LORA_SEED_VARIANTS]
            summaries[metric] = {
                "mean": mean(values),
                "sample_standard_deviation": stdev(values),
                "minimum": min(values),
                "maximum": max(values),
                "values_by_seed": {
                    variant: value for variant, value in zip(LORA_SEED_VARIANTS, values)
                },
            }
        outputs.append({
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "scope": scope,
            "seeds": list(LORA_SEED_VARIANTS),
            "seed_count": 3,
            "metrics": summaries,
        })
    return outputs


def _bootstrap_metric_replicates(
    rows: Sequence[Mapping[str, Any]],
    cluster_ids: Sequence[str],
    cluster_counts: Any,
    *,
    bins: int,
    nll_epsilon: float,
) -> dict[str, Any]:
    import numpy as np

    cluster_lookup = {value: index for index, value in enumerate(cluster_ids)}
    cluster_index = np.asarray([cluster_lookup[str(row["input_id"])] for row in rows], dtype=np.int64)
    rows_per_cluster = len(rows) // len(cluster_ids)
    require(rows_per_cluster * len(cluster_ids) == len(rows), "nonuniform bootstrap clusters")
    require(
        all(sum(str(row["input_id"]) == cluster for row in rows) == rows_per_cluster for cluster in cluster_ids),
        "bootstrap cluster width drift",
    )

    targets = np.asarray([1.0 if row["ground_truth"] == "Yes" else 0.0 for row in rows])
    probabilities = np.clip(
        np.asarray([float(row["p_yes"]) for row in rows]),
        nll_epsilon,
        1.0 - nll_epsilon,
    )
    confidence = np.asarray([float(row["confidence"]) for row in rows])
    correct = np.asarray([float(bool(row["correct"])) for row in rows])
    losses = -(targets * np.log(probabilities) + (1.0 - targets) * np.log(1.0 - probabilities))
    squared_errors = (probabilities - targets) ** 2

    def cluster_sums(values: Any) -> Any:
        result = np.zeros(len(cluster_ids), dtype=np.float64)
        np.add.at(result, cluster_index, values)
        return result

    denominator = cluster_counts.sum(axis=1).astype(np.float64) * rows_per_cluster
    outputs = {
        "accuracy": cluster_counts @ cluster_sums(correct) / denominator,
        "nll": cluster_counts @ cluster_sums(losses) / denominator,
        "brier": cluster_counts @ cluster_sums(squared_errors) / denominator,
    }

    bin_index = np.ceil(confidence * bins).astype(np.int64) - 1
    bin_index = np.clip(bin_index, 0, bins - 1)
    bin_counts = np.zeros((len(cluster_ids), bins), dtype=np.float64)
    bin_correct = np.zeros_like(bin_counts)
    bin_confidence = np.zeros_like(bin_counts)
    np.add.at(bin_counts, (cluster_index, bin_index), 1.0)
    np.add.at(bin_correct, (cluster_index, bin_index), correct)
    np.add.at(bin_confidence, (cluster_index, bin_index), confidence)
    sampled_counts = cluster_counts @ bin_counts
    sampled_correct = cluster_counts @ bin_correct
    sampled_confidence = cluster_counts @ bin_confidence
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy_by_bin = np.divide(
            sampled_correct,
            sampled_counts,
            out=np.zeros_like(sampled_correct),
            where=sampled_counts > 0,
        )
        confidence_by_bin = np.divide(
            sampled_confidence,
            sampled_counts,
            out=np.zeros_like(sampled_confidence),
            where=sampled_counts > 0,
        )
    outputs["ece_10_bin"] = np.sum(
        sampled_counts / denominator[:, None] * np.abs(accuracy_by_bin - confidence_by_bin),
        axis=1,
        where=sampled_counts > 0,
    )

    scores = 1.0 - confidence
    labels = 1.0 - correct
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ordered_clusters = cluster_index[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(ordered_scores) != 0.0) + 1))
    aurocs = np.empty(cluster_counts.shape[0], dtype=np.float64)
    for start in range(0, cluster_counts.shape[0], 128):
        stop = min(start + 128, cluster_counts.shape[0])
        weights = cluster_counts[start:stop, ordered_clusters]
        positives = np.add.reduceat(weights * ordered_labels, starts, axis=1)
        negatives = np.add.reduceat(weights * (1.0 - ordered_labels), starts, axis=1)
        lower_negatives = np.cumsum(negatives, axis=1) - negatives
        numerator = np.sum(positives * (lower_negatives + 0.5 * negatives), axis=1)
        total_positives = np.sum(positives, axis=1)
        total_negatives = np.sum(negatives, axis=1)
        denominator_auc = total_positives * total_negatives
        aurocs[start:stop] = np.divide(
            numerator,
            denominator_auc,
            out=np.full(stop - start, np.nan),
            where=denominator_auc > 0,
        )
    outputs["error_detection_auroc"] = aurocs
    return outputs


def build_clustered_bootstrap_differences(
    normalized: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
    confidence_level: float = 0.95,
    nll_epsilon: float = 1e-12,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """Percentile intervals for paired method differences clustered by input ID."""
    import numpy as np

    require(repetitions >= 100, "too few bootstrap repetitions")
    require(0.5 < confidence_level < 1.0, "invalid bootstrap confidence level")
    by_variant = defaultdict(list)
    for row in normalized:
        by_variant[str(row["variant"])].append(row)
    cluster_ids = sorted({str(row["input_id"]) for row in by_variant["baseline_raw"]})
    require(len(cluster_ids) == 400, "bootstrap requires 400 input clusters")
    for variant in PRIMARY_VARIANTS:
        require(len(by_variant[variant]) == 2400, f"bootstrap variant cardinality drift: {variant}")
        require({str(row["input_id"]) for row in by_variant[variant]} == set(cluster_ids), "bootstrap alignment drift")

    rng = np.random.default_rng(seed)
    cluster_counts = rng.multinomial(
        len(cluster_ids),
        np.full(len(cluster_ids), 1.0 / len(cluster_ids)),
        size=repetitions,
    ).astype(np.float64)
    lower_quantile = (1.0 - confidence_level) / 2.0
    upper_quantile = 1.0 - lower_quantile
    outputs: list[dict[str, Any]] = []
    for scope in ("overall", *CONDITIONS):
        replicate_cache: dict[str, dict[str, Any]] = {}
        point_cache: dict[str, dict[str, Any]] = {}
        variants_needed = sorted({variant for _, candidates, reference in COMPARISONS for variant in (*candidates, reference)})
        for variant in variants_needed:
            scoped = _scope_rows(by_variant[variant], scope)
            replicate_cache[variant] = _bootstrap_metric_replicates(
                scoped,
                cluster_ids,
                cluster_counts,
                bins=bins,
                nll_epsilon=nll_epsilon,
            )
            point_cache[variant] = summarize_rows(scoped, nll_epsilon=nll_epsilon, bins=bins)

        for comparison, candidates, reference in COMPARISONS:
            metric_records: dict[str, Any] = {}
            for metric in BOOTSTRAP_METRICS:
                candidate_replicates = np.mean(
                    np.stack([replicate_cache[variant][metric] for variant in candidates]),
                    axis=0,
                )
                reference_replicates = replicate_cache[reference][metric]
                differences = candidate_replicates - reference_replicates
                finite = differences[np.isfinite(differences)]
                require(finite.size >= repetitions * 0.99, "too many undefined bootstrap replicates")
                candidate_point = mean(float(point_cache[variant][metric]) for variant in candidates)
                reference_point = float(point_cache[reference][metric])
                metric_records[metric] = {
                    "candidate": candidate_point,
                    "reference": reference_point,
                    "difference_candidate_minus_reference": candidate_point - reference_point,
                    "percentile_interval_lower": float(np.quantile(finite, lower_quantile, method="linear")),
                    "percentile_interval_upper": float(np.quantile(finite, upper_quantile, method="linear")),
                    "valid_replicates": int(finite.size),
                    "favourable_difference_direction": "positive" if metric in ("accuracy", "error_detection_auroc") else "negative",
                }
            outputs.append({
                "schema_version": 1,
                "protocol_version": PROTOCOL_VERSION,
                "comparison": comparison,
                "candidate_variants": list(candidates),
                "candidate_aggregation": "identity" if len(candidates) == 1 else "arithmetic-mean-across-three-seeds",
                "reference_variant": reference,
                "scope": scope,
                "cluster_unit": "input_id",
                "clusters": len(cluster_ids),
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed,
                "confidence_level": confidence_level,
                "interval_method": "paired-cluster-percentile",
                "metrics": metric_records,
            })
    return outputs


def _signal_sources(
    normalized: Sequence[Mapping[str, Any]],
    ensemble_rows: Sequence[Mapping[str, Any]],
    laplace_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_variant = defaultdict(list)
    for row in normalized:
        by_variant[str(row["variant"])].append(row)
    outputs: list[dict[str, Any]] = []
    for variant in PRIMARY_VARIANTS:
        outputs.append({
            "variant": variant,
            "signal": "predictive_entropy",
            "rows": [
                {
                    "input_id": str(row["input_id"]),
                    "condition": str(row["condition"]),
                    "correct": bool(row["correct"]),
                    "score": float(row["predictive_entropy"]),
                }
                for row in by_variant[variant]
            ],
        })

    ensemble_base = {
        (str(row["input_id"]), str(row["condition"])): row for row in ensemble_rows
    }
    laplace_base = {
        (str(row["example_id"]), str(row["condition"])): row for row in laplace_rows
    }
    require(len(ensemble_base) == len(laplace_base) == 2400, "UQ source cardinality drift")
    ensemble_correct = {
        (str(row["input_id"]), str(row["condition"])): bool(row["correct"])
        for row in by_variant["three_lora_ensemble_calibrated"]
    }
    laplace_correct = {
        (str(row["input_id"]), str(row["condition"])): bool(row["correct"])
        for row in by_variant["laplace_head"]
    }
    for signal, field in (
        ("ensemble_member_probability_variance", "calibrated_member_probability_variance"),
        ("ensemble_mi_style_disagreement", "calibrated_mutual_information_style"),
    ):
        outputs.append({
            "variant": "three_lora_ensemble_calibrated",
            "signal": signal,
            "rows": [
                {
                    "input_id": key[0],
                    "condition": key[1],
                    "correct": ensemble_correct[key],
                    "score": float(row[field]),
                }
                for key, row in sorted(ensemble_base.items())
            ],
        })
    for signal, field in (
        ("laplace_posterior_predictive_variance", "posterior_predictive_variance"),
        ("laplace_mutual_information", "mutual_information"),
    ):
        outputs.append({
            "variant": "laplace_head",
            "signal": signal,
            "rows": [
                {
                    "input_id": key[0],
                    "condition": key[1],
                    "correct": laplace_correct[key],
                    "score": float(row[field]),
                }
                for key, row in sorted(laplace_base.items())
            ],
        })
    require(all(len(source["rows"]) == 2400 for source in outputs), "UQ signal row drift")
    return outputs


def build_uq_signal_evaluations(
    normalized: Sequence[Mapping[str, Any]],
    ensemble_rows: Sequence[Mapping[str, Any]],
    laplace_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate UQ scores for error ranking and evidence-degradation ranking."""
    sources = _signal_sources(normalized, ensemble_rows, laplace_rows)
    error_outputs: list[dict[str, Any]] = []
    degradation_outputs: list[dict[str, Any]] = []
    for source in sources:
        rows = source["rows"]
        scores = [float(row["score"]) for row in rows]
        error_labels = [not bool(row["correct"]) for row in rows]
        error_outputs.append({
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "variant": source["variant"],
            "signal": source["signal"],
            "positive_class": "prediction_error",
            "higher_score_means": "more_uncertain",
            "rows": len(rows),
            "positives": sum(error_labels),
            "positive_rate": sum(error_labels) / len(rows),
            "auroc": binary_auroc(error_labels, scores),
            "average_precision": binary_average_precision(error_labels, scores),
        })

        original = [row for row in rows if row["condition"] == "original"]
        require(len(original) == 400, "original UQ signal cardinality drift")
        for target in ("any_degraded", *CONDITIONS[1:]):
            degraded = (
                [row for row in rows if row["condition"] != "original"]
                if target == "any_degraded"
                else [row for row in rows if row["condition"] == target]
            )
            combined = original + degraded
            labels = [False] * len(original) + [True] * len(degraded)
            combined_scores = [float(row["score"]) for row in combined]
            degradation_outputs.append({
                "schema_version": 1,
                "protocol_version": PROTOCOL_VERSION,
                "variant": source["variant"],
                "signal": source["signal"],
                "target": target,
                "positive_class": "degraded_evidence",
                "higher_score_means": "more_uncertain",
                "rows": len(combined),
                "original_rows": len(original),
                "degraded_rows": len(degraded),
                "positive_rate": len(degraded) / len(combined),
                "mean_original_score": mean(float(row["score"]) for row in original),
                "mean_degraded_score": mean(float(row["score"]) for row in degraded),
                "mean_difference_degraded_minus_original": mean(float(row["score"]) for row in degraded) - mean(float(row["score"]) for row in original),
                "auroc": binary_auroc(labels, combined_scores),
                "average_precision": binary_average_precision(labels, combined_scores),
            })

    seed_error = [
        row for row in error_outputs
        if row["variant"] in LORA_SEED_VARIANTS and row["signal"] == "predictive_entropy"
    ]
    seed_degradation = [
        row for row in degradation_outputs
        if row["variant"] in LORA_SEED_VARIANTS and row["signal"] == "predictive_entropy"
    ]
    grouped: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "signal": "predictive_entropy",
        "seeds": list(LORA_SEED_VARIANTS),
        "error_detection": {},
        "degradation_detection": {},
    }
    for metric in ("auroc", "average_precision"):
        values = [float(row[metric]) for row in seed_error]
        grouped["error_detection"][metric] = {
            "mean": mean(values), "sample_standard_deviation": stdev(values),
            "minimum": min(values), "maximum": max(values),
        }
    for target in ("any_degraded", *CONDITIONS[1:]):
        grouped["degradation_detection"][target] = {}
        members = [row for row in seed_degradation if row["target"] == target]
        for metric in ("auroc", "average_precision"):
            values = [float(row[metric]) for row in members]
            grouped["degradation_detection"][target][metric] = {
                "mean": mean(values), "sample_standard_deviation": stdev(values),
                "minimum": min(values), "maximum": max(values),
            }
    return error_outputs, degradation_outputs, [grouped]


def validate_statistical_outputs(
    bootstrap_rows: Sequence[Mapping[str, Any]],
    seed_rows: Sequence[Mapping[str, Any]],
    error_rows: Sequence[Mapping[str, Any]],
    degradation_rows: Sequence[Mapping[str, Any]],
) -> None:
    require(len(bootstrap_rows) == len(COMPARISONS) * 7, "bootstrap output cardinality drift")
    require(len(seed_rows) == 7, "seed-summary cardinality drift")
    require(len(error_rows) == len(PRIMARY_VARIANTS) + 4, "UQ error output cardinality drift")
    require(len(degradation_rows) == len(error_rows) * 6, "UQ degradation output cardinality drift")
    require(all(row["clusters"] == 400 for row in bootstrap_rows), "bootstrap cluster drift")
    encoded = canonical_json({
        "bootstrap": bootstrap_rows,
        "seed_summary": seed_rows,
        "error": error_rows,
        "degradation": degradation_rows,
    })
    require("NaN" not in encoded and "Infinity" not in encoded, "non-finite statistical output")


__all__ = [
    "BOOTSTRAP_METRICS",
    "COMPARISONS",
    "LORA_SEED_VARIANTS",
    "PROTOCOL_VERSION",
    "binary_auroc",
    "binary_average_precision",
    "build_clustered_bootstrap_differences",
    "build_lora_seed_summary",
    "build_uq_signal_evaluations",
    "validate_statistical_outputs",
]
