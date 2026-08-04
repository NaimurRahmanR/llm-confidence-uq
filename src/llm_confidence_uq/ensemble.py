"""Deterministic three-member LoRA ensemble aggregation and evaluation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-three-lora-ensemble-v1"
SOURCE_PROTOCOL_VERSION = "boolq-calibrated-test-evaluation-v1"
MEMBERS = ("full_seed_1", "full_seed_2", "full_seed_3")
CONDITIONS = (
    "original",
    "lexical_evidence_removal",
    "prefix_truncation_50",
    "irrelevant_distractor",
    "lexical_contradiction",
    "no_passage",
)
CLASS_ORDER = ("Yes", "No")


class EnsembleError(RuntimeError):
    """Raised when ensemble inputs violate the locked protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EnsembleError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _validate_source_row(row: Mapping[str, Any], method: str) -> None:
    require(row.get("schema_version") == 1, "source schema drift")
    require(row.get("protocol_version") == SOURCE_PROTOCOL_VERSION, "source protocol drift")
    require(row.get("method") == method, "member identity drift")
    require(row.get("class_order") == list(CLASS_ORDER), "class-order drift")
    require(row.get("condition") in CONDITIONS, "unknown condition")
    require(int(row.get("condition_index")) == CONDITIONS.index(row["condition"]), "condition-index drift")
    require(row.get("ground_truth") in CLASS_ORDER, "invalid ground truth")
    require(row.get("token_prediction") in CLASS_ORDER, "invalid member prediction")
    require(type(row.get("correct")) is bool, "invalid member correctness")
    require(row["correct"] == (row["token_prediction"] == row["ground_truth"]), "member correctness drift")
    require(isinstance(row.get("input_id"), str) and row["input_id"], "invalid input ID")
    require(type(row.get("source_index")) is int, "invalid source index")
    require(isinstance(row.get("source_prediction_row_sha256"), str) and len(row["source_prediction_row_sha256"]) == 64, "invalid source-row hash")
    require(isinstance(row.get("row_sha256"), str) and len(row["row_sha256"]) == 64, "invalid evaluation-row hash")
    for family in ("raw", "calibrated"):
        p_yes = _finite_probability(row.get(f"{family}_p_yes"), f"{family} Yes probability")
        p_no = _finite_probability(row.get(f"{family}_p_no"), f"{family} No probability")
        require(abs(p_yes + p_no - 1.0) <= 1e-6, f"{family} probabilities do not normalize")
    require(float(row.get("temperature")) > 0.0, "non-positive member temperature")
    calibrated_prediction = "Yes" if float(row["calibrated_p_yes"]) >= float(row["calibrated_p_no"]) else "No"
    require(calibrated_prediction == row["token_prediction"], "member calibration changed class")
    unhashed = dict(row)
    observed_hash = unhashed.pop("row_sha256")
    require(sha256_text(canonical_json(unhashed)) == observed_hash, "evaluation-row hash drift")


def aggregate_rows(
    member_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    member_metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align three independently seeded adapters and compute ensemble UQ rows."""
    require(tuple(member_rows) == MEMBERS, "members are absent or out of locked order")
    require(tuple(member_metadata) == MEMBERS, "member metadata are absent or out of locked order")
    lengths = {len(member_rows[member]) for member in MEMBERS}
    require(len(lengths) == 1 and next(iter(lengths)) > 0, "member row cardinality drift")
    adapter_hashes = []
    seeds = []
    for member in MEMBERS:
        metadata = member_metadata[member]
        require(metadata.get("method") == member, "metadata member identity drift")
        require(type(metadata.get("seed")) is int, "invalid member seed")
        require(isinstance(metadata.get("adapter_sha256"), str) and len(metadata["adapter_sha256"]) == 64, "invalid adapter hash")
        seeds.append(metadata["seed"])
        adapter_hashes.append(metadata["adapter_sha256"])
    require(len(set(seeds)) == len(MEMBERS), "ensemble seeds are not unique")
    require(len(set(adapter_hashes)) == len(MEMBERS), "ensemble checkpoints are not unique")

    outputs: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int, str]] = set()
    for ordinal in range(next(iter(lengths))):
        aligned = {member: member_rows[member][ordinal] for member in MEMBERS}
        for member, row in aligned.items():
            _validate_source_row(row, member)
        anchor = aligned[MEMBERS[0]]
        key = (str(anchor["input_id"]), int(anchor["source_index"]), str(anchor["condition"]))
        require(key not in seen_keys, "duplicate aligned ensemble key")
        seen_keys.add(key)
        for member in MEMBERS[1:]:
            row = aligned[member]
            require(
                (row["input_id"], row["source_index"], row["condition"], row["condition_index"])
                == (anchor["input_id"], anchor["source_index"], anchor["condition"], anchor["condition_index"]),
                "member alignment drift",
            )
            require(row["ground_truth"] == anchor["ground_truth"], "ground-truth alignment drift")
        source_hashes = [aligned[member]["source_prediction_row_sha256"] for member in MEMBERS]
        require(len(set(source_hashes)) == len(MEMBERS), "member prediction identities are not distinct")

        member_payloads = []
        votes = Counter()
        families: dict[str, dict[str, float]] = {}
        for member in MEMBERS:
            row = aligned[member]
            votes[row["token_prediction"]] += 1
            member_payloads.append({
                "method": member,
                "seed": member_metadata[member]["seed"],
                "adapter_sha256": member_metadata[member]["adapter_sha256"],
                "source_prediction_row_sha256": row["source_prediction_row_sha256"],
                "token_prediction": row["token_prediction"],
                "raw_p_yes": float(row["raw_p_yes"]),
                "calibrated_p_yes": float(row["calibrated_p_yes"]),
                "temperature": float(row["temperature"]),
            })
        for family in ("raw", "calibrated"):
            probabilities = [float(aligned[member][f"{family}_p_yes"]) for member in MEMBERS]
            mean = sum(probabilities) / len(probabilities)
            variance = sum((probability - mean) ** 2 for probability in probabilities) / len(probabilities)
            predictive_entropy = entropy(mean)
            expected_entropy = sum(entropy(probability) for probability in probabilities) / len(probabilities)
            information = predictive_entropy - expected_entropy
            require(variance >= 0.0 and math.isfinite(variance), "invalid member variance")
            require(information >= -1e-12 and math.isfinite(information), "invalid mutual-information-style value")
            families[family] = {
                "p_yes": mean,
                "p_no": 1.0 - mean,
                "confidence": max(mean, 1.0 - mean),
                "predictive_entropy": predictive_entropy,
                "expected_entropy": expected_entropy,
                "mutual_information_style": max(0.0, information),
                "member_probability_variance": variance,
            }

        identity = {
            "input_id": anchor["input_id"],
            "source_index": anchor["source_index"],
            "condition": anchor["condition"],
            "protocol_version": PROTOCOL_VERSION,
            "source_prediction_row_sha256s": source_hashes,
        }
        row = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "ensemble_id": "boolqensemble-" + sha256_text(canonical_json(identity)),
            "input_id": anchor["input_id"],
            "source_index": anchor["source_index"],
            "condition": anchor["condition"],
            "condition_index": anchor["condition_index"],
            "class_order": list(CLASS_ORDER),
            "ground_truth": anchor["ground_truth"],
            "members": member_payloads,
            "vote_counts": {label: votes.get(label, 0) for label in CLASS_ORDER},
            "any_vote_disagreement": len(votes) > 1,
            "vote_disagreement_rate": 1.0 - max(votes.values()) / len(MEMBERS),
        }
        for family, values in families.items():
            prediction = "Yes" if values["p_yes"] >= values["p_no"] else "No"
            row[f"{family}_ensemble_p_yes"] = values["p_yes"]
            row[f"{family}_ensemble_p_no"] = values["p_no"]
            row[f"{family}_ensemble_prediction"] = prediction
            row[f"{family}_ensemble_correct"] = prediction == anchor["ground_truth"]
            row[f"{family}_ensemble_confidence"] = values["confidence"]
            row[f"{family}_predictive_entropy"] = values["predictive_entropy"]
            row[f"{family}_expected_entropy"] = values["expected_entropy"]
            row[f"{family}_mutual_information_style"] = values["mutual_information_style"]
            row[f"{family}_member_probability_variance"] = values["member_probability_variance"]
        row["row_sha256"] = sha256_text(canonical_json(row))
        outputs.append(row)
    require(len({row["ensemble_id"] for row in outputs}) == len(outputs), "duplicate ensemble ID")
    return outputs


def expected_calibration_error(rows: Sequence[Mapping[str, Any]], family: str, bins: int = 10) -> float:
    require(family in ("raw", "calibrated") and bins >= 2 and rows, "invalid ECE input")
    total = len(rows)
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [row for row in rows if lower < float(row[f"{family}_ensemble_confidence"]) <= upper]
        if members:
            accuracy = sum(bool(row[f"{family}_ensemble_correct"]) for row in members) / len(members)
            confidence = sum(float(row[f"{family}_ensemble_confidence"]) for row in members) / len(members)
            result += len(members) / total * abs(accuracy - confidence)
    return result


def error_detection_auroc(rows: Sequence[Mapping[str, Any]], family: str) -> float | None:
    require(family in ("raw", "calibrated") and rows, "invalid AUROC input")
    positives = [row for row in rows if not row[f"{family}_ensemble_correct"]]
    negatives = [row for row in rows if row[f"{family}_ensemble_correct"]]
    if not positives or not negatives:
        return None
    score = 0.0
    for positive in positives:
        positive_score = 1.0 - float(positive[f"{family}_ensemble_confidence"])
        for negative in negatives:
            negative_score = 1.0 - float(negative[f"{family}_ensemble_confidence"])
            score += 1.0 if positive_score > negative_score else 0.5 if positive_score == negative_score else 0.0
    return score / (len(positives) * len(negatives))


def summarize(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    require(family in ("raw", "calibrated") and rows, "invalid metric input")
    count = len(rows)
    nll = 0.0
    brier = 0.0
    for row in rows:
        target = 1.0 if row["ground_truth"] == "Yes" else 0.0
        p_yes = min(max(float(row[f"{family}_ensemble_p_yes"]), 1e-12), 1.0 - 1e-12)
        nll -= target * math.log(p_yes) + (1.0 - target) * math.log(1.0 - p_yes)
        brier += (p_yes - target) ** 2
    return {
        "rows": count,
        "accuracy": sum(bool(row[f"{family}_ensemble_correct"]) for row in rows) / count,
        "errors": sum(not bool(row[f"{family}_ensemble_correct"]) for row in rows),
        "nll": nll / count,
        "brier": brier / count,
        "ece_10_bin": expected_calibration_error(rows, family),
        "mean_confidence": sum(float(row[f"{family}_ensemble_confidence"]) for row in rows) / count,
        "mean_predictive_entropy": sum(float(row[f"{family}_predictive_entropy"]) for row in rows) / count,
        "mean_expected_entropy": sum(float(row[f"{family}_expected_entropy"]) for row in rows) / count,
        "mean_mutual_information_style": sum(float(row[f"{family}_mutual_information_style"]) for row in rows) / count,
        "mean_member_probability_variance": sum(float(row[f"{family}_member_probability_variance"]) for row in rows) / count,
        "error_detection_auroc": error_detection_auroc(rows, family),
        "prediction_counts": dict(sorted(Counter(row[f"{family}_ensemble_prediction"] for row in rows).items())),
        "vote_disagreement_rows": sum(bool(row["any_vote_disagreement"]) for row in rows),
        "mean_vote_disagreement_rate": sum(float(row["vote_disagreement_rate"]) for row in rows) / count,
    }


def build_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 2400, "ensemble metrics require 2,400 rows")
    require(len({row["input_id"] for row in rows}) == 400, "ensemble input cardinality drift")
    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    require(all(len(group) == 400 for group in groups.values()), "ensemble condition cardinality drift")
    original = {row["input_id"]: row for row in groups["original"]}
    paired_changes: dict[str, Any] = {}
    for condition, group in groups.items():
        pairs = [(original[row["input_id"]], row) for row in group]
        paired_changes[condition] = {}
        for family in ("raw", "calibrated"):
            paired_changes[condition][family] = {
                "mean_confidence_change_from_original": sum(float(b[f"{family}_ensemble_confidence"]) - float(a[f"{family}_ensemble_confidence"]) for a, b in pairs) / len(pairs),
                "mean_predictive_entropy_change_from_original": sum(float(b[f"{family}_predictive_entropy"]) - float(a[f"{family}_predictive_entropy"]) for a, b in pairs) / len(pairs),
                "mean_mutual_information_style_change_from_original": sum(float(b[f"{family}_mutual_information_style"]) - float(a[f"{family}_mutual_information_style"]) for a, b in pairs) / len(pairs),
                "mean_member_probability_variance_change_from_original": sum(float(b[f"{family}_member_probability_variance"]) - float(a[f"{family}_member_probability_variance"]) for a, b in pairs) / len(pairs),
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "method": "three_lora_ensemble",
        "members": list(MEMBERS),
        "rows": 2400,
        "inputs": 400,
        "conditions": list(CONDITIONS),
        "raw": {"overall": summarize(rows, "raw"), "by_condition": {condition: summarize(group, "raw") for condition, group in groups.items()}},
        "calibrated": {"overall": summarize(rows, "calibrated"), "by_condition": {condition: summarize(group, "calibrated") for condition, group in groups.items()}},
        "paired_changes": paired_changes,
        "aggregation": "arithmetic-mean-of-member-binary-probabilities",
        "variance_convention": "population-variance-across-three-members",
        "mutual_information_claim": "finite-ensemble-entropy-Jensen-gap; members are not posterior samples",
        "test_labels_used_for_fitting_or_tuning": False,
    }


def risk_coverage_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    scopes = {"overall": list(rows), **{condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}}
    for scope, members in scopes.items():
        for family in ("raw", "calibrated"):
            ordered = sorted(members, key=lambda row: (-float(row[f"{family}_ensemble_confidence"]), row["ensemble_id"]))
            errors = 0
            for rank, row in enumerate(ordered, start=1):
                errors += not bool(row[f"{family}_ensemble_correct"])
                outputs.append({
                    "schema_version": SCHEMA_VERSION,
                    "protocol_version": PROTOCOL_VERSION,
                    "method": "three_lora_ensemble",
                    "scope": scope,
                    "probability_type": family,
                    "retained": rank,
                    "total": len(ordered),
                    "coverage": rank / len(ordered),
                    "risk": errors / rank,
                    "selective_accuracy": 1.0 - errors / rank,
                    "confidence_threshold": float(row[f"{family}_ensemble_confidence"]),
                })
    return outputs


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)
