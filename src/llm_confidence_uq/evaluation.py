from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-calibrated-test-evaluation-v1"
METHODS = ("baseline", "full_seed_1", "full_seed_2", "full_seed_3")
CONDITIONS = (
    "original",
    "lexical_evidence_removal",
    "prefix_truncation_50",
    "irrelevant_distractor",
    "lexical_contradiction",
    "no_passage",
)
CLASS_ORDER = ("Yes", "No")
FORBIDDEN_PREDICTION_FIELDS = {"answer", "correct", "ground_truth", "label", "target"}


class EvaluationContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationContractError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _binary_probability(yes_logit: float, no_logit: float, temperature: float) -> tuple[float, float]:
    require(math.isfinite(yes_logit) and math.isfinite(no_logit), "nonfinite logits")
    require(math.isfinite(temperature) and temperature > 0.0, "temperature must be finite and positive")
    margin = (yes_logit - no_logit) / temperature
    if margin >= 0.0:
        p_no = math.exp(-margin) / (1.0 + math.exp(-margin))
        p_yes = 1.0 - p_no
    else:
        p_yes = math.exp(margin) / (1.0 + math.exp(margin))
        p_no = 1.0 - p_yes
    return p_yes, p_no


def _entropy(p_yes: float, p_no: float) -> float:
    result = 0.0
    for probability in (p_yes, p_no):
        if probability > 0.0:
            result -= probability * math.log(probability)
    return result


def build_evaluation_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    temperature: float,
    temperature_artifact_sha256: str,
) -> list[dict[str, Any]]:
    require(method in METHODS, "unknown method")
    require(len(manifest_rows) == 400, "test manifest must contain 400 rows")
    require(len(prediction_rows) == 2400, "prediction file must contain 2,400 rows")
    require(math.isfinite(temperature) and temperature > 0.0, "invalid frozen temperature")
    require(isinstance(temperature_artifact_sha256, str) and len(temperature_artifact_sha256) == 64, "invalid temperature artifact hash")

    for manifest in manifest_rows:
        require(manifest.get("research_split") == "test", "non-test label entered evaluation")
        require(manifest.get("source_split") == "validation", "test label is not from official validation")
        require(type(manifest.get("answer")) is bool, "invalid test label")
    require(len({int(row["source_index"]) for row in manifest_rows}) == 400, "duplicate test source index")
    manifest_by_source_index = {int(row["source_index"]): row for row in manifest_rows}

    rows: list[dict[str, Any]] = []
    seen_input_ids: set[str] = set()
    active_input_id: str | None = None
    active_source_index: int | None = None
    for ordinal, prediction in enumerate(prediction_rows):
        _, condition_index = divmod(ordinal, len(CONDITIONS))
        condition = CONDITIONS[condition_index]
        require(FORBIDDEN_PREDICTION_FIELDS.isdisjoint(prediction), "label-bearing field found in inference predictions")
        require(prediction.get("source_split") == "validation", "prediction source-split drift")
        source_index = int(prediction.get("source_index"))
        require(source_index in manifest_by_source_index, "prediction source index absent from test manifest")
        manifest = manifest_by_source_index[source_index]
        require(prediction.get("condition") == condition, "condition order drift")
        require(int(prediction.get("condition_index")) == condition_index, "condition-index drift")
        require(prediction.get("class_order") == list(CLASS_ORDER), "class-order drift")
        require(prediction.get("inference_status") == "ok", "failed inference row entered evaluation")
        if condition_index == 0:
            active_input_id = str(prediction["input_id"])
            active_source_index = source_index
            require(active_input_id not in seen_input_ids, "duplicate prediction input group")
            seen_input_ids.add(active_input_id)
        else:
            require(prediction["input_id"] == active_input_id, "prediction input group is not contiguous")
            require(source_index == active_source_index, "prediction source group is not contiguous")
        if method == "baseline":
            require(prediction.get("adapter_stage") is None or "adapter_stage" not in prediction, "baseline row has adapter identity")
            source_row_sha256 = prediction["row_sha256"]
        else:
            require(prediction.get("adapter_stage") == method, "adapter identity drift")
            source_row_sha256 = prediction["adapter_row_sha256"]

        input_id = prediction["input_id"]
        yes_logit = float(prediction["yes_logit"])
        no_logit = float(prediction["no_logit"])
        raw_yes, raw_no = _binary_probability(yes_logit, no_logit, 1.0)
        calibrated_yes, calibrated_no = _binary_probability(yes_logit, no_logit, temperature)
        require(abs(raw_yes - float(prediction["p_yes_binary"])) <= 1e-5, "stored raw Yes probability drift")
        require(abs(raw_no - float(prediction["p_no_binary"])) <= 1e-5, "stored raw No probability drift")
        token_prediction = "Yes" if yes_logit >= no_logit else "No"
        require(prediction["token_prediction"] == token_prediction, "token prediction/logit disagreement")
        ground_truth = "Yes" if manifest["answer"] else "No"
        correct = token_prediction == ground_truth
        raw_confidence = max(raw_yes, raw_no)
        calibrated_confidence = max(calibrated_yes, calibrated_no)
        expressed_valid = bool(prediction["expressed_parser_valid"])
        expressed = prediction["expressed_confidence"] if expressed_valid else None
        require(expressed is None or type(expressed) is int and 0 <= expressed <= 100, "invalid expressed confidence")

        identity = {
            "input_id": input_id,
            "condition": condition,
            "method": method,
            "protocol_version": PROTOCOL_VERSION,
            "source_prediction_row_sha256": source_row_sha256,
            "temperature_artifact_sha256": temperature_artifact_sha256,
        }
        row = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "evaluation_id": "boolqeval-" + sha256_text(canonical_json(identity)),
            "method": method,
            "input_id": input_id,
            "source_index": int(manifest["source_index"]),
            "condition": condition,
            "condition_index": condition_index,
            "class_order": list(CLASS_ORDER),
            "ground_truth": ground_truth,
            "token_prediction": token_prediction,
            "correct": correct,
            "yes_logit": yes_logit,
            "no_logit": no_logit,
            "raw_p_yes": raw_yes,
            "raw_p_no": raw_no,
            "calibrated_p_yes": calibrated_yes,
            "calibrated_p_no": calibrated_no,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
            "raw_entropy": _entropy(raw_yes, raw_no),
            "calibrated_entropy": _entropy(calibrated_yes, calibrated_no),
            "temperature": temperature,
            "temperature_artifact_sha256": temperature_artifact_sha256,
            "source_prediction_row_sha256": source_row_sha256,
            "expressed_parser_valid": expressed_valid,
            "expressed_parser_reason_code": prediction["expressed_parser_reason_code"],
            "expressed_confidence": expressed,
            "expressed_raw_absolute_divergence": abs(expressed / 100.0 - raw_confidence) if expressed is not None else None,
            "expressed_calibrated_absolute_divergence": abs(expressed / 100.0 - calibrated_confidence) if expressed is not None else None,
        }
        row["row_sha256"] = sha256_text(canonical_json(row))
        rows.append(row)

    require(len(seen_input_ids) == 400, "prediction input-group cardinality drift")
    require(len({row["evaluation_id"] for row in rows}) == 2400, "duplicate evaluation ID")
    return rows


def expected_calibration_error(rows: Sequence[Mapping[str, Any]], probability_type: str, bins: int = 10) -> float:
    require(probability_type in ("raw", "calibrated"), "unknown probability type")
    require(bins >= 2 and len(rows) > 0, "invalid ECE input")
    confidence_key = f"{probability_type}_confidence"
    total = len(rows)
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [row for row in rows if lower < float(row[confidence_key]) <= upper]
        if members:
            accuracy = sum(bool(row["correct"]) for row in members) / len(members)
            confidence = sum(float(row[confidence_key]) for row in members) / len(members)
            ece += len(members) / total * abs(accuracy - confidence)
    return ece


def error_detection_auroc(rows: Sequence[Mapping[str, Any]], probability_type: str) -> float | None:
    require(probability_type in ("raw", "calibrated"), "unknown probability type")
    values = [(1.0 - float(row[f"{probability_type}_confidence"]), not bool(row["correct"])) for row in rows]
    positives = sum(label for _, label in values)
    negatives = len(values) - positives
    if positives == 0 or negatives == 0:
        return None
    favourable = 0.0
    positive_scores = [score for score, label in values if label]
    negative_scores = [score for score, label in values if not label]
    for positive in positive_scores:
        for negative in negative_scores:
            favourable += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return favourable / (positives * negatives)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) > 0, "empty evaluation rows")
    epsilon = 1e-15
    count = len(rows)
    raw_nll = 0.0
    calibrated_nll = 0.0
    raw_brier = 0.0
    calibrated_brier = 0.0
    for row in rows:
        true_yes = row["ground_truth"] == "Yes"
        raw_true = float(row["raw_p_yes"] if true_yes else row["raw_p_no"])
        calibrated_true = float(row["calibrated_p_yes"] if true_yes else row["calibrated_p_no"])
        raw_nll -= math.log(max(raw_true, epsilon))
        calibrated_nll -= math.log(max(calibrated_true, epsilon))
        target = 1.0 if true_yes else 0.0
        raw_brier += (float(row["raw_p_yes"]) - target) ** 2
        calibrated_brier += (float(row["calibrated_p_yes"]) - target) ** 2
    valid = [row for row in rows if row["expressed_parser_valid"]]
    return {
        "rows": count,
        "accuracy": sum(bool(row["correct"]) for row in rows) / count,
        "errors": sum(not bool(row["correct"]) for row in rows),
        "ground_truth_counts": dict(sorted(Counter(row["ground_truth"] for row in rows).items())),
        "prediction_counts": dict(sorted(Counter(row["token_prediction"] for row in rows).items())),
        "raw_nll": raw_nll / count,
        "calibrated_nll": calibrated_nll / count,
        "raw_brier": raw_brier / count,
        "calibrated_brier": calibrated_brier / count,
        "raw_ece_10_bin": expected_calibration_error(rows, "raw"),
        "calibrated_ece_10_bin": expected_calibration_error(rows, "calibrated"),
        "raw_mean_confidence": sum(float(row["raw_confidence"]) for row in rows) / count,
        "calibrated_mean_confidence": sum(float(row["calibrated_confidence"]) for row in rows) / count,
        "raw_mean_entropy": sum(float(row["raw_entropy"]) for row in rows) / count,
        "calibrated_mean_entropy": sum(float(row["calibrated_entropy"]) for row in rows) / count,
        "raw_error_detection_auroc": error_detection_auroc(rows, "raw"),
        "calibrated_error_detection_auroc": error_detection_auroc(rows, "calibrated"),
        "expressed_parser_valid_rows": len(valid),
        "expressed_parser_valid_rate": len(valid) / count,
        "expressed_mean_confidence_valid_only": sum(row["expressed_confidence"] for row in valid) / (100.0 * len(valid)) if valid else None,
        "expressed_raw_mean_absolute_divergence_valid_only": sum(row["expressed_raw_absolute_divergence"] for row in valid) / len(valid) if valid else None,
        "expressed_calibrated_mean_absolute_divergence_valid_only": sum(row["expressed_calibrated_absolute_divergence"] for row in valid) / len(valid) if valid else None,
    }


def build_metrics(rows: Sequence[Mapping[str, Any]], *, method: str) -> dict[str, Any]:
    require(len(rows) == 2400, "metrics require 2,400 aligned rows")
    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    require(all(len(group) == 400 for group in groups.values()), "condition cardinality drift")
    original = {row["input_id"]: row for row in groups["original"]}
    deltas = {}
    for condition, group in groups.items():
        paired = [(original[row["input_id"]], row) for row in group]
        deltas[condition] = {
            "raw_mean_confidence_change_from_original": sum(b["raw_confidence"] - a["raw_confidence"] for a, b in paired) / len(paired),
            "calibrated_mean_confidence_change_from_original": sum(b["calibrated_confidence"] - a["calibrated_confidence"] for a, b in paired) / len(paired),
            "raw_mean_entropy_change_from_original": sum(b["raw_entropy"] - a["raw_entropy"] for a, b in paired) / len(paired),
            "calibrated_mean_entropy_change_from_original": sum(b["calibrated_entropy"] - a["calibrated_entropy"] for a, b in paired) / len(paired),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "rows": 2400,
        "inputs": 400,
        "conditions": list(CONDITIONS),
        "temperature": rows[0]["temperature"],
        "temperature_artifact_sha256": rows[0]["temperature_artifact_sha256"],
        "overall": summarize_rows(rows),
        "by_condition": {condition: summarize_rows(group) for condition, group in groups.items()},
        "paired_changes": deltas,
        "predictions_unchanged_by_calibration": all(row["token_prediction"] == ("Yes" if row["calibrated_p_yes"] >= row["calibrated_p_no"] else "No") for row in rows),
        "temperature_refit_on_test": False,
    }


def risk_coverage_rows(rows: Sequence[Mapping[str, Any]], *, method: str) -> list[dict[str, Any]]:
    outputs = []
    scopes = {"overall": list(rows), **{condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}}
    for scope, members in scopes.items():
        for probability_type in ("raw", "calibrated"):
            key = f"{probability_type}_confidence"
            ordered = sorted(members, key=lambda row: (-float(row[key]), row["evaluation_id"]))
            errors = 0
            for rank, row in enumerate(ordered, start=1):
                errors += not bool(row["correct"])
                outputs.append({
                    "schema_version": SCHEMA_VERSION,
                    "protocol_version": PROTOCOL_VERSION,
                    "method": method,
                    "scope": scope,
                    "probability_type": probability_type,
                    "retained": rank,
                    "total": len(ordered),
                    "coverage": rank / len(ordered),
                    "risk": errors / rank,
                    "selective_accuracy": 1.0 - errors / rank,
                    "confidence_threshold": float(row[key]),
                })
    return outputs


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)
