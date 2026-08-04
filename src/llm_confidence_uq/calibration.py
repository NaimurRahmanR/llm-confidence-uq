from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-temperature-scaling-v1"
CLASS_ORDER = ("Yes", "No")
METHODS = ("baseline", "full_seed_1", "full_seed_2", "full_seed_3")
FORBIDDEN_PREDICTION_FIELDS = {
    "answer", "correct", "ground_truth", "label", "target"
}


class TemperatureScalingContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TemperatureScalingContractError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TemperatureFitInputs:
    method: str
    input_ids: tuple[str, ...]
    source_indices: tuple[int, ...]
    logits: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]
    alignment_sha256: str

    def __post_init__(self) -> None:
        size = len(self.input_ids)
        require(self.method in METHODS, "unknown method")
        require(size > 0, "empty temperature-fit inputs")
        require(len(self.source_indices) == size, "source-index length mismatch")
        require(len(self.logits) == size, "logit length mismatch")
        require(len(self.labels) == size, "label length mismatch")
        require(len(set(self.input_ids)) == size, "duplicate input ID")
        require(len(set(self.source_indices)) == size, "duplicate source index")
        require(all(label in (0, 1) for label in self.labels), "invalid class index")
        require(
            all(len(pair) == 2 and all(math.isfinite(x) for x in pair) for pair in self.logits),
            "nonfinite or malformed logits",
        )


@dataclass(frozen=True)
class TemperatureFitResult:
    temperature: float
    initial_nll: float
    final_nll: float
    initial_brier: float
    final_brier: float
    initial_ece: float
    final_ece: float
    optimizer_evaluations: int
    predictions_unchanged: bool
    boundary_hit: bool

    def __post_init__(self) -> None:
        for name in (
            "temperature", "initial_nll", "final_nll", "initial_brier",
            "final_brier", "initial_ece", "final_ece",
        ):
            require(math.isfinite(float(getattr(self, name))), f"nonfinite {name}")
        require(self.temperature > 0.0, "temperature is not positive")
        require(self.optimizer_evaluations > 0, "optimizer did not evaluate the objective")
        require(self.predictions_unchanged, "positive scaling changed class predictions")


def build_fit_inputs(
    manifest_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
) -> TemperatureFitInputs:
    require(method in METHODS, "unknown method")
    require(len(manifest_rows) == 200, "temperature fitting requires 200 calibration rows")
    require(len(prediction_rows) == 200, "temperature fitting requires 200 predictions")

    input_ids: list[str] = []
    source_indices: list[int] = []
    logits: list[tuple[float, float]] = []
    labels: list[int] = []
    alignment: list[dict[str, Any]] = []

    for ordinal, (manifest, prediction) in enumerate(zip(manifest_rows, prediction_rows)):
        require(manifest.get("research_split") == "calibration", "non-calibration label entered fitting")
        require(manifest.get("source_split") == "train", "calibration label is not from official train")
        require(manifest.get("answer") in (False, True), "invalid calibration label")
        require(FORBIDDEN_PREDICTION_FIELDS.isdisjoint(prediction), "prediction contains a label-bearing field")
        require(prediction.get("method") == method, "prediction method mismatch")
        require(prediction.get("source_split") == "train", "prediction source-split mismatch")
        require(prediction.get("evidence_condition") == "original", "degraded evidence entered temperature fitting")
        require(prediction.get("class_order") == list(CLASS_ORDER), "class-order drift")
        require(prediction.get("inference_status", "ok") == "ok", "failed inference row entered fitting")

        source_index = int(manifest["source_index"])
        require(int(prediction["source_index"]) == source_index, "manifest/prediction order mismatch")
        input_id = prediction.get("input_id")
        require(isinstance(input_id, str) and input_id.startswith("boolqinput-"), "invalid input ID")
        yes_logit = float(prediction["yes_logit"])
        no_logit = float(prediction["no_logit"])
        require(math.isfinite(yes_logit) and math.isfinite(no_logit), "nonfinite logits")

        label = 0 if manifest["answer"] is True else 1
        input_ids.append(input_id)
        source_indices.append(source_index)
        logits.append((yes_logit, no_logit))
        labels.append(label)
        alignment.append({
            "input_id": input_id,
            "manifest_record_sha256": manifest["record_sha256"],
            "ordinal": ordinal,
            "prediction_row_sha256": prediction["row_sha256"],
            "source_index": source_index,
        })

    return TemperatureFitInputs(
        method=method,
        input_ids=tuple(input_ids),
        source_indices=tuple(source_indices),
        logits=tuple(logits),
        labels=tuple(labels),
        alignment_sha256=sha256_text(canonical_json(alignment)),
    )


def _metrics(logits: Any, labels: Any, torch_module: Any, *, ece_bins: int) -> tuple[float, float, float]:
    probabilities = torch_module.softmax(logits, dim=1)
    nll = torch_module.nn.functional.cross_entropy(logits, labels)
    targets_yes = (labels == 0).to(dtype=probabilities.dtype)
    brier = torch_module.mean((probabilities[:, 0] - targets_yes) ** 2)
    confidence, prediction = probabilities.max(dim=1)
    correct = (prediction == labels).to(dtype=probabilities.dtype)
    ece = torch_module.zeros((), dtype=probabilities.dtype, device=probabilities.device)
    boundaries = torch_module.linspace(0.0, 1.0, ece_bins + 1, dtype=probabilities.dtype, device=probabilities.device)
    for index in range(ece_bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if bool(mask.any().item()):
            weight = mask.to(dtype=probabilities.dtype).mean()
            ece = ece + weight * torch_module.abs(correct[mask].mean() - confidence[mask].mean())
    return float(nll.item()), float(brier.item()), float(ece.item())


def fit_temperature(
    inputs: TemperatureFitInputs,
    torch_module: Any,
    *,
    minimum_temperature: float = 0.01,
    maximum_temperature: float = 100.0,
    maximum_iterations: int = 100,
    ece_bins: int = 10,
) -> TemperatureFitResult:
    require(0.0 < minimum_temperature < 1.0 < maximum_temperature, "invalid temperature bounds")
    require(maximum_iterations >= 1, "invalid iteration limit")
    require(ece_bins >= 2, "invalid ECE bin count")

    torch_module.manual_seed(0)
    logits = torch_module.tensor(inputs.logits, dtype=torch_module.float64, device="cpu")
    labels = torch_module.tensor(inputs.labels, dtype=torch_module.long, device="cpu")
    require(tuple(logits.shape) == (len(inputs.labels), 2), "logit tensor shape mismatch")

    span = maximum_temperature - minimum_temperature
    initial_fraction = (1.0 - minimum_temperature) / span
    initial_raw = math.log(initial_fraction / (1.0 - initial_fraction))
    raw_temperature = torch_module.nn.Parameter(
        torch_module.tensor(initial_raw, dtype=torch_module.float64, device="cpu")
    )

    def temperature_tensor() -> Any:
        return minimum_temperature + span * torch_module.sigmoid(raw_temperature)

    initial_nll, initial_brier, initial_ece = _metrics(logits, labels, torch_module, ece_bins=ece_bins)
    optimizer = torch_module.optim.LBFGS(
        [raw_temperature],
        lr=1.0,
        max_iter=maximum_iterations,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure() -> Any:
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = torch_module.nn.functional.cross_entropy(logits / temperature_tensor(), labels)
        require(bool(torch_module.isfinite(loss).item()), "nonfinite calibration loss")
        loss.backward()
        evaluations += 1
        return loss

    optimizer.step(closure)

    with torch_module.no_grad():
        temperature = float(temperature_tensor().item())
        calibrated_logits = logits / temperature
        final_nll, final_brier, final_ece = _metrics(
            calibrated_logits, labels, torch_module, ece_bins=ece_bins
        )
        unchanged = bool(torch_module.equal(logits.argmax(dim=1), calibrated_logits.argmax(dim=1)))

    require(final_nll <= initial_nll + 1e-10, "temperature optimization increased NLL")
    margin = (maximum_temperature - minimum_temperature) * 1e-6
    boundary_hit = temperature <= minimum_temperature + margin or temperature >= maximum_temperature - margin
    return TemperatureFitResult(
        temperature=temperature,
        initial_nll=initial_nll,
        final_nll=final_nll,
        initial_brier=initial_brier,
        final_brier=final_brier,
        initial_ece=initial_ece,
        final_ece=final_ece,
        optimizer_evaluations=evaluations,
        predictions_unchanged=unchanged,
        boundary_hit=boundary_hit,
    )


def make_temperature_artifact(
    inputs: TemperatureFitInputs,
    result: TemperatureFitResult,
    *,
    calibration_manifest_sha256: str,
    prediction_jsonl_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "method": inputs.method,
        "class_order": list(CLASS_ORDER),
        "temperature": result.temperature,
        "parameterization": "bounded-sigmoid-positive-temperature-v1",
        "objective": "mean-cross-entropy-on-calibration-logits-v1",
        "optimizer": "torch.optim.LBFGS-strong-wolfe-float64-cpu-v1",
        "optimizer_evaluations": result.optimizer_evaluations,
        "rows": len(inputs.labels),
        "source_split": "train",
        "research_split": "calibration",
        "evidence_condition": "original",
        "class_counts": {
            "No": sum(label == 1 for label in inputs.labels),
            "Yes": sum(label == 0 for label in inputs.labels),
        },
        "initial_nll": result.initial_nll,
        "final_nll": result.final_nll,
        "initial_brier": result.initial_brier,
        "final_brier": result.final_brier,
        "initial_ece_10_bin": result.initial_ece,
        "final_ece_10_bin": result.final_ece,
        "predictions_unchanged": result.predictions_unchanged,
        "boundary_hit": result.boundary_hit,
        "labels_used": "calibration-only",
        "test_labels_used": False,
        "expressed_confidence_used": False,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "prediction_jsonl_sha256": prediction_jsonl_sha256,
        "alignment_sha256": inputs.alignment_sha256,
        "config_sha256": config_sha256,
    }
    artifact["artifact_id"] = "boolqtemp-" + sha256_text(canonical_json(artifact))
    return artifact
