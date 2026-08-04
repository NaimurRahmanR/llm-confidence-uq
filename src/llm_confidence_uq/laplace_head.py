"""Diagonal-Laplace Bayesian linear head over frozen LLM representations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-frozen-qwen-diagonal-laplace-head-v1"
CONDITIONS = (
    "original",
    "lexical_evidence_removal",
    "prefix_truncation_50",
    "irrelevant_distractor",
    "lexical_contradiction",
    "no_passage",
)


class LaplaceHeadError(RuntimeError):
    """Raised when the Bayesian-head protocol fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaplaceHeadError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class FrozenPromptExample:
    ordinal: int
    example_id: str
    source_index: int
    condition: str
    condition_index: int
    prompt_token_ids: tuple[int, ...]
    target: int
    source_row_sha256: str

    def __post_init__(self) -> None:
        require(self.ordinal >= 0, "negative example ordinal")
        require(bool(self.example_id), "empty example ID")
        require(self.source_index >= 0, "negative source index")
        require(self.condition in CONDITIONS, "unknown condition")
        require(self.condition_index == CONDITIONS.index(self.condition), "condition-index drift")
        require(bool(self.prompt_token_ids) and all(type(value) is int and value >= 0 for value in self.prompt_token_ids), "invalid prompt token IDs")
        require(self.target in (0, 1), "invalid binary target")
        require(isinstance(self.source_row_sha256, str) and len(self.source_row_sha256) == 64, "invalid source-row hash")


class FrozenPromptDataset(Dataset[FrozenPromptExample]):
    def __init__(self, examples: Sequence[FrozenPromptExample]) -> None:
        self._examples = tuple(examples)
        require(bool(self._examples), "empty frozen-prompt dataset")
        require(tuple(item.ordinal for item in self._examples) == tuple(range(len(self._examples))), "non-contiguous example ordinals")
        require(len({(item.example_id, item.condition) for item in self._examples}) == len(self._examples), "duplicate example identity")

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> FrozenPromptExample:
        return self._examples[index]


class FrozenPromptCollator:
    """Left-pad prompts without exposing labels to the frozen transformer."""

    def __init__(self, pad_token_id: int) -> None:
        require(type(pad_token_id) is int and pad_token_id >= 0, "invalid pad token ID")
        self.pad_token_id = pad_token_id

    def __call__(self, examples: Sequence[FrozenPromptExample]) -> dict[str, Any]:
        require(bool(examples), "empty representation batch")
        width = max(len(item.prompt_token_ids) for item in examples)
        input_ids = torch.full((len(examples), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
        for row_index, item in enumerate(examples):
            length = len(item.prompt_token_ids)
            input_ids[row_index, width - length :] = torch.tensor(item.prompt_token_ids, dtype=torch.long)
            attention_mask[row_index, width - length :] = 1
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "targets": torch.tensor([item.target for item in examples], dtype=torch.float64),
            "examples": tuple(examples),
        }


def gather_last_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    require(last_hidden_state.ndim == 3 and attention_mask.ndim == 2, "invalid hidden-state or mask rank")
    require(last_hidden_state.shape[:2] == attention_mask.shape, "hidden-state/mask shape drift")
    require(bool(torch.all(attention_mask.sum(dim=1) > 0).item()), "empty attended sequence")
    indices = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
    last = indices.masked_fill(attention_mask == 0, -1).max(dim=1).values
    result = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last.device), last]
    require(bool(torch.isfinite(result).all().item()), "non-finite frozen representation")
    return result


def fit_standardizer(features: torch.Tensor, minimum_scale: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    require(features.ndim == 2 and features.shape[0] >= 2, "invalid training features")
    require(bool(torch.isfinite(features).all().item()), "non-finite training features")
    require(minimum_scale > 0.0, "invalid minimum standardization scale")
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(minimum_scale)
    require(bool(torch.isfinite(mean).all().item()) and bool(torch.isfinite(scale).all().item()), "non-finite standardizer")
    return mean, scale


def apply_standardizer(features: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    require(features.ndim == 2 and mean.ndim == scale.ndim == 1, "invalid standardization tensors")
    require(features.shape[1] == mean.numel() == scale.numel(), "standardization dimension drift")
    require(bool(torch.all(scale > 0).item()), "non-positive standardization scale")
    result = (features - mean) / scale
    require(bool(torch.isfinite(result).all().item()), "non-finite standardized features")
    return result


class BayesianLinearHead(nn.Module):
    """A binary linear prediction head; the transformer remains frozen/non-Bayesian."""

    def __init__(self, feature_dimension: int) -> None:
        super().__init__()
        require(feature_dimension > 0, "invalid feature dimension")
        self.linear = nn.Linear(feature_dimension, 1, bias=True, dtype=torch.float64)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def parameter_vector(head: BayesianLinearHead) -> torch.Tensor:
    return torch.cat((head.linear.weight.reshape(-1), head.linear.bias.reshape(-1)))


def negative_log_posterior(
    head: BayesianLinearHead,
    features: torch.Tensor,
    targets: torch.Tensor,
    prior_precision: float,
) -> torch.Tensor:
    require(features.ndim == 2 and targets.shape == (features.shape[0],), "MAP input shape drift")
    require(prior_precision > 0.0 and math.isfinite(prior_precision), "invalid prior precision")
    logits = head(features)
    likelihood = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
    prior = 0.5 * prior_precision * parameter_vector(head).square().sum()
    result = likelihood + prior
    require(bool(torch.isfinite(result).item()), "non-finite negative log posterior")
    return result


def fit_map(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    prior_precision: float,
    maximum_iterations: int,
) -> tuple[BayesianLinearHead, dict[str, Any]]:
    require(maximum_iterations > 0, "invalid MAP iteration limit")
    features = features.detach().to(device="cpu", dtype=torch.float64)
    targets = targets.detach().to(device="cpu", dtype=torch.float64)
    require(bool(torch.all((targets == 0) | (targets == 1)).item()), "non-binary MAP target")
    require(len(torch.unique(targets)) == 2, "MAP training requires both classes")
    head = BayesianLinearHead(features.shape[1])
    initial_vector = parameter_vector(head).detach().clone()
    initial_loss = float(negative_log_posterior(head, features, targets, prior_precision).item())
    optimizer = torch.optim.LBFGS(
        head.parameters(),
        lr=1.0,
        max_iter=maximum_iterations,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0
    gradients_seen = {name: False for name, _ in head.named_parameters()}

    def closure() -> torch.Tensor:
        nonlocal closure_calls
        optimizer.zero_grad(set_to_none=True)
        loss = negative_log_posterior(head, features, targets, prior_precision)
        loss.backward()
        closure_calls += 1
        for name, parameter in head.named_parameters():
            if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item()):
                gradients_seen[name] = True
        return loss

    optimizer.step(closure)
    final_loss = float(negative_log_posterior(head, features, targets, prior_precision).item())
    final_vector = parameter_vector(head).detach()
    require(all(gradients_seen.values()), "not all MAP parameters received finite gradients")
    require(bool(torch.isfinite(final_vector).all().item()), "non-finite MAP parameter")
    require(not torch.equal(initial_vector, final_vector), "MAP parameters did not change")
    require(final_loss < initial_loss, "MAP objective did not decrease")
    return head, {
        "initial_negative_log_posterior": initial_loss,
        "final_negative_log_posterior": final_loss,
        "closure_calls": closure_calls,
        "all_parameters_received_finite_gradients": all(gradients_seen.values()),
        "parameters_changed": True,
        "initial_parameter_sha256": tensor_sha256(initial_vector),
        "map_parameter_sha256": tensor_sha256(final_vector),
    }


def diagonal_laplace(
    head: BayesianLinearHead,
    standardized_training_features: torch.Tensor,
    *,
    prior_precision: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = standardized_training_features.detach().to(device="cpu", dtype=torch.float64)
    require(features.ndim == 2 and features.shape[1] == head.linear.in_features, "curvature feature drift")
    augmented = torch.cat((features, torch.ones((features.shape[0], 1), dtype=torch.float64)), dim=1)
    with torch.no_grad():
        probabilities = torch.sigmoid(head(features))
        weights = probabilities * (1.0 - probabilities)
        precision = prior_precision + (weights[:, None] * augmented.square()).sum(dim=0)
        variance = precision.reciprocal()
    require(bool(torch.isfinite(precision).all().item()) and bool(torch.all(precision > 0).item()), "invalid diagonal posterior precision")
    require(bool(torch.isfinite(variance).all().item()) and bool(torch.all(variance >= 0).item()), "invalid diagonal posterior variance")
    return precision, variance


def posterior_predictive(
    standardized_features: torch.Tensor,
    map_parameters: torch.Tensor,
    posterior_variance: torch.Tensor,
    *,
    samples: int,
    seed: int,
    chunk_size: int = 128,
) -> dict[str, torch.Tensor]:
    features = standardized_features.detach().to(device="cpu", dtype=torch.float64)
    parameters = map_parameters.detach().to(device="cpu", dtype=torch.float64)
    variance = posterior_variance.detach().to(device="cpu", dtype=torch.float64)
    require(features.ndim == 2 and parameters.shape == variance.shape == (features.shape[1] + 1,), "posterior-predictive dimension drift")
    require(samples >= 2 and chunk_size > 0, "invalid posterior sampling contract")
    require(bool(torch.isfinite(variance).all().item()) and bool(torch.all(variance >= 0).item()), "invalid posterior variance")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    draws = parameters + torch.randn((samples, parameters.numel()), generator=generator, dtype=torch.float64) * variance.sqrt()
    augmented = torch.cat((features, torch.ones((features.shape[0], 1), dtype=torch.float64)), dim=1)
    probability_chunks = []
    for start in range(0, features.shape[0], chunk_size):
        logits = augmented[start : start + chunk_size] @ draws.T
        probability_chunks.append(torch.sigmoid(logits))
    probabilities = torch.cat(probability_chunks, dim=0)
    mean = probabilities.mean(dim=1)
    predictive_variance = probabilities.var(dim=1, unbiased=False)
    epsilon = torch.finfo(torch.float64).eps
    safe_mean = mean.clamp(epsilon, 1.0 - epsilon)
    predictive_entropy = -(safe_mean * safe_mean.log() + (1.0 - safe_mean) * (1.0 - safe_mean).log())
    safe_samples = probabilities.clamp(epsilon, 1.0 - epsilon)
    sample_entropies = -(safe_samples * safe_samples.log() + (1.0 - safe_samples) * (1.0 - safe_samples).log())
    expected_entropy = sample_entropies.mean(dim=1)
    information = (predictive_entropy - expected_entropy).clamp_min(0.0)
    for name, tensor in {
        "mean": mean,
        "variance": predictive_variance,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": information,
    }.items():
        require(bool(torch.isfinite(tensor).all().item()), f"non-finite posterior-predictive {name}")
    require(bool(torch.all(predictive_variance >= 0).item()), "negative predictive variance")
    return {
        "mean_p_yes": mean,
        "variance": predictive_variance,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": information,
    }


def checkpoint_payload(
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    map_parameters: torch.Tensor,
    posterior_precision: torch.Tensor,
    posterior_variance: torch.Tensor,
    prior_precision: float,
    training_report: Mapping[str, Any],
) -> dict[str, Any]:
    size = feature_mean.numel()
    require(feature_scale.shape == feature_mean.shape == (size,), "checkpoint standardizer drift")
    require(map_parameters.shape == posterior_precision.shape == posterior_variance.shape == (size + 1,), "checkpoint posterior drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "claim": "Laplace-approximated Bayesian prediction head over frozen LLM representations",
        "full_transformer_is_bayesian": False,
        "feature_dimension": size,
        "prior_precision": prior_precision,
        "feature_mean": feature_mean.detach().cpu().double().tolist(),
        "feature_scale": feature_scale.detach().cpu().double().tolist(),
        "map_parameters": map_parameters.detach().cpu().double().tolist(),
        "posterior_precision_diagonal": posterior_precision.detach().cpu().double().tolist(),
        "posterior_variance_diagonal": posterior_variance.detach().cpu().double().tolist(),
        "training_report": dict(training_report),
    }


def checkpoint_tensors(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    require(payload.get("protocol_version") == PROTOCOL_VERSION, "checkpoint protocol drift")
    require(payload.get("full_transformer_is_bayesian") is False, "checkpoint overclaims Bayesian scope")
    dimension = int(payload["feature_dimension"])
    result = {
        "feature_mean": torch.tensor(payload["feature_mean"], dtype=torch.float64),
        "feature_scale": torch.tensor(payload["feature_scale"], dtype=torch.float64),
        "map_parameters": torch.tensor(payload["map_parameters"], dtype=torch.float64),
        "posterior_precision": torch.tensor(payload["posterior_precision_diagonal"], dtype=torch.float64),
        "posterior_variance": torch.tensor(payload["posterior_variance_diagonal"], dtype=torch.float64),
    }
    require(result["feature_mean"].shape == result["feature_scale"].shape == (dimension,), "checkpoint feature dimension drift")
    require(result["map_parameters"].shape == result["posterior_precision"].shape == result["posterior_variance"].shape == (dimension + 1,), "checkpoint parameter dimension drift")
    require(bool(torch.all(result["feature_scale"] > 0).item()), "checkpoint scale is non-positive")
    require(bool(torch.all(result["posterior_precision"] > 0).item()), "checkpoint precision is non-positive")
    require(torch.allclose(result["posterior_precision"].reciprocal(), result["posterior_variance"], rtol=1e-12, atol=1e-12), "checkpoint precision/variance mismatch")
    return result


def expected_calibration_error(rows: Sequence[Mapping[str, Any]], bins: int = 10) -> float:
    require(rows and bins >= 2, "invalid ECE input")
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [row for row in rows if lower < float(row["confidence"]) <= upper]
        if members:
            accuracy = sum(bool(row["correct"]) for row in members) / len(members)
            confidence = sum(float(row["confidence"]) for row in members) / len(members)
            result += len(members) / len(rows) * abs(accuracy - confidence)
    return result


def error_detection_auroc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    positives = [row for row in rows if not row["correct"]]
    negatives = [row for row in rows if row["correct"]]
    if not positives or not negatives:
        return None
    total = 0.0
    for positive in positives:
        p_score = 1.0 - float(positive["confidence"])
        for negative in negatives:
            n_score = 1.0 - float(negative["confidence"])
            total += 1.0 if p_score > n_score else 0.5 if p_score == n_score else 0.0
    return total / (len(positives) * len(negatives))


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(rows, "empty Bayesian-head metric input")
    nll = 0.0
    brier = 0.0
    for row in rows:
        target = 1.0 if row["ground_truth"] == "Yes" else 0.0
        probability = min(max(float(row["posterior_mean_p_yes"]), 1e-12), 1.0 - 1e-12)
        nll -= target * math.log(probability) + (1.0 - target) * math.log(1.0 - probability)
        brier += (probability - target) ** 2
    count = len(rows)
    return {
        "rows": count,
        "accuracy": sum(bool(row["correct"]) for row in rows) / count,
        "errors": sum(not bool(row["correct"]) for row in rows),
        "nll": nll / count,
        "brier": brier / count,
        "ece_10_bin": expected_calibration_error(rows),
        "mean_confidence": sum(float(row["confidence"]) for row in rows) / count,
        "mean_predictive_entropy": sum(float(row["predictive_entropy"]) for row in rows) / count,
        "mean_expected_entropy": sum(float(row["expected_entropy"]) for row in rows) / count,
        "mean_mutual_information": sum(float(row["mutual_information"]) for row in rows) / count,
        "mean_posterior_predictive_variance": sum(float(row["posterior_predictive_variance"]) for row in rows) / count,
        "error_detection_auroc": error_detection_auroc(rows),
        "prediction_counts": dict(sorted(Counter(row["prediction"] for row in rows).items())),
    }


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)
