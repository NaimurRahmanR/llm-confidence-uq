#!/usr/bin/env python3
"""Fit and evaluate a diagonal-Laplace head over frozen Qwen representations."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for location in (SRC, SCRIPTS):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from llm_confidence_uq.laplace_head import (  # noqa: E402
    CONDITIONS,
    PROTOCOL_VERSION,
    FrozenPromptCollator,
    FrozenPromptDataset,
    FrozenPromptExample,
    apply_standardizer,
    canonical_json,
    checkpoint_payload,
    checkpoint_tensors,
    diagonal_laplace,
    fit_map,
    fit_standardizer,
    gather_last_hidden,
    jsonl_bytes,
    parameter_vector,
    posterior_predictive,
    sha256_text,
    summarize_rows,
    tensor_sha256,
)


class LaplaceRunError(RuntimeError):
    """Raised when a staged Bayesian-head run fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaplaceRunError(message)


def repo_path(value: str | Path) -> Path:
    candidate = Path(value)
    absolute = candidate if candidate.is_absolute() else REPO / candidate
    absolute = Path(os.path.abspath(absolute))
    root = Path(os.path.abspath(REPO))
    require(absolute == root or root in absolute.parents, f"path escapes repository: {absolute}")
    return absolute


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def read_jsonl(path: Path, expected_sha256: str, expected_rows: int) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"input hash drift: {path.relative_to(REPO)}")
    require(payload.endswith(b"\n") and b"\r" not in payload, f"input newline drift: {path.relative_to(REPO)}")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    require(len(rows) == expected_rows and all(isinstance(row, dict) for row in rows), f"input row-count drift: {path.relative_to(REPO)}")
    return rows


def validate_config(config: Mapping[str, Any], config_payload: bytes) -> None:
    require(config.get("schema_version") == 1 and config.get("protocol_version") == PROTOCOL_VERSION, "configuration protocol drift")
    model = config.get("model")
    require(model == {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "tokenizer_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "tokenizer_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "expected_model_class": "Qwen2Model",
        "expected_tokenizer_class": "Qwen2Tokenizer",
        "hidden_size": 1536,
        "dtype": "bfloat16",
        "transformer_frozen": True,
        "adapter_attached": False,
        "representation": "final_hidden_state_at_last_attended_answer_prefix_token",
    }, "model contract drift")
    require(config["map"]["likelihood"] == "bernoulli-logistic", "likelihood drift")
    require(config["map"]["prior"] == "zero-mean-isotropic-gaussian", "prior drift")
    require(float(config["map"]["prior_precision"]) == 1.0, "prior precision drift")
    require(config["map"]["optimizer"] == "full-batch-lbfgs", "MAP optimizer drift")
    require(config["laplace"]["curvature"] == "exact-logistic-negative-log-posterior-diagonal", "curvature contract drift")
    require(tuple(config["stages"]) == ("smoke", "full"), "stage order drift")
    require(config["stages"]["smoke"]["training_examples"] == 32 and config["stages"]["smoke"]["test_rows"] == 12, "smoke stage drift")
    require(config["stages"]["full"]["training_examples"] == 800 and config["stages"]["full"]["test_rows"] == 2400, "full stage drift")
    require(config["claim_boundaries"] == {
        "full_transformer_is_bayesian": False,
        "transformer_weights_frozen": True,
        "approximate_posterior_scope": "binary_linear_prediction_head_only",
        "diagonal_covariance_ignores_parameter_correlations": True,
        "local_gaussian_approximation": True,
        "test_labels_used_for_fitting_or_tuning": False,
    }, "claim-boundary drift")
    for relative, expected in config["implementation"].items():
        require(expected != "PENDING", f"pending implementation hash: {relative}")
        path = repo_path(relative)
        require(path.is_file() and file_sha256(path) == expected, f"implementation hash drift: {relative}")
    for relative, expected in config["prerequisites"].items():
        path = repo_path(relative)
        require(path.is_file() and file_sha256(path) == expected, f"prerequisite hash drift: {relative}")
    require(config_payload.endswith(b"\n"), "configuration newline drift")


def validate_environment(config: Mapping[str, Any], torch: Any) -> None:
    environment = config["environment"]
    require(sys.version.split()[0] == environment["python"], "Python version drift")
    require(torch.__version__ == environment["pytorch"], "PyTorch version drift")
    require(torch.version.cuda == environment["pytorch_cuda_build"], "PyTorch CUDA build drift")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "CUDA device contract drift")
    require(torch.cuda.get_device_name(0) == environment["gpu_name"], "unexpected GPU")
    require(list(torch.cuda.get_device_capability(0)) == environment["compute_capability"], "compute capability drift")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    require(abs(vram - float(environment["vram_gib"])) <= 0.1, "GPU VRAM drift")
    require(torch.cuda.is_bf16_supported(), "BF16 is unsupported")
    require(importlib.util.find_spec("torchao") is None, "torchao must remain absent")


def runtime_dependencies() -> tuple[Any, ...]:
    import prepare_degradations as preparation
    import run_baseline as baseline
    import train_lora as training
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    import torch

    return preparation, baseline, training, load_dataset, AutoModel, AutoTokenizer, torch


def make_train_examples(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_split: Any,
    tokenizer: Any,
    prompt_spec: Any,
    count: int,
    training: Any,
) -> list[FrozenPromptExample]:
    reconstructed = training.reconstruct_training_examples(rows, source_split, tokenizer, prompt_spec)
    selected = reconstructed[:count]
    require(len(selected) == count and len({item.target_class for item in selected}) == 2, "training stage lacks both classes")
    outputs = []
    for ordinal, item in enumerate(selected):
        manifest = rows[item.ordinal]
        outputs.append(FrozenPromptExample(
            ordinal=ordinal,
            example_id=item.example_id,
            source_index=int(manifest["source_index"]),
            condition="original",
            condition_index=0,
            prompt_token_ids=tuple(item.prompt_token_ids),
            target=1 if item.target_class == "Yes" else 0,
            source_row_sha256=sha256_text(canonical_json(manifest)),
        ))
    return outputs


def make_test_examples(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    degradation_rows: Sequence[Mapping[str, Any]],
    validation_split: Any,
    tokenizer: Any,
    prompt_spec: Any,
    stage: str,
    config: Mapping[str, Any],
    preparation: Any,
    baseline: Any,
) -> tuple[list[FrozenPromptExample], Mapping[str, Any]]:
    baseline.validate_degradation_rows(degradation_rows)
    stage_rows = baseline.select_stage_rows(degradation_rows, "smoke12" if stage == "smoke" else "full")
    selected_inputs, selection_report = preparation.reconstruct_selected_inputs(
        manifest_rows,
        validation_split,
        tokenizer,
        prompt_spec,
        dataset_id=config["data"]["dataset_id"],
        dataset_revision=config["data"]["dataset_revision"],
        selection_seed=20260803,
        eligibility_max_prompt_tokens=768,
    )
    degradation_config = __import__("yaml").safe_load(repo_path("configs/degradations.yaml").read_text(encoding="utf-8"))
    reconstructed = baseline.reconstruct_stage_examples(
        stage_rows=stage_rows,
        selected_inputs=selected_inputs,
        tokenizer=tokenizer,
        prompt_spec=prompt_spec,
        seed=int(degradation_config["protocol"]["seed"]),
        maximum_fragment_tokens=int(degradation_config["distractor"]["maximum_fragment_tokens"]),
    )
    labels = {int(row["source_index"]): 1 if bool(row["answer"]) else 0 for row in manifest_rows}
    outputs = []
    for ordinal, item in enumerate(reconstructed):
        token_ids = tuple(int(value) for value in tokenizer.encode(item.prompt, add_special_tokens=False))
        require(len(token_ids) == item.prompt_tokens, "reconstructed test prompt-token drift")
        outputs.append(FrozenPromptExample(
            ordinal=ordinal,
            example_id=item.input_id,
            source_index=item.source_index,
            condition=item.condition,
            condition_index=item.condition_index,
            prompt_token_ids=token_ids,
            target=labels[item.source_index],
            source_row_sha256=item.degradation_row_sha256,
        ))
    require(len(outputs) == int(config["stages"][stage]["test_rows"]), "test stage cardinality drift")
    return outputs, selection_report


def extract_features(
    examples: Sequence[FrozenPromptExample],
    *,
    model: Any,
    pad_token_id: int,
    batch_size: int,
    torch: Any,
) -> tuple[Any, Any]:
    dataset = FrozenPromptDataset(examples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=FrozenPromptCollator(pad_token_id),
    )
    features = []
    targets = []
    device = torch.device("cuda:0")
    model.eval()
    require(model.training is False, "frozen model is not in evaluation mode")
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            hidden = gather_last_hidden(outputs.last_hidden_state, attention_mask)
            features.append(hidden.detach().float().cpu())
            targets.append(batch["targets"].detach().double().cpu())
    result_features = torch.cat(features, dim=0)
    result_targets = torch.cat(targets, dim=0)
    require(result_features.shape[0] == len(examples), "feature extraction cardinality drift")
    require(bool(torch.isfinite(result_features).all().item()), "non-finite extracted features")
    return result_features, result_targets


def build_prediction_rows(
    examples: Sequence[FrozenPromptExample],
    posterior: Mapping[str, Any],
    *,
    stage: str,
    checkpoint_sha256: str,
) -> list[dict[str, Any]]:
    rows = []
    for index, example in enumerate(examples):
        mean = float(posterior["mean_p_yes"][index].item())
        prediction = "Yes" if mean >= 0.5 else "No"
        truth = "Yes" if example.target == 1 else "No"
        identity = {
            "example_id": example.example_id,
            "condition": example.condition,
            "source_row_sha256": example.source_row_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "protocol_version": PROTOCOL_VERSION,
        }
        row = {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "prediction_id": "boolqlaplace-" + sha256_text(canonical_json(identity)),
            "stage": stage,
            "example_id": example.example_id,
            "source_index": example.source_index,
            "condition": example.condition,
            "condition_index": example.condition_index,
            "source_row_sha256": example.source_row_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "ground_truth": truth,
            "prediction": prediction,
            "correct": prediction == truth,
            "posterior_mean_p_yes": mean,
            "posterior_mean_p_no": 1.0 - mean,
            "confidence": max(mean, 1.0 - mean),
            "posterior_predictive_variance": float(posterior["variance"][index].item()),
            "predictive_entropy": float(posterior["predictive_entropy"][index].item()),
            "expected_entropy": float(posterior["expected_entropy"][index].item()),
            "mutual_information": float(posterior["mutual_information"][index].item()),
            "full_transformer_is_bayesian": False,
        }
        row["row_sha256"] = sha256_text(canonical_json(row))
        rows.append(row)
    require(len({row["prediction_id"] for row in rows}) == len(rows), "duplicate Bayesian-head prediction ID")
    return rows


def build_metrics(rows: Sequence[Mapping[str, Any]], *, stage: str) -> dict[str, Any]:
    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    require(all(groups.values()), "missing Bayesian-head condition")
    original = {row["example_id"]: row for row in groups["original"]}
    changes = {}
    for condition, group in groups.items():
        pairs = [(original[row["example_id"]], row) for row in group]
        changes[condition] = {
            "mean_confidence_change_from_original": sum(float(b["confidence"]) - float(a["confidence"]) for a, b in pairs) / len(pairs),
            "mean_predictive_entropy_change_from_original": sum(float(b["predictive_entropy"]) - float(a["predictive_entropy"]) for a, b in pairs) / len(pairs),
            "mean_mutual_information_change_from_original": sum(float(b["mutual_information"]) - float(a["mutual_information"]) for a, b in pairs) / len(pairs),
            "mean_posterior_predictive_variance_change_from_original": sum(float(b["posterior_predictive_variance"]) - float(a["posterior_predictive_variance"]) for a, b in pairs) / len(pairs),
        }
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "rows": len(rows),
        "inputs": len({row["example_id"] for row in rows}),
        "conditions": list(CONDITIONS),
        "overall": summarize_rows(rows),
        "by_condition": {condition: summarize_rows(group) for condition, group in groups.items()},
        "paired_changes": changes,
        "full_transformer_is_bayesian": False,
        "test_labels_used_for_fitting_or_tuning": False,
    }


def risk_coverage_rows(rows: Sequence[Mapping[str, Any]], *, stage: str) -> list[dict[str, Any]]:
    outputs = []
    scopes = {"overall": list(rows), **{condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}}
    for scope, members in scopes.items():
        ordered = sorted(members, key=lambda row: (-float(row["confidence"]), row["prediction_id"]))
        errors = 0
        for rank, row in enumerate(ordered, start=1):
            errors += not bool(row["correct"])
            outputs.append({
                "schema_version": 1,
                "protocol_version": PROTOCOL_VERSION,
                "stage": stage,
                "scope": scope,
                "retained": rank,
                "total": len(ordered),
                "coverage": rank / len(ordered),
                "risk": errors / rank,
                "selective_accuracy": 1.0 - errors / rank,
                "confidence_threshold": float(row["confidence"]),
            })
    return outputs


def atomic_refuse_or_verify(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, f"refusing to replace differing artifact: {path.relative_to(REPO)}")
        return "verified_existing"
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "created"


def publish(directory: Path, payloads: Mapping[str, bytes]) -> tuple[dict[str, str], dict[str, Any]]:
    ledger = {name: {"bytes": len(payload), "sha256": sha256_bytes(payload)} for name, payload in sorted(payloads.items())}
    complete_payloads = {**payloads, "artifact_hashes.json": canonical_bytes(ledger)}
    complete = canonical_bytes({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "artifact_hashes_sha256": sha256_bytes(complete_payloads["artifact_hashes.json"]), "complete": True})
    statuses = {name: atomic_refuse_or_verify(directory / name, payload) for name, payload in complete_payloads.items()}
    statuses["COMPLETE.json"] = atomic_refuse_or_verify(directory / "COMPLETE.json", complete)
    return statuses, ledger


def preflight(config_path: Path) -> dict[str, Any]:
    """Validate the frozen contract and Bayesian-head math without loading Qwen."""
    import torch
    import yaml

    config_path = repo_path(config_path)
    config_payload = config_path.read_bytes()
    config = yaml.safe_load(config_payload.decode("utf-8"))
    require(isinstance(config, dict), "configuration is not a mapping")
    validate_config(config, config_payload)

    features = torch.tensor(
        [
            [-2.0, -1.0, 0.5],
            [-1.5, -0.5, -0.5],
            [-1.0, -1.5, 1.0],
            [-0.5, -1.0, -1.0],
            [0.5, 1.0, -1.0],
            [1.0, 1.5, 1.0],
            [1.5, 0.5, -0.5],
            [2.0, 1.0, 0.5],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float64)
    mean, scale = fit_standardizer(features, float(config["map"]["minimum_scale"]))
    standardized = apply_standardizer(features, mean, scale)
    head, training_report = fit_map(
        standardized,
        targets,
        prior_precision=float(config["map"]["prior_precision"]),
        maximum_iterations=int(config["map"]["maximum_iterations"]),
    )
    precision, variance = diagonal_laplace(
        head,
        standardized,
        prior_precision=float(config["map"]["prior_precision"]),
    )
    posterior = posterior_predictive(
        standardized,
        parameter_vector(head),
        variance,
        samples=64,
        seed=int(config["laplace"]["sampling_seed"]),
        chunk_size=4,
    )
    repeated = posterior_predictive(
        standardized,
        parameter_vector(head),
        variance,
        samples=64,
        seed=int(config["laplace"]["sampling_seed"]),
        chunk_size=4,
    )
    require(torch.equal(posterior["mean_p_yes"], repeated["mean_p_yes"]), "preflight posterior is nondeterministic")
    require(bool(torch.all(precision > 0).item()), "preflight precision is non-positive")
    require(bool(torch.all(variance >= 0).item()), "preflight variance is negative")
    require(bool(torch.all(posterior["mutual_information"] >= 0).item()), "preflight mutual information is negative")
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "qwen_weights_loaded": False,
        "dataset_loaded": False,
        "repository_files_modified": False,
        "synthetic_examples": int(features.shape[0]),
        "synthetic_feature_dimension": int(features.shape[1]),
        "posterior_parameters": int(variance.numel()),
        "minimum_precision": float(precision.min().item()),
        "maximum_variance": float(variance.max().item()),
        "posterior_deterministic": True,
        "all_mutual_information_nonnegative": True,
        "training": training_report,
    }


def run(stage: str, config_path: Path) -> tuple[dict[str, Any], Path]:
    import yaml

    config_path = repo_path(config_path)
    config_payload = config_path.read_bytes()
    config = yaml.safe_load(config_payload.decode("utf-8"))
    require(isinstance(config, dict), "configuration is not a mapping")
    validate_config(config, config_payload)
    require(stage in config["stages"], "unknown stage")
    preparation, baseline, training, load_dataset, AutoModel, AutoTokenizer, torch = runtime_dependencies()
    validate_environment(config, torch)
    seed = int(config["laplace"]["sampling_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_manifest = read_jsonl(repo_path(config["data"]["train_manifest_path"]), config["data"]["train_manifest_sha256"], 800)
    test_manifest = read_jsonl(repo_path(config["data"]["test_manifest_path"]), config["data"]["test_manifest_sha256"], 400)
    degradation_rows = read_jsonl(repo_path(config["data"]["degradation_manifest_path"]), config["data"]["degradation_manifest_sha256"], 2400)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["tokenizer_id"],
        revision=config["model"]["tokenizer_revision"],
        token=False,
        trust_remote_code=False,
        use_fast=True,
    )
    require(tokenizer.__class__.__name__ == config["model"]["expected_tokenizer_class"], "tokenizer class drift")
    tokenizer.padding_side = "left"
    require(tokenizer.pad_token_id is not None, "tokenizer lacks pad token")
    data_config = yaml.safe_load(repo_path("configs/data.yaml").read_text(encoding="utf-8"))
    degradation_config = yaml.safe_load(repo_path("configs/degradations.yaml").read_text(encoding="utf-8"))
    prompt_spec = preparation.prompt_spec_from_configs(data_config, degradation_config)
    train_source = load_dataset(config["data"]["dataset_id"], revision=config["data"]["dataset_revision"], split="train", verification_mode="all_checks", token=False)
    validation_source = load_dataset(config["data"]["dataset_id"], revision=config["data"]["dataset_revision"], split="validation", verification_mode="all_checks", token=False)
    train_examples = make_train_examples(
        rows=train_manifest,
        source_split=train_source,
        tokenizer=tokenizer,
        prompt_spec=prompt_spec,
        count=int(config["stages"][stage]["training_examples"]),
        training=training,
    )
    test_examples, selection_report = make_test_examples(
        manifest_rows=test_manifest,
        degradation_rows=degradation_rows,
        validation_split=validation_source,
        tokenizer=tokenizer,
        prompt_spec=prompt_spec,
        stage=stage,
        config=config,
        preparation=preparation,
        baseline=baseline,
    )
    del train_source, validation_source

    model = AutoModel.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        token=False,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map=0,
    )
    require(model.__class__.__name__ == config["model"]["expected_model_class"], "frozen model class drift")
    require(int(model.config.hidden_size) == int(config["model"]["hidden_size"]), "hidden-size drift")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(not any(parameter.requires_grad for parameter in model.parameters()), "transformer parameter remains trainable")
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_features, train_targets = extract_features(train_examples, model=model, pad_token_id=tokenizer.pad_token_id, batch_size=int(config["feature_extraction"]["batch_size"]), torch=torch)
    test_features, _ = extract_features(test_examples, model=model, pad_token_id=tokenizer.pad_token_id, batch_size=int(config["feature_extraction"]["batch_size"]), torch=torch)
    torch.cuda.synchronize()
    feature_seconds = time.perf_counter() - started
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
    del model
    torch.cuda.empty_cache()

    feature_mean, feature_scale = fit_standardizer(train_features, float(config["map"]["minimum_scale"]))
    standardized_train = apply_standardizer(train_features.double(), feature_mean.double(), feature_scale.double())
    standardized_test = apply_standardizer(test_features.double(), feature_mean.double(), feature_scale.double())
    head, training_report = fit_map(
        standardized_train,
        train_targets,
        prior_precision=float(config["map"]["prior_precision"]),
        maximum_iterations=int(config["map"]["maximum_iterations"]),
    )
    precision, variance = diagonal_laplace(head, standardized_train, prior_precision=float(config["map"]["prior_precision"]))
    map_parameters = parameter_vector(head).detach().double()
    checkpoint = checkpoint_payload(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        map_parameters=map_parameters,
        posterior_precision=precision,
        posterior_variance=variance,
        prior_precision=float(config["map"]["prior_precision"]),
        training_report=training_report,
    )
    checkpoint_bytes = canonical_bytes(checkpoint)
    checkpoint_sha256 = sha256_bytes(checkpoint_bytes)
    sample_count = int(config["laplace"]["posterior_samples"][stage])
    posterior = posterior_predictive(
        standardized_test,
        map_parameters,
        variance,
        samples=sample_count,
        seed=seed,
        chunk_size=int(config["laplace"]["predictive_chunk_size"]),
    )
    reloaded = checkpoint_tensors(json.loads(checkpoint_bytes))
    reloaded_test = apply_standardizer(test_features.double(), reloaded["feature_mean"], reloaded["feature_scale"])
    repeated = posterior_predictive(reloaded_test, reloaded["map_parameters"], reloaded["posterior_variance"], samples=sample_count, seed=seed, chunk_size=int(config["laplace"]["predictive_chunk_size"]))
    maximum_reload_difference = float((posterior["mean_p_yes"] - repeated["mean_p_yes"]).abs().max().item())
    require(maximum_reload_difference == 0.0, "reloaded Bayesian-head predictions differ")

    predictions = build_prediction_rows(test_examples, posterior, stage=stage, checkpoint_sha256=checkpoint_sha256)
    metrics = build_metrics(predictions, stage=stage)
    curves = risk_coverage_rows(predictions, stage=stage)
    provenance = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "model_class": config["model"]["expected_model_class"],
        "transformer_frozen": True,
        "adapter_attached": False,
        "full_transformer_is_bayesian": False,
        "approximate_posterior_scope": "binary_linear_prediction_head_only",
        "train_examples": len(train_examples),
        "test_rows": len(test_examples),
        "posterior_samples": sample_count,
        "prior_precision": float(config["map"]["prior_precision"]),
        "feature_dimension": int(train_features.shape[1]),
        "train_feature_sha256": tensor_sha256(train_features),
        "test_feature_sha256": tensor_sha256(test_features),
        "feature_extraction_seconds": feature_seconds,
        "peak_allocated_gpu_bytes": peak_gpu_bytes,
        "checkpoint_reload_maximum_absolute_difference": maximum_reload_difference,
        "selection_report": selection_report,
        "config_sha256": sha256_bytes(config_payload),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("datasets", "torch", "transformers")},
        "test_labels_used_for_fitting_or_tuning": False,
        "representations_persisted": False,
    }
    payloads = {
        "head_checkpoint.json": checkpoint_bytes,
        "predictions.jsonl": jsonl_bytes(predictions),
        "metrics.json": canonical_bytes(metrics),
        "risk_coverage.jsonl": jsonl_bytes(curves),
        "provenance.json": canonical_bytes(provenance),
    }
    output_directory = repo_path(config["output"]["root"]) / stage
    statuses, ledger = publish(output_directory, payloads)
    report = {
        "stage": stage,
        "output_directory": output_directory.relative_to(REPO).as_posix(),
        "training": training_report,
        "posterior": {
            "parameters": int(variance.numel()),
            "samples": sample_count,
            "minimum_precision": float(precision.min().item()),
            "maximum_precision": float(precision.max().item()),
            "minimum_variance": float(variance.min().item()),
            "maximum_variance": float(variance.max().item()),
            "all_variances_finite_nonnegative": bool(torch.isfinite(variance).all().item() and torch.all(variance >= 0).item()),
        },
        "reload_maximum_absolute_difference": maximum_reload_difference,
        "metrics": metrics,
        "artifact_statuses": statuses,
        "artifact_hashes": ledger,
    }
    return report, output_directory


def write_run_manifest(stage: str, status: str, details: Mapping[str, Any], config_path: str) -> Path:
    root = repo_path("outputs/manifests/laplace_head")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    path = root / f"{stage}-{status}-{timestamp.isoformat().replace(':', '-')}.json"
    path.write_bytes(canonical_bytes({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "stage": stage, "status": status, "timestamp_utc": timestamp.isoformat(), "config_path": config_path, "details": details}))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen-Qwen diagonal-Laplace prediction head")
    parser.add_argument("--stage", required=True, choices=("smoke", "full"))
    parser.add_argument("--config", default="configs/laplace_head.yaml")
    parser.add_argument("--preflight", action="store_true", help="check contracts and synthetic Bayesian-head math without loading Qwen")
    args = parser.parse_args()
    if args.preflight:
        details = preflight(Path(args.config))
        print("LAPLACE HEAD PREFLIGHT: PASS")
        print("report=" + canonical_json(details))
        print("SCOPE: synthetic CPU math and frozen-file contracts only; no dataset or Qwen weights were loaded and no files were modified.")
        return 0
    try:
        details, _ = run(args.stage, Path(args.config))
        manifest = write_run_manifest(args.stage, "success", details, args.config)
        print("LAPLACE BAYESIAN HEAD: PASS")
        print(f"stage={args.stage}")
        print("report=" + canonical_json(details))
        print(f"run_manifest={manifest.relative_to(REPO).as_posix()}")
        print("SCOPE: Laplace-approximated Bayesian linear prediction head over frozen Qwen representations; the transformer is not Bayesian.")
        return 0
    except Exception as error:
        details = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        try:
            manifest = write_run_manifest(args.stage, "failed", details, args.config)
            print(f"FAILED_RUN_MANIFEST={manifest.relative_to(REPO).as_posix()}", file=sys.stderr)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
