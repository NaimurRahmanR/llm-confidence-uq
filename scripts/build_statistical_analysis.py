#!/usr/bin/env python3
"""Build clustered bootstrap intervals and direct UQ-signal diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq.analysis import METHODS, build_normalized_rows, canonical_json, jsonl_bytes, require  # noqa: E402
from llm_confidence_uq.statistics import (  # noqa: E402
    PROTOCOL_VERSION,
    build_clustered_bootstrap_differences,
    build_lora_seed_summary,
    build_uq_signal_evaluations,
    validate_statistical_outputs,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def canonical_tracked_payload(path: Path) -> bytes:
    """Return Git-canonical bytes for text artifacts on every checkout OS."""
    payload = path.read_bytes()
    if path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml", ".md", ".txt", ".py", ".toml", ".cff"}:
        payload = payload.replace(b"\r\n", b"\n")
    return payload


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    absolute = path.resolve() if path.is_absolute() else (REPO / path).resolve()
    require(absolute == REPO or REPO in absolute.parents, f"path escapes repository: {absolute}")
    return absolute


def verify_ledger(directory: Path) -> None:
    ledger_path = directory / "artifact_hashes.json"
    complete_path = directory / "COMPLETE.json"
    require(ledger_path.is_file() and complete_path.is_file(), f"artifact ledger absent: {directory.relative_to(REPO)}")
    ledger_payload = canonical_tracked_payload(ledger_path)
    ledger = json.loads(ledger_payload)
    require(isinstance(ledger, dict) and ledger, "invalid source ledger")
    for name, expected in ledger.items():
        payload = canonical_tracked_payload(directory / name)
        require(len(payload) == int(expected["bytes"]), f"source artifact byte drift: {name}")
        require(sha256_bytes(payload) == expected["sha256"], f"source artifact hash drift: {name}")
    completion = json.loads(canonical_tracked_payload(complete_path))
    require(completion.get("complete") is True, "source completion marker invalid")
    require(completion.get("artifact_hashes_sha256") == sha256_bytes(ledger_payload), "source ledger binding drift")


def read_jsonl(specification: Mapping[str, Any], expected_rows: int) -> tuple[list[dict[str, Any]], str]:
    path = repo_path(specification["path"])
    verify_ledger(path.parent)
    payload = canonical_tracked_payload(path)
    require(sha256_bytes(payload) == specification["sha256"], f"input hash drift: {specification['path']}")
    require(payload.endswith(b"\n") and b"\r" not in payload, "input newline drift")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    require(len(rows) == expected_rows and jsonl_bytes(rows) == payload, f"input JSONL drift: {specification['path']}")
    return rows, sha256_bytes(payload)


def load_inputs(config: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    evaluations: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for method in METHODS:
        rows, digest = read_jsonl(config["inputs"]["evaluations"][method], 2400)
        evaluations[method] = rows
        hashes[f"evaluation_{method}"] = digest
    ensemble, hashes["ensemble"] = read_jsonl(config["inputs"]["ensemble"], 2400)
    laplace, hashes["laplace_head"] = read_jsonl(config["inputs"]["laplace_head"], 2400)
    return evaluations, ensemble, laplace, hashes


def validate_config(config: Mapping[str, Any]) -> None:
    require(config.get("schema_version") == 1 and config.get("protocol_version") == PROTOCOL_VERSION, "statistical configuration protocol drift")
    require(config["bootstrap"] == {
        "cluster_unit": "input_id",
        "clusters": 400,
        "repetitions": 2000,
        "seed": 20260804,
        "confidence_level": 0.95,
        "interval_method": "paired-cluster-percentile",
        "numpy_bit_generator": "PCG64",
    }, "bootstrap contract drift")
    require(config["metrics"] == {
        "ece_bins": 10,
        "nll_epsilon": 1e-12,
        "auprc_definition": "threshold-grouped-average-precision",
        "higher_uq_score_means": "more_uncertain",
    }, "statistical metric contract drift")
    require(config["claim_boundaries"]["ensemble_mi_style_is_exact_bayesian_mi"] is False, "ensemble MI overclaim")


def build_payloads(config: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    evaluations, ensemble, laplace, input_hashes = load_inputs(config)
    normalized, _ = build_normalized_rows(evaluations, ensemble, laplace)
    bootstrap = config["bootstrap"]
    metric_config = config["metrics"]
    bootstrap_rows = build_clustered_bootstrap_differences(
        normalized,
        repetitions=int(bootstrap["repetitions"]),
        seed=int(bootstrap["seed"]),
        confidence_level=float(bootstrap["confidence_level"]),
        nll_epsilon=float(metric_config["nll_epsilon"]),
        bins=int(metric_config["ece_bins"]),
    )
    seed_rows = build_lora_seed_summary(
        normalized,
        nll_epsilon=float(metric_config["nll_epsilon"]),
        bins=int(metric_config["ece_bins"]),
    )
    error_rows, degradation_rows, seed_uq_rows = build_uq_signal_evaluations(normalized, ensemble, laplace)
    validate_statistical_outputs(bootstrap_rows, seed_rows, error_rows, degradation_rows)

    overall_bootstrap = next(
        row for row in bootstrap_rows
        if row["comparison"] == "lora_seed_mean_vs_baseline_calibrated" and row["scope"] == "overall"
    )
    overall_seeds = next(row for row in seed_rows if row["scope"] == "overall")
    summary = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "clusters": 400,
        "conditions_per_cluster": 6,
        "bootstrap_repetitions": int(bootstrap["repetitions"]),
        "bootstrap_seed": int(bootstrap["seed"]),
        "confidence_level": float(bootstrap["confidence_level"]),
        "lora_seed_overall": overall_seeds,
        "lora_seed_mean_vs_baseline_calibrated": overall_bootstrap,
        "uq_error_signal_rows": len(error_rows),
        "uq_degradation_signal_rows": len(degradation_rows),
        "claim_boundaries": dict(config["claim_boundaries"]),
    }
    provenance = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "input_sha256": input_hashes,
        "config_sha256": sha256_bytes(repo_path("configs/statistical_analysis.yaml").read_bytes()),
        "implementation_sha256": {
            "src/llm_confidence_uq/statistics.py": sha256_bytes(repo_path("src/llm_confidence_uq/statistics.py").read_bytes()),
            "scripts/build_statistical_analysis.py": sha256_bytes(repo_path("scripts/build_statistical_analysis.py").read_bytes()),
        },
        "labels_used_only_for_post_hoc_evaluation": True,
        "model_inference_or_training_performed": False,
        "bootstrap": dict(bootstrap),
    }
    payloads = {
        "bootstrap_differences.jsonl": jsonl_bytes(bootstrap_rows),
        "lora_seed_summary.jsonl": jsonl_bytes(seed_rows),
        "lora_seed_uq_summary.jsonl": jsonl_bytes(seed_uq_rows),
        "uq_degradation_detection.jsonl": jsonl_bytes(degradation_rows),
        "uq_error_detection.jsonl": jsonl_bytes(error_rows),
        "summary.json": canonical_bytes(summary),
        "provenance.json": canonical_bytes(provenance),
    }
    return payloads, summary


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


def publish(config: Mapping[str, Any], payloads: Mapping[str, bytes]) -> dict[str, str]:
    directory = repo_path(config["output"]["directory"])
    ledger = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(payloads.items())
    }
    complete_payloads = dict(payloads)
    complete_payloads["artifact_hashes.json"] = canonical_bytes(ledger)
    complete_payloads["COMPLETE.json"] = canonical_bytes({
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "complete": True,
        "artifact_hashes_sha256": sha256_bytes(complete_payloads["artifact_hashes.json"]),
    })
    statuses: dict[str, str] = {}
    for name, payload in complete_payloads.items():
        path = directory / name
        if path.exists():
            require(path.read_bytes() == payload, f"refusing incompatible existing statistical artifact: {name}")
            statuses[name] = "verified_existing"
        else:
            _atomic_write(path, payload)
            statuses[name] = "created"
        require(path.read_bytes() == payload, f"post-write statistical artifact drift: {name}")
    return statuses


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clustered bootstrap and UQ-signal results")
    parser.add_argument("--config", default="configs/statistical_analysis.yaml")
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    try:
        import yaml

        config_path = repo_path(arguments.config)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        require(isinstance(config, dict), "statistical configuration is not a mapping")
        validate_config(config)
        payloads, summary = build_payloads(config)
        if arguments.preflight:
            print("STATISTICAL ANALYSIS PREFLIGHT: PASS")
            print("summary=" + canonical_json(summary))
            print("SCOPE: deterministic post-hoc analysis only; no files were modified.")
            return 0
        statuses = publish(config, payloads)
        print("STATISTICAL ANALYSIS: PASS")
        print("summary=" + canonical_json(summary))
        print("artifact_statuses=" + canonical_json(statuses))
        print("SCOPE: paired input-cluster bootstrap and UQ-signal evaluation; no model inference or training was performed.")
        return 0
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
