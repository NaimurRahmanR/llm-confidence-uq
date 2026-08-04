from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq.evaluation import (  # noqa: E402
    METHODS,
    PROTOCOL_VERSION,
    build_evaluation_rows,
    build_metrics,
    canonical_json,
    jsonl_bytes,
    require,
    risk_coverage_rows,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def repo_path(value: str | Path) -> Path:
    path = (REPO / value).resolve()
    require(path == REPO or REPO in path.parents, "path escapes repository")
    return path


def read_jsonl(path: Path, expected_sha256: str, expected_rows: int) -> tuple[list[dict], bytes]:
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"hash mismatch: {path.relative_to(REPO)}")
    require(payload.endswith(b"\n"), f"missing terminal newline: {path.relative_to(REPO)}")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    require(len(rows) == expected_rows, f"row-count mismatch: {path.relative_to(REPO)}")
    require(jsonl_bytes(rows) == payload, f"noncanonical JSONL: {path.relative_to(REPO)}")
    return rows, payload


def verify_ledger(directory: Path) -> None:
    ledger_payload = (directory / "artifact_hashes.json").read_bytes()
    ledger = json.loads(ledger_payload)
    for name, expected in ledger.items():
        payload = (directory / name).read_bytes()
        require(len(payload) == int(expected["bytes"]), f"byte-count mismatch: {directory / name}")
        require(sha256_bytes(payload) == expected["sha256"], f"artifact hash mismatch: {directory / name}")
    complete = json.loads((directory / "COMPLETE.json").read_text("utf-8"))
    require(complete.get("complete") is True, f"incomplete artifact directory: {directory}")
    require(complete.get("artifact_hashes_sha256") == sha256_bytes(ledger_payload), f"ledger binding mismatch: {directory}")


def validate_config(config: Mapping[str, Any], config_path: Path) -> None:
    require(config.get("schema_version") == 1 and config.get("protocol_version") == PROTOCOL_VERSION, "configuration protocol drift")
    require(config.get("classes") == ["Yes", "No"], "configuration class-order drift")
    test = config.get("test", {})
    require(test.get("research_split") == "test" and test.get("source_split") == "validation", "test split drift")
    require(int(test.get("inputs", 0)) == 400 and int(test.get("prediction_rows", 0)) == 2400, "test cardinality drift")
    require(test.get("labels_available_during_inference") is False, "label-isolation drift")
    require(set(config.get("methods", {})) == set(METHODS), "method configuration drift")
    require(config.get("calibration", {}).get("refit_on_test") is False, "test-time temperature refitting enabled")
    for relative, expected in config.get("implementation", {}).items():
        require(sha256_bytes(repo_path(relative).read_bytes()) == expected, f"implementation hash mismatch: {relative}")
    require(config_path == repo_path("configs/evaluation.yaml"), "unexpected configuration path")


def load_inputs(method: str, config: Mapping[str, Any]) -> tuple[list[dict], list[dict], dict, dict[str, str]]:
    test = config["test"]
    manifest_rows, manifest_payload = read_jsonl(repo_path(test["manifest_path"]), test["manifest_sha256"], 400)
    specification = config["methods"][method]
    prediction_path = repo_path(specification["predictions_path"])
    verify_ledger(prediction_path.parent)
    prediction_rows, prediction_payload = read_jsonl(prediction_path, specification["predictions_sha256"], 2400)
    temperature_path = repo_path(specification["temperature_path"])
    verify_ledger(temperature_path.parent)
    temperature_payload = temperature_path.read_bytes()
    require(sha256_bytes(temperature_payload) == specification["temperature_sha256"], "temperature artifact hash mismatch")
    temperature = json.loads(temperature_payload)
    require(temperature.get("method") == method, "temperature method mismatch")
    require(temperature.get("labels_used") == "calibration-only" and temperature.get("test_labels_used") is False, "temperature provenance drift")
    require(temperature.get("predictions_unchanged") is True and temperature.get("class_order") == ["Yes", "No"], "temperature class contract drift")
    return manifest_rows, prediction_rows, temperature, {
        "manifest_sha256": sha256_bytes(manifest_payload),
        "predictions_sha256": sha256_bytes(prediction_payload),
        "temperature_sha256": sha256_bytes(temperature_payload),
    }


def atomic_refuse_or_verify(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and path.read_bytes() == payload, f"refusing to replace differing artifact: {path}")
        return "verified_existing"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def publish(directory: Path, payloads: Mapping[str, bytes]) -> dict[str, str]:
    ledger = {name: {"bytes": len(payload), "sha256": sha256_bytes(payload)} for name, payload in sorted(payloads.items())}
    complete_payloads = {**payloads, "artifact_hashes.json": canonical_bytes(ledger)}
    complete = canonical_bytes({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "artifact_hashes_sha256": sha256_bytes(complete_payloads["artifact_hashes.json"]), "complete": True})
    statuses = {name: atomic_refuse_or_verify(directory / name, payload) for name, payload in complete_payloads.items()}
    statuses["COMPLETE.json"] = atomic_refuse_or_verify(directory / "COMPLETE.json", complete)
    return statuses


def run(method: str, config_path: Path, *, preflight: bool) -> dict[str, Any]:
    require(method in METHODS, "unknown method")
    import yaml
    config_path = repo_path(config_path)
    config_payload = config_path.read_bytes()
    config = yaml.safe_load(config_payload.decode("utf-8"))
    require(isinstance(config, dict), "configuration is not a mapping")
    validate_config(config, config_path)
    manifest, predictions, temperature_artifact, hashes = load_inputs(method, config)
    evaluation_rows = build_evaluation_rows(
        manifest,
        predictions,
        method=method,
        temperature=float(temperature_artifact["temperature"]),
        temperature_artifact_sha256=hashes["temperature_sha256"],
    )
    preflight_report = {
        "method": method,
        "rows": len(evaluation_rows),
        "inputs": len({row["input_id"] for row in evaluation_rows}),
        "conditions": len({row["condition"] for row in evaluation_rows}),
        "temperature": temperature_artifact["temperature"],
        "temperature_refit_on_test": False,
        "prediction_classes_unchanged": all(row["token_prediction"] == ("Yes" if row["calibrated_p_yes"] >= row["calibrated_p_no"] else "No") for row in evaluation_rows),
    }
    if preflight:
        return {"preflight": preflight_report}

    metrics = build_metrics(evaluation_rows, method=method)
    curves = risk_coverage_rows(evaluation_rows, method=method)
    provenance = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_sha256": sha256_bytes(config_payload),
        **hashes,
        "labels_joined_after_inference": True,
        "temperature_refit_on_test": False,
    }
    payloads = {
        "evaluations.jsonl": jsonl_bytes(evaluation_rows),
        "metrics.json": canonical_bytes(metrics),
        "risk_coverage.jsonl": jsonl_bytes(curves),
        "provenance.json": canonical_bytes(provenance),
    }
    output_directory = repo_path(config["output"]["root"]) / method
    statuses = publish(output_directory, payloads)
    return {
        "output_directory": output_directory.relative_to(REPO).as_posix(),
        "metrics": metrics,
        "artifact_statuses": statuses,
        "artifact_hashes": {name: sha256_bytes(payload) for name, payload in payloads.items()},
    }


def write_run_manifest(method: str, status: str, details: Mapping[str, Any], config_path: str) -> Path:
    root = repo_path("outputs/manifests/evaluation")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    filename = f"{method}-{status}-{timestamp.isoformat().replace(':', '-')}.json"
    path = root / filename
    path.write_bytes(canonical_bytes({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "method": method, "status": status, "timestamp_utc": timestamp.isoformat(), "config_path": config_path, "details": details}))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate raw and frozen-temperature BoolQ test predictions")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        details = run(args.method, Path(args.config), preflight=args.preflight)
        if args.preflight:
            print("TEST EVALUATION PREFLIGHT: PASS")
            print(canonical_json(details["preflight"]))
            print("SCOPE: alignment and frozen-temperature application only; no metrics or files written.")
            return 0
        manifest = write_run_manifest(args.method, "success", details, args.config)
        print("CALIBRATED TEST EVALUATION: PASS")
        print(f"method={args.method}")
        print(f"output_directory={details['output_directory']}")
        print("metrics=" + canonical_json(details["metrics"]))
        print("artifact_statuses=" + canonical_json(details["artifact_statuses"]))
        print("artifact_hashes=" + canonical_json(details["artifact_hashes"]))
        print(f"run_manifest={manifest.relative_to(REPO).as_posix()}")
        print("SCOPE: test labels joined after inference; frozen calibration temperature applied without test refitting.")
        return 0
    except Exception as error:
        details = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        try:
            manifest = write_run_manifest(args.method, "failed", details, args.config)
            print(f"FAILED_RUN_MANIFEST={manifest.relative_to(REPO).as_posix()}", file=sys.stderr)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
