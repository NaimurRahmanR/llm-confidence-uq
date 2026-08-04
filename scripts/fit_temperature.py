from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
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

from llm_confidence_uq.calibration import (  # noqa: E402
    METHODS,
    PROTOCOL_VERSION,
    build_fit_inputs,
    canonical_json,
    fit_temperature,
    make_temperature_artifact,
    require,
)


class TemperatureRunError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def repo_path(value: str | Path) -> Path:
    path = (REPO / Path(value)).resolve()
    require(path == REPO or REPO in path.parents, "path escapes repository")
    return path


def read_canonical_jsonl(path: Path, expected_sha256: str, expected_rows: int) -> tuple[list[dict], bytes]:
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"hash mismatch: {path.relative_to(REPO)}")
    require(payload.endswith(b"\n"), f"missing final newline: {path.relative_to(REPO)}")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    require(len(rows) == expected_rows, f"row-count mismatch: {path.relative_to(REPO)}")
    reconstructed = b"".join(canonical_bytes(row) for row in rows)
    require(reconstructed == payload, f"noncanonical JSONL: {path.relative_to(REPO)}")
    return rows, payload


def validate_config(config: Mapping[str, Any], config_path: Path) -> None:
    require(config.get("schema_version") == 1, "configuration schema drift")
    require(config.get("protocol_version") == PROTOCOL_VERSION, "configuration protocol drift")
    require(config.get("classes") == ["Yes", "No"], "configuration class-order drift")
    data = config.get("data", {})
    require(data.get("labels") == "calibration-only", "labels are not calibration-only")
    require(data.get("test_labels_used") is False, "test-label use is not forbidden")
    require(data.get("research_split") == "calibration", "wrong research split")
    require(data.get("source_split") == "train", "wrong source split")
    require(data.get("evidence_condition") == "original", "wrong evidence condition")
    require(int(data.get("rows", 0)) == 200, "wrong calibration row count")
    require(set(config.get("methods", {})) == set(METHODS), "method configuration drift")
    fitting = config.get("fitting", {})
    require(fitting.get("objective") == "mean-cross-entropy", "objective drift")
    require(fitting.get("dtype") == "float64", "fitting dtype drift")
    require(fitting.get("device") == "cpu", "fitting device drift")
    require(float(fitting.get("minimum_temperature")) > 0.0, "nonpositive minimum temperature")
    require(float(fitting.get("maximum_temperature")) > 1.0, "invalid maximum temperature")
    require(int(fitting.get("maximum_iterations")) >= 1, "invalid iteration limit")
    require(int(fitting.get("ece_bins")) >= 2, "invalid ECE bins")
    for relative, expected in config.get("implementation", {}).items():
        payload = repo_path(relative).read_bytes()
        require(sha256_bytes(payload) == expected, f"implementation hash mismatch: {relative}")
    require(config_path == repo_path("configs/temperature.yaml"), "unexpected configuration path")


def verify_artifact_ledger(directory: Path) -> None:
    ledger_path = directory / "artifact_hashes.json"
    complete_path = directory / "COMPLETE.json"
    require(ledger_path.is_file() and complete_path.is_file(), f"incomplete prediction directory: {directory}")
    ledger_payload = ledger_path.read_bytes()
    ledger = json.loads(ledger_payload)
    for name, expected in ledger.items():
        payload = (directory / name).read_bytes()
        require(len(payload) == int(expected["bytes"]), f"artifact byte-count mismatch: {directory / name}")
        require(sha256_bytes(payload) == expected["sha256"], f"artifact hash mismatch: {directory / name}")
    complete = json.loads(complete_path.read_text("utf-8"))
    require(complete.get("complete") is True, f"prediction completion marker is false: {directory}")
    require(complete.get("artifact_hashes_sha256") == sha256_bytes(ledger_payload), f"prediction ledger binding mismatch: {directory}")


def load_inputs(method: str, config: Mapping[str, Any]) -> tuple[Any, dict[str, str]]:
    data = config["data"]
    manifest_rows, manifest_payload = read_canonical_jsonl(
        repo_path(data["manifest_path"]),
        str(data["manifest_sha256"]),
        int(data["rows"]),
    )
    specification = config["methods"][method]
    prediction_path = repo_path(specification["predictions_path"])
    verify_artifact_ledger(prediction_path.parent)
    prediction_rows, prediction_payload = read_canonical_jsonl(
        prediction_path,
        str(specification["predictions_sha256"]),
        int(data["rows"]),
    )
    inputs = build_fit_inputs(manifest_rows, prediction_rows, method=method)
    return inputs, {
        "manifest_sha256": sha256_bytes(manifest_payload),
        "predictions_sha256": sha256_bytes(prediction_payload),
    }


def atomic_refuse_or_verify(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and path.read_bytes() == payload, f"refusing to replace differing artifact: {path}")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
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


def publish(directory: Path, *, artifact: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, str]:
    summary = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "method": artifact["method"],
        "rows": artifact["rows"],
        "temperature": artifact["temperature"],
        "initial_nll": artifact["initial_nll"],
        "final_nll": artifact["final_nll"],
        "initial_brier": artifact["initial_brier"],
        "final_brier": artifact["final_brier"],
        "initial_ece_10_bin": artifact["initial_ece_10_bin"],
        "final_ece_10_bin": artifact["final_ece_10_bin"],
        "predictions_unchanged": artifact["predictions_unchanged"],
        "boundary_hit": artifact["boundary_hit"],
        "labels_used": "calibration-only",
        "test_labels_used": False,
    }
    base_payloads = {
        "temperature.json": canonical_bytes(artifact),
        "provenance.json": canonical_bytes(provenance),
        "summary.json": canonical_bytes(summary),
    }
    ledger = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(base_payloads.items())
    }
    payloads = {**base_payloads, "artifact_hashes.json": canonical_bytes(ledger)}
    complete = canonical_bytes({
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_hashes_sha256": sha256_bytes(payloads["artifact_hashes.json"]),
        "complete": True,
    })
    statuses = {name: atomic_refuse_or_verify(directory / name, payload) for name, payload in payloads.items()}
    statuses["COMPLETE.json"] = atomic_refuse_or_verify(directory / "COMPLETE.json", complete)
    return statuses


def package_versions() -> dict[str, str | None]:
    values = {}
    for name in ("torch", "pyyaml"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def run(method: str, config_path: Path, *, preflight: bool) -> dict[str, Any]:
    require(method in METHODS, "unknown method")
    import yaml

    config_path = repo_path(config_path)
    config_payload = config_path.read_bytes()
    config = yaml.safe_load(config_payload.decode("utf-8"))
    require(isinstance(config, dict), "configuration is not a mapping")
    validate_config(config, config_path)
    inputs, hashes = load_inputs(method, config)

    preflight_report = {
        "method": method,
        "rows": len(inputs.labels),
        "class_counts": {"No": inputs.labels.count(1), "Yes": inputs.labels.count(0)},
        "alignment_sha256": inputs.alignment_sha256,
        "manifest_sha256": hashes["manifest_sha256"],
        "predictions_sha256": hashes["predictions_sha256"],
        "labels_used": "calibration-only",
        "test_labels_used": False,
    }
    if preflight:
        return {"preflight": preflight_report}

    import torch

    fitting = config["fitting"]
    result = fit_temperature(
        inputs,
        torch,
        minimum_temperature=float(fitting["minimum_temperature"]),
        maximum_temperature=float(fitting["maximum_temperature"]),
        maximum_iterations=int(fitting["maximum_iterations"]),
        ece_bins=int(fitting["ece_bins"]),
    )
    artifact = make_temperature_artifact(
        inputs,
        result,
        calibration_manifest_sha256=hashes["manifest_sha256"],
        prediction_jsonl_sha256=hashes["predictions_sha256"],
        config_sha256=sha256_bytes(config_payload),
    )
    provenance = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "config_sha256": sha256_bytes(config_payload),
        "manifest_sha256": hashes["manifest_sha256"],
        "predictions_sha256": hashes["predictions_sha256"],
        "alignment_sha256": inputs.alignment_sha256,
        "labels_used": "calibration-only",
        "test_labels_used": False,
    }
    output_directory = repo_path(config["output"]["root"]) / method
    statuses = publish(output_directory, artifact=artifact, provenance=provenance)
    return {
        "artifact": artifact,
        "artifact_statuses": statuses,
        "output_directory": output_directory.relative_to(REPO).as_posix(),
    }


def write_run_manifest(method: str, status: str, details: Mapping[str, Any], config_path: str) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat().replace(":", "-")
    root = repo_path("outputs/manifests/temperature")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{method}-{status}-{timestamp}.json"
    payload = canonical_bytes({
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "details": details,
    })
    destination.write_bytes(payload)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit scalar temperature on the frozen BoolQ calibration split")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--config", default="configs/temperature.yaml")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        details = run(args.method, Path(args.config), preflight=args.preflight)
        if args.preflight:
            print("TEMPERATURE PREFLIGHT: PASS")
            print(canonical_json(details["preflight"]))
            print("SCOPE: input integrity and calibration-label isolation only; no optimization and no files written.")
            return 0
        manifest = write_run_manifest(args.method, "success", details, args.config)
        print("TEMPERATURE SCALING: PASS")
        print(f"method={args.method}")
        print(f"output_directory={details['output_directory']}")
        print("result=" + canonical_json(details["artifact"]))
        print("artifact_statuses=" + canonical_json(details["artifact_statuses"]))
        print(f"run_manifest={manifest.relative_to(REPO).as_posix()}")
        print("SCOPE: one scalar temperature fitted using calibration labels only; no test labels were loaded.")
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
