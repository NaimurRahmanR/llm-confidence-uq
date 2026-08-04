#!/usr/bin/env python3
"""Build and evaluate the locked three-adapter BoolQ ensemble."""

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
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq.ensemble import (  # noqa: E402
    CONDITIONS,
    MEMBERS,
    PROTOCOL_VERSION,
    aggregate_rows,
    build_metrics,
    canonical_json,
    jsonl_bytes,
    risk_coverage_rows,
)


class EnsembleRunError(RuntimeError):
    """Raised when the ensemble runner fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EnsembleRunError(message)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def load_jsonl(path: Path, expected_sha256: str) -> tuple[list[dict[str, Any]], bytes]:
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"input hash drift: {path.relative_to(REPO)}")
    require(payload.endswith(b"\n"), f"JSONL is not newline terminated: {path.relative_to(REPO)}")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    require(all(isinstance(row, dict) for row in rows), "non-object JSONL row")
    return rows, payload


def validate_config(config: Mapping[str, Any], config_payload: bytes) -> None:
    require(config.get("schema_version") == 1, "configuration schema drift")
    require(config.get("protocol_version") == PROTOCOL_VERSION, "configuration protocol drift")
    implementation = config.get("implementation")
    require(isinstance(implementation, dict), "missing implementation contract")
    module_path = repo_path(implementation.get("module_path", ""))
    require(module_path.is_file(), "ensemble implementation is missing")
    require(sha256_bytes(module_path.read_bytes()) == implementation.get("module_sha256"), "ensemble implementation hash drift")
    members = config.get("members")
    require(isinstance(members, dict) and tuple(members) == MEMBERS, "locked member order drift")
    seeds: list[int] = []
    adapters: list[str] = []
    for method in MEMBERS:
        member = members[method]
        require(member.get("method") == method, "configured method drift")
        require(type(member.get("seed")) is int, "configured seed is invalid")
        require(isinstance(member.get("adapter_sha256"), str) and len(member["adapter_sha256"]) == 64, "configured adapter hash is invalid")
        require(isinstance(member.get("evaluation_sha256"), str) and len(member["evaluation_sha256"]) == 64, "configured evaluation hash is invalid")
        require(isinstance(member.get("evaluation_path"), str), "configured evaluation path is invalid")
        seeds.append(member["seed"])
        adapters.append(member["adapter_sha256"])
    require(len(set(seeds)) == 3, "configured seeds are not unique")
    require(len(set(adapters)) == 3, "configured checkpoints are not unique")
    expected = config.get("expected")
    require(expected == {"rows": 2400, "inputs": 400, "conditions": list(CONDITIONS)}, "expected cardinality contract drift")
    claims = config.get("claim_boundaries")
    require(claims == {
        "members_are_posterior_samples": False,
        "full_transformer_is_bayesian": False,
        "test_labels_used_for_fitting_or_tuning": False,
        "mutual_information_name": "finite-ensemble-entropy-Jensen-gap",
    }, "ensemble claim-boundary drift")
    require(config_payload.endswith(b"\n"), "configuration is not newline terminated")


def load_inputs(config: Mapping[str, Any]) -> tuple[dict[str, Sequence[Mapping[str, Any]]], dict[str, Mapping[str, Any]], dict[str, str]]:
    member_rows: dict[str, Sequence[Mapping[str, Any]]] = {}
    member_metadata: dict[str, Mapping[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for method in MEMBERS:
        member = config["members"][method]
        path = repo_path(member["evaluation_path"])
        rows, payload = load_jsonl(path, member["evaluation_sha256"])
        require(len(rows) == config["expected"]["rows"], f"{method} row cardinality drift")
        member_rows[method] = rows
        member_metadata[method] = {
            "method": method,
            "seed": member["seed"],
            "adapter_sha256": member["adapter_sha256"],
        }
        input_hashes[method] = sha256_bytes(payload)
    return member_rows, member_metadata, input_hashes


def atomic_refuse_or_verify(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, f"refusing to replace differing artifact: {path.relative_to(REPO)}")
        return "verified_existing"
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
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


def publish(directory: Path, payloads: Mapping[str, bytes]) -> dict[str, str]:
    ledger = {name: {"bytes": len(payload), "sha256": sha256_bytes(payload)} for name, payload in sorted(payloads.items())}
    all_payloads = {**payloads, "artifact_hashes.json": canonical_bytes(ledger)}
    complete = canonical_bytes({
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_hashes_sha256": sha256_bytes(all_payloads["artifact_hashes.json"]),
        "complete": True,
    })
    statuses = {name: atomic_refuse_or_verify(directory / name, payload) for name, payload in all_payloads.items()}
    statuses["COMPLETE.json"] = atomic_refuse_or_verify(directory / "COMPLETE.json", complete)
    return statuses


def run(config_path: Path, *, preflight: bool) -> dict[str, Any]:
    import yaml

    config_path = repo_path(config_path)
    config_payload = config_path.read_bytes()
    config = yaml.safe_load(config_payload.decode("utf-8"))
    require(isinstance(config, dict), "configuration is not a mapping")
    validate_config(config, config_payload)
    member_rows, member_metadata, input_hashes = load_inputs(config)
    rows = aggregate_rows(member_rows, member_metadata)
    variation_rows = sum(
        len({member["raw_p_yes"] for member in row["members"]}) > 1
        for row in rows
    )
    preflight_report = {
        "rows": len(rows),
        "inputs": len({row["input_id"] for row in rows}),
        "conditions": len({row["condition"] for row in rows}),
        "members": list(MEMBERS),
        "unique_seeds": len({metadata["seed"] for metadata in member_metadata.values()}),
        "unique_adapter_checkpoints": len({metadata["adapter_sha256"] for metadata in member_metadata.values()}),
        "member_probability_variation_rows": variation_rows,
        "vote_disagreement_rows": sum(bool(row["any_vote_disagreement"]) for row in rows),
        "aggregation": "arithmetic-mean-of-member-binary-probabilities",
        "test_labels_used_for_fitting_or_tuning": False,
    }
    require(variation_rows > 0, "member predictions contain no probability variation")
    if preflight:
        return {"preflight": preflight_report}

    metrics = build_metrics(rows)
    curves = risk_coverage_rows(rows)
    provenance = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config_sha256": sha256_bytes(config_payload),
        "input_evaluation_sha256": input_hashes,
        "members": member_metadata,
        "aggregation": "arithmetic-mean-of-member-binary-probabilities",
        "test_labels_used_for_fitting_or_tuning": False,
        "members_are_posterior_samples": False,
        "full_transformer_is_bayesian": False,
    }
    payloads = {
        "ensemble_predictions.jsonl": jsonl_bytes(rows),
        "metrics.json": canonical_bytes(metrics),
        "risk_coverage.jsonl": jsonl_bytes(curves),
        "provenance.json": canonical_bytes(provenance),
    }
    output_directory = repo_path(config["output"]["directory"])
    statuses = publish(output_directory, payloads)
    return {
        "output_directory": output_directory.relative_to(REPO).as_posix(),
        "preflight": preflight_report,
        "metrics": metrics,
        "artifact_statuses": statuses,
        "artifact_hashes": {name: sha256_bytes(payload) for name, payload in payloads.items()},
    }


def write_run_manifest(status: str, details: Mapping[str, Any], config_path: str) -> Path:
    root = repo_path("outputs/manifests/ensemble")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    path = root / f"three-lora-{status}-{timestamp.isoformat().replace(':', '-')}.json"
    path.write_bytes(canonical_bytes({
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "timestamp_utc": timestamp.isoformat(),
        "config_path": config_path,
        "details": details,
    }))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the locked three-adapter LoRA ensemble")
    parser.add_argument("--config", default="configs/ensemble.yaml")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        details = run(Path(args.config), preflight=args.preflight)
        if args.preflight:
            print("THREE-LORA ENSEMBLE PREFLIGHT: PASS")
            print(canonical_json(details["preflight"]))
            print("SCOPE: deterministic member alignment and probability aggregation only; no artifacts written.")
            return 0
        manifest = write_run_manifest("success", details, args.config)
        print("THREE-LORA ENSEMBLE EVALUATION: PASS")
        print(f"output_directory={details['output_directory']}")
        print("preflight=" + canonical_json(details["preflight"]))
        print("metrics=" + canonical_json(details["metrics"]))
        print("artifact_statuses=" + canonical_json(details["artifact_statuses"]))
        print("artifact_hashes=" + canonical_json(details["artifact_hashes"]))
        print(f"run_manifest={manifest.relative_to(REPO).as_posix()}")
        print("SCOPE: finite three-adapter ensemble; not posterior sampling and not a Bayesian transformer.")
        return 0
    except Exception as error:
        details = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        try:
            manifest = write_run_manifest("failed", details, args.config)
            print(f"FAILED_RUN_MANIFEST={manifest.relative_to(REPO).as_posix()}", file=sys.stderr)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
