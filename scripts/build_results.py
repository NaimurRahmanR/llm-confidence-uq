#!/usr/bin/env python3
"""Build unified metrics and eight reproducible figures from frozen artifacts."""

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
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq.analysis import (  # noqa: E402
    ALL_VARIANTS,
    CONDITIONS,
    DISPLAY_LABELS,
    FIGURES,
    METHODS,
    PRIMARY_VARIANTS,
    PROTOCOL_VERSION,
    build_expressed_summary,
    build_method_metrics,
    build_normalized_rows,
    build_paired_changes,
    build_reliability_table,
    build_risk_coverage,
    canonical_json,
    jsonl_bytes,
    render_figures,
    require,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    absolute = path.resolve() if path.is_absolute() else (REPO / path).resolve()
    require(absolute == REPO or REPO in absolute.parents, f"path escapes repository: {absolute}")
    return absolute


def verify_ledger(directory: Path) -> None:
    ledger_path = directory / "artifact_hashes.json"
    complete_path = directory / "COMPLETE.json"
    require(ledger_path.is_file() and complete_path.is_file(), f"artifact ledger absent: {directory.relative_to(REPO)}")
    ledger_payload = ledger_path.read_bytes()
    ledger = json.loads(ledger_payload)
    require(isinstance(ledger, dict) and ledger, f"invalid ledger: {directory.relative_to(REPO)}")
    for name, expected in ledger.items():
        path = directory / name
        payload = path.read_bytes()
        require(len(payload) == int(expected["bytes"]), f"artifact byte-count drift: {path.relative_to(REPO)}")
        require(sha256_bytes(payload) == expected["sha256"], f"artifact hash drift: {path.relative_to(REPO)}")
    complete = json.loads(complete_path.read_bytes())
    require(complete.get("complete") is True, f"artifact directory incomplete: {directory.relative_to(REPO)}")
    require(complete.get("artifact_hashes_sha256") == sha256_bytes(ledger_payload), f"ledger binding drift: {directory.relative_to(REPO)}")


def read_jsonl(path: Path, expected_sha256: str, expected_rows: int) -> tuple[list[dict[str, Any]], bytes]:
    verify_ledger(path.parent)
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"input hash drift: {path.relative_to(REPO)}")
    require(payload.endswith(b"\n") and b"\r" not in payload, f"input newline drift: {path.relative_to(REPO)}")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    require(len(rows) == expected_rows and all(isinstance(row, dict) for row in rows), f"input cardinality drift: {path.relative_to(REPO)}")
    require(jsonl_bytes(rows) == payload, f"input JSONL is noncanonical: {path.relative_to(REPO)}")
    return rows, payload


def validate_config(config: Mapping[str, Any], config_path: Path) -> None:
    require(config.get("schema_version") == 1 and config.get("protocol_version") == PROTOCOL_VERSION, "results configuration protocol drift")
    expected = config.get("expected")
    require(expected == {
        "rows_per_variant": 2400,
        "inputs": 400,
        "conditions": list(CONDITIONS),
        "all_variants": list(ALL_VARIANTS),
        "primary_variants": list(PRIMARY_VARIANTS),
        "figures": list(FIGURES),
    }, "results cardinality or figure contract drift")
    require(config["metrics"] == {
        "ece_bins": 10,
        "nll_epsilon": 1e-12,
        "brier_definition": "binary-yes-probability",
        "error_detection_score": "one-minus-selected-class-confidence",
        "expressed_invalid_policy": "retain-missing-no-imputation",
    }, "metric contract drift")
    require(config["claim_boundaries"] == {
        "ensemble_members_are_posterior_samples": False,
        "full_transformer_is_bayesian": False,
        "laplace_scope": "binary-linear-head-over-frozen-representations",
        "test_labels_used_for_fitting_or_tuning": False,
        "post_hoc_test_results_are_descriptive_not_model_selection": True,
        "lower_ece_alone_proves_superior_reliability": False,
    }, "claim-boundary drift")
    for relative, expected_hash in config["implementation"].items():
        require(expected_hash != "PENDING", f"pending implementation hash: {relative}")
        path = repo_path(relative)
        require(path.is_file() and sha256_bytes(path.read_bytes()) == expected_hash, f"implementation hash drift: {relative}")
    for relative, expected_hash in config["prerequisites"].items():
        path = repo_path(relative)
        require(path.is_file() and sha256_bytes(path.read_bytes()) == expected_hash, f"prerequisite hash drift: {relative}")
    require(config_path == repo_path("configs/results.yaml"), "unexpected results configuration path")


def load_inputs(config: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    evaluation_rows: dict[str, list[dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}
    for method in METHODS:
        specification = config["inputs"]["evaluations"][method]
        rows, payload = read_jsonl(repo_path(specification["path"]), specification["sha256"], 2400)
        evaluation_rows[method] = rows
        input_hashes[f"evaluation_{method}"] = sha256_bytes(payload)
    ensemble_specification = config["inputs"]["ensemble"]
    ensemble_rows, ensemble_payload = read_jsonl(repo_path(ensemble_specification["path"]), ensemble_specification["sha256"], 2400)
    input_hashes["ensemble"] = sha256_bytes(ensemble_payload)
    laplace_specification = config["inputs"]["laplace_head"]
    laplace_rows, laplace_payload = read_jsonl(repo_path(laplace_specification["path"]), laplace_specification["sha256"], 2400)
    input_hashes["laplace_head"] = sha256_bytes(laplace_payload)
    return evaluation_rows, ensemble_rows, laplace_rows, input_hashes


def build_tables(config: Mapping[str, Any], inputs: tuple[Any, ...]) -> dict[str, Any]:
    evaluation_rows, ensemble_rows, laplace_rows, input_hashes = inputs
    normalized, expressed = build_normalized_rows(evaluation_rows, ensemble_rows, laplace_rows)
    bins = int(config["metrics"]["ece_bins"])
    epsilon = float(config["metrics"]["nll_epsilon"])
    method_metrics = build_method_metrics(normalized, nll_epsilon=epsilon, bins=bins)
    reliability = build_reliability_table(normalized, bins=bins)
    risk_coverage = build_risk_coverage(normalized)
    paired_changes = build_paired_changes(normalized)
    expressed_summary = build_expressed_summary(expressed)
    primary_overall = {
        variant: next(row for row in method_metrics if row["variant"] == variant and row["scope"] == "overall")
        for variant in PRIMARY_VARIANTS
    }
    summary = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "rows_per_variant": 2400,
        "inputs": 400,
        "conditions": list(CONDITIONS),
        "all_variants": list(ALL_VARIANTS),
        "primary_variants": list(PRIMARY_VARIANTS),
        "primary_overall": primary_overall,
        "expressed_overall": {
            method: next(row for row in expressed_summary if row["method"] == method and row["scope"] == "overall")
            for method in METHODS
        },
        "claim_boundaries": dict(config["claim_boundaries"]),
        "input_sha256": input_hashes,
    }
    return {
        "normalized": normalized,
        "expressed": expressed,
        "method_metrics": method_metrics,
        "reliability": reliability,
        "risk_coverage": risk_coverage,
        "paired_changes": paired_changes,
        "expressed_summary": expressed_summary,
        "summary": summary,
        "input_hashes": input_hashes,
    }


def preflight(config_path: Path) -> dict[str, Any]:
    import yaml

    config_path = repo_path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "results configuration is not a mapping")
    validate_config(config, config_path)
    tables = build_tables(config, load_inputs(config))
    valid_counts = {
        method: sum(row["parser_valid"] for row in tables["expressed"] if row["method"] == method)
        for method in METHODS
    }
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "source_prediction_rows": len(METHODS) * 2400 + 2400 + 2400,
        "normalized_variant_rows": len(tables["normalized"]),
        "variants": len(ALL_VARIANTS),
        "primary_variants": len(PRIMARY_VARIANTS),
        "method_metric_rows": len(tables["method_metrics"]),
        "reliability_bin_rows": len(tables["reliability"]),
        "risk_coverage_rows": len(tables["risk_coverage"]),
        "paired_change_rows": len(tables["paired_changes"]),
        "expressed_rows": len(tables["expressed"]),
        "expressed_valid_rows": valid_counts,
        "figures_planned": list(FIGURES),
        "input_sha256": tables["input_hashes"],
        "model_or_gpu_loaded": False,
        "files_modified": False,
    }


def atomic_refuse_or_verify(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and path.read_bytes() == payload, f"refusing to replace differing result artifact: {path.relative_to(REPO)}")
        return "verified_existing"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    ledger_payload = canonical_bytes(ledger)
    complete_payload = canonical_bytes({
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_hashes_sha256": sha256_bytes(ledger_payload),
        "complete": True,
    })
    statuses = {name: atomic_refuse_or_verify(directory / name, payload) for name, payload in payloads.items()}
    statuses["artifact_hashes.json"] = atomic_refuse_or_verify(directory / "artifact_hashes.json", ledger_payload)
    statuses["COMPLETE.json"] = atomic_refuse_or_verify(directory / "COMPLETE.json", complete_payload)
    return statuses, ledger


def run(config_path: Path) -> dict[str, Any]:
    import yaml

    config_path = repo_path(config_path)
    config_payload = config_path.read_bytes()
    config = yaml.safe_load(config_payload.decode("utf-8"))
    require(isinstance(config, dict), "results configuration is not a mapping")
    validate_config(config, config_path)
    tables = build_tables(config, load_inputs(config))

    table_payloads = {
        "method_metrics.jsonl": jsonl_bytes(tables["method_metrics"]),
        "reliability_bins.jsonl": jsonl_bytes(tables["reliability"]),
        "risk_coverage.jsonl": jsonl_bytes(tables["risk_coverage"]),
        "paired_changes.jsonl": jsonl_bytes(tables["paired_changes"]),
        "expressed_alignment.jsonl": jsonl_bytes(tables["expressed"]),
        "expressed_summary.jsonl": jsonl_bytes(tables["expressed_summary"]),
        "summary.json": canonical_bytes(tables["summary"]),
    }
    with tempfile.TemporaryDirectory(prefix="llm-confidence-results-") as temporary_name:
        figure_paths = render_figures(
            normalized=tables["normalized"],
            expressed=tables["expressed"],
            method_metrics=tables["method_metrics"],
            reliability=tables["reliability"],
            risk_coverage=tables["risk_coverage"],
            paired_changes=tables["paired_changes"],
            directory=Path(temporary_name),
        )
        figure_payloads = {f"figures/{path.name}": path.read_bytes() for path in figure_paths}
    require(tuple(name.removeprefix("figures/") for name in figure_payloads) == FIGURES, "figure payload order drift")

    table_hashes = {name: sha256_bytes(payload) for name, payload in table_payloads.items()}
    figure_sources = {
        "01_reliability_diagram.png": ("reliability_bins.jsonl",),
        "02_risk_coverage.png": ("risk_coverage.jsonl",),
        "03_accuracy_by_condition.png": ("method_metrics.jsonl",),
        "04_ece_by_condition.png": ("method_metrics.jsonl",),
        "05_expressed_vs_token_confidence.png": ("expressed_alignment.jsonl", "expressed_summary.jsonl"),
        "06_confidence_change.png": ("paired_changes.jsonl",),
        "07_uncertainty_change.png": ("paired_changes.jsonl",),
        "08_method_comparison.png": ("method_metrics.jsonl",),
    }
    figure_manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "manual_result_values_used": False,
        "input_prediction_sha256": tables["input_hashes"],
        "figures": {
            name: {
                "bytes": len(figure_payloads[f"figures/{name}"]),
                "sha256": sha256_bytes(figure_payloads[f"figures/{name}"]),
                "source_tables": {source: table_hashes[source] for source in figure_sources[name]},
            }
            for name in FIGURES
        },
    }
    provenance = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("matplotlib", "numpy", "pyyaml")},
        "config_sha256": sha256_bytes(config_payload),
        "input_prediction_sha256": tables["input_hashes"],
        "manual_result_values_used": False,
        "test_labels_used_for_fitting_or_tuning": False,
        "full_transformer_is_bayesian": False,
        "ensemble_members_are_posterior_samples": False,
    }
    payloads = {
        **table_payloads,
        **figure_payloads,
        "figure_manifest.json": canonical_bytes(figure_manifest),
        "provenance.json": canonical_bytes(provenance),
    }
    output_directory = repo_path(config["output"]["directory"])
    statuses, ledger = publish(output_directory, payloads)
    return {
        "output_directory": output_directory.relative_to(REPO).as_posix(),
        "rows_per_variant": 2400,
        "variants": len(ALL_VARIANTS),
        "primary_variants": len(PRIMARY_VARIANTS),
        "figures": list(FIGURES),
        "primary_overall": tables["summary"]["primary_overall"],
        "expressed_overall": tables["summary"]["expressed_overall"],
        "artifact_statuses": statuses,
        "artifact_hashes": ledger,
    }


def write_run_manifest(status: str, details: Mapping[str, Any], config_path: str) -> Path:
    root = repo_path("outputs/manifests/results")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    path = root / f"gate8-{status}-{timestamp.isoformat().replace(':', '-')}.json"
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
    parser = argparse.ArgumentParser(description="Build unified research results and figures")
    parser.add_argument("--config", default="configs/results.yaml")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            details = preflight(Path(args.config))
            print("GATE 8 RESULTS PREFLIGHT: PASS")
            print("report=" + canonical_json(details))
            print("SCOPE: read-only schema, alignment, and metric preflight; no model, GPU, plots, or files were created.")
            return 0
        details = run(Path(args.config))
        manifest = write_run_manifest("success", details, args.config)
        print("GATE 8 RESULTS AND FIGURES: PASS")
        print("report=" + canonical_json(details))
        print(f"run_manifest={manifest.relative_to(REPO).as_posix()}")
        print("SCOPE: descriptive post-hoc analysis from frozen artifacts; no model fitting, test tuning, or manually entered result values.")
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
