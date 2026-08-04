from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for directory in (SRC_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_baseline as baseline  # noqa: E402
from llm_confidence_uq.adapter_inference import (  # noqa: E402
    PROTOCOL_VERSION,
    adapter_prediction_jsonl_bytes,
    summarize_adapter_prediction_rows,
    validate_adapter_prediction_rows,
    wrap_prediction_row,
)
from llm_confidence_uq.inference import (  # noqa: E402
    EXPECTED_CLASS_TOKEN_IDS,
    canonical_bytes,
    canonical_json,
    sha256_bytes,
)


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET_ID = "google/boolq"
DATASET_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_GPU_VRAM_GIB = 79.250732421875
EXPECTED_COMPUTE_CAPABILITY = (8, 0)
EXPECTED_BASE_PARAMETERS = 1_543_714_304
EXPECTED_TOTAL_PARAMETERS = 1_552_946_688
EXPECTED_PACKAGE_VERSIONS = dict(baseline.EXPECTED_PACKAGE_VERSIONS)
EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
ALLOWED_ADAPTERS = ("full_seed_1", "full_seed_2", "full_seed_3")
ALLOWED_STAGES = ("smoke12", "full")


class LoRAInferenceRunError(RuntimeError):
    """Raised when a pinned LoRA-inference invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LoRAInferenceRunError(message)


def _repo_path(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    root = Path(os.path.abspath(REPO_ROOT))
    absolute = Path(os.path.abspath(candidate))
    require(
        absolute == root or root in absolute.parents,
        f"path escapes repository: {absolute}",
    )
    current = absolute
    while True:
        require(not current.is_symlink(), f"symbolic link forbidden: {current}")
        if current == root:
            break
        current = current.parent
    return absolute.resolve(strict=False)


def file_sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    values = {}
    for package in EXPECTED_PACKAGE_VERSIONS:
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = "not-installed"
    return values


def validate_torchao_absent() -> None:
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        version = None
    require(version is None, f"excluded package installed: torchao=={version}")
    require(importlib.util.find_spec("torchao") is None, "excluded package importable: torchao")


def semantic_adapter_config_sha256(payload: bytes) -> str:
    config = json.loads(payload)
    targets = config.get("target_modules")
    require(isinstance(targets, list), "adapter target_modules is not a list")
    require(len(targets) == len(set(targets)), "adapter target_modules has duplicates")
    require(set(targets) == EXPECTED_TARGET_MODULES, "adapter target_modules drift")
    normalized = dict(config)
    normalized["target_modules"] = sorted(targets)
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(canonical)


def checkpoint_ledger(directory: Path) -> dict[str, dict[str, Any]]:
    ledger = {}
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative in {"artifact_hashes.json", "COMPLETE.json"}:
            continue
        payload = path.read_bytes()
        ledger[relative] = {"bytes": len(payload), "sha256": sha256_bytes(payload)}
    return ledger


def validate_checkpoint(
    adapter_stage: str,
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    directory = _repo_path(specification["checkpoint_path"])
    require(directory.is_dir() and not directory.is_symlink(), "adapter checkpoint missing")
    for path in directory.rglob("*"):
        require(not path.is_symlink(), f"checkpoint contains symbolic link: {path}")
    ledger_payload = (directory / "artifact_hashes.json").read_bytes()
    ledger = json.loads(ledger_payload)
    require(ledger == checkpoint_ledger(directory), "adapter checkpoint ledger drift")
    complete = json.loads((directory / "COMPLETE.json").read_text("utf-8"))
    require(complete["complete"] is True and complete["stage"] == adapter_stage, "adapter completion drift")
    require(
        complete["artifact_hashes_sha256"] == sha256_bytes(ledger_payload),
        "adapter completion binding drift",
    )
    require(
        complete["artifact_hashes_sha256"] == specification["checkpoint_ledger_sha256"],
        "configured adapter ledger digest drift",
    )
    adapter_relative = "adapter/adapter_model.safetensors"
    config_relative = "adapter/adapter_config.json"
    require(adapter_relative in ledger and config_relative in ledger, "adapter artifacts missing")
    require(
        ledger[adapter_relative]["sha256"] == specification["adapter_weights_sha256"],
        "adapter weights digest drift",
    )
    config_payload = (directory / config_relative).read_bytes()
    require(
        sha256_bytes(config_payload) == specification["raw_adapter_config_sha256"],
        "raw adapter config digest drift",
    )
    semantic_sha256 = semantic_adapter_config_sha256(config_payload)
    require(
        semantic_sha256 == specification["semantic_adapter_config_sha256"],
        "semantic adapter config digest drift",
    )
    report = json.loads((directory / "training_report.json").read_text("utf-8"))
    require(
        report["stage"] == adapter_stage
        and report["seed"] == specification["training_seed"]
        and report["examples"] == 800
        and report["optimizer_steps"] == 75
        and report["all_losses_finite"] is True
        and report["all_trainable_parameters_received_gradients"] is True,
        "adapter training report drift",
    )
    reload_report = json.loads((directory / "reload_validation.json").read_text("utf-8"))
    require(reload_report["passed"] is True, "adapter reload validation failed")
    return {
        "directory": directory,
        "ledger": ledger,
        "complete": complete,
        "report": report,
        "semantic_config_sha256": semantic_sha256,
    }


def validate_config_contract(config: Mapping[str, Any]) -> None:
    require(config["schema_version"] == 1, "config schema drift")
    require(config["protocol_version"] == PROTOCOL_VERSION, "protocol drift")
    require(
        config["dataset"]["id"] == DATASET_ID
        and config["dataset"]["revision"] == DATASET_REVISION
        and config["dataset"]["source_split"] == "validation"
        and config["dataset"]["expected_source_rows"] == 3270,
        "dataset contract drift",
    )
    model = config["model"]
    require(
        model["id"] == MODEL_ID
        and model["revision"] == MODEL_REVISION
        and model["dtype"] == "bfloat16"
        and model["peft_adapters_required"] is True
        and model["base_parameter_count"] == EXPECTED_BASE_PARAMETERS
        and model["total_parameter_count_with_adapter"] == EXPECTED_TOTAL_PARAMETERS,
        "model contract drift",
    )
    environment = config["environment"]
    require(
        environment["python"] == EXPECTED_PYTHON_VERSION
        and environment["gpu"] == EXPECTED_GPU_NAME
        and float(environment["gpu_vram_gib"]) == EXPECTED_GPU_VRAM_GIB
        and tuple(environment["compute_capability"]) == EXPECTED_COMPUTE_CAPABILITY
        and tuple(environment["excluded_packages"]) == ("torchao",)
        and dict(environment["package_versions"]) == EXPECTED_PACKAGE_VERSIONS,
        "environment contract drift",
    )
    input_config = config["input"]
    require(
        input_config["source_manifest_sha256"]
        == baseline.EXPECTED_FILE_SHA256["data/manifests/test.jsonl"]
        and input_config["degradation_manifest_sha256"]
        == baseline.EXPECTED_FILE_SHA256["data/degradations/test.jsonl"]
        and input_config["labels_available_to_model_inference"] is False
        and input_config["degradation_rows"] == 2400,
        "input contract drift",
    )
    require(tuple(config["adapters"]) == ALLOWED_ADAPTERS, "adapter order drift")
    expected_seeds = (20260811, 20260812, 20260813)
    require(
        tuple(config["adapters"][name]["training_seed"] for name in ALLOWED_ADAPTERS)
        == expected_seeds,
        "adapter seed drift",
    )
    require(tuple(config["stages"]) == ALLOWED_STAGES, "stage order drift")
    require(
        config["stages"]["smoke12"]["rows"] == 12
        and config["stages"]["full"]["rows"] == 2400,
        "stage size drift",
    )
    require(
        config["output"]["include_ground_truth"] is False
        and config["output"]["include_correctness"] is False
        and config["output"]["include_question"] is False
        and config["output"]["include_passage"] is False
        and config["output"]["include_prompt"] is False,
        "output isolation drift",
    )


def verify_static_files(config: Mapping[str, Any]) -> None:
    for mapping_name in ("prerequisites", "implementation"):
        for relative, expected in config[mapping_name].items():
            require(expected != "PENDING", f"pending digest: {relative}")
            path = _repo_path(relative)
            require(path.is_file() and not path.is_symlink(), f"missing file: {relative}")
            require(file_sha256(path) == expected, f"file digest drift: {relative}")


def _load_environment_and_adapter(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any]:
    validate_torchao_absent()
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(sys.version.split()[0] == EXPECTED_PYTHON_VERSION, "Python version drift")
    require(platform.platform() == config["environment"]["platform"], "platform drift")
    require(package_versions() == EXPECTED_PACKAGE_VERSIONS, "package version drift")
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(torch.cuda.device_count() == 1, "exactly one CUDA device required")
    require(torch.cuda.get_device_name(0) == EXPECTED_GPU_NAME, "unexpected GPU")
    require(tuple(torch.cuda.get_device_capability(0)) == EXPECTED_COMPUTE_CAPABILITY, "compute capability drift")
    observed_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    require(abs(observed_vram - EXPECTED_GPU_VRAM_GIB) <= 0.10, "GPU memory drift")
    require(torch.version.cuda == "12.8" and torch.cuda.is_bf16_supported(), "CUDA/BF16 drift")
    api = HfApi()
    require(api.model_info(MODEL_ID, revision=MODEL_REVISION, token=False).sha == MODEL_REVISION, "model revision resolution drift")
    require(api.dataset_info(DATASET_ID, revision=DATASET_REVISION, token=False).sha == DATASET_REVISION, "dataset revision resolution drift")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=False,
        trust_remote_code=False,
        use_fast=True,
        padding_side="left",
    )
    require(tokenizer.__class__.__name__ == "Qwen2Tokenizer", "tokenizer class drift")
    generation = config["inference"]["generation"]
    require(
        tokenizer.pad_token_id == generation["pad_token_id"]
        and tokenizer.eos_token_id == generation["eos_token_ids"][0],
        "tokenizer special-token drift",
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=False,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map=0,
    )
    require(base.__class__.__name__ == "Qwen2ForCausalLM", "base model class drift")
    require(sum(parameter.numel() for parameter in base.parameters()) == EXPECTED_BASE_PARAMETERS, "base parameter count drift")
    model = PeftModel.from_pretrained(
        base,
        checkpoint["directory"] / "adapter",
        is_trainable=False,
    )
    require(model.__class__.__name__ == "PeftModelForCausalLM", "PEFT model class drift")
    require(sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_TOTAL_PARAMETERS, "adapter total parameter count drift")
    require(not any(parameter.requires_grad for parameter in model.parameters()), "inference model has trainable parameters")
    require("logits_to_keep" in inspect.signature(base.forward).parameters, "model lacks logits_to_keep")
    require(model.get_input_embeddings().weight.dtype == torch.bfloat16, "model dtype drift")
    model.eval()
    require(model.training is False, "adapter model failed to enter eval mode")
    seed = int(config["inference"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    validation_split = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="validation",
        verification_mode="all_checks",
        token=False,
    )
    return torch, tokenizer, model, validation_split


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def publish_artifacts(
    output_directory: Path,
    *,
    predictions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    output_directory = baseline._ensure_safe_directory(output_directory)
    prediction_payload = adapter_prediction_jsonl_bytes(predictions)
    base_payloads = {
        "predictions.jsonl": prediction_payload,
        "provenance.json": canonical_bytes(provenance),
        "summary.json": canonical_bytes(summary),
    }
    ledger = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(base_payloads.items())
    }
    payloads = dict(base_payloads)
    payloads["artifact_hashes.json"] = canonical_bytes(ledger)
    completion = canonical_bytes(
        {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "artifact_hashes_sha256": sha256_bytes(payloads["artifact_hashes.json"]),
            "complete": True,
        }
    )
    statuses = {}
    for name in ("predictions.jsonl", "provenance.json", "summary.json", "artifact_hashes.json"):
        statuses[name] = baseline.atomic_refuse_or_verify(output_directory / name, payloads[name])
    statuses["COMPLETE.json"] = baseline.atomic_refuse_or_verify(output_directory / "COMPLETE.json", completion)
    hashes = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(payloads.items())
    }
    hashes["COMPLETE.json"] = {"bytes": len(completion), "sha256": sha256_bytes(completion)}
    return statuses, hashes


def write_run_manifest(
    *,
    adapter_stage: str,
    inference_stage: str,
    status: str,
    started_at: str,
    details: Mapping[str, Any],
) -> Path:
    ended_at = _timestamp()
    root = baseline._ensure_safe_directory(
        Path("outputs/manifests/lora_inference") / adapter_stage
    )
    stamp = ended_at.replace(":", "-").replace("+", "_")
    destination = root / f"{inference_stage}-{status}-{stamp}.json"
    payload = canonical_bytes(
        {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "adapter_stage": adapter_stage,
            "inference_stage": inference_stage,
            "status": status,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "git_head": git_head(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
            **dict(details),
        }
    )
    require(baseline.atomic_refuse_or_verify(destination, payload) == "created", "run manifest collision")
    return destination


def run(
    adapter_stage: str,
    inference_stage: str,
    config_path: Path,
) -> tuple[dict[str, Any], Path]:
    import yaml

    require(adapter_stage in ALLOWED_ADAPTERS, "unknown adapter")
    require(inference_stage in ALLOWED_STAGES, "unknown inference stage")
    config_path = _repo_path(config_path)
    config_payload = config_path.read_bytes()
    config = yaml.safe_load(config_payload.decode("utf-8"))
    require(isinstance(config, dict), "configuration is not a mapping")
    validate_config_contract(config)
    verify_static_files(config)
    specification = config["adapters"][adapter_stage]
    checkpoint = validate_checkpoint(adapter_stage, specification)

    baseline.validate_hash_ledgers(config)
    preparation, *_ = baseline._runtime_dependencies()
    source_rows = preparation.read_verified_jsonl(
        _repo_path(config["input"]["source_manifest_path"]),
        expected_sha256=config["input"]["source_manifest_sha256"],
        expected_rows=int(config["input"]["source_examples"]),
    )
    degradation_rows = baseline.read_jsonl_verified(
        _repo_path(config["input"]["degradation_manifest_path"]),
        expected_sha256=config["input"]["degradation_manifest_sha256"],
        expected_rows=int(config["input"]["degradation_rows"]),
    )
    baseline.validate_degradation_rows(degradation_rows)
    stage_rows = baseline.select_stage_rows(degradation_rows, inference_stage)
    stage_projection_sha256 = baseline.selected_stage_projection_sha256(stage_rows)

    torch_module, tokenizer, model, validation_split = _load_environment_and_adapter(config, checkpoint)
    data_config = yaml.safe_load(_repo_path("configs/data.yaml").read_text("utf-8"))
    degradation_config = yaml.safe_load(_repo_path("configs/degradations.yaml").read_text("utf-8"))
    prompt_spec = preparation.prompt_spec_from_configs(data_config, degradation_config)
    source_report = preparation.verify_full_validation_source(
        validation_split,
        expected_rows=int(config["dataset"]["expected_source_rows"]),
        expected_sha256=str(config["dataset"]["canonical_source_sha256"]),
    )
    selected_inputs, selection_report = preparation.reconstruct_selected_inputs(
        source_rows,
        validation_split,
        tokenizer,
        prompt_spec,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        selection_seed=int(data_config["selection"]["seed"]),
        eligibility_max_prompt_tokens=int(data_config["selection"]["original_prompt_eligibility_max_tokens"]),
    )
    examples = baseline.reconstruct_stage_examples(
        stage_rows=stage_rows,
        selected_inputs=selected_inputs,
        tokenizer=tokenizer,
        prompt_spec=prompt_spec,
        seed=int(degradation_config["protocol"]["seed"]),
        maximum_fragment_tokens=int(degradation_config["distractor"]["maximum_fragment_tokens"]),
    )
    del source_rows, validation_split, selected_inputs

    config_sha256 = sha256_bytes(config_payload)
    effective_config_sha256 = sha256_bytes(
        canonical_bytes(
            {
                "adapter_checkpoint_ledger_sha256": specification["checkpoint_ledger_sha256"],
                "adapter_stage": adapter_stage,
                "adapter_weights_sha256": specification["adapter_weights_sha256"],
                "config_sha256": config_sha256,
                "protocol_version": PROTOCOL_VERSION,
            }
        )
    )
    output_directory = _repo_path(config["output"]["root"]) / adapter_stage / inference_stage
    batch_directory = output_directory / config["output"]["batch_shard_directory"]
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    core_predictions, shard_counts = baseline.run_model_batches(
        examples=examples,
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        config=config,
        config_sha256=effective_config_sha256,
        batch_directory=batch_directory,
    )
    torch_module.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    checkpoint_relative = Path(specification["checkpoint_path"]).as_posix()
    predictions = [
        wrap_prediction_row(
            row,
            adapter_stage=adapter_stage,
            adapter_training_seed=int(specification["training_seed"]),
            adapter_checkpoint_path=checkpoint_relative,
            adapter_checkpoint_ledger_sha256=str(specification["checkpoint_ledger_sha256"]),
            adapter_weights_sha256=str(specification["adapter_weights_sha256"]),
            adapter_semantic_config_sha256=str(specification["semantic_adapter_config_sha256"]),
        )
        for row in core_predictions
    ]
    validate_adapter_prediction_rows(predictions)
    prediction_payload = adapter_prediction_jsonl_bytes(predictions)
    summary = {
        **summarize_adapter_prediction_rows(predictions),
        "inference_stage": inference_stage,
        "base_inputs": len({row["input_id"] for row in predictions}),
        "prediction_jsonl_sha256": sha256_bytes(prediction_payload),
        "stage_projection_sha256": stage_projection_sha256,
        "selection": config["stages"][inference_stage]["selection"],
    }
    provenance = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "adapter_stage": adapter_stage,
        "adapter_training_seed": specification["training_seed"],
        "adapter_checkpoint_path": checkpoint_relative,
        "adapter_checkpoint_ledger_sha256": specification["checkpoint_ledger_sha256"],
        "adapter_weights_sha256": specification["adapter_weights_sha256"],
        "adapter_raw_config_sha256": specification["raw_adapter_config_sha256"],
        "adapter_semantic_config_sha256": specification["semantic_adapter_config_sha256"],
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": str(model.get_input_embeddings().weight.dtype)},
        "inputs": {
            "source_manifest_sha256": config["input"]["source_manifest_sha256"],
            "degradation_manifest_sha256": config["input"]["degradation_manifest_sha256"],
            "full_selected_label_blind_projection_sha256": selection_report["label_blind_projection_sha256"],
            "stage_projection_sha256": stage_projection_sha256,
            "full_validation_source_sha256": source_report["canonical_sha256"],
        },
        "configuration": {"path": config_path.relative_to(REPO_ROOT).as_posix(), "sha256": config_sha256, "effective_adapter_sha256": effective_config_sha256},
        "labels_available_to_model_inference": False,
        "ground_truth_persisted": False,
        "correctness_persisted": False,
        "generated_continuation_persisted": True,
    }
    statuses, artifact_hashes = publish_artifacts(
        output_directory,
        predictions=predictions,
        summary=summary,
        provenance=provenance,
    )
    details = {
        "command": [sys.executable, "scripts/run_lora_inference.py", "--adapter", adapter_stage, "--stage", inference_stage, "--config", config_path.relative_to(REPO_ROOT).as_posix()],
        "inference_seconds": inference_seconds,
        "gpu": torch_module.cuda.get_device_name(0),
        "peak_allocated_gpu_gib": torch_module.cuda.max_memory_allocated() / (1024**3),
        "output_directory": output_directory.relative_to(REPO_ROOT).as_posix(),
        "artifact_hashes": artifact_hashes,
        "artifact_statuses": statuses,
        "batch_shards": shard_counts,
        "summary": summary,
    }
    return details, output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run audited LoRA-adapter confidence inference")
    parser.add_argument("--adapter", required=True, choices=ALLOWED_ADAPTERS)
    parser.add_argument("--stage", required=True, choices=ALLOWED_STAGES)
    parser.add_argument("--config", type=Path, default=Path("configs/lora_inference.yaml"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = _timestamp()
    started = time.perf_counter()
    try:
        details, output_directory = run(args.adapter, args.stage, args.config)
        manifest = write_run_manifest(
            adapter_stage=args.adapter,
            inference_stage=args.stage,
            status="success",
            started_at=started_at,
            details={**details, "elapsed_seconds": time.perf_counter() - started},
        )
        print("LORA ADAPTER INFERENCE: PASS")
        print(f"adapter={args.adapter}")
        print(f"stage={args.stage}")
        print(f"output_directory={output_directory.relative_to(REPO_ROOT).as_posix()}")
        print("summary=" + canonical_json(details["summary"]))
        print(f"run_manifest={manifest.relative_to(REPO_ROOT).as_posix()}")
        print("SCOPE: label-blind LoRA-adapter predictions only; no calibration, aggregation, or accuracy claim is made.")
        return 0
    except Exception as error:
        try:
            manifest = write_run_manifest(
                adapter_stage=args.adapter,
                inference_stage=args.stage,
                status="failed",
                started_at=started_at,
                details={
                    "elapsed_seconds": time.perf_counter() - started,
                    "failure_type": error.__class__.__name__,
                    "failure_message_sanitized": " ".join(str(error).split())[:500],
                },
            )
            print(f"FAILED_RUN_MANIFEST={manifest.relative_to(REPO_ROOT).as_posix()}", file=sys.stderr)
        except Exception as manifest_error:
            print(f"FAILED_RUN_MANIFEST_WRITE_ERROR={manifest_error.__class__.__name__}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
