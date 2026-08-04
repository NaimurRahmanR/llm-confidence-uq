from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for directory in (SRC_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_baseline as baseline  # noqa: E402
import run_lora_inference as lora  # noqa: E402
from llm_confidence_uq.calibration_inference import (  # noqa: E402
    CalibrationDataset,
    PROTOCOL_VERSION,
    STAGES,
    label_blind_projection_sha256,
    make_prediction_row,
    prediction_jsonl_bytes,
    reconstruct_original_examples,
    select_stage_rows,
    summarize_prediction_rows,
    validate_manifest_rows,
    validate_prediction_rows,
)
from llm_confidence_uq.inference import (  # noqa: E402
    EXPECTED_CLASS_TOKEN_IDS,
    PromptCollator,
    binary_confidence_from_logits,
    canonical_bytes,
    canonical_json,
    effective_expressed_parse,
    gather_last_token_logits,
    position_ids_from_attention_mask,
    sha256_bytes,
)


ALLOWED_METHODS = ("baseline", "full_seed_1", "full_seed_2", "full_seed_3")
DATASET_ID = "google/boolq"
DATASET_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class CalibrationInferenceRunError(RuntimeError):
    """Raised when a calibration inference invariant fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationInferenceRunError(message)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    root = Path(os.path.abspath(REPO_ROOT))
    absolute = Path(os.path.abspath(candidate))
    require(absolute == root or root in absolute.parents, f"path escapes repository: {absolute}")
    current = absolute
    while True:
        require(not current.is_symlink(), f"symbolic link forbidden: {current}")
        if current == root:
            break
        current = current.parent
    return absolute.resolve(strict=False)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _read_canonical_manifest(path: Path, expected_sha256: str, expected_rows: int) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), "calibration manifest missing or unsafe")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, "calibration manifest SHA-256 mismatch")
    require(payload.endswith(b"\n") and b"\r\n" not in payload, "calibration manifest newline drift")
    lines = payload.splitlines()
    require(len(lines) == expected_rows and all(lines), "calibration manifest row-count drift")
    rows = [json.loads(line.decode("utf-8")) for line in lines]
    require(
        b"".join(canonical_bytes(row) for row in rows) == payload,
        "calibration manifest is not canonical JSONL",
    )
    return rows


def validate_config(config: Mapping[str, Any], config_path: Path) -> None:
    require(config["schema_version"] == 1, "configuration schema drift")
    require(config["protocol_version"] == PROTOCOL_VERSION, "configuration protocol drift")
    require(config["dataset"]["id"] == DATASET_ID, "dataset ID drift")
    require(config["dataset"]["revision"] == DATASET_REVISION, "dataset revision drift")
    require(config["dataset"]["source_split"] == "train", "source split drift")
    require(config["input"]["research_split"] == "calibration", "research split drift")
    require(config["input"]["evidence_condition"] == "original", "calibration evidence drift")
    require(config["input"]["labels_available_to_model_inference"] is False, "labels exposed to inference")
    require(int(config["input"]["rows"]) == 200, "calibration size drift")
    require(set(config["methods"]) == set(ALLOWED_METHODS), "method set drift")
    require(set(config["selection"]["stages"]) == set(STAGES), "stage set drift")
    for stage, expected_rows in STAGES.items():
        require(int(config["selection"]["stages"][stage]["rows"]) == expected_rows, f"{stage} size drift")
    for relative, expected_sha256 in config["implementation"].items():
        path = _repo_path(relative)
        require(path.is_file(), f"missing implementation file: {relative}")
        require(lora.file_sha256(path) == expected_sha256, f"implementation hash drift: {relative}")
    require(config_path == _repo_path("configs/calibration_inference.yaml"), "unexpected config path")


def _load_method(method: str, config: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Mapping[str, Any], Mapping[str, Any] | None]:
    specification = config["methods"][method]
    inference_config_path = _repo_path(specification["inference_config_path"])
    payload = inference_config_path.read_bytes()
    require(sha256_bytes(payload) == specification["inference_config_sha256"], "inference config hash drift")
    import yaml

    inference_config = yaml.safe_load(payload.decode("utf-8"))
    require(isinstance(inference_config, dict), "inference configuration is not a mapping")
    if method == "baseline":
        baseline.validate_config_contract(inference_config)
        torch_module, tokenizer, model, validation_split = baseline._load_and_validate_environment(inference_config)
        return torch_module, tokenizer, model, validation_split, inference_config, None

    lora.validate_config_contract(inference_config)
    adapter_specification = inference_config["adapters"][method]
    checkpoint = lora.validate_checkpoint(method, adapter_specification)
    torch_module, tokenizer, model, validation_split = lora._load_environment_and_adapter(
        inference_config,
        checkpoint,
    )
    return torch_module, tokenizer, model, validation_split, inference_config, adapter_specification


def _write_run_manifest(
    *,
    method: str,
    stage: str,
    status: str,
    started_at: str,
    details: Mapping[str, Any],
) -> Path:
    ended_at = _timestamp()
    root = baseline._ensure_safe_directory(
        _repo_path("outputs/manifests/calibration_inference") / method
    )
    stamp = ended_at.replace(":", "-").replace("+", "_")
    destination = root / f"{stage}-{status}-{stamp}.json"
    payload = canonical_bytes(
        {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "method": method,
            "stage": stage,
            "status": status,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "git_head": _git_head(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **dict(details),
        }
    )
    require(baseline.atomic_refuse_or_verify(destination, payload) == "created", "run-manifest collision")
    return destination


def _run_batches(*, examples, model, tokenizer, torch_module, config, config_sha256: str, method: str, adapter_specification):
    from transformers import RepetitionPenaltyLogitsProcessor

    require("prompt_ignore_length" in inspect.signature(RepetitionPenaltyLogitsProcessor).parameters, "repetition processor contract drift")
    dataset = CalibrationDataset(examples)
    collator = PromptCollator(tokenizer, maximum_prompt_tokens=int(config["prompt"]["maximum_prompt_tokens"]))
    loader = torch_module.utils.data.DataLoader(
        dataset,
        batch_size=int(config["inference"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["inference"]["num_workers"]),
        pin_memory=bool(config["inference"]["pin_memory"]),
        drop_last=False,
        collate_fn=collator,
    )
    generation = config["inference"]["generation"]
    maximum = int(generation["max_new_tokens"])
    eos_ids = tuple(int(value) for value in generation["eos_token_ids"])
    pad_id = int(generation["pad_token_id"])
    device = model.get_input_embeddings().weight.device
    adapter = None if adapter_specification is None else {
        "adapter_stage": method,
        "adapter_training_seed": int(adapter_specification["training_seed"]),
        "adapter_checkpoint_ledger_sha256": str(adapter_specification["checkpoint_ledger_sha256"]),
        "adapter_weights_sha256": str(adapter_specification["adapter_weights_sha256"]),
        "adapter_semantic_config_sha256": str(adapter_specification["semantic_adapter_config_sha256"]),
    }
    rows = []
    for batch in loader:
        batch_examples = tuple(batch["examples"])
        model_inputs = {key: value.to(device, non_blocking=False) for key, value in batch["model_inputs"].items()}
        attention_mask = model_inputs["attention_mask"]
        forward_inputs = dict(model_inputs)
        forward_inputs["position_ids"] = position_ids_from_attention_mask(attention_mask)
        input_width = int(model_inputs["input_ids"].shape[1])
        processor = RepetitionPenaltyLogitsProcessor(
            penalty=float(generation["repetition_penalty"]),
            prompt_ignore_length=input_width,
        )
        with torch_module.inference_mode():
            outputs = model(**forward_inputs, use_cache=False, return_dict=True, output_hidden_states=False, output_attentions=False, logits_to_keep=1)
            confidence = binary_confidence_from_logits(
                gather_last_token_logits(outputs.logits, attention_mask),
                yes_token_id=EXPECTED_CLASS_TOKEN_IDS[0],
                no_token_id=EXPECTED_CLASS_TOKEN_IDS[1],
            )
            generated = model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=maximum,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=list(eos_ids),
                repetition_penalty=float(generation["inherited_repetition_penalty_override"]),
                logits_processor=[processor],
            )
        require(generated.shape[0] == len(batch_examples) and generated.shape[1] >= input_width, "generated shape drift")
        require(bool(torch_module.equal(generated[:, :input_width], model_inputs["input_ids"])), "generated prefix drift")
        for index, example in enumerate(batch_examples):
            suffix, ended, reached = baseline._generation_content(
                generated[index, input_width:].detach().cpu().tolist(),
                eos_token_ids=eos_ids,
                pad_token_id=pad_id,
                special_token_ids=tokenizer.all_special_ids,
                maximum_new_tokens=maximum,
            )
            completion = tokenizer.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            parsed = effective_expressed_parse(completion, generation_reached_max_new_tokens=reached)
            prediction_index = int(confidence.token_prediction_index[index].detach().cpu().item())
            token_prediction = ("Yes", "No")[prediction_index]
            rows.append(make_prediction_row(
                example,
                method=method,
                model_id=MODEL_ID,
                model_revision=lora.MODEL_REVISION,
                inference_config_sha256=config_sha256,
                inference_seed=int(config["inference"]["seed"]),
                yes_token_id=EXPECTED_CLASS_TOKEN_IDS[0], no_token_id=EXPECTED_CLASS_TOKEN_IDS[1],
                yes_logit=baseline._float_at(confidence.yes_logit, index), no_logit=baseline._float_at(confidence.no_logit, index),
                p_yes_binary=baseline._float_at(confidence.p_yes_binary, index), p_no_binary=baseline._float_at(confidence.p_no_binary, index),
                p_yes_full_vocabulary=baseline._float_at(confidence.p_yes_full_vocabulary, index),
                p_no_full_vocabulary=baseline._float_at(confidence.p_no_full_vocabulary, index),
                yes_no_full_vocabulary_mass=baseline._float_at(confidence.yes_no_full_vocabulary_mass, index),
                token_prediction=token_prediction, token_confidence=baseline._float_at(confidence.token_confidence, index),
                predictive_entropy=baseline._float_at(confidence.predictive_entropy, index),
                generated_continuation=completion, generated_continuation_tokens=len(suffix),
                generation_ended_with_eos=ended, generation_reached_max_new_tokens=reached,
                parsed=parsed, adapter=adapter,
            ))
    validate_prediction_rows(rows)
    return rows


def _publish(output_directory: Path, *, predictions, summary, provenance):
    output_directory = baseline._ensure_safe_directory(output_directory)
    base_payloads = {
        "predictions.jsonl": prediction_jsonl_bytes(predictions),
        "provenance.json": canonical_bytes(provenance),
        "summary.json": canonical_bytes(summary),
    }
    ledger = {name: {"bytes": len(payload), "sha256": sha256_bytes(payload)} for name, payload in sorted(base_payloads.items())}
    payloads = {**base_payloads, "artifact_hashes.json": canonical_bytes(ledger)}
    completion = canonical_bytes({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "artifact_hashes_sha256": sha256_bytes(payloads["artifact_hashes.json"]), "complete": True})
    statuses = {name: baseline.atomic_refuse_or_verify(output_directory / name, payload) for name, payload in payloads.items()}
    statuses["COMPLETE.json"] = baseline.atomic_refuse_or_verify(output_directory / "COMPLETE.json", completion)
    hashes = {name: {"bytes": len(payload), "sha256": sha256_bytes(payload)} for name, payload in payloads.items()}
    hashes["COMPLETE.json"] = {"bytes": len(completion), "sha256": sha256_bytes(completion)}
    return statuses, hashes


def run(method: str, stage: str, config_path: Path) -> tuple[dict[str, Any], Path]:
    import yaml

    require(method in ALLOWED_METHODS, "unknown calibration inference method")
    require(stage in STAGES, "unknown calibration inference stage")
    config_path = _repo_path(config_path)
    config_payload = config_path.read_bytes()
    config = yaml.safe_load(config_payload.decode("utf-8"))
    require(isinstance(config, dict), "calibration configuration is not a mapping")
    validate_config(config, config_path)

    manifest_rows = _read_canonical_manifest(
        _repo_path(config["input"]["manifest_path"]),
        str(config["input"]["manifest_sha256"]),
        int(config["input"]["rows"]),
    )
    validate_manifest_rows(
        manifest_rows,
        expected_rows=int(config["input"]["rows"]),
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        selection_seed=int(config["selection"]["seed"]),
        eligibility_max_prompt_tokens=int(config["selection"]["eligibility_max_prompt_tokens"]),
    )
    selected_rows = select_stage_rows(manifest_rows, stage)
    projection_sha256 = label_blind_projection_sha256(selected_rows)

    torch_module, tokenizer, model, validation_split, inference_config, adapter_specification = _load_method(method, config)
    del validation_split
    from datasets import load_dataset
    source_split = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
        verification_mode="all_checks",
        token=False,
    )
    preparation, *_ = baseline._runtime_dependencies()
    data_config = yaml.safe_load(_repo_path("configs/data.yaml").read_text("utf-8"))
    degradation_config = yaml.safe_load(_repo_path("configs/degradations.yaml").read_text("utf-8"))
    prompt_spec = preparation.prompt_spec_from_configs(data_config, degradation_config)
    source_report = preparation.verify_full_validation_source(
        source_split,
        expected_rows=int(config["dataset"]["expected_source_rows"]),
        expected_sha256=str(config["dataset"]["canonical_source_sha256"]),
    )
    examples = reconstruct_original_examples(
        selected_rows,
        source_split,
        render_prompt=lambda passage, question: preparation.render_locked_prompt(
            tokenizer,
            prompt_spec,
            passage=passage,
            question=question,
        ),
        encode_prompt=lambda prompt: preparation.encode_text(tokenizer, prompt),
        make_input_id=preparation.make_label_blind_input_id,
        dataset_revision=DATASET_REVISION,
    )
    del manifest_rows, source_split

    effective_config_sha256 = sha256_bytes(
        canonical_bytes(
            {
                "calibration_config_sha256": sha256_bytes(config_payload),
                "method": method,
                "method_inference_config_sha256": config["methods"][method]["inference_config_sha256"],
                "protocol_version": PROTOCOL_VERSION,
                "selection_projection_sha256": projection_sha256,
                "stage": stage,
                "adapter_checkpoint_ledger_sha256": (
                    None if adapter_specification is None else adapter_specification["checkpoint_ledger_sha256"]
                ),
            }
        )
    )
    output_directory = _repo_path(config["output"]["root"]) / method / stage
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    predictions = _run_batches(
        examples=examples,
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        config=inference_config,
        config_sha256=effective_config_sha256,
        method=method,
        adapter_specification=adapter_specification,
    )
    torch_module.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    summary = summarize_prediction_rows(predictions)

    summary = {
        **summary,
        "calibration_protocol_version": PROTOCOL_VERSION,
        "calibration_stage": stage,
        "research_split": "calibration",
        "evidence_condition": "original",
        "label_blind_selection_projection_sha256": projection_sha256,
        "labels_available_to_model_inference": False,
    }
    provenance = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "stage": stage,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
        "source_train_sha256": source_report["canonical_sha256"],
        "source_manifest_sha256": config["input"]["manifest_sha256"],
        "label_blind_selection_projection_sha256": projection_sha256,
        "configuration": {
            "path": config_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_bytes(config_payload),
            "effective_inference_sha256": effective_config_sha256,
        },
        "labels_available_to_model_inference": False,
        "ground_truth_persisted": False,
        "correctness_persisted": False,
    }
    statuses, artifact_hashes = _publish(
        output_directory,
        predictions=predictions,
        summary=summary,
        provenance=provenance,
    )
    details = {
        "command": [
            sys.executable,
            "scripts/run_calibration_inference.py",
            "--method",
            method,
            "--stage",
            stage,
            "--config",
            config_path.relative_to(REPO_ROOT).as_posix(),
        ],
        "output_directory": output_directory.relative_to(REPO_ROOT).as_posix(),
        "inference_seconds": inference_seconds,
        "gpu": torch_module.cuda.get_device_name(0),
        "peak_allocated_gpu_gib": torch_module.cuda.max_memory_allocated() / (1024**3),
        "artifact_statuses": statuses,
        "artifact_hashes": artifact_hashes,
        "summary": summary,
    }
    return details, output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run label-blind original-evidence calibration inference")
    parser.add_argument("--method", required=True, choices=ALLOWED_METHODS)
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
    parser.add_argument("--config", type=Path, default=Path("configs/calibration_inference.yaml"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = _timestamp()
    started = time.perf_counter()
    try:
        details, output_directory = run(args.method, args.stage, args.config)
        manifest = _write_run_manifest(
            method=args.method,
            stage=args.stage,
            status="success",
            started_at=started_at,
            details={**details, "elapsed_seconds": time.perf_counter() - started},
        )
        print("CALIBRATION INFERENCE: PASS")
        print(f"method={args.method}")
        print(f"stage={args.stage}")
        print(f"output_directory={output_directory.relative_to(REPO_ROOT).as_posix()}")
        print("summary=" + canonical_json(details["summary"]))
        print("artifact_statuses=" + canonical_json(details["artifact_statuses"]))
        print(f"run_manifest={manifest.relative_to(REPO_ROOT).as_posix()}")
        print("SCOPE: original-evidence calibration predictions only; labels were unavailable to model inference and no temperature was fitted.")
        return 0
    except Exception as error:
        failure_details = {
            "command": [
                sys.executable,
                "scripts/run_calibration_inference.py",
                "--method",
                args.method,
                "--stage",
                args.stage,
                "--config",
                Path(args.config).as_posix(),
            ],
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        try:
            manifest = _write_run_manifest(
                method=args.method,
                stage=args.stage,
                status="failed",
                started_at=started_at,
                details=failure_details,
            )
            print(f"FAILED_RUN_MANIFEST={manifest.relative_to(REPO_ROOT).as_posix()}", file=sys.stderr)
        finally:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
