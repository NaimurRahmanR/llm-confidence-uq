from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import errno
import importlib.metadata
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
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from llm_confidence_uq.inference import (  # noqa: E402
    EXPECTED_CLASS_TOKEN_IDS,
    InferenceDataset,
    InferenceExample,
    PROTOCOL_VERSION,
    PromptCollator,
    binary_confidence_from_logits,
    canonical_bytes,
    canonical_json,
    canonical_jsonl_bytes,
    contextual_class_token_ids,
    gather_last_token_logits,
    make_prediction_row,
    parse_expressed_continuation,
    position_ids_from_attention_mask,
    prediction_jsonl_bytes,
    sha256_bytes,
    sha256_text,
    summarize_prediction_rows,
    validate_prediction_rows,
)


EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_GPU_VRAM_GIB = 79.250732421875
EXPECTED_COMPUTE_CAPABILITY = (8, 0)
EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.14.0",
    "datasets": "4.0.0",
    "huggingface-hub": "1.23.0",
    "numpy": "2.0.2",
    "peft": "0.19.1",
    "pyarrow": "18.1.0",
    "pyyaml": "6.0.3",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.11.0+cu128",
    "transformers": "5.13.1",
}
EXPECTED_FILE_SHA256 = {
    "configs/data.yaml": "412fd6da74212e071159463e104329efefbc8fbe6b853c29c825e3ab37cb64ac",
    "configs/degradations.yaml": "303238ae02025c9e5cb59bbd061949b6798f5fde8936a1f0604d15f0552cf2e3",
    "src/llm_confidence_uq/degradations.py": "8d71c5305d04d888b68181df9d43e30e76ea4c6f6f56494a65e53697d8544337",
    "scripts/prepare_degradations.py": "aee819f1727a9113a4b2df2c8a946747dd90565b2659bd3dcbf635a4b3d13e86",
    "data/manifests/artifact_hashes.json": "8916df7fd59606377a7701f9216179a3a521f4f0ccbfba3a1ea113017353e640",
    "data/manifests/test.jsonl": "42a346cb58d0ef1530b5fb805cc798c0191a6f2884d2f1766a4d86d24817d588",
    "data/degradations/artifact_hashes.json": "5a6bce031837eaf76869f4ebefe15801dd0009df05fc125c0673c0c76773be9f",
    "data/degradations/test.jsonl": "82c6285ee0c236c8b75d65632b1fda3e85f8e0c492f89aef877d06c17fa1255c",
}
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_PARAMETER_COUNT = 1_543_714_304
DATASET_ID = "google/boolq"
DATASET_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
EXPECTED_CONDITIONS = (
    "original",
    "lexical_evidence_removal",
    "prefix_truncation_50",
    "irrelevant_distractor",
    "lexical_contradiction",
    "no_passage",
)
ALLOWED_STAGES = ("smoke12", "smoke60", "full")
RUN_MANIFEST_ROOT = Path("outputs/manifests/baseline")


class BaselineRunError(RuntimeError):
    """Raised when a pinned baseline-inference invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineRunError(message)


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
        require(not current.is_symlink(), f"symbolic links are forbidden: {current}")
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


def read_jsonl_verified(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> list[dict[str, Any]]:
    path = _repo_path(path)
    require(path.exists() and path.is_file(), f"missing JSONL file: {path}")
    require(not path.is_symlink(), f"JSONL file is a symlink: {path}")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"JSONL SHA-256 mismatch: {path}")
    require(payload.endswith(b"\n"), f"JSONL lacks terminal newline: {path}")
    require(b"\r" not in payload, f"JSONL contains carriage returns: {path}")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    require(len(rows) == expected_rows, f"JSONL row-count mismatch: {path}")
    require(all(isinstance(row, dict) for row in rows), "JSONL rows must be objects")
    return rows


def read_json_verified(path: Path, *, expected_sha256: str) -> Any:
    path = _repo_path(path)
    require(path.exists() and path.is_file(), f"missing JSON file: {path}")
    require(not path.is_symlink(), f"JSON file is a symlink: {path}")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"JSON SHA-256 mismatch: {path}")
    return json.loads(payload.decode("utf-8"))


def package_versions() -> dict[str, str]:
    values: dict[str, str] = {}
    for name in EXPECTED_PACKAGE_VERSIONS:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = "not-installed"
    return values


def validate_config_contract(config: Mapping[str, Any]) -> None:
    require(config.get("schema_version") == 1, "baseline schema version drift")
    require(config.get("protocol_version") == PROTOCOL_VERSION, "baseline protocol drift")

    dataset = config["dataset"]
    require(
        dataset["id"] == DATASET_ID
        and dataset["revision"] == DATASET_REVISION
        and dataset["source_split"] == "validation"
        and dataset["expected_source_rows"] == 3270,
        "dataset contract drift",
    )
    require(
        dataset["canonical_source_sha256"]
        == "c81b6520f1a51a8b96de45420b3d1f78184aee00373c77852132b297f2e567ef",
        "dataset source digest drift",
    )

    model = config["model"]
    require(
        model["id"] == MODEL_ID
        and model["revision"] == MODEL_REVISION
        and model["tokenizer_id"] == MODEL_ID
        and model["tokenizer_revision"] == MODEL_REVISION,
        "model/tokenizer identity drift",
    )
    require(
        model["tokenizer_class"] == "Qwen2Tokenizer"
        and model["model_class"] == "Qwen2ForCausalLM",
        "model/tokenizer class drift",
    )
    require(model["parameter_count"] == MODEL_PARAMETER_COUNT, "parameter-count drift")
    require(
        model["dtype"] == "float16"
        and model["device"] == "cuda:0"
        and model["trust_remote_code"] is False
        and model["use_safetensors"] is True
        and model["peft_adapters_allowed"] is False,
        "model load policy drift",
    )

    environment = config["environment"]
    require(environment["python"] == EXPECTED_PYTHON_VERSION, "Python contract drift")
    require(
        environment["gpu"] == EXPECTED_GPU_NAME
        and float(environment["gpu_vram_gib"]) == EXPECTED_GPU_VRAM_GIB
        and tuple(environment["compute_capability"]) == EXPECTED_COMPUTE_CAPABILITY,
        "GPU contract drift",
    )
    require(
        dict(environment["package_versions"]) == EXPECTED_PACKAGE_VERSIONS,
        "package-version contract drift",
    )

    input_config = config["input"]
    require(
        input_config["source_manifest_path"] == "data/manifests/test.jsonl"
        and input_config["source_manifest_ledger_path"]
        == "data/manifests/artifact_hashes.json"
        and input_config["source_manifest_sha256"]
        == EXPECTED_FILE_SHA256["data/manifests/test.jsonl"]
        and input_config["source_manifest_ledger_sha256"]
        == EXPECTED_FILE_SHA256["data/manifests/artifact_hashes.json"]
        and input_config["source_examples"] == 400,
        "source-manifest contract drift",
    )
    require(
        input_config["degradation_manifest_path"] == "data/degradations/test.jsonl"
        and input_config["degradation_manifest_ledger_path"]
        == "data/degradations/artifact_hashes.json"
        and input_config["degradation_manifest_sha256"]
        == EXPECTED_FILE_SHA256["data/degradations/test.jsonl"]
        and input_config["degradation_manifest_ledger_sha256"]
        == EXPECTED_FILE_SHA256["data/degradations/artifact_hashes.json"]
        and input_config["degradation_rows"] == 2400
        and input_config["condition_count"] == 6,
        "degradation-manifest contract drift",
    )
    require(
        input_config["require_all_transformations_ok"] is True
        and input_config["require_all_runtime_status_ok"] is True
        and input_config["labels_available_to_model_inference"] is False,
        "input isolation/status policy drift",
    )

    prompt = config["prompt"]
    require(
        prompt["renderer_version"] == "qwen-chat-boolq-confidence-v1"
        and prompt["maximum_prompt_tokens"] == 1024
        and prompt["truncation_allowed"] is False
        and prompt["padding_side"] == "left"
        and prompt["add_special_tokens"] is False,
        "prompt contract drift",
    )

    classification = config["classification"]
    require(tuple(classification["class_order"]) == ("Yes", "No"), "class order drift")
    require(
        (
            classification["token_ids"]["Yes"],
            classification["token_ids"]["No"],
        )
        == EXPECTED_CLASS_TOKEN_IDS,
        "class token IDs drift",
    )
    require(
        classification["continuations"] == {"Yes": " Yes", "No": " No"}
        and classification["contextual_single_token_validation"] is True
        and classification["binary_probability_definition"]
        == "softmax-over-yes-no-logits"
        and classification["record_full_vocabulary_yes_no_mass"] is True
        and classification["argmax_tie_policy"] == "Yes",
        "classification policy drift",
    )

    forward = config["forward"]
    require(
        forward
        == {
            "use_cache": False,
            "return_dict": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "logits_to_keep": 1,
            "attention_derived_position_ids": True,
            "require_finite_full_vocabulary_logits": True,
        },
        "forward-pass contract drift",
    )

    inference_config = config["inference"]
    require(
        inference_config["seed"] == 20260803
        and inference_config["batch_size"] == 4
        and inference_config["num_workers"] == 0
        and inference_config["pin_memory"] is True
        and inference_config["shuffle"] is False
        and inference_config["model_mode"] == "eval"
        and inference_config["autograd_enabled"] is False,
        "inference contract drift",
    )
    generation = inference_config["generation"]
    require(
        generation["do_sample"] is False
        and generation["num_beams"] == 1
        and generation["max_new_tokens"] == 24
        and generation["use_cache"] is True
        and generation["eos_token_ids"] == [151645, 151643]
        and generation["pad_token_id"] == 151643
        and generation["repetition_penalty"] == 1.1
        and generation["repetition_penalty_scope"]
        == "generated-continuation-only"
        and generation["inherited_repetition_penalty_override"] == 1.0
        and generation["skip_special_tokens_on_decode"] is True
        and generation["clean_up_tokenization_spaces"] is False
        and generation["parser"] == "exact-continuation-v1"
        and generation["reject_max_new_tokens_reached"] is True
        and generation["persist_generated_continuation"] is True
        and generation["retain_invalid_parse_rows"] is True,
        "generation contract drift",
    )

    stages = config["stages"]
    require(
        stages["smoke12"]["base_inputs"] == 2
        and stages["smoke12"]["rows"] == 12
        and stages["smoke12"]["selection"]
        == "first-two-label-blind-input-ids"
        and stages["smoke60"]["base_inputs"] == 10
        and stages["smoke60"]["rows"] == 60
        and stages["smoke60"]["selection"]
        == "first-nine-plus-longest-unless-already-selected-then-first-unused"
        and stages["full"]["base_inputs"] == 400
        and stages["full"]["rows"] == 2400
        and stages["full"]["selection"] == "all-label-blind-inputs",
        "stage cardinality drift",
    )

    output = config["output"]
    require(
        output["root"] == "outputs/predictions/baseline"
        and output["run_manifest_root"] == "outputs/manifests/baseline"
        and output["batch_shards"] is True
        and output["batch_shard_directory"] == "batches"
        and output["resume_policy"] == "verify-identity-row-hashes-and-order"
        and output["atomic_writes"] is True
        and output["publish_completion_marker_last"] is True
        and output["refuse_differing_existing_artifacts"] is True
        and output["include_generated_continuation"] is True
        and output["include_ground_truth"] is False
        and output["include_correctness"] is False
        and output["include_question"] is False
        and output["include_passage"] is False
        and output["include_prompt"] is False
        and output["canonical_json"] is True
        and output["jsonl_terminal_newline"] is True,
        "output policy drift",
    )
    require(
        dict(config["prerequisites"]) == EXPECTED_FILE_SHA256,
        "prerequisite hash contract drift",
    )
    implementation = config["implementation"]
    require(
        set(implementation)
        == {
            "src/llm_confidence_uq/inference.py",
            "scripts/run_baseline.py",
            "tests/test_inference.py",
        }
        and all(
            isinstance(value, str)
            and len(value) == 64
            and set(value) <= set("0123456789abcdef")
            for value in implementation.values()
        ),
        "implementation hash contract drift",
    )


def validate_hash_ledgers(config: Mapping[str, Any]) -> None:
    input_config = config["input"]
    source_ledger = read_json_verified(
        _repo_path(input_config["source_manifest_ledger_path"]),
        expected_sha256=input_config["source_manifest_ledger_sha256"],
    )
    degradation_ledger = read_json_verified(
        _repo_path(input_config["degradation_manifest_ledger_path"]),
        expected_sha256=input_config["degradation_manifest_ledger_sha256"],
    )
    require(
        source_ledger["test.jsonl"]["sha256"]
        == input_config["source_manifest_sha256"],
        "source ledger does not bind test.jsonl",
    )
    require(
        degradation_ledger["test.jsonl"]["sha256"]
        == input_config["degradation_manifest_sha256"],
        "degradation ledger does not bind test.jsonl",
    )


def validate_degradation_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    require(len(rows) == 2400, "wrong degradation row count")
    require(len(rows) % len(EXPECTED_CONDITIONS) == 0, "incomplete input group")
    forbidden = {"answer", "correct", "label", "target"}
    transformation_ids: set[str] = set()
    observed_input_ids: list[str] = []
    for start in range(0, len(rows), len(EXPECTED_CONDITIONS)):
        group = rows[start : start + len(EXPECTED_CONDITIONS)]
        input_id = str(group[0]["input_id"])
        observed_input_ids.append(input_id)
        require(
            all(str(row["input_id"]) == input_id for row in group),
            "degradation input group is not contiguous",
        )
        for condition_index, (row, condition) in enumerate(
            zip(group, EXPECTED_CONDITIONS, strict=True)
        ):
            require(not (set(row) & forbidden), "degradation row exposes a target")
            without_hash = {
                key: value for key, value in row.items() if key != "row_sha256"
            }
            require(
                row["row_sha256"] == sha256_text(canonical_json(without_hash)),
                "degradation row hash mismatch",
            )
            require(
                int(row["condition_index"]) == condition_index
                and row["condition"] == condition,
                "per-input condition order drift",
            )
            require(
                row["transformation_status"] == "ok"
                and row["runtime_status"] == "ok"
                and row["runtime_reason_code"] is None,
                "frozen degradation row is not runnable",
            )
            prompt_tokens = row["prompt_tokens"]
            require(
                isinstance(prompt_tokens, int)
                and not isinstance(prompt_tokens, bool)
                and 0 < prompt_tokens <= 1024,
                "invalid frozen prompt-token count",
            )
            require(
                isinstance(row["rendered_prompt_sha256"], str)
                and len(row["rendered_prompt_sha256"]) == 64,
                "missing rendered-prompt digest",
            )
            transformation_id = str(row["transformation_id"])
            require(
                transformation_id not in transformation_ids,
                "duplicate transformation ID",
            )
            transformation_ids.add(transformation_id)
    require(len(observed_input_ids) == 400, "wrong base-input count")
    require(
        len(set(observed_input_ids)) == 400
        and observed_input_ids == sorted(observed_input_ids),
        "base-input IDs are duplicate or unordered",
    )


def select_stage_rows(
    rows: Sequence[Mapping[str, Any]],
    stage: str,
) -> list[dict[str, Any]]:
    """Select only identifiers and prompt-length metadata."""

    require(stage in ALLOWED_STAGES, f"unsupported stage: {stage}")
    input_order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        input_id = str(row["input_id"])
        if input_id not in seen:
            input_order.append(input_id)
            seen.add(input_id)
    require(bool(input_order), "no base inputs in degradation rows")
    if stage == "full":
        selected_ids = set(input_order)
    elif stage == "smoke12":
        require(len(input_order) >= 2, "smoke12 needs two base inputs")
        selected_ids = set(input_order[:2])
    else:
        require(len(input_order) >= 10, "smoke60 needs ten base inputs")
        runnable = [
            row
            for row in rows
            if row["runtime_status"] == "ok"
            and isinstance(row["prompt_tokens"], int)
            and not isinstance(row["prompt_tokens"], bool)
        ]
        require(bool(runnable), "smoke60 has no runnable prompt")
        longest = sorted(
            runnable,
            key=lambda row: (-int(row["prompt_tokens"]), str(row["input_id"])),
        )[0]
        longest_id = str(longest["input_id"])
        chosen = list(input_order[:9])
        if longest_id not in chosen:
            chosen.append(longest_id)
        else:
            chosen.append(
                next(input_id for input_id in input_order if input_id not in chosen)
            )
        require(
            len(chosen) == 10 and len(set(chosen)) == 10,
            "smoke60 selection is not unique",
        )
        selected_ids = set(chosen)
    selected = [
        dict(row) for row in rows if str(row["input_id"]) in selected_ids
    ]
    expected = {"smoke12": 12, "smoke60": 60, "full": 2400}[stage]
    require(
        len(selected) == expected,
        f"{stage} selected {len(selected)} rows, expected {expected}",
    )
    return selected


def selected_stage_projection_sha256(
    stage_rows: Sequence[Mapping[str, Any]],
) -> str:
    projection = [
        {
            "condition_index": int(row["condition_index"]),
            "input_id": str(row["input_id"]),
            "row_sha256": str(row["row_sha256"]),
            "transformation_id": str(row["transformation_id"]),
        }
        for row in stage_rows
    ]
    return sha256_bytes(canonical_jsonl_bytes(projection))


def _runtime_dependencies() -> tuple[Any, ...]:
    # These imports are intentionally deferred so parser/config/unit use is offline.
    import prepare_degradations as preparation
    from llm_confidence_uq.degradations import (
        CONDITIONS,
        DonorText,
        TransformationExample,
        build_all_conditions,
    )

    return (
        preparation,
        CONDITIONS,
        DonorText,
        TransformationExample,
        build_all_conditions,
    )


def reconstruct_stage_examples(
    *,
    stage_rows: Sequence[Mapping[str, Any]],
    selected_inputs: Sequence[Any],
    tokenizer: Any,
    prompt_spec: Any,
    seed: int,
    maximum_fragment_tokens: int,
) -> list[InferenceExample]:
    (
        preparation,
        conditions,
        DonorText,
        TransformationExample,
        build_all_conditions,
    ) = _runtime_dependencies()
    require(tuple(conditions) == EXPECTED_CONDITIONS, "condition protocol drift")
    selected_ids = [str(item.input_id) for item in selected_inputs]
    require(
        len(selected_ids) == len(set(selected_ids)),
        "duplicate reconstructed selected input ID",
    )
    selected_by_id = {str(item.input_id): item for item in selected_inputs}
    stage_input_order = [
        str(row["input_id"])
        for row in stage_rows
        if int(row["condition_index"]) == 0
    ]
    require(
        all(input_id in selected_by_id for input_id in stage_input_order),
        "stage input missing from reconstructed source",
    )
    stored_by_key = {
        (str(row["input_id"]), int(row["condition_index"])): dict(row)
        for row in stage_rows
    }
    require(
        len(stored_by_key) == len(stage_rows),
        "duplicate stage input/condition pair",
    )
    donors = [DonorText(item.input_id, item.passage) for item in selected_inputs]
    cached_tokenizer = preparation.CachingTokenizer(tokenizer)
    examples: list[InferenceExample] = []
    for input_id in stage_input_order:
        selected = selected_by_id[input_id]
        source = TransformationExample(
            selected.input_id,
            selected.question,
            selected.passage,
        )
        results = build_all_conditions(
            source,
            donors,
            cached_tokenizer,
            seed=seed,
            maximum_fragment_tokens=maximum_fragment_tokens,
        )
        require(
            tuple(result.condition for result in results) == EXPECTED_CONDITIONS,
            "condition reconstruction drift",
        )
        for condition_index, result in enumerate(results):
            reconstructed, _ = preparation.materialize_runtime_row(
                selected=selected,
                result=result,
                condition_index=condition_index,
                tokenizer=tokenizer,
                cached_tokenizer=cached_tokenizer,
                prompt_spec=prompt_spec,
            )
            stored = stored_by_key[(input_id, condition_index)]
            require(
                stored == reconstructed,
                "reconstructed degradation row differs from frozen manifest",
            )
            require(
                result.transformed_passage is not None,
                "runnable transformation lacks a passage",
            )
            prompt = preparation.render_locked_prompt(
                tokenizer,
                prompt_spec,
                question=selected.question,
                passage=result.transformed_passage,
            )
            require(
                sha256_text(prompt) == stored["rendered_prompt_sha256"],
                "rendered prompt hash drift",
            )
            contextual_class_token_ids(
                tokenizer,
                prompt,
                expected_ids=EXPECTED_CLASS_TOKEN_IDS,
            )
            examples.append(
                InferenceExample(
                    ordinal=len(examples),
                    transformation_id=str(stored["transformation_id"]),
                    degradation_row_sha256=str(stored["row_sha256"]),
                    input_id=input_id,
                    source_split=str(stored["source_split"]),
                    source_index=int(stored["source_index"]),
                    condition_index=condition_index,
                    condition=str(stored["condition"]),
                    runtime_status="ok",
                    runtime_reason_code=None,
                    prompt=prompt,
                    prompt_tokens=int(stored["prompt_tokens"]),
                    rendered_prompt_sha256=str(
                        stored["rendered_prompt_sha256"]
                    ),
                )
            )
    require(len(examples) == len(stage_rows), "reconstructed stage size drift")
    return examples


def _float_at(tensor: Any, index: int) -> float:
    return float(tensor[index].detach().cpu().item())


def _generation_content(
    token_ids: Sequence[int],
    *,
    eos_token_ids: Sequence[int],
    pad_token_id: int | None,
    special_token_ids: Sequence[int],
    maximum_new_tokens: int,
) -> tuple[list[int], bool, bool]:
    ids = [int(token_id) for token_id in token_ids]
    eos_ids = tuple(int(token_id) for token_id in eos_token_ids)
    require(eos_ids, "at least one EOS token ID is required")
    require(len(eos_ids) == len(set(eos_ids)), "duplicate EOS token IDs")
    require(
        len(ids) <= maximum_new_tokens,
        "generated suffix exceeds max_new_tokens",
    )
    eos_position = (
        next(
            (
                index
                for index, token_id in enumerate(ids)
                if token_id in eos_ids
            ),
            None,
        )
    )
    if eos_position is not None:
        content = ids[:eos_position]
        allowed_tail = {
            token_id
            for token_id in (*eos_ids, pad_token_id)
            if token_id is not None
        }
        require(
            all(token_id in allowed_tail for token_id in ids[eos_position + 1 :]),
            "generated suffix contains a non-padding token after EOS",
        )
        require(
            set(content).isdisjoint({int(value) for value in special_token_ids}),
            "generated content contains a tokenizer special token",
        )
        return content, True, False
    if pad_token_id is not None and pad_token_id in ids:
        raise BaselineRunError("generated padding appeared before an EOS token")
    require(
        len(ids) == maximum_new_tokens,
        "generation stopped without EOS or max_new_tokens",
    )
    content = ids
    require(
        set(content).isdisjoint({int(value) for value in special_token_ids}),
        "generated content contains a tokenizer special token",
    )
    return content, False, True


def _batch_identity_matches(
    rows: Sequence[Mapping[str, Any]],
    examples: Sequence[InferenceExample],
    *,
    config_sha256: str,
    inference_seed: int,
) -> None:
    validate_prediction_rows(rows)
    require(len(rows) == len(examples), "resumed batch row count drift")
    for row, example in zip(rows, examples, strict=True):
        require(
            row["model_id"] == MODEL_ID
            and row["model_revision"] == MODEL_REVISION
            and row["inference_config_sha256"] == config_sha256
            and row["inference_seed"] == inference_seed
            and row["transformation_id"] == example.transformation_id
            and row["degradation_row_sha256"]
            == example.degradation_row_sha256
            and row["input_id"] == example.input_id
            and row["source_split"] == example.source_split
            and row["source_index"] == example.source_index
            and row["condition_index"] == example.condition_index
            and row["condition"] == example.condition
            and row["rendered_prompt_sha256"]
            == example.rendered_prompt_sha256
            and row["prompt_tokens"] == example.prompt_tokens,
            "resumed batch identity drift",
        )


def _read_prediction_shard(
    path: Path,
    *,
    examples: Sequence[InferenceExample],
    config_sha256: str,
    inference_seed: int,
) -> list[dict[str, Any]]:
    path = _repo_path(path)
    require(path.exists() and path.is_file(), "prediction shard is missing")
    payload = path.read_bytes()
    require(payload.endswith(b"\n") and b"\r" not in payload, "invalid shard framing")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    _batch_identity_matches(
        rows,
        examples,
        config_sha256=config_sha256,
        inference_seed=inference_seed,
    )
    return rows


def _ensure_safe_directory(path: Path) -> Path:
    path = _repo_path(path)
    if path.exists():
        require(path.is_dir() and not path.is_symlink(), "invalid output directory")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            # Google Drive's Linux FUSE mount can reject directory fsync even
            # though file fsync and atomic rename succeeded.  Ignore only the
            # documented "operation unsupported" family; all other I/O errors
            # still fail closed.
            if error.errno not in {
                errno.EBADF,
                errno.EINVAL,
                errno.ENOSYS,
                errno.ENOTSUP,
            }:
                raise
    finally:
        os.close(descriptor)


def atomic_refuse_or_verify(path: Path, payload: bytes) -> str:
    path = _repo_path(path)
    _ensure_safe_directory(path.parent)
    require(not path.is_symlink(), f"symlinked output rejected: {path}")
    if path.exists():
        require(path.is_file(), f"output is not a file: {path}")
        require(path.read_bytes() == payload, f"differing output already exists: {path}")
        return "verified_existing"
    temporary = path.with_name(f".{path.name}.tmp")
    require(
        not temporary.exists() and not temporary.is_symlink(),
        f"unresolved temporary output exists: {temporary}",
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        require(not path.exists(), f"output appeared during publication: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    require(path.read_bytes() == payload, f"output round-trip mismatch: {path}")
    return "created"


def run_model_batches(
    *,
    examples: Sequence[InferenceExample],
    model: Any,
    tokenizer: Any,
    torch_module: Any,
    config: Mapping[str, Any],
    config_sha256: str,
    batch_directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from transformers import RepetitionPenaltyLogitsProcessor

    require(
        "prompt_ignore_length"
        in inspect.signature(RepetitionPenaltyLogitsProcessor).parameters,
        "repetition-penalty processor lacks prompt exclusion",
    )
    require(
        all(example.runtime_status == "ok" for example in examples),
        "baseline runner received a non-runnable example",
    )
    runnable = [
        replace(example, ordinal=index)
        for index, example in enumerate(examples)
    ]
    dataset = InferenceDataset(runnable)
    collator = PromptCollator(
        tokenizer,
        maximum_prompt_tokens=int(config["prompt"]["maximum_prompt_tokens"]),
    )
    loader = torch_module.utils.data.DataLoader(
        dataset,
        batch_size=int(config["inference"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["inference"]["num_workers"]),
        pin_memory=bool(config["inference"]["pin_memory"]),
        drop_last=False,
        collate_fn=collator,
    )
    batch_directory = _ensure_safe_directory(batch_directory)
    device = model.get_input_embeddings().weight.device
    require(
        device.type == "cuda" and str(device) == "cuda:0",
        f"unexpected model device: {device}",
    )
    generation = config["inference"]["generation"]
    max_new_tokens = int(generation["max_new_tokens"])
    eos_token_ids = tuple(int(value) for value in generation["eos_token_ids"])
    pad_token_id = int(generation["pad_token_id"])
    repetition_penalty = float(generation["repetition_penalty"])
    built_in_repetition_penalty = float(
        generation["inherited_repetition_penalty_override"]
    )
    all_rows: list[dict[str, Any]] = []
    counters = {"created_shards": 0, "resumed_shards": 0}
    for batch_index, batch in enumerate(loader):
        batch_examples = tuple(batch["examples"])
        shard_path = batch_directory / f"batch-{batch_index:05d}.jsonl"
        if shard_path.exists():
            rows = _read_prediction_shard(
                shard_path,
                examples=batch_examples,
                config_sha256=config_sha256,
                inference_seed=int(config["inference"]["seed"]),
            )
            counters["resumed_shards"] += 1
            all_rows.extend(rows)
            continue

        model_inputs = {
            key: value.to(device, non_blocking=False)
            for key, value in batch["model_inputs"].items()
        }
        attention_mask = model_inputs["attention_mask"]
        forward_inputs = dict(model_inputs)
        forward_inputs["position_ids"] = position_ids_from_attention_mask(
            attention_mask
        )
        input_width = int(model_inputs["input_ids"].shape[1])
        repetition_processor = RepetitionPenaltyLogitsProcessor(
            penalty=repetition_penalty,
            prompt_ignore_length=input_width,
        )
        with torch_module.inference_mode():
            outputs = model(
                **forward_inputs,
                use_cache=False,
                return_dict=True,
                output_hidden_states=False,
                output_attentions=False,
                logits_to_keep=1,
            )
            next_logits = gather_last_token_logits(
                outputs.logits,
                attention_mask,
            )
            confidence = binary_confidence_from_logits(
                next_logits,
                yes_token_id=EXPECTED_CLASS_TOKEN_IDS[0],
                no_token_id=EXPECTED_CLASS_TOKEN_IDS[1],
            )
            generated = model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=list(eos_token_ids),
                repetition_penalty=built_in_repetition_penalty,
                logits_processor=[repetition_processor],
            )
        require(
            generated.ndim == 2
            and generated.shape[0] == len(batch_examples),
            "generated batch shape drift",
        )
        require(generated.shape[1] >= input_width, "generated sequence is too short")
        require(
            bool(
                torch_module.equal(
                    generated[:, :input_width],
                    model_inputs["input_ids"],
                )
            ),
            "generated sequence did not preserve the padded input prefix",
        )
        rows: list[dict[str, Any]] = []
        for index, example in enumerate(batch_examples):
            raw_suffix = generated[index, input_width:].detach().cpu().tolist()
            completion_ids, ended_with_eos, reached_max = _generation_content(
                raw_suffix,
                eos_token_ids=eos_token_ids,
                pad_token_id=pad_token_id,
                special_token_ids=tokenizer.all_special_ids,
                maximum_new_tokens=max_new_tokens,
            )
            completion = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            parsed = parse_expressed_continuation(completion)
            prediction_index = int(
                confidence.token_prediction_index[index].detach().cpu().item()
            )
            token_prediction = ("Yes", "No")[prediction_index]
            rows.append(
                make_prediction_row(
                    example,
                    model_id=MODEL_ID,
                    model_revision=MODEL_REVISION,
                    inference_config_sha256=config_sha256,
                    inference_seed=int(config["inference"]["seed"]),
                    yes_token_id=EXPECTED_CLASS_TOKEN_IDS[0],
                    no_token_id=EXPECTED_CLASS_TOKEN_IDS[1],
                    yes_logit=_float_at(confidence.yes_logit, index),
                    no_logit=_float_at(confidence.no_logit, index),
                    p_yes_binary=_float_at(confidence.p_yes_binary, index),
                    p_no_binary=_float_at(confidence.p_no_binary, index),
                    p_yes_full_vocabulary=_float_at(
                        confidence.p_yes_full_vocabulary,
                        index,
                    ),
                    p_no_full_vocabulary=_float_at(
                        confidence.p_no_full_vocabulary,
                        index,
                    ),
                    yes_no_full_vocabulary_mass=_float_at(
                        confidence.yes_no_full_vocabulary_mass,
                        index,
                    ),
                    token_prediction=token_prediction,
                    token_confidence=_float_at(
                        confidence.token_confidence,
                        index,
                    ),
                    predictive_entropy=_float_at(
                        confidence.predictive_entropy,
                        index,
                    ),
                    generated_continuation=completion,
                    generated_continuation_tokens=len(completion_ids),
                    generation_ended_with_eos=ended_with_eos,
                    generation_reached_max_new_tokens=reached_max,
                    expressed_parse=parsed,
                )
            )
        payload = prediction_jsonl_bytes(rows)
        require(
            atomic_refuse_or_verify(shard_path, payload) == "created",
            "new prediction shard was not created",
        )
        counters["created_shards"] += 1
        all_rows.extend(rows)
        del (
            outputs,
            generated,
            next_logits,
            confidence,
            forward_inputs,
            model_inputs,
        )
    expected_shards = {
        f"batch-{batch_index:05d}.jsonl" for batch_index in range(len(loader))
    }
    observed_shards = {path.name for path in batch_directory.iterdir()}
    require(
        observed_shards == expected_shards,
        "prediction batch directory contains missing or unexpected artifacts",
    )
    validate_prediction_rows(all_rows)
    require(len(all_rows) == len(examples), "prediction cardinality drift")
    require(
        [row["transformation_id"] for row in all_rows]
        == [example.transformation_id for example in examples],
        "prediction/example order drift",
    )
    return all_rows, counters


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitized_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    message = message.replace(str(REPO_ROOT), "<repo>")
    return message[:500]


def write_run_manifest(
    *,
    stage: str,
    status: str,
    started_at: str,
    ended_at: str,
    elapsed_seconds: float,
    details: Mapping[str, Any],
) -> Path:
    root = _ensure_safe_directory(RUN_MANIFEST_ROOT)
    safe_stamp = ended_at.replace(":", "-").replace("+", "_")
    destination = root / f"{stage}-{status}-{safe_stamp}.json"
    payload = canonical_bytes(
        {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "stage": stage,
            "status": status,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "elapsed_seconds": elapsed_seconds,
            "git_head": git_head(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
            **dict(details),
        }
    )
    require(
        atomic_refuse_or_verify(destination, payload) == "created",
        "run manifest collision",
    )
    return destination


def _load_and_validate_environment(
    config: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any]:
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(
        sys.version.split()[0] == EXPECTED_PYTHON_VERSION,
        "Python version drift",
    )
    require(
        platform.platform() == config["environment"]["platform"],
        "platform contract drift",
    )
    observed_versions = package_versions()
    require(
        observed_versions == EXPECTED_PACKAGE_VERSIONS,
        f"package version drift: {observed_versions}",
    )
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.get_device_name(0) == EXPECTED_GPU_NAME, "unexpected GPU")
    require(torch.version.cuda == "12.8", "PyTorch CUDA build drift")
    require(
        tuple(torch.cuda.get_device_capability(0)) == EXPECTED_COMPUTE_CAPABILITY,
        "unexpected compute capability",
    )
    observed_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    require(
        abs(observed_vram_gib - EXPECTED_GPU_VRAM_GIB) <= 0.10,
        "GPU memory contract drift",
    )
    api = HfApi()
    require(
        api.model_info(MODEL_ID, revision=MODEL_REVISION, token=False).sha
        == MODEL_REVISION,
        "model revision resolution drift",
    )
    require(
        api.dataset_info(DATASET_ID, revision=DATASET_REVISION, token=False).sha
        == DATASET_REVISION,
        "dataset revision resolution drift",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=False,
        trust_remote_code=False,
        use_fast=True,
        padding_side="left",
    )
    require(
        tokenizer.__class__.__name__ == config["model"]["tokenizer_class"],
        "tokenizer class drift",
    )
    require(tokenizer.padding_side == "left", "tokenizer padding-side drift")
    require(
        tokenizer.pad_token_id is not None
        and tokenizer.eos_token_id is not None,
        "tokenizer special IDs are missing",
    )
    generation = config["inference"]["generation"]
    require(
        tokenizer.pad_token_id == generation["pad_token_id"]
        and tokenizer.eos_token_id == generation["eos_token_ids"][0],
        "tokenizer special-token contract drift",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=False,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float16,
        device_map=0,
    )
    require(
        model.__class__.__name__ == config["model"]["model_class"],
        "model class drift",
    )
    observed_eos_token_ids = model.generation_config.eos_token_id
    require(
        isinstance(observed_eos_token_ids, (list, tuple))
        and list(observed_eos_token_ids) == generation["eos_token_ids"]
        and model.generation_config.pad_token_id == generation["pad_token_id"]
        and model.generation_config.repetition_penalty
        == generation["repetition_penalty"],
        "model generation-config drift",
    )
    require(model.dtype == torch.float16, "model dtype drift")
    require(
        sum(parameter.numel() for parameter in model.parameters())
        == MODEL_PARAMETER_COUNT,
        "loaded parameter count drift",
    )
    require(
        getattr(model, "peft_config", None) in (None, {}),
        "unexpected PEFT adapter attached to baseline model",
    )
    require(
        "logits_to_keep" in inspect.signature(model.forward).parameters,
        "model forward does not expose logits_to_keep",
    )
    require(
        model.get_input_embeddings().weight.shape[0]
        > max(EXPECTED_CLASS_TOKEN_IDS),
        "class token ID exceeds model vocabulary",
    )
    model.eval()
    require(model.training is False, "model failed to enter evaluation mode")
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


def _publish_final_artifacts(
    output_directory: Path,
    *,
    predictions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    output_directory = _ensure_safe_directory(output_directory)
    prediction_payload = prediction_jsonl_bytes(predictions)
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
    completion = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_hashes_sha256": sha256_bytes(
            payloads["artifact_hashes.json"]
        ),
        "complete": True,
    }
    statuses: dict[str, str] = {}
    for name in (
        "predictions.jsonl",
        "provenance.json",
        "summary.json",
        "artifact_hashes.json",
    ):
        statuses[name] = atomic_refuse_or_verify(
            output_directory / name,
            payloads[name],
        )
    completion_payload = canonical_bytes(completion)
    statuses["COMPLETE.json"] = atomic_refuse_or_verify(
        output_directory / "COMPLETE.json",
        completion_payload,
    )
    all_hashes = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(payloads.items())
    }
    all_hashes["COMPLETE.json"] = {
        "bytes": len(completion_payload),
        "sha256": sha256_bytes(completion_payload),
    }
    return statuses, all_hashes


def run(stage: str, config_path: Path) -> tuple[dict[str, Any], Path]:
    import yaml

    config_path = _repo_path(config_path)
    config_payload = config_path.read_bytes()
    config = yaml.safe_load(config_payload.decode("utf-8"))
    require(isinstance(config, dict), "baseline configuration is not a mapping")
    validate_config_contract(config)
    config_sha256 = sha256_bytes(config_payload)
    for relative_name, expected_hash in EXPECTED_FILE_SHA256.items():
        path = _repo_path(relative_name)
        require(path.exists() and path.is_file(), f"missing prerequisite: {relative_name}")
        require(
            file_sha256(path) == expected_hash,
            f"prerequisite hash mismatch: {relative_name}",
        )
    for relative_name, expected_hash in config["implementation"].items():
        path = _repo_path(relative_name)
        require(path.exists() and path.is_file(), f"missing implementation: {relative_name}")
        require(
            file_sha256(path) == expected_hash,
            f"implementation hash mismatch: {relative_name}",
        )
    validate_hash_ledgers(config)

    preparation, *_ = _runtime_dependencies()
    source_rows = preparation.read_verified_jsonl(
        _repo_path(config["input"]["source_manifest_path"]),
        expected_sha256=config["input"]["source_manifest_sha256"],
        expected_rows=int(config["input"]["source_examples"]),
    )
    degradation_rows = read_jsonl_verified(
        _repo_path(config["input"]["degradation_manifest_path"]),
        expected_sha256=config["input"]["degradation_manifest_sha256"],
        expected_rows=int(config["input"]["degradation_rows"]),
    )
    validate_degradation_rows(degradation_rows)
    stage_rows = select_stage_rows(degradation_rows, stage)
    stage_projection_sha256 = selected_stage_projection_sha256(stage_rows)

    torch_module, tokenizer, model, validation_split = (
        _load_and_validate_environment(config)
    )
    data_config = yaml.safe_load(
        _repo_path("configs/data.yaml").read_text(encoding="utf-8")
    )
    degradation_config = yaml.safe_load(
        _repo_path("configs/degradations.yaml").read_text(encoding="utf-8")
    )
    prompt_spec = preparation.prompt_spec_from_configs(
        data_config,
        degradation_config,
    )
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
        eligibility_max_prompt_tokens=int(
            data_config["selection"]["original_prompt_eligibility_max_tokens"]
        ),
    )
    del source_rows, validation_split
    examples = reconstruct_stage_examples(
        stage_rows=stage_rows,
        selected_inputs=selected_inputs,
        tokenizer=tokenizer,
        prompt_spec=prompt_spec,
        seed=int(degradation_config["protocol"]["seed"]),
        maximum_fragment_tokens=int(
            degradation_config["distractor"]["maximum_fragment_tokens"]
        ),
    )
    del selected_inputs

    output_directory = (
        _repo_path(config["output"]["root"]) / stage
    )
    batch_directory = (
        output_directory / config["output"]["batch_shard_directory"]
    )
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    predictions, shard_counts = run_model_batches(
        examples=examples,
        model=model,
        tokenizer=tokenizer,
        torch_module=torch_module,
        config=config,
        config_sha256=config_sha256,
        batch_directory=batch_directory,
    )
    torch_module.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    prediction_payload = prediction_jsonl_bytes(predictions)
    summary = {
        **summarize_prediction_rows(predictions),
        "stage": stage,
        "base_inputs": len({row["input_id"] for row in predictions}),
        "prediction_jsonl_sha256": sha256_bytes(prediction_payload),
        "stage_projection_sha256": stage_projection_sha256,
        "selection": config["stages"][stage]["selection"],
    }
    relative_config_path = config_path.relative_to(REPO_ROOT).as_posix()
    provenance = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "class": model.__class__.__name__,
            "dtype": str(model.dtype),
            "parameter_count": MODEL_PARAMETER_COUNT,
            "peft_adapter_attached": False,
        },
        "tokenizer": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "class": tokenizer.__class__.__name__,
            "padding_side": tokenizer.padding_side,
            "class_order": ["Yes", "No"],
            "class_token_ids": {"Yes": 7414, "No": 2308},
        },
        "inputs": {
            "source_manifest_sha256": config["input"]["source_manifest_sha256"],
            "degradation_manifest_sha256": config["input"][
                "degradation_manifest_sha256"
            ],
            "full_selected_label_blind_projection_sha256": selection_report[
                "label_blind_projection_sha256"
            ],
            "stage_projection_sha256": stage_projection_sha256,
            "full_validation_source_sha256": source_report["canonical_sha256"],
        },
        "configuration": {
            "path": relative_config_path,
            "sha256": config_sha256,
        },
        "implementation": {
            "inference_module_sha256": file_sha256(
                _repo_path("src/llm_confidence_uq/inference.py")
            ),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
        },
        "inference_seed": int(config["inference"]["seed"]),
        "labels_available_to_model_inference": False,
        "ground_truth_persisted": False,
        "correctness_persisted": False,
        "source_fields_copied_to_predictions": False,
        "generated_continuation_persisted": True,
        "dynamic_metadata_location": RUN_MANIFEST_ROOT.as_posix(),
    }
    statuses, artifact_hashes = _publish_final_artifacts(
        output_directory,
        predictions=predictions,
        summary=summary,
        provenance=provenance,
    )
    details = {
        "command": [
            sys.executable,
            "scripts/run_baseline.py",
            "--stage",
            stage,
            "--config",
            relative_config_path,
        ],
        "inference_seconds": inference_seconds,
        "gpu": torch_module.cuda.get_device_name(0),
        "peak_allocated_gpu_gib": (
            torch_module.cuda.max_memory_allocated() / (1024**3)
        ),
        "output_directory": output_directory.relative_to(REPO_ROOT).as_posix(),
        "artifact_hashes": artifact_hashes,
        "artifact_statuses": statuses,
        "batch_shards": shard_counts,
        "summary": summary,
    }
    return details, output_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run audited frozen-base-model confidence inference."
    )
    parser.add_argument("--stage", required=True, choices=ALLOWED_STAGES)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = _timestamp()
    started = time.perf_counter()
    try:
        details, output_directory = run(args.stage, args.config)
        ended_at = _timestamp()
        manifest_path = write_run_manifest(
            stage=args.stage,
            status="success",
            started_at=started_at,
            ended_at=ended_at,
            elapsed_seconds=time.perf_counter() - started,
            details=details,
        )
        print("BASELINE INFERENCE: PASS")
        print(f"stage={args.stage}")
        print(
            "output_directory="
            + output_directory.relative_to(REPO_ROOT).as_posix()
        )
        print("summary=" + canonical_json(details["summary"]))
        print("artifact_statuses=" + canonical_json(details["artifact_statuses"]))
        print(f"run_manifest={manifest_path.relative_to(REPO_ROOT).as_posix()}")
        print(
            "SCOPE: frozen base-model predictions only; labels were unavailable "
            "to model inference, and no calibration, fine-tuning, or accuracy "
            "claim is made."
        )
        return 0
    except Exception as error:
        ended_at = _timestamp()
        try:
            failure_manifest = write_run_manifest(
                stage=args.stage,
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                elapsed_seconds=time.perf_counter() - started,
                details={
                    "failure_type": error.__class__.__name__,
                    "failure_message_sanitized": _sanitized_error(error),
                    "command": [
                        sys.executable,
                        "scripts/run_baseline.py",
                        "--stage",
                        args.stage,
                        "--config",
                        str(args.config),
                    ],
                },
            )
            print(
                "FAILED_RUN_MANIFEST="
                + failure_manifest.relative_to(REPO_ROOT).as_posix(),
                file=sys.stderr,
            )
        except Exception as manifest_error:
            print(
                "FAILED_RUN_MANIFEST_WRITE_ERROR="
                + _sanitized_error(manifest_error),
                file=sys.stderr,
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
