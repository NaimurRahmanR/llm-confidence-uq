from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from llm_confidence_uq.inference import contextual_class_token_ids  # noqa: E402
from llm_confidence_uq.training import (  # noqa: E402
    AnswerTokenCollator,
    EXPECTED_CLASS_TOKEN_IDS,
    PROTOCOL_VERSION,
    SupervisedTrainingDataset,
    SupervisedTrainingExample,
    build_cosine_scheduler,
    canonical_json,
    expected_optimizer_steps,
    make_epoch_dataloader,
    set_deterministic_seed,
    sha256_bytes,
    sha256_text,
    total_parameter_count,
    train_one_epoch,
    trainable_parameter_count,
    trainable_parameter_sha256,
    validate_independent_seeds,
)
import prepare_degradations as preparation  # noqa: E402


ALLOWED_STAGES = ("smoke32", "full_seed_1", "full_seed_2", "full_seed_3")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL_PARAMETER_COUNT = 1_543_714_304
DATASET_ID = "google/boolq"
DATASET_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_GPU_VRAM_GIB = 79.250732421875
EXPECTED_COMPUTE_CAPABILITY = (8, 0)
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_EXCLUDED_PACKAGES = ("torchao",)
EXPECTED_TRAINABLE_PARAMETERS = 9_232_384
EXPECTED_TOTAL_WITH_ADAPTER = 1_552_946_688
EXPECTED_TARGET_INSTANCES = 196
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
EXPECTED_PREREQUISITES = {
    "configs/data.yaml": "412fd6da74212e071159463e104329efefbc8fbe6b853c29c825e3ab37cb64ac",
    "configs/degradations.yaml": "303238ae02025c9e5cb59bbd061949b6798f5fde8936a1f0604d15f0552cf2e3",
    "configs/baseline.yaml": "021e8dcdb7e316d83fa9528aea5d21b427a80161c398c86d6bf26ae834c43c85",
    "data/manifests/artifact_hashes.json": "8916df7fd59606377a7701f9216179a3a521f4f0ccbfba3a1ea113017353e640",
    "data/manifests/train.jsonl": "8427a69570e3457fcff5a5482c205117ced2f3c6f98ca225156aebf7283991c3",
    "scripts/prepare_degradations.py": "aee819f1727a9113a4b2df2c8a946747dd90565b2659bd3dcbf635a4b3d13e86",
}
EXPECTED_MANIFEST_FIELDS = {
    "answer",
    "dataset_id",
    "dataset_revision",
    "eligibility_max_prompt_tokens",
    "example_id",
    "input_sha256",
    "original_passage_sha256",
    "prompt_tokens",
    "record_sha256",
    "research_split",
    "schema_version",
    "selection_rank_sha256",
    "selection_seed",
    "source_index",
    "source_split",
}
TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRARunError(RuntimeError):
    """Raised when a pinned LoRA-training invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LoRARunError(message)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    root = Path(os.path.abspath(REPO_ROOT))
    absolute = Path(os.path.abspath(candidate))
    require(absolute == root or root in absolute.parents, f"path escapes repository: {absolute}")
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


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in EXPECTED_PACKAGE_VERSIONS
    }


def validate_excluded_runtime_packages() -> None:
    for package in EXPECTED_EXCLUDED_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        require(version is None, f"excluded package installed: {package}=={version}")
        require(importlib.util.find_spec(package) is None, f"excluded package importable: {package}")


def validate_config_contract(config: Mapping[str, Any]) -> None:
    require(config["schema_version"] == 1, "config schema drift")
    require(config["protocol_version"] == PROTOCOL_VERSION, "protocol drift")
    dataset = config["dataset"]
    require(
        dataset["id"] == DATASET_ID
        and dataset["revision"] == DATASET_REVISION
        and dataset["source_split"] == "train"
        and dataset["manifest_rows"] == 800
        and dataset["calibration_examples_used"] == 0
        and dataset["test_examples_used"] == 0,
        "dataset contract drift",
    )
    model = config["model"]
    require(
        model["id"] == MODEL_ID
        and model["revision"] == MODEL_REVISION
        and model["tokenizer_id"] == MODEL_ID
        and model["tokenizer_revision"] == MODEL_REVISION
        and model["dtype"] == "bfloat16"
        and model["quantization"] == "none",
        "model contract drift",
    )
    environment = config["environment"]
    require(
        environment["python"] == EXPECTED_PYTHON_VERSION
        and environment["gpu"] == EXPECTED_GPU_NAME
        and float(environment["gpu_vram_gib"]) == EXPECTED_GPU_VRAM_GIB
        and tuple(environment["compute_capability"]) == EXPECTED_COMPUTE_CAPABILITY
        and environment["pytorch_cuda_build"] == "12.8"
        and tuple(environment["excluded_packages"]) == EXPECTED_EXCLUDED_PACKAGES
        and dict(environment["package_versions"]) == EXPECTED_PACKAGE_VERSIONS,
        "environment contract drift",
    )
    objective = config["objective"]
    require(
        objective["name"] == "full-vocabulary-next-token-cross-entropy"
        and objective["prompt_loss_weight"] == 0.0
        and objective["answer_tokens_per_example"] == 1
        and objective["numerical_confidence_targets_used"] is False
        and list(objective["class_order"]) == ["Yes", "No"]
        and dict(objective["class_token_ids"]) == {"Yes": 7414, "No": 2308}
        and objective["maximum_prompt_tokens"] == 768
        and objective["padding_side"] == "left"
        and objective["truncation_allowed"] is False
        and objective["logits_to_keep"] == 1,
        "objective contract drift",
    )
    lora = config["lora"]
    require(
        lora["task_type"] == "CAUSAL_LM"
        and lora["rank"] == 8
        and lora["alpha"] == 16
        and float(lora["dropout"]) == 0.05
        and lora["bias"] == "none"
        and lora["initialization"] == "default"
        and lora["use_rslora"] is False
        and tuple(lora["target_modules"]) == TARGET_MODULES
        and lora["expected_target_module_instances"] == EXPECTED_TARGET_INSTANCES
        and lora["expected_trainable_parameter_count"] == EXPECTED_TRAINABLE_PARAMETERS
        and lora["expected_total_parameter_count_with_adapter"] == EXPECTED_TOTAL_WITH_ADAPTER,
        "LoRA contract drift",
    )
    optimization = config["optimization"]
    require(
        optimization["implementation"] == "torch-explicit-loop-v1"
        and optimization["trainer_abstraction_used"] is False
        and optimization["optimizer"] == "torch.optim.AdamW"
        and float(optimization["learning_rate"]) == 0.0002
        and list(optimization["betas"]) == [0.9, 0.999]
        and float(optimization["epsilon"]) == 1e-8
        and float(optimization["weight_decay"]) == 0.01
        and optimization["batch_size"] == 4
        and optimization["gradient_accumulation_steps"] == 8
        and optimization["effective_batch_size"] == 32
        and float(optimization["maximum_gradient_norm"]) == 1.0
        and optimization["full_epochs"] == 3
        and optimization["full_warmup_steps"] == 8
        and optimization["full_optimizer_steps"] == 75
        and optimization["deterministic_algorithms"] is True
        and optimization["cublas_workspace_config"] == ":4096:8"
        and optimization["allow_tf32"] is False,
        "optimization contract drift",
    )
    stages = config["stages"]
    require(tuple(stages) == ALLOWED_STAGES, "training stage order drift")
    require(stages["smoke32"] == {"seed": 20260811, "examples": 32, "epochs": 1, "warmup_steps": 0, "optimizer_steps": 1}, "smoke stage drift")
    require(
        [stages[name]["seed"] for name in ALLOWED_STAGES[1:]]
        == [20260811, 20260812, 20260813],
        "full-stage seed drift",
    )
    validate_independent_seeds([stages[name]["seed"] for name in ALLOWED_STAGES[1:]])
    for stage_name, stage in stages.items():
        observed = expected_optimizer_steps(
            examples=int(stage["examples"]),
            batch_size=int(optimization["batch_size"]),
            gradient_accumulation_steps=int(optimization["gradient_accumulation_steps"]),
            epochs=int(stage["epochs"]),
        )
        require(observed == int(stage["optimizer_steps"]), f"optimizer-step drift: {stage_name}")
    require(dict(config["prerequisites"]) == EXPECTED_PREREQUISITES, "prerequisite map drift")


def verify_static_files(config: Mapping[str, Any]) -> None:
    for relative, expected in EXPECTED_PREREQUISITES.items():
        path = _repo_path(relative)
        require(path.is_file() and not path.is_symlink(), f"missing prerequisite: {relative}")
        require(file_sha256(path) == expected, f"prerequisite SHA-256 drift: {relative}")
    for relative, expected in config["implementation"].items():
        require(expected != "PENDING", f"implementation digest is pending: {relative}")
        path = _repo_path(relative)
        require(path.is_file() and not path.is_symlink(), f"missing implementation: {relative}")
        require(file_sha256(path) == expected, f"implementation SHA-256 drift: {relative}")


def read_training_manifest(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = _repo_path(config["dataset"]["manifest_path"])
    payload = path.read_bytes()
    require(sha256_bytes(payload) == config["dataset"]["manifest_sha256"], "training manifest SHA-256 drift")
    require(payload.endswith(b"\n") and b"\r" not in payload, "training manifest line endings drift")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    require(len(rows) == 800, "training manifest row count drift")
    require(all(set(row) == EXPECTED_MANIFEST_FIELDS for row in rows), "training manifest schema drift")
    require(all(row["research_split"] == "train" and row["source_split"] == "train" for row in rows), "non-training row in training manifest")
    require(len({row["example_id"] for row in rows}) == 800, "duplicate training example ID")
    require(Counter(bool(row["answer"]) for row in rows) == Counter({False: 301, True: 499}), "training class-count drift")
    return rows


def reconstruct_training_examples(
    manifest_rows: Sequence[Mapping[str, Any]],
    dataset_split: Any,
    tokenizer: Any,
    prompt_spec: Any,
) -> list[SupervisedTrainingExample]:
    examples: list[SupervisedTrainingExample] = []
    for ordinal, row in enumerate(manifest_rows):
        require(row["dataset_id"] == DATASET_ID and row["dataset_revision"] == DATASET_REVISION, "manifest dataset drift")
        source_index = int(row["source_index"])
        require(0 <= source_index < len(dataset_split), "training source index out of range")
        source = dataset_split[source_index]
        question = source["question"]
        passage = source["passage"]
        answer = bool(source["answer"])
        require(isinstance(question, str) and question.strip(), "invalid training question")
        require(isinstance(passage, str) and passage.strip(), "invalid training passage")
        require(bool(row["answer"]) == answer, "training label mismatch")
        record_sha256 = sha256_text(canonical_json({"answer": answer, "passage": passage, "question": question}))
        input_sha256 = sha256_text(canonical_json({"passage": passage, "question": question}))
        require(row["record_sha256"] == record_sha256, "training record hash mismatch")
        require(row["input_sha256"] == input_sha256, "training input hash mismatch")
        require(row["original_passage_sha256"] == sha256_text(passage), "training passage hash mismatch")
        expected_id = f"boolq-train-{source_index:05d}-{record_sha256[:12]}"
        require(row["example_id"] == expected_id, "training example ID mismatch")
        prompt = preparation.render_locked_prompt(
            tokenizer,
            prompt_spec,
            question=question,
            passage=passage,
        )
        prompt_ids = tuple(int(token_id) for token_id in tokenizer.encode(prompt, add_special_tokens=False))
        require(len(prompt_ids) == int(row["prompt_tokens"]), "training prompt-token count mismatch")
        require(0 < len(prompt_ids) <= 768, "training prompt length outside contract")
        contextual_ids = contextual_class_token_ids(tokenizer, prompt)
        require(tuple(contextual_ids) == EXPECTED_CLASS_TOKEN_IDS, "contextual class-token drift")
        target_class = "Yes" if answer else "No"
        target_id = EXPECTED_CLASS_TOKEN_IDS[0 if answer else 1]
        examples.append(
            SupervisedTrainingExample(
                ordinal=ordinal,
                example_id=expected_id,
                input_sha256=input_sha256,
                prompt_sha256=sha256_text(prompt),
                prompt_token_ids=prompt_ids,
                target_class=target_class,
                target_token_id=target_id,
            )
        )
    require(len(examples) == len(manifest_rows), "training reconstruction cardinality drift")
    return examples


def validate_environment(config: Mapping[str, Any], torch: Any) -> None:
    validate_excluded_runtime_packages()
    require(sys.version.split()[0] == EXPECTED_PYTHON_VERSION, "Python version drift")
    require(platform.platform() == config["environment"]["platform"], "platform drift")
    require(package_versions() == EXPECTED_PACKAGE_VERSIONS, "package version drift")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.device_count() == 1, "exactly one CUDA device is required")
    require(torch.cuda.get_device_name(0) == EXPECTED_GPU_NAME, "unexpected GPU")
    require(tuple(torch.cuda.get_device_capability(0)) == EXPECTED_COMPUTE_CAPABILITY, "unexpected compute capability")
    observed_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    require(abs(observed_vram - EXPECTED_GPU_VRAM_GIB) <= 0.10, "GPU memory contract drift")
    require(torch.version.cuda == "12.8", "PyTorch CUDA build drift")
    require(torch.cuda.is_bf16_supported(), "BF16 is unsupported")


def build_peft_model(config: Mapping[str, Any], torch: Any, AutoModelForCausalLM: Any, LoraConfig: Any, get_peft_model: Any) -> Any:
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=False,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map=0,
    )
    require(base.__class__.__name__ == config["model"]["model_class"], "model class drift")
    require(base.dtype == torch.bfloat16, "base-model dtype drift")
    require(total_parameter_count(base) == MODEL_PARAMETER_COUNT, "base parameter-count drift")
    base.config.use_cache = False
    adapter_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=list(TARGET_MODULES),
        init_lora_weights=True,
        use_rslora=False,
    )
    model = get_peft_model(base, adapter_config)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    require(bool(trainable_names) and all("lora_" in name for name in trainable_names), "non-LoRA parameter is trainable")
    require(trainable_parameter_count(model) == EXPECTED_TRAINABLE_PARAMETERS, "trainable parameter-count drift")
    require(total_parameter_count(model) == EXPECTED_TOTAL_WITH_ADAPTER, "adapter total parameter-count drift")
    lora_a = [name for name in trainable_names if ".lora_A." in name]
    lora_b = [name for name in trainable_names if ".lora_B." in name]
    require(len(lora_a) == EXPECTED_TARGET_INSTANCES and len(lora_b) == EXPECTED_TARGET_INSTANCES, "LoRA target-instance drift")
    return model


def probe_logits(model: Any, examples: Sequence[SupervisedTrainingExample], collator: AnswerTokenCollator, torch: Any) -> Any:
    model.eval()
    batch = collator(examples[:4])
    device = torch.device("cuda:0")
    with torch.inference_mode():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            position_ids=batch["position_ids"].to(device),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    logits = outputs.logits[:, 0, :].index_select(
        1,
        torch.tensor(EXPECTED_CLASS_TOKEN_IDS, device=device),
    )
    require(bool(torch.isfinite(logits).all().item()), "probe logits are non-finite")
    return logits.detach().float().cpu()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def directory_ledger(root: Path, *, excluded: set[str] | None = None) -> dict[str, dict[str, Any]]:
    excluded = excluded or set()
    ledger = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        payload = path.read_bytes()
        ledger[relative] = {"bytes": len(payload), "sha256": sha256_bytes(payload)}
    return ledger


def verify_checkpoint_directory(directory: Path) -> dict[str, Any]:
    require(directory.is_dir() and not directory.is_symlink(), "checkpoint directory missing")
    ledger_payload = (directory / "artifact_hashes.json").read_bytes()
    ledger = json.loads(ledger_payload)
    require(ledger == directory_ledger(directory, excluded={"artifact_hashes.json", "COMPLETE.json"}), "checkpoint ledger drift")
    complete = json.loads((directory / "COMPLETE.json").read_text("utf-8"))
    require(complete["complete"] is True, "checkpoint completion flag is false")
    require(complete["artifact_hashes_sha256"] == sha256_bytes(ledger_payload), "checkpoint completion binding drift")
    return {"ledger": ledger, "complete": complete}


def publish_checkpoint(
    *,
    stage: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    report: Mapping[str, Any],
    probe_after_training: Any,
    config: Mapping[str, Any],
    torch: Any,
    AutoModelForCausalLM: Any,
    PeftModel: Any,
    examples: Sequence[SupervisedTrainingExample],
    collator: AnswerTokenCollator,
) -> tuple[Path, dict[str, Any]]:
    root = _repo_path(config["checkpoint"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    destination = root / stage
    require(not destination.exists(), f"checkpoint already exists: {destination}")
    temporary = root / f".{stage}.tmp-{os.getpid()}"
    require(not temporary.exists(), f"temporary checkpoint already exists: {temporary}")
    temporary.mkdir()
    try:
        adapter_directory = temporary / "adapter"
        model.save_pretrained(
            adapter_directory,
            safe_serialization=True,
            save_embedding_layers=False,
        )
        torch.save(
            {
                "schema_version": 1,
                "protocol_version": PROTOCOL_VERSION,
                "stage": stage,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            temporary / "training_state.pt",
        )
        _write_bytes(temporary / "training_report.json", canonical_bytes(report))

        fresh_base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            token=False,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.bfloat16,
            device_map=0,
        )
        fresh_base.config.use_cache = False
        reloaded = PeftModel.from_pretrained(
            fresh_base,
            adapter_directory,
            is_trainable=False,
        )
        reloaded_logits = probe_logits(reloaded, examples, collator, torch)
        torch.testing.assert_close(
            reloaded_logits,
            probe_after_training,
            rtol=float(config["checkpoint"]["reload_logit_rtol"]),
            atol=float(config["checkpoint"]["reload_logit_atol"]),
        )
        reload_report = {
            "schema_version": 1,
            "rows": int(reloaded_logits.shape[0]),
            "classes": ["Yes", "No"],
            "atol": float(config["checkpoint"]["reload_logit_atol"]),
            "rtol": float(config["checkpoint"]["reload_logit_rtol"]),
            "maximum_absolute_difference": float(
                torch.max(torch.abs(reloaded_logits - probe_after_training)).item()
            ),
            "passed": True,
        }
        _write_bytes(temporary / "reload_validation.json", canonical_bytes(reload_report))
        del reloaded
        del fresh_base
        torch.cuda.empty_cache()

        ledger = directory_ledger(temporary)
        ledger_payload = canonical_bytes(ledger)
        _write_bytes(temporary / "artifact_hashes.json", ledger_payload)
        completion = {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "stage": stage,
            "artifact_hashes_sha256": sha256_bytes(ledger_payload),
            "complete": True,
        }
        _write_bytes(temporary / "COMPLETE.json", canonical_bytes(completion))
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    verification = verify_checkpoint_directory(destination)
    return destination, verification


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def write_run_manifest(stage: str, status: str, started_at: str, details: Mapping[str, Any]) -> Path:
    root = _repo_path("outputs/manifests/lora")
    root.mkdir(parents=True, exist_ok=True)
    ended_at = _timestamp()
    destination = root / f"{stage}-{status}-{ended_at.replace(':', '-').replace('+', '_')}.json"
    payload = canonical_bytes(
        {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "stage": stage,
            "status": status,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "git_head": git_head(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **dict(details),
        }
    )
    _write_bytes(destination, payload)
    return destination


def run(stage: str, config_path: Path) -> tuple[dict[str, Any], Path]:
    validate_excluded_runtime_packages()
    import torch
    import yaml
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(stage in ALLOWED_STAGES, "unknown training stage")
    config_path = _repo_path(config_path)
    config = yaml.safe_load(config_path.read_text("utf-8"))
    validate_config_contract(config)
    verify_static_files(config)
    validate_environment(config, torch)
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
    require(tokenizer.__class__.__name__ == config["model"]["tokenizer_class"], "tokenizer class drift")
    require(tokenizer.pad_token_id == 151643 and tokenizer.eos_token_id == 151645, "tokenizer special-token drift")
    data_config = yaml.safe_load(_repo_path("configs/data.yaml").read_text("utf-8"))
    degradation_config = yaml.safe_load(_repo_path("configs/degradations.yaml").read_text("utf-8"))
    prompt_spec = preparation.prompt_spec_from_configs(data_config, degradation_config)
    manifest_rows = read_training_manifest(config)
    dataset_split = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
        token=False,
    )
    all_examples = reconstruct_training_examples(manifest_rows, dataset_split, tokenizer, prompt_spec)
    stage_config = config["stages"][stage]
    selected = all_examples[: int(stage_config["examples"])]
    selected = [
        SupervisedTrainingExample(
            ordinal=index,
            example_id=example.example_id,
            input_sha256=example.input_sha256,
            prompt_sha256=example.prompt_sha256,
            prompt_token_ids=example.prompt_token_ids,
            target_class=example.target_class,
            target_token_id=example.target_token_id,
        )
        for index, example in enumerate(selected)
    ]
    dataset = SupervisedTrainingDataset(selected)
    seed = int(stage_config["seed"])
    set_deterministic_seed(seed)
    model = build_peft_model(config, torch, AutoModelForCausalLM, LoraConfig, get_peft_model)
    collator = AnswerTokenCollator(
        pad_token_id=int(tokenizer.pad_token_id),
        maximum_prompt_tokens=int(config["objective"]["maximum_prompt_tokens"]),
    )
    initial_trainable_sha256 = trainable_parameter_sha256(model)
    optimizer_config = config["optimization"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        total_steps=int(stage_config["optimizer_steps"]),
        warmup_steps=int(stage_config["warmup_steps"]),
    )
    epoch_reports = []
    completed_steps = 0
    inference_start = time.perf_counter()
    for epoch_index in range(int(stage_config["epochs"])):
        dataloader = make_epoch_dataloader(
            dataset,
            collator=collator,
            batch_size=int(optimizer_config["batch_size"]),
            seed=seed,
            epoch_index=epoch_index,
        )
        report = train_one_epoch(
            model,
            dataloader,
            optimizer,
            scheduler,
            device=torch.device("cuda:0"),
            gradient_accumulation_steps=int(optimizer_config["gradient_accumulation_steps"]),
            maximum_gradient_norm=float(optimizer_config["maximum_gradient_norm"]),
            starting_optimizer_step=completed_steps,
        )
        completed_steps += int(report["optimizer_steps"])
        epoch_reports.append({"epoch_index": epoch_index, **report})
    training_seconds = time.perf_counter() - inference_start
    require(completed_steps == int(stage_config["optimizer_steps"]), "completed optimizer-step drift")
    final_trainable_sha256 = trainable_parameter_sha256(model)
    require(final_trainable_sha256 != initial_trainable_sha256, "adapter did not change")
    probe = probe_logits(model, selected, collator, torch)
    report = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "seed": seed,
        "examples": len(selected),
        "class_counts": dict(sorted(Counter(example.target_class for example in selected).items())),
        "epochs": int(stage_config["epochs"]),
        "optimizer_steps": completed_steps,
        "trainable_parameter_count": trainable_parameter_count(model),
        "total_parameter_count": total_parameter_count(model),
        "trainable_percentage": 100.0 * trainable_parameter_count(model) / total_parameter_count(model),
        "trainable_parameter_sha256_before": initial_trainable_sha256,
        "trainable_parameter_sha256_after": final_trainable_sha256,
        "parameters_changed": True,
        "all_losses_finite": all(epoch["all_losses_finite"] for epoch in epoch_reports),
        "all_trainable_parameters_received_gradients": all(epoch["all_trainable_parameters_received_gradients"] for epoch in epoch_reports),
        "epoch_reports": epoch_reports,
        "training_input_projection_sha256": sha256_text(canonical_json([{"example_id": example.example_id, "input_sha256": example.input_sha256, "target_token_id": example.target_token_id} for example in selected])),
        "raw_prompts_persisted": False,
        "questions_persisted": False,
        "passages_persisted": False,
        "numerical_confidence_targets_used": False,
    }
    destination, verification = publish_checkpoint(
        stage=stage,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        report=report,
        probe_after_training=probe,
        config=config,
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        PeftModel=PeftModel,
        examples=selected,
        collator=collator,
    )
    details = {
        "command": [sys.executable, "scripts/train_lora.py", "--stage", stage, "--config", config_path.relative_to(REPO_ROOT).as_posix()],
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "checkpoint_directory": destination.relative_to(REPO_ROOT).as_posix(),
        "checkpoint_artifact_hashes_sha256": verification["complete"]["artifact_hashes_sha256"],
        "training_seconds": training_seconds,
        "report": report,
    }
    return details, destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pinned answer-token LoRA training")
    parser.add_argument("--stage", required=True, choices=ALLOWED_STAGES)
    parser.add_argument("--config", type=Path, default=Path("configs/lora.yaml"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = _timestamp()
    started = time.perf_counter()
    try:
        details, destination = run(args.stage, args.config)
        details = {**details, "elapsed_seconds": time.perf_counter() - started}
        manifest = write_run_manifest(args.stage, "success", started_at, details)
        print("LORA TRAINING: PASS")
        print(f"stage={args.stage}")
        print(f"checkpoint_directory={destination.relative_to(REPO_ROOT).as_posix()}")
        print("report=" + canonical_json(details["report"]))
        print(f"run_manifest={manifest.relative_to(REPO_ROOT).as_posix()}")
        print("SCOPE: supervised answer-token LoRA adaptation only; no numerical-confidence target, calibration data, or test data was used.")
        return 0
    except Exception as error:
        try:
            manifest = write_run_manifest(
                args.stage,
                "failed",
                started_at,
                {
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
