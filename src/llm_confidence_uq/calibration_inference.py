from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
import math
from typing import Any, Callable, Mapping, Sequence

from .degradations import canonical_json, sha256_text
from .inference import ExpressedConfidenceParse, canonical_jsonl_bytes


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-original-calibration-inference-v1"
EXPECTED_MANIFEST_FIELDS = {
    "answer", "dataset_id", "dataset_revision", "eligibility_max_prompt_tokens",
    "example_id", "input_sha256", "original_passage_sha256", "prompt_tokens",
    "record_sha256", "research_split", "schema_version", "selection_rank_sha256",
    "selection_seed", "source_index", "source_split",
}
STAGES = {"smoke8": 8, "full": 200}
METHODS = ("baseline", "full_seed_1", "full_seed_2", "full_seed_3")
PREDICTION_FIELDS = {
    "schema_version", "protocol_version", "calibration_prediction_id", "row_sha256",
    "method", "model_id", "model_revision", "inference_config_sha256", "inference_seed",
    "input_id", "source_split", "source_index", "evidence_condition",
    "rendered_prompt_sha256", "prompt_tokens", "class_order", "yes_token_id", "no_token_id",
    "yes_logit", "no_logit", "p_yes_binary", "p_no_binary", "binary_probability_sum",
    "p_yes_full_vocabulary", "p_no_full_vocabulary", "yes_no_full_vocabulary_mass",
    "token_prediction", "token_confidence", "predictive_entropy", "generated_continuation",
    "generated_continuation_sha256", "generated_continuation_tokens", "generation_ended_with_eos",
    "generation_reached_max_new_tokens", "expressed_parser_version", "expressed_parser_valid",
    "expressed_parser_reason_code", "generated_answer", "expressed_confidence",
    "generated_answer_matches_token_prediction", "adapter_stage", "adapter_training_seed",
    "adapter_checkpoint_ledger_sha256", "adapter_weights_sha256", "adapter_semantic_config_sha256",
}


class CalibrationInferenceContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationInferenceContractError(message)


def _sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"invalid {label}")
    return value


@dataclass(frozen=True)
class CalibrationExample:
    ordinal: int
    input_id: str
    source_split: str
    source_index: int
    prompt: str
    prompt_tokens: int
    rendered_prompt_sha256: str
    runtime_status: str = "ok"

    def __post_init__(self) -> None:
        require(isinstance(self.ordinal, int) and self.ordinal >= 0, "invalid ordinal")
        require(isinstance(self.input_id, str) and self.input_id.startswith("boolqinput-"), "invalid input ID")
        _sha(self.input_id.removeprefix("boolqinput-"), "input ID")
        require(self.source_split == "train", "calibration source must be official train")
        require(isinstance(self.source_index, int) and self.source_index >= 0, "invalid source index")
        require(isinstance(self.prompt, str) and bool(self.prompt), "missing prompt")
        require(isinstance(self.prompt_tokens, int) and 0 < self.prompt_tokens <= 1024, "invalid prompt length")
        require(self.rendered_prompt_sha256 == sha256_text(self.prompt), "prompt hash drift")
        require(self.runtime_status == "ok", "calibration example is not runnable")


require({f.name for f in fields(CalibrationExample)}.isdisjoint({"answer", "label", "target", "correct"}), "example exposes label")


class CalibrationDataset:
    def __init__(self, examples: Sequence[CalibrationExample]) -> None:
        self._examples = tuple(examples)
        require(bool(self._examples), "empty calibration dataset")
        require([x.ordinal for x in self._examples] == list(range(len(self._examples))), "non-contiguous ordinals")
        require(len({x.input_id for x in self._examples}) == len(self._examples), "duplicate input ID")

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> CalibrationExample:
        return self._examples[index]


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]], *, expected_rows: int, dataset_id: str, dataset_revision: str, selection_seed: int, eligibility_max_prompt_tokens: int) -> None:
    require(len(rows) == expected_rows and expected_rows > 0, "calibration manifest row-count mismatch")
    for row in rows:
        require(set(row) == EXPECTED_MANIFEST_FIELDS, "calibration manifest schema drift")
        require(row["schema_version"] == 1 and row["dataset_id"] == dataset_id and row["dataset_revision"] == dataset_revision, "dataset contract drift")
        require(row["research_split"] == "calibration" and row["source_split"] == "train", "calibration split contract drift")
        require(row["answer"] in (False, True), "invalid calibration label")
        require(int(row["selection_seed"]) == selection_seed and int(row["eligibility_max_prompt_tokens"]) == eligibility_max_prompt_tokens, "selection contract drift")
    require(len({str(row["example_id"]) for row in rows}) == expected_rows, "duplicate example ID")
    require(len({int(row["source_index"]) for row in rows}) == expected_rows, "duplicate source location")


def select_stage_rows(rows: Sequence[Mapping[str, Any]], stage: str) -> list[Mapping[str, Any]]:
    require(stage in STAGES, "unknown calibration stage")
    require(len(rows) >= STAGES[stage], "manifest too small for stage")
    return list(rows[: STAGES[stage]])


def label_blind_projection_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json([{k: row[k] for k in ("input_sha256", "original_passage_sha256", "prompt_tokens", "source_index", "source_split")} for row in rows]))


def reconstruct_original_examples(manifest_rows: Sequence[Mapping[str, Any]], dataset_split: Any, *, render_prompt: Callable[[str, str], str], encode_prompt: Callable[[str], Sequence[int]], make_input_id: Callable[..., str], dataset_revision: str) -> list[CalibrationExample]:
    examples = []
    for ordinal, row in enumerate(manifest_rows):
        index = int(row["source_index"])
        require(0 <= index < len(dataset_split), "source index out of range")
        source = dataset_split[index]
        question, passage, answer = source["question"], source["passage"], source["answer"]
        require(isinstance(question, str) and isinstance(passage, str) and answer in (False, True), "invalid source record")
        record = canonical_json({"answer": bool(answer), "passage": passage, "question": question})
        input_payload = canonical_json({"passage": passage, "question": question})
        require(row["answer"] is bool(answer) and row["record_sha256"] == sha256_text(record), "record integrity mismatch")
        require(row["input_sha256"] == sha256_text(input_payload) and row["original_passage_sha256"] == sha256_text(passage), "input integrity mismatch")
        input_id = make_input_id(dataset_revision=dataset_revision, source_split="train", source_index=index, input_sha256=row["input_sha256"])
        prompt = render_prompt(passage, question)
        ids = list(encode_prompt(prompt))
        require(len(ids) == int(row["prompt_tokens"]), "prompt-token count drift")
        examples.append(CalibrationExample(ordinal, input_id, "train", index, prompt, len(ids), sha256_text(prompt)))
    return examples


def make_prediction_row(example: CalibrationExample, *, method: str, model_id: str, model_revision: str, inference_config_sha256: str, inference_seed: int, yes_token_id: int, no_token_id: int, yes_logit: float, no_logit: float, p_yes_binary: float, p_no_binary: float, p_yes_full_vocabulary: float, p_no_full_vocabulary: float, yes_no_full_vocabulary_mass: float, token_prediction: str, token_confidence: float, predictive_entropy: float, generated_continuation: str, generated_continuation_tokens: int, generation_ended_with_eos: bool, generation_reached_max_new_tokens: bool, parsed: ExpressedConfidenceParse, adapter: Mapping[str, Any] | None) -> dict[str, Any]:
    require(method in METHODS, "unknown method")
    adapter_values = {k: None for k in ("adapter_stage", "adapter_training_seed", "adapter_checkpoint_ledger_sha256", "adapter_weights_sha256", "adapter_semantic_config_sha256")}
    if adapter is not None:
        adapter_values.update(adapter)
    identity = {"input_id": example.input_id, "inference_config_sha256": inference_config_sha256, "method": method, "protocol_version": PROTOCOL_VERSION}
    row = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION,
        "calibration_prediction_id": "boolqcalpred-" + sha256_text(canonical_json(identity)),
        "method": method, "model_id": model_id, "model_revision": model_revision,
        "inference_config_sha256": inference_config_sha256, "inference_seed": inference_seed,
        "input_id": example.input_id, "source_split": "train", "source_index": example.source_index,
        "evidence_condition": "original", "rendered_prompt_sha256": example.rendered_prompt_sha256,
        "prompt_tokens": example.prompt_tokens, "class_order": ["Yes", "No"],
        "yes_token_id": yes_token_id, "no_token_id": no_token_id, "yes_logit": yes_logit, "no_logit": no_logit,
        "p_yes_binary": p_yes_binary, "p_no_binary": p_no_binary, "binary_probability_sum": p_yes_binary + p_no_binary,
        "p_yes_full_vocabulary": p_yes_full_vocabulary, "p_no_full_vocabulary": p_no_full_vocabulary,
        "yes_no_full_vocabulary_mass": yes_no_full_vocabulary_mass, "token_prediction": token_prediction,
        "token_confidence": token_confidence, "predictive_entropy": predictive_entropy,
        "generated_continuation": generated_continuation, "generated_continuation_sha256": sha256_text(generated_continuation),
        "generated_continuation_tokens": generated_continuation_tokens, "generation_ended_with_eos": generation_ended_with_eos,
        "generation_reached_max_new_tokens": generation_reached_max_new_tokens,
        "expressed_parser_version": parsed.parser_version, "expressed_parser_valid": parsed.valid,
        "expressed_parser_reason_code": parsed.reason_code, "generated_answer": parsed.answer,
        "expressed_confidence": parsed.confidence,
        "generated_answer_matches_token_prediction": parsed.answer == token_prediction if parsed.valid else None,
        **adapter_values,
    }
    row["row_sha256"] = sha256_text(canonical_json(row))
    validate_prediction_row(row)
    return row


def validate_prediction_row(row: Mapping[str, Any]) -> None:
    require(set(row) == PREDICTION_FIELDS, "calibration prediction schema drift")
    require(row["schema_version"] == 1 and row["protocol_version"] == PROTOCOL_VERSION, "protocol drift")
    require(row["method"] in METHODS and row["source_split"] == "train" and row["evidence_condition"] == "original", "identity drift")
    require(row["class_order"] == ["Yes", "No"] and row["token_prediction"] in ("Yes", "No"), "class drift")
    for key in ("yes_logit", "no_logit", "p_yes_binary", "p_no_binary", "token_confidence", "predictive_entropy"):
        require(isinstance(row[key], (int, float)) and math.isfinite(float(row[key])), f"nonfinite {key}")
    require(abs(float(row["p_yes_binary"]) + float(row["p_no_binary"]) - 1.0) <= 1e-5, "probability normalization drift")
    require(row["row_sha256"] == sha256_text(canonical_json({k: v for k, v in row.items() if k != "row_sha256"})), "row hash drift")
    require({"answer", "label", "correct", "target"}.isdisjoint(row), "label leakage")
    if row["method"] == "baseline":
        require(all(row[k] is None for k in ("adapter_stage", "adapter_training_seed", "adapter_checkpoint_ledger_sha256", "adapter_weights_sha256", "adapter_semantic_config_sha256")), "baseline has adapter identity")
    else:
        require(row["adapter_stage"] == row["method"], "adapter identity drift")


def validate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), "empty prediction rows")
    for row in rows:
        validate_prediction_row(row)
    require(len({row["calibration_prediction_id"] for row in rows}) == len(rows), "duplicate prediction ID")
    require(len({row["method"] for row in rows}) == 1, "mixed methods")


def prediction_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    validate_prediction_rows(rows)
    return canonical_jsonl_bytes(rows)


def summarize_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_prediction_rows(rows)
    return {"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "method": rows[0]["method"], "rows": len(rows), "source_split": "train", "evidence_condition": "original", "token_prediction_counts": dict(sorted(Counter(row["token_prediction"] for row in rows).items())), "expressed_parser_counts": dict(sorted(Counter("valid" if row["expressed_parser_valid"] else row["expressed_parser_reason_code"] for row in rows).items())), "contains_ground_truth": False, "contains_correctness": False}
