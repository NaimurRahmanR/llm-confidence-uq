from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, fields
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_confidence_uq.degradations import (  # noqa: E402
    BOUNDARY_PATTERN_TEXT,
    CONDITIONS,
    CONTRADICTION_TEMPLATE,
    DISTRACTOR_SELECTOR,
    LEXICAL_ALGORITHM,
    NORMALIZATION_VERSION,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SENTENCE_SPLITTER,
    TOKEN_PATTERN_TEXT,
    DonorText,
    TransformationExample,
    TransformationResult,
    build_all_conditions,
    canonical_json,
    encode_text,
    sha256_text,
)


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
ROW_FIELDS = {
    "changed_from_original",
    "condition",
    "condition_index",
    "input_id",
    "input_sha256",
    "original_passage_sha256",
    "original_passage_tokens",
    "original_prompt_sha256",
    "original_prompt_tokens",
    "parameters",
    "prompt_tokens",
    "protocol_version",
    "quality_flags",
    "rendered_prompt_sha256",
    "renderer_version",
    "research_split",
    "row_sha256",
    "runtime_maximum_prompt_tokens",
    "runtime_reason_code",
    "runtime_status",
    "schema_version",
    "source_index",
    "source_split",
    "transformation_id",
    "transformation_reason_code",
    "transformation_status",
    "transformed_passage_sha256",
    "transformed_passage_tokens",
}
FORBIDDEN_KEYS = {
    "answer",
    "label",
    "messages",
    "passage",
    "prompt",
    "question",
    "raw_text",
    "record_sha256",
    "selection_rank_sha256",
    "target",
    "text",
}
QUALITY_FLAG_ALLOWLIST = {
    "fragment_shorter_than_requested",
    "token_boundary_adjusted",
    "whole_passage_removed",
    "zero_lexical_overlap",
}
TRANSFORMATION_REASON_ALLOWLIST = {
    "empty_passage_tokenization",
    "empty_selected_sentence_tokenization",
    "no_eligible_donor",
    "no_exact_decodable_prefix",
    "no_question_lexical_tokens",
    "no_sentence",
    "unchanged_result",
}
RUNTIME_OVERFLOW_REASON = "runtime_sequence_overflow"
RUNTIME_NOT_RUN_REASON = "transformation_not_runnable"
PACKAGE_NAMES = (
    "datasets",
    "huggingface-hub",
    "numpy",
    "pyarrow",
    "pyyaml",
    "tokenizers",
    "torch",
    "transformers",
)
EXPECTED_STATIC_FILE_SHA256 = {
    "data_config": "412fd6da74212e071159463e104329efefbc8fbe6b853c29c825e3ab37cb64ac",
    "degradation_config": "303238ae02025c9e5cb59bbd061949b6798f5fde8936a1f0604d15f0552cf2e3",
    "degradation_module": "8d71c5305d04d888b68181df9d43e30e76ea4c6f6f56494a65e53697d8544337",
}
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_PACKAGE_VERSIONS = {
    "datasets": "4.0.0",
    "huggingface-hub": "1.23.0",
    "pyyaml": "6.0.3",
    "transformers": "5.13.1",
}
EXPECTED_TOKENIZER_CLASS = "Qwen2Tokenizer"


@dataclass(frozen=True)
class PromptSpec:
    system: str
    user_template: str
    classification_prefix: str
    renderer_version: str
    maximum_prompt_tokens: int

    def __post_init__(self) -> None:
        if not self.system:
            raise ValueError("system prompt must be non-empty")
        if self.user_template != "Passage: {passage}\nQuestion: {question}":
            raise ValueError("unexpected BoolQ user template")
        if self.classification_prefix != "Answer:":
            raise ValueError("unexpected classification prefix")
        if not self.renderer_version:
            raise ValueError("renderer_version must be non-empty")
        if self.maximum_prompt_tokens <= 0:
            raise ValueError("maximum_prompt_tokens must be positive")


@dataclass(frozen=True)
class SelectedInput:
    input_id: str
    source_split: str
    source_index: int
    input_sha256: str
    original_passage_sha256: str
    original_passage_tokens: int
    original_prompt_sha256: str
    original_prompt_tokens: int
    question: str
    passage: str

    def __post_init__(self) -> None:
        if not self.input_id.startswith("boolqinput-"):
            raise ValueError("input_id must use the label-blind boolqinput namespace")
        if self.source_split != "validation":
            raise ValueError("degradation inputs must come from validation")
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        for name in (
            "input_sha256",
            "original_passage_sha256",
            "original_prompt_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.original_passage_tokens <= 0:
            raise ValueError("original passage must have at least one token")
        if self.original_prompt_tokens <= 0:
            raise ValueError("original prompt must have at least one token")
        if not self.question.strip() or not self.passage.strip():
            raise ValueError("question and passage must be non-empty")


class CachingTokenizer:
    """Memoize deterministic scalar encode/decode operations without persisting text."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._encode_cache: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[int, ...]] = {}
        self._decode_cache: dict[
            tuple[tuple[int, ...], tuple[tuple[str, str], ...]], str
        ] = {}

    @staticmethod
    def _kwargs_key(kwargs: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(key), repr(value)) for key, value in kwargs.items()))

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        if not isinstance(text, str):
            raise TypeError("tokenizer input must be a scalar string")
        key = (text, self._kwargs_key(kwargs))
        if key not in self._encode_cache:
            encoded = self._tokenizer(text, **kwargs)
            token_ids = encoded["input_ids"]
            if not isinstance(token_ids, (list, tuple)) or not all(
                isinstance(token_id, int) for token_id in token_ids
            ):
                raise TypeError("tokenizer must return flat integer input_ids")
            self._encode_cache[key] = tuple(token_ids)
        return {"input_ids": list(self._encode_cache[key])}

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        ids = tuple(int(token_id) for token_id in token_ids)
        key = (ids, self._kwargs_key(kwargs))
        if key not in self._decode_cache:
            decoded = self._tokenizer.decode(list(ids), **kwargs)
            if not isinstance(decoded, str):
                raise TypeError("tokenizer decode must return a string")
            self._decode_cache[key] = decoded
        return self._decode_cache[key]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    require(bool(rows), "JSONL rows must not be empty")
    return "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _require_sha256(value: Any, name: str) -> None:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )


def _require_revision(value: Any, name: str) -> None:
    require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase 40-character Git revision",
    )


def _require_blind_input_id(value: Any, name: str = "input_id") -> None:
    prefix = "boolqinput-"
    require(
        isinstance(value, str)
        and value.startswith(prefix)
        and len(value) == len(prefix) + 64
        and all(character in "0123456789abcdef" for character in value[len(prefix) :]),
        f"{name} must be a full label-blind input identifier",
    )


def _all_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _all_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_keys(nested)


def _all_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _all_string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_string_values(nested)


def make_label_blind_input_id(
    *,
    dataset_revision: str,
    source_split: str,
    source_index: int,
    input_sha256: str,
) -> str:
    _require_revision(dataset_revision, "dataset_revision")
    _require_sha256(input_sha256, "input_sha256")
    require(source_index >= 0, "source_index must be non-negative")
    identity = {
        "dataset_revision": dataset_revision,
        "input_sha256": input_sha256,
        "source_index": source_index,
        "source_split": source_split,
    }
    return "boolqinput-" + sha256_text(canonical_json(identity))


def render_locked_prompt(
    tokenizer: Any,
    prompt_spec: PromptSpec,
    *,
    question: str,
    passage: str,
) -> str:
    messages = [
        {"role": "system", "content": prompt_spec.system},
        {
            "role": "user",
            "content": prompt_spec.user_template.format(
                passage=passage,
                question=question,
            ),
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    require(isinstance(rendered, str), "chat template must return text")
    return rendered + prompt_spec.classification_prefix


def read_verified_jsonl(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> list[dict[str, Any]]:
    _require_sha256(expected_sha256, "expected manifest SHA-256")
    require(path.is_file(), f"manifest not found: {path}")
    require(not path.is_symlink(), f"manifest must not be a symbolic link: {path}")
    payload = path.read_bytes()
    require(
        sha256_bytes(payload) == expected_sha256,
        f"manifest SHA-256 mismatch: {path}",
    )
    require(payload.endswith(b"\n"), "manifest must end with LF")
    require(b"\r\n" not in payload, "manifest must use LF line endings")
    raw_lines = payload.splitlines()
    require(len(raw_lines) == expected_rows, "unexpected manifest row count")
    require(all(raw_lines), "manifest contains a blank line")
    rows = [json.loads(line.decode("utf-8")) for line in raw_lines]
    require(
        all(isinstance(row, dict) for row in rows),
        "manifest rows must be JSON objects",
    )
    require(
        all(set(row) == EXPECTED_MANIFEST_FIELDS for row in rows),
        "manifest schema mismatch",
    )
    require(
        canonical_jsonl_bytes(rows) == payload,
        "manifest is not canonical compact JSONL",
    )
    require(
        len({str(row["example_id"]) for row in rows}) == expected_rows,
        "duplicate source-manifest example_id",
    )
    locations = {
        (str(row["source_split"]), int(row["source_index"])) for row in rows
    }
    require(len(locations) == expected_rows, "duplicate source location")
    require(
        all(row["research_split"] == "test" for row in rows),
        "non-test row in source manifest",
    )
    require(
        all(row["source_split"] == "validation" for row in rows),
        "test manifest must use the official validation split",
    )
    return rows


def prompt_spec_from_configs(
    data_config: Mapping[str, Any],
    degradation_config: Mapping[str, Any],
) -> PromptSpec:
    require(data_config["schema_version"] == 1, "unsupported data config schema")
    require(
        degradation_config["schema_version"] == 1,
        "unsupported degradation config schema",
    )
    prompt = data_config["prompt"]
    runtime = degradation_config["runtime"]
    selection = data_config["selection"]
    protocol = degradation_config["protocol"]
    tokenizer = degradation_config["tokenizer"]
    model = data_config["model"]

    require(prompt["apply_chat_template"] is True, "chat template must be enabled")
    require(prompt["add_generation_prompt"] is True, "generation prompt must be enabled")
    require(prompt["add_special_tokens"] is False, "special tokens must not be added twice")
    require(prompt["padding"] is False, "prompt padding must be disabled")
    require(prompt["truncation"] is False, "prompt truncation must be disabled")
    require(runtime["truncation_allowed"] is False, "runtime truncation must be disabled")
    require(
        runtime["overflow_policy"] == "preserve_as_failed_row",
        "unexpected overflow policy",
    )
    require(
        int(runtime["maximum_sequence_tokens"])
        == int(selection["runtime_max_sequence_tokens"]),
        "runtime ceilings disagree",
    )
    require(
        runtime["renderer_version"] == prompt["renderer_version"],
        "renderer versions disagree",
    )
    require(
        runtime["renderer_version"] == "qwen-chat-boolq-confidence-v1",
        "approved renderer version drift",
    )
    require(
        int(runtime["maximum_sequence_tokens"]) == 1024,
        "approved runtime prompt ceiling drift",
    )
    require(tokenizer["id"] == model["tokenizer_id"], "tokenizer IDs disagree")
    require(
        tokenizer["revision"] == model["tokenizer_revision"],
        "tokenizer revisions disagree",
    )
    require(
        tokenizer["add_special_tokens_for_passage_operations"] is False,
        "passage operations must not add special tokens",
    )
    require(
        tokenizer["exact_decode_reencode_required"] is True,
        "exact decode/re-encode must be required",
    )
    require(tuple(protocol["condition_order"]) == CONDITIONS, "condition order drift")
    require(protocol["version"] == PROTOCOL_VERSION, "protocol version drift")
    require(protocol["labels_available_to_transformations"] is False, "labels exposed")
    require(protocol["persist_raw_text"] is False, "raw-text persistence enabled")
    require(protocol["preserve_not_applicable_rows"] is True, "rows may be dropped")
    require(protocol["preserve_failed_rows"] is True, "failed rows may be dropped")
    require(protocol["ordered_severity_claim_allowed"] is False, "severity claim enabled")
    require(
        int(protocol["seed"]) == int(selection["seed"]),
        "selection and transformation seeds disagree",
    )
    require(
        degradation_config["lexical_proxy"]["normalization"] == NORMALIZATION_VERSION,
        "normalization version drift",
    )
    require(
        degradation_config["lexical_proxy"]["algorithm"] == LEXICAL_ALGORITHM,
        "lexical algorithm drift",
    )
    require(
        degradation_config["lexical_proxy"]["token_pattern"] == TOKEN_PATTERN_TEXT,
        "token pattern drift",
    )
    require(
        degradation_config["lexical_proxy"]["sentence_splitter"] == SENTENCE_SPLITTER,
        "sentence splitter drift",
    )
    require(
        degradation_config["lexical_proxy"]["sentence_boundary_pattern"]
        == BOUNDARY_PATTERN_TEXT,
        "sentence boundary drift",
    )
    require(
        degradation_config["distractor"]["algorithm"] == DISTRACTOR_SELECTOR,
        "distractor selector drift",
    )
    require(
        degradation_config["distractor"]["source_pool"] == "selected_test_examples",
        "unexpected distractor source pool",
    )
    require(
        degradation_config["distractor"]["transductive_input_use"] is True,
        "transductive distractor use must be disclosed",
    )
    require(
        degradation_config["distractor"]["uses_labels"] is False,
        "distractor labels enabled",
    )
    require(
        degradation_config["contradiction"]["algorithm"] == LEXICAL_ALGORITHM,
        "contradiction selector drift",
    )
    require(
        degradation_config["contradiction"]["template"] == CONTRADICTION_TEMPLATE,
        "contradiction template drift",
    )
    require(
        degradation_config["contradiction"]["uses_labels"] is False,
        "contradiction labels enabled",
    )
    require(
        degradation_config["input"]["research_split"] == "test",
        "unexpected degradation research split",
    )
    require(
        int(degradation_config["input"]["expected_examples"]) == 400,
        "unexpected degradation input count",
    )
    require(
        degradation_config["prefix_truncation"]["algorithm"]
        == "token-prefix-half-ceiling-v1",
        "prefix truncation algorithm drift",
    )
    require(
        int(degradation_config["prefix_truncation"]["ratio_numerator"]) == 1
        and int(degradation_config["prefix_truncation"]["ratio_denominator"]) == 2
        and degradation_config["prefix_truncation"]["rounding"] == "ceiling",
        "prefix truncation ratio drift",
    )
    require(
        degradation_config["lexical_proxy"]["tie_break"]
        == "earliest-character-span-v1",
        "lexical tie-break drift",
    )
    require(
        degradation_config["lexical_proxy"]["zero_overlap_policy"]
        == "select_earliest-and-flag",
        "zero-overlap policy drift",
    )
    require(
        degradation_config["distractor"]["exclude_target_example"] is True
        and degradation_config["distractor"]["exclude_identical_passage"] is True,
        "distractor exclusion policy drift",
    )

    return PromptSpec(
        system=str(prompt["system"]),
        user_template=str(prompt["user_template"]),
        classification_prefix=str(prompt["classification_prefix"]),
        renderer_version=str(prompt["renderer_version"]),
        maximum_prompt_tokens=int(runtime["maximum_sequence_tokens"]),
    )


def verify_full_validation_source(
    dataset_split: Any,
    *,
    expected_rows: int,
    expected_sha256: str,
) -> dict[str, Any]:
    require(len(dataset_split) == expected_rows, "validation source row-count mismatch")
    require(
        set(dataset_split.column_names) == {"question", "passage", "answer"},
        "validation source schema mismatch",
    )
    digest = hashlib.sha256()
    counts = Counter()
    for source_index, source in enumerate(dataset_split):
        question = source["question"]
        passage = source["passage"]
        answer = source["answer"]
        require(isinstance(question, str) and question.strip(), "invalid source question")
        require(isinstance(passage, str) and passage.strip(), "invalid source passage")
        require(answer in (False, True), "invalid source answer")
        answer = bool(answer)
        counts[answer] += 1
        canonical_record = canonical_json(
            {"answer": answer, "passage": passage, "question": question}
        )
        digest.update(canonical_record.encode("utf-8"))
        digest.update(b"\n")
        require(source_index >= 0, "invalid source index")
    observed_sha256 = digest.hexdigest()
    require(observed_sha256 == expected_sha256, "validation source SHA-256 mismatch")
    return {
        "rows": expected_rows,
        "class_counts": {
            "false": int(counts.get(False, 0)),
            "true": int(counts.get(True, 0)),
        },
        "canonical_sha256": observed_sha256,
    }


def reconstruct_selected_inputs(
    manifest_rows: Sequence[Mapping[str, Any]],
    dataset_split: Any,
    tokenizer: Any,
    prompt_spec: PromptSpec,
    *,
    dataset_id: str,
    dataset_revision: str,
    selection_seed: int,
    eligibility_max_prompt_tokens: int,
) -> tuple[list[SelectedInput], dict[str, Any]]:
    cached = CachingTokenizer(tokenizer)
    selected: list[SelectedInput] = []
    class_counts = Counter()
    seen_input_ids: set[str] = set()

    for manifest_row in manifest_rows:
        require(manifest_row["dataset_id"] == dataset_id, "manifest dataset ID mismatch")
        require(
            manifest_row["dataset_revision"] == dataset_revision,
            "manifest dataset revision mismatch",
        )
        require(manifest_row["schema_version"] == 1, "manifest schema version mismatch")
        require(
            int(manifest_row["selection_seed"]) == selection_seed,
            "manifest selection seed mismatch",
        )
        require(
            int(manifest_row["eligibility_max_prompt_tokens"])
            == eligibility_max_prompt_tokens,
            "manifest eligibility ceiling mismatch",
        )
        source_index = int(manifest_row["source_index"])
        require(0 <= source_index < len(dataset_split), "source index out of range")
        source = dataset_split[source_index]
        question = source["question"]
        passage = source["passage"]
        source_answer = source["answer"]
        require(isinstance(question, str) and question.strip(), "invalid selected question")
        require(isinstance(passage, str) and passage.strip(), "invalid selected passage")
        require(source_answer in (False, True), "invalid selected answer")
        answer = bool(source_answer)
        require(manifest_row["answer"] in (False, True), "invalid manifest answer")
        require(bool(manifest_row["answer"]) == answer, "selected label mismatch")
        class_counts[answer] += 1

        canonical_record = canonical_json(
            {"answer": answer, "passage": passage, "question": question}
        )
        record_sha256 = sha256_text(canonical_record)
        input_sha256 = sha256_text(
            canonical_json({"passage": passage, "question": question})
        )
        passage_sha256 = sha256_text(passage)
        require(
            manifest_row["record_sha256"] == record_sha256,
            "selected record SHA-256 mismatch",
        )
        require(
            manifest_row["input_sha256"] == input_sha256,
            "selected input SHA-256 mismatch",
        )
        require(
            manifest_row["original_passage_sha256"] == passage_sha256,
            "selected passage SHA-256 mismatch",
        )
        expected_source_id = (
            f"boolq-validation-{source_index:05d}-{record_sha256[:12]}"
        )
        require(
            manifest_row["example_id"] == expected_source_id,
            "source manifest example_id mismatch",
        )

        input_id = make_label_blind_input_id(
            dataset_revision=dataset_revision,
            source_split="validation",
            source_index=source_index,
            input_sha256=input_sha256,
        )
        require(input_id not in seen_input_ids, "duplicate label-blind input ID")
        seen_input_ids.add(input_id)
        prompt = render_locked_prompt(
            tokenizer,
            prompt_spec,
            question=question,
            passage=passage,
        )
        prompt_tokens = len(encode_text(cached, prompt))
        require(
            prompt_tokens == int(manifest_row["prompt_tokens"]),
            "original prompt-token count mismatch",
        )
        require(
            prompt_tokens <= eligibility_max_prompt_tokens,
            "selected prompt exceeds eligibility ceiling",
        )
        passage_tokens = len(encode_text(cached, passage))
        require(passage_tokens > 0, "selected passage tokenizes to empty")

        selected.append(
            SelectedInput(
                input_id=input_id,
                source_split="validation",
                source_index=source_index,
                input_sha256=input_sha256,
                original_passage_sha256=passage_sha256,
                original_passage_tokens=passage_tokens,
                original_prompt_sha256=sha256_text(prompt),
                original_prompt_tokens=prompt_tokens,
                question=question,
                passage=passage,
            )
        )

    require(
        {field.name for field in fields(SelectedInput)}.isdisjoint(
            {"answer", "label", "record_sha256", "selection_rank_sha256", "target"}
        ),
        "SelectedInput contains a label-dependent field",
    )
    ordered = sorted(selected, key=lambda item: item.input_id)
    projection = [
        {
            "input_id": item.input_id,
            "input_sha256": item.input_sha256,
            "original_passage_sha256": item.original_passage_sha256,
            "original_passage_tokens": item.original_passage_tokens,
            "original_prompt_sha256": item.original_prompt_sha256,
            "original_prompt_tokens": item.original_prompt_tokens,
            "source_index": item.source_index,
            "source_split": item.source_split,
        }
        for item in ordered
    ]
    projection_payload = canonical_jsonl_bytes(projection)
    return ordered, {
        "selected_rows": len(ordered),
        "selected_class_counts_for_source_integrity_only": {
            "false": int(class_counts.get(False, 0)),
            "true": int(class_counts.get(True, 0)),
        },
        "label_blind_projection_sha256": sha256_bytes(projection_payload),
    }


def _transformation_identity(
    selected: SelectedInput,
    result: TransformationResult,
    transformed_passage_sha256: str | None,
    changed_from_original: bool | None,
) -> str:
    identity = {
        "changed_from_original": changed_from_original,
        "condition": result.condition,
        "input_id": selected.input_id,
        "original_passage_sha256": result.original_passage_sha256,
        "parameters": dict(result.parameters),
        "protocol_version": PROTOCOL_VERSION,
        "quality_flags": list(result.quality_flags),
        "schema_version": SCHEMA_VERSION,
        "transformation_reason_code": result.reason_code,
        "transformation_status": result.status,
        "transformed_passage_sha256": transformed_passage_sha256,
    }
    return "boolqdeg-" + sha256_text(canonical_json(identity))


def materialize_runtime_row(
    *,
    selected: SelectedInput,
    result: TransformationResult,
    condition_index: int,
    tokenizer: Any,
    cached_tokenizer: CachingTokenizer,
    prompt_spec: PromptSpec,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    require(result.example_id == selected.input_id, "transformation input ID drift")
    require(
        result.original_passage_sha256 == selected.original_passage_sha256,
        "transformation original-passage hash drift",
    )
    metadata = result.manifest_metadata()
    transformed_sha256 = metadata["transformed_passage_sha256"]
    changed_from_original = metadata["changed_from_original"]
    raw_values = [selected.question, selected.passage]

    if result.transformed_passage is None:
        transformed_tokens = None
        rendered_prompt = None
        rendered_prompt_sha256 = None
        prompt_tokens = None
        runtime_status = "not_run"
        runtime_reason_code = RUNTIME_NOT_RUN_REASON
    else:
        raw_values.append(result.transformed_passage)
        transformed_tokens = len(encode_text(cached_tokenizer, result.transformed_passage))
        rendered_prompt = render_locked_prompt(
            tokenizer,
            prompt_spec,
            question=selected.question,
            passage=result.transformed_passage,
        )
        raw_values.append(rendered_prompt)
        rendered_prompt_sha256 = sha256_text(rendered_prompt)
        prompt_tokens = len(encode_text(cached_tokenizer, rendered_prompt))
        if prompt_tokens > prompt_spec.maximum_prompt_tokens:
            runtime_status = "failed"
            runtime_reason_code = RUNTIME_OVERFLOW_REASON
        else:
            runtime_status = "ok"
            runtime_reason_code = None

    transformation_id = _transformation_identity(
        selected,
        result,
        transformed_sha256,
        changed_from_original,
    )
    row_without_hash = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "transformation_id": transformation_id,
        "input_id": selected.input_id,
        "research_split": "test",
        "source_split": selected.source_split,
        "source_index": selected.source_index,
        "input_sha256": selected.input_sha256,
        "original_passage_sha256": selected.original_passage_sha256,
        "original_passage_tokens": selected.original_passage_tokens,
        "original_prompt_sha256": selected.original_prompt_sha256,
        "original_prompt_tokens": selected.original_prompt_tokens,
        "condition_index": condition_index,
        "condition": result.condition,
        "transformation_status": result.status,
        "transformation_reason_code": result.reason_code,
        "runtime_status": runtime_status,
        "runtime_reason_code": runtime_reason_code,
        "quality_flags": list(result.quality_flags),
        "transformed_passage_sha256": transformed_sha256,
        "transformed_passage_tokens": transformed_tokens,
        "changed_from_original": changed_from_original,
        "rendered_prompt_sha256": rendered_prompt_sha256,
        "prompt_tokens": prompt_tokens,
        "runtime_maximum_prompt_tokens": prompt_spec.maximum_prompt_tokens,
        "renderer_version": prompt_spec.renderer_version,
        "parameters": dict(result.parameters),
    }
    row = dict(row_without_hash)
    row["row_sha256"] = sha256_text(canonical_json(row_without_hash))
    row_string_values = set(_all_string_values(row))
    require(
        all(raw not in row_string_values for raw in raw_values),
        "raw text leaked into transformation metadata",
    )
    return row, tuple(raw_values)


def validate_core_rows(
    rows: Sequence[Mapping[str, Any]],
    selected_inputs: Sequence[SelectedInput],
    prompt_spec: PromptSpec,
) -> dict[str, Any]:
    expected_rows = len(selected_inputs) * len(CONDITIONS)
    require(len(rows) == expected_rows, "wrong transformation row count")
    require(all(set(row) == ROW_FIELDS for row in rows), "unstable row schema")
    selected_by_id = {item.input_id: item for item in selected_inputs}
    require(
        len(selected_by_id) == len(selected_inputs),
        "duplicate selected input ID",
    )
    expected_order = [
        (item.input_id, condition_index)
        for item in sorted(selected_inputs, key=lambda value: value.input_id)
        for condition_index in range(len(CONDITIONS))
    ]
    observed_order = [
        (str(row["input_id"]), int(row["condition_index"])) for row in rows
    ]
    require(observed_order == expected_order, "transformation row order drift")
    require(
        len({str(row["transformation_id"]) for row in rows}) == expected_rows,
        "duplicate transformation ID",
    )
    require(
        len({str(row["row_sha256"]) for row in rows}) == expected_rows,
        "duplicate row SHA-256",
    )
    per_condition = Counter()
    runtime_failures = 0
    transformation_failures = 0
    not_applicable = 0

    for row in rows:
        require(
            not (set(_all_keys(row)) & FORBIDDEN_KEYS),
            "forbidden raw or label-dependent key in row",
        )
        input_id = str(row["input_id"])
        _require_blind_input_id(input_id)
        require(input_id in selected_by_id, "unknown row input ID")
        selected = selected_by_id[input_id]
        row_string_values = set(_all_string_values(row))
        require(
            selected.question not in row_string_values
            and selected.passage not in row_string_values,
            "source raw text leaked into row",
        )
        condition_index = int(row["condition_index"])
        require(
            0 <= condition_index < len(CONDITIONS)
            and row["condition"] == CONDITIONS[condition_index],
            "condition index mismatch",
        )
        per_condition[str(row["condition"])] += 1
        require(row["schema_version"] == SCHEMA_VERSION, "row schema version drift")
        require(row["protocol_version"] == PROTOCOL_VERSION, "row protocol drift")
        require(row["research_split"] == "test", "row research split drift")
        require(row["source_split"] == selected.source_split, "row source split drift")
        require(row["source_index"] == selected.source_index, "row source index drift")
        require(row["input_sha256"] == selected.input_sha256, "row input hash drift")
        require(
            row["original_passage_sha256"] == selected.original_passage_sha256,
            "row original passage hash drift",
        )
        require(
            row["original_prompt_sha256"] == selected.original_prompt_sha256,
            "row original prompt hash drift",
        )
        require(
            row["original_prompt_tokens"] == selected.original_prompt_tokens,
            "row original prompt-token drift",
        )
        require(
            row["original_passage_tokens"] == selected.original_passage_tokens,
            "row original passage-token drift",
        )
        require(
            row["runtime_maximum_prompt_tokens"]
            == prompt_spec.maximum_prompt_tokens,
            "row runtime ceiling drift",
        )
        require(
            row["renderer_version"] == prompt_spec.renderer_version,
            "row renderer version drift",
        )
        _require_sha256(row["input_sha256"], "row input_sha256")
        _require_sha256(
            row["original_passage_sha256"],
            "row original_passage_sha256",
        )
        _require_sha256(row["original_prompt_sha256"], "row original_prompt_sha256")
        _require_sha256(row["row_sha256"], "row_sha256")
        require(
            isinstance(row["transformation_id"], str)
            and row["transformation_id"].startswith("boolqdeg-")
            and len(row["transformation_id"]) == len("boolqdeg-") + 64,
            "invalid transformation ID",
        )
        _require_sha256(
            str(row["transformation_id"])[len("boolqdeg-") :],
            "transformation ID digest",
        )
        require(
            row["quality_flags"]
            == sorted(set(row["quality_flags"])),
            "quality flags must be sorted and unique",
        )
        require(
            set(row["quality_flags"]).issubset(QUALITY_FLAG_ALLOWLIST),
            "unknown quality flag",
        )
        require(isinstance(row["parameters"], dict), "parameters must be an object")
        require(
            not (set(_all_keys(row["parameters"])) & FORBIDDEN_KEYS),
            "forbidden parameter key",
        )
        require(
            all(
                value is None or isinstance(value, (bool, int, str))
                for value in row["parameters"].values()
            ),
            "unsafe parameter value",
        )
        for parameter_name, parameter_value in row["parameters"].items():
            if parameter_name.endswith("sha256") and parameter_value is not None:
                _require_sha256(parameter_value, parameter_name)
            if parameter_name.endswith("example_id") and parameter_value is not None:
                _require_blind_input_id(parameter_value, parameter_name)

        row_without_hash = {key: value for key, value in row.items() if key != "row_sha256"}
        require(
            row["row_sha256"] == sha256_text(canonical_json(row_without_hash)),
            "row SHA-256 mismatch",
        )
        transformation_status = row["transformation_status"]
        transformation_reason = row["transformation_reason_code"]
        runtime_status = row["runtime_status"]
        runtime_reason = row["runtime_reason_code"]
        transformed_sha256 = row["transformed_passage_sha256"]
        transformed_tokens = row["transformed_passage_tokens"]
        prompt_sha256 = row["rendered_prompt_sha256"]
        prompt_tokens = row["prompt_tokens"]

        require(
            transformation_status in {"ok", "not_applicable", "failed"},
            "unknown transformation status",
        )
        if transformation_status == "ok":
            require(transformation_reason is None, "ok transformation has a reason")
            _require_sha256(transformed_sha256, "transformed_passage_sha256")
            require(
                isinstance(transformed_tokens, int) and transformed_tokens >= 0,
                "invalid transformed passage token count",
            )
            require(isinstance(row["changed_from_original"], bool), "invalid changed flag")
            require(
                row["changed_from_original"]
                == (transformed_sha256 != row["original_passage_sha256"]),
                "changed flag disagrees with passage hashes",
            )
            _require_sha256(prompt_sha256, "rendered_prompt_sha256")
            require(
                isinstance(prompt_tokens, int) and prompt_tokens > 0,
                "invalid prompt token count",
            )
            if runtime_status == "ok":
                require(runtime_reason is None, "ok runtime has a reason")
                require(
                    prompt_tokens <= prompt_spec.maximum_prompt_tokens,
                    "runtime ok above prompt ceiling",
                )
            elif runtime_status == "failed":
                runtime_failures += 1
                require(
                    runtime_reason == RUNTIME_OVERFLOW_REASON,
                    "unexpected runtime failure reason",
                )
                require(
                    prompt_tokens > prompt_spec.maximum_prompt_tokens,
                    "overflow failure at or below prompt ceiling",
                )
            else:
                raise RuntimeError("successful transformation was not run")
        else:
            transformation_failures += int(transformation_status == "failed")
            not_applicable += int(transformation_status == "not_applicable")
            require(
                transformation_reason in TRANSFORMATION_REASON_ALLOWLIST,
                "unexpected transformation reason",
            )
            require(transformed_sha256 is None, "non-ok transformation has output hash")
            require(transformed_tokens is None, "non-ok transformation has output tokens")
            require(row["changed_from_original"] is None, "non-ok transformation changed flag")
            require(runtime_status == "not_run", "non-ok transformation was run")
            require(
                runtime_reason == RUNTIME_NOT_RUN_REASON,
                "unexpected not-run reason",
            )
            require(prompt_sha256 is None, "not-run row has prompt hash")
            require(prompt_tokens is None, "not-run row has prompt tokens")

        transformation_identity = {
            "changed_from_original": row["changed_from_original"],
            "condition": row["condition"],
            "input_id": row["input_id"],
            "original_passage_sha256": row["original_passage_sha256"],
            "parameters": row["parameters"],
            "protocol_version": row["protocol_version"],
            "quality_flags": row["quality_flags"],
            "schema_version": row["schema_version"],
            "transformation_reason_code": row["transformation_reason_code"],
            "transformation_status": row["transformation_status"],
            "transformed_passage_sha256": row["transformed_passage_sha256"],
        }
        require(
            row["transformation_id"]
            == "boolqdeg-" + sha256_text(canonical_json(transformation_identity)),
            "transformation ID payload mismatch",
        )

        if row["condition"] == "original":
            require(transformation_status == "ok", "original transformation failed")
            require(runtime_status == "ok", "original runtime failed")
            require(
                transformed_sha256 == selected.original_passage_sha256,
                "original condition changed passage",
            )
            require(row["changed_from_original"] is False, "original changed flag")
            require(
                prompt_tokens == selected.original_prompt_tokens,
                "original prompt-token mismatch",
            )
            require(
                prompt_sha256 == selected.original_prompt_sha256,
                "original prompt hash mismatch",
            )
        if row["condition"] == "no_passage":
            require(transformation_status == "ok", "no-passage transformation failed")
            require(runtime_status == "ok", "no-passage runtime failed")
            require(
                transformed_sha256 == sha256_text(""),
                "no-passage output is not empty",
            )
            require(transformed_tokens == 0, "empty passage has nonzero token count")
        if row["condition"] == "irrelevant_distractor" and transformation_status == "ok":
            donor_id = str(row["parameters"]["distractor_example_id"])
            require(donor_id in selected_by_id, "distractor donor is outside blind pool")
            require(donor_id != input_id, "distractor selected the target itself")
            donor = selected_by_id[donor_id]
            require(
                row["parameters"]["distractor_original_passage_sha256"]
                == donor.original_passage_sha256,
                "distractor donor passage hash mismatch",
            )
            require(
                donor.original_passage_sha256 != selected.original_passage_sha256,
                "distractor donor has identical passage",
            )

    require(
        per_condition == Counter({condition: len(selected_inputs) for condition in CONDITIONS}),
        "wrong per-condition row counts",
    )
    return {
        "rows": len(rows),
        "inputs": len(selected_inputs),
        "conditions": len(CONDITIONS),
        "runtime_failures": runtime_failures,
        "transformation_failures": transformation_failures,
        "not_applicable": not_applicable,
    }


def _nearest_rank(values: Sequence[int], percent: int) -> int | None:
    if not values:
        return None
    require(0 <= percent <= 100, "percent must be in [0, 100]")
    ordered = sorted(values)
    index = ((len(ordered) - 1) * percent + 50) // 100
    return int(ordered[index])


def integer_statistics(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "p50": _nearest_rank(values, 50),
        "p90": _nearest_rank(values, 90),
        "p95": _nearest_rank(values, 95),
        "p99": _nearest_rank(values, 99),
        "max": max(values),
    }


def _counter_json(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {key: counts[key] for key in sorted(counts)}


def _fraction_counter(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    fractions = []
    for row in rows:
        parameters = row["parameters"]
        if "overlap_numerator" in parameters and "overlap_denominator" in parameters:
            value = Fraction(
                int(parameters["overlap_numerator"]),
                int(parameters["overlap_denominator"]),
            )
            fractions.append(f"{value.numerator}/{value.denominator}")
    return _counter_json(fractions)


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_blind_projection_sha256: str,
    manifest_payload: bytes,
) -> dict[str, Any]:
    original_prompt_by_id = {
        str(row["input_id"]): int(row["prompt_tokens"])
        for row in rows
        if row["condition"] == "original"
    }
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        prompt_values = [
            int(row["prompt_tokens"])
            for row in condition_rows
            if row["prompt_tokens"] is not None
        ]
        passage_values = [
            int(row["transformed_passage_tokens"])
            for row in condition_rows
            if row["transformed_passage_tokens"] is not None
        ]
        deltas = [
            int(row["prompt_tokens"]) - original_prompt_by_id[str(row["input_id"])]
            for row in condition_rows
            if row["prompt_tokens"] is not None
        ]
        ratio_ppm = [
            (
                int(row["transformed_passage_tokens"]) * 1_000_000
                + int(row["original_passage_tokens"]) // 2
            )
            // int(row["original_passage_tokens"])
            for row in condition_rows
            if row["transformed_passage_tokens"] is not None
            and int(row["original_passage_tokens"]) > 0
        ]
        conditions[condition] = {
            "rows": len(condition_rows),
            "transformation_status_counts": _counter_json(
                row["transformation_status"] for row in condition_rows
            ),
            "runtime_status_counts": _counter_json(
                row["runtime_status"] for row in condition_rows
            ),
            "transformation_reason_counts": _counter_json(
                row["transformation_reason_code"]
                for row in condition_rows
                if row["transformation_reason_code"] is not None
            ),
            "runtime_reason_counts": _counter_json(
                row["runtime_reason_code"]
                for row in condition_rows
                if row["runtime_reason_code"] is not None
            ),
            "quality_flag_counts": _counter_json(
                flag for row in condition_rows for flag in row["quality_flags"]
            ),
            "changed_counts": {
                "false": sum(row["changed_from_original"] is False for row in condition_rows),
                "null": sum(row["changed_from_original"] is None for row in condition_rows),
                "true": sum(row["changed_from_original"] is True for row in condition_rows),
            },
            "prompt_token_statistics": {
                **integer_statistics(prompt_values),
                "over_512": sum(value > 512 for value in prompt_values),
                "over_768": sum(value > 768 for value in prompt_values),
                "over_1024": sum(value > 1024 for value in prompt_values),
            },
            "prompt_token_delta_from_original_statistics": integer_statistics(deltas),
            "transformed_passage_token_statistics": integer_statistics(passage_values),
            "retained_passage_token_ratio_ppm_statistics": integer_statistics(ratio_ppm),
            "exact_overlap_fraction_counts": _fraction_counter(condition_rows),
        }

    distractor_rows = [
        row
        for row in rows
        if row["condition"] == "irrelevant_distractor"
        and row["transformation_status"] == "ok"
    ]
    donor_counts = Counter(
        str(row["parameters"]["distractor_example_id"]) for row in distractor_rows
    )
    all_input_ids = {str(row["input_id"]) for row in rows}
    used_donors = set(donor_counts)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "label_blind_projection_sha256": label_blind_projection_sha256,
        "manifest": {
            "bytes": len(manifest_payload),
            "sha256": sha256_bytes(manifest_payload),
        },
        "inputs": len(all_input_ids),
        "condition_count": len(CONDITIONS),
        "rows": len(rows),
        "condition_order": list(CONDITIONS),
        "conditions_are_ordered_by_severity": False,
        "percentile_definition": "nearest-index-round-half-up-v1",
        "conditions": conditions,
        "distractor_reuse": {
            "successful_rows": len(distractor_rows),
            "unique_donors_used": len(used_donors),
            "unused_selected_inputs": len(all_input_ids - used_donors),
            "maximum_reuse": max(donor_counts.values(), default=0),
            "counts_by_blind_input_id": {
                key: donor_counts[key] for key in sorted(donor_counts)
            },
        },
        "empty_passage_sha256": sha256_text(""),
    }
    return summary


def build_core(
    selected_inputs: Sequence[SelectedInput],
    tokenizer: Any,
    prompt_spec: PromptSpec,
    *,
    seed: int,
    maximum_fragment_tokens: int,
    label_blind_projection_sha256: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    require(seed >= 0, "seed must be non-negative")
    require(maximum_fragment_tokens > 0, "maximum fragment tokens must be positive")
    ordered = sorted(selected_inputs, key=lambda item: item.input_id)
    require(len({item.input_id for item in ordered}) == len(ordered), "duplicate input ID")
    cached = CachingTokenizer(tokenizer)
    donors = [
        DonorText(item.input_id, item.passage)
        for item in ordered
    ]
    rows: list[dict[str, Any]] = []
    for selected in ordered:
        require(
            len(encode_text(cached, selected.passage))
            == selected.original_passage_tokens,
            "original passage-token drift",
        )
        example = TransformationExample(
            selected.input_id,
            selected.question,
            selected.passage,
        )
        results = build_all_conditions(
            example,
            donors,
            cached,
            seed=seed,
            maximum_fragment_tokens=maximum_fragment_tokens,
        )
        require(
            tuple(result.condition for result in results) == CONDITIONS,
            "condition order drift during build",
        )
        for condition_index, result in enumerate(results):
            row, _ = materialize_runtime_row(
                selected=selected,
                result=result,
                condition_index=condition_index,
                tokenizer=tokenizer,
                cached_tokenizer=cached,
                prompt_spec=prompt_spec,
            )
            rows.append(row)

    validation = validate_core_rows(rows, ordered, prompt_spec)
    manifest_payload = canonical_jsonl_bytes(rows)
    summary = summarize_rows(
        rows,
        label_blind_projection_sha256=label_blind_projection_sha256,
        manifest_payload=manifest_payload,
    )
    payloads = {
        "test.jsonl": manifest_payload,
        "summary.json": canonical_bytes(summary),
    }
    return payloads, validation


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def resolve_hub_revision(
    *,
    api: Any,
    repo_type: str,
    repo_id: str,
    revision: str,
) -> str:
    if repo_type == "dataset":
        info = api.dataset_info(repo_id=repo_id, revision=revision, token=False)
    elif repo_type == "model":
        info = api.model_info(repo_id=repo_id, revision=revision, token=False)
    else:
        raise ValueError(f"unsupported repo type: {repo_type}")
    resolved = str(info.sha)
    require(resolved == revision, f"{repo_type} revision mismatch")
    return resolved


def _resolve_repo_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else REPO_ROOT / path
    root_absolute = Path(os.path.abspath(REPO_ROOT))
    candidate_absolute = Path(os.path.abspath(candidate))
    require(
        candidate_absolute == root_absolute
        or root_absolute in candidate_absolute.parents,
        f"path escapes repository: {candidate_absolute}",
    )
    current = candidate_absolute
    while True:
        require(not current.is_symlink(), f"symbolic links are forbidden: {current}")
        if current == root_absolute:
            break
        current = current.parent
    return candidate_absolute.resolve(strict=False)


def preflight_and_write(
    output_dir: Path,
    payloads: Mapping[str, bytes],
) -> dict[str, str]:
    output_dir = _resolve_repo_path(output_dir)
    if output_dir.exists():
        require(output_dir.is_dir(), "output path exists but is not a directory")
        require(not output_dir.is_symlink(), "output directory must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {name: output_dir / name for name in payloads}
    for name, destination in destinations.items():
        require(not destination.is_symlink(), f"artifact is a symlink: {destination}")
        if destination.exists():
            require(destination.is_file(), f"artifact is not a file: {destination}")
            require(
                destination.read_bytes() == payloads[name],
                f"refusing to overwrite differing artifact: {destination}",
            )

    statuses: dict[str, str] = {}
    for name, destination in destinations.items():
        payload = payloads[name]
        if destination.exists():
            statuses[name] = "verified_existing"
            continue
        temporary = output_dir / f".{name}.tmp-{os.getpid()}"
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        statuses[name] = "created"
    for name, destination in destinations.items():
        require(destination.read_bytes() == payloads[name], f"round-trip failure: {name}")
    return statuses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data.yaml"),
    )
    parser.add_argument(
        "--degradation-config",
        type=Path,
        default=Path("configs/degradations.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/degradations"),
    )
    args = parser.parse_args()

    import yaml
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    data_config_path = _resolve_repo_path(args.data_config)
    degradation_config_path = _resolve_repo_path(args.degradation_config)
    output_dir = _resolve_repo_path(args.output_dir)
    data_config = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    degradation_config = yaml.safe_load(
        degradation_config_path.read_text(encoding="utf-8")
    )
    prompt_spec = prompt_spec_from_configs(data_config, degradation_config)
    degradation_module_path = REPO_ROOT / "src/llm_confidence_uq/degradations.py"
    observed_static_hashes = {
        "data_config": file_sha256(data_config_path),
        "degradation_config": file_sha256(degradation_config_path),
        "degradation_module": file_sha256(degradation_module_path),
    }
    require(
        observed_static_hashes == EXPECTED_STATIC_FILE_SHA256,
        f"audited static-file hashes drifted: {observed_static_hashes}",
    )
    require(
        sys.version.split()[0] == EXPECTED_PYTHON_VERSION,
        f"audited Python version drifted: {sys.version.split()[0]}",
    )
    observed_package_versions = package_versions()
    require(
        all(
            observed_package_versions[name] == expected
            for name, expected in EXPECTED_PACKAGE_VERSIONS.items()
        ),
        "audited package versions drifted",
    )
    expected_examples = int(degradation_config["input"]["expected_examples"])
    require(
        expected_examples == int(data_config["selection"]["test_size"]) == 400,
        "expected test size drift",
    )
    manifest_relative = Path(str(degradation_config["input"]["manifest_path"]))
    require(not manifest_relative.is_absolute(), "manifest path must be repository-relative")
    source_manifest_path = _resolve_repo_path(manifest_relative)
    source_manifest_sha256 = str(
        degradation_config["input"]["manifest_sha256"]
    )
    manifest_rows = read_verified_jsonl(
        source_manifest_path,
        expected_sha256=source_manifest_sha256,
        expected_rows=expected_examples,
    )

    dataset_id = str(data_config["data"]["dataset_id"])
    dataset_revision = str(data_config["data"]["dataset_revision"])
    tokenizer_id = str(data_config["model"]["tokenizer_id"])
    tokenizer_revision = str(data_config["model"]["tokenizer_revision"])
    api = HfApi()
    resolved_dataset_revision = resolve_hub_revision(
        api=api,
        repo_type="dataset",
        repo_id=dataset_id,
        revision=dataset_revision,
    )
    resolved_tokenizer_revision = resolve_hub_revision(
        api=api,
        repo_type="model",
        repo_id=tokenizer_id,
        revision=tokenizer_revision,
    )
    validation_split = load_dataset(
        dataset_id,
        revision=dataset_revision,
        split="validation",
        verification_mode="all_checks",
        token=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        trust_remote_code=False,
        token=False,
    )
    require(
        tokenizer.__class__.__name__ == EXPECTED_TOKENIZER_CLASS,
        f"unexpected tokenizer class: {tokenizer.__class__.__name__}",
    )
    source_report = verify_full_validation_source(
        validation_split,
        expected_rows=int(data_config["data"]["source_expected_rows"]["validation"]),
        expected_sha256=str(
            data_config["data"]["source_expected_sha256"]["validation"]
        ),
    )
    selection_seed = int(data_config["selection"]["seed"])
    eligibility_max = int(
        data_config["selection"]["original_prompt_eligibility_max_tokens"]
    )
    require(eligibility_max == 768, "approved eligibility ceiling drift")
    selected_inputs, selection_report = reconstruct_selected_inputs(
        manifest_rows,
        validation_split,
        tokenizer,
        prompt_spec,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        selection_seed=selection_seed,
        eligibility_max_prompt_tokens=eligibility_max,
    )
    maximum_fragment_tokens = int(
        degradation_config["distractor"]["maximum_fragment_tokens"]
    )
    require(
        maximum_fragment_tokens
        == int(degradation_config["contradiction"]["maximum_fragment_tokens"]),
        "fragment-token ceilings disagree",
    )
    require(maximum_fragment_tokens == 64, "approved fragment ceiling drift")

    first_payloads, first_validation = build_core(
        selected_inputs,
        tokenizer,
        prompt_spec,
        seed=selection_seed,
        maximum_fragment_tokens=maximum_fragment_tokens,
        label_blind_projection_sha256=selection_report[
            "label_blind_projection_sha256"
        ],
    )
    second_payloads, second_validation = build_core(
        list(reversed(selected_inputs)),
        tokenizer,
        prompt_spec,
        seed=selection_seed,
        maximum_fragment_tokens=maximum_fragment_tokens,
        label_blind_projection_sha256=selection_report[
            "label_blind_projection_sha256"
        ],
    )
    require(first_payloads == second_payloads, "in-process build bytes differ")
    require(
        first_validation == second_validation,
        "in-process validation reports differ",
    )

    implementation_hashes = {
        "data_config_sha256": file_sha256(data_config_path),
        "degradation_config_sha256": file_sha256(degradation_config_path),
        "degradation_module_sha256": file_sha256(
            degradation_module_path
        ),
        "integration_script_sha256": file_sha256(Path(__file__).resolve()),
        "integration_test_sha256": file_sha256(
            REPO_ROOT / "tests/test_prepare_degradations.py"
        ),
    }
    base_hashes = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(first_payloads.items())
    }
    chat_template = getattr(tokenizer, "chat_template", None)
    require(isinstance(chat_template, str) and chat_template, "tokenizer chat template missing")
    placeholder_rendered_prompt = render_locked_prompt(
        tokenizer,
        prompt_spec,
        question="{question}",
        passage="{passage}",
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": {
            "id": dataset_id,
            "requested_revision": dataset_revision,
            "resolved_revision": resolved_dataset_revision,
            "validation_fingerprint_supporting_only": str(
                getattr(validation_split, "_fingerprint", "")
            ),
            "validation_source": source_report,
        },
        "source_selection": {
            "manifest_bytes": source_manifest_path.stat().st_size,
            "manifest_sha256": source_manifest_sha256,
            "research_split": "test",
            **selection_report,
            "selection_was_predeclared_and_label_stratified": True,
            "labels_passed_to_transformation_operators": False,
            "label_permutation_invariance_scope": (
                "Core test.jsonl and summary.json are invariant conditional on the "
                "frozen selected inputs. Provenance intentionally retains aggregate "
                "source-integrity counts and the pinned source-manifest hash."
            ),
        },
        "tokenizer": {
            "id": tokenizer_id,
            "requested_revision": tokenizer_revision,
            "resolved_revision": resolved_tokenizer_revision,
            "class": tokenizer.__class__.__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "model_weights_loaded": False,
        },
        "prompt_renderer": {
            "version": prompt_spec.renderer_version,
            "system_sha256": sha256_text(prompt_spec.system),
            "user_template_sha256": sha256_text(prompt_spec.user_template),
            "tokenizer_chat_template_sha256": sha256_text(chat_template),
            "placeholder_rendered_prompt_sha256": sha256_text(
                placeholder_rendered_prompt
            ),
            "apply_chat_template": True,
            "add_generation_prompt": True,
            "classification_prefix": prompt_spec.classification_prefix,
            "tokenization": {
                "add_special_tokens": False,
                "padding": False,
                "truncation": False,
            },
            "maximum_input_prompt_tokens": prompt_spec.maximum_prompt_tokens,
        },
        "transformation": {
            "condition_order": list(CONDITIONS),
            "conditions_are_ordered_by_severity": False,
            "seed": selection_seed,
            "maximum_fragment_tokens": maximum_fragment_tokens,
            "distractor_pool": "selected-test-inputs-transductive",
            "labels_available_to_operators": False,
            "raw_text_persisted": False,
            "input_id_policy": (
                "sha256(dataset_revision,source_split,source_index,input_sha256); "
                "answer-derived source-manifest IDs are validated then discarded"
            ),
            "interpretation": (
                "Evaluation uses inherited original BoolQ targets under stress; "
                "transformations are not guaranteed to preserve answer validity."
            ),
            "experimental_unit_policy": (
                "The 2,400 rows are repeated conditions on 400 base inputs and "
                "must not be analysed as 2,400 independent examples."
            ),
            "claim_boundaries": dict(degradation_config["claim_boundary"]),
        },
        "verification": {
            "independent_in_process_builds": 2,
            "second_build_input_order_reversed": True,
            "payloads_byte_identical": True,
            **first_validation,
        },
        "implementation": {
            **implementation_hashes,
            "unicode_database_version": unicodedata.unidata_version,
        },
        "environment": {
            "python": sys.version.split()[0],
            "packages": package_versions(),
            "unicode_database_version": unicodedata.unidata_version,
        },
        "base_artifacts": base_hashes,
        "dynamic_metadata_policy": (
            "Timestamp and Git HEAD are excluded from deterministic data artifacts "
            "and belong in later experiment run manifests."
        ),
    }
    all_payloads = dict(first_payloads)
    all_payloads["provenance.json"] = canonical_bytes(provenance)
    ledger = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in sorted(all_payloads.items())
    }
    all_payloads["artifact_hashes.json"] = canonical_bytes(ledger)
    statuses = preflight_and_write(output_dir, all_payloads)

    reloaded_rows = [
        json.loads(line)
        for line in (output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    round_trip = validate_core_rows(reloaded_rows, selected_inputs, prompt_spec)
    require(round_trip == first_validation, "round-trip validation report drift")
    stored_ledger = json.loads(
        (output_dir / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    for name, expected in stored_ledger.items():
        payload = (output_dir / name).read_bytes()
        require(len(payload) == expected["bytes"], f"ledger byte mismatch: {name}")
        require(
            sha256_bytes(payload) == expected["sha256"],
            f"ledger SHA-256 mismatch: {name}",
        )

    gate_passed = (
        first_validation["runtime_failures"] == 0
        and first_validation["transformation_failures"] == 0
    )
    print(
        "DEGRADATION MANIFEST INTEGRATION:",
        "PASS" if gate_passed else "FAIL",
    )
    print(f"dataset_revision={resolved_dataset_revision}")
    print(f"tokenizer_revision={resolved_tokenizer_revision}")
    print(f"tokenizer_class={tokenizer.__class__.__name__}")
    print(
        "source_selection="
        + json.dumps(
            {
                **selection_report,
                "manifest_sha256": source_manifest_sha256,
            },
            sort_keys=True,
        )
    )
    print("validation=" + json.dumps(first_validation, sort_keys=True))
    summary = json.loads(first_payloads["summary.json"])
    for condition in CONDITIONS:
        report = summary["conditions"][condition]
        print(
            f"condition={condition} rows={report['rows']} "
            f"transformation_status={report['transformation_status_counts']} "
            f"runtime_status={report['runtime_status_counts']} "
            f"prompt_tokens={report['prompt_token_statistics']} "
            f"quality_flags={report['quality_flag_counts']}"
        )
    print(
        "distractor_reuse="
        + json.dumps(
            {
                key: value
                for key, value in summary["distractor_reuse"].items()
                if key != "counts_by_blind_input_id"
            },
            sort_keys=True,
        )
    )
    print("independent_builds_byte_identical=True")
    print("round_trip_validation=True")
    print(
        "runtime_environment="
        + json.dumps(
            {
                "git_head": git_head(),
                "packages": package_versions(),
                "python": sys.version.split()[0],
                "unicode_database": unicodedata.unidata_version,
            },
            sort_keys=True,
        )
    )
    for name in sorted(all_payloads):
        payload = all_payloads[name]
        print(
            f"artifact={name} status={statuses[name]} bytes={len(payload)} "
            f"sha256={sha256_bytes(payload)}"
        )
    print(
        "SCOPE: label-blind deterministic transformation manifests only; "
        "no model weights loaded and no inference performed."
    )
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
