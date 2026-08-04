"""Deterministic confidence extraction for the frozen BoolQ baseline.

This module deliberately separates label-free model inference from later
evaluation.  Questions, passages, prompts, targets, and correctness values are
never serialized into prediction rows.  Raw prompts exist only in
``InferenceExample`` instances while the model is being called.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
import re
from typing import Any, Mapping, Sequence

try:  # Keep parsing/serialization usable in a non-PyTorch audit runtime.
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError:  # pragma: no cover - Colab requires PyTorch.
    torch = None

    class TorchDataset:  # type: ignore[no-redef]
        """Import-only fallback; tensor operations still fail closed."""


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-baseline-inference-v1"
PARSER_VERSION = "exact-continuation-v1"
METHOD = "frozen-base-model"
CLASS_ORDER = ("Yes", "No")
CLASS_CONTINUATIONS = (" Yes", " No")
EXPECTED_CLASS_TOKEN_IDS = (7414, 2308)
DEFAULT_INFERENCE_SEED = 20260803
SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
GIT_REVISION_PATTERN = re.compile(r"\A[0-9a-f]{40}\Z")
EXPRESSED_CONTINUATION_PATTERN = re.compile(
    r"\A (Yes|No)\nConfidence: (100|[1-9][0-9]?|0)\n?\Z"
)
FORBIDDEN_PREDICTION_KEYS = {
    "answer",
    "correct",
    "label",
    "messages",
    "passage",
    "prompt",
    "question",
    "raw_text",
    "record_sha256",
    "selection_rank_sha256",
    "target",
}
_PROBABILITY_TOLERANCE = 1e-6


class InferenceContractError(RuntimeError):
    """Raised when an audited inference invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InferenceContractError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""
    return ("\n".join(canonical_json(dict(row)) for row in rows) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, name: str) -> None:
    require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256 digest",
    )


def _require_git_revision(value: Any, name: str) -> None:
    require(
        isinstance(value, str) and GIT_REVISION_PATTERN.fullmatch(value) is not None,
        f"{name} must be a lowercase 40-character Git revision",
    )


def _require_finite(value: Any, name: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{name} must be finite")
    return result


def _close(left: float, right: float, *, tolerance: float = _PROBABILITY_TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tolerance


@dataclass(frozen=True)
class ExpressedConfidenceParse:
    parser_version: str
    valid: bool
    answer: str | None
    confidence: int | None
    reason_code: str | None

    def __post_init__(self) -> None:
        require(self.parser_version == PARSER_VERSION, "parser version drift")
        if self.valid:
            require(self.answer in CLASS_ORDER, "valid parse has invalid answer")
            require(
                isinstance(self.confidence, int)
                and not isinstance(self.confidence, bool)
                and 0 <= self.confidence <= 100,
                "valid parse has invalid confidence",
            )
            require(self.reason_code is None, "valid parse has a reason code")
        else:
            require(self.answer is None, "invalid parse exposes an answer")
            require(self.confidence is None, "invalid parse exposes confidence")
            require(
                self.reason_code
                in {
                    "empty_completion",
                    "answer_line_invalid",
                    "confidence_line_missing",
                    "format_mismatch",
                    "generation_max_new_tokens_reached",
                    "not_run",
                },
                "invalid parse has an unsupported reason code",
            )


def parse_expressed_continuation(completion: str) -> ExpressedConfidenceParse:
    """Parse only the exact continuation after the locked ``Answer:`` prefix."""

    if not isinstance(completion, str):
        raise TypeError("completion must be a string")
    match = EXPRESSED_CONTINUATION_PATTERN.fullmatch(completion)
    if match is not None:
        return ExpressedConfidenceParse(
            parser_version=PARSER_VERSION,
            valid=True,
            answer=match.group(1),
            confidence=int(match.group(2)),
            reason_code=None,
        )
    if completion == "":
        reason = "empty_completion"
    elif not (completion.startswith(" Yes\n") or completion.startswith(" No\n")):
        reason = "answer_line_invalid"
    elif "\nConfidence: " not in completion:
        reason = "confidence_line_missing"
    else:
        reason = "format_mismatch"
    return ExpressedConfidenceParse(
        parser_version=PARSER_VERSION,
        valid=False,
        answer=None,
        confidence=None,
        reason_code=reason,
    )


def effective_expressed_parse(
    completion: str,
    *,
    generation_reached_max_new_tokens: bool,
) -> ExpressedConfidenceParse:
    """Apply the generation-stop contract to the strict visible-text parse."""

    require(
        isinstance(generation_reached_max_new_tokens, bool),
        "generation max-token flag must be boolean",
    )
    parsed = parse_expressed_continuation(completion)
    if not generation_reached_max_new_tokens:
        return parsed
    return ExpressedConfidenceParse(
        parser_version=PARSER_VERSION,
        valid=False,
        answer=None,
        confidence=None,
        reason_code="generation_max_new_tokens_reached",
    )


@dataclass(frozen=True)
class InferenceExample:
    """One label-free in-memory prompt bound to a degradation row."""

    ordinal: int
    transformation_id: str
    degradation_row_sha256: str
    input_id: str
    source_split: str
    source_index: int
    condition_index: int
    condition: str
    runtime_status: str
    runtime_reason_code: str | None
    prompt: str | None
    prompt_tokens: int | None
    rendered_prompt_sha256: str | None = None

    def __post_init__(self) -> None:
        require(
            isinstance(self.ordinal, int)
            and not isinstance(self.ordinal, bool)
            and self.ordinal >= 0,
            "ordinal must be a non-negative integer",
        )
        require(
            isinstance(self.transformation_id, str)
            and self.transformation_id.startswith("boolqdeg-"),
            "unexpected transformation ID namespace",
        )
        _require_sha256(
            self.transformation_id.removeprefix("boolqdeg-"), "transformation ID"
        )
        _require_sha256(self.degradation_row_sha256, "degradation row SHA-256")
        require(
            isinstance(self.input_id, str) and self.input_id.startswith("boolqinput-"),
            "unexpected input ID namespace",
        )
        _require_sha256(self.input_id.removeprefix("boolqinput-"), "input ID")
        require(self.source_split == "validation", "unexpected source split")
        require(
            isinstance(self.source_index, int)
            and not isinstance(self.source_index, bool)
            and self.source_index >= 0,
            "negative source index",
        )
        require(
            isinstance(self.condition_index, int)
            and not isinstance(self.condition_index, bool)
            and 0 <= self.condition_index < 6,
            "condition index out of range",
        )
        require(isinstance(self.condition, str) and bool(self.condition), "condition missing")
        require(
            self.runtime_status in {"ok", "not_run", "failed"},
            "bad runtime status",
        )
        if self.runtime_status == "ok":
            require(
                isinstance(self.prompt, str) and bool(self.prompt),
                "runnable prompt missing",
            )
            require(
                isinstance(self.prompt_tokens, int)
                and not isinstance(self.prompt_tokens, bool)
                and 0 < self.prompt_tokens <= 1024,
                "runnable prompt-token count invalid",
            )
            expected_prompt_sha256 = sha256_text(self.prompt)
            if self.rendered_prompt_sha256 is None:
                object.__setattr__(
                    self,
                    "rendered_prompt_sha256",
                    expected_prompt_sha256,
                )
            else:
                _require_sha256(
                    self.rendered_prompt_sha256,
                    "rendered prompt SHA-256",
                )
                require(
                    self.rendered_prompt_sha256 == expected_prompt_sha256,
                    "rendered prompt SHA-256 disagrees with in-memory prompt",
                )
            require(self.runtime_reason_code is None, "runnable row has reason code")
        else:
            require(self.prompt is None, "non-runnable row retains a prompt")
            require(bool(self.runtime_reason_code), "non-runnable row lacks reason code")
            require(
                (self.rendered_prompt_sha256 is None) == (self.prompt_tokens is None),
                "non-runnable prompt provenance must be jointly present or absent",
            )
            if self.rendered_prompt_sha256 is not None:
                _require_sha256(
                    self.rendered_prompt_sha256,
                    "rendered prompt SHA-256",
                )
                require(
                    isinstance(self.prompt_tokens, int)
                    and not isinstance(self.prompt_tokens, bool)
                    and self.prompt_tokens > 0,
                    "non-runnable prompt-token count invalid",
                )


require(
    {field.name for field in fields(InferenceExample)}.isdisjoint(
        {"answer", "correct", "label", "target"}
    ),
    "InferenceExample contains a target field",
)


class InferenceDataset(TorchDataset):
    """Ordered map-style dataset for audited label-free inference examples."""

    def __init__(self, examples: Sequence[InferenceExample]) -> None:
        self._examples = tuple(examples)
        require(bool(self._examples), "inference dataset must not be empty")
        require(
            [example.ordinal for example in self._examples]
            == list(range(len(self._examples))),
            "inference ordinals must be contiguous and ordered",
        )
        identifiers = [example.transformation_id for example in self._examples]
        require(
            len(identifiers) == len(set(identifiers)),
            "duplicate transformation ID",
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> InferenceExample:
        return self._examples[index]


class PromptCollator:
    """Tokenize an ordered batch without truncation using left padding."""

    def __init__(self, tokenizer: Any, *, maximum_prompt_tokens: int) -> None:
        require(maximum_prompt_tokens > 0, "maximum prompt tokens must be positive")
        require(
            getattr(tokenizer, "padding_side", None) == "left",
            "tokenizer padding_side must be left",
        )
        self._tokenizer = tokenizer
        self._maximum_prompt_tokens = maximum_prompt_tokens

    def __call__(self, examples: Sequence[InferenceExample]) -> dict[str, Any]:
        require(bool(examples), "cannot collate an empty batch")
        require(
            all(
                example.runtime_status == "ok" and example.prompt is not None
                for example in examples
            ),
            "collator received a non-runnable example",
        )
        prompts = [str(example.prompt) for example in examples]
        encoded = self._tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        require(
            "input_ids" in encoded and "attention_mask" in encoded,
            "tokenizer batch missing tensors",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        require(getattr(input_ids, "ndim", None) == 2, "input IDs must be rank two")
        require(getattr(attention_mask, "ndim", None) == 2, "attention mask must be rank two")
        require(
            tuple(input_ids.shape) == tuple(attention_mask.shape),
            "input IDs and attention mask disagree",
        )
        observed_lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
        expected_lengths = [int(example.prompt_tokens) for example in examples]
        require(observed_lengths == expected_lengths, "batched prompt-token count drift")
        require(
            max(observed_lengths) <= self._maximum_prompt_tokens,
            "prompt exceeds runtime ceiling",
        )
        # Pass only the two audited causal-LM inputs.  Tokenizer-specific extras
        # (for example token-type IDs) must not silently alter the model call.
        return {
            "examples": tuple(examples),
            "model_inputs": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        }


def contextual_class_token_ids(
    tokenizer: Any,
    prompt: str,
    *,
    expected_ids: tuple[int, int] | None = EXPECTED_CLASS_TOKEN_IDS,
) -> tuple[int, int]:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    require(
        isinstance(prefix_ids, (list, tuple))
        and all(isinstance(token_id, int) for token_id in prefix_ids),
        "tokenizer returned invalid prompt IDs",
    )
    class_ids: list[int] = []
    for continuation in CLASS_CONTINUATIONS:
        combined = tokenizer.encode(prompt + continuation, add_special_tokens=False)
        require(
            isinstance(combined, (list, tuple))
            and all(isinstance(token_id, int) for token_id in combined),
            "tokenizer returned invalid continuation IDs",
        )
        require(
            list(combined[: len(prefix_ids)]) == list(prefix_ids),
            f"{continuation!r} changed the tokenized prompt prefix",
        )
        suffix = list(combined[len(prefix_ids) :])
        require(len(suffix) == 1, f"{continuation!r} is not one contextual token")
        class_ids.append(int(suffix[0]))
    result = (class_ids[0], class_ids[1])
    require(result[0] != result[1], "Yes and No resolved to the same token")
    if expected_ids is not None:
        require(result == expected_ids, f"class token IDs drifted: {result}")
    return result


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor confidence operations")
    return torch


def last_nonpadding_indices(attention_mask: Any) -> Any:
    torch_module = _require_torch()
    require(getattr(attention_mask, "ndim", None) == 2, "attention mask must be rank two")
    require(attention_mask.shape[1] > 0, "attention mask has zero sequence length")
    require(
        bool(((attention_mask == 0) | (attention_mask == 1)).all().item()),
        "attention mask must be binary",
    )
    mask = attention_mask.to(dtype=torch_module.bool)
    require(bool(mask.any(dim=1).all().item()), "attention mask contains an all-padding row")
    positions = torch_module.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    positions = positions.expand(mask.shape[0], -1)
    masked_positions = torch_module.where(
        mask,
        positions,
        torch_module.full_like(positions, -1),
    )
    return masked_positions.max(dim=1).values


def position_ids_from_attention_mask(attention_mask: Any) -> Any:
    """Build Qwen-compatible position IDs for left or right padded batches."""

    torch_module = _require_torch()
    # Reuse the complete mask validation, including binary and non-empty rows.
    last_nonpadding_indices(attention_mask)
    mask = attention_mask.to(dtype=torch_module.long)
    position_ids = mask.cumsum(dim=1) - 1
    return position_ids.masked_fill(mask == 0, 0).to(dtype=torch_module.long)


def gather_last_token_logits(logits: Any, attention_mask: Any) -> Any:
    torch_module = _require_torch()
    require(getattr(logits, "ndim", None) == 3, "logits must be rank three")
    require(getattr(attention_mask, "ndim", None) == 2, "attention mask must be rank two")
    require(
        logits.shape[0] == attention_mask.shape[0],
        "logits and attention-mask batch dimensions disagree",
    )
    indices = last_nonpadding_indices(attention_mask)
    if logits.shape[1] == 1:
        # ``logits_to_keep=1`` deliberately returns only the final model position,
        # while the attention mask continues to describe the complete prompt.
        require(
            bool((indices == attention_mask.shape[1] - 1).all().item()),
            "logits_to_keep=1 requires the final prompt position to be attended",
        )
        return logits[:, 0, :]
    require(
        logits.shape[1] == attention_mask.shape[1],
        "logits and attention-mask sequence dimensions disagree",
    )
    indices = indices.to(device=logits.device)
    batch_indices = torch_module.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, indices]


@dataclass(frozen=True)
class BinaryConfidenceBatch:
    yes_logit: Any
    no_logit: Any
    p_yes_binary: Any
    p_no_binary: Any
    p_yes_full_vocabulary: Any
    p_no_full_vocabulary: Any
    yes_no_full_vocabulary_mass: Any
    token_prediction_index: Any
    token_confidence: Any
    predictive_entropy: Any


def binary_confidence_from_logits(
    next_token_logits: Any,
    *,
    yes_token_id: int,
    no_token_id: int,
) -> BinaryConfidenceBatch:
    torch_module = _require_torch()
    require(
        getattr(next_token_logits, "ndim", None) == 2,
        "next-token logits must be rank two",
    )
    vocabulary_size = int(next_token_logits.shape[1])
    require(
        0 <= yes_token_id < vocabulary_size
        and 0 <= no_token_id < vocabulary_size
        and yes_token_id != no_token_id,
        "class token IDs are invalid for the vocabulary",
    )
    logits32 = next_token_logits.float()
    require(
        bool(torch_module.isfinite(logits32).all().item()),
        "non-finite full-vocabulary logits",
    )
    class_logits = torch_module.stack(
        [logits32[:, yes_token_id], logits32[:, no_token_id]], dim=1
    )
    require(bool(torch_module.isfinite(class_logits).all().item()), "non-finite class logits")
    log_normalizer = torch_module.logsumexp(logits32, dim=1)
    require(
        bool(torch_module.isfinite(log_normalizer).all().item()),
        "non-finite vocabulary normalizer",
    )
    binary_probabilities = torch_module.softmax(class_logits, dim=1)
    require(
        bool(torch_module.isfinite(binary_probabilities).all().item()),
        "non-finite binary probabilities",
    )
    probability_sums = binary_probabilities.sum(dim=1)
    require(
        bool(
            torch_module.allclose(
                probability_sums,
                torch_module.ones_like(probability_sums),
                atol=_PROBABILITY_TOLERANCE,
                rtol=_PROBABILITY_TOLERANCE,
            )
        ),
        "binary probabilities do not normalize",
    )
    full_yes = torch_module.exp(class_logits[:, 0] - log_normalizer)
    full_no = torch_module.exp(class_logits[:, 1] - log_normalizer)
    full_mass = full_yes + full_no
    require(
        bool(((full_mass >= 0.0) & (full_mass <= 1.0 + _PROBABILITY_TOLERANCE)).all().item()),
        "Yes/No full-vocabulary mass is outside [0,1]",
    )
    prediction_index = binary_probabilities.argmax(dim=1)
    token_confidence = binary_probabilities.gather(
        1, prediction_index.unsqueeze(1)
    ).squeeze(1)
    entropy = -(
        binary_probabilities * binary_probabilities.clamp_min(1e-12).log()
    ).sum(dim=1)
    return BinaryConfidenceBatch(
        yes_logit=class_logits[:, 0],
        no_logit=class_logits[:, 1],
        p_yes_binary=binary_probabilities[:, 0],
        p_no_binary=binary_probabilities[:, 1],
        p_yes_full_vocabulary=full_yes,
        p_no_full_vocabulary=full_no,
        yes_no_full_vocabulary_mass=full_mass,
        token_prediction_index=prediction_index,
        token_confidence=token_confidence,
        predictive_entropy=entropy,
    )


def trim_generated_token_ids(
    token_ids: Sequence[int],
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> list[int]:
    trimmed: list[int] = []
    for token_id in token_ids:
        value = int(token_id)
        if eos_token_id is not None and value == eos_token_id:
            break
        if pad_token_id is not None and value == pad_token_id:
            break
        trimmed.append(value)
    return trimmed


PREDICTION_FIELDS = {
    "schema_version",
    "protocol_version",
    "method",
    "prediction_id",
    "model_id",
    "model_revision",
    "inference_config_sha256",
    "inference_seed",
    "transformation_id",
    "degradation_row_sha256",
    "input_id",
    "source_split",
    "source_index",
    "condition_index",
    "condition",
    "rendered_prompt_sha256",
    "prompt_tokens",
    "inference_status",
    "inference_reason_code",
    "class_order",
    "yes_token_id",
    "no_token_id",
    "yes_logit",
    "no_logit",
    "p_yes_binary",
    "p_no_binary",
    "binary_probability_sum",
    "p_yes_full_vocabulary",
    "p_no_full_vocabulary",
    "yes_no_full_vocabulary_mass",
    "token_prediction",
    "token_confidence",
    "predictive_entropy",
    "generated_continuation",
    "generated_continuation_sha256",
    "generated_continuation_tokens",
    "generation_ended_with_eos",
    "generation_reached_max_new_tokens",
    "expressed_parser_version",
    "expressed_parser_valid",
    "expressed_parser_reason_code",
    "generated_answer",
    "expressed_confidence",
    "generated_answer_matches_token_prediction",
    "row_sha256",
}


def _prediction_identifier(
    *,
    model_revision: str,
    inference_config_sha256: str,
    inference_seed: int,
    transformation_id: str,
    degradation_row_sha256: str,
) -> str:
    identity = {
        "degradation_row_sha256": degradation_row_sha256,
        "inference_config_sha256": inference_config_sha256,
        "inference_seed": inference_seed,
        "method": METHOD,
        "model_revision": model_revision,
        "protocol_version": PROTOCOL_VERSION,
        "transformation_id": transformation_id,
    }
    return "boolqpred-" + sha256_text(canonical_json(identity))


def _validate_success_metrics(
    *,
    yes_logit: Any,
    no_logit: Any,
    p_yes_binary: Any,
    p_no_binary: Any,
    binary_probability_sum: Any,
    p_yes_full_vocabulary: Any,
    p_no_full_vocabulary: Any,
    yes_no_full_vocabulary_mass: Any,
    token_prediction: Any,
    token_confidence: Any,
    predictive_entropy: Any,
) -> None:
    yes_logit_value = _require_finite(yes_logit, "yes_logit")
    no_logit_value = _require_finite(no_logit, "no_logit")
    p_yes = _require_finite(p_yes_binary, "p_yes_binary")
    p_no = _require_finite(p_no_binary, "p_no_binary")
    stored_sum = _require_finite(binary_probability_sum, "binary_probability_sum")
    full_yes = _require_finite(p_yes_full_vocabulary, "p_yes_full_vocabulary")
    full_no = _require_finite(p_no_full_vocabulary, "p_no_full_vocabulary")
    full_mass = _require_finite(
        yes_no_full_vocabulary_mass, "yes_no_full_vocabulary_mass"
    )
    confidence = _require_finite(token_confidence, "token_confidence")
    entropy = _require_finite(predictive_entropy, "predictive_entropy")

    require(0.0 <= p_yes <= 1.0 and 0.0 <= p_no <= 1.0, "binary probabilities invalid")
    require(_close(p_yes + p_no, 1.0), "binary probabilities do not sum to one")
    require(_close(stored_sum, p_yes + p_no), "stored binary sum is inconsistent")

    logit_max = max(yes_logit_value, no_logit_value)
    yes_weight = math.exp(yes_logit_value - logit_max)
    no_weight = math.exp(no_logit_value - logit_max)
    expected_yes = yes_weight / (yes_weight + no_weight)
    expected_no = no_weight / (yes_weight + no_weight)
    require(
        _close(p_yes, expected_yes, tolerance=2e-6)
        and _close(p_no, expected_no, tolerance=2e-6),
        "binary probabilities disagree with Yes/No logits",
    )

    expected_prediction = "Yes" if p_yes >= p_no else "No"
    require(token_prediction == expected_prediction, "token prediction disagrees with class order")
    require(_close(confidence, max(p_yes, p_no)), "token confidence is not the class maximum")

    expected_entropy = -sum(
        probability * math.log(max(probability, 1e-12))
        for probability in (p_yes, p_no)
    )
    require(_close(entropy, expected_entropy, tolerance=2e-6), "predictive entropy drift")
    require(-_PROBABILITY_TOLERANCE <= entropy <= math.log(2.0) + 2e-6, "entropy out of range")

    require(
        0.0 <= full_yes <= 1.0
        and 0.0 <= full_no <= 1.0
        and 0.0 <= full_mass <= 1.0 + _PROBABILITY_TOLERANCE,
        "full-vocabulary probabilities invalid",
    )
    require(
        _close(full_yes + full_no, full_mass),
        "full-vocabulary mass does not equal its components",
    )


def make_prediction_row(
    example: InferenceExample,
    *,
    model_id: str,
    model_revision: str,
    inference_config_sha256: str,
    yes_token_id: int,
    no_token_id: int,
    inference_seed: int = DEFAULT_INFERENCE_SEED,
    yes_logit: float | None = None,
    no_logit: float | None = None,
    p_yes_binary: float | None = None,
    p_no_binary: float | None = None,
    p_yes_full_vocabulary: float | None = None,
    p_no_full_vocabulary: float | None = None,
    yes_no_full_vocabulary_mass: float | None = None,
    token_prediction: str | None = None,
    token_confidence: float | None = None,
    predictive_entropy: float | None = None,
    generated_continuation: str | None = None,
    generated_continuation_tokens: int | None = None,
    generation_ended_with_eos: bool = False,
    generation_reached_max_new_tokens: bool = False,
    expressed_parse: ExpressedConfidenceParse | None = None,
    inference_status: str = "ok",
    inference_reason_code: str | None = None,
) -> dict[str, Any]:
    require(inference_status in {"ok", "not_run"}, "unsupported inference status")
    _require_git_revision(model_revision, "model revision")
    _require_sha256(inference_config_sha256, "inference config SHA-256")
    require(isinstance(model_id, str) and bool(model_id), "model ID must be non-empty")
    require(
        isinstance(inference_seed, int)
        and not isinstance(inference_seed, bool)
        and inference_seed >= 0,
        "inference seed must be a non-negative integer",
    )
    require((yes_token_id, no_token_id) == EXPECTED_CLASS_TOKEN_IDS, "class token IDs drifted")
    require(
        isinstance(generation_ended_with_eos, bool),
        "generation EOS flag must be boolean",
    )
    require(
        isinstance(generation_reached_max_new_tokens, bool),
        "generation max-token flag must be boolean",
    )
    require(
        not (generation_ended_with_eos and generation_reached_max_new_tokens),
        "generation cannot both end with EOS and reach max_new_tokens",
    )

    if inference_status == "ok":
        require(example.runtime_status == "ok", "cannot run a non-runnable example")
        require(inference_reason_code is None, "successful inference has a reason code")
        require(isinstance(generated_continuation, str), "generated continuation missing")
        require(
            isinstance(generated_continuation_tokens, int)
            and not isinstance(generated_continuation_tokens, bool)
            and generated_continuation_tokens >= 0,
            "generated continuation token count invalid",
        )
        require(expressed_parse is not None, "expressed parser result missing")
        raw_parse = parse_expressed_continuation(generated_continuation)
        require(expressed_parse == raw_parse, "supplied parser result disagrees with completion")
        expressed_parse = effective_expressed_parse(
            generated_continuation,
            generation_reached_max_new_tokens=generation_reached_max_new_tokens,
        )
        stored_generation_ended_with_eos: bool | None = generation_ended_with_eos
        stored_generation_reached_max_new_tokens: bool | None = (
            generation_reached_max_new_tokens
        )
        binary_sum = float(p_yes_binary) + float(p_no_binary) if (
            p_yes_binary is not None and p_no_binary is not None
        ) else None
        _validate_success_metrics(
            yes_logit=yes_logit,
            no_logit=no_logit,
            p_yes_binary=p_yes_binary,
            p_no_binary=p_no_binary,
            binary_probability_sum=binary_sum,
            p_yes_full_vocabulary=p_yes_full_vocabulary,
            p_no_full_vocabulary=p_no_full_vocabulary,
            yes_no_full_vocabulary_mass=yes_no_full_vocabulary_mass,
            token_prediction=token_prediction,
            token_confidence=token_confidence,
            predictive_entropy=predictive_entropy,
        )
    else:
        require(example.runtime_status != "ok", "runnable example was marked not_run")
        require(bool(inference_reason_code), "not_run inference lacks a reason code")
        require(
            not generation_ended_with_eos
            and not generation_reached_max_new_tokens,
            "not_run row contains generation-stop flags",
        )
        require(
            all(
                value is None
                for value in (
                    yes_logit,
                    no_logit,
                    p_yes_binary,
                    p_no_binary,
                    p_yes_full_vocabulary,
                    p_no_full_vocabulary,
                    yes_no_full_vocabulary_mass,
                    token_confidence,
                    predictive_entropy,
                )
            ),
            "not_run row contains metrics",
        )
        require(token_prediction is None, "not_run row contains a prediction")
        require(generated_continuation is None, "not_run row contains generated text")
        require(generated_continuation_tokens is None, "not_run row contains token count")
        require(expressed_parse is None, "not_run row contains a parser result")
        expressed_parse = ExpressedConfidenceParse(
            parser_version=PARSER_VERSION,
            valid=False,
            answer=None,
            confidence=None,
            reason_code="not_run",
        )
        binary_sum = None
        stored_generation_ended_with_eos = None
        stored_generation_reached_max_new_tokens = None

    generated_hash = (
        sha256_text(generated_continuation)
        if generated_continuation is not None
        else None
    )
    prediction_id = _prediction_identifier(
        model_revision=model_revision,
        inference_config_sha256=inference_config_sha256,
        inference_seed=inference_seed,
        transformation_id=example.transformation_id,
        degradation_row_sha256=example.degradation_row_sha256,
    )
    row_without_hash: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "method": METHOD,
        "prediction_id": prediction_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "inference_config_sha256": inference_config_sha256,
        "inference_seed": inference_seed,
        "transformation_id": example.transformation_id,
        "degradation_row_sha256": example.degradation_row_sha256,
        "input_id": example.input_id,
        "source_split": example.source_split,
        "source_index": example.source_index,
        "condition_index": example.condition_index,
        "condition": example.condition,
        "rendered_prompt_sha256": example.rendered_prompt_sha256,
        "prompt_tokens": example.prompt_tokens,
        "inference_status": inference_status,
        "inference_reason_code": inference_reason_code,
        "class_order": list(CLASS_ORDER),
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "p_yes_binary": p_yes_binary,
        "p_no_binary": p_no_binary,
        "binary_probability_sum": binary_sum,
        "p_yes_full_vocabulary": p_yes_full_vocabulary,
        "p_no_full_vocabulary": p_no_full_vocabulary,
        "yes_no_full_vocabulary_mass": yes_no_full_vocabulary_mass,
        "token_prediction": token_prediction,
        "token_confidence": token_confidence,
        "predictive_entropy": predictive_entropy,
        "generated_continuation": generated_continuation,
        "generated_continuation_sha256": generated_hash,
        "generated_continuation_tokens": generated_continuation_tokens,
        "generation_ended_with_eos": stored_generation_ended_with_eos,
        "generation_reached_max_new_tokens": (
            stored_generation_reached_max_new_tokens
        ),
        "expressed_parser_version": expressed_parse.parser_version,
        "expressed_parser_valid": expressed_parse.valid,
        "expressed_parser_reason_code": expressed_parse.reason_code,
        "generated_answer": expressed_parse.answer,
        "expressed_confidence": expressed_parse.confidence,
        "generated_answer_matches_token_prediction": (
            expressed_parse.answer == token_prediction
            if expressed_parse.answer is not None and token_prediction is not None
            else None
        ),
    }
    row = dict(row_without_hash)
    row["row_sha256"] = sha256_text(canonical_json(row_without_hash))
    validate_prediction_row(row)
    return row


def validate_prediction_row(row: Mapping[str, Any]) -> None:
    require(set(row) == PREDICTION_FIELDS, "prediction row schema drift")
    require(
        set(row).isdisjoint(FORBIDDEN_PREDICTION_KEYS),
        "prediction row contains a forbidden key",
    )
    require(row["schema_version"] == SCHEMA_VERSION, "prediction schema version drift")
    require(row["protocol_version"] == PROTOCOL_VERSION, "prediction protocol drift")
    require(row["method"] == METHOD, "prediction method drift")
    require(row["class_order"] == list(CLASS_ORDER), "prediction class order drift")
    require(
        (row["yes_token_id"], row["no_token_id"]) == EXPECTED_CLASS_TOKEN_IDS,
        "prediction class token IDs drifted",
    )
    require(
        isinstance(row["prediction_id"], str)
        and str(row["prediction_id"]).startswith("boolqpred-"),
        "prediction ID namespace drift",
    )
    _require_sha256(str(row["prediction_id"]).removeprefix("boolqpred-"), "prediction ID")
    _require_git_revision(row["model_revision"], "model revision")
    _require_sha256(row["inference_config_sha256"], "inference config SHA-256")
    _require_sha256(row["degradation_row_sha256"], "degradation row SHA-256")
    _require_sha256(row["row_sha256"], "row SHA-256")
    require(
        isinstance(row["inference_seed"], int)
        and not isinstance(row["inference_seed"], bool)
        and row["inference_seed"] >= 0,
        "inference seed must be a non-negative integer",
    )
    require(
        isinstance(row["transformation_id"], str)
        and str(row["transformation_id"]).startswith("boolqdeg-"),
        "transformation ID namespace drift",
    )
    _require_sha256(
        str(row["transformation_id"]).removeprefix("boolqdeg-"),
        "transformation ID",
    )
    require(
        isinstance(row["input_id"], str)
        and str(row["input_id"]).startswith("boolqinput-"),
        "input ID namespace drift",
    )
    _require_sha256(str(row["input_id"]).removeprefix("boolqinput-"), "input ID")
    require(row["source_split"] == "validation", "prediction source split drift")
    require(
        isinstance(row["source_index"], int)
        and not isinstance(row["source_index"], bool)
        and row["source_index"] >= 0,
        "prediction source index invalid",
    )
    require(
        isinstance(row["condition_index"], int)
        and not isinstance(row["condition_index"], bool)
        and 0 <= row["condition_index"] < 6,
        "prediction condition index invalid",
    )
    require(isinstance(row["condition"], str) and bool(row["condition"]), "condition missing")
    rendered_prompt_sha256 = row["rendered_prompt_sha256"]
    prompt_tokens = row["prompt_tokens"]
    require(
        (rendered_prompt_sha256 is None) == (prompt_tokens is None),
        "prompt provenance must be jointly present or absent",
    )
    if rendered_prompt_sha256 is not None:
        _require_sha256(rendered_prompt_sha256, "rendered prompt SHA-256")
        require(
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
            and prompt_tokens > 0,
            "prompt-token count invalid",
        )

    expected_prediction_id = _prediction_identifier(
        model_revision=str(row["model_revision"]),
        inference_config_sha256=str(row["inference_config_sha256"]),
        inference_seed=int(row["inference_seed"]),
        transformation_id=str(row["transformation_id"]),
        degradation_row_sha256=str(row["degradation_row_sha256"]),
    )
    require(row["prediction_id"] == expected_prediction_id, "prediction ID digest mismatch")
    without_hash = {key: value for key, value in row.items() if key != "row_sha256"}
    require(
        row["row_sha256"] == sha256_text(canonical_json(without_hash)),
        "prediction row SHA-256 mismatch",
    )

    status = row["inference_status"]
    require(status in {"ok", "not_run"}, "invalid prediction status")
    if status == "ok":
        require(row["inference_reason_code"] is None, "ok prediction has reason code")
        require(
            rendered_prompt_sha256 is not None
            and isinstance(prompt_tokens, int)
            and prompt_tokens <= 1024,
            "ok prediction lacks valid runnable prompt provenance",
        )
        _validate_success_metrics(
            yes_logit=row["yes_logit"],
            no_logit=row["no_logit"],
            p_yes_binary=row["p_yes_binary"],
            p_no_binary=row["p_no_binary"],
            binary_probability_sum=row["binary_probability_sum"],
            p_yes_full_vocabulary=row["p_yes_full_vocabulary"],
            p_no_full_vocabulary=row["p_no_full_vocabulary"],
            yes_no_full_vocabulary_mass=row["yes_no_full_vocabulary_mass"],
            token_prediction=row["token_prediction"],
            token_confidence=row["token_confidence"],
            predictive_entropy=row["predictive_entropy"],
        )
        require(
            isinstance(row["generated_continuation"], str),
            "ok prediction lacks continuation",
        )
        require(
            isinstance(row["generated_continuation_tokens"], int)
            and not isinstance(row["generated_continuation_tokens"], bool)
            and row["generated_continuation_tokens"] >= 0,
            "generated continuation token count invalid",
        )
        require(
            row["generated_continuation_sha256"]
            == sha256_text(str(row["generated_continuation"])),
            "generated-continuation hash mismatch",
        )
        require(
            isinstance(row["generation_ended_with_eos"], bool),
            "generation EOS flag must be boolean",
        )
        require(
            isinstance(row["generation_reached_max_new_tokens"], bool),
            "generation max-token flag must be boolean",
        )
        require(
            not (
                row["generation_ended_with_eos"]
                and row["generation_reached_max_new_tokens"]
            ),
            "generation cannot both end with EOS and reach max_new_tokens",
        )
        reparsed = effective_expressed_parse(
            str(row["generated_continuation"]),
            generation_reached_max_new_tokens=row[
                "generation_reached_max_new_tokens"
            ],
        )
        require(
            row["expressed_parser_version"] == reparsed.parser_version,
            "expressed parser version drift",
        )
        require(
            row["expressed_parser_valid"] is reparsed.valid,
            "expressed parser validity drift",
        )
        require(
            row["expressed_parser_reason_code"] == reparsed.reason_code,
            "expressed parser reason drift",
        )
        require(row["generated_answer"] == reparsed.answer, "generated answer parse drift")
        require(
            row["expressed_confidence"] == reparsed.confidence,
            "expressed confidence parse drift",
        )
        expected_match = (
            reparsed.answer == row["token_prediction"]
            if reparsed.answer is not None
            else None
        )
        require(
            row["generated_answer_matches_token_prediction"] == expected_match,
            "generated/token answer agreement drift",
        )
    else:
        require(bool(row["inference_reason_code"]), "not_run prediction lacks reason")
        for name in (
            "yes_logit",
            "no_logit",
            "p_yes_binary",
            "p_no_binary",
            "binary_probability_sum",
            "p_yes_full_vocabulary",
            "p_no_full_vocabulary",
            "yes_no_full_vocabulary_mass",
            "token_confidence",
            "predictive_entropy",
        ):
            require(row[name] is None, f"not_run prediction has {name}")
        require(row["token_prediction"] is None, "not_run prediction has class")
        require(row["generated_continuation"] is None, "not_run prediction has continuation")
        require(
            row["generated_continuation_sha256"] is None,
            "not_run prediction has continuation hash",
        )
        require(
            row["generated_continuation_tokens"] is None,
            "not_run prediction has continuation token count",
        )
        require(
            row["generation_ended_with_eos"] is None,
            "not_run prediction has EOS flag",
        )
        require(
            row["generation_reached_max_new_tokens"] is None,
            "not_run prediction has max-token flag",
        )
        require(row["expressed_parser_version"] == PARSER_VERSION, "parser version drift")
        require(row["expressed_parser_valid"] is False, "not_run parser flag must be false")
        require(
            row["expressed_parser_reason_code"] == "not_run",
            "not_run parser reason drift",
        )
        require(row["generated_answer"] is None, "not_run row exposes generated answer")
        require(row["expressed_confidence"] is None, "not_run row exposes confidence")
        require(
            row["generated_answer_matches_token_prediction"] is None,
            "not_run row contains agreement",
        )


def validate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), "prediction rows must not be empty")
    for row in rows:
        validate_prediction_row(row)
    prediction_ids = [str(row["prediction_id"]) for row in rows]
    transformation_ids = [str(row["transformation_id"]) for row in rows]
    require(len(prediction_ids) == len(set(prediction_ids)), "duplicate prediction ID")
    require(
        len(transformation_ids) == len(set(transformation_ids)),
        "duplicate transformation ID",
    )
    observed_order = [
        (
            str(row["input_id"]),
            int(row["condition_index"]),
            str(row["transformation_id"]),
        )
        for row in rows
    ]
    require(
        observed_order == sorted(observed_order),
        "prediction row order is not deterministic",
    )


def prediction_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    validate_prediction_rows(rows)
    return canonical_jsonl_bytes(rows)


def summarize_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_prediction_rows(rows)
    status_counts = Counter(str(row["inference_status"]) for row in rows)
    parser_counts = Counter(
        "valid"
        if row["expressed_parser_valid"]
        else str(row["expressed_parser_reason_code"])
        for row in rows
    )
    condition_counts = Counter(str(row["condition"]) for row in rows)
    token_counts = Counter(
        str(row["token_prediction"])
        for row in rows
        if row["token_prediction"] is not None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "rows": len(rows),
        "condition_counts": dict(sorted(condition_counts.items())),
        "inference_status_counts": dict(sorted(status_counts.items())),
        "expressed_parser_counts": dict(sorted(parser_counts.items())),
        "token_prediction_counts": dict(sorted(token_counts.items())),
        "contains_ground_truth": False,
        "contains_correctness": False,
    }


__all__ = [
    "BinaryConfidenceBatch",
    "CLASS_CONTINUATIONS",
    "CLASS_ORDER",
    "DEFAULT_INFERENCE_SEED",
    "EXPECTED_CLASS_TOKEN_IDS",
    "ExpressedConfidenceParse",
    "FORBIDDEN_PREDICTION_KEYS",
    "InferenceContractError",
    "InferenceDataset",
    "InferenceExample",
    "METHOD",
    "PARSER_VERSION",
    "PREDICTION_FIELDS",
    "PROTOCOL_VERSION",
    "PromptCollator",
    "binary_confidence_from_logits",
    "canonical_bytes",
    "canonical_json",
    "canonical_jsonl_bytes",
    "contextual_class_token_ids",
    "effective_expressed_parse",
    "gather_last_token_logits",
    "last_nonpadding_indices",
    "make_prediction_row",
    "parse_expressed_continuation",
    "prediction_jsonl_bytes",
    "position_ids_from_attention_mask",
    "sha256_bytes",
    "sha256_text",
    "summarize_prediction_rows",
    "trim_generated_token_ids",
    "validate_prediction_row",
    "validate_prediction_rows",
]
