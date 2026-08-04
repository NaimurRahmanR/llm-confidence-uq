from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import re
import unicodedata
from typing import Any, Mapping, Protocol, Sequence


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-evidence-degradation-v1"
CONDITIONS = (
    "original",
    "lexical_evidence_removal",
    "prefix_truncation_50",
    "irrelevant_distractor",
    "lexical_contradiction",
    "no_passage",
)
TOKEN_PATTERN_TEXT = r"[^\W_]+(?:['’][^\W_]+)*"
TOKEN_PATTERN = re.compile(TOKEN_PATTERN_TEXT, re.UNICODE)
BOUNDARY_PATTERN_TEXT = r"""[.!?](?:["')\]]*)?(?=\s|$)|\n+"""
BOUNDARY_PATTERN = re.compile(BOUNDARY_PATTERN_TEXT)
NORMALIZATION_VERSION = "unicode-nfkc-casefold-unique-v1"
LEXICAL_ALGORITHM = "unique-token-jaccard-v1"
SENTENCE_SPLITTER = "terminal-punctuation-or-newline-spans-v1"
DISTRACTOR_SELECTOR = "minimum-jaccard-then-seeded-sha256-v1"
CONTRADICTION_TEMPLATE = "However, it is not true that {fragment}"
EXAMPLE_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]+")


class TokenizerLike(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str: ...


@dataclass(frozen=True)
class TransformationExample:
    example_id: str
    question: str
    passage: str

    def __post_init__(self) -> None:
        _validate_example_id(self.example_id)
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(self.passage, str) or not self.passage.strip():
            raise ValueError("passage must be a non-empty string")


@dataclass(frozen=True)
class DonorText:
    example_id: str
    passage: str

    def __post_init__(self) -> None:
        _validate_example_id(self.example_id)
        if not isinstance(self.passage, str) or not self.passage.strip():
            raise ValueError("donor passage must be a non-empty string")


@dataclass(frozen=True)
class SentenceSpan:
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class SentenceSelection:
    span: SentenceSpan
    overlap_numerator: int
    overlap_denominator: int
    sentence_count: int


@dataclass(frozen=True)
class ExactPrefix:
    text: str
    source_tokens: int
    requested_tokens: int
    actual_tokens: int
    boundary_adjustment_tokens: int


@dataclass(frozen=True)
class TransformationResult:
    example_id: str
    condition: str
    status: str
    reason_code: str | None
    quality_flags: tuple[str, ...]
    original_passage_sha256: str
    transformed_passage: str | None
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_example_id(self.example_id)
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {self.condition}")
        if self.status not in {"ok", "not_applicable", "failed"}:
            raise ValueError(f"unknown status: {self.status}")
        if tuple(sorted(set(self.quality_flags))) != self.quality_flags:
            raise ValueError("quality_flags must be unique and sorted")
        if self.status == "ok":
            if self.reason_code is not None or self.transformed_passage is None:
                raise ValueError("ok result has inconsistent fields")
        elif self.reason_code is None or self.transformed_passage is not None:
            raise ValueError("non-ok result has inconsistent fields")
        _validate_safe_parameters(self.parameters)

    def manifest_metadata(self) -> dict[str, Any]:
        transformed_sha256 = (
            sha256_text(self.transformed_passage)
            if self.transformed_passage is not None
            else None
        )
        changed_from_original = (
            transformed_sha256 != self.original_passage_sha256
            if transformed_sha256 is not None
            else None
        )
        identity_payload = {
            "condition": self.condition,
            "example_id": self.example_id,
            "original_passage_sha256": self.original_passage_sha256,
            "parameters": dict(self.parameters),
            "protocol_version": PROTOCOL_VERSION,
            "quality_flags": list(self.quality_flags),
            "reason_code": self.reason_code,
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "transformed_passage_sha256": transformed_sha256,
            "changed_from_original": changed_from_original,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "transformation_id": "boolqdeg-" + sha256_text(canonical_json(identity_payload)),
            "example_id": self.example_id,
            "condition": self.condition,
            "status": self.status,
            "reason_code": self.reason_code,
            "quality_flags": list(self.quality_flags),
            "original_passage_sha256": self.original_passage_sha256,
            "transformed_passage_sha256": transformed_sha256,
            "changed_from_original": changed_from_original,
            "parameters": dict(self.parameters),
        }


def _validate_example_id(example_id: str) -> None:
    if not isinstance(example_id, str) or EXAMPLE_ID_PATTERN.fullmatch(example_id) is None:
        raise ValueError("example_id must match [A-Za-z0-9._:-]+")


def _validate_safe_parameters(parameters: Mapping[str, Any]) -> None:
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    for key, value in parameters.items():
        if not isinstance(key, str) or not key:
            raise TypeError("parameter keys must be non-empty strings")
        if value is not None and not isinstance(value, (bool, int, str)):
            raise TypeError(f"unsafe parameter value for {key}")
        if key.endswith("sha256") and value is not None:
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"invalid SHA-256 parameter: {key}")
        if key.endswith("example_id") and value is not None:
            _validate_example_id(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def normalized_tokens(text: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return frozenset(match.group(0) for match in TOKEN_PATTERN.finditer(normalized))


def jaccard_overlap(left: str, right: str) -> tuple[int, int]:
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    union = left_tokens | right_tokens
    if not union:
        return 0, 1
    return len(left_tokens & right_tokens), len(union)


def sentence_spans(passage: str) -> tuple[SentenceSpan, ...]:
    spans: list[SentenceSpan] = []
    segment_start = 0

    def append_span(raw_start: int, raw_end: int) -> None:
        start, end = raw_start, raw_end
        while start < end and passage[start].isspace():
            start += 1
        while end > start and passage[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append(SentenceSpan(len(spans), start, end, passage[start:end]))

    for match in BOUNDARY_PATTERN.finditer(passage):
        token = match.group(0)
        if token.startswith("\n"):
            append_span(segment_start, match.start())
            segment_start = match.end()
        else:
            append_span(segment_start, match.end())
            segment_start = match.end()
            while segment_start < len(passage) and passage[segment_start].isspace():
                segment_start += 1
    append_span(segment_start, len(passage))
    return tuple(spans)


def select_lexical_sentence(question: str, passage: str) -> SentenceSelection | None:
    if not normalized_tokens(question):
        return None
    spans = sentence_spans(passage)
    if not spans:
        return None
    best_span = spans[0]
    best_pair = jaccard_overlap(question, best_span.text)
    best_score = Fraction(*best_pair)
    for span in spans[1:]:
        pair = jaccard_overlap(question, span.text)
        score = Fraction(*pair)
        if score > best_score:
            best_span, best_pair, best_score = span, pair, score
    return SentenceSelection(
        span=best_span,
        overlap_numerator=best_pair[0],
        overlap_denominator=best_pair[1],
        sentence_count=len(spans),
    )


def encode_text(tokenizer: TokenizerLike, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoded["input_ids"]
    if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
        raise TypeError("tokenizer must return a flat list of integer input_ids")
    return token_ids


def decode_tokens(tokenizer: TokenizerLike, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def exact_decodable_prefix(
    tokenizer: TokenizerLike,
    text: str,
    requested_tokens: int,
) -> ExactPrefix | None:
    if requested_tokens <= 0:
        raise ValueError("requested_tokens must be positive")
    source_ids = encode_text(tokenizer, text)
    target = min(len(source_ids), requested_tokens)
    for actual in range(target, 0, -1):
        fragment = decode_tokens(tokenizer, source_ids[:actual])
        if fragment and encode_text(tokenizer, fragment) == source_ids[:actual]:
            return ExactPrefix(
                text=fragment,
                source_tokens=len(source_ids),
                requested_tokens=requested_tokens,
                actual_tokens=actual,
                boundary_adjustment_tokens=target - actual,
            )
    return None


def _result(
    example: TransformationExample,
    condition: str,
    *,
    status: str,
    transformed_passage: str | None,
    reason_code: str | None = None,
    quality_flags: Sequence[str] = (),
    parameters: Mapping[str, Any],
) -> TransformationResult:
    return TransformationResult(
        example_id=example.example_id,
        condition=condition,
        status=status,
        reason_code=reason_code,
        quality_flags=tuple(sorted(set(quality_flags))),
        original_passage_sha256=sha256_text(example.passage),
        transformed_passage=transformed_passage,
        parameters=dict(parameters),
    )


def original(example: TransformationExample) -> TransformationResult:
    return _result(
        example, "original", status="ok", transformed_passage=example.passage,
        parameters={"algorithm": "identity-v1"},
    )


def lexical_evidence_removal(example: TransformationExample) -> TransformationResult:
    selected = select_lexical_sentence(example.question, example.passage)
    if selected is None:
        reason = (
            "no_question_lexical_tokens"
            if not normalized_tokens(example.question)
            else "no_sentence"
        )
        return _result(
            example, "lexical_evidence_removal", status="not_applicable",
            transformed_passage=None, reason_code=reason,
            parameters={"algorithm": LEXICAL_ALGORITHM},
        )
    span = selected.span
    left = example.passage[: span.start].rstrip()
    right = example.passage[span.end :].lstrip()
    transformed = (left + " " + right) if left and right else (left + right)
    flags = []
    if selected.overlap_numerator == 0:
        flags.append("zero_lexical_overlap")
    if not transformed:
        flags.append("whole_passage_removed")
    return _result(
        example, "lexical_evidence_removal", status="ok",
        transformed_passage=transformed, quality_flags=flags,
        parameters={
            "algorithm": LEXICAL_ALGORITHM,
            "normalization": NORMALIZATION_VERSION,
            "sentence_splitter": SENTENCE_SPLITTER,
            "sentence_count": selected.sentence_count,
            "selected_sentence_index": span.index,
            "selected_sentence_start": span.start,
            "selected_sentence_end": span.end,
            "selected_sentence_sha256": sha256_text(span.text),
            "overlap_numerator": selected.overlap_numerator,
            "overlap_denominator": selected.overlap_denominator,
            "tie_break": "earliest-character-span-v1",
        },
    )


def prefix_truncation_50(
    example: TransformationExample,
    tokenizer: TokenizerLike,
) -> TransformationResult:
    source_ids = encode_text(tokenizer, example.passage)
    if not source_ids:
        return _result(
            example, "prefix_truncation_50", status="not_applicable",
            transformed_passage=None, reason_code="empty_passage_tokenization",
            parameters={"algorithm": "token-prefix-half-ceiling-v1", "original_tokens": 0},
        )
    target = (len(source_ids) + 1) // 2
    prefix = exact_decodable_prefix(tokenizer, example.passage, target)
    if prefix is None:
        return _result(
            example, "prefix_truncation_50", status="not_applicable",
            transformed_passage=None, reason_code="no_exact_decodable_prefix",
            parameters={
                "algorithm": "token-prefix-half-ceiling-v1",
                "original_tokens": len(source_ids),
                "target_tokens": target,
            },
        )
    if prefix.text == example.passage:
        return _result(
            example, "prefix_truncation_50", status="not_applicable",
            transformed_passage=None, reason_code="unchanged_result",
            parameters={
                "algorithm": "token-prefix-half-ceiling-v1",
                "original_tokens": len(source_ids),
                "target_tokens": target,
                "retained_tokens": prefix.actual_tokens,
            },
        )
    flags = (
        ("token_boundary_adjusted",)
        if prefix.boundary_adjustment_tokens
        else ()
    )
    return _result(
        example, "prefix_truncation_50", status="ok",
        transformed_passage=prefix.text, quality_flags=flags,
        parameters={
            "algorithm": "token-prefix-half-ceiling-v1",
            "original_tokens": len(source_ids),
            "target_tokens": target,
            "retained_tokens": prefix.actual_tokens,
            "boundary_adjustment_tokens": prefix.boundary_adjustment_tokens,
            "ratio_numerator": 1,
            "ratio_denominator": 2,
            "rounding": "ceiling",
        },
    )


def _validate_donors(donors: Sequence[DonorText]) -> None:
    donor_ids = [donor.example_id for donor in donors]
    if len(donor_ids) != len(set(donor_ids)):
        raise ValueError("duplicate donor example_id")


def irrelevant_distractor(
    example: TransformationExample,
    donors: Sequence[DonorText],
    tokenizer: TokenizerLike,
    *,
    seed: int,
    maximum_fragment_tokens: int = 64,
) -> TransformationResult:
    _validate_donors(donors)
    original_sha256 = sha256_text(example.passage)
    ranked: list[tuple[Fraction, str, str, DonorText, ExactPrefix, int, int]] = []
    for donor in donors:
        donor_sha256 = sha256_text(donor.passage)
        if donor.example_id == example.example_id or donor_sha256 == original_sha256:
            continue
        prefix = exact_decodable_prefix(tokenizer, donor.passage, maximum_fragment_tokens)
        if prefix is None:
            continue
        numerator, denominator = jaccard_overlap(example.question, prefix.text)
        tie_hash = sha256_text(
            canonical_json(
                {
                    "candidate_example_id": donor.example_id,
                    "condition_version": PROTOCOL_VERSION,
                    "seed": seed,
                    "target_example_id": example.example_id,
                }
            )
        )
        ranked.append(
            (
                Fraction(numerator, denominator), tie_hash, donor.example_id,
                donor, prefix, numerator, denominator,
            )
        )
    if not ranked:
        return _result(
            example, "irrelevant_distractor", status="not_applicable",
            transformed_passage=None, reason_code="no_eligible_donor",
            parameters={
                "algorithm": DISTRACTOR_SELECTOR,
                "donor_pool_size": len(donors),
                "maximum_fragment_tokens": maximum_fragment_tokens,
                "seed": seed,
            },
        )
    _, tie_hash, _, donor, prefix, numerator, denominator = min(ranked)
    flags = []
    if numerator == 0:
        flags.append("zero_lexical_overlap")
    if prefix.actual_tokens < maximum_fragment_tokens:
        flags.append("fragment_shorter_than_requested")
    if prefix.boundary_adjustment_tokens:
        flags.append("token_boundary_adjusted")
    transformed = example.passage.rstrip() + "\n\n" + prefix.text
    return _result(
        example, "irrelevant_distractor", status="ok",
        transformed_passage=transformed, quality_flags=flags,
        parameters={
            "algorithm": DISTRACTOR_SELECTOR,
            "eligible_donor_count": len(ranked),
            "distractor_example_id": donor.example_id,
            "distractor_original_passage_sha256": sha256_text(donor.passage),
            "fragment_sha256": sha256_text(prefix.text),
            "fragment_source_tokens": prefix.source_tokens,
            "requested_fragment_tokens": maximum_fragment_tokens,
            "actual_fragment_tokens": prefix.actual_tokens,
            "boundary_adjustment_tokens": prefix.boundary_adjustment_tokens,
            "overlap_numerator": numerator,
            "overlap_denominator": denominator,
            "seed": seed,
            "tie_break_sha256": tie_hash,
        },
    )


def lexical_contradiction(
    example: TransformationExample,
    tokenizer: TokenizerLike,
    *,
    maximum_fragment_tokens: int = 64,
) -> TransformationResult:
    selected = select_lexical_sentence(example.question, example.passage)
    if selected is None:
        reason = (
            "no_question_lexical_tokens"
            if not normalized_tokens(example.question)
            else "no_sentence"
        )
        return _result(
            example, "lexical_contradiction", status="not_applicable",
            transformed_passage=None, reason_code=reason,
            parameters={"algorithm": LEXICAL_ALGORITHM},
        )
    prefix = exact_decodable_prefix(
        tokenizer, selected.span.text, maximum_fragment_tokens
    )
    if prefix is None:
        return _result(
            example, "lexical_contradiction", status="not_applicable",
            transformed_passage=None, reason_code="empty_selected_sentence_tokenization",
            parameters={
                "algorithm": LEXICAL_ALGORITHM,
                "selected_sentence_index": selected.span.index,
                "maximum_fragment_tokens": maximum_fragment_tokens,
            },
        )
    flags = []
    if selected.overlap_numerator == 0:
        flags.append("zero_lexical_overlap")
    if prefix.actual_tokens < maximum_fragment_tokens:
        flags.append("fragment_shorter_than_requested")
    if prefix.boundary_adjustment_tokens:
        flags.append("token_boundary_adjusted")
    contradiction = CONTRADICTION_TEMPLATE.format(fragment=prefix.text)
    transformed = example.passage.rstrip() + "\n\n" + contradiction
    return _result(
        example, "lexical_contradiction", status="ok",
        transformed_passage=transformed, quality_flags=flags,
        parameters={
            "algorithm": LEXICAL_ALGORITHM,
            "normalization": NORMALIZATION_VERSION,
            "sentence_splitter": SENTENCE_SPLITTER,
            "selected_sentence_index": selected.span.index,
            "selected_sentence_start": selected.span.start,
            "selected_sentence_end": selected.span.end,
            "selected_sentence_sha256": sha256_text(selected.span.text),
            "fragment_sha256": sha256_text(prefix.text),
            "fragment_source_tokens": prefix.source_tokens,
            "requested_fragment_tokens": maximum_fragment_tokens,
            "actual_fragment_tokens": prefix.actual_tokens,
            "boundary_adjustment_tokens": prefix.boundary_adjustment_tokens,
            "overlap_numerator": selected.overlap_numerator,
            "overlap_denominator": selected.overlap_denominator,
            "template_sha256": sha256_text(CONTRADICTION_TEMPLATE),
            "tie_break": "earliest-character-span-v1",
        },
    )


def no_passage(example: TransformationExample) -> TransformationResult:
    return _result(
        example, "no_passage", status="ok", transformed_passage="",
        parameters={"algorithm": "empty-passage-v1"},
    )


def build_all_conditions(
    example: TransformationExample,
    donors: Sequence[DonorText],
    tokenizer: TokenizerLike,
    *,
    seed: int,
    maximum_fragment_tokens: int = 64,
) -> list[TransformationResult]:
    results = [
        original(example),
        lexical_evidence_removal(example),
        prefix_truncation_50(example, tokenizer),
        irrelevant_distractor(
            example, donors, tokenizer, seed=seed,
            maximum_fragment_tokens=maximum_fragment_tokens,
        ),
        lexical_contradiction(
            example, tokenizer, maximum_fragment_tokens=maximum_fragment_tokens,
        ),
        no_passage(example),
    ]
    if tuple(result.condition for result in results) != CONDITIONS:
        raise RuntimeError("condition order changed")
    return results


def metadata_jsonl(results: Sequence[TransformationResult]) -> str:
    if not results:
        return ""
    return "".join(canonical_json(result.manifest_metadata()) + "\n" for result in results)
