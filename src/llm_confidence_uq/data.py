"""Deterministic BoolQ record identity and split selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Mapping, Sequence

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SOURCE_SPLITS = frozenset({"train", "validation"})
RESEARCH_SPLITS = ("train", "calibration", "test")


def canonical_json(value: Any) -> str:
    """Serialize a value with stable JSON ordering and no extra spaces."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(text: str) -> str:
    """Return a lowercase SHA-256 hexadecimal digest."""
    return sha256(text.encode("utf-8")).hexdigest()


def canonical_source_record(
    question: str,
    passage: str,
    answer: bool,
) -> str:
    """Serialize the unmodified source fields used for provenance."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(passage, str) or not passage.strip():
        raise ValueError("passage must be a non-empty string")
    if type(answer) is not bool:
        raise TypeError("answer must be bool")

    return canonical_json(
        {
            "question": question,
            "passage": passage,
            "answer": answer,
        }
    )


@dataclass(frozen=True, slots=True)
class EligibleRecord:
    """A source record that fits the original-prompt token budget."""

    dataset_id: str
    dataset_revision: str
    source_split: str
    source_index: int
    question: str
    passage: str
    answer: bool
    prompt_tokens: int

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if COMMIT_PATTERN.fullmatch(self.dataset_revision) is None:
            raise ValueError("dataset_revision must be a 40-character SHA")
        if self.source_split not in SOURCE_SPLITS:
            raise ValueError(
                f"unsupported source split: {self.source_split!r}"
            )
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if not self.question.strip():
            raise ValueError("question must be non-empty")
        if not self.passage.strip():
            raise ValueError("passage must be non-empty")
        if type(self.answer) is not bool:
            raise TypeError("answer must be bool")
        if self.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")

    @property
    def canonical_record(self) -> str:
        return canonical_source_record(
            self.question,
            self.passage,
            self.answer,
        )

    @property
    def record_sha256(self) -> str:
        return sha256_text(self.canonical_record)

    @property
    def input_sha256(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "question": self.question,
                    "passage": self.passage,
                }
            )
        )

    @property
    def original_passage_sha256(self) -> str:
        return sha256_text(self.passage)

    @property
    def example_id(self) -> str:
        dataset_slug = re.sub(
            r"[^a-z0-9]+",
            "-",
            self.dataset_id.rsplit("/", maxsplit=1)[-1].lower(),
        ).strip("-")
        return (
            f"{dataset_slug}-{self.source_split}-"
            f"{self.source_index:05d}-{self.record_sha256[:12]}"
        )


@dataclass(frozen=True, slots=True)
class SelectedRecord:
    """An eligible source record assigned to a research split."""

    record: EligibleRecord
    research_split: str
    selection_rank_sha256: str

    def __post_init__(self) -> None:
        if self.research_split not in RESEARCH_SPLITS:
            raise ValueError(
                f"invalid research split: {self.research_split!r}"
            )
        if (
            SHA256_PATTERN.fullmatch(self.selection_rank_sha256)
            is None
        ):
            raise ValueError("selection rank must be a SHA-256 digest")

    def to_manifest_row(
        self,
        *,
        selection_seed: int,
        eligibility_max_prompt_tokens: int,
    ) -> dict[str, Any]:
        """Return a manifest row without redistributing source text."""
        source = self.record
        return {
            "schema_version": 1,
            "dataset_id": source.dataset_id,
            "dataset_revision": source.dataset_revision,
            "example_id": source.example_id,
            "source_split": source.source_split,
            "source_index": source.source_index,
            "research_split": self.research_split,
            "answer": source.answer,
            "prompt_tokens": source.prompt_tokens,
            "eligibility_max_prompt_tokens":
                eligibility_max_prompt_tokens,
            "selection_seed": selection_seed,
            "selection_rank_sha256":
                self.selection_rank_sha256,
            "record_sha256": source.record_sha256,
            "input_sha256": source.input_sha256,
            "original_passage_sha256":
                source.original_passage_sha256,
        }


def source_split_sha256(
    records: Sequence[EligibleRecord],
) -> str:
    """Hash a complete source split in original index order."""
    if not records:
        raise ValueError("records must not be empty")

    source_splits = {record.source_split for record in records}
    if len(source_splits) != 1:
        raise ValueError("records must come from one source split")

    ordered = sorted(records, key=lambda record: record.source_index)
    observed_indices = [record.source_index for record in ordered]
    expected_indices = list(range(len(ordered)))

    if observed_indices != expected_indices:
        raise ValueError(
            "source indices must be unique and contiguous from zero"
        )

    digest = sha256()
    for record in ordered:
        digest.update(record.canonical_record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def class_counts(
    records: Sequence[EligibleRecord],
) -> dict[bool, int]:
    """Count False and True labels explicitly."""
    counts = Counter(record.answer for record in records)
    return {False: counts[False], True: counts[True]}


def largest_remainder_allocation(
    available_by_label: Mapping[bool, int],
    target_size: int,
) -> dict[bool, int]:
    """Allocate an exact target while preserving label proportions."""
    if set(available_by_label) != {False, True}:
        raise ValueError("counts must contain exactly False and True")
    if target_size < 0:
        raise ValueError("target_size must be non-negative")

    available = {
        label: int(available_by_label[label])
        for label in (False, True)
    }
    if any(count < 0 for count in available.values()):
        raise ValueError("available counts must be non-negative")

    total_available = sum(available.values())
    if total_available == 0:
        if target_size == 0:
            return {False: 0, True: 0}
        raise ValueError("cannot allocate from an empty population")
    if target_size > total_available:
        raise ValueError(
            f"target {target_size} exceeds population "
            f"{total_available}"
        )

    numerators = {
        label: target_size * available[label]
        for label in (False, True)
    }
    allocation = {
        label: numerators[label] // total_available
        for label in (False, True)
    }
    remainders = {
        label: numerators[label] % total_available
        for label in (False, True)
    }

    remaining = target_size - sum(allocation.values())
    order = sorted(
        (False, True),
        key=lambda label: (-remainders[label], int(label)),
    )

    for label in order[:remaining]:
        allocation[label] += 1

    if sum(allocation.values()) != target_size:
        raise AssertionError("allocation does not match target size")
    if any(
        allocation[label] > available[label]
        for label in (False, True)
    ):
        raise AssertionError("allocation exceeds available examples")

    return allocation


def selection_rank(
    record: EligibleRecord,
    *,
    seed: int,
    salt: str,
) -> str:
    """Create a deterministic pseudo-random rank for one record."""
    if not isinstance(seed, int):
        raise TypeError("seed must be int")
    if not salt:
        raise ValueError("salt must be non-empty")

    return sha256_text(
        f"sha256-stratified-v1|{seed}|{salt}|{record.example_id}"
    )


def stratified_hash_select(
    records: Sequence[EligibleRecord],
    *,
    quotas: Mapping[bool, int],
    skip_by_label: Mapping[bool, int] | None,
    seed: int,
    salt: str,
    research_split: str,
) -> list[SelectedRecord]:
    """Select deterministic per-label slices from hash-ranked rows."""
    if research_split not in RESEARCH_SPLITS:
        raise ValueError("unsupported research split")
    if set(quotas) != {False, True}:
        raise ValueError("quotas must contain False and True")

    skips = (
        {False: 0, True: 0}
        if skip_by_label is None
        else {
            False: int(skip_by_label[False]),
            True: int(skip_by_label[True]),
        }
    )

    selected: list[SelectedRecord] = []

    for label in (False, True):
        quota = int(quotas[label])
        skip = skips[label]
        if quota < 0 or skip < 0:
            raise ValueError("quota and skip must be non-negative")

        ranked = sorted(
            (
                (
                    selection_rank(
                        record,
                        seed=seed,
                        salt=salt,
                    ),
                    record.example_id,
                    record,
                )
                for record in records
                if record.answer is label
            ),
            key=lambda item: (item[0], item[1]),
        )

        stop = skip + quota
        if stop > len(ranked):
            raise ValueError(
                f"label {label}: requested through index {stop}, "
                f"but only {len(ranked)} records exist"
            )

        for rank, _, record in ranked[skip:stop]:
            selected.append(
                SelectedRecord(
                    record=record,
                    research_split=research_split,
                    selection_rank_sha256=rank,
                )
            )

    return sorted(
        selected,
        key=lambda item: (
            item.selection_rank_sha256,
            item.record.example_id,
        ),
    )


def _stringify_counts(
    counts: Mapping[bool, int],
) -> dict[str, int]:
    return {
        "false": int(counts[False]),
        "true": int(counts[True]),
    }


def build_research_manifest(
    official_train_records: Sequence[EligibleRecord],
    official_validation_records: Sequence[EligibleRecord],
    *,
    train_size: int,
    calibration_size: int,
    test_size: int,
    selection_seed: int,
    eligibility_max_prompt_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic train, calibration, and test manifests."""
    if not official_train_records:
        raise ValueError("official_train_records must not be empty")
    if not official_validation_records:
        raise ValueError(
            "official_validation_records must not be empty"
        )
    if any(
        record.source_split != "train"
        for record in official_train_records
    ):
        raise ValueError("training records must use source split train")
    if any(
        record.source_split != "validation"
        for record in official_validation_records
    ):
        raise ValueError(
            "test records must use source split validation"
        )

    all_records = [
        *official_train_records,
        *official_validation_records,
    ]

    if any(
        record.prompt_tokens > eligibility_max_prompt_tokens
        for record in all_records
    ):
        raise ValueError(
            "ineligible record passed to manifest construction"
        )

    dataset_ids = {record.dataset_id for record in all_records}
    revisions = {
        record.dataset_revision for record in all_records
    }
    if len(dataset_ids) != 1 or len(revisions) != 1:
        raise ValueError(
            "all records must share one dataset and revision"
        )

    coordinates = [
        (record.source_split, record.source_index)
        for record in all_records
    ]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("duplicate source coordinates detected")

    input_hashes = [record.input_sha256 for record in all_records]
    if len(input_hashes) != len(set(input_hashes)):
        raise ValueError("duplicate input hashes detected")

    train_available = class_counts(official_train_records)
    validation_available = class_counts(
        official_validation_records
    )

    train_quotas = largest_remainder_allocation(
        train_available,
        train_size,
    )
    calibration_quotas = largest_remainder_allocation(
        train_available,
        calibration_size,
    )
    test_quotas = largest_remainder_allocation(
        validation_available,
        test_size,
    )

    for label in (False, True):
        if (
            train_quotas[label] + calibration_quotas[label]
            > train_available[label]
        ):
            raise ValueError(
                f"insufficient official-train label {label} rows"
            )

    selected_train = stratified_hash_select(
        official_train_records,
        quotas=train_quotas,
        skip_by_label=None,
        seed=selection_seed,
        salt="official-train-v1",
        research_split="train",
    )
    selected_calibration = stratified_hash_select(
        official_train_records,
        quotas=calibration_quotas,
        skip_by_label=train_quotas,
        seed=selection_seed,
        salt="official-train-v1",
        research_split="calibration",
    )
    selected_test = stratified_hash_select(
        official_validation_records,
        quotas=test_quotas,
        skip_by_label=None,
        seed=selection_seed,
        salt="official-validation-v1",
        research_split="test",
    )

    selected = [
        *selected_train,
        *selected_calibration,
        *selected_test,
    ]
    rows = [
        item.to_manifest_row(
            selection_seed=selection_seed,
            eligibility_max_prompt_tokens=(
                eligibility_max_prompt_tokens
            ),
        )
        for item in selected
    ]

    split_order = {
        name: index
        for index, name in enumerate(RESEARCH_SPLITS)
    }
    rows.sort(
        key=lambda row: (
            split_order[row["research_split"]],
            row["selection_rank_sha256"],
            row["example_id"],
        )
    )

    expected_sizes = {
        "train": train_size,
        "calibration": calibration_size,
        "test": test_size,
    }
    validate_manifest_rows(
        rows,
        expected_sizes=expected_sizes,
        selection_seed=selection_seed,
        eligibility_max_prompt_tokens=(
            eligibility_max_prompt_tokens
        ),
    )

    summary = {
        "schema_version": 1,
        "selection_algorithm": "sha256-stratified-v1",
        "selection_seed": selection_seed,
        "dataset_id": next(iter(dataset_ids)),
        "dataset_revision": next(iter(revisions)),
        "eligibility_max_prompt_tokens":
            eligibility_max_prompt_tokens,
        "source_eligible_class_counts": {
            "train": _stringify_counts(train_available),
            "validation":
                _stringify_counts(validation_available),
        },
        "selected_sizes": expected_sizes,
        "selected_class_counts": {
            "train": _stringify_counts(train_quotas),
            "calibration":
                _stringify_counts(calibration_quotas),
            "test": _stringify_counts(test_quotas),
        },
    }
    return rows, summary


def validate_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_sizes: Mapping[str, int],
    selection_seed: int,
    eligibility_max_prompt_tokens: int,
) -> None:
    """Validate manifest integrity and split-use boundaries."""
    required_fields = {
        "schema_version",
        "dataset_id",
        "dataset_revision",
        "example_id",
        "source_split",
        "source_index",
        "research_split",
        "answer",
        "prompt_tokens",
        "eligibility_max_prompt_tokens",
        "selection_seed",
        "selection_rank_sha256",
        "record_sha256",
        "input_sha256",
        "original_passage_sha256",
    }

    if set(expected_sizes) != set(RESEARCH_SPLITS):
        raise ValueError("expected_sizes must define all research splits")
    if len(rows) != sum(expected_sizes.values()):
        raise ValueError("manifest row count does not match expected sizes")

    ids: list[str] = []
    coordinates: list[tuple[str, int]] = []
    record_hashes: list[str] = []
    input_hashes: list[str] = []
    observed_splits: Counter[str] = Counter()
    dataset_ids: set[str] = set()
    revisions: set[str] = set()

    for row in rows:
        missing = required_fields - set(row)
        if missing:
            raise ValueError(
                f"manifest row is missing fields: {sorted(missing)}"
            )
        if {"question", "passage"} & set(row):
            raise ValueError("manifest must not contain raw source text")
        if row["schema_version"] != 1:
            raise ValueError("unsupported manifest schema")
        if row["research_split"] not in RESEARCH_SPLITS:
            raise ValueError("invalid research split")
        if row["source_split"] not in SOURCE_SPLITS:
            raise ValueError("invalid source split")
        if (
            row["research_split"] in {"train", "calibration"}
            and row["source_split"] != "train"
        ):
            raise ValueError(
                "train/calibration must derive from official train"
            )
        if (
            row["research_split"] == "test"
            and row["source_split"] != "validation"
        ):
            raise ValueError(
                "test must derive from official validation"
            )
        if type(row["answer"]) is not bool:
            raise TypeError("manifest answer must be bool")
        if row["selection_seed"] != selection_seed:
            raise ValueError("selection seed mismatch")
        if (
            row["eligibility_max_prompt_tokens"]
            != eligibility_max_prompt_tokens
        ):
            raise ValueError("eligibility ceiling mismatch")
        if not (
            1
            <= row["prompt_tokens"]
            <= eligibility_max_prompt_tokens
        ):
            raise ValueError("invalid prompt-token count")

        for hash_field in (
            "selection_rank_sha256",
            "record_sha256",
            "input_sha256",
            "original_passage_sha256",
        ):
            if (
                SHA256_PATTERN.fullmatch(row[hash_field])
                is None
            ):
                raise ValueError(f"invalid {hash_field}")

        if (
            COMMIT_PATTERN.fullmatch(row["dataset_revision"])
            is None
        ):
            raise ValueError("invalid dataset revision")
        if not row["example_id"].endswith(
            row["record_sha256"][:12]
        ):
            raise ValueError("example ID does not match record hash")

        ids.append(row["example_id"])
        coordinates.append(
            (row["source_split"], row["source_index"])
        )
        record_hashes.append(row["record_sha256"])
        input_hashes.append(row["input_sha256"])
        observed_splits[row["research_split"]] += 1
        dataset_ids.add(row["dataset_id"])
        revisions.add(row["dataset_revision"])

    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example IDs detected")
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("split leakage through source coordinates")
    if len(record_hashes) != len(set(record_hashes)):
        raise ValueError("duplicate source records detected")
    if len(input_hashes) != len(set(input_hashes)):
        raise ValueError("split leakage through duplicate inputs")
    if dict(observed_splits) != dict(expected_sizes):
        raise ValueError(
            f"unexpected split sizes: {dict(observed_splits)}"
        )
    if len(dataset_ids) != 1 or len(revisions) != 1:
        raise ValueError("manifest mixes datasets or revisions")


def manifest_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize manifest rows deterministically as JSON Lines."""
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def summary_json(summary: Mapping[str, Any]) -> str:
    """Serialize a deterministic human-readable summary."""
    return json.dumps(
        dict(summary),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
