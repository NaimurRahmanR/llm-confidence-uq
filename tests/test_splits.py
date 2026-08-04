"""Unit tests for deterministic split construction."""

from __future__ import annotations

from copy import deepcopy
import unittest

from llm_confidence_uq.data import (
    EligibleRecord,
    build_research_manifest,
    largest_remainder_allocation,
    manifest_jsonl,
    source_split_sha256,
    validate_manifest_rows,
)

DATASET_REVISION = "a" * 40


def make_records(
    source_split: str,
    *,
    false_count: int,
    true_count: int,
) -> list[EligibleRecord]:
    records: list[EligibleRecord] = []
    labels = [False] * false_count + [True] * true_count

    for source_index, answer in enumerate(labels):
        records.append(
            EligibleRecord(
                dataset_id="google/boolq",
                dataset_revision=DATASET_REVISION,
                source_split=source_split,
                source_index=source_index,
                question=f"question {source_split} {source_index}",
                passage=f"passage {source_split} {source_index}",
                answer=answer,
                prompt_tokens=100 + source_index % 20,
            )
        )
    return records


class AllocationTests(unittest.TestCase):
    def test_verified_source_proportions(self) -> None:
        train_available = {False: 3553, True: 5874}
        validation_available = {False: 1237, True: 2033}

        self.assertEqual(
            largest_remainder_allocation(train_available, 800),
            {False: 302, True: 498},
        )
        self.assertEqual(
            largest_remainder_allocation(train_available, 200),
            {False: 75, True: 125},
        )
        self.assertEqual(
            largest_remainder_allocation(
                validation_available,
                400,
            ),
            {False: 151, True: 249},
        )

    def test_allocation_rejects_oversized_target(self) -> None:
        with self.assertRaises(ValueError):
            largest_remainder_allocation(
                {False: 2, True: 3},
                6,
            )


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.official_train = make_records(
            "train",
            false_count=60,
            true_count=90,
        )
        self.official_validation = make_records(
            "validation",
            false_count=40,
            true_count=60,
        )
        self.arguments = {
            "train_size": 50,
            "calibration_size": 20,
            "test_size": 30,
            "selection_seed": 20260803,
            "eligibility_max_prompt_tokens": 768,
        }

    def test_manifest_is_deterministic_and_disjoint(self) -> None:
        rows_one, summary_one = build_research_manifest(
            self.official_train,
            self.official_validation,
            **self.arguments,
        )
        rows_two, summary_two = build_research_manifest(
            self.official_train,
            self.official_validation,
            **self.arguments,
        )

        self.assertEqual(rows_one, rows_two)
        self.assertEqual(summary_one, summary_two)

        ids_by_split = {
            split: {
                row["example_id"]
                for row in rows_one
                if row["research_split"] == split
            }
            for split in ("train", "calibration", "test")
        }
        self.assertTrue(
            ids_by_split["train"].isdisjoint(
                ids_by_split["calibration"]
            )
        )
        self.assertTrue(
            ids_by_split["train"].isdisjoint(
                ids_by_split["test"]
            )
        )
        self.assertTrue(
            ids_by_split["calibration"].isdisjoint(
                ids_by_split["test"]
            )
        )

        self.assertEqual(
            summary_one["selected_class_counts"],
            {
                "train": {"false": 20, "true": 30},
                "calibration": {"false": 8, "true": 12},
                "test": {"false": 12, "true": 18},
            },
        )

    def test_manifest_changes_when_seed_changes(self) -> None:
        rows_one, _ = build_research_manifest(
            self.official_train,
            self.official_validation,
            **self.arguments,
        )
        changed_arguments = {
            **self.arguments,
            "selection_seed": 20260804,
        }
        rows_two, _ = build_research_manifest(
            self.official_train,
            self.official_validation,
            **changed_arguments,
        )
        self.assertNotEqual(rows_one, rows_two)

    def test_manifest_excludes_raw_text(self) -> None:
        rows, _ = build_research_manifest(
            self.official_train,
            self.official_validation,
            **self.arguments,
        )
        for row in rows:
            self.assertNotIn("question", row)
            self.assertNotIn("passage", row)

    def test_duplicate_row_is_rejected(self) -> None:
        rows, _ = build_research_manifest(
            self.official_train,
            self.official_validation,
            **self.arguments,
        )
        corrupted = deepcopy(rows)
        corrupted[-1] = deepcopy(corrupted[0])

        with self.assertRaises(ValueError):
            validate_manifest_rows(
                corrupted,
                expected_sizes={
                    "train": 50,
                    "calibration": 20,
                    "test": 30,
                },
                selection_seed=20260803,
                eligibility_max_prompt_tokens=768,
            )

    def test_source_hash_requires_contiguous_indices(self) -> None:
        digest_one = source_split_sha256(self.official_train)
        digest_two = source_split_sha256(
            list(reversed(self.official_train))
        )
        self.assertEqual(digest_one, digest_two)
        self.assertEqual(len(digest_one), 64)

        with self.assertRaises(ValueError):
            source_split_sha256(self.official_train[1:])

    def test_jsonl_is_stable_and_newline_terminated(self) -> None:
        rows, _ = build_research_manifest(
            self.official_train,
            self.official_validation,
            **self.arguments,
        )
        first = manifest_jsonl(rows)
        second = manifest_jsonl(rows)

        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))
        self.assertEqual(first.count("\n"), len(rows))


if __name__ == "__main__":
    unittest.main()
