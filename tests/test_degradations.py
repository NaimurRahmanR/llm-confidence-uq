from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_confidence_uq.degradations import (
    BOUNDARY_PATTERN_TEXT,
    CONDITIONS,
    CONTRADICTION_TEMPLATE,
    DISTRACTOR_SELECTOR,
    NORMALIZATION_VERSION,
    PROTOCOL_VERSION,
    TOKEN_PATTERN_TEXT,
    DonorText,
    TransformationExample,
    TransformationResult,
    build_all_conditions,
    canonical_json,
    irrelevant_distractor,
    jaccard_overlap,
    lexical_contradiction,
    lexical_evidence_removal,
    metadata_jsonl,
    no_passage,
    normalized_tokens,
    original,
    prefix_truncation_50,
    select_lexical_sentence,
    sentence_spans,
)


class WhitespaceTokenizer:
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        tokens = re.findall(r"\S+", text)
        ids = []
        for token in tokens:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id) + 1
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            ids.append(self.token_to_id[token])
        return {"input_ids": ids}

    def decode(self, token_ids: list[int], **_: object) -> str:
        return " ".join(self.id_to_token[token_id] for token_id in token_ids)


def example(example_id: str, question: str, passage: str) -> TransformationExample:
    return TransformationExample(example_id, question, passage)


class LexicalTests(unittest.TestCase):
    def test_unicode_nfkc_and_casefold_normalization(self) -> None:
        self.assertEqual(normalized_tokens("ＣＡＦÉ café"), frozenset({"café"}))
        self.assertEqual(jaccard_overlap("Red PLANET", "red moon"), (1, 3))

    def test_sentence_spans_preserve_offsets_and_newlines(self) -> None:
        passage = "First sentence.  Second line!\nThird?"
        spans = sentence_spans(passage)
        self.assertEqual([span.text for span in spans], ["First sentence.", "Second line!", "Third?"])
        self.assertEqual([passage[span.start:span.end] for span in spans], [span.text for span in spans])

    def test_highest_jaccard_sentence_and_earliest_tie(self) -> None:
        item = example(
            "target", "Is Mars the red planet?",
            "Venus is bright. Mars is the red planet. Mars has two moons.",
        )
        selected = select_lexical_sentence(item.question, item.passage)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.span.index, 1)
        tied = select_lexical_sentence(
            "Does Mars rotate?", "Mars can rotate. Mars may rotate. Earth also rotates."
        )
        self.assertIsNotNone(tied)
        assert tied is not None
        self.assertEqual(tied.span.index, 0)

    def test_zero_overlap_is_retained_and_flagged(self) -> None:
        item = example("target", "Mars planet?", "Ocean waves move. Forest trees grow.")
        result = lexical_evidence_removal(item)
        self.assertEqual(result.status, "ok")
        self.assertIn("zero_lexical_overlap", result.quality_flags)
        self.assertEqual(result.parameters["selected_sentence_index"], 0)

    def test_removal_deletes_selected_span_without_raw_metadata(self) -> None:
        item = example(
            "target", "Is Mars the red planet?",
            "Venus is bright.  Mars is the red planet.\nMars has two moons.",
        )
        result = lexical_evidence_removal(item)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.transformed_passage, "Venus is bright. Mars has two moons.")
        serialized = canonical_json(result.manifest_metadata())
        self.assertNotIn(item.question, serialized)
        self.assertNotIn(item.passage, serialized)
        self.assertNotIn(result.transformed_passage, serialized)

    def test_single_sentence_removal_is_visible_empty_result(self) -> None:
        item = example("target", "Is Mars red?", "Mars is red.")
        result = lexical_evidence_removal(item)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.transformed_passage, "")
        self.assertIn("whole_passage_removed", result.quality_flags)


class TokenTransformationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()

    def test_prefix_truncation_uses_ceiling_half(self) -> None:
        item = example("target", "Question?", "one two three four five")
        result = prefix_truncation_50(item, self.tokenizer)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.transformed_passage, "one two three")
        self.assertEqual(result.parameters["target_tokens"], 3)
        self.assertEqual(result.parameters["retained_tokens"], 3)

    def test_one_token_truncation_is_not_applicable(self) -> None:
        result = prefix_truncation_50(
            example("target", "Question?", "single"), self.tokenizer
        )
        self.assertEqual(result.status, "not_applicable")
        self.assertEqual(result.reason_code, "unchanged_result")

    def test_distractor_is_order_independent_and_excludes_same_passage(self) -> None:
        target = example("target", "Is Mars a red planet?", "Target passage text.")
        donors = [
            DonorText("same", target.passage),
            DonorText("red", "Mars is a red planet with dust."),
            DonorText("ocean", "Ocean waves carry blue water."),
            DonorText("forest", "Forest trees grow beside rivers."),
        ]
        first = irrelevant_distractor(target, donors, self.tokenizer, seed=20260803)
        second = irrelevant_distractor(
            target, list(reversed(donors)), self.tokenizer, seed=20260803
        )
        self.assertEqual(first.manifest_metadata(), second.manifest_metadata())
        self.assertEqual(first.transformed_passage, second.transformed_passage)
        self.assertNotIn(first.parameters["distractor_example_id"], {"same", "red"})
        self.assertEqual(first.parameters["algorithm"], DISTRACTOR_SELECTOR)

    def test_distractor_tie_break_matches_seeded_hash(self) -> None:
        target = example("target", "Is Mars red?", "Target passage.")
        donors = [
            DonorText("alpha", "Ocean water moves."),
            DonorText("beta", "Forest trees grow."),
        ]
        result = irrelevant_distractor(target, donors, self.tokenizer, seed=7)
        expected = min(
            donors,
            key=lambda donor: hashlib.sha256(
                json.dumps(
                    {
                        "candidate_example_id": donor.example_id,
                        "condition_version": "boolq-evidence-degradation-v1",
                        "seed": 7,
                        "target_example_id": target.example_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(result.parameters["distractor_example_id"], expected.example_id)

    def test_duplicate_donor_ids_fail_closed(self) -> None:
        target = example("target", "Question?", "Target passage.")
        donors = [DonorText("dup", "One passage."), DonorText("dup", "Other passage.")]
        with self.assertRaisesRegex(ValueError, "duplicate donor"):
            irrelevant_distractor(target, donors, self.tokenizer, seed=1)

    def test_contradiction_caps_excerpt_and_has_no_label_field(self) -> None:
        long_sentence = " ".join(["Mars"] + [f"token{i}" for i in range(90)]) + "."
        item = example("target", "Is Mars present?", long_sentence)
        result = lexical_contradiction(item, self.tokenizer, maximum_fragment_tokens=64)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.parameters["actual_fragment_tokens"], 64)
        self.assertTrue(
            result.transformed_passage.endswith(
                CONTRADICTION_TEMPLATE.format(fragment=" ".join(long_sentence.split()[:64]))
            )
        )
        self.assertNotIn("answer", canonical_json(result.manifest_metadata()).lower())


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = WhitespaceTokenizer()
        self.target = example(
            "target", "Is Mars the red planet?",
            "Venus is bright. Mars is the red planet. Mars has moons.",
        )
        self.donors = [
            DonorText("target", self.target.passage),
            DonorText("other", "Ocean waves move across blue water."),
        ]

    def test_data_classes_structurally_exclude_labels(self) -> None:
        for data_class in (TransformationExample, DonorText):
            names = {field.name for field in fields(data_class)}
            self.assertTrue(names.isdisjoint({"answer", "label", "target"}))

    def test_original_and_no_passage_are_exact(self) -> None:
        unchanged = original(self.target)
        empty = no_passage(self.target)
        self.assertEqual(unchanged.transformed_passage, self.target.passage)
        self.assertFalse(unchanged.manifest_metadata()["changed_from_original"])
        self.assertEqual(empty.transformed_passage, "")
        self.assertTrue(empty.manifest_metadata()["changed_from_original"])

    def test_all_conditions_have_fixed_order_and_unique_ids(self) -> None:
        results = build_all_conditions(
            self.target, self.donors, self.tokenizer,
            seed=20260803, maximum_fragment_tokens=64,
        )
        self.assertEqual(tuple(result.condition for result in results), CONDITIONS)
        metadata = [result.manifest_metadata() for result in results]
        self.assertEqual(len({row["transformation_id"] for row in metadata}), 6)
        self.assertTrue(metadata_jsonl(results).endswith("\n"))
        self.assertEqual(metadata_jsonl(results), metadata_jsonl(results))

    def test_transformation_id_binds_complete_safe_result(self) -> None:
        first = original(self.target)
        changed_output = TransformationResult(
            example_id=first.example_id,
            condition=first.condition,
            status=first.status,
            reason_code=first.reason_code,
            quality_flags=first.quality_flags,
            original_passage_sha256=first.original_passage_sha256,
            transformed_passage=first.transformed_passage + " changed",
            parameters=first.parameters,
        )
        unavailable_a = TransformationResult(
            example_id=first.example_id,
            condition="prefix_truncation_50",
            status="not_applicable",
            reason_code="reason_a",
            quality_flags=(),
            original_passage_sha256=first.original_passage_sha256,
            transformed_passage=None,
            parameters={"algorithm": "synthetic-test-v1"},
        )
        unavailable_b = TransformationResult(
            example_id=first.example_id,
            condition="prefix_truncation_50",
            status="not_applicable",
            reason_code="reason_b",
            quality_flags=(),
            original_passage_sha256=first.original_passage_sha256,
            transformed_passage=None,
            parameters={"algorithm": "synthetic-test-v1"},
        )

        identifiers = [
            result.manifest_metadata()["transformation_id"]
            for result in (first, changed_output, unavailable_a, unavailable_b)
        ]
        self.assertEqual(len(set(identifiers)), len(identifiers))
        self.assertTrue(
            all(
                re.fullmatch(r"boolqdeg-[0-9a-f]{64}", identifier)
                for identifier in identifiers
            )
        )

    def test_config_matches_implementation_contract(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/degradations.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(tuple(config["protocol"]["condition_order"]), CONDITIONS)
        self.assertEqual(config["protocol"]["version"], PROTOCOL_VERSION)
        self.assertFalse(config["protocol"]["labels_available_to_transformations"])
        self.assertFalse(config["protocol"]["persist_raw_text"])
        self.assertFalse(config["protocol"]["ordered_severity_claim_allowed"])
        self.assertTrue(config["distractor"]["transductive_input_use"])
        self.assertEqual(config["lexical_proxy"]["algorithm"], "unique-token-jaccard-v1")
        self.assertEqual(config["lexical_proxy"]["normalization"], NORMALIZATION_VERSION)
        self.assertEqual(config["lexical_proxy"]["token_pattern"], TOKEN_PATTERN_TEXT)
        self.assertEqual(
            config["lexical_proxy"]["sentence_boundary_pattern"],
            BOUNDARY_PATTERN_TEXT,
        )
        self.assertEqual(config["contradiction"]["template"], CONTRADICTION_TEMPLATE)
        self.assertEqual(config["runtime"]["maximum_sequence_tokens"], 1024)


if __name__ == "__main__":
    unittest.main()
