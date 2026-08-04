from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/prepare_degradations.py"
SPEC = importlib.util.spec_from_file_location("prepare_degradations", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load prepare_degradations.py")
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)

from llm_confidence_uq.degradations import (  # noqa: E402
    BOUNDARY_PATTERN_TEXT,
    CONDITIONS,
    CONTRADICTION_TEMPLATE,
    DISTRACTOR_SELECTOR,
    LEXICAL_ALGORITHM,
    NORMALIZATION_VERSION,
    PROTOCOL_VERSION,
    SENTENCE_SPLITTER,
    TOKEN_PATTERN_TEXT,
    TransformationExample,
    build_all_conditions,
    original,
)


DATASET_ID = "google/boolq"
DATASET_REVISION = "a" * 40
SELECTION_SEED = 20260803
ELIGIBILITY_MAX = 768


class CharacterTokenizer:
    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []
        self.encode_calls: list[tuple[str, dict[str, object]]] = []
        self.decode_calls: list[tuple[list[int], dict[str, object]]] = []
        self.is_fast = True

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        self.encode_calls.append((text, dict(kwargs)))
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        self.decode_calls.append((list(token_ids), dict(kwargs)))
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return (
            "<system>"
            + messages[0]["content"]
            + "</system><user>"
            + messages[1]["content"]
            + "</user><assistant>"
        )


class FakeSplit:
    column_names = ["question", "passage", "answer"]

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)


def prompt_spec(maximum: int = 10_000) -> prepare.PromptSpec:
    return prepare.PromptSpec(
        system="Use evidence only.",
        user_template="Passage: {passage}\nQuestion: {question}",
        classification_prefix="Answer:",
        renderer_version="qwen-chat-boolq-confidence-v1",
        maximum_prompt_tokens=maximum,
    )


def selected_input(
    tokenizer: CharacterTokenizer,
    *,
    source_index: int,
    question: str,
    passage: str,
) -> prepare.SelectedInput:
    specification = prompt_spec()
    input_sha256 = prepare.sha256_text(
        prepare.canonical_json({"passage": passage, "question": question})
    )
    rendered = prepare.render_locked_prompt(
        tokenizer,
        specification,
        question=question,
        passage=passage,
    )
    return prepare.SelectedInput(
        input_id=prepare.make_label_blind_input_id(
            dataset_revision=DATASET_REVISION,
            source_split="validation",
            source_index=source_index,
            input_sha256=input_sha256,
        ),
        source_split="validation",
        source_index=source_index,
        input_sha256=input_sha256,
        original_passage_sha256=prepare.sha256_text(passage),
        original_passage_tokens=len(passage),
        original_prompt_sha256=prepare.sha256_text(rendered),
        original_prompt_tokens=len(rendered),
        question=question,
        passage=passage,
    )


def projection_sha256(items: list[prepare.SelectedInput]) -> str:
    rows = [
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
        for item in sorted(items, key=lambda value: value.input_id)
    ]
    return prepare.sha256_bytes(prepare.canonical_jsonl_bytes(rows))


def manifest_row(
    tokenizer: CharacterTokenizer,
    source: dict[str, object],
    source_index: int,
) -> dict[str, object]:
    question = str(source["question"])
    passage = str(source["passage"])
    answer = bool(source["answer"])
    canonical_record = prepare.canonical_json(
        {"answer": answer, "passage": passage, "question": question}
    )
    record_sha256 = prepare.sha256_text(canonical_record)
    input_sha256 = prepare.sha256_text(
        prepare.canonical_json({"passage": passage, "question": question})
    )
    rendered = prepare.render_locked_prompt(
        tokenizer,
        prompt_spec(),
        question=question,
        passage=passage,
    )
    return {
        "answer": answer,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "eligibility_max_prompt_tokens": ELIGIBILITY_MAX,
        "example_id": f"boolq-validation-{source_index:05d}-{record_sha256[:12]}",
        "input_sha256": input_sha256,
        "original_passage_sha256": prepare.sha256_text(passage),
        "prompt_tokens": len(rendered),
        "record_sha256": record_sha256,
        "research_split": "test",
        "schema_version": 1,
        "selection_rank_sha256": "b" * 64,
        "selection_seed": SELECTION_SEED,
        "source_index": source_index,
        "source_split": "validation",
    }


def rebuild_row_hash(row: dict[str, object]) -> None:
    without_hash = {key: value for key, value in row.items() if key != "row_sha256"}
    row["row_sha256"] = prepare.sha256_text(prepare.canonical_json(without_hash))


class PromptAndManifestTests(unittest.TestCase):
    def test_locked_prompt_rendering_and_tokenization_flags(self) -> None:
        tokenizer = CharacterTokenizer()
        rendered = prepare.render_locked_prompt(
            tokenizer,
            prompt_spec(),
            question="Distinct question sentinel?",
            passage="Distinct passage sentinel.",
        )
        self.assertTrue(rendered.endswith("Answer:"))
        call = tokenizer.template_calls[-1]
        self.assertEqual(
            call["messages"],
            [
                {"role": "system", "content": "Use evidence only."},
                {
                    "role": "user",
                    "content": (
                        "Passage: Distinct passage sentinel.\n"
                        "Question: Distinct question sentinel?"
                    ),
                },
            ],
        )
        self.assertFalse(call["tokenize"])
        self.assertTrue(call["add_generation_prompt"])
        prepare.encode_text(prepare.CachingTokenizer(tokenizer), rendered)
        _, kwargs = tokenizer.encode_calls[-1]
        self.assertEqual(
            kwargs,
            {
                "add_special_tokens": False,
                "padding": False,
                "truncation": False,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            },
        )

    def test_verified_jsonl_is_byte_exact(self) -> None:
        tokenizer = CharacterTokenizer()
        rows = [
            manifest_row(
                tokenizer,
                {
                    "question": "Question one?",
                    "passage": "Passage one.",
                    "answer": True,
                },
                0,
            ),
            manifest_row(
                tokenizer,
                {
                    "question": "Question two?",
                    "passage": "Passage two.",
                    "answer": False,
                },
                1,
            ),
        ]
        payload = prepare.canonical_jsonl_bytes(rows)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "test.jsonl"
            path.write_bytes(payload)
            loaded = prepare.read_verified_jsonl(
                path,
                expected_sha256=prepare.sha256_bytes(payload),
                expected_rows=2,
            )
            self.assertEqual(loaded, rows)
            path.write_bytes(payload[:-1])
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                prepare.read_verified_jsonl(
                    path,
                    expected_sha256=prepare.sha256_bytes(payload),
                    expected_rows=2,
                )


class LabelIsolationTests(unittest.TestCase):
    def test_reconstruction_is_invariant_to_synchronized_label_permutation(self) -> None:
        original_sources = [
            {
                "question": "Is alpha present?",
                "passage": "Alpha appears in this passage.",
                "answer": True,
            },
            {
                "question": "Is beta absent?",
                "passage": "Beta appears in a different passage.",
                "answer": False,
            },
        ]
        permuted_sources = [
            {**original_sources[0], "answer": False},
            {**original_sources[1], "answer": True},
        ]
        tokenizer_a = CharacterTokenizer()
        tokenizer_b = CharacterTokenizer()
        manifest_a = [
            manifest_row(tokenizer_a, source, index)
            for index, source in enumerate(original_sources)
        ]
        manifest_b = [
            manifest_row(tokenizer_b, source, index)
            for index, source in enumerate(permuted_sources)
        ]
        selected_a, _ = prepare.reconstruct_selected_inputs(
            manifest_a,
            FakeSplit(original_sources),
            tokenizer_a,
            prompt_spec(),
            dataset_id=DATASET_ID,
            dataset_revision=DATASET_REVISION,
            selection_seed=SELECTION_SEED,
            eligibility_max_prompt_tokens=ELIGIBILITY_MAX,
        )
        selected_b, _ = prepare.reconstruct_selected_inputs(
            manifest_b,
            FakeSplit(permuted_sources),
            tokenizer_b,
            prompt_spec(),
            dataset_id=DATASET_ID,
            dataset_revision=DATASET_REVISION,
            selection_seed=SELECTION_SEED,
            eligibility_max_prompt_tokens=ELIGIBILITY_MAX,
        )
        self.assertEqual(selected_a, selected_b)
        forbidden = {
            "answer",
            "label",
            "record_sha256",
            "selection_rank_sha256",
            "target",
        }
        self.assertTrue(
            {field.name for field in fields(prepare.SelectedInput)}.isdisjoint(forbidden)
        )
        original_source_ids = {str(row["example_id"]) for row in manifest_a + manifest_b}
        self.assertTrue(
            all(item.input_id not in original_source_ids for item in selected_a)
        )

        payloads_a, _ = prepare.build_core(
            selected_a,
            CharacterTokenizer(),
            prompt_spec(),
            seed=SELECTION_SEED,
            maximum_fragment_tokens=8,
            label_blind_projection_sha256=projection_sha256(selected_a),
        )
        payloads_b, _ = prepare.build_core(
            selected_b,
            CharacterTokenizer(),
            prompt_spec(),
            seed=SELECTION_SEED,
            maximum_fragment_tokens=8,
            label_blind_projection_sha256=projection_sha256(selected_b),
        )
        self.assertEqual(payloads_a, payloads_b)

    def test_reconstruction_rejects_label_or_prompt_drift(self) -> None:
        source = {
            "question": "Is gamma present?",
            "passage": "Gamma is present.",
            "answer": True,
        }
        tokenizer = CharacterTokenizer()
        row = manifest_row(tokenizer, source, 0)
        bad_label = dict(row)
        bad_label["answer"] = False
        with self.assertRaisesRegex(RuntimeError, "label mismatch"):
            prepare.reconstruct_selected_inputs(
                [bad_label],
                FakeSplit([source]),
                tokenizer,
                prompt_spec(),
                dataset_id=DATASET_ID,
                dataset_revision=DATASET_REVISION,
                selection_seed=SELECTION_SEED,
                eligibility_max_prompt_tokens=ELIGIBILITY_MAX,
            )
        bad_prompt = dict(row)
        bad_prompt["prompt_tokens"] = int(row["prompt_tokens"]) + 1
        with self.assertRaisesRegex(RuntimeError, "prompt-token"):
            prepare.reconstruct_selected_inputs(
                [bad_prompt],
                FakeSplit([source]),
                tokenizer,
                prompt_spec(),
                dataset_id=DATASET_ID,
                dataset_revision=DATASET_REVISION,
                selection_seed=SELECTION_SEED,
                eligibility_max_prompt_tokens=ELIGIBILITY_MAX,
            )


class CoreBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        tokenizer = CharacterTokenizer()
        self.inputs = [
            selected_input(
                tokenizer,
                source_index=2,
                question="Does crimson appear?",
                passage="Crimson appears here. Ocean waves move elsewhere.",
            ),
            selected_input(
                tokenizer,
                source_index=0,
                question="Does cobalt appear?",
                passage="Cobalt appears here. Forest trees grow elsewhere.",
            ),
            selected_input(
                tokenizer,
                source_index=1,
                question="Does amber appear?",
                passage="Amber appears here. Mountain snow falls elsewhere.",
            ),
        ]

    def build(self):
        tokenizer = CharacterTokenizer()
        return prepare.build_core(
            self.inputs,
            tokenizer,
            prompt_spec(),
            seed=SELECTION_SEED,
            maximum_fragment_tokens=8,
            label_blind_projection_sha256=projection_sha256(self.inputs),
        )

    def test_exact_cardinality_order_and_repeatability(self) -> None:
        payloads_a, report_a = self.build()
        payloads_b, report_b = prepare.build_core(
            list(reversed(self.inputs)),
            CharacterTokenizer(),
            prompt_spec(),
            seed=SELECTION_SEED,
            maximum_fragment_tokens=8,
            label_blind_projection_sha256=projection_sha256(self.inputs),
        )
        self.assertEqual(payloads_a, payloads_b)
        self.assertEqual(report_a, report_b)
        rows = [
            json.loads(line)
            for line in payloads_a["test.jsonl"].decode("utf-8").splitlines()
        ]
        self.assertEqual(len(rows), len(self.inputs) * len(CONDITIONS))
        self.assertEqual(
            [(row["input_id"], row["condition_index"]) for row in rows],
            [
                (item.input_id, condition_index)
                for item in sorted(self.inputs, key=lambda value: value.input_id)
                for condition_index in range(len(CONDITIONS))
            ],
        )
        self.assertEqual(
            len({row["transformation_id"] for row in rows}),
            len(rows),
        )

    def test_serialization_contains_no_raw_or_label_dependent_data(self) -> None:
        payloads, _ = self.build()
        serialized = payloads["test.jsonl"].decode("utf-8")
        for item in self.inputs:
            self.assertNotIn(item.question, serialized)
            self.assertNotIn(item.passage, serialized)
        rows = [json.loads(line) for line in serialized.splitlines()]
        for row in rows:
            self.assertTrue(
                set(prepare._all_keys(row)).isdisjoint(prepare.FORBIDDEN_KEYS)
            )
            self.assertEqual(set(row), prepare.ROW_FIELDS)
        self.assertTrue(payloads["test.jsonl"].endswith(b"\n"))
        self.assertNotIn(b"\r\n", payloads["test.jsonl"])

    def test_empty_string_is_rendered_but_none_is_not_run(self) -> None:
        tokenizer = CharacterTokenizer()
        one_character = selected_input(
            tokenizer,
            source_index=4,
            question="X?",
            passage="X",
        )
        donor = selected_input(
            tokenizer,
            source_index=5,
            question="Y?",
            passage="Donor text.",
        )
        payloads, _ = prepare.build_core(
            [one_character, donor],
            CharacterTokenizer(),
            prompt_spec(),
            seed=SELECTION_SEED,
            maximum_fragment_tokens=4,
            label_blind_projection_sha256=projection_sha256([one_character, donor]),
        )
        rows = [
            row
            for row in (
                json.loads(line)
                for line in payloads["test.jsonl"].decode("utf-8").splitlines()
            )
            if row["input_id"] == one_character.input_id
        ]
        by_condition = {row["condition"]: row for row in rows}
        for condition in ("lexical_evidence_removal", "no_passage"):
            row = by_condition[condition]
            self.assertEqual(row["transformation_status"], "ok")
            self.assertEqual(row["runtime_status"], "ok")
            self.assertEqual(row["transformed_passage_tokens"], 0)
            self.assertIsNotNone(row["rendered_prompt_sha256"])
            self.assertIsNotNone(row["prompt_tokens"])
        prefix = by_condition["prefix_truncation_50"]
        self.assertEqual(prefix["transformation_status"], "not_applicable")
        self.assertEqual(prefix["runtime_status"], "not_run")
        self.assertIsNone(prefix["rendered_prompt_sha256"])
        self.assertIsNone(prefix["prompt_tokens"])

    def test_runtime_overflow_boundary_preserves_transformation(self) -> None:
        tokenizer = CharacterTokenizer()
        item = self.inputs[0]
        result = original(
            TransformationExample(item.input_id, item.question, item.passage)
        )
        rendered = prepare.render_locked_prompt(
            tokenizer,
            prompt_spec(),
            question=item.question,
            passage=item.passage,
        )
        exact_limit = len(rendered)
        exact_row, _ = prepare.materialize_runtime_row(
            selected=item,
            result=result,
            condition_index=0,
            tokenizer=tokenizer,
            cached_tokenizer=prepare.CachingTokenizer(tokenizer),
            prompt_spec=prompt_spec(exact_limit),
        )
        overflow_row, _ = prepare.materialize_runtime_row(
            selected=item,
            result=result,
            condition_index=0,
            tokenizer=tokenizer,
            cached_tokenizer=prepare.CachingTokenizer(tokenizer),
            prompt_spec=prompt_spec(exact_limit - 1),
        )
        self.assertEqual(exact_row["runtime_status"], "ok")
        self.assertIsNone(exact_row["runtime_reason_code"])
        self.assertEqual(overflow_row["transformation_status"], "ok")
        self.assertEqual(overflow_row["runtime_status"], "failed")
        self.assertEqual(
            overflow_row["runtime_reason_code"],
            prepare.RUNTIME_OVERFLOW_REASON,
        )
        self.assertEqual(overflow_row["prompt_tokens"], exact_limit)
        self.assertIsNotNone(overflow_row["transformed_passage_sha256"])
        self.assertIsNotNone(overflow_row["rendered_prompt_sha256"])

    def test_validator_rejects_semantic_mutation(self) -> None:
        payloads, _ = self.build()
        rows = [
            json.loads(line)
            for line in payloads["test.jsonl"].decode("utf-8").splitlines()
        ]
        mutated = [dict(row) for row in rows]
        target = next(
            row
            for row in mutated
            if row["condition"] == "original"
        )
        target["runtime_status"] = "failed"
        target["runtime_reason_code"] = prepare.RUNTIME_OVERFLOW_REASON
        rebuild_row_hash(target)
        with self.assertRaisesRegex(RuntimeError, "overflow"):
            prepare.validate_core_rows(mutated, self.inputs, prompt_spec())

        identity_mutation = [dict(row) for row in rows]
        identity_target = next(
            row
            for row in identity_mutation
            if row["condition"] == "original"
        )
        identity_target["parameters"] = {"algorithm": "altered-identity-v1"}
        rebuild_row_hash(identity_target)
        with self.assertRaisesRegex(RuntimeError, "transformation ID payload"):
            prepare.validate_core_rows(
                identity_mutation,
                self.inputs,
                prompt_spec(),
            )


class ConfigurationTests(unittest.TestCase):
    def test_preflight_verifies_existing_and_rejects_differences(self) -> None:
        with tempfile.TemporaryDirectory(dir=prepare.REPO_ROOT) as temporary_directory:
            output_dir = Path(temporary_directory) / "artifacts"
            payloads = {"test.json": b'{"value":1}\n'}
            first = prepare.preflight_and_write(output_dir, payloads)
            second = prepare.preflight_and_write(output_dir, payloads)
            self.assertEqual(first, {"test.json": "created"})
            self.assertEqual(second, {"test.json": "verified_existing"})
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                prepare.preflight_and_write(
                    output_dir,
                    {"test.json": b'{"value":2}\n'},
                )

            dangling = Path(temporary_directory) / "dangling-output"
            try:
                dangling.symlink_to(
                    Path(temporary_directory) / "missing-target",
                    target_is_directory=True,
                )
            except OSError:
                pass
            else:
                with self.assertRaisesRegex(RuntimeError, "symbolic links"):
                    prepare.preflight_and_write(
                        dangling,
                        {"test.json": b'{"value":1}\n'},
                    )

    def test_repository_configs_match_approved_numerical_protocol(self) -> None:
        data_path = ROOT / "configs/data.yaml"
        degradation_path = ROOT / "configs/degradations.yaml"
        if not data_path.is_file() or not degradation_path.is_file():
            self.skipTest("repository YAML files are unavailable in the local harness")
        try:
            import yaml
        except ModuleNotFoundError:
            self.skipTest("PyYAML is unavailable in the local harness")
        data_config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        degradation_config = yaml.safe_load(
            degradation_path.read_text(encoding="utf-8")
        )
        specification = prepare.prompt_spec_from_configs(
            data_config,
            degradation_config,
        )
        self.assertEqual(specification.maximum_prompt_tokens, 1024)
        self.assertEqual(
            data_config["selection"]["original_prompt_eligibility_max_tokens"],
            768,
        )
        self.assertEqual(
            degradation_config["distractor"]["maximum_fragment_tokens"],
            64,
        )
        self.assertEqual(
            degradation_config["contradiction"]["maximum_fragment_tokens"],
            64,
        )

    def test_configuration_contract_constructs_locked_prompt_spec(self) -> None:
        data_config = {
            "schema_version": 1,
            "model": {
                "tokenizer_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "tokenizer_revision": "c" * 40,
            },
            "prompt": {
                "system": "Use evidence only.",
                "user_template": "Passage: {passage}\nQuestion: {question}",
                "classification_prefix": "Answer:",
                "renderer_version": "qwen-chat-boolq-confidence-v1",
                "apply_chat_template": True,
                "add_generation_prompt": True,
                "add_special_tokens": False,
                "padding": False,
                "truncation": False,
            },
            "selection": {
                "seed": SELECTION_SEED,
                "runtime_max_sequence_tokens": 1024,
            },
        }
        degradation_config = {
            "schema_version": 1,
            "input": {
                "research_split": "test",
                "expected_examples": 400,
            },
            "tokenizer": {
                "id": "Qwen/Qwen2.5-1.5B-Instruct",
                "revision": "c" * 40,
                "add_special_tokens_for_passage_operations": False,
                "exact_decode_reencode_required": True,
            },
            "protocol": {
                "version": PROTOCOL_VERSION,
                "seed": SELECTION_SEED,
                "condition_order": list(CONDITIONS),
                "labels_available_to_transformations": False,
                "persist_raw_text": False,
                "preserve_not_applicable_rows": True,
                "preserve_failed_rows": True,
                "ordered_severity_claim_allowed": False,
            },
            "lexical_proxy": {
                "normalization": NORMALIZATION_VERSION,
                "algorithm": LEXICAL_ALGORITHM,
                "token_pattern": TOKEN_PATTERN_TEXT,
                "sentence_splitter": SENTENCE_SPLITTER,
                "sentence_boundary_pattern": BOUNDARY_PATTERN_TEXT,
                "tie_break": "earliest-character-span-v1",
                "zero_overlap_policy": "select_earliest-and-flag",
            },
            "prefix_truncation": {
                "algorithm": "token-prefix-half-ceiling-v1",
                "ratio_numerator": 1,
                "ratio_denominator": 2,
                "rounding": "ceiling",
            },
            "distractor": {
                "algorithm": DISTRACTOR_SELECTOR,
                "source_pool": "selected_test_examples",
                "transductive_input_use": True,
                "exclude_target_example": True,
                "exclude_identical_passage": True,
                "uses_labels": False,
            },
            "contradiction": {
                "algorithm": LEXICAL_ALGORITHM,
                "template": CONTRADICTION_TEMPLATE,
                "uses_labels": False,
            },
            "runtime": {
                "renderer_version": "qwen-chat-boolq-confidence-v1",
                "maximum_sequence_tokens": 1024,
                "truncation_allowed": False,
                "overflow_policy": "preserve_as_failed_row",
            },
        }
        specification = prepare.prompt_spec_from_configs(
            data_config,
            degradation_config,
        )
        self.assertEqual(specification.maximum_prompt_tokens, 1024)
        self.assertEqual(
            specification.user_template,
            "Passage: {passage}\nQuestion: {question}",
        )


if __name__ == "__main__":
    unittest.main()
