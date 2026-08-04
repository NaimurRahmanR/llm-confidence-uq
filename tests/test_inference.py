from __future__ import annotations

import ast
import builtins
import contextlib
from dataclasses import fields, replace
import importlib.util
import io
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq import inference  # noqa: E402


try:
    import torch
except ModuleNotFoundError:  # The Colab gate requires Torch; local pure tests do not.
    torch = None


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CONFIG_SHA256 = "c" * 64
RAW_PROMPT_SENTINEL = "RAW_PROMPT_SENTINEL_DO_NOT_SERIALIZE"


def runnable_example(
    *,
    ordinal: int = 0,
    digest_character: str = "a",
    input_digest_character: str | None = None,
    condition_index: int = 0,
    prompt: str = RAW_PROMPT_SENTINEL,
    prompt_tokens: int = 3,
) -> inference.InferenceExample:
    input_character = input_digest_character or digest_character
    return inference.InferenceExample(
        ordinal=ordinal,
        transformation_id="boolqdeg-" + digest_character * 64,
        degradation_row_sha256=digest_character * 64,
        input_id="boolqinput-" + input_character * 64,
        source_split="validation",
        source_index=ordinal,
        condition_index=condition_index,
        condition="original" if condition_index == 0 else "no_passage",
        runtime_status="ok",
        runtime_reason_code=None,
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        rendered_prompt_sha256=inference.sha256_text(prompt),
    )


def not_run_example() -> inference.InferenceExample:
    return inference.InferenceExample(
        ordinal=0,
        transformation_id="boolqdeg-" + "d" * 64,
        degradation_row_sha256="e" * 64,
        input_id="boolqinput-" + "f" * 64,
        source_split="validation",
        source_index=17,
        condition_index=2,
        condition="prefix_truncation_50",
        runtime_status="failed",
        runtime_reason_code="runtime_sequence_overflow",
        prompt=None,
        prompt_tokens=1100,
        rendered_prompt_sha256="1" * 64,
    )


def valid_prediction_row(
    *,
    example: inference.InferenceExample | None = None,
    completion: str = " Yes\nConfidence: 82",
    reached_max: bool = False,
) -> dict[str, object]:
    item = example or runnable_example()
    parsed = inference.parse_expressed_continuation(completion)
    return inference.make_prediction_row(
        item,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        inference_config_sha256=CONFIG_SHA256,
        inference_seed=20260803,
        yes_token_id=7414,
        no_token_id=2308,
        yes_logit=math.log(3.0),
        no_logit=0.0,
        p_yes_binary=0.75,
        p_no_binary=0.25,
        p_yes_full_vocabulary=0.15,
        p_no_full_vocabulary=0.05,
        yes_no_full_vocabulary_mass=0.20,
        token_prediction="Yes",
        token_confidence=0.75,
        predictive_entropy=-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)),
        generated_continuation=completion,
        generated_continuation_tokens=24 if reached_max else 5,
        generation_ended_with_eos=not reached_max,
        generation_reached_max_new_tokens=reached_max,
        expressed_parse=parsed,
    )


class ParserTests(unittest.TestCase):
    def test_valid_boundaries_and_optional_terminal_newline(self) -> None:
        cases = (
            (" Yes\nConfidence: 0", "Yes", 0),
            (" No\nConfidence: 100\n", "No", 100),
        )
        for text, answer, confidence in cases:
            with self.subTest(text=text):
                parsed = inference.parse_expressed_continuation(text)
                self.assertTrue(parsed.valid)
                self.assertEqual(parsed.answer, answer)
                self.assertEqual(parsed.confidence, confidence)
                self.assertIsNone(parsed.reason_code)

    def test_parser_consumes_only_continuation_after_answer_prefix(self) -> None:
        valid = inference.parse_expressed_continuation(" Yes\nConfidence: 82")
        wrong = inference.parse_expressed_continuation(
            "Answer: Yes\nConfidence: 82"
        )
        self.assertTrue(valid.valid)
        self.assertFalse(wrong.valid)
        self.assertEqual(wrong.reason_code, "answer_line_invalid")

    def test_whitespace_variants_are_invalid_without_fallback(self) -> None:
        invalid = (
            "Yes\nConfidence: 82",
            "  Yes\nConfidence: 82",
            " Yes \nConfidence: 82",
            " Yes\n Confidence: 82",
            " Yes\nConfidence: 82 ",
            " Yes\r\nConfidence: 82",
            "\n Yes\nConfidence: 82",
        )
        for text in invalid:
            with self.subTest(text=repr(text)):
                parsed = inference.parse_expressed_continuation(text)
                self.assertFalse(parsed.valid)
                self.assertIsNone(parsed.answer)
                self.assertIsNone(parsed.confidence)
                self.assertIsNotNone(parsed.reason_code)

    def test_extra_duplicate_and_missing_lines_are_not_salvaged(self) -> None:
        invalid = (
            "",
            " Yes",
            " Yes\n",
            " Yes\nConfidence: 82\nExplanation: certain",
            " Yes\nConfidence: 82\nConfidence: 81",
            "Confidence: 82\n Yes",
            " Yes\n\nConfidence: 82",
        )
        for text in invalid:
            with self.subTest(text=repr(text)):
                parsed = inference.parse_expressed_continuation(text)
                self.assertFalse(parsed.valid)
                self.assertIsNone(parsed.answer)
                self.assertIsNone(parsed.confidence)

    def test_noncanonical_and_out_of_range_numbers_are_invalid(self) -> None:
        invalid = ("-1", "101", "00", "01", "+7", "7.0", "82%", "٨٢")
        for value in invalid:
            with self.subTest(value=value):
                parsed = inference.parse_expressed_continuation(
                    f" Yes\nConfidence: {value}"
                )
                self.assertFalse(parsed.valid)
                self.assertIsNone(parsed.answer)
                self.assertIsNone(parsed.confidence)


class ContextualTokenTests(unittest.TestCase):
    class Tokenizer:
        def __init__(
            self,
            *,
            yes_suffix: list[int] | None = None,
            no_suffix: list[int] | None = None,
            drift: bool = False,
        ) -> None:
            self.yes_suffix = yes_suffix if yes_suffix is not None else [7414]
            self.no_suffix = no_suffix if no_suffix is not None else [2308]
            self.drift = drift

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            if add_special_tokens:
                raise AssertionError("special tokens must be disabled")
            prefix = [10, 20, 30]
            if text.endswith(" Yes"):
                return ([99, 20, 30] if self.drift else prefix) + self.yes_suffix
            if text.endswith(" No"):
                return prefix + self.no_suffix
            return prefix

    def test_contextual_tokens_are_single_distinct_and_pinned(self) -> None:
        observed = inference.contextual_class_token_ids(
            self.Tokenizer(), "locked prompt ending Answer:"
        )
        self.assertEqual(observed, (7414, 2308))

    def test_contextual_prefix_multitoken_and_id_drift_fail_closed(self) -> None:
        cases = (
            self.Tokenizer(drift=True),
            self.Tokenizer(yes_suffix=[7, 8]),
            self.Tokenizer(yes_suffix=[7]),
            self.Tokenizer(yes_suffix=[7], no_suffix=[7]),
        )
        for tokenizer in cases:
            with self.subTest(tokenizer=tokenizer.__dict__):
                with self.assertRaises(inference.InferenceContractError):
                    inference.contextual_class_token_ids(
                        tokenizer, "locked prompt ending Answer:"
                    )


class ExampleAndDatasetTests(unittest.TestCase):
    def test_example_is_structurally_label_free_and_binds_prompt_hash(self) -> None:
        names = {field.name for field in fields(inference.InferenceExample)}
        forbidden = {
            "answer", "correct", "label", "target", "question", "passage"
        }
        self.assertTrue(names.isdisjoint(forbidden))
        example = runnable_example()
        self.assertEqual(
            example.rendered_prompt_sha256,
            inference.sha256_text(RAW_PROMPT_SENTINEL),
        )
        with self.assertRaisesRegex(
            inference.InferenceContractError, "rendered prompt SHA"
        ):
            replace(example, rendered_prompt_sha256="9" * 64)

    def test_dataset_preserves_order_and_indexing(self) -> None:
        examples = (
            runnable_example(ordinal=0, digest_character="a"),
            runnable_example(ordinal=1, digest_character="b"),
            runnable_example(ordinal=2, digest_character="c"),
        )
        dataset = inference.InferenceDataset(examples)
        self.assertEqual(len(dataset), 3)
        self.assertEqual([dataset[index] for index in range(3)], list(examples))

    def test_dataset_rejects_duplicate_ids_and_noncontiguous_ordinals(self) -> None:
        first = runnable_example(ordinal=0, digest_character="a")
        duplicate = replace(first, ordinal=1, source_index=1)
        skipped = runnable_example(ordinal=2, digest_character="b")
        with self.assertRaisesRegex(
            inference.InferenceContractError, "duplicate transformation"
        ):
            inference.InferenceDataset((first, duplicate))
        with self.assertRaisesRegex(inference.InferenceContractError, "contiguous"):
            inference.InferenceDataset((first, skipped))


class SerializationAndPredictionTests(unittest.TestCase):
    def test_canonical_jsonl_is_stable_utf8_and_newline_terminated(self) -> None:
        value = {"z": 1, "a": "café"}
        self.assertEqual(inference.canonical_json(value), '{"a":"café","z":1}')
        payload = inference.canonical_jsonl_bytes((value, value))
        self.assertEqual(payload, b'{"a":"caf\xc3\xa9","z":1}\n' * 2)
        self.assertTrue(payload.endswith(b"\n"))
        self.assertNotIn(b"\r\n", payload)
        with self.assertRaises(ValueError):
            inference.canonical_json({"bad": float("nan")})

    def test_prediction_identity_hash_and_label_isolation_are_deterministic(self) -> None:
        first = valid_prediction_row()
        second = valid_prediction_row()
        self.assertEqual(first, second)
        self.assertEqual(first["class_order"], ["Yes", "No"])
        self.assertEqual(first["inference_seed"], 20260803)
        self.assertEqual(
            first["row_sha256"],
            inference.sha256_text(
                inference.canonical_json(
                    {key: value for key, value in first.items() if key != "row_sha256"}
                )
            ),
        )
        serialized = inference.prediction_jsonl_bytes((first,))
        self.assertNotIn(RAW_PROMPT_SENTINEL.encode("utf-8"), serialized)
        self.assertTrue(set(first).isdisjoint(inference.FORBIDDEN_PREDICTION_KEYS))
        self.assertNotIn(b'"answer":', serialized)
        self.assertNotIn(b'"label":', serialized)
        self.assertNotIn(b'"correct":', serialized)

    def test_invalid_expressed_parse_is_retained_without_imputation(self) -> None:
        row = valid_prediction_row(completion=" Yes\nConfidence: 101")
        self.assertEqual(row["inference_status"], "ok")
        self.assertFalse(row["expressed_parser_valid"])
        self.assertIsNone(row["generated_answer"])
        self.assertIsNone(row["expressed_confidence"])
        self.assertIsNotNone(row["expressed_parser_reason_code"])
        self.assertEqual(row["generated_continuation"], " Yes\nConfidence: 101")
        self.assertEqual(row["p_yes_binary"], 0.75)

    def test_max_token_completion_is_invalid_even_when_text_parses(self) -> None:
        row = valid_prediction_row(reached_max=True)
        self.assertFalse(row["expressed_parser_valid"])
        self.assertEqual(
            row["expressed_parser_reason_code"],
            "generation_max_new_tokens_reached",
        )
        self.assertIsNone(row["generated_answer"])
        self.assertIsNone(row["expressed_confidence"])
        inference.validate_prediction_row(row)

    def test_not_run_row_has_complete_null_schema(self) -> None:
        row = inference.make_prediction_row(
            not_run_example(),
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            inference_config_sha256=CONFIG_SHA256,
            yes_token_id=7414,
            no_token_id=2308,
            inference_status="not_run",
            inference_reason_code="degradation_runtime_failed",
        )
        inference.validate_prediction_row(row)
        self.assertEqual(row["inference_status"], "not_run")
        self.assertIsNone(row["yes_logit"])
        self.assertIsNone(row["token_prediction"])
        self.assertIsNone(row["generated_continuation"])
        self.assertIsNone(row["generation_ended_with_eos"])
        self.assertFalse(row["expressed_parser_valid"])
        self.assertEqual(row["expressed_parser_reason_code"], "not_run")

    def test_validator_rejects_rehashed_semantic_and_order_mutations(self) -> None:
        row = valid_prediction_row()
        forbidden = dict(row)
        forbidden["answer"] = True
        corrupt = dict(row)
        corrupt["p_yes_binary"] = 0.5
        corrupt["row_sha256"] = inference.sha256_text(
            inference.canonical_json(
                {key: value for key, value in corrupt.items() if key != "row_sha256"}
            )
        )
        for mutated in (forbidden, corrupt):
            with self.subTest(extra=set(mutated) - set(row)):
                with self.assertRaises(inference.InferenceContractError):
                    inference.validate_prediction_row(mutated)
        later = valid_prediction_row(
            example=runnable_example(
                ordinal=1,
                digest_character="b",
                input_digest_character="b",
            )
        )
        with self.assertRaisesRegex(
            inference.InferenceContractError, "order"
        ):
            inference.validate_prediction_rows((later, row))


class ConfigurationAndRunnerTests(unittest.TestCase):
    def test_baseline_config_contains_exact_pinned_label_blind_contract(self) -> None:
        text = (ROOT / "configs/baseline.yaml").read_text(encoding="utf-8")
        required = (
            "protocol_version: boolq-baseline-inference-v1",
            "id: Qwen/Qwen2.5-1.5B-Instruct",
            "revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "parameter_count: 1543714304",
            'class_order: ["Yes", "No"]',
            '"Yes": " Yes"',
            '"No": " No"',
            '"Yes": 7414',
            '"No": 2308',
            'argmax_tie_policy: "Yes"',
            "logits_to_keep: 1",
            "attention_derived_position_ids: true",
            "persist_generated_continuation: true",
            "reject_max_new_tokens_reached: true",
            "batch_shards: true",
            "labels_available_to_model_inference: false",
            "include_ground_truth: false",
            "include_correctness: false",
            "do_sample: false",
            "num_beams: 1",
            "eos_token_ids: [151645, 151643]",
            "pad_token_id: 151643",
            "repetition_penalty: 1.1",
            "repetition_penalty_scope: generated-continuation-only",
            "inherited_repetition_penalty_override: 1.0",
            "parser: exact-continuation-v1",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)
        try:
            import yaml
        except ModuleNotFoundError:
            self.skipTest("PyYAML is unavailable in this local audit runtime")
        runner = self._load_runner()
        runner.validate_config_contract(yaml.safe_load(text))

    @staticmethod
    def _load_runner(*, block_network_and_model_imports: bool = False):
        path = ROOT / "scripts/run_baseline.py"
        spec = importlib.util.spec_from_file_location("baseline_runner_test", path)
        if spec is None or spec.loader is None:
            raise AssertionError("runner import spec missing")
        module = importlib.util.module_from_spec(spec)
        if not block_network_and_model_imports:
            spec.loader.exec_module(module)
            return module
        original_import = builtins.__import__
        blocked = {
            "datasets",
            "huggingface_hub",
            "prepare_degradations",
            "transformers",
            "yaml",
        }

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in blocked:
                raise AssertionError(f"heavy import at module load: {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        try:
            spec.loader.exec_module(module)
        finally:
            builtins.__import__ = original_import
        return module

    def test_runner_import_does_not_load_hub_dataset_or_model_libraries(self) -> None:
        before = {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        runner = self._load_runner(block_network_and_model_imports=True)
        after = {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        self.assertEqual(before, after)
        self.assertEqual(runner.ALLOWED_STAGES, ("smoke12", "smoke60", "full"))

    def test_runner_cli_requires_explicit_locked_stage(self) -> None:
        runner = self._load_runner()
        parser = runner.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as missing:
                parser.parse_args([])
        self.assertEqual(missing.exception.code, 2)
        for stage in runner.ALLOWED_STAGES:
            self.assertEqual(parser.parse_args(["--stage", stage]).stage, stage)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as invalid:
                parser.parse_args(["--stage", "everything"])
        self.assertEqual(invalid.exception.code, 2)

    def test_stage_selection_is_exact_deterministic_and_label_blind(self) -> None:
        runner = self._load_runner()
        conditions = (
            "original",
            "lexical_evidence_removal",
            "prefix_truncation_50",
            "irrelevant_distractor",
            "lexical_contradiction",
            "no_passage",
        )
        rows: list[dict[str, object]] = []
        for input_index in range(400):
            input_id = f"boolqinput-{input_index:064x}"
            for condition_index, condition in enumerate(conditions):
                rows.append(
                    {
                        "input_id": input_id,
                        "condition_index": condition_index,
                        "condition": condition,
                        "runtime_status": "ok",
                        "prompt_tokens": (
                            1000 if input_index == 399 and condition_index == 4
                            else 100 + condition_index
                        ),
                    }
                )
        expected = {"smoke12": (12, 2), "smoke60": (60, 10), "full": (2400, 400)}
        for stage, (row_count, input_count) in expected.items():
            with self.subTest(stage=stage):
                selected = runner.select_stage_rows(rows, stage)
                self.assertEqual(len(selected), row_count)
                self.assertEqual(
                    len({str(row["input_id"]) for row in selected}),
                    input_count,
                )
                self.assertFalse(any("answer" in row or "label" in row for row in selected))
        smoke60 = runner.select_stage_rows(rows, "smoke60")
        self.assertIn(
            f"boolqinput-{399:064x}",
            {str(row["input_id"]) for row in smoke60},
        )
        source = (ROOT / "scripts/run_baseline.py").read_text(encoding="utf-8")
        function = source[source.index("def select_stage_rows"):source.index(
            "\ndef ", source.index("def select_stage_rows") + 5
        )]
        self.assertNotIn("answer", function)
        self.assertNotIn("label", function)

    def test_runner_reuses_frozen_transformations_instead_of_reimplementing(self) -> None:
        path = ROOT / "scripts/run_baseline.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        forbidden = {
            "original",
            "lexical_evidence_removal",
            "prefix_truncation_50",
            "irrelevant_distractor",
            "lexical_contradiction",
            "no_passage",
        }
        self.assertTrue(function_names.isdisjoint(forbidden))
        self.assertIn("build_all_conditions", source)
        self.assertIn("materialize_runtime_row", source)
        self.assertIn("eos_token_id=list(eos_token_ids)", source)
        self.assertIn("prompt_ignore_length=input_width", source)
        self.assertIn(
            "repetition_penalty=built_in_repetition_penalty",
            source,
        )
        self.assertIn("logits_processor=[repetition_processor]", source)
        self.assertIn("model.generation_config.eos_token_id", source)

    def test_resumed_shards_bind_model_config_seed_and_input_identity(self) -> None:
        runner = self._load_runner()
        example = runnable_example()
        row = valid_prediction_row(example=example)
        runner._batch_identity_matches(
            (row,),
            (example,),
            config_sha256=CONFIG_SHA256,
            inference_seed=20260803,
        )
        with self.assertRaises(runner.BaselineRunError):
            runner._batch_identity_matches(
                (row,),
                (example,),
                config_sha256="d" * 64,
                inference_seed=20260803,
            )

    def test_generation_suffix_distinguishes_eos_max_and_invalid_padding(self) -> None:
        runner = self._load_runner()
        content, ended, reached = runner._generation_content(
            (11, 12, 99, 0),
            eos_token_ids=(99, 77),
            pad_token_id=0,
            special_token_ids=(0, 77, 99),
            maximum_new_tokens=4,
        )
        self.assertEqual(content, [11, 12])
        self.assertTrue(ended)
        self.assertFalse(reached)
        content, ended, reached = runner._generation_content(
            (11, 77, 0, 0),
            eos_token_ids=(99, 77),
            pad_token_id=0,
            special_token_ids=(0, 77, 99),
            maximum_new_tokens=4,
        )
        self.assertEqual(content, [11])
        self.assertTrue(ended)
        self.assertFalse(reached)
        content, ended, reached = runner._generation_content(
            (11, 12, 13, 14),
            eos_token_ids=(99, 77),
            pad_token_id=0,
            special_token_ids=(0, 99),
            maximum_new_tokens=4,
        )
        self.assertEqual(content, [11, 12, 13, 14])
        self.assertFalse(ended)
        self.assertTrue(reached)
        with self.assertRaises(runner.BaselineRunError):
            runner._generation_content(
                (11, 0, 0, 0),
                eos_token_ids=(99, 77),
                pad_token_id=0,
                special_token_ids=(0, 99),
                maximum_new_tokens=4,
            )
        with self.assertRaises(runner.BaselineRunError):
            runner._generation_content(
                (11, 88, 99, 0),
                eos_token_ids=(99, 77),
                pad_token_id=0,
                special_token_ids=(0, 77, 88, 99),
                maximum_new_tokens=4,
            )

    def test_degradation_validator_enforces_per_input_order_and_no_targets(self) -> None:
        runner = self._load_runner()
        rows: list[dict[str, object]] = []
        for input_index in range(400):
            input_id = f"boolqinput-{input_index:064x}"
            for condition_index, condition in enumerate(runner.EXPECTED_CONDITIONS):
                without_hash: dict[str, object] = {
                    "input_id": input_id,
                    "transformation_id": (
                        "boolqdeg-" + f"{input_index * 6 + condition_index:064x}"
                    ),
                    "condition_index": condition_index,
                    "condition": condition,
                    "transformation_status": "ok",
                    "runtime_status": "ok",
                    "runtime_reason_code": None,
                    "prompt_tokens": 100 + condition_index,
                    "rendered_prompt_sha256": "a" * 64,
                }
                row = dict(without_hash)
                row["row_sha256"] = inference.sha256_text(
                    inference.canonical_json(without_hash)
                )
                rows.append(row)
        runner.validate_degradation_rows(rows)
        leaked = [dict(row) for row in rows]
        leaked[0]["answer"] = True
        leaked[0]["row_sha256"] = inference.sha256_text(
            inference.canonical_json(
                {key: value for key, value in leaked[0].items() if key != "row_sha256"}
            )
        )
        with self.assertRaises(runner.BaselineRunError):
            runner.validate_degradation_rows(leaked)


@unittest.skipUnless(torch is not None, "PyTorch is unavailable in this local audit runtime")
class TensorAndBatchingTests(unittest.TestCase):
    def test_last_nonpadding_gather_supports_left_and_right_padding(self) -> None:
        logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        for mask, expected_indices in (
            (torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]), [3, 3]),
            (torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]), [1, 2]),
        ):
            with self.subTest(mask=mask.tolist()):
                indices = inference.last_nonpadding_indices(mask)
                gathered = inference.gather_last_token_logits(logits, mask)
                self.assertEqual(indices.tolist(), expected_indices)
                expected = torch.stack(
                    [logits[row, index] for row, index in enumerate(expected_indices)]
                )
                self.assertTrue(torch.equal(gathered, expected))

    def test_logits_to_keep_one_requires_last_position_attended(self) -> None:
        logits = torch.arange(2 * 1 * 3, dtype=torch.float32).reshape(2, 1, 3)
        left = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        self.assertTrue(
            torch.equal(inference.gather_last_token_logits(logits, left), logits[:, 0])
        )
        right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        with self.assertRaises(inference.InferenceContractError):
            inference.gather_last_token_logits(logits, right)

    def test_position_ids_are_derived_from_attention_mask(self) -> None:
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        expected = torch.tensor([[0, 0, 0, 1], [0, 1, 2, 3]])
        self.assertTrue(
            torch.equal(inference.position_ids_from_attention_mask(mask), expected)
        )

    def test_padding_helpers_reject_invalid_masks_and_shape_drift(self) -> None:
        for mask in (
            torch.zeros((1, 3), dtype=torch.long),
            torch.tensor([[0, 2, 1]], dtype=torch.long),
        ):
            with self.subTest(mask=mask.tolist()):
                with self.assertRaises(inference.InferenceContractError):
                    inference.last_nonpadding_indices(mask)
        with self.assertRaises(inference.InferenceContractError):
            inference.gather_last_token_logits(
                torch.zeros((2, 3, 4)), torch.ones((2, 2), dtype=torch.long)
            )

    def test_binary_probabilities_known_values_entropy_and_yes_tie(self) -> None:
        logits = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, math.log(3.0), 0.0, -2.0]],
            dtype=torch.float16,
        )
        result = inference.binary_confidence_from_logits(
            logits, yes_token_id=1, no_token_id=2
        )
        self.assertTrue(
            torch.allclose(
                result.p_yes_binary + result.p_no_binary,
                torch.ones(2),
                atol=1e-6,
            )
        )
        self.assertAlmostEqual(float(result.p_yes_binary[0]), 0.5, places=6)
        self.assertAlmostEqual(float(result.predictive_entropy[0]), math.log(2), places=6)
        self.assertEqual(int(result.token_prediction_index[0]), 0)
        self.assertAlmostEqual(float(result.p_yes_binary[1]), 0.75, places=4)
        self.assertTrue(torch.isfinite(result.predictive_entropy).all())

    def test_nonfinite_logits_and_invalid_class_ids_fail_closed(self) -> None:
        for nonfinite in (float("nan"), float("inf"), float("-inf")):
            bad = torch.zeros((1, 4))
            bad[0, 0] = nonfinite  # Deliberately outside the two class IDs.
            with self.subTest(nonfinite=nonfinite):
                with self.assertRaises(inference.InferenceContractError):
                    inference.binary_confidence_from_logits(
                        bad,
                        yes_token_id=1,
                        no_token_id=2,
                    )
        with self.assertRaises(inference.InferenceContractError):
            inference.binary_confidence_from_logits(
                torch.zeros((1, 4)), yes_token_id=4, no_token_id=2
            )

    def test_full_vocabulary_mass_is_separate_from_binary_normalization(self) -> None:
        logits = torch.tensor([[10.0, 0.0, 0.0, -1.0]])
        result = inference.binary_confidence_from_logits(
            logits, yes_token_id=1, no_token_id=2
        )
        self.assertAlmostEqual(float(result.p_yes_binary[0]), 0.5, places=6)
        self.assertAlmostEqual(float(result.p_no_binary[0]), 0.5, places=6)
        self.assertLess(float(result.yes_no_full_vocabulary_mass[0]), 0.001)
        self.assertAlmostEqual(
            float(result.yes_no_full_vocabulary_mass[0]),
            float(
                result.p_yes_full_vocabulary[0]
                + result.p_no_full_vocabulary[0]
            ),
            places=7,
        )

    def test_prompt_collator_left_pads_without_labels_or_reordering(self) -> None:
        from transformers import RepetitionPenaltyLogitsProcessor

        class FakeTokenizer:
            padding_side = "left"

            def __init__(self) -> None:
                self.calls: list[tuple[list[str], dict[str, object]]] = []

            def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, object]:
                self.calls.append((list(prompts), dict(kwargs)))
                return {
                    "input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]]),
                    "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
                }

        tokenizer = FakeTokenizer()
        examples = (
            runnable_example(
                ordinal=0,
                digest_character="a",
                prompt="short prompt",
                prompt_tokens=2,
            ),
            runnable_example(
                ordinal=1,
                digest_character="b",
                prompt="longer prompt here",
                prompt_tokens=3,
            ),
        )
        batch = inference.PromptCollator(
            tokenizer, maximum_prompt_tokens=1024
        )(examples)
        self.assertEqual(batch["examples"], examples)
        self.assertNotIn("labels", batch["model_inputs"])
        prompts, kwargs = tokenizer.calls[0]
        self.assertEqual(prompts, ["short prompt", "longer prompt here"])
        self.assertEqual(
            kwargs,
            {
                "add_special_tokens": False,
                "padding": True,
                "truncation": False,
                "return_tensors": "pt",
            },
        )
        scores = torch.ones((2, 64), dtype=torch.float32)
        padded_prompts_plus_one_generated_token = torch.tensor(
            [[0, 0, 10, 20, 30], [40, 41, 10, 20, 30]],
            dtype=torch.long,
        )
        processor = RepetitionPenaltyLogitsProcessor(
            penalty=1.1,
            prompt_ignore_length=4,
        )
        processed = processor(
            padded_prompts_plus_one_generated_token,
            scores,
        )
        self.assertTrue(torch.equal(processed[0], processed[1]))
        self.assertEqual(float(processed[0, 0]), 1.0)
        self.assertAlmostEqual(
            float(processed[0, 30]),
            1.0 / 1.1,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
