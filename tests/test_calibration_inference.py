from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from llm_confidence_uq.calibration_inference import (  # noqa: E402
    CalibrationExample,
    CalibrationInferenceContractError,
    PROTOCOL_VERSION,
    label_blind_projection_sha256,
    make_prediction_row,
    prediction_jsonl_bytes,
    reconstruct_original_examples,
    select_stage_rows,
    summarize_prediction_rows,
    validate_manifest_rows,
    validate_prediction_rows,
)
from llm_confidence_uq.degradations import canonical_json, sha256_text  # noqa: E402
from llm_confidence_uq.inference import ExpressedConfidenceParse  # noqa: E402


DATASET_ID = "google/boolq"
DATASET_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_row(index: int, *, answer: bool | None = None) -> tuple[dict, dict]:
    if answer is None:
        answer = bool(index % 2)
    source = {
        "question": f"question {index}?",
        "passage": f"passage {index}",
        "answer": answer,
    }
    record_sha256 = sha256_text(canonical_json(source))
    input_sha256 = sha256_text(
        canonical_json({"passage": source["passage"], "question": source["question"]})
    )
    prompt = f"PASSAGE={source['passage']} QUESTION={source['question']}"
    row = {
        "answer": answer,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "eligibility_max_prompt_tokens": 768,
        "example_id": f"boolq-train-{index:05d}-{record_sha256[:12]}",
        "input_sha256": input_sha256,
        "original_passage_sha256": sha256_text(source["passage"]),
        "prompt_tokens": len(prompt.encode("utf-8")),
        "record_sha256": record_sha256,
        "research_split": "calibration",
        "schema_version": 1,
        "selection_rank_sha256": sha256_text(f"rank-{index}"),
        "selection_seed": 20260803,
        "source_index": index,
        "source_split": "train",
    }
    return row, source


def validate(rows: list[dict]) -> None:
    validate_manifest_rows(
        rows,
        expected_rows=len(rows),
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        selection_seed=20260803,
        eligibility_max_prompt_tokens=768,
    )


class ManifestAndSelectionTests(unittest.TestCase):
    def test_manifest_contract_accepts_only_calibration_train_rows(self) -> None:
        rows = [make_row(index)[0] for index in range(4)]
        validate(rows)
        for field, bad_value in (
            ("research_split", "test"),
            ("source_split", "validation"),
            ("dataset_revision", "0" * 40),
        ):
            candidate = copy.deepcopy(rows)
            candidate[0][field] = bad_value
            with self.assertRaises(CalibrationInferenceContractError):
                validate(candidate)

    def test_duplicate_source_location_and_schema_drift_fail_closed(self) -> None:
        rows = [make_row(index)[0] for index in range(4)]
        duplicate = copy.deepcopy(rows)
        duplicate[1]["source_index"] = duplicate[0]["source_index"]
        with self.assertRaises(CalibrationInferenceContractError):
            validate(duplicate)
        extra = copy.deepcopy(rows)
        extra[0]["ground_truth"] = True
        with self.assertRaises(CalibrationInferenceContractError):
            validate(extra)

    def test_stage_selection_is_ordered_and_label_blind(self) -> None:
        rows = [make_row(index)[0] for index in range(200)]
        self.assertEqual(select_stage_rows(rows, "smoke8"), rows[:8])
        self.assertEqual(select_stage_rows(rows, "full"), rows)
        flipped = copy.deepcopy(rows)
        for row in flipped:
            row["answer"] = not row["answer"]
            row["record_sha256"] = "f" * 64
        self.assertEqual(
            label_blind_projection_sha256(rows),
            label_blind_projection_sha256(flipped),
        )

    def test_unknown_or_oversized_stage_fails_closed(self) -> None:
        rows = [make_row(index)[0] for index in range(8)]
        with self.assertRaises(CalibrationInferenceContractError):
            select_stage_rows(rows, "unknown")
        with self.assertRaises(CalibrationInferenceContractError):
            select_stage_rows(rows, "full")


class ReconstructionTests(unittest.TestCase):
    @staticmethod
    def render(passage: str, question: str) -> str:
        return f"PASSAGE={passage} QUESTION={question}"

    @staticmethod
    def encode(prompt: str) -> list[int]:
        return list(prompt.encode("utf-8"))

    @staticmethod
    def input_id(**kwargs) -> str:
        return "boolqinput-" + sha256_text(canonical_json(kwargs))

    def test_reconstruction_is_deterministic_and_structurally_label_free(self) -> None:
        pairs = [make_row(index) for index in range(3)]
        rows = [pair[0] for pair in pairs]
        dataset = [pair[1] for pair in pairs]
        first = reconstruct_original_examples(
            rows,
            dataset,
            render_prompt=self.render,
            encode_prompt=self.encode,
            make_input_id=self.input_id,
            dataset_revision=DATASET_REVISION,
        )
        second = reconstruct_original_examples(
            rows,
            dataset,
            render_prompt=self.render,
            encode_prompt=self.encode,
            make_input_id=self.input_id,
            dataset_revision=DATASET_REVISION,
        )
        self.assertEqual(first, second)
        self.assertEqual([item.ordinal for item in first], [0, 1, 2])
        self.assertTrue(all(item.source_split == "train" for item in first))
        self.assertTrue(all("answer" not in item.__dict__ for item in first))
        self.assertTrue(all("label" not in item.__dict__ for item in first))

    def test_source_label_and_prompt_drift_fail_closed(self) -> None:
        row, source = make_row(0)
        bad_source = dict(source)
        bad_source["answer"] = not source["answer"]
        with self.assertRaises(CalibrationInferenceContractError):
            reconstruct_original_examples(
                [row],
                [bad_source],
                render_prompt=self.render,
                encode_prompt=self.encode,
                make_input_id=self.input_id,
                dataset_revision=DATASET_REVISION,
            )
        bad_row = dict(row)
        bad_row["prompt_tokens"] += 1
        with self.assertRaises(CalibrationInferenceContractError):
            reconstruct_original_examples(
                [bad_row],
                [source],
                render_prompt=self.render,
                encode_prompt=self.encode,
                make_input_id=self.input_id,
                dataset_revision=DATASET_REVISION,
            )


class ConfigurationAndRunnerTests(unittest.TestCase):
    def test_config_is_exact_and_hash_binds_implementation(self) -> None:
        text = (REPO / "configs/calibration_inference.yaml").read_text("utf-8")
        self.assertIn(f"protocol_version: {PROTOCOL_VERSION}\n", text)
        self.assertIn("  research_split: calibration\n", text)
        self.assertIn("  evidence_condition: original\n", text)
        self.assertIn("  labels_available_to_model_inference: false\n", text)
        self.assertIn("  rows: 200\n", text)
        for method in ("baseline", "full_seed_1", "full_seed_2", "full_seed_3"):
            self.assertIn(f"  {method}:\n", text)
        implementation_block = text.split("implementation:\n", 1)[1].split("\noutput:\n", 1)[0]
        entries = {}
        for line in implementation_block.splitlines():
            if not line.strip():
                continue
            relative, expected = line.strip().rsplit(": ", 1)
            entries[relative] = expected
        for relative, expected in entries.items():
            self.assertEqual(file_sha256(REPO / relative), expected)

    def test_runner_import_defers_hub_model_and_adapter_libraries(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import importlib.util, json; "
                f"p={str(SCRIPTS / 'run_calibration_inference.py')!r}; "
                "s=importlib.util.spec_from_file_location('calibration_runner_probe', p); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "import sys; print(json.dumps({k:(k in sys.modules) for k in ['torch','datasets','transformers','peft']}))"
            ),
        ]
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        imported = json.loads(result.stdout)
        self.assertIsInstance(imported["torch"], bool)
        self.assertEqual(
            {key: imported[key] for key in ("datasets", "transformers", "peft")},
            {"datasets": False, "transformers": False, "peft": False},
        )

    def test_runner_cli_requires_explicit_method_and_stage(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "run_calibration_inference.py")],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--method", result.stderr)
        self.assertIn("--stage", result.stderr)


class PredictionSchemaTests(unittest.TestCase):
    def make_prediction(self, method: str) -> dict:
        example = CalibrationExample(0, "boolqinput-" + "a" * 64, "train", 9, "prompt", 1, sha256_text("prompt"))
        adapter = None if method == "baseline" else {
            "adapter_stage": method,
            "adapter_training_seed": 20260811,
            "adapter_checkpoint_ledger_sha256": "b" * 64,
            "adapter_weights_sha256": "c" * 64,
            "adapter_semantic_config_sha256": "d" * 64,
        }
        return make_prediction_row(
            example, method=method, model_id="model", model_revision="e" * 40,
            inference_config_sha256="f" * 64, inference_seed=1, yes_token_id=1,
            no_token_id=2, yes_logit=2.0, no_logit=1.0, p_yes_binary=0.75,
            p_no_binary=0.25, p_yes_full_vocabulary=0.6,
            p_no_full_vocabulary=0.2, yes_no_full_vocabulary_mass=0.8,
            token_prediction="Yes", token_confidence=0.75,
            predictive_entropy=0.56, generated_continuation=" Yes\nConfidence: 75",
            generated_continuation_tokens=5, generation_ended_with_eos=True,
            generation_reached_max_new_tokens=False,
            parsed=ExpressedConfidenceParse("exact-continuation-v1", True, "Yes", 75, None),
            adapter=adapter,
        )

    def test_prediction_schema_is_label_free_and_deterministic(self) -> None:
        row = self.make_prediction("baseline")
        validate_prediction_rows([row])
        self.assertTrue({"answer", "label", "correct", "target"}.isdisjoint(row))
        self.assertEqual(prediction_jsonl_bytes([row]), prediction_jsonl_bytes([row]))
        self.assertFalse(summarize_prediction_rows([row])["contains_ground_truth"])

    def test_adapter_identity_and_row_hash_mutations_fail_closed(self) -> None:
        row = self.make_prediction("full_seed_1")
        validate_prediction_rows([row])
        bad = dict(row)
        bad["p_yes_binary"] = 0.9
        with self.assertRaises(CalibrationInferenceContractError):
            validate_prediction_rows([bad])


if __name__ == "__main__":
    unittest.main()
