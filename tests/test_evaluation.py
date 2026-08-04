from __future__ import annotations

import copy
import hashlib
import json
import math
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

from llm_confidence_uq.evaluation import (  # noqa: E402
    CONDITIONS,
    PROTOCOL_VERSION,
    EvaluationContractError,
    build_evaluation_rows,
    build_metrics,
    error_detection_auroc,
    risk_coverage_rows,
    summarize_rows,
)


def probability(yes: float, no: float) -> tuple[float, float]:
    exp_yes, exp_no = math.exp(yes), math.exp(no)
    return exp_yes / (exp_yes + exp_no), exp_no / (exp_yes + exp_no)


def synthetic_inputs(method: str = "baseline") -> tuple[list[dict], list[dict]]:
    manifest = []
    predictions = []
    for index in range(400):
        answer = index % 2 == 0
        manifest.append({
            "answer": answer,
            "research_split": "test",
            "source_index": index,
            "source_split": "validation",
        })
        for condition_index, condition in enumerate(CONDITIONS):
            predicted_yes = answer if (index + condition_index) % 4 != 0 else not answer
            yes, no = (2.0, -2.0) if predicted_yes else (-2.0, 2.0)
            p_yes, p_no = probability(yes, no)
            base = {
                "class_order": ["Yes", "No"],
                "condition": condition,
                "condition_index": condition_index,
                "expressed_confidence": 90,
                "expressed_parser_reason_code": None,
                "expressed_parser_valid": True,
                "inference_status": "ok",
                "input_id": "boolqinput-" + hashlib.sha256(f"input-{index}".encode()).hexdigest(),
                "no_logit": no,
                "p_no_binary": p_no,
                "p_yes_binary": p_yes,
                "row_sha256": hashlib.sha256(f"core-{index}-{condition}".encode()).hexdigest(),
                "source_index": index,
                "source_split": "validation",
                "token_prediction": "Yes" if predicted_yes else "No",
                "yes_logit": yes,
            }
            if method != "baseline":
                base["adapter_stage"] = method
                base["adapter_row_sha256"] = hashlib.sha256(f"adapter-{method}-{index}-{condition}".encode()).hexdigest()
            predictions.append(base)
    return manifest, predictions


class JoinAndCalibrationTests(unittest.TestCase):
    def test_exact_join_and_positive_temperature_preserve_predictions(self) -> None:
        manifest, predictions = synthetic_inputs()
        rows = build_evaluation_rows(manifest, predictions, method="baseline", temperature=2.0, temperature_artifact_sha256="a" * 64)
        self.assertEqual(len(rows), 2400)
        self.assertEqual(len({row["input_id"] for row in rows}), 400)
        self.assertTrue(all(row["raw_confidence"] > row["calibrated_confidence"] for row in rows))
        self.assertTrue(all(row["token_prediction"] == ("Yes" if row["calibrated_p_yes"] >= row["calibrated_p_no"] else "No") for row in rows))

    def test_manifest_order_is_independent_and_source_keyed(self) -> None:
        manifest, predictions = synthetic_inputs()
        reordered = list(reversed(manifest))
        first = build_evaluation_rows(manifest, predictions, method="baseline", temperature=2.0, temperature_artifact_sha256="a" * 64)
        second = build_evaluation_rows(reordered, predictions, method="baseline", temperature=2.0, temperature_artifact_sha256="a" * 64)
        self.assertEqual(first, second)

    def test_adapter_identity_is_bound(self) -> None:
        manifest, predictions = synthetic_inputs("full_seed_1")
        rows = build_evaluation_rows(manifest, predictions, method="full_seed_1", temperature=2.0, temperature_artifact_sha256="b" * 64)
        self.assertEqual(rows[0]["method"], "full_seed_1")
        bad = copy.deepcopy(predictions)
        bad[0]["adapter_stage"] = "full_seed_2"
        with self.assertRaises(EvaluationContractError):
            build_evaluation_rows(manifest, bad, method="full_seed_1", temperature=2.0, temperature_artifact_sha256="b" * 64)

    def test_non_test_labels_and_prediction_label_leakage_fail_closed(self) -> None:
        manifest, predictions = synthetic_inputs()
        wrong = copy.deepcopy(manifest)
        wrong[0]["research_split"] = "calibration"
        with self.assertRaises(EvaluationContractError):
            build_evaluation_rows(wrong, predictions, method="baseline", temperature=2.0, temperature_artifact_sha256="a" * 64)
        leaked = copy.deepcopy(predictions)
        leaked[0]["ground_truth"] = "Yes"
        with self.assertRaises(EvaluationContractError):
            build_evaluation_rows(manifest, leaked, method="baseline", temperature=2.0, temperature_artifact_sha256="a" * 64)

    def test_condition_and_source_order_drift_fail_closed(self) -> None:
        manifest, predictions = synthetic_inputs()
        bad_condition = copy.deepcopy(predictions)
        bad_condition[0]["condition"] = "no_passage"
        with self.assertRaises(EvaluationContractError):
            build_evaluation_rows(manifest, bad_condition, method="baseline", temperature=2.0, temperature_artifact_sha256="a" * 64)
        bad_source = copy.deepcopy(predictions)
        bad_source[0]["source_index"] = 999
        with self.assertRaises(EvaluationContractError):
            build_evaluation_rows(manifest, bad_source, method="baseline", temperature=2.0, temperature_artifact_sha256="a" * 64)

    def test_nonpositive_temperature_fails_closed(self) -> None:
        manifest, predictions = synthetic_inputs()
        for value in (0.0, -1.0, float("nan")):
            with self.assertRaises(EvaluationContractError):
                build_evaluation_rows(manifest, predictions, method="baseline", temperature=value, temperature_artifact_sha256="a" * 64)


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest, predictions = synthetic_inputs()
        cls.rows = build_evaluation_rows(manifest, predictions, method="baseline", temperature=2.0, temperature_artifact_sha256="a" * 64)

    def test_metrics_are_finite_and_condition_complete(self) -> None:
        metrics = build_metrics(self.rows, method="baseline")
        self.assertEqual(set(metrics["by_condition"]), set(CONDITIONS))
        self.assertTrue(metrics["predictions_unchanged_by_calibration"])
        self.assertFalse(metrics["temperature_refit_on_test"])
        for group in [metrics["overall"], *metrics["by_condition"].values()]:
            for key in ("accuracy", "raw_nll", "calibrated_nll", "raw_brier", "calibrated_brier", "raw_ece_10_bin", "calibrated_ece_10_bin"):
                self.assertTrue(math.isfinite(group[key]))

    def test_auroc_known_perfect_ranking(self) -> None:
        rows = [
            {"raw_confidence": 0.9, "calibrated_confidence": 0.8, "correct": True},
            {"raw_confidence": 0.8, "calibrated_confidence": 0.7, "correct": True},
            {"raw_confidence": 0.4, "calibrated_confidence": 0.6, "correct": False},
            {"raw_confidence": 0.3, "calibrated_confidence": 0.5, "correct": False},
        ]
        self.assertEqual(error_detection_auroc(rows, "raw"), 1.0)
        self.assertEqual(error_detection_auroc(rows, "calibrated"), 1.0)

    def test_risk_coverage_is_deterministic_and_complete(self) -> None:
        first = risk_coverage_rows(self.rows, method="baseline")
        second = risk_coverage_rows(self.rows, method="baseline")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2 * (2400 + 6 * 400))
        self.assertEqual(first[2399]["coverage"], 1.0)

    def test_expressed_metrics_use_valid_rows_only(self) -> None:
        subset = [dict(row) for row in self.rows[:2]]
        subset[1]["expressed_parser_valid"] = False
        subset[1]["expressed_confidence"] = None
        subset[1]["expressed_raw_absolute_divergence"] = None
        subset[1]["expressed_calibrated_absolute_divergence"] = None
        summary = summarize_rows(subset)
        self.assertEqual(summary["expressed_parser_valid_rows"], 1)
        self.assertEqual(summary["expressed_parser_valid_rate"], 0.5)


class ConfigurationAndRunnerTests(unittest.TestCase):
    def test_configuration_hash_binds_implementation(self) -> None:
        text = (REPO / "configs/evaluation.yaml").read_text("utf-8")
        self.assertIn(f"protocol_version: {PROTOCOL_VERSION}\n", text)
        self.assertIn("  refit_on_test: false\n", text)
        block = text.split("implementation:\n", 1)[1].split("\noutput:\n", 1)[0]
        for line in block.splitlines():
            if line.strip():
                relative, expected = line.strip().rsplit(": ", 1)
                self.assertEqual(hashlib.sha256((REPO / relative).read_bytes()).hexdigest(), expected)

    def test_runner_import_defers_torch_and_hub_libraries(self) -> None:
        command = [sys.executable, "-c", ("import importlib.util,json,sys;" f"p={str(SCRIPTS / 'evaluate_predictions.py')!r};" "s=importlib.util.spec_from_file_location('eval_probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);" "print(json.dumps({k:(k in sys.modules) for k in ['torch','datasets','transformers','peft','yaml']}))")]
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"torch": False, "datasets": False, "transformers": False, "peft": False, "yaml": False})

    def test_runner_requires_explicit_method(self) -> None:
        result = subprocess.run([sys.executable, str(SCRIPTS / "evaluate_predictions.py")], cwd=REPO, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--method", result.stderr)


if __name__ == "__main__":
    unittest.main()
