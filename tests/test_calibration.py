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

from llm_confidence_uq.calibration import (  # noqa: E402
    CLASS_ORDER,
    PROTOCOL_VERSION,
    TemperatureScalingContractError,
    build_fit_inputs,
    fit_temperature,
    make_temperature_artifact,
)


def manifest_row(index: int, answer: bool) -> dict:
    return {
        "answer": answer,
        "record_sha256": hashlib.sha256(f"record-{index}".encode()).hexdigest(),
        "research_split": "calibration",
        "source_index": index,
        "source_split": "train",
    }


def prediction_row(index: int, method: str, logits: tuple[float, float]) -> dict:
    return {
        "class_order": list(CLASS_ORDER),
        "evidence_condition": "original",
        "input_id": "boolqinput-" + hashlib.sha256(f"input-{index}".encode()).hexdigest(),
        "method": method,
        "no_logit": logits[1],
        "row_sha256": hashlib.sha256(f"prediction-{index}".encode()).hexdigest(),
        "source_index": index,
        "source_split": "train",
        "yes_logit": logits[0],
    }


def synthetic_rows(method: str = "baseline") -> tuple[list[dict], list[dict]]:
    manifests = []
    predictions = []
    for index in range(200):
        answer = index % 2 == 0
        manifests.append(manifest_row(index, answer))
        # Correct direction but deliberately too sharp, plus regular errors.
        correct = index % 5 != 0
        predicted_yes = answer if correct else not answer
        logits = (6.0, -6.0) if predicted_yes else (-6.0, 6.0)
        predictions.append(prediction_row(index, method, logits))
    return manifests, predictions


class InputIsolationTests(unittest.TestCase):
    def test_build_inputs_uses_only_calibration_train_labels(self) -> None:
        manifests, predictions = synthetic_rows()
        inputs = build_fit_inputs(manifests, predictions, method="baseline")
        self.assertEqual(len(inputs.labels), 200)
        self.assertEqual(inputs.labels.count(0), 100)
        self.assertEqual(inputs.labels.count(1), 100)
        # Index 0 is one of the deliberately incorrect synthetic predictions.
        self.assertEqual(inputs.logits[0], (-6.0, 6.0))

    def test_test_or_validation_labels_fail_closed(self) -> None:
        manifests, predictions = synthetic_rows()
        for field, value in (("research_split", "test"), ("source_split", "validation")):
            candidate = copy.deepcopy(manifests)
            candidate[0][field] = value
            with self.assertRaises(TemperatureScalingContractError):
                build_fit_inputs(candidate, predictions, method="baseline")

    def test_prediction_label_leakage_and_order_drift_fail_closed(self) -> None:
        manifests, predictions = synthetic_rows()
        leaked = copy.deepcopy(predictions)
        leaked[0]["ground_truth"] = True
        with self.assertRaises(TemperatureScalingContractError):
            build_fit_inputs(manifests, leaked, method="baseline")
        misordered = copy.deepcopy(predictions)
        misordered[0]["source_index"] = 999
        with self.assertRaises(TemperatureScalingContractError):
            build_fit_inputs(manifests, misordered, method="baseline")

    def test_degraded_evidence_and_class_order_fail_closed(self) -> None:
        manifests, predictions = synthetic_rows()
        degraded = copy.deepcopy(predictions)
        degraded[0]["evidence_condition"] = "no_passage"
        with self.assertRaises(TemperatureScalingContractError):
            build_fit_inputs(manifests, degraded, method="baseline")
        reversed_classes = copy.deepcopy(predictions)
        reversed_classes[0]["class_order"] = ["No", "Yes"]
        with self.assertRaises(TemperatureScalingContractError):
            build_fit_inputs(manifests, reversed_classes, method="baseline")


class TorchTemperatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        cls.torch = torch

    def test_positive_temperature_reduces_nll_and_preserves_predictions(self) -> None:
        manifests, predictions = synthetic_rows()
        inputs = build_fit_inputs(manifests, predictions, method="baseline")
        result = fit_temperature(inputs, self.torch)
        self.assertGreater(result.temperature, 0.0)
        self.assertGreater(result.temperature, 1.0)
        self.assertLessEqual(result.final_nll, result.initial_nll)
        self.assertTrue(result.predictions_unchanged)

    def test_fit_is_deterministic(self) -> None:
        manifests, predictions = synthetic_rows("full_seed_1")
        inputs = build_fit_inputs(manifests, predictions, method="full_seed_1")
        first = fit_temperature(inputs, self.torch)
        second = fit_temperature(inputs, self.torch)
        self.assertEqual(first, second)

    def test_artifact_records_claim_boundaries(self) -> None:
        manifests, predictions = synthetic_rows()
        inputs = build_fit_inputs(manifests, predictions, method="baseline")
        result = fit_temperature(inputs, self.torch)
        artifact = make_temperature_artifact(
            inputs,
            result,
            calibration_manifest_sha256="a" * 64,
            prediction_jsonl_sha256="b" * 64,
            config_sha256="c" * 64,
        )
        self.assertEqual(artifact["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(artifact["class_order"], ["Yes", "No"])
        self.assertEqual(artifact["labels_used"], "calibration-only")
        self.assertFalse(artifact["test_labels_used"])
        self.assertFalse(artifact["expressed_confidence_used"])
        self.assertTrue(artifact["predictions_unchanged"])


class ConfigurationAndRunnerTests(unittest.TestCase):
    def test_configuration_binds_implementation_hashes(self) -> None:
        text = (REPO / "configs/temperature.yaml").read_text("utf-8")
        self.assertIn(f"protocol_version: {PROTOCOL_VERSION}\n", text)
        self.assertIn("  labels: calibration-only\n", text)
        self.assertIn("  test_labels_used: false\n", text)
        block = text.split("implementation:\n", 1)[1].split("\noutput:\n", 1)[0]
        for line in block.splitlines():
            if not line.strip():
                continue
            relative, expected = line.strip().rsplit(": ", 1)
            observed = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
            self.assertEqual(observed, expected)

    def test_runner_import_defers_torch_and_yaml(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import importlib.util,json,sys;"
                f"p={str(SCRIPTS / 'fit_temperature.py')!r};"
                "s=importlib.util.spec_from_file_location('temperature_probe',p);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "print(json.dumps({k:(k in sys.modules) for k in ['torch','yaml']}))"
            ),
        ]
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"torch": False, "yaml": False})

    def test_runner_requires_explicit_method(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "fit_temperature.py")],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--method", result.stderr)


if __name__ == "__main__":
    unittest.main()
