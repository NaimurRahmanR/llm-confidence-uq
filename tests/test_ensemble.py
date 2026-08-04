from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq.ensemble import (
    CONDITIONS,
    MEMBERS,
    PROTOCOL_VERSION,
    SOURCE_PROTOCOL_VERSION,
    EnsembleError,
    aggregate_rows,
    build_metrics,
    canonical_json,
    entropy,
    error_detection_auroc,
    risk_coverage_rows,
    sha256_text,
)


ADAPTERS = {
    "full_seed_1": "3bcd570b3ec3616ecaf3b7cc874a81fff84b3899334028d4d8179a367aea35b4",
    "full_seed_2": "0e393dc7cddf1f7183c17af5995d519de016f46d6151c5465f81845bb060e691",
    "full_seed_3": "a665864d76931f59650dec517ce66632233117685b690bde17c6138911dae46c",
}
SEEDS = {"full_seed_1": 20260811, "full_seed_2": 20260812, "full_seed_3": 20260813}


def metadata() -> dict[str, dict[str, object]]:
    return {
        method: {"method": method, "seed": SEEDS[method], "adapter_sha256": ADAPTERS[method]}
        for method in MEMBERS
    }


def source_row(
    method: str,
    *,
    input_id: str = "input-1",
    source_index: int = 7,
    condition: str = "original",
    ground_truth: str = "Yes",
    raw_p_yes: float = 0.8,
    calibrated_p_yes: float = 0.7,
) -> dict[str, object]:
    prediction = "Yes" if raw_p_yes >= 0.5 else "No"
    calibrated_prediction = "Yes" if calibrated_p_yes >= 0.5 else "No"
    if calibrated_prediction != prediction:
        raise ValueError("synthetic member calibration cannot change class")
    row: dict[str, object] = {
        "schema_version": 1,
        "protocol_version": SOURCE_PROTOCOL_VERSION,
        "evaluation_id": f"evaluation-{method}-{input_id}-{condition}",
        "method": method,
        "input_id": input_id,
        "source_index": source_index,
        "condition": condition,
        "condition_index": CONDITIONS.index(condition),
        "class_order": ["Yes", "No"],
        "ground_truth": ground_truth,
        "token_prediction": prediction,
        "correct": prediction == ground_truth,
        "yes_logit": 1.0,
        "no_logit": 0.0,
        "raw_p_yes": raw_p_yes,
        "raw_p_no": 1.0 - raw_p_yes,
        "calibrated_p_yes": calibrated_p_yes,
        "calibrated_p_no": 1.0 - calibrated_p_yes,
        "raw_confidence": max(raw_p_yes, 1.0 - raw_p_yes),
        "calibrated_confidence": max(calibrated_p_yes, 1.0 - calibrated_p_yes),
        "raw_entropy": entropy(raw_p_yes),
        "calibrated_entropy": entropy(calibrated_p_yes),
        "temperature": 2.0,
        "temperature_artifact_sha256": "a" * 64,
        "source_prediction_row_sha256": sha256_text(f"{method}-{input_id}-{condition}"),
        "expressed_parser_valid": False,
        "expressed_parser_reason_code": "format_mismatch",
        "expressed_confidence": None,
        "expressed_raw_absolute_divergence": None,
        "expressed_calibrated_absolute_divergence": None,
    }
    row["row_sha256"] = sha256_text(canonical_json(row))
    return row


class AggregationTests(unittest.TestCase):
    def test_probability_mean_population_variance_and_entropy_decomposition(self) -> None:
        raw = (0.9, 0.7, 0.5)
        calibrated = (0.8, 0.6, 0.5)
        rows = {
            method: [source_row(method, raw_p_yes=raw[index], calibrated_p_yes=calibrated[index])]
            for index, method in enumerate(MEMBERS)
        }
        result = aggregate_rows(rows, metadata())[0]
        self.assertAlmostEqual(result["raw_ensemble_p_yes"], 0.7)
        self.assertAlmostEqual(result["raw_member_probability_variance"], 0.02666666666666667)
        expected_entropy = sum(entropy(value) for value in raw) / 3.0
        self.assertAlmostEqual(result["raw_expected_entropy"], expected_entropy)
        self.assertAlmostEqual(
            result["raw_mutual_information_style"],
            entropy(0.7) - expected_entropy,
        )
        self.assertGreaterEqual(result["raw_mutual_information_style"], 0.0)
        self.assertEqual(result["vote_counts"], {"Yes": 3, "No": 0})
        self.assertFalse(result["any_vote_disagreement"])

    def test_vote_disagreement_is_reported_not_required(self) -> None:
        probabilities = (0.8, 0.7, 0.2)
        rows = {
            method: [source_row(method, raw_p_yes=value, calibrated_p_yes=value)]
            for method, value in zip(MEMBERS, probabilities)
        }
        result = aggregate_rows(rows, metadata())[0]
        self.assertTrue(result["any_vote_disagreement"])
        self.assertAlmostEqual(result["vote_disagreement_rate"], 1.0 / 3.0)
        self.assertEqual(result["vote_counts"], {"Yes": 2, "No": 1})

    def test_member_order_seed_and_checkpoint_uniqueness_fail_closed(self) -> None:
        rows = {method: [source_row(method)] for method in MEMBERS}
        reversed_rows = dict(reversed(tuple(rows.items())))
        with self.assertRaisesRegex(EnsembleError, "locked order"):
            aggregate_rows(reversed_rows, metadata())
        duplicate_seed = metadata()
        duplicate_seed["full_seed_3"]["seed"] = 20260811
        with self.assertRaisesRegex(EnsembleError, "seeds are not unique"):
            aggregate_rows(rows, duplicate_seed)
        duplicate_adapter = metadata()
        duplicate_adapter["full_seed_3"]["adapter_sha256"] = ADAPTERS["full_seed_1"]
        with self.assertRaisesRegex(EnsembleError, "checkpoints are not unique"):
            aggregate_rows(rows, duplicate_adapter)

    def test_alignment_hash_and_distinct_prediction_identity_fail_closed(self) -> None:
        rows = {method: [source_row(method)] for method in MEMBERS}
        rows["full_seed_2"][0]["source_index"] = 8
        rows["full_seed_2"][0]["row_sha256"] = sha256_text(canonical_json({k: v for k, v in rows["full_seed_2"][0].items() if k != "row_sha256"}))
        with self.assertRaisesRegex(EnsembleError, "alignment drift"):
            aggregate_rows(rows, metadata())

        rows = {method: [source_row(method)] for method in MEMBERS}
        rows["full_seed_2"][0]["raw_p_yes"] = 0.9
        rows["full_seed_2"][0]["raw_p_no"] = 0.1
        with self.assertRaisesRegex(EnsembleError, "hash drift"):
            aggregate_rows(rows, metadata())

        rows = {method: [source_row(method)] for method in MEMBERS}
        shared = rows["full_seed_1"][0]["source_prediction_row_sha256"]
        rows["full_seed_2"][0]["source_prediction_row_sha256"] = shared
        rows["full_seed_2"][0]["row_sha256"] = sha256_text(canonical_json({k: v for k, v in rows["full_seed_2"][0].items() if k != "row_sha256"}))
        with self.assertRaisesRegex(EnsembleError, "identities are not distinct"):
            aggregate_rows(rows, metadata())

    def test_non_normalized_nonfinite_and_class_changing_inputs_fail_closed(self) -> None:
        for field, value, complement_field, complement_value, message in (
            ("raw_p_no", 0.3, None, None, "do not normalize"),
            ("raw_p_yes", float("nan"), None, None, "invalid raw Yes"),
            ("calibrated_p_yes", 0.4, "calibrated_p_no", 0.6, "changed class"),
        ):
            rows = {method: [source_row(method)] for method in MEMBERS}
            rows["full_seed_1"][0][field] = value
            if complement_field is not None:
                rows["full_seed_1"][0][complement_field] = complement_value
            rows["full_seed_1"][0]["row_sha256"] = sha256_text(canonical_json({k: v for k, v in rows["full_seed_1"][0].items() if k != "row_sha256"})) if math.isfinite(value) else "f" * 64
            with self.assertRaisesRegex(EnsembleError, message):
                aggregate_rows(rows, metadata())


class MetricTests(unittest.TestCase):
    def test_error_detection_auroc_known_ranking(self) -> None:
        rows = [
            {"raw_ensemble_correct": False, "raw_ensemble_confidence": 0.55},
            {"raw_ensemble_correct": False, "raw_ensemble_confidence": 0.60},
            {"raw_ensemble_correct": True, "raw_ensemble_confidence": 0.80},
            {"raw_ensemble_correct": True, "raw_ensemble_confidence": 0.90},
        ]
        self.assertEqual(error_detection_auroc(rows, "raw"), 1.0)

    def test_complete_metrics_accept_zero_disagreement_honestly(self) -> None:
        member_rows = {method: [] for method in MEMBERS}
        for input_index in range(400):
            for condition in CONDITIONS:
                for method in MEMBERS:
                    member_rows[method].append(source_row(
                        method,
                        input_id=f"input-{input_index:04d}",
                        source_index=input_index,
                        condition=condition,
                        raw_p_yes=0.8,
                        calibrated_p_yes=0.7,
                    ))
        rows = aggregate_rows(member_rows, metadata())
        metrics = build_metrics(rows)
        self.assertEqual(metrics["rows"], 2400)
        self.assertEqual(metrics["raw"]["overall"]["accuracy"], 1.0)
        self.assertEqual(metrics["raw"]["overall"]["vote_disagreement_rows"], 0)
        self.assertEqual(metrics["raw"]["overall"]["mean_mutual_information_style"], 0.0)
        self.assertFalse(metrics["test_labels_used_for_fitting_or_tuning"])
        curves = risk_coverage_rows(rows[:6])
        self.assertEqual(len(curves), 24)
        self.assertEqual(curves[-1]["coverage"], 1.0)


class ConfigurationAndRunnerTests(unittest.TestCase):
    def test_configuration_binds_implementation_and_three_distinct_members(self) -> None:
        config_path = REPO / "configs/ensemble.yaml"
        source = config_path.read_text(encoding="utf-8")
        module_path = "src/llm_confidence_uq/ensemble.py"
        module_hash = hashlib.sha256((REPO / module_path).read_bytes()).hexdigest()
        self.assertIn(f"protocol_version: {PROTOCOL_VERSION}", source)
        self.assertIn(f"module_path: {module_path}", source)
        self.assertIn(f"module_sha256: {module_hash}", source)
        for method, seed in SEEDS.items():
            self.assertIn(f"  {method}:\n    method: {method}\n    seed: {seed}", source)
            self.assertIn(f"adapter_sha256: {ADAPTERS[method]}", source)
        self.assertIn("members_are_posterior_samples: false", source)
        self.assertIn("full_transformer_is_bayesian: false", source)

    def test_runner_import_defers_gpu_and_model_libraries(self) -> None:
        before = set(sys.modules)
        script = REPO / "scripts/evaluate_ensemble.py"
        specification = importlib.util.spec_from_file_location("ensemble_runner_test", script)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        added = set(sys.modules) - before
        self.assertFalse(any(name == "torch" or name.startswith("transformers") or name.startswith("peft") for name in added))


if __name__ == "__main__":
    unittest.main()
