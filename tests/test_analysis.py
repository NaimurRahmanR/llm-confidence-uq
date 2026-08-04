from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import yaml


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq.analysis import (
    ALL_VARIANTS,
    CONDITIONS,
    FIGURES,
    METHODS,
    PRIMARY_VARIANTS,
    PROTOCOL_VERSION,
    AnalysisError,
    _normalized_row,
    build_expressed_summary,
    build_method_metrics,
    build_normalized_rows,
    build_paired_changes,
    build_reliability_table,
    build_risk_coverage,
    canonical_json,
    entropy,
    error_detection_auroc,
    normalize_ensemble_rows,
    normalize_evaluation_rows,
    normalize_laplace_rows,
    reliability_bins,
    sha256_text,
    summarize_rows,
)


def bind(row: dict) -> dict:
    result = dict(row)
    result["row_sha256"] = sha256_text(canonical_json(result))
    return result


def source_coordinates(ordinal: int) -> tuple[int, int, str, str]:
    input_index, condition_index = divmod(ordinal, len(CONDITIONS))
    condition = CONDITIONS[condition_index]
    input_id = f"boolqinput-{input_index:04d}"
    return input_index, condition_index, condition, input_id


def evaluation_rows(method: str) -> list[dict]:
    rows = []
    for ordinal in range(2400):
        input_index, condition_index, condition, input_id = source_coordinates(ordinal)
        truth = "Yes" if input_index % 2 == 0 else "No"
        raw_p_yes = 0.7 if truth == "Yes" else 0.3
        calibrated_p_yes = 0.62 if truth == "Yes" else 0.38
        valid = (input_index + condition_index) % 4 != 0
        expressed = 80 if valid else None
        rows.append(bind({
            "schema_version": 1,
            "protocol_version": "boolq-calibrated-test-evaluation-v1",
            "method": method,
            "evaluation_id": f"evaluation-{method}-{ordinal}",
            "input_id": input_id,
            "source_index": input_index,
            "condition": condition,
            "condition_index": condition_index,
            "ground_truth": truth,
            "token_prediction": truth,
            "correct": True,
            "raw_p_yes": raw_p_yes,
            "raw_p_no": 1.0 - raw_p_yes,
            "calibrated_p_yes": calibrated_p_yes,
            "calibrated_p_no": 1.0 - calibrated_p_yes,
            "raw_confidence": max(raw_p_yes, 1.0 - raw_p_yes),
            "calibrated_confidence": max(calibrated_p_yes, 1.0 - calibrated_p_yes),
            "raw_entropy": entropy(raw_p_yes),
            "calibrated_entropy": entropy(calibrated_p_yes),
            "expressed_parser_valid": valid,
            "expressed_parser_reason_code": "valid" if valid else "format_mismatch",
            "expressed_confidence": expressed,
            "expressed_raw_absolute_divergence": abs(0.8 - max(raw_p_yes, 1.0 - raw_p_yes)) if valid else None,
            "expressed_calibrated_absolute_divergence": abs(0.8 - max(calibrated_p_yes, 1.0 - calibrated_p_yes)) if valid else None,
        }))
    return rows


def ensemble_rows() -> list[dict]:
    rows = []
    for ordinal in range(2400):
        input_index, condition_index, condition, input_id = source_coordinates(ordinal)
        truth = "Yes" if input_index % 2 == 0 else "No"
        raw_p_yes = 0.72 if truth == "Yes" else 0.28
        calibrated_p_yes = 0.64 if truth == "Yes" else 0.36
        rows.append(bind({
            "schema_version": 1,
            "protocol_version": "boolq-three-lora-ensemble-v1",
            "ensemble_id": f"ensemble-{ordinal}",
            "input_id": input_id,
            "source_index": input_index,
            "condition": condition,
            "condition_index": condition_index,
            "ground_truth": truth,
            "members": [{"method": method} for method in METHODS[1:]],
            "raw_ensemble_p_yes": raw_p_yes,
            "raw_ensemble_p_no": 1.0 - raw_p_yes,
            "raw_predictive_entropy": entropy(raw_p_yes),
            "raw_expected_entropy": entropy(raw_p_yes) - 0.02,
            "raw_mutual_information_style": 0.02,
            "calibrated_ensemble_p_yes": calibrated_p_yes,
            "calibrated_ensemble_p_no": 1.0 - calibrated_p_yes,
            "calibrated_predictive_entropy": entropy(calibrated_p_yes),
            "calibrated_expected_entropy": entropy(calibrated_p_yes) - 0.01,
            "calibrated_mutual_information_style": 0.01,
        }))
    return rows


def laplace_rows() -> list[dict]:
    rows = []
    for ordinal in range(2400):
        input_index, condition_index, condition, input_id = source_coordinates(ordinal)
        truth = "Yes" if input_index % 2 == 0 else "No"
        p_yes = 0.66 if truth == "Yes" else 0.34
        predictive = entropy(p_yes)
        rows.append(bind({
            "schema_version": 1,
            "protocol_version": "boolq-frozen-qwen-diagonal-laplace-head-v1",
            "prediction_id": f"laplace-{ordinal}",
            "example_id": input_id,
            "source_index": input_index,
            "condition": condition,
            "condition_index": condition_index,
            "ground_truth": truth,
            "posterior_mean_p_yes": p_yes,
            "posterior_mean_p_no": 1.0 - p_yes,
            "predictive_entropy": predictive,
            "expected_entropy": predictive - 0.03,
            "mutual_information": 0.03,
            "full_transformer_is_bayesian": False,
        }))
    return rows


class CoreMetricTests(unittest.TestCase):
    def test_entropy_known_boundaries(self) -> None:
        self.assertEqual(entropy(0.0), 0.0)
        self.assertAlmostEqual(entropy(0.5), math.log(2.0), places=14)
        with self.assertRaisesRegex(AnalysisError, "invalid"):
            entropy(1.01)

    def test_normalized_row_binds_scope_and_entropy_decomposition(self) -> None:
        row = _normalized_row(
            variant="laplace_head",
            input_id="x",
            source_index=1,
            condition="original",
            condition_index=0,
            ground_truth="Yes",
            p_yes=0.7,
            predictive_entropy=0.6,
            expected_entropy=0.5,
            mutual_information=0.1,
            source_row_sha256="a" * 64,
        )
        self.assertTrue(row["correct"])
        self.assertAlmostEqual(row["p_no"], 0.3, places=15)
        with self.assertRaisesRegex(AnalysisError, "decomposition"):
            _normalized_row(
                variant="laplace_head", input_id="x", source_index=1,
                condition="original", condition_index=0, ground_truth="Yes",
                p_yes=0.7, predictive_entropy=0.6, expected_entropy=0.5,
                mutual_information=0.2, source_row_sha256="a" * 64,
            )

    def test_reliability_bins_use_locked_open_closed_boundaries(self) -> None:
        rows = [
            {"confidence": 0.5, "correct": True},
            {"confidence": 0.6, "correct": False},
            {"confidence": 1.0, "correct": True},
        ]
        bins = reliability_bins(rows, 10)
        self.assertEqual(sum(row["count"] for row in bins), 3)
        self.assertEqual(bins[4]["count"], 1)
        self.assertEqual(bins[5]["count"], 1)
        self.assertEqual(bins[9]["count"], 1)

    def test_error_detection_auroc_handles_perfect_order_and_ties(self) -> None:
        perfect = [
            {"confidence": 0.9, "correct": True},
            {"confidence": 0.8, "correct": True},
            {"confidence": 0.6, "correct": False},
            {"confidence": 0.5, "correct": False},
        ]
        self.assertEqual(error_detection_auroc(perfect), 1.0)
        tied = [{"confidence": 0.7, "correct": True}, {"confidence": 0.7, "correct": False}]
        self.assertEqual(error_detection_auroc(tied), 0.5)

    def test_summary_metrics_match_known_binary_values(self) -> None:
        rows = []
        for index, (truth, p_yes) in enumerate((("Yes", 0.8), ("No", 0.2), ("Yes", 0.4), ("No", 0.6))):
            rows.append(_normalized_row(
                variant="baseline_raw", input_id=str(index), source_index=index,
                condition="original", condition_index=0, ground_truth=truth,
                p_yes=p_yes, predictive_entropy=entropy(p_yes), source_row_sha256=f"{index:064d}",
            ))
        summary = summarize_rows(rows)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertAlmostEqual(summary["brier"], 0.2, places=14)
        self.assertTrue(math.isfinite(summary["nll"]))


class SourceNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluation = evaluation_rows("baseline")
        cls.ensemble = ensemble_rows()
        cls.laplace = laplace_rows()

    def test_evaluation_normalization_retains_invalid_expressed_rows(self) -> None:
        normalized, expressed = normalize_evaluation_rows("baseline", self.evaluation)
        self.assertEqual(len(normalized), 4800)
        self.assertEqual(len(expressed), 2400)
        invalid = [row for row in expressed if not row["parser_valid"]]
        self.assertTrue(invalid)
        self.assertTrue(all(row["expressed_confidence"] is None for row in invalid))

    def test_source_hash_mutation_fails_closed(self) -> None:
        changed = [dict(row) for row in self.evaluation]
        changed[0]["raw_p_yes"] = 0.9
        with self.assertRaisesRegex(AnalysisError, "hash drift"):
            normalize_evaluation_rows("baseline", changed)

    def test_ensemble_normalization_retains_nonposterior_entropy_gap(self) -> None:
        normalized = normalize_ensemble_rows(self.ensemble)
        self.assertEqual(len(normalized), 4800)
        self.assertTrue(all(row["mutual_information"] is not None for row in normalized))

    def test_laplace_scope_overclaim_fails_closed(self) -> None:
        changed = [dict(row) for row in self.laplace]
        changed[0]["full_transformer_is_bayesian"] = True
        changed[0] = bind({key: value for key, value in changed[0].items() if key != "row_sha256"})
        with self.assertRaisesRegex(AnalysisError, "overclaim"):
            normalize_laplace_rows(changed)


class UnifiedTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        evaluations = {method: evaluation_rows(method) for method in METHODS}
        cls.normalized, cls.expressed = build_normalized_rows(evaluations, ensemble_rows(), laplace_rows())

    def test_all_variants_are_complete_and_aligned(self) -> None:
        self.assertEqual(len(self.normalized), len(ALL_VARIANTS) * 2400)
        self.assertEqual(Counter(row["variant"] for row in self.normalized), Counter({variant: 2400 for variant in ALL_VARIANTS}))
        self.assertEqual(len({row["analysis_id"] for row in self.normalized}), len(self.normalized))

    def test_metric_reliability_and_risk_tables_have_locked_cardinality(self) -> None:
        metrics = build_method_metrics(self.normalized)
        reliability = build_reliability_table(self.normalized)
        risk = build_risk_coverage(self.normalized)
        self.assertEqual(len(metrics), len(ALL_VARIANTS) * 7)
        self.assertEqual(len(reliability), len(PRIMARY_VARIANTS) * 10)
        self.assertEqual(len(risk), len(PRIMARY_VARIANTS) * 2400)
        self.assertTrue(all(abs(row["risk"] + row["selective_accuracy"] - 1.0) <= 1e-15 for row in risk))

    def test_paired_and_expressed_summaries_are_complete(self) -> None:
        paired = build_paired_changes(self.normalized)
        expressed = build_expressed_summary(self.expressed)
        self.assertEqual(len(paired), len(ALL_VARIANTS) * len(CONDITIONS))
        self.assertEqual(len(expressed), len(METHODS) * 7)
        self.assertTrue(all(row["invalid_rows_retained_without_imputation"] >= 0 for row in expressed))


class ContractTests(unittest.TestCase):
    def test_configuration_binds_inputs_implementation_figures_and_claims(self) -> None:
        config = yaml.safe_load((REPO / "configs/results.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(tuple(config["expected"]["figures"]), FIGURES)
        self.assertEqual(tuple(config["expected"]["primary_variants"]), PRIMARY_VARIANTS)
        self.assertFalse(config["claim_boundaries"]["full_transformer_is_bayesian"])
        self.assertFalse(config["claim_boundaries"]["ensemble_members_are_posterior_samples"])
        self.assertTrue(config["claim_boundaries"]["post_hoc_test_results_are_descriptive_not_model_selection"])
        for relative, expected in config["implementation"].items():
            self.assertNotEqual(expected, "PENDING")
            self.assertEqual(hashlib.sha256((REPO / relative).read_bytes()).hexdigest(), expected)

    def test_runner_import_defers_matplotlib_and_model_libraries(self) -> None:
        before = set(sys.modules)
        script = REPO / "scripts/build_results.py"
        specification = importlib.util.spec_from_file_location("results_runner_test", script)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        added = set(sys.modules) - before
        self.assertFalse(any(name.startswith(("matplotlib", "transformers", "datasets", "peft")) for name in added))


if __name__ == "__main__":
    unittest.main()
