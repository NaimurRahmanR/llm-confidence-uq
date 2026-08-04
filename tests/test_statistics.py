from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq.analysis import CONDITIONS, PRIMARY_VARIANTS, canonical_json, entropy  # noqa: E402
from llm_confidence_uq.statistics import (  # noqa: E402
    binary_auroc,
    binary_average_precision,
    build_clustered_bootstrap_differences,
    build_lora_seed_summary,
    build_uq_signal_evaluations,
    validate_statistical_outputs,
)


def synthetic_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    error_limits = {
        "baseline_raw": 120,
        "baseline_calibrated": 120,
        "full_seed_1_calibrated": 70,
        "full_seed_2_calibrated": 60,
        "full_seed_3_calibrated": 80,
        "three_lora_ensemble_calibrated": 65,
        "laplace_head": 100,
    }
    normalized: list[dict] = []
    ensemble: list[dict] = []
    laplace: list[dict] = []
    for input_index in range(400):
        input_id = f"input-{input_index:03d}"
        truth = "Yes" if input_index % 2 == 0 else "No"
        for condition_index, condition in enumerate(CONDITIONS):
            degraded_offset = condition_index * 0.02
            for variant in PRIMARY_VARIANTS:
                wrong = input_index < error_limits[variant]
                prediction = "No" if truth == "Yes" else "Yes" if wrong else truth
                if not wrong:
                    prediction = truth
                p_yes = 0.75 if prediction == "Yes" else 0.25
                uncertainty = 0.8 if wrong else 0.2
                uncertainty = min(0.99, uncertainty + degraded_offset)
                normalized.append({
                    "variant": variant,
                    "input_id": input_id,
                    "source_index": input_index,
                    "condition": condition,
                    "condition_index": condition_index,
                    "ground_truth": truth,
                    "prediction": prediction,
                    "correct": not wrong,
                    "p_yes": p_yes,
                    "p_no": 1.0 - p_yes,
                    "confidence": 0.75,
                    "predictive_entropy": uncertainty,
                    "expected_entropy": uncertainty - 0.01 if variant in ("three_lora_ensemble_calibrated", "laplace_head") else None,
                    "mutual_information": 0.01 if variant in ("three_lora_ensemble_calibrated", "laplace_head") else None,
                    "analysis_id": f"{variant}-{input_id}-{condition}",
                })
            ensemble_wrong = input_index < error_limits["three_lora_ensemble_calibrated"]
            ensemble.append({
                "input_id": input_id,
                "condition": condition,
                "calibrated_member_probability_variance": (0.2 if ensemble_wrong else 0.01) + degraded_offset,
                "calibrated_mutual_information_style": (0.1 if ensemble_wrong else 0.005) + degraded_offset,
            })
            laplace_wrong = input_index < error_limits["laplace_head"]
            laplace.append({
                "example_id": input_id,
                "condition": condition,
                "posterior_predictive_variance": (0.2 if laplace_wrong else 0.01) + degraded_offset,
                "mutual_information": (0.1 if laplace_wrong else 0.005) + degraded_offset,
            })
    return normalized, ensemble, laplace


class RankingMetricTests(unittest.TestCase):
    def test_auroc_and_average_precision_handle_perfect_order_and_ties(self) -> None:
        labels = [False, False, True, True]
        self.assertEqual(binary_auroc(labels, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(binary_average_precision(labels, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(binary_auroc([False, True], [0.5, 0.5]), 0.5)
        self.assertEqual(binary_average_precision([False, True], [0.5, 0.5]), 0.5)


class ClusteredAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.normalized, cls.ensemble, cls.laplace = synthetic_inputs()

    def test_lora_summary_centres_all_three_seeds(self) -> None:
        summary = build_lora_seed_summary(self.normalized)
        overall = next(row for row in summary if row["scope"] == "overall")
        accuracy = overall["metrics"]["accuracy"]
        self.assertAlmostEqual(accuracy["mean"], (0.825 + 0.85 + 0.8) / 3.0)
        self.assertEqual(accuracy["minimum"], 0.8)
        self.assertEqual(accuracy["maximum"], 0.85)
        self.assertGreater(accuracy["sample_standard_deviation"], 0.0)

    def test_clustered_bootstrap_is_deterministic_and_paired(self) -> None:
        first = build_clustered_bootstrap_differences(
            self.normalized, repetitions=100, seed=123, confidence_level=0.95
        )
        second = build_clustered_bootstrap_differences(
            self.normalized, repetitions=100, seed=123, confidence_level=0.95
        )
        self.assertEqual(canonical_json(first), canonical_json(second))
        row = next(
            item for item in first
            if item["comparison"] == "lora_seed_mean_vs_baseline_calibrated"
            and item["scope"] == "overall"
        )
        self.assertEqual(row["clusters"], 400)
        self.assertAlmostEqual(
            row["metrics"]["accuracy"]["difference_candidate_minus_reference"],
            (0.825 + 0.85 + 0.8) / 3.0 - 0.7,
        )
        self.assertGreater(
            row["metrics"]["accuracy"]["percentile_interval_lower"], 0.0
        )

    def test_direct_uq_signals_rank_errors_and_degradation(self) -> None:
        error, degradation, seed_uq = build_uq_signal_evaluations(
            self.normalized, self.ensemble, self.laplace
        )
        validate_statistical_outputs(
            build_clustered_bootstrap_differences(
                self.normalized, repetitions=100, seed=7
            ),
            build_lora_seed_summary(self.normalized),
            error,
            degradation,
        )
        ensemble_variance = next(
            row for row in error
            if row["signal"] == "ensemble_member_probability_variance"
        )
        self.assertEqual(ensemble_variance["auroc"], 1.0)
        self.assertEqual(ensemble_variance["average_precision"], 1.0)
        laplace_degradation = next(
            row for row in degradation
            if row["signal"] == "laplace_mutual_information"
            and row["target"] == "no_passage"
        )
        self.assertGreater(laplace_degradation["auroc"], 0.5)
        self.assertEqual(len(seed_uq), 1)
        self.assertTrue(math.isfinite(seed_uq[0]["error_detection"]["auroc"]["mean"]))


if __name__ == "__main__":
    unittest.main()
