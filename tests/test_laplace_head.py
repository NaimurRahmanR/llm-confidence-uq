from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

import torch
import yaml


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq.laplace_head import (
    CONDITIONS,
    PROTOCOL_VERSION,
    BayesianLinearHead,
    FrozenPromptCollator,
    FrozenPromptDataset,
    FrozenPromptExample,
    LaplaceHeadError,
    apply_standardizer,
    canonical_json,
    checkpoint_payload,
    checkpoint_tensors,
    diagonal_laplace,
    fit_map,
    fit_standardizer,
    gather_last_hidden,
    negative_log_posterior,
    parameter_vector,
    posterior_predictive,
    sha256_text,
    summarize_rows,
)


def example(ordinal: int, *, length: int, target: int = 1, condition: str = "original") -> FrozenPromptExample:
    return FrozenPromptExample(
        ordinal=ordinal,
        example_id=f"example-{ordinal}",
        source_index=ordinal,
        condition=condition,
        condition_index=CONDITIONS.index(condition),
        prompt_token_ids=tuple(range(1, length + 1)),
        target=target,
        source_row_sha256=sha256_text(f"source-{ordinal}-{condition}"),
    )


class DatasetAndRepresentationTests(unittest.TestCase):
    def test_dataset_identity_and_contiguous_order(self) -> None:
        dataset = FrozenPromptDataset([example(0, length=2), example(1, length=3)])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[1].ordinal, 1)
        with self.assertRaisesRegex(LaplaceHeadError, "non-contiguous"):
            FrozenPromptDataset([example(1, length=2)])

    def test_collator_left_pads_without_transformer_labels(self) -> None:
        batch = FrozenPromptCollator(0)([example(0, length=2), example(1, length=4, target=0)])
        self.assertEqual(batch["input_ids"].tolist(), [[0, 0, 1, 2], [1, 2, 3, 4]])
        self.assertEqual(batch["attention_mask"].tolist(), [[0, 0, 1, 1], [1, 1, 1, 1]])
        self.assertEqual(batch["position_ids"].tolist(), [[0, 0, 0, 1], [0, 1, 2, 3]])
        self.assertNotIn("labels", batch)
        self.assertEqual(batch["targets"].tolist(), [1.0, 0.0])

    def test_last_attended_hidden_gather_supports_left_and_right_padding(self) -> None:
        hidden = torch.arange(2 * 4 * 3, dtype=torch.float64).reshape(2, 4, 3)
        mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
        gathered = gather_last_hidden(hidden, mask)
        self.assertTrue(torch.equal(gathered[0], hidden[0, 3]))
        self.assertTrue(torch.equal(gathered[1], hidden[1, 1]))

    def test_standardizer_uses_population_statistics_and_rejects_bad_scale(self) -> None:
        features = torch.tensor([[1.0, 3.0], [3.0, 7.0]], dtype=torch.float64)
        mean, scale = fit_standardizer(features)
        self.assertTrue(torch.allclose(mean, torch.tensor([2.0, 5.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(scale, torch.tensor([1.0, 2.0], dtype=torch.float64)))
        standardized = apply_standardizer(features, mean, scale)
        self.assertTrue(torch.allclose(standardized.mean(dim=0), torch.zeros(2, dtype=torch.float64)))
        with self.assertRaisesRegex(LaplaceHeadError, "non-positive"):
            apply_standardizer(features, mean, torch.tensor([1.0, 0.0], dtype=torch.float64))


class MapAndLaplaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = torch.tensor(
            [[-2.0, -1.0], [-1.5, -0.5], [-1.0, -1.5], [-0.5, -1.0], [0.5, 1.0], [1.0, 1.5], [1.5, 0.5], [2.0, 1.0]],
            dtype=torch.float64,
        )
        self.targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float64)

    def test_map_objective_is_bernoulli_likelihood_plus_gaussian_prior(self) -> None:
        head = BayesianLinearHead(2)
        observed = negative_log_posterior(head, self.features, self.targets, 1.0)
        self.assertAlmostEqual(float(observed.item()), 8.0 * math.log(2.0), places=10)
        with torch.no_grad():
            head.linear.weight.fill_(1.0)
            head.linear.bias.fill_(1.0)
        without_prior = torch.nn.functional.binary_cross_entropy_with_logits(head(self.features), self.targets, reduction="sum")
        with_prior = negative_log_posterior(head, self.features, self.targets, 2.0)
        self.assertAlmostEqual(float((with_prior - without_prior).item()), 3.0, places=10)

    def test_map_fit_changes_parameters_receives_gradients_and_reduces_objective(self) -> None:
        head, report = fit_map(self.features, self.targets, prior_precision=1.0, maximum_iterations=50)
        self.assertTrue(report["parameters_changed"])
        self.assertTrue(report["all_parameters_received_finite_gradients"])
        self.assertLess(report["final_negative_log_posterior"], report["initial_negative_log_posterior"])
        self.assertNotEqual(report["initial_parameter_sha256"], report["map_parameter_sha256"])
        self.assertTrue(torch.isfinite(parameter_vector(head)).all())

    def test_diagonal_curvature_matches_logistic_formula(self) -> None:
        head = BayesianLinearHead(2)
        with torch.no_grad():
            head.linear.weight.copy_(torch.tensor([[0.4, -0.2]], dtype=torch.float64))
            head.linear.bias.copy_(torch.tensor([0.1], dtype=torch.float64))
        precision, variance = diagonal_laplace(head, self.features, prior_precision=1.5)
        augmented = torch.cat((self.features, torch.ones((len(self.features), 1), dtype=torch.float64)), dim=1)
        probabilities = torch.sigmoid(head(self.features))
        expected = 1.5 + ((probabilities * (1.0 - probabilities))[:, None] * augmented.square()).sum(dim=0)
        self.assertTrue(torch.allclose(precision, expected, rtol=0.0, atol=1e-12))
        self.assertTrue(torch.allclose(variance, expected.reciprocal(), rtol=0.0, atol=1e-12))
        self.assertTrue(torch.all(variance >= 0))

    def test_posterior_sampling_is_deterministic_variable_and_nonnegative(self) -> None:
        head, _ = fit_map(self.features, self.targets, prior_precision=1.0, maximum_iterations=50)
        _, variance = diagonal_laplace(head, self.features, prior_precision=1.0)
        first = posterior_predictive(self.features, parameter_vector(head), variance, samples=128, seed=123)
        second = posterior_predictive(self.features, parameter_vector(head), variance, samples=128, seed=123)
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]))
            self.assertTrue(torch.isfinite(first[key]).all())
        self.assertTrue(torch.all(first["variance"] > 0))
        self.assertTrue(torch.all(first["mutual_information"] >= 0))
        changed = posterior_predictive(self.features, parameter_vector(head), variance, samples=128, seed=124)
        self.assertFalse(torch.equal(first["mean_p_yes"], changed["mean_p_yes"]))

    def test_checkpoint_round_trip_is_exact_and_scope_safe(self) -> None:
        head, report = fit_map(self.features, self.targets, prior_precision=1.0, maximum_iterations=50)
        precision, variance = diagonal_laplace(head, self.features, prior_precision=1.0)
        payload = checkpoint_payload(
            feature_mean=torch.zeros(2, dtype=torch.float64),
            feature_scale=torch.ones(2, dtype=torch.float64),
            map_parameters=parameter_vector(head),
            posterior_precision=precision,
            posterior_variance=variance,
            prior_precision=1.0,
            training_report=report,
        )
        serialized = (canonical_json(payload) + "\n").encode("utf-8")
        restored = checkpoint_tensors(json.loads(serialized))
        self.assertTrue(torch.equal(restored["map_parameters"], parameter_vector(head)))
        self.assertFalse(payload["full_transformer_is_bayesian"])
        bad = dict(payload)
        bad["full_transformer_is_bayesian"] = True
        with self.assertRaisesRegex(LaplaceHeadError, "overclaims"):
            checkpoint_tensors(bad)


class MetricAndContractTests(unittest.TestCase):
    def test_metrics_are_finite_and_error_detection_is_defined(self) -> None:
        rows = []
        for index, (truth, mean) in enumerate((("Yes", 0.9), ("No", 0.2), ("Yes", 0.4), ("No", 0.7))):
            prediction = "Yes" if mean >= 0.5 else "No"
            rows.append({
                "ground_truth": truth,
                "prediction": prediction,
                "correct": prediction == truth,
                "posterior_mean_p_yes": mean,
                "confidence": max(mean, 1.0 - mean),
                "predictive_entropy": 0.5,
                "expected_entropy": 0.4,
                "mutual_information": 0.1,
                "posterior_predictive_variance": 0.02,
            })
        metrics = summarize_rows(rows)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertIsNotNone(metrics["error_detection_auroc"])
        for key in ("nll", "brier", "ece_10_bin", "mean_confidence", "mean_mutual_information"):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))

    def test_configuration_binds_math_scope_and_implementation_hashes(self) -> None:
        config = yaml.safe_load((REPO / "configs/laplace_head.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(config["map"]["likelihood"], "bernoulli-logistic")
        self.assertEqual(config["map"]["prior_precision"], 1.0)
        self.assertEqual(config["laplace"]["curvature"], "exact-logistic-negative-log-posterior-diagonal")
        self.assertFalse(config["claim_boundaries"]["full_transformer_is_bayesian"])
        self.assertEqual(config["claim_boundaries"]["approximate_posterior_scope"], "binary_linear_prediction_head_only")
        for relative, expected in config["implementation"].items():
            self.assertNotEqual(expected, "PENDING")
            self.assertEqual(hashlib.sha256((REPO / relative).read_bytes()).hexdigest(), expected)

    def test_runner_import_does_not_load_dataset_transformer_or_model_weights(self) -> None:
        before = set(sys.modules)
        script = REPO / "scripts/run_laplace_head.py"
        specification = importlib.util.spec_from_file_location("laplace_runner_test", script)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        added = set(sys.modules) - before
        self.assertFalse(any(name == "datasets" or name.startswith("transformers") or name.startswith("peft") for name in added))


if __name__ == "__main__":
    unittest.main()
