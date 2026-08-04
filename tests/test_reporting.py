from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq.reporting import (
    FIGURES,
    PRIMARY_VARIANTS,
    PROTOCOL_VERSION,
    ReportingError,
    build_documents,
    load_config,
    load_inputs,
    pct,
    preflight,
)


class ReportingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = Path("configs/report.json")
        cls.config = load_config(REPO / cls.config_path)
        cls.inputs = load_inputs(REPO, cls.config)
        cls.documents = build_documents(cls.inputs, cls.config)
        cls.text = {name: payload.decode("utf-8") for name, payload in cls.documents.items()}

    def test_config_binds_every_input_hash(self) -> None:
        self.assertEqual(self.config["protocol_version"], PROTOCOL_VERSION)
        for specification in self.config["inputs"].values():
            payload = (REPO / specification["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), specification["sha256"])

    def test_document_set_and_encoding_are_fixed(self) -> None:
        self.assertEqual(set(self.documents), set(self.config["outputs"]))
        for name, payload in self.documents.items():
            self.assertTrue(payload.endswith(b"\n"), name)
            self.assertNotIn(b"\r", payload, name)
            self.assertGreater(len(payload), 100, name)

    def test_report_has_every_required_section(self) -> None:
        report = self.text["report/technical_report.md"]
        required = (
            "## Abstract",
            "## 1. Research question",
            "## 2. Connection to Evidence-State Reliability",
            "## 3. Dataset and evidence conditions",
            "## 4. Model and fine-tuning method",
            "## 5. Confidence definitions",
            "## 6. Calibration method",
            "## 7. Ensemble method",
            "## 8. Bayesian prediction-head method",
            "## 9. Experimental protocol",
            "## 10. Metrics",
            "## 11. Results",
            "## 12. Failure analysis",
            "## 13. Limitations",
            "## 14. Reproducibility statement",
            "## 15. Claim boundaries",
            "## 16. Future work",
            "## 17. References",
        )
        for heading in required:
            self.assertIn(heading, report)

    def test_reported_headline_numbers_derive_from_summary(self) -> None:
        summary = self.inputs["results_summary"]["primary_overall"]
        readme = self.text["README.md"]
        report = self.text["report/technical_report.md"]
        for variant in PRIMARY_VARIANTS:
            accuracy = pct(summary[variant]["accuracy"])
            self.assertIn(accuracy, readme)
            self.assertIn(accuracy, report)
        source = (REPO / "src/llm_confidence_uq/reporting.py").read_text(encoding="utf-8")
        for forbidden_literal in ("0.73875", "0.7101502201251497", "0.0229594628967815"):
            self.assertNotIn(forbidden_literal, source)

    def test_scientific_headline_and_three_seed_summary_are_primary(self) -> None:
        readme = self.text["README.md"]
        self.assertIn(
            "LoRA adaptation improved binary QA performance, but generated confidence\n"
            "formatting became substantially less reliable; temperature scaling improved\n"
            "probabilistic scores, while ensemble and Laplace methods showed different\n"
            "calibration and error-ranking trade-offs.",
            readme,
        )
        self.assertIn("Three-seed LoRA summary", readme)
        self.assertIn("mean ±", readme)
        self.assertNotIn("strongest single adapter", readme.lower())

    def test_clustered_intervals_cover_every_requested_metric(self) -> None:
        rows = self.inputs["bootstrap_differences"]
        self.assertEqual(len(rows), 49)
        self.assertTrue(all(row["cluster_unit"] == "input_id" for row in rows))
        self.assertTrue(all(row["clusters"] == 400 for row in rows))
        self.assertTrue(all(row["bootstrap_repetitions"] == 2000 for row in rows))
        expected = {"accuracy", "nll", "brier", "ece_10_bin", "error_detection_auroc"}
        self.assertTrue(all(set(row["metrics"]) == expected for row in rows))
        report = self.text["report/technical_report.md"]
        for label in ("Δ accuracy", "Δ NLL", "Δ Brier", "Δ ECE", "Δ error AUROC"):
            self.assertIn(label, report)

    def test_direct_uq_signal_results_are_reported(self) -> None:
        error_rows = self.inputs["uq_error_detection"]
        signals = {row["signal"] for row in error_rows}
        self.assertTrue({
            "predictive_entropy",
            "ensemble_member_probability_variance",
            "ensemble_mi_style_disagreement",
            "laplace_posterior_predictive_variance",
            "laplace_mutual_information",
        }.issubset(signals))
        report = self.text["report/technical_report.md"]
        readme = self.text["README.md"]
        self.assertIn("Direct evaluation of uncertainty signals", report)
        self.assertIn("Any-degraded AUROC", report)
        self.assertIn("5/6 positive", report)
        for rendered in (readme, report):
            self.assertIn("Any-degraded AUPRC baseline prevalence = 0.8333", rendered)
            self.assertIn(
                "Error AUPRC should not be compared naively across models because each model has a different error prevalence.",
                rendered,
            )

    def test_stale_and_unsupported_claims_are_absent(self) -> None:
        combined = "\n".join(self.text.values())
        self.assertNotIn("no empirical research results are claimed", combined.lower())
        self.assertNotIn("these are planned methods", combined.lower())
        self.assertNotIn("nvidia tesla t4", combined.lower())
        self.assertNotIn("fully bayesian transformer", combined.lower())
        self.assertIn("not peer reviewed", combined.lower())
        self.assertIn("transformer is not bayesian", combined.lower())
        self.assertIn("not posterior samples", combined.lower())

    def test_readme_commands_match_locked_cli_names(self) -> None:
        readme = self.text["README.md"]
        self.assertIn("prepare_data.py --config configs/data.yaml --output-dir data/manifests", readme)
        self.assertIn("run_lora_inference.py --adapter full_seed_1", readme)
        self.assertIn("run_calibration_inference.py --method baseline --stage full", readme)
        self.assertIn("evaluate_predictions.py --method full_seed_3", readme)
        self.assertIn("build_report.py --config configs/report.json", readme)
        self.assertIn("build_statistical_analysis.py --config configs/statistical_analysis.yaml", readme)
        self.assertIn("bash colab/setup.sh", readme)
        self.assertNotIn("configs/report.yaml", readme)
        self.assertNotIn("evaluate_predictions.py --all", readme)
        self.assertNotIn("run_lora_inference.py --method", readme)

    def test_claims_cover_missing_expressed_confidence(self) -> None:
        claims = self.text["report/claim_boundaries.md"]
        report = self.text["report/technical_report.md"]
        self.assertIn("invalid rows remain missing", claims)
        self.assertIn("selected parser-valid subset", report)
        for method in ("baseline", "full_seed_1", "full_seed_2", "full_seed_3"):
            rate = pct(self.inputs["results_summary"]["expressed_overall"][method]["valid_rate"])
            self.assertIn(rate, report)

    def test_citation_and_reproduction_metadata_parse(self) -> None:
        citation = self.text["CITATION.cff"]
        self.assertIn("cff-version: 1.2.0", citation)
        self.assertIn('version: "0.1.1"', citation)
        self.assertIn('repository-code: "https://github.com/NaimurRahmanR/llm-confidence-uq"', citation)
        self.assertIn('alias: "NaimurRahmanR"', citation)
        reproducibility = self.text["report/reproducibility.md"]
        for specification in self.config["inputs"].values():
            self.assertIn(specification["sha256"], reproducibility)

    def test_figures_and_results_are_bound_to_machine_readable_sources(self) -> None:
        manifest = self.inputs["figure_manifest"]
        self.assertFalse(manifest["manual_result_values_used"])
        self.assertEqual(tuple(manifest["figures"]), FIGURES)
        for record in manifest["figures"].values():
            self.assertTrue(record["source_tables"])

    def test_preflight_is_read_only_and_complete(self) -> None:
        paths = [REPO / name for name in self.config["outputs"]]
        before = {
            path.as_posix(): (
                path.exists(),
                hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            )
            for path in paths
        }
        report = preflight(REPO, self.config_path)
        after = {
            path.as_posix(): (
                path.exists(),
                hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            )
            for path in paths
        }
        self.assertEqual(before, after)
        self.assertEqual(report["documents"], 7)
        self.assertEqual(report["figures"], 8)
        self.assertFalse(report["files_modified"])

    def test_mutated_input_fails_closed(self) -> None:
        changed = dict(self.config)
        changed["inputs"] = {name: dict(value) for name, value in self.config["inputs"].items()}
        changed["inputs"]["results_summary"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ReportingError, "hash drift"):
            load_inputs(REPO, changed)


if __name__ == "__main__":
    unittest.main()
