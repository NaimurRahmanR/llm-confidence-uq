from __future__ import annotations

import builtins
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq import adapter_inference
from llm_confidence_uq.inference import (
    InferenceExample,
    make_prediction_row,
    parse_expressed_continuation,
)


REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LEDGER = "1" * 64
WEIGHTS = "2" * 64
SEMANTIC = "3" * 64


def _core_row(ordinal: int = 0):
    example = InferenceExample(
        ordinal=ordinal,
        transformation_id="boolqdeg-" + f"{ordinal + 1:064x}",
        degradation_row_sha256=f"{ordinal + 101:064x}",
        input_id="boolqinput-" + f"{ordinal + 201:064x}",
        source_split="validation",
        source_index=ordinal,
        condition_index=ordinal % 6,
        condition=("original", "no_passage")[ordinal % 2],
        runtime_status="ok",
        runtime_reason_code=None,
        prompt=f"prompt-{ordinal}",
        prompt_tokens=3,
    )
    p_yes = 1.0 / (1.0 + math.exp(-1.0))
    p_no = 1.0 - p_yes
    return make_prediction_row(
        example,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision=REVISION,
        inference_config_sha256="4" * 64,
        inference_seed=20260803,
        yes_token_id=7414,
        no_token_id=2308,
        yes_logit=1.0,
        no_logit=0.0,
        p_yes_binary=p_yes,
        p_no_binary=p_no,
        p_yes_full_vocabulary=0.2,
        p_no_full_vocabulary=0.1,
        yes_no_full_vocabulary_mass=0.3,
        token_prediction="Yes",
        token_confidence=p_yes,
        predictive_entropy=-(p_yes * math.log(p_yes) + p_no * math.log(p_no)),
        generated_continuation=" Yes\nConfidence: 80",
        generated_continuation_tokens=5,
        generation_ended_with_eos=True,
        expressed_parse=parse_expressed_continuation(" Yes\nConfidence: 80"),
    )


def _wrapped(ordinal: int = 0, *, stage: str = "full_seed_1", seed: int = 20260811):
    return adapter_inference.wrap_prediction_row(
        _core_row(ordinal),
        adapter_stage=stage,
        adapter_training_seed=seed,
        adapter_checkpoint_path=f"outputs/checkpoints/lora/{stage}",
        adapter_checkpoint_ledger_sha256=LEDGER,
        adapter_weights_sha256=WEIGHTS,
        adapter_semantic_config_sha256=SEMANTIC,
    )


class AdapterEnvelopeTests(unittest.TestCase):
    def test_wrap_preserves_valid_core_and_binds_adapter(self) -> None:
        row = _wrapped()
        adapter_inference.validate_adapter_prediction_row(row)
        self.assertEqual(row["adapter_stage"], "full_seed_1")
        self.assertEqual(row["adapter_training_seed"], 20260811)
        self.assertTrue(row["adapter_prediction_id"].startswith("boolqlorapred-"))
        self.assertNotIn("ground_truth", row)
        self.assertNotIn("correctness", row)

    def test_adapter_identity_and_row_mutations_are_rejected(self) -> None:
        row = _wrapped()
        row["adapter_weights_sha256"] = "9" * 64
        with self.assertRaises(adapter_inference.AdapterInferenceContractError):
            adapter_inference.validate_adapter_prediction_row(row)
        row = _wrapped()
        row["adapter_row_sha256"] = "8" * 64
        with self.assertRaises(adapter_inference.AdapterInferenceContractError):
            adapter_inference.validate_adapter_prediction_row(row)

    def test_different_adapters_have_different_bound_prediction_ids(self) -> None:
        first = _wrapped()
        second = _wrapped(stage="full_seed_2", seed=20260812)
        self.assertEqual(first["prediction_id"], second["prediction_id"])
        self.assertNotEqual(
            first["adapter_prediction_id"],
            second["adapter_prediction_id"],
        )

    def test_prediction_file_rejects_mixed_adapter_identity(self) -> None:
        first = _wrapped(0)
        second = _wrapped(1, stage="full_seed_2", seed=20260812)
        with self.assertRaises(adapter_inference.AdapterInferenceContractError):
            adapter_inference.validate_adapter_prediction_rows([first, second])

    def test_jsonl_and_summary_are_deterministic_and_label_free(self) -> None:
        rows = [_wrapped(0), _wrapped(1)]
        first = adapter_inference.adapter_prediction_jsonl_bytes(rows)
        second = adapter_inference.adapter_prediction_jsonl_bytes(rows)
        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        summary = adapter_inference.summarize_adapter_prediction_rows(rows)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["adapter_stage"], "full_seed_1")
        self.assertFalse(summary["contains_ground_truth"])
        self.assertFalse(summary["contains_correctness"])


class SemanticConfigTests(unittest.TestCase):
    @staticmethod
    def _load_runner(*, block_heavy: bool = False):
        path = ROOT / "scripts/run_lora_inference.py"
        spec = importlib.util.spec_from_file_location("lora_inference_runner_test", path)
        if spec is None or spec.loader is None:
            raise AssertionError("runner import spec missing")
        module = importlib.util.module_from_spec(spec)
        if not block_heavy:
            spec.loader.exec_module(module)
            return module
        original_import = builtins.__import__
        blocked = {"datasets", "huggingface_hub", "peft", "torch", "transformers"}

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

    def test_semantic_adapter_hash_is_order_invariant(self) -> None:
        runner = self._load_runner()
        left = json.dumps({"r": 8, "target_modules": sorted(runner.EXPECTED_TARGET_MODULES)}).encode()
        right = json.dumps({"r": 8, "target_modules": sorted(runner.EXPECTED_TARGET_MODULES, reverse=True)}).encode()
        self.assertEqual(
            runner.semantic_adapter_config_sha256(left),
            runner.semantic_adapter_config_sha256(right),
        )

    def test_semantic_adapter_hash_rejects_membership_drift(self) -> None:
        runner = self._load_runner()
        payload = json.dumps({"target_modules": ["q_proj"]}).encode()
        with self.assertRaises(runner.LoRAInferenceRunError):
            runner.semantic_adapter_config_sha256(payload)

    def test_runner_import_defers_model_dataset_and_peft_libraries(self) -> None:
        runner = self._load_runner(block_heavy=True)
        self.assertEqual(runner.ALLOWED_ADAPTERS, ("full_seed_1", "full_seed_2", "full_seed_3"))
        self.assertEqual(runner.ALLOWED_STAGES, ("smoke12", "full"))

    def test_runner_cli_requires_explicit_adapter_and_stage(self) -> None:
        runner = self._load_runner()
        parser = runner.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        parsed = parser.parse_args(["--adapter", "full_seed_2", "--stage", "smoke12"])
        self.assertEqual((parsed.adapter, parsed.stage), ("full_seed_2", "smoke12"))

    def test_repository_config_matches_adapter_inference_contract(self) -> None:
        try:
            import yaml
        except ModuleNotFoundError:
            self.skipTest("PyYAML unavailable in local audit runtime")
        runner = self._load_runner()
        config = yaml.safe_load((ROOT / "configs/lora_inference.yaml").read_text("utf-8"))
        runner.validate_config_contract(config)
        self.assertEqual(
            {config["adapters"][stage]["adapter_weights_sha256"] for stage in runner.ALLOWED_ADAPTERS},
            {
                "3bcd570b3ec3616ecaf3b7cc874a81fff84b3899334028d4d8179a367aea35b4",
                "0e393dc7cddf1f7183c17af5995d519de016f46d6151c5465f81845bb060e691",
                "a665864d76931f59650dec517ce66632233117685b690bde17c6138911dae46c",
            },
        )


if __name__ == "__main__":
    unittest.main()
