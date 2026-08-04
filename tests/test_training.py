from __future__ import annotations

import builtins
import contextlib
from dataclasses import FrozenInstanceError
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_confidence_uq import training


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        import torch
    except ModuleNotFoundError:
        test_case.skipTest("PyTorch is unavailable in this local audit runtime")
    return torch


def _example(ordinal: int, *, target_class: str | None = None):
    target_class = target_class or ("Yes" if ordinal % 2 == 0 else "No")
    target_id = 7414 if target_class == "Yes" else 2308
    return training.SupervisedTrainingExample(
        ordinal=ordinal,
        example_id=f"example-{ordinal}",
        input_sha256=f"{ordinal + 1:064x}",
        prompt_sha256=f"{ordinal + 101:064x}",
        prompt_token_ids=tuple(range(1, 3 + ordinal % 3)),
        target_class=target_class,
        target_token_id=target_id,
    )


class ExampleDatasetTests(unittest.TestCase):
    def test_example_is_validated_and_immutable(self) -> None:
        example = _example(0)
        self.assertEqual(example.target_class, "Yes")
        self.assertEqual(example.target_token_id, 7414)
        with self.assertRaises(FrozenInstanceError):
            example.ordinal = 3

    def test_example_rejects_class_token_disagreement(self) -> None:
        with self.assertRaises(training.TrainingContractError):
            training.SupervisedTrainingExample(
                ordinal=0,
                example_id="bad",
                input_sha256="1" * 64,
                prompt_sha256="2" * 64,
                prompt_token_ids=(1, 2),
                target_class="Yes",
                target_token_id=2308,
            )

    def test_dataset_rejects_duplicate_and_noncontiguous_identity(self) -> None:
        first = _example(0)
        duplicate = training.SupervisedTrainingExample(
            ordinal=1,
            example_id=first.example_id,
            input_sha256="f" * 64,
            prompt_sha256="e" * 64,
            prompt_token_ids=(1,),
            target_class="No",
            target_token_id=2308,
        )
        with self.assertRaises(training.TrainingContractError):
            training.SupervisedTrainingDataset((first, duplicate))
        with self.assertRaises(training.TrainingContractError):
            training.SupervisedTrainingDataset((_example(1),))

    def test_dataset_preserves_order_and_indexing(self) -> None:
        examples = tuple(_example(index) for index in range(4))
        dataset = training.SupervisedTrainingDataset(examples)
        self.assertEqual(len(dataset), 4)
        self.assertEqual([dataset[index] for index in range(4)], list(examples))


class TensorTrainingTests(unittest.TestCase):
    def test_position_ids_are_padding_invariant(self) -> None:
        torch = _torch_or_skip(self)
        mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        positions = training.position_ids_from_attention_mask(mask)
        self.assertEqual(positions.tolist(), [[0, 0, 0, 1], [0, 0, 1, 2]])
        with self.assertRaises(training.TrainingContractError):
            training.position_ids_from_attention_mask(torch.tensor([[0, 2]]))

    def test_collator_left_pads_and_keeps_one_separate_target(self) -> None:
        torch = _torch_or_skip(self)
        del torch
        examples = (_example(0), _example(2))
        batch = training.AnswerTokenCollator(
            pad_token_id=99,
            maximum_prompt_tokens=8,
        )(examples)
        self.assertEqual(
            batch["input_ids"].tolist(),
            [[99, 99, 1, 2], [1, 2, 3, 4]],
        )
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[0, 0, 1, 1], [1, 1, 1, 1]],
        )
        self.assertEqual(batch["target_token_ids"].tolist(), [7414, 7414])
        self.assertNotIn("labels", batch)
        self.assertNotIn("prompt", batch)

    def test_next_token_loss_is_finite_and_backpropagates(self) -> None:
        torch = _torch_or_skip(self)
        logits = torch.zeros((2, 1, 8000), requires_grad=True)
        targets = torch.tensor([7414, 2308])
        loss = training.next_token_cross_entropy(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(torch.abs(logits.grad).sum()), 0.0)

    def test_nonfinite_logits_fail_closed(self) -> None:
        torch = _torch_or_skip(self)
        logits = torch.zeros((1, 1, 8000))
        logits[0, 0, 3] = float("nan")
        with self.assertRaises(training.TrainingContractError):
            training.next_token_cross_entropy(logits, torch.tensor([7414]))

    def test_expected_optimizer_steps_require_exact_windows(self) -> None:
        self.assertEqual(
            training.expected_optimizer_steps(
                examples=800,
                batch_size=4,
                gradient_accumulation_steps=8,
                epochs=3,
            ),
            75,
        )
        with self.assertRaises(training.TrainingContractError):
            training.expected_optimizer_steps(
                examples=36,
                batch_size=4,
                gradient_accumulation_steps=8,
                epochs=1,
            )

    def test_cosine_warmup_boundaries_are_exact(self) -> None:
        self.assertEqual(
            training.cosine_warmup_factor(0, total_steps=10, warmup_steps=2),
            0.5,
        )
        self.assertEqual(
            training.cosine_warmup_factor(1, total_steps=10, warmup_steps=2),
            1.0,
        )
        self.assertAlmostEqual(
            training.cosine_warmup_factor(10, total_steps=10, warmup_steps=2),
            0.0,
        )

    def test_seeded_dataloader_order_repeats_and_changes(self) -> None:
        _torch_or_skip(self)
        dataset = training.SupervisedTrainingDataset(tuple(_example(i) for i in range(8)))
        collator = training.AnswerTokenCollator(pad_token_id=99, maximum_prompt_tokens=8)

        def order(seed):
            loader = training.make_epoch_dataloader(
                dataset,
                collator=collator,
                batch_size=2,
                seed=seed,
                epoch_index=0,
            )
            return [ordinal for batch in loader for ordinal in batch["ordinals"]]

        self.assertEqual(order(101), order(101))
        self.assertNotEqual(order(101), order(102))

    def test_explicit_training_loop_changes_parameters_and_audits_gradients(self) -> None:
        torch = _torch_or_skip(self)

        class ToyCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(128, 8)
                self.output = torch.nn.Linear(8, 8000)

            def forward(
                self,
                *,
                input_ids,
                attention_mask,
                position_ids,
                use_cache,
                return_dict,
                logits_to_keep,
            ):
                self.last_contract = {
                    "use_cache": use_cache,
                    "return_dict": return_dict,
                    "logits_to_keep": logits_to_keep,
                    "position_ids": position_ids.detach().clone(),
                }
                hidden = self.embedding(input_ids[:, -1])
                return SimpleNamespace(logits=self.output(hidden).unsqueeze(1))

        training.set_deterministic_seed(7)
        model = ToyCausalLM()
        dataset = training.SupervisedTrainingDataset(tuple(_example(i) for i in range(8)))
        collator = training.AnswerTokenCollator(pad_token_id=99, maximum_prompt_tokens=8)
        loader = training.make_epoch_dataloader(
            dataset,
            collator=collator,
            batch_size=4,
            seed=7,
            epoch_index=0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = training.build_cosine_scheduler(
            optimizer,
            total_steps=1,
            warmup_steps=0,
        )
        report = training.train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            device=torch.device("cpu"),
            gradient_accumulation_steps=2,
            maximum_gradient_norm=1.0,
            starting_optimizer_step=0,
        )
        self.assertEqual(report["microbatches"], 2)
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertTrue(report["all_losses_finite"])
        self.assertTrue(report["parameters_changed"])
        self.assertTrue(report["all_trainable_parameters_received_gradients"])
        self.assertFalse(model.last_contract["use_cache"])
        self.assertEqual(model.last_contract["logits_to_keep"], 1)

    def test_trainable_parameter_digest_and_counts_bind_values(self) -> None:
        torch = _torch_or_skip(self)
        model = torch.nn.Linear(3, 2)
        before = training.trainable_parameter_sha256(model)
        self.assertEqual(training.trainable_parameter_count(model), 8)
        self.assertEqual(training.total_parameter_count(model), 8)
        with torch.no_grad():
            model.weight[0, 0] += 1
        after = training.trainable_parameter_sha256(model)
        self.assertNotEqual(before, after)


class ProtocolAndRunnerTests(unittest.TestCase):
    @staticmethod
    def _load_runner(*, block_heavy: bool = False):
        path = ROOT / "scripts/train_lora.py"
        spec = importlib.util.spec_from_file_location("lora_runner_test", path)
        if spec is None or spec.loader is None:
            raise AssertionError("runner import spec missing")
        module = importlib.util.module_from_spec(spec)
        if not block_heavy:
            spec.loader.exec_module(module)
            return module
        original_import = builtins.__import__
        blocked = {"datasets", "huggingface_hub", "peft", "torch", "transformers", "yaml"}

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

    def test_independent_seed_contract_rejects_duplicates(self) -> None:
        self.assertEqual(training.validate_independent_seeds([3, 4, 5]), (3, 4, 5))
        with self.assertRaises(training.TrainingContractError):
            training.validate_independent_seeds([3, 3, 5])

    def test_runner_import_does_not_load_gpu_model_or_dataset_libraries(self) -> None:
        runner = self._load_runner(block_heavy=True)
        self.assertEqual(
            runner.ALLOWED_STAGES,
            ("smoke32", "full_seed_1", "full_seed_2", "full_seed_3"),
        )

    def test_runner_cli_requires_explicit_locked_stage(self) -> None:
        runner = self._load_runner()
        parser = runner.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
        for stage in runner.ALLOWED_STAGES:
            self.assertEqual(parser.parse_args(["--stage", stage]).stage, stage)

    def test_repository_config_matches_exact_training_contract(self) -> None:
        try:
            import yaml
        except ModuleNotFoundError:
            self.skipTest("PyYAML is unavailable in this local audit runtime")
        config = yaml.safe_load((ROOT / "configs/lora.yaml").read_text("utf-8"))
        runner = self._load_runner()
        runner.validate_config_contract(config)

    def test_torchao_is_explicitly_excluded_and_checked_before_training(self) -> None:
        runner = self._load_runner()
        self.assertEqual(runner.EXPECTED_EXCLUDED_PACKAGES, ("torchao",))
        runner_source = (ROOT / "scripts/train_lora.py").read_text("utf-8")
        run_source = runner_source.split("def run(", 1)[1]
        self.assertLess(
            run_source.index("validate_excluded_runtime_packages()"),
            run_source.index("    import torch"),
        )

        with (
            mock.patch.object(
                runner.importlib.metadata,
                "version",
                side_effect=runner.importlib.metadata.PackageNotFoundError,
            ),
            mock.patch.object(runner.importlib.util, "find_spec", return_value=None),
        ):
            runner.validate_excluded_runtime_packages()

        with mock.patch.object(
            runner.importlib.metadata,
            "version",
            return_value="0.10.0",
        ):
            with self.assertRaisesRegex(
                runner.LoRARunError,
                "excluded package installed: torchao==0.10.0",
            ):
                runner.validate_excluded_runtime_packages()

    def test_implementation_uses_explicit_pytorch_not_trainer(self) -> None:
        module_source = (ROOT / "src/llm_confidence_uq/training.py").read_text("utf-8")
        runner_source = (ROOT / "scripts/train_lora.py").read_text("utf-8")
        required = (
            "loss / gradient_accumulation_steps",
            ".backward()",
            "clip_grad_norm_",
            "optimizer.step()",
            "scheduler.step()",
            "torch.optim.AdamW",
        )
        joined = module_source + runner_source
        for fragment in required:
            self.assertIn(fragment, joined)
        self.assertNotIn("Trainer(", joined)

    def test_config_forbids_confidence_targets_and_split_leakage(self) -> None:
        text = (ROOT / "configs/lora.yaml").read_text("utf-8")
        for fragment in (
            "numerical_confidence_targets_used: false",
            "calibration_examples_used: 0",
            "test_examples_used: 0",
            "trainer_abstraction_used: false",
            "persist_raw_prompt: false",
            "persist_question: false",
            "persist_passage: false",
        ):
            self.assertIn(fragment, text)

    def test_checkpoint_ledger_round_trip_and_mutation_detection(self) -> None:
        runner = self._load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner._write_bytes(root / "adapter/model.bin", b"adapter")
            runner._write_bytes(root / "training_report.json", b"{}\n")
            ledger = runner.directory_ledger(root)
            payload = runner.canonical_bytes(ledger)
            runner._write_bytes(root / "artifact_hashes.json", payload)
            runner._write_bytes(
                root / "COMPLETE.json",
                runner.canonical_bytes(
                    {
                        "schema_version": 1,
                        "protocol_version": training.PROTOCOL_VERSION,
                        "stage": "test",
                        "artifact_hashes_sha256": training.sha256_bytes(payload),
                        "complete": True,
                    }
                ),
            )
            runner.verify_checkpoint_directory(root)
            (root / "adapter/model.bin").write_bytes(b"changed")
            with self.assertRaises(runner.LoRARunError):
                runner.verify_checkpoint_directory(root)


if __name__ == "__main__":
    unittest.main()
