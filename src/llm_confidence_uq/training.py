"""Auditable PyTorch primitives for answer-token LoRA fine-tuning.

The supervised objective is deliberately narrow: given the frozen BoolQ prompt
ending in ``Answer:``, predict exactly one contextual class token (`` Yes`` or
`` No``).  Prompt tokens are inputs, never loss targets, and no numerical
confidence target is manufactured from the ground-truth answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import random
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-answer-token-lora-v1"
CLASS_ORDER = ("Yes", "No")
EXPECTED_CLASS_TOKEN_IDS = (7414, 2308)


class TrainingContractError(RuntimeError):
    """Raised when a locked training invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingContractError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _require_sha256(value: str, name: str) -> None:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )


@dataclass(frozen=True)
class SupervisedTrainingExample:
    """One in-memory prompt with one answer-token target."""

    ordinal: int
    example_id: str
    input_sha256: str
    prompt_sha256: str
    prompt_token_ids: tuple[int, ...]
    target_class: str
    target_token_id: int

    def __post_init__(self) -> None:
        require(
            isinstance(self.ordinal, int)
            and not isinstance(self.ordinal, bool)
            and self.ordinal >= 0,
            "ordinal must be a non-negative integer",
        )
        require(isinstance(self.example_id, str) and self.example_id, "example ID missing")
        _require_sha256(self.input_sha256, "input_sha256")
        _require_sha256(self.prompt_sha256, "prompt_sha256")
        require(bool(self.prompt_token_ids), "prompt token IDs must not be empty")
        require(
            all(
                isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and token_id >= 0
                for token_id in self.prompt_token_ids
            ),
            "prompt token IDs are invalid",
        )
        require(self.target_class in CLASS_ORDER, "target class is invalid")
        expected = EXPECTED_CLASS_TOKEN_IDS[CLASS_ORDER.index(self.target_class)]
        require(self.target_token_id == expected, "target token ID disagrees with class")


class SupervisedTrainingDataset:
    """Minimal ordered dataset; shuffling belongs to the seeded DataLoader."""

    def __init__(self, examples: Sequence[SupervisedTrainingExample]) -> None:
        require(bool(examples), "training dataset must not be empty")
        copied = tuple(examples)
        require(
            [example.ordinal for example in copied] == list(range(len(copied))),
            "training ordinals must be contiguous",
        )
        require(
            len({example.example_id for example in copied}) == len(copied),
            "duplicate training example ID",
        )
        require(
            len({example.input_sha256 for example in copied}) == len(copied),
            "duplicate training input",
        )
        self._examples = copied

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> SupervisedTrainingExample:
        return self._examples[index]


def position_ids_from_attention_mask(attention_mask: Any) -> Any:
    """Create padding-invariant causal positions for a binary attention mask."""

    import torch

    require(isinstance(attention_mask, torch.Tensor), "attention mask must be a tensor")
    require(attention_mask.ndim == 2, "attention mask must be rank two")
    require(attention_mask.numel() > 0, "attention mask must not be empty")
    require(
        bool(torch.all((attention_mask == 0) | (attention_mask == 1)).item()),
        "attention mask must be binary",
    )
    require(
        bool(torch.all(attention_mask.sum(dim=1) > 0).item()),
        "every sequence must contain an attended token",
    )
    positions = attention_mask.long().cumsum(dim=1) - 1
    return positions.masked_fill(attention_mask == 0, 0)


class AnswerTokenCollator:
    """Left-pad prompts and keep targets as a separate one-token class vector."""

    def __init__(self, *, pad_token_id: int, maximum_prompt_tokens: int) -> None:
        require(pad_token_id >= 0, "pad token ID must be non-negative")
        require(maximum_prompt_tokens > 0, "prompt ceiling must be positive")
        self._pad_token_id = int(pad_token_id)
        self._maximum_prompt_tokens = int(maximum_prompt_tokens)

    def __call__(
        self,
        examples: Sequence[SupervisedTrainingExample],
    ) -> dict[str, Any]:
        import torch

        require(bool(examples), "cannot collate an empty batch")
        maximum = max(len(example.prompt_token_ids) for example in examples)
        require(maximum <= self._maximum_prompt_tokens, "prompt exceeds training ceiling")
        batch = len(examples)
        input_ids = torch.full(
            (batch, maximum),
            self._pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch, maximum), dtype=torch.long)
        target_token_ids = torch.empty((batch,), dtype=torch.long)
        for row_index, example in enumerate(examples):
            tokens = torch.tensor(example.prompt_token_ids, dtype=torch.long)
            start = maximum - tokens.numel()
            input_ids[row_index, start:] = tokens
            attention_mask[row_index, start:] = 1
            target_token_ids[row_index] = example.target_token_id
        require(
            [int(value) for value in attention_mask.sum(dim=1).tolist()]
            == [len(example.prompt_token_ids) for example in examples],
            "collated prompt lengths drifted",
        )
        require(
            set(int(value) for value in target_token_ids.tolist())
            <= set(EXPECTED_CLASS_TOKEN_IDS),
            "collated targets contain a non-class token",
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids_from_attention_mask(attention_mask),
            "target_token_ids": target_token_ids,
            "ordinals": tuple(example.ordinal for example in examples),
            "example_ids": tuple(example.example_id for example in examples),
        }


def next_token_cross_entropy(logits: Any, target_token_ids: Any) -> Any:
    """Full-vocabulary cross entropy for exactly one next-token target per row."""

    import torch
    import torch.nn.functional as functional

    require(isinstance(logits, torch.Tensor), "logits must be a tensor")
    require(isinstance(target_token_ids, torch.Tensor), "targets must be a tensor")
    if logits.ndim == 3:
        require(logits.shape[1] == 1, "training logits must retain exactly one position")
        logits = logits[:, 0, :]
    require(logits.ndim == 2, "training logits must be rank two")
    require(target_token_ids.ndim == 1, "targets must be rank one")
    require(logits.shape[0] == target_token_ids.shape[0], "batch dimensions disagree")
    require(logits.shape[1] > max(EXPECTED_CLASS_TOKEN_IDS), "class token exceeds vocabulary")
    require(bool(torch.isfinite(logits).all().item()), "training logits are non-finite")
    require(
        set(int(value) for value in target_token_ids.detach().cpu().tolist())
        <= set(EXPECTED_CLASS_TOKEN_IDS),
        "training target is not Yes or No",
    )
    loss = functional.cross_entropy(logits.float(), target_token_ids, reduction="mean")
    require(bool(torch.isfinite(loss).item()), "training loss is non-finite")
    return loss


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic kernels."""

    import numpy as np
    import torch

    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "invalid seed")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def trainable_parameters(model: Any) -> list[tuple[str, Any]]:
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    require(bool(parameters), "model has no trainable parameters")
    require(len({name for name, _ in parameters}) == len(parameters), "duplicate parameter name")
    return parameters


def trainable_parameter_count(model: Any) -> int:
    return sum(parameter.numel() for _, parameter in trainable_parameters(model))


def total_parameter_count(model: Any) -> int:
    count = sum(parameter.numel() for parameter in model.parameters())
    require(count > 0, "model has no parameters")
    return count


def trainable_parameter_sha256(model: Any) -> str:
    """Hash names, shapes, dtypes and raw bytes of trainable parameters."""

    import torch

    digest = sha256()
    for name, parameter in trainable_parameters(model):
        tensor = parameter.detach().cpu().contiguous()
        header = canonical_json(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            }
        )
        digest.update(header.encode("utf-8"))
        digest.update(b"\n")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def cosine_warmup_factor(
    completed_step: int,
    *,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Warm up linearly, then decay with a half cosine to zero."""

    require(total_steps > 0, "total steps must be positive")
    require(0 <= warmup_steps < total_steps, "warmup steps are invalid")
    require(0 <= completed_step <= total_steps, "completed step is out of range")
    if warmup_steps and completed_step < warmup_steps:
        return float(completed_step + 1) / float(warmup_steps)
    progress = float(completed_step - warmup_steps) / float(total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_cosine_scheduler(
    optimizer: Any,
    *,
    total_steps: int,
    warmup_steps: int,
) -> Any:
    import torch

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_factor(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        ),
    )


def make_epoch_dataloader(
    dataset: SupervisedTrainingDataset,
    *,
    collator: AnswerTokenCollator,
    batch_size: int,
    seed: int,
    epoch_index: int,
) -> Any:
    import torch

    require(batch_size > 0, "batch size must be positive")
    require(epoch_index >= 0, "epoch index must be non-negative")
    generator = torch.Generator()
    generator.manual_seed(seed + epoch_index)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
        generator=generator,
    )


def expected_optimizer_steps(
    *,
    examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> int:
    require(examples > 0 and batch_size > 0 and epochs > 0, "invalid training cardinality")
    require(gradient_accumulation_steps > 0, "invalid accumulation count")
    require(examples % batch_size == 0, "examples must divide evenly by batch size")
    microbatches = examples // batch_size
    require(
        microbatches % gradient_accumulation_steps == 0,
        "microbatches must divide evenly by accumulation count",
    )
    return (microbatches // gradient_accumulation_steps) * epochs


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if name in {"input_ids", "attention_mask", "position_ids", "target_token_ids"}
    }


def train_one_epoch(
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: Any,
    scheduler: Any,
    *,
    device: Any,
    gradient_accumulation_steps: int,
    maximum_gradient_norm: float,
    starting_optimizer_step: int,
) -> dict[str, Any]:
    """Run one exact epoch using explicit forward/backward/clip/step operations."""

    import torch

    require(gradient_accumulation_steps > 0, "invalid accumulation count")
    require(maximum_gradient_norm > 0, "gradient norm ceiling must be positive")
    require(starting_optimizer_step >= 0, "starting step must be non-negative")
    named_trainable = trainable_parameters(model)
    initial_sha256 = trainable_parameter_sha256(model)
    model.train()
    require(model.training is True, "model failed to enter training mode")
    optimizer.zero_grad(set_to_none=True)
    microbatch_losses: list[float] = []
    optimizer_records: list[dict[str, Any]] = []
    order: list[int] = []
    microbatch_count = 0
    gradient_non_null_names: set[str] = set()

    for batch in dataloader:
        microbatch_count += 1
        order.extend(int(value) for value in batch["ordinals"])
        tensors = _move_batch(batch, device)
        outputs = model(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            position_ids=tensors["position_ids"],
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        loss = next_token_cross_entropy(outputs.logits, tensors["target_token_ids"])
        microbatch_losses.append(float(loss.detach().cpu().item()))
        (loss / gradient_accumulation_steps).backward()

        if microbatch_count % gradient_accumulation_steps != 0:
            continue

        gradients = []
        missing = []
        for name, parameter in named_trainable:
            if parameter.grad is None:
                missing.append(name)
                continue
            gradient_non_null_names.add(name)
            require(
                bool(torch.isfinite(parameter.grad).all().item()),
                f"non-finite gradient: {name}",
            )
            gradients.append(parameter.grad)
        require(not missing, f"trainable parameters lack gradients: {missing[:5]}")
        require(
            any(bool(torch.any(gradient != 0).item()) for gradient in gradients),
            "all trainable gradients are zero",
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named_trainable],
            max_norm=maximum_gradient_norm,
            error_if_nonfinite=True,
        )
        require(bool(torch.isfinite(gradient_norm).item()), "gradient norm is non-finite")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step = starting_optimizer_step + len(optimizer_records) + 1
        window = microbatch_losses[-gradient_accumulation_steps:]
        optimizer_records.append(
            {
                "optimizer_step": optimizer_step,
                "mean_microbatch_loss": sum(window) / len(window),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu().item()),
                "learning_rates_after_step": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
            }
        )

    require(microbatch_count > 0, "epoch produced no microbatches")
    require(
        microbatch_count % gradient_accumulation_steps == 0,
        "partial accumulation window is forbidden",
    )
    final_sha256 = trainable_parameter_sha256(model)
    require(final_sha256 != initial_sha256, "no trainable parameter changed")
    require(
        len(gradient_non_null_names) == len(named_trainable),
        "some trainable parameters never received gradients",
    )
    return {
        "microbatches": microbatch_count,
        "optimizer_steps": len(optimizer_records),
        "mean_microbatch_loss": sum(microbatch_losses) / len(microbatch_losses),
        "all_losses_finite": all(math.isfinite(value) for value in microbatch_losses),
        "all_trainable_parameters_received_gradients": True,
        "trainable_parameter_sha256_before": initial_sha256,
        "trainable_parameter_sha256_after": final_sha256,
        "parameters_changed": True,
        "example_order_sha256": sha256_text(canonical_json(order)),
        "optimizer_records": optimizer_records,
    }


def validate_independent_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    require(len(normalized) >= 2, "at least two seeds are required")
    require(len(set(normalized)) == len(normalized), "ensemble seeds must be distinct")
    require(all(seed >= 0 for seed in normalized), "ensemble seed is negative")
    return normalized


__all__ = [
    "AnswerTokenCollator",
    "CLASS_ORDER",
    "EXPECTED_CLASS_TOKEN_IDS",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "SupervisedTrainingDataset",
    "SupervisedTrainingExample",
    "TrainingContractError",
    "build_cosine_scheduler",
    "canonical_json",
    "cosine_warmup_factor",
    "expected_optimizer_steps",
    "make_epoch_dataloader",
    "next_token_cross_entropy",
    "position_ids_from_attention_mask",
    "set_deterministic_seed",
    "sha256_bytes",
    "sha256_text",
    "total_parameter_count",
    "train_one_epoch",
    "trainable_parameter_count",
    "trainable_parameter_sha256",
    "validate_independent_seeds",
]
