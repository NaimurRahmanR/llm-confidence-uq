from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .inference import (
    PREDICTION_FIELDS,
    canonical_json,
    canonical_jsonl_bytes,
    sha256_text,
    summarize_prediction_rows,
    validate_prediction_row,
)


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "boolq-lora-inference-v1"
ADAPTER_PREDICTION_PREFIX = "boolqlorapred-"
ADAPTER_FIELDS = {
    "adapter_protocol_version",
    "adapter_prediction_id",
    "adapter_stage",
    "adapter_training_seed",
    "adapter_checkpoint_path",
    "adapter_checkpoint_ledger_sha256",
    "adapter_weights_sha256",
    "adapter_semantic_config_sha256",
    "adapter_row_sha256",
}
PREDICTION_FIELDS_WITH_ADAPTER = PREDICTION_FIELDS | ADAPTER_FIELDS


class AdapterInferenceContractError(ValueError):
    """Raised when an adapter-bound prediction violates its contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterInferenceContractError(message)


def _require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _adapter_prediction_id(
    *,
    core_prediction_id: str,
    adapter_stage: str,
    adapter_training_seed: int,
    adapter_checkpoint_ledger_sha256: str,
    adapter_weights_sha256: str,
    adapter_semantic_config_sha256: str,
) -> str:
    identity = {
        "adapter_checkpoint_ledger_sha256": adapter_checkpoint_ledger_sha256,
        "adapter_semantic_config_sha256": adapter_semantic_config_sha256,
        "adapter_stage": adapter_stage,
        "adapter_training_seed": adapter_training_seed,
        "adapter_weights_sha256": adapter_weights_sha256,
        "core_prediction_id": core_prediction_id,
        "protocol_version": PROTOCOL_VERSION,
    }
    return ADAPTER_PREDICTION_PREFIX + sha256_text(canonical_json(identity))


def wrap_prediction_row(
    core_row: Mapping[str, Any],
    *,
    adapter_stage: str,
    adapter_training_seed: int,
    adapter_checkpoint_path: str,
    adapter_checkpoint_ledger_sha256: str,
    adapter_weights_sha256: str,
    adapter_semantic_config_sha256: str,
) -> dict[str, Any]:
    validate_prediction_row(core_row)
    require(
        isinstance(adapter_stage, str) and adapter_stage.startswith("full_seed_"),
        "adapter stage is invalid",
    )
    require(
        isinstance(adapter_training_seed, int)
        and not isinstance(adapter_training_seed, bool)
        and adapter_training_seed >= 0,
        "adapter training seed is invalid",
    )
    require(
        isinstance(adapter_checkpoint_path, str)
        and adapter_checkpoint_path
        == f"outputs/checkpoints/lora/{adapter_stage}",
        "adapter checkpoint path is invalid",
    )
    for value, label in (
        (adapter_checkpoint_ledger_sha256, "adapter checkpoint ledger SHA-256"),
        (adapter_weights_sha256, "adapter weights SHA-256"),
        (adapter_semantic_config_sha256, "adapter semantic config SHA-256"),
    ):
        _require_sha256(value, label)
    adapter_prediction_id = _adapter_prediction_id(
        core_prediction_id=str(core_row["prediction_id"]),
        adapter_stage=adapter_stage,
        adapter_training_seed=adapter_training_seed,
        adapter_checkpoint_ledger_sha256=adapter_checkpoint_ledger_sha256,
        adapter_weights_sha256=adapter_weights_sha256,
        adapter_semantic_config_sha256=adapter_semantic_config_sha256,
    )
    row_without_adapter_hash = {
        **dict(core_row),
        "adapter_protocol_version": PROTOCOL_VERSION,
        "adapter_prediction_id": adapter_prediction_id,
        "adapter_stage": adapter_stage,
        "adapter_training_seed": adapter_training_seed,
        "adapter_checkpoint_path": adapter_checkpoint_path,
        "adapter_checkpoint_ledger_sha256": adapter_checkpoint_ledger_sha256,
        "adapter_weights_sha256": adapter_weights_sha256,
        "adapter_semantic_config_sha256": adapter_semantic_config_sha256,
    }
    row = dict(row_without_adapter_hash)
    row["adapter_row_sha256"] = sha256_text(
        canonical_json(row_without_adapter_hash)
    )
    validate_adapter_prediction_row(row)
    return row


def validate_adapter_prediction_row(row: Mapping[str, Any]) -> None:
    require(
        set(row) == PREDICTION_FIELDS_WITH_ADAPTER,
        "adapter prediction row schema drift",
    )
    core_row = {field: row[field] for field in PREDICTION_FIELDS}
    validate_prediction_row(core_row)
    require(
        row["adapter_protocol_version"] == PROTOCOL_VERSION,
        "adapter protocol drift",
    )
    adapter_stage = row["adapter_stage"]
    adapter_seed = row["adapter_training_seed"]
    require(
        isinstance(adapter_stage, str) and adapter_stage.startswith("full_seed_"),
        "adapter stage drift",
    )
    require(
        isinstance(adapter_seed, int)
        and not isinstance(adapter_seed, bool)
        and adapter_seed >= 0,
        "adapter training seed drift",
    )
    require(
        row["adapter_checkpoint_path"]
        == f"outputs/checkpoints/lora/{adapter_stage}",
        "adapter checkpoint path drift",
    )
    for key in (
        "adapter_checkpoint_ledger_sha256",
        "adapter_weights_sha256",
        "adapter_semantic_config_sha256",
        "adapter_row_sha256",
    ):
        _require_sha256(row[key], key)
    expected_identifier = _adapter_prediction_id(
        core_prediction_id=str(row["prediction_id"]),
        adapter_stage=str(adapter_stage),
        adapter_training_seed=int(adapter_seed),
        adapter_checkpoint_ledger_sha256=str(
            row["adapter_checkpoint_ledger_sha256"]
        ),
        adapter_weights_sha256=str(row["adapter_weights_sha256"]),
        adapter_semantic_config_sha256=str(
            row["adapter_semantic_config_sha256"]
        ),
    )
    require(
        row["adapter_prediction_id"] == expected_identifier,
        "adapter prediction identity drift",
    )
    _require_sha256(
        str(row["adapter_prediction_id"]).removeprefix(
            ADAPTER_PREDICTION_PREFIX
        ),
        "adapter prediction ID",
    )
    require(
        row["adapter_row_sha256"]
        == sha256_text(
            canonical_json(
                {
                    key: value
                    for key, value in row.items()
                    if key != "adapter_row_sha256"
                }
            )
        ),
        "adapter row hash drift",
    )


def validate_adapter_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    require(bool(rows), "adapter prediction rows must not be empty")
    for row in rows:
        validate_adapter_prediction_row(row)
    identifiers = [str(row["adapter_prediction_id"]) for row in rows]
    require(
        len(identifiers) == len(set(identifiers)),
        "duplicate adapter prediction ID",
    )
    identity_tuples = {
        (
            row["adapter_stage"],
            row["adapter_training_seed"],
            row["adapter_checkpoint_path"],
            row["adapter_checkpoint_ledger_sha256"],
            row["adapter_weights_sha256"],
            row["adapter_semantic_config_sha256"],
        )
        for row in rows
    }
    require(
        len(identity_tuples) == 1,
        "prediction file mixes adapter identities",
    )


def adapter_prediction_jsonl_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    validate_adapter_prediction_rows(rows)
    return canonical_jsonl_bytes(rows)


def summarize_adapter_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_adapter_prediction_rows(rows)
    core_rows = [
        {field: row[field] for field in PREDICTION_FIELDS}
        for row in rows
    ]
    core_summary = summarize_prediction_rows(core_rows)
    first = rows[0]
    return {
        **core_summary,
        "adapter_protocol_version": PROTOCOL_VERSION,
        "adapter_stage": first["adapter_stage"],
        "adapter_training_seed": first["adapter_training_seed"],
        "adapter_checkpoint_path": first["adapter_checkpoint_path"],
        "adapter_checkpoint_ledger_sha256": first[
            "adapter_checkpoint_ledger_sha256"
        ],
        "adapter_weights_sha256": first["adapter_weights_sha256"],
        "adapter_semantic_config_sha256": first[
            "adapter_semantic_config_sha256"
        ],
        "condition_counts": dict(
            sorted(Counter(str(row["condition"]) for row in rows).items())
        ),
    }


__all__ = [
    "ADAPTER_FIELDS",
    "ADAPTER_PREDICTION_PREFIX",
    "AdapterInferenceContractError",
    "PREDICTION_FIELDS_WITH_ADAPTER",
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "adapter_prediction_jsonl_bytes",
    "summarize_adapter_prediction_rows",
    "validate_adapter_prediction_row",
    "validate_adapter_prediction_rows",
    "wrap_prediction_row",
]
