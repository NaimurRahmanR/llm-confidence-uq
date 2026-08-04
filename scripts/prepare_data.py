from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_confidence_uq.data import (  # noqa: E402
    EligibleRecord,
    build_research_manifest,
    canonical_source_record,
    class_counts,
    manifest_jsonl,
    sha256_text,
    source_split_sha256,
    summary_json,
    validate_manifest_rows,
)

EXPECTED_CLASSES = {
    "train": {False: 3553, True: 5874},
    "validation": {False: 1237, True: 2033},
}
EXPECTED_CONFIG_SHA256 = "412fd6da74212e071159463e104329efefbc8fbe6b853c29c825e3ab37cb64ac"
EXPECTED_DATA_MODULE_SHA256 = "542157fa0c40853118a6607aa4f265f4e5993e00f88a6b97f2b31f9f6ec78eac"
EXPECTED_TEST_MODULE_SHA256 = "3ee20b48a377a293264ea967f54b284373f0a729475af267badb03c58aaa0e9b"
EXPECTED_DATASET_ID = "google/boolq"
EXPECTED_DATASET_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
EXPECTED_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EXPECTED_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EXPECTED_SYSTEM = (
    "Use only the supplied passage to answer the binary question. Return exactly two "
    "lines. Line 1 must be 'Answer: Yes' or 'Answer: No'. Line 2 must be "
    "'Confidence: N', where N is an integer from 0 to 100."
)
EXPECTED_VERSIONS = {
    "datasets": "4.0.0",
    "huggingface-hub": "1.23.0",
    "numpy": "2.0.2",
    "PyYAML": "6.0.3",
    "transformers": "5.13.1",
}
EXPECTED_EXCLUDED = {"train": 4, "validation": 2}
EXPECTED_PROMPT_LENGTHS = {
    "train": {
        "min": 93, "p50": 201, "p90": 302, "p95": 343, "p99": 445,
        "max": 962, "over_512": 41, "over_768": 4, "over_1024": 0,
    },
    "validation": {
        "min": 100, "p50": 199, "p90": 303, "p95": 343, "p99": 451,
        "max": 1336, "over_512": 18, "over_768": 2, "over_1024": 1,
    },
}
EXPECTED_SELECTED = {
    "train": {False: 301, True: 499},
    "calibration": {False: 75, True: 125},
    "test": {False: 151, True: 249},
}
EXPECTED_SOURCE = {"train": "train", "calibration": "train", "test": "validation"}
FORBIDDEN_FIELDS = {
    "question", "question_text", "passage", "passage_text", "prompt",
    "rendered_prompt", "input_text", "raw_text", "text", "messages",
}
TOKENIZATION = {
    "apply_chat_template": True,
    "add_generation_prompt": True,
    "add_special_tokens": False,
    "padding": False,
    "truncation": False,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def render_prompt(
    tokenizer,
    system,
    user_template,
    passage,
    question,
    prefix,
):
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": user_template.format(
                passage=passage,
                question=question,
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) + prefix


def input_digest(record):
    return sha256_text(
        json.dumps(
            {"passage": record.passage, "question": record.question},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
    )


def resolve_revision(api, repo_type, repo_id, revision):
    info = (
        api.dataset_info(repo_id=repo_id, revision=revision)
        if repo_type == "dataset"
        else api.model_info(repo_id=repo_id, revision=revision)
    )
    require(info.sha == revision, f"{repo_type} revision mismatch: {info.sha}")
    return info.sha


def make_records(split, split_name, cfg, tokenizer):
    records = []
    for start in range(0, len(split), 128):
        batch = split[start : min(start + 128, len(split))]
        require(
            all(isinstance(value, str) for value in batch["question"]),
            f"Non-string question in {split_name}",
        )
        require(
            all(isinstance(value, str) for value in batch["passage"]),
            f"Non-string passage in {split_name}",
        )
        require(
            all(isinstance(value, bool) for value in batch["answer"]),
            f"Non-boolean answer in {split_name}",
        )
        prompts = [
            render_prompt(
                tokenizer,
                cfg["prompt"]["system"],
                cfg["prompt"]["user_template"],
                passage,
                question,
                cfg["prompt"]["classification_prefix"],
            )
            for question, passage in zip(batch["question"], batch["passage"], strict=True)
        ]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        require(len(encoded) == len(prompts), f"Tokenizer batch mismatch in {split_name}")
        require(
            all(isinstance(ids, list) and all(isinstance(i, int) for i in ids) for ids in encoded),
            f"Unexpected tokenizer output type in {split_name}",
        )
        for offset, (question, passage, answer, ids) in enumerate(
            zip(batch["question"], batch["passage"], batch["answer"], encoded, strict=True)
        ):
            records.append(
                EligibleRecord(
                    dataset_id=cfg["data"]["dataset_id"],
                    dataset_revision=cfg["data"]["dataset_revision"],
                    source_split=split_name,
                    source_index=start + offset,
                    question=question,
                    passage=passage,
                    answer=answer,
                    prompt_tokens=len(ids),
                )
            )
    require(len(records) == len(split), f"Conversion failed for {split_name}")
    return records


def stated_label(row):
    for key in ("label", "answer"):
        if key in row:
            require(row[key] in (False, True, 0, 1), f"Invalid label: {row[key]!r}")
            return bool(row[key])
    return None


def build_once(dataset, tokenizer, cfg):
    selection = cfg["selection"]
    eligibility = int(selection["original_prompt_eligibility_max_tokens"])
    sizes = {
        "train": int(selection["train_size"]),
        "calibration": int(selection["calibration_size"]),
        "test": int(selection["test_size"]),
    }
    records = {
        name: make_records(dataset[name], name, cfg, tokenizer)
        for name in ("train", "validation")
    }

    source_input_hashes = {}
    all_source_input_hashes = set()
    for name, source_records in records.items():
        hashes = [input_digest(record) for record in source_records]
        require(len(hashes) == len(set(hashes)), f"Duplicate source input in {name}")
        source_input_hashes[name] = set(hashes)
        require(
            all_source_input_hashes.isdisjoint(source_input_hashes[name]),
            f"Question-passage overlap involving source split {name}",
        )
        all_source_input_hashes.update(source_input_hashes[name])

    source_report, excluded = {}, {}
    for name, source_records in records.items():
        expected_rows = int(cfg["data"]["source_expected_rows"][name])
        require(len(source_records) == expected_rows, f"Wrong {name} row count")
        counts = class_counts(source_records)
        counts = {False: counts.get(False, 0), True: counts.get(True, 0)}
        require(counts == EXPECTED_CLASSES[name], f"Wrong {name} class counts: {counts}")
        source_hash = source_split_sha256(source_records)
        require(
            source_hash == cfg["data"]["source_expected_sha256"][name],
            f"Wrong {name} source SHA-256: {source_hash}",
        )
        excluded[name] = [
            {
                "answer": record.answer,
                "prompt_tokens": record.prompt_tokens,
                "source_index": record.source_index,
                "source_record_sha256": sha256_text(
                    canonical_source_record(record.question, record.passage, record.answer)
                ),
            }
            for record in source_records
            if record.prompt_tokens > eligibility
        ]
        require(
            len(excluded[name]) == EXPECTED_EXCLUDED[name],
            f"{name} exclusions differ from the prior audit: {len(excluded[name])}",
        )
        lengths = np.asarray(
            [record.prompt_tokens for record in source_records],
            dtype=np.int64,
        )
        prompt_statistics = {
            "min": int(lengths.min()),
            "p50": int(np.percentile(lengths, 50, method="nearest")),
            "p90": int(np.percentile(lengths, 90, method="nearest")),
            "p95": int(np.percentile(lengths, 95, method="nearest")),
            "p99": int(np.percentile(lengths, 99, method="nearest")),
            "max": int(lengths.max()),
            "over_512": int((lengths > 512).sum()),
            "over_768": int((lengths > 768).sum()),
            "over_1024": int((lengths > 1024).sum()),
        }
        require(
            prompt_statistics == EXPECTED_PROMPT_LENGTHS[name],
            f"Prompt-length audit mismatch for {name}: {prompt_statistics}",
        )
        source_report[name] = {
            "rows": len(source_records),
            "class_counts": {"false": counts[False], "true": counts[True]},
            "canonical_sha256": source_hash,
            "eligible_rows": len(source_records) - len(excluded[name]),
            "excluded_rows": len(excluded[name]),
            "prompt_token_statistics": prompt_statistics,
        }

    eligible_records = {
        name: [
            record
            for record in source_records
            if record.prompt_tokens <= eligibility
        ]
        for name, source_records in records.items()
    }
    require(
        len(eligible_records["train"])
        == len(records["train"]) - EXPECTED_EXCLUDED["train"],
        "Eligible train count mismatch",
    )
    require(
        len(eligible_records["validation"])
        == len(records["validation"]) - EXPECTED_EXCLUDED["validation"],
        "Eligible validation count mismatch",
    )

    rows, summary = build_research_manifest(
        eligible_records["train"], eligible_records["validation"],
        train_size=sizes["train"], calibration_size=sizes["calibration"],
        test_size=sizes["test"], selection_seed=int(selection["seed"]),
        eligibility_max_prompt_tokens=eligibility,
    )
    validate_manifest_rows(
        rows, expected_sizes=sizes, selection_seed=int(selection["seed"]),
        eligibility_max_prompt_tokens=eligibility,
    )

    fields = sorted(rows[0])
    require(all(sorted(row) == fields for row in rows), "Manifest schema varies by row")
    require(
        set(map(str.lower, fields)).isdisjoint(FORBIDDEN_FIELDS),
        "A raw-text field is present in the manifest",
    )

    lookup = {
        (record.source_split, record.source_index): record
        for source_records in records.values() for record in source_records
    }
    content_hashes, split_report = {}, {}
    for research_split, size in sizes.items():
        selected = [row for row in rows if row["research_split"] == research_split]
        require(len(selected) == size, f"Wrong {research_split} size")
        require(
            all(row["source_split"] == EXPECTED_SOURCE[research_split] for row in selected),
            f"Wrong official source for {research_split}",
        )
        content_hashes[research_split] = set()
        selected_answers = []
        for row in selected:
            location = (row["source_split"], int(row["source_index"]))
            record = lookup[location]
            selected_answers.append(record.answer)
            if stated_label(row) is not None:
                require(stated_label(row) == record.answer, f"Label mismatch at {location}")
            require(row["prompt_tokens"] == record.prompt_tokens, f"Length mismatch at {location}")
            require(record.prompt_tokens <= eligibility, f"Ineligible row selected at {location}")
            require(record.question not in row.values(), f"Question leaked at {location}")
            require(record.passage not in row.values(), f"Passage leaked at {location}")
            input_hash = input_digest(record)
            require(input_hash not in content_hashes[research_split], "Duplicate selected input")
            content_hashes[research_split].add(input_hash)
        counts_counter = Counter(selected_answers)
        counts = {False: counts_counter[False], True: counts_counter[True]}
        require(counts == EXPECTED_SELECTED[research_split], f"Wrong {research_split} counts")
        split_report[research_split] = {
            "rows": len(selected),
            "class_counts": {"false": counts[False], "true": counts[True]},
            "maximum_prompt_tokens": max(row["prompt_tokens"] for row in selected),
        }

    split_names = list(sizes)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            require(content_hashes[left].isdisjoint(content_hashes[right]), f"Leak: {left}/{right}")

    payloads = {
        f"{name}.jsonl": manifest_jsonl(
            [row for row in rows if row["research_split"] == name]
        ).encode("utf-8")
        for name in sizes
    }
    payloads["summary.json"] = summary_json(summary).encode("utf-8")
    report = {
        "excluded": excluded, "manifest_fields": fields,
        "source": source_report, "splits": split_report,
    }
    return rows, payloads, report


def write_verified(output_dir, payloads):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = output_dir / name
        require(not path.is_symlink(), f"Artifact target may not be a symlink: {path}")
        if path.exists() and path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite non-identical artifact: {path}")

    statuses = {}
    for name, payload in payloads.items():
        path = output_dir / name
        if path.exists():
            statuses[name] = "verified_existing"
            continue
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=output_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            require(not path.exists(), f"Target appeared during write: {path}")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        statuses[name] = "created"

    require(
        all((output_dir / name).read_bytes() == payload for name, payload in payloads.items()),
        "Read-back mismatch",
    )
    return statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config_argument = args.config.absolute()
    output_argument = args.output_dir.absolute()
    require(not config_argument.is_symlink(), "Config path may not be a symlink")
    require(not output_argument.is_symlink(), "Output directory may not be a symlink")
    config_path, output_dir = config_argument.resolve(), output_argument.resolve()
    require(
        config_path == (ROOT / "configs/data.yaml").resolve(),
        "Config path must be repository configs/data.yaml",
    )
    require(
        output_dir == (ROOT / "data/manifests").resolve(),
        "Output directory must be repository data/manifests",
    )
    require(digest(config_path.read_bytes()) == EXPECTED_CONFIG_SHA256, "Config SHA-256 changed")

    data_module_path = ROOT / "src/llm_confidence_uq/data.py"
    test_module_path = ROOT / "tests/test_splits.py"
    require(
        digest(data_module_path.read_bytes()) == EXPECTED_DATA_MODULE_SHA256,
        "Tested data module SHA-256 changed",
    )
    require(
        digest(test_module_path.read_bytes()) == EXPECTED_TEST_MODULE_SHA256,
        "Split-test module SHA-256 changed",
    )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(cfg["schema_version"] == 1, "Unsupported config schema")
    require(cfg["selection"]["algorithm"] == "sha256-stratified-v1", "Wrong algorithm")
    require(cfg["data"]["dataset_id"] == EXPECTED_DATASET_ID, "Dataset ID changed")
    require(cfg["data"]["dataset_revision"] == EXPECTED_DATASET_REVISION, "Dataset revision changed")
    require(cfg["data"]["licence"] == "cc-by-sa-3.0", "Dataset licence changed")
    require(cfg["model"]["id"] == EXPECTED_MODEL_ID, "Model ID changed")
    require(cfg["model"]["revision"] == EXPECTED_MODEL_REVISION, "Model revision changed")
    require(cfg["model"]["tokenizer_id"] == EXPECTED_MODEL_ID, "Tokenizer ID changed")
    require(cfg["model"]["tokenizer_revision"] == EXPECTED_MODEL_REVISION, "Tokenizer revision changed")
    require(
        cfg["prompt"] == {
            "renderer_version": "qwen-chat-boolq-confidence-v1",
            "system": EXPECTED_SYSTEM,
            "user_template": "Passage: {passage}\nQuestion: {question}",
            "classification_prefix": "Answer:",
            "apply_chat_template": True,
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
        },
        "Prompt configuration changed",
    )
    require(
        cfg["selection"] == {
            "algorithm": "sha256-stratified-v1", "seed": 20260803,
            "train_size": 800, "calibration_size": 200, "test_size": 400,
            "original_prompt_eligibility_max_tokens": 768,
            "runtime_max_sequence_tokens": 1024,
        },
        "Selection configuration changed",
    )
    require(
        cfg["selection"]["original_prompt_eligibility_max_tokens"]
        <= cfg["selection"]["runtime_max_sequence_tokens"],
        "Eligibility ceiling exceeds runtime ceiling",
    )

    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    require(versions == EXPECTED_VERSIONS, f"Package versions changed: {versions}")
    python_version = sys.version.split()[0]
    require(python_version == "3.12.13", f"Python version changed: {python_version}")

    api = HfApi(token=False)
    dataset_revision = resolve_revision(
        api, "dataset", cfg["data"]["dataset_id"], cfg["data"]["dataset_revision"]
    )
    tokenizer_revision = resolve_revision(
        api, "model", cfg["model"]["tokenizer_id"], cfg["model"]["tokenizer_revision"]
    )
    dataset = load_dataset(
        cfg["data"]["dataset_id"], revision=dataset_revision, token=False
    )
    require(set(dataset) == {"train", "validation"}, f"Unexpected splits: {list(dataset)}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["tokenizer_id"], revision=tokenizer_revision,
        trust_remote_code=False, token=False,
    )
    require(
        tokenizer.__class__.__name__ == "Qwen2Tokenizer",
        f"Tokenizer class changed: {tokenizer.__class__.__name__}",
    )

    rows_a, payloads_a, report_a = build_once(dataset, tokenizer, cfg)
    rows_b, payloads_b, report_b = build_once(dataset, tokenizer, cfg)
    require(
        rows_a == rows_b and payloads_a == payloads_b and report_a == report_b,
        "Two repeat in-process builds differ",
    )
    for name in ("train", "calibration", "test"):
        candidate = [
            json.loads(line)
            for line in payloads_a[f"{name}.jsonl"].decode().splitlines()
        ]
        expected = [row for row in rows_a if row["research_split"] == name]
        require(candidate == expected, f"Candidate serialization mismatch: {name}")

    template = render_prompt(
        tokenizer,
        cfg["prompt"]["system"],
        cfg["prompt"]["user_template"],
        "{passage}",
        "{question}",
        cfg["prompt"]["classification_prefix"],
    )
    provenance = {
        "schema_version": 1,
        "dataset": {
            "id": cfg["data"]["dataset_id"],
            "licence": cfg["data"]["licence"],
            "requested_revision": cfg["data"]["dataset_revision"],
            "resolved_revision": dataset_revision,
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "id": cfg["model"]["tokenizer_id"],
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "requested_revision": cfg["model"]["tokenizer_revision"],
            "resolved_revision": tokenizer_revision,
        },
        "model": {
            "id": cfg["model"]["id"],
            "requested_revision": cfg["model"]["revision"],
            "resolved_revision": tokenizer_revision,
            "weights_loaded_during_data_preparation": False,
        },
        "prompt_renderer": {
            "version": "qwen-chat-boolq-confidence-v1",
            "template": template,
            "template_sha256": sha256_text(template),
            "tokenization": TOKENIZATION,
        },
        "selection": {
            **cfg["selection"],
            "test_label_policy": (
                "Validation labels are used only for source-integrity checks and the "
                "predeclared deterministic stratified test sample, including count "
                "verification; they must not be used for calibration, hyperparameter "
                "tuning, checkpoint selection, or method revision."
            ),
        },
        "verification": {"repeat_in_process_builds": 2, **report_a},
        "implementation_sha256": {
            "config": digest(config_path.read_bytes()),
            "data_module": digest(data_module_path.read_bytes()),
            "prepare_data": digest(Path(__file__).read_bytes()),
            "test_splits": digest(test_module_path.read_bytes()),
        },
        "software_versions": versions,
        "python_version": python_version,
        "artifact_sha256": {
            name: digest(payload) for name, payload in payloads_a.items()
        },
    }

    payloads = {**payloads_a, "provenance.json": json_bytes(provenance)}
    ledger = {
        name: {"bytes": len(payload), "sha256": digest(payload)}
        for name, payload in sorted(payloads.items())
    }
    payloads["artifact_hashes.json"] = json_bytes(ledger)
    statuses = write_verified(output_dir, payloads)

    reloaded = []
    for name in ("train", "calibration", "test"):
        raw = (output_dir / f"{name}.jsonl").read_bytes()
        require(raw.endswith(b"\n"), f"No terminal newline: {name}")
        parsed = [json.loads(line) for line in raw.decode().splitlines()]
        expected = [row for row in rows_a if row["research_split"] == name]
        require(parsed == expected, f"Parsed round-trip mismatch: {name}")
        reloaded.extend(parsed)

    validate_manifest_rows(
        reloaded,
        expected_sizes={
            key: cfg["selection"][f"{key}_size"]
            for key in ("train", "calibration", "test")
        },
        selection_seed=cfg["selection"]["seed"],
        eligibility_max_prompt_tokens=(
            cfg["selection"]["original_prompt_eligibility_max_tokens"]
        ),
    )

    print("DATA MANIFEST INTEGRATION: PASS")
    print("dataset_revision:", dataset_revision)
    print("tokenizer_revision:", tokenizer_revision)
    print("tokenizer_class:", tokenizer.__class__.__name__)
    for name, report in report_a["source"].items():
        print("source", name, report)
    for name, report in report_a["splits"].items():
        print("research_split", name, report)
    print("manifest_fields:", report_a["manifest_fields"])
    print("repeat_in_process_builds_byte_identical: True")
    for name in sorted(payloads):
        print(
            name, statuses[name], "bytes=", len(payloads[name]),
            "sha256=", digest(payloads[name]),
        )
    print("round_trip_validation: True")
    print("SCOPE: original-data manifests passed; degradation checks remain pending.")


if __name__ == "__main__":
    main()
