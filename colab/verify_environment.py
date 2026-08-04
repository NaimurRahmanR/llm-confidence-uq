#!/usr/bin/env python3
"""Fail-closed verification for the documented Colab experiment runtime."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import sys

import torch


EXPECTED = {
    "accelerate": "1.14.0",
    "datasets": "4.0.0",
    "huggingface-hub": "1.23.0",
    "matplotlib": "3.10.0",
    "numpy": "2.0.2",
    "pandas": "2.2.2",
    "peft": "0.19.1",
    "PyYAML": "6.0.3",
    "safetensors": "0.8.0",
    "scikit-learn": "1.6.1",
    "seaborn": "0.13.2",
    "torch": "2.11.0+cu128",
    "transformers": "5.13.1",
}


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"expected Python 3.12, observed {platform.python_version()}")
    observed = {name: importlib.metadata.version(name) for name in EXPECTED}
    mismatches = {
        name: {"expected": expected, "observed": observed[name]}
        for name, expected in EXPECTED.items()
        if observed[name] != expected
    }
    if mismatches:
        raise RuntimeError("package version drift: " + json.dumps(mismatches, sort_keys=True))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"expected CUDA build 12.8, observed {torch.version.cuda}")
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": observed,
        "cuda_available": True,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "torchao_installed": importlib.util.find_spec("torchao") is not None,
    }
    if report["torchao_installed"]:
        raise RuntimeError("torchao must remain absent; PEFT rejected the Colab-provided 0.10.0 build")
    print("COLAB ENVIRONMENT CONTRACT: PASS")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
