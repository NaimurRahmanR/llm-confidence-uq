#!/usr/bin/env bash
set -euo pipefail

# Reconstruct the verified Python 3.12 / CUDA 12.8 Colab environment.
# Run from the repository root in a fresh GPU-enabled Colab runtime.
python -m pip install --upgrade pip
python -m pip uninstall -y torchao
python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0"
python -m pip install -r requirements.txt
python -m pip install "jedi==0.19.2"
python colab/verify_environment.py
