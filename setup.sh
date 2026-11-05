#!/usr/bin/env bash
# Install from PyPI only (Python 3.12). Tests run on CPU; training and evaluation also need vLLM on a Linux GPU.
set -euo pipefail
python -m pip install 'torch>=2.4' 'transformers>=4.51' 'peft>=0.13' 'datasets>=3.0' 'math-verify>=0.7' 'pytest>=8'
