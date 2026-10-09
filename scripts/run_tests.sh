#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements-test.txt
# Requires an already-installed CUDA-enabled PyTorch build and a working nvcc.
python -m pip install -e . --no-build-isolation
pytest -v tests
