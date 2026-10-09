#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python benchmarks/benchmark.py --seq-lens 128,256,512 --batch 1 --heads 8 --head-dim 64 --dtype fp16 --causal --output benchmark-results/fp16-report.json
