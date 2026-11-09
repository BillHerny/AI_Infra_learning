#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "== correctness tests: MiniDDP vs torch DDP =="
"$PYTHON_BIN" -m pytest tests -q

echo
echo "== MiniDDP training smoke test on CPU (2 virtual ranks) =="
"$PYTHON_BIN" examples/train_mnist.py \
  --impl miniddp \
  --world-size 2 \
  --device cpu \
  --max-steps 20 \
  --batch-size 64 \
  --bucket-mb 1.0 \
  --output results/miniddp_cpu.pt
