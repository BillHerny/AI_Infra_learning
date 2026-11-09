#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "== MiniDDP on the single RTX 4060, 2 virtual ranks (Gloo) =="
"$PYTHON_BIN" examples/train_mnist.py \
  --impl miniddp \
  --world-size 2 \
  --device cuda \
  --backend gloo \
  --max-steps 20 \
  --batch-size 64 \
  --bucket-mb 1.0 \
  --output results/miniddp_gpu.pt

echo
echo "== torch DDP reference on the same single GPU (Gloo) =="
"$PYTHON_BIN" examples/train_mnist.py \
  --impl torchddp \
  --world-size 2 \
  --device cuda \
  --backend gloo \
  --max-steps 20 \
  --batch-size 64 \
  --bucket-mb 1.0 \
  --output results/torchddp_gpu.pt

echo
echo "== GPU correctness comparison =="
RUN_GPU_TESTS=1 "$PYTHON_BIN" -m pytest tests -q
