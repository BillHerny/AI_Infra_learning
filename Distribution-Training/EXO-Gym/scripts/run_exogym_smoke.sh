#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh
export MPLBACKEND=Agg

if ! "$PYTHON_BIN" -c "import exogym" >/dev/null 2>&1; then
  echo "exogym is not installed in: $PYTHON_BIN" >&2
  exit 1
fi

echo "== 10-step EXO Gym smoke test: 2 nodes, DDP only =="
"$PYTHON_BIN" experiments/mnist_exogym.py \
  --num-nodes 2 \
  --max-steps 10 \
  --batch-size 64 \
  --minibatch-size 64 \
  --val-size 512 \
  --val-interval 10 \
  --strategies ddp \
  --run-tag smoke \
  --plot
