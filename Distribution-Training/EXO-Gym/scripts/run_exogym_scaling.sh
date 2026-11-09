#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/common.sh
export MPLBACKEND=Agg

if ! "$PYTHON_BIN" -c "import exogym" >/dev/null 2>&1; then
  echo "exogym is not installed in: $PYTHON_BIN" >&2
  exit 1
fi

# Keep the global batch fixed at 1024 so the N sweep is comparable.
for n in 1 2 4; do
  bs=$((1024 / n))
  mb=64
  if [ "$bs" -lt "$mb" ]; then
    mb=$bs
  fi

  for strategy in ddp diloco sparta; do
    echo
    echo "== EXO Gym N=${n} strategy=${strategy} batch/rank=${bs} =="
    "$PYTHON_BIN" experiments/mnist_exogym.py \
      --num-nodes "$n" \
      --max-steps 150 \
      --batch-size "$bs" \
      --minibatch-size "$mb" \
      --val-size 2000 \
      --val-interval 25 \
      --strategies "$strategy" \
      --run-tag "n${n}" \
      --plot
  done
done

# Optional 8-node smoke run; uncomment only after N=2/4 are stable.
# "$PYTHON_BIN" experiments/mnist_exogym.py \
#   --num-nodes 8 --max-steps 50 --batch-size 128 --minibatch-size 32 \
#   --val-size 1024 --val-interval 25 --strategies ddp --run-tag n8 --plot
