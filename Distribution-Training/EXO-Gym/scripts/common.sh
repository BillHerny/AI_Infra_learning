#!/usr/bin/env bash
# Shared interpreter selection for all run_*.sh scripts.
#
# The project is installed in the conda environment named "exogym". Using an
# absolute interpreter path here means the scripts work even when the user
# forgot to `conda activate exogym` in the current terminal.
#
# Override if needed:
#   PYTHON_BIN=/path/to/python bash scripts/run_exogym_smoke.sh
PYTHON_BIN="${PYTHON_BIN:-/home/fate-whispering/miniconda3/envs/exogym/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  echo "Install the exogym env or set PYTHON_BIN explicitly." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  echo "torch cannot be imported by: $PYTHON_BIN" >&2
  exit 1
fi
