"""Pytest wrappers around tests/compare_impls.py.

Each case runs in a subprocess because torch.multiprocessing.spawn needs a real
Python file as the entry point. This also keeps pytest's own process clean of
CUDA/process-group state.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = PROJECT_ROOT / "tests" / "compare_impls.py"


def run_compare(*args: str) -> None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        [sys.executable, str(COMPARE_SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            "compare_impls failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def test_single_step_world2_small_buckets() -> None:
    run_compare(
        "--world-size", "2",
        "--steps", "1",
        "--bucket-mb", "0.001",
    )


def test_two_steps_world2_one_mb_bucket() -> None:
    run_compare(
        "--world-size", "2",
        "--steps", "2",
        "--bucket-mb", "1.0",
    )


def test_no_sync_gradient_accumulation() -> None:
    run_compare(
        "--world-size", "2",
        "--steps", "2",
        "--accum-steps", "3",
        "--bucket-mb", "0.01",
    )


def test_four_ranks() -> None:
    run_compare(
        "--world-size", "4",
        "--steps", "2",
        "--bucket-mb", "0.01",
    )


@pytest.mark.skipif(
    os.environ.get("RUN_GPU_TESTS") != "1",
    reason="set RUN_GPU_TESTS=1 to run the single-GPU Gloo comparison",
)
def test_gpu_two_ranks() -> None:
    run_compare(
        "--world-size", "2",
        "--steps", "1",
        "--bucket-mb", "0.01",
        "--device", "cuda",
    )
