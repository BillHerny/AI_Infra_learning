"""Small spawn helper used by examples and benchmarks."""

from __future__ import annotations

from typing import Any, Callable

import torch.multiprocessing as mp


def spawn(fn: Callable[..., Any], world_size: int, args: tuple = ()) -> None:
    """Spawn one process per virtual rank.

    `start_method="spawn"` is the safest choice when CUDA is involved. The
    target function and all arguments must be picklable, and callers should run
    from a real .py file (not stdin or a notebook).
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    mp.spawn(
        fn,
        args=(world_size, *args),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
