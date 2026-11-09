"""Gradient buckets, autograd hooks, and asynchronous all-reduce.

This is the heart of MiniDDP. The implementation mirrors the three important
ideas in PyTorch DDP:

1. Register a hook on every parameter. The hook runs when that parameter's
   gradient has been accumulated by autograd.
2. Copy the gradient into a flat bucket buffer. Launch all-reduce as soon as
   every parameter in that bucket is ready.
3. Wait for all bucket all-reduces in finalize(), then copy reduced gradients
   back into param.grad before the optimizer step.

Buckets are built in reverse parameter order because backward computes the
last layers first. That ordering lets communication for late layers overlap
with backward computation for earlier layers.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist


@dataclass
class BucketStats:
    """Communication counters for one or more optimizer steps."""

    allreduce_calls: int = 0
    payload_bytes: int = 0
    wait_time_s: float = 0.0

    def reset(self) -> None:
        self.allreduce_calls = 0
        self.payload_bytes = 0
        self.wait_time_s = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "allreduce_calls": float(self.allreduce_calls),
            "payload_mb": self.payload_bytes / (1024.0 * 1024.0),
            "wait_time_ms": self.wait_time_s * 1000.0,
        }


class GradientBucket:
    """A flat buffer plus views that alias each parameter's gradient slot."""

    def __init__(self, params: List[torch.nn.Parameter]):
        if not params:
            raise ValueError("a bucket needs at least one parameter")

        self.params = params
        self.device = params[0].device
        self.dtype = params[0].dtype
        self.numel = sum(p.numel() for p in params)

        self.buffer = torch.empty(self.numel, device=self.device, dtype=self.dtype)
        self.views: List[torch.Tensor] = []
        self.param_to_view: Dict[int, torch.Tensor] = {}

        offset = 0
        for param in params:
            if param.device != self.device:
                raise ValueError(
                    f"parameter on {param.device} cannot share a bucket on {self.device}"
                )
            if param.dtype != self.dtype:
                raise ValueError(
                    f"parameter dtype {param.dtype} cannot share a bucket of {self.dtype}"
                )
            n = param.numel()
            view = self.buffer[offset : offset + n].view_as(param)
            self.views.append(view)
            self.param_to_view[id(param)] = view
            offset += n

        self.ready_ids: set[int] = set()
        self.work: Optional[object] = None

    @property
    def nbytes(self) -> int:
        return self.buffer.numel() * self.buffer.element_size()

    def reset(self) -> None:
        self.ready_ids.clear()
        self.work = None

    def __repr__(self) -> str:
        return (
            f"GradientBucket(params={len(self.params)}, numel={self.numel}, "
            f"bytes={self.nbytes}, ready={len(self.ready_ids)})"
        )


class Reducer:
    """Owns buckets and parameter hooks for one model replica."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        bucket_cap_mb: float = 1.0,
        world_size: Optional[int] = None,
        divide_by_world_size: bool = True,
    ):
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("the model has no trainable parameters")

        self.world_size = world_size if world_size is not None else dist.get_world_size()
        self.divide_by_world_size = divide_by_world_size
        self.bucket_cap_bytes = max(1, int(bucket_cap_mb * 1024 * 1024))

        self.buckets = self._build_buckets()
        self.sync_enabled = True
        self.stats = BucketStats()

        # Keep hook handles alive for the lifetime of the reducer.
        self._hook_handles: List[object] = []
        self._register_hooks()

    # ------------------------------------------------------------------ setup
    def _build_buckets(self) -> List[GradientBucket]:
        # Group by (device, dtype) first, then chunk into size-limited buckets.
        groups: Dict[tuple, List[torch.nn.Parameter]] = defaultdict(list)
        for param in reversed(self.params):
            groups[(param.device, param.dtype)].append(param)

        buckets: List[GradientBucket] = []
        for group in groups.values():
            current: List[torch.nn.Parameter] = []
            current_bytes = 0
            for param in group:
                nbytes = param.numel() * param.element_size()
                if current and current_bytes + nbytes > self.bucket_cap_bytes:
                    buckets.append(GradientBucket(current))
                    current = []
                    current_bytes = 0
                current.append(param)
                current_bytes += nbytes
            if current:
                buckets.append(GradientBucket(current))
        return buckets

    def _register_hooks(self) -> None:
        for bucket in self.buckets:
            for param in bucket.params:
                handle = param.register_post_accumulate_grad_hook(
                    self._make_hook(bucket)
                )
                self._hook_handles.append(handle)

    # --------------------------------------------------------------- backward
    def _make_hook(self, bucket: GradientBucket):
        def hook(param: torch.nn.Parameter) -> None:
            # no_sync(): let .grad accumulate. The next synchronized backward
            # will copy the accumulated gradient into the bucket.
            if not self.sync_enabled:
                return

            if param.grad is None:
                raise RuntimeError(
                    "MiniDDP hook fired but param.grad is None; "
                    "this should not happen for a used parameter"
                )

            # Defensive: a parameter may be used more than once in the graph.
            if id(param) in bucket.ready_ids:
                return
            bucket.ready_ids.add(id(param))

            view = bucket.param_to_view[id(param)]
            with torch.no_grad():
                view.copy_(param.grad)

            if len(bucket.ready_ids) == len(bucket.params):
                if self.world_size > 1:
                    bucket.work = dist.all_reduce(
                        bucket.buffer,
                        op=dist.ReduceOp.SUM,
                        async_op=True,
                    )
                    self.stats.allreduce_calls += 1
                    self.stats.payload_bytes += bucket.nbytes
                # world_size == 1: no communication, finalize() just copies back.

        return hook

    # ---------------------------------------------------------------- finalize
    def finalize(self) -> None:
        """Wait for all asynchronous all-reduces and copy gradients back.

        Must be called before optimizer.step() and before gradient clipping.
        """
        if not self.sync_enabled:
            raise RuntimeError("finalize() cannot be called inside no_sync()")

        wait_start = time.perf_counter()

        for bucket in self.buckets:
            if len(bucket.ready_ids) != len(bucket.params):
                missing = len(bucket.params) - len(bucket.ready_ids)
                raise RuntimeError(
                    f"bucket is not ready ({missing} parameter(s) missing grads). "
                    "MiniDDP requires every trainable parameter to participate in "
                    "the forward/backward pass. `find_unused_parameters` is not "
                    "implemented yet."
                )

            if self.world_size > 1 and bucket.work is None:
                raise RuntimeError(
                    "bucket is ready but has no all-reduce work; "
                    "this indicates a hook/state bug"
                )

            if bucket.work is not None:
                bucket.work.wait()

            if self.divide_by_world_size and self.world_size > 1:
                bucket.buffer.div_(self.world_size)

            with torch.no_grad():
                for param, view in zip(bucket.params, bucket.views):
                    if param.grad is None:
                        raise RuntimeError("param.grad became None during finalize")
                    param.grad.copy_(view)

            bucket.reset()

        self.stats.wait_time_s += time.perf_counter() - wait_start

    # ------------------------------------------------------------------- stats
    def reset_stats(self) -> None:
        self.stats.reset()

    @property
    def num_buckets(self) -> int:
        return len(self.buckets)

    def bucket_summary(self) -> List[dict]:
        return [
            {
                "num_params": len(bucket.params),
                "numel": bucket.numel,
                "bytes": bucket.nbytes,
            }
            for bucket in self.buckets
        ]
