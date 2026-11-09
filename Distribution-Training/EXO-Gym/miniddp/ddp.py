"""A small DDP-like wrapper built on top of Reducer."""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch
import torch.distributed as dist

from .comm import broadcast_model
from .reducer import BucketStats, Reducer


class MiniDDP:
    """Minimal replacement for torch.nn.parallel.DistributedDataParallel.

    Example
    -------
    model = MyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    ddp = MiniDDP(model, optimizer, bucket_cap_mb=1.0)

    loss = criterion(ddp(x), y)
    loss.backward()
    ddp.step()       # waits for all-reduce, then optimizer.step()
    ddp.zero_grad()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        bucket_cap_mb: float = 1.0,
        divide_by_world_size: bool = True,
    ):
        self.module = model
        self.model = model
        self.optimizer = optimizer
        self.reducer = Reducer(
            model.parameters(),
            bucket_cap_mb=bucket_cap_mb,
            divide_by_world_size=divide_by_world_size,
        )

        if dist.is_available() and dist.is_initialized():
            # Match DDP: every rank starts from rank 0's weights and buffers.
            broadcast_model(model, src=0)

    # Let attribute access fall through to the wrapped model: ddp.train(),
    # ddp.eval(), ddp.parameters(), etc. work as users expect.
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in {
            "module",
            "model",
            "optimizer",
            "reducer",
        }:
            raise AttributeError(name)
        return getattr(self.model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    @contextlib.contextmanager
    def no_sync(self) -> Iterator[None]:
        """Skip all-reduce during gradient accumulation.

        Gradients keep accumulating in param.grad. The next backward performed
        outside this context will copy the accumulated gradients into buckets
        and synchronize them.
        """
        previous = self.reducer.sync_enabled
        self.reducer.sync_enabled = False
        try:
            yield
        finally:
            self.reducer.sync_enabled = previous

    def finalize(self) -> None:
        """Wait for bucket all-reduces without stepping the optimizer."""
        self.reducer.finalize()

    def step(self) -> Any:
        """Finalize gradient synchronization, then run the optimizer step."""
        self.reducer.finalize()
        return self.optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def stats(self) -> BucketStats:
        return self.reducer.stats

    def reset_stats(self) -> None:
        self.reducer.reset_stats()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def __repr__(self) -> str:
        return (
            f"MiniDDP(buckets={self.reducer.num_buckets}, "
            f"bucket_cap_mb={self.reducer.bucket_cap_bytes / (1024 * 1024):.3f})"
        )
