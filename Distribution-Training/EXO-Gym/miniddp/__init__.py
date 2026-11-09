"""MiniDDP: a small, readable DDP implementation for learning.

The package intentionally implements only the core mechanics:
- process-group setup with Gloo (works for CPU and single-GPU multi-process)
- parameter/buffer broadcast at startup
- per-parameter autograd hooks
- gradient buckets in reverse parameter order
- asynchronous all-reduce with overlap
- no_sync() for gradient accumulation
"""

from .ddp import MiniDDP
from .reducer import BucketStats, GradientBucket, Reducer

__all__ = ["MiniDDP", "Reducer", "GradientBucket", "BucketStats"]
