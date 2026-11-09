"""Process-group and model-broadcast helpers."""

from __future__ import annotations

import os
import socket
from typing import Optional

import torch
import torch.distributed as dist


def find_free_port() -> int:
    """Ask the OS for an unused localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def init_process_group(
    rank: int,
    world_size: int,
    port: Optional[int] = None,
    backend: str = "gloo",
    master_addr: str = "127.0.0.1",
) -> None:
    """Initialize the default process group.

    Gloo is the safe default on this machine because a single RTX 4060 is
    shared by all virtual ranks. NCCL rejects two ranks on the same GPU with
    "Duplicate GPU detected", while Gloo supports CUDA tensors and works.
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if port is None:
        port = find_free_port()

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def broadcast_model(model: torch.nn.Module, src: int = 0) -> None:
    """Make every rank start from rank 0's parameters and buffers."""
    if not dist.is_initialized():
        return
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param.data, src=src)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=src)


def cleanup() -> None:
    """Barrier and destroy the process group, tolerating partial failures."""
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        finally:
            dist.destroy_process_group()
