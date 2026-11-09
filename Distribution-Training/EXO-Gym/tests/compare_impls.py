"""Compare MiniDDP against torch.nn.parallel.DistributedDataParallel.

The parent process runs the same training job twice (MiniDDP and torch DDP),
each with a fresh set of spawned ranks, then compares the final state dicts.
This is deliberately a separate script so `mp.spawn` always has a real file as
its entry point.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miniddp.comm import cleanup, find_free_port, init_process_group
from miniddp.ddp import MiniDDP
from miniddp.launch import spawn


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 20, hidden: int = 64, classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dataset(seed: int, n: int = 512, in_dim: int = 20, classes: int = 3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=generator)
    w = torch.randn(in_dim, classes, generator=generator)
    logits = x @ w + 0.1 * torch.randn(n, classes, generator=generator)
    y = logits.argmax(dim=1)
    return TensorDataset(x, y)


def worker(
    rank: int,
    world_size: int,
    impl: str,
    bucket_mb: float,
    steps: int,
    accum_steps: int,
    device_str: str,
    seed: int,
    port: int,
    out_path: str,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    if device_str == "cuda":
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    init_process_group(rank, world_size, port=port, backend="gloo")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    dataset = make_dataset(seed)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )

    model = TinyMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    ddp = None
    sync_wrapper = None
    if impl == "miniddp":
        ddp = MiniDDP(model, optimizer, bucket_cap_mb=bucket_mb)
        model = ddp
        sync_wrapper = ddp
    elif impl == "torchddp":
        kwargs = {"bucket_cap_mb": bucket_mb}
        if device.type == "cuda":
            kwargs["device_ids"] = [0]
        model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
        sync_wrapper = model
    else:
        raise ValueError(f"unknown impl: {impl}")

    stream = iter(loader)

    def next_batch():
        nonlocal stream
        try:
            return next(stream)
        except StopIteration:
            stream = iter(loader)
            return next(stream)

    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum_steps):
            x, y = next_batch()
            x = x.to(device)
            y = y.to(device)
            loss = F.cross_entropy(model(x), y)

            if micro < accum_steps - 1:
                context = (
                    sync_wrapper.no_sync()
                    if sync_wrapper is not None
                    else nullcontext()
                )
                with context:
                    loss.backward()
            else:
                loss.backward()

        if ddp is not None:
            ddp.step()
        else:
            optimizer.step()

    if rank == 0:
        if impl == "torchddp":
            state = model.module.state_dict()
        elif impl == "miniddp":
            state = ddp.model.state_dict()
        else:
            state = model.state_dict()
        torch.save({k: v.detach().cpu() for k, v in state.items()}, out_path)

    dist.barrier()
    cleanup()


def run_impl(
    impl: str,
    world_size: int,
    bucket_mb: float,
    steps: int,
    accum_steps: int,
    device: str,
    seed: int,
    out_path: Path,
) -> None:
    port = find_free_port()
    spawn(
        worker,
        world_size,
        (impl, bucket_mb, steps, accum_steps, device, seed, port, str(out_path)),
    )


def compare(
    world_size: int = 2,
    bucket_mb: float = 0.01,
    steps: int = 2,
    accum_steps: int = 1,
    device: str = "cpu",
    seed: int = 42,
    tol: float = 1e-5,
) -> float:
    out_dir = Path(tempfile.mkdtemp(prefix="miniddp_compare_"))
    miniddp_path = out_dir / "miniddp.pt"
    torchddp_path = out_dir / "torchddp.pt"

    run_impl(
        "miniddp", world_size, bucket_mb, steps, accum_steps, device, seed, miniddp_path
    )
    run_impl(
        "torchddp", world_size, bucket_mb, steps, accum_steps, device, seed, torchddp_path
    )

    a = torch.load(miniddp_path, map_location="cpu")
    b = torch.load(torchddp_path, map_location="cpu")

    if set(a) != set(b):
        raise AssertionError(f"state dict keys differ: {set(a) ^ set(b)}")

    max_diff = 0.0
    worst = ""
    for key in a:
        diff = (a[key].float() - b[key].float()).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            worst = key

    print(
        f"world_size={world_size} bucket_mb={bucket_mb} steps={steps} "
        f"accum_steps={accum_steps} device={device} "
        f"max_abs_diff={max_diff:.3e} worst={worst}"
    )
    if max_diff > tol:
        raise AssertionError(
            f"MiniDDP and torch DDP diverged: max_abs_diff={max_diff:.3e} "
            f"> tol={tol:.3e} at {worst}"
        )
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--bucket-mb", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    compare(
        world_size=args.world_size,
        bucket_mb=args.bucket_mb,
        steps=args.steps,
        accum_steps=args.accum_steps,
        device=args.device,
        seed=args.seed,
        tol=args.tol,
    )


if __name__ == "__main__":
    main()
