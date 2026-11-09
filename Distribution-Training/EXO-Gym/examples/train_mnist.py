"""Train a small MLP on MNIST with MiniDDP, torch DDP, or a single process.

Examples
--------
CPU, two virtual ranks, compare MiniDDP with torch DDP:
    python examples/train_mnist.py --impl miniddp --world-size 2 --device cpu
    python examples/train_mnist.py --impl torchddp --world-size 2 --device cpu

Single RTX 4060, two virtual ranks sharing GPU 0 (Gloo backend):
    python examples/train_mnist.py --impl miniddp --world-size 2 --device cuda

Gradient accumulation with no_sync:
    python examples/train_mnist.py --impl miniddp --world-size 2 --device cpu \
        --accum-steps 4 --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision import datasets, transforms

# Make `miniddp` importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miniddp.comm import cleanup, find_free_port, init_process_group
from miniddp.ddp import MiniDDP
from miniddp.launch import spawn


class MLP(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BatchStream:
    """Infinite stream over a DataLoader, advancing the sampler epoch."""

    def __init__(self, loader: DataLoader, sampler):
        self.loader = loader
        self.sampler = sampler
        self.epoch = 0
        self.iterator = iter(loader)

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            return next(self.iterator)


def prepare_mnist(data_dir: str, retries: int = 3) -> None:
    """Download/extract MNIST once in the parent, before spawning ranks.

    torchvision's download is not safe when several ranks start at the same
    time: all of them may write the same archive. Doing it here also gives us
    a place to retry transient network resets.
    """
    for attempt in range(1, retries + 1):
        try:
            datasets.MNIST(data_dir, train=True, download=True)
            datasets.MNIST(data_dir, train=False, download=True)
            return
        except Exception as exc:
            print(
                f"[data] MNIST preparation attempt {attempt}/{retries} failed: {exc}",
                flush=True,
            )
            if attempt == retries:
                raise RuntimeError(
                    "Could not prepare MNIST. Check your network, or manually put "
                    "the four *.gz files under <data-dir>/MNIST/raw/ and rerun. "
                    "Expected files: train-images-idx3-ubyte.gz, "
                    "train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz, "
                    "t10k-labels-idx1-ubyte.gz."
                ) from exc
            time.sleep(2 * attempt)


def build_loaders(args, rank: int, world_size: int, device: torch.device):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # The parent process already prepared the files; see prepare_mnist().
    # Using download=False avoids concurrent writes from multiple ranks.
    train_ds = datasets.MNIST(
        args.data_dir, train=True, download=False, transform=transform
    )
    test_ds = datasets.MNIST(
        args.data_dir, train=False, download=False, transform=transform
    )

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = RandomSampler(
            train_ds, generator=torch.Generator().manual_seed(args.seed)
        )
        test_sampler = torch.utils.data.SequentialSampler(test_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return train_loader, test_loader, train_sampler


def evaluate(model, loader, device: torch.device, world_size: int):
    model.eval()
    correct = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)
    loss_sum = torch.zeros(1, device=device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum")
            correct += (logits.argmax(dim=1) == y).sum()
            total += y.numel()

    if world_size > 1:
        for tensor in (correct, total, loss_sum):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return (loss_sum / total).item(), (correct / total).item()


def unwrap_state_dict(model, impl: str):
    if impl == "torchddp" and hasattr(model, "module"):
        return model.module.state_dict()
    if impl == "miniddp" and hasattr(model, "model"):
        return model.model.state_dict()
    return model.state_dict()


def worker(rank: int, world_size: int, args_dict: dict, port: int) -> None:
    args = argparse.Namespace(**args_dict)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    distributed = not (args.impl == "single" and world_size == 1)
    if distributed:
        init_process_group(rank, world_size, port=port, backend=args.backend)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = MLP(args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ddp: Optional[MiniDDP] = None
    sync_wrapper = None

    if args.impl == "miniddp":
        ddp = MiniDDP(model, optimizer, bucket_cap_mb=args.bucket_mb)
        model = ddp
        sync_wrapper = ddp
    elif args.impl == "torchddp":
        if world_size > 1:
            ddp_kwargs = {"bucket_cap_mb": args.bucket_mb}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [device]
            model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
            sync_wrapper = model
    elif args.impl == "single":
        if world_size != 1:
            raise ValueError("--impl single requires --world-size 1")
    else:
        raise ValueError(f"unknown impl: {args.impl}")

    train_loader, test_loader, train_sampler = build_loaders(
        args, rank, world_size, device
    )
    stream = BatchStream(train_loader, train_sampler)

    steps_per_epoch = len(train_loader)
    total_steps = (
        args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    )

    if rank == 0:
        if ddp is not None:
            print(f"MiniDDP buckets: {ddp.reducer.bucket_summary()}")
        print(
            f"impl={args.impl} world_size={world_size} device={device} "
            f"steps={total_steps} batch/rank={args.batch_size} "
            f"accum={args.accum_steps} backend={args.backend}"
        )
        print(
            f"global batch per optimizer step = "
            f"{args.batch_size * args.accum_steps * world_size}"
        )

    model.train()
    start = time.perf_counter()
    last_loss = float("nan")

    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        for micro in range(args.accum_steps):
            x, y = stream.next()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            loss = F.cross_entropy(model(x), y)
            last_loss = loss.item()

            if micro < args.accum_steps - 1:
                # Intermediate microbatches: do not all-reduce.
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

        if step % args.log_interval == 0 or step == 1:
            elapsed = time.perf_counter() - start
            samples = step * args.batch_size * args.accum_steps * world_size
            msg = (
                f"step {step:4d}/{total_steps} loss={last_loss:.4f} "
                f"elapsed={elapsed:.1f}s samples/s={samples / elapsed:.0f}"
            )
            if ddp is not None:
                stats = ddp.stats.as_dict()
                msg += (
                    f" allreduce_calls={int(stats['allreduce_calls'])} "
                    f"payload={stats['payload_mb']:.3f}MB "
                    f"wait={stats['wait_time_ms']:.1f}ms"
                )
                ddp.reset_stats()
            if rank == 0:
                print(msg, flush=True)

    train_time = time.perf_counter() - start
    test_loss, test_acc = evaluate(model, test_loader, device, world_size)

    if rank == 0:
        print(
            f"final test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"train_time={train_time:.2f}s"
        )

        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            state = unwrap_state_dict(model, args.impl)
            cpu_state = {k: v.detach().cpu() for k, v in state.items()}
            torch.save(cpu_state, output)
            metrics = {
                "impl": args.impl,
                "world_size": world_size,
                "device": str(device),
                "backend": args.backend,
                "bucket_mb": args.bucket_mb,
                "batch_size_per_rank": args.batch_size,
                "accum_steps": args.accum_steps,
                "global_batch": args.batch_size * args.accum_steps * world_size,
                "steps": total_steps,
                "train_time_s": train_time,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "buckets": ddp.reducer.bucket_summary() if ddp is not None else None,
            }
            metrics_path = output.with_suffix(".json")
            metrics_path.write_text(json.dumps(metrics, indent=2))
            print(f"saved state to {output} and metrics to {metrics_path}")

    if distributed:
        dist.barrier()
        cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniDDP MNIST example")
    parser.add_argument("--impl", choices=["miniddp", "torchddp", "single"], default="miniddp")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--bucket-mb", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if args.impl == "single" and args.world_size != 1:
        raise SystemExit("--impl single requires --world-size 1")
    if args.backend == "nccl" and args.device == "cpu":
        raise SystemExit("NCCL cannot be used with --device cpu")
    if args.backend == "nccl" and args.world_size > 1:
        if torch.cuda.device_count() < args.world_size:
            raise SystemExit(
                "NCCL needs one process per GPU. With one 4060, use --backend gloo "
                "for world_size > 1."
            )

    # Do this exactly once, before mp.spawn(), so the two ranks never race on
    # writing the same MNIST archive.
    prepare_mnist(args.data_dir)

    port = find_free_port()
    spawn(worker, args.world_size, (vars(args), port))


if __name__ == "__main__":
    main()
