"""Parameterized EXO Gym MNIST experiment for this single-GPU laptop.

Why this file exists
--------------------
The upstream example/example/mnist.py hardcodes num_nodes=4, 150 steps and a
large validation set. This script exposes those values as CLI flags so you can
run a 10-step smoke test, then sweep N=1/2/4 without editing upstream code.

Important local details
-----------------------
- On one RTX 4060, EXO Gym automatically uses Gloo when num_nodes > #GPUs.
  All virtual ranks share cuda:0. This is functional simulation, not multi-GPU
  speedup.
- `device` is passed to Trainer(...), not to fit(...). In the current EXO Gym
  source, fit() has no device parameter; a `device=` keyword there would be
  swallowed by **kwargs.
- CSV logs go to logs/<run_name>/train.csv and validation.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from exogym.strategy.diloco import DiLoCoStrategy
    from exogym.strategy.optim import OptimSpec
    from exogym.strategy.sparta import SPARTAStrategy
    from exogym.strategy.strategy import SimpleReduceStrategy
    from exogym.trainer import Trainer
except ImportError as exc:
    raise SystemExit(
        "EXO Gym is not installed in this Python environment.\n"
        "Follow README step 3, then rerun from the project root."
    ) from exc


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Lighter than the upstream CNN; safer for 4 ranks sharing 8GB VRAM."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class OfficialCNN(nn.Module):
    """The CNN from EXO Gym's upstream MNIST example."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ModelWrapper(nn.Module):
    """EXO Gym expects model(batch) -> scalar loss."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch):
        images, labels = batch
        return F.cross_entropy(self.backbone(images), labels)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_datasets(data_dir: str):
    train_tfm = transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    val_tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=train_tfm)
    val_ds = datasets.MNIST(data_dir, train=False, download=True, transform=val_tfm)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_runs(run_names: list[tuple[str, str]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    plotted = 0
    for label, run_name in run_names:
        csv_path = Path("logs") / run_name / "validation.csv"
        if not csv_path.exists():
            print(f"[plot] missing {csv_path}, skipping {label}")
            continue

        steps, losses = [], []
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                value = row.get("global_loss") or row.get("local_loss")
                if value:
                    steps.append(int(row["step"]))
                    losses.append(float(value))
        if steps:
            plt.plot(steps, losses, label=label, linewidth=2)
            plotted += 1

    if plotted == 0:
        print("[plot] no validation points found")
        return

    plt.xlabel("Step")
    plt.ylabel("Validation loss")
    plt.title("EXO Gym MNIST: global validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[plot] saved {output_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
STRATEGY_MAP = {
    "ddp": SimpleReduceStrategy,
    "diloco": DiLoCoStrategy,
    "sparta": SPARTAStrategy,
}


def largest_divisor(value: int, upper: int) -> int:
    candidate = min(value, upper)
    while candidate > 1 and value % candidate != 0:
        candidate -= 1
    return max(1, candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXO Gym MNIST on a single GPU")
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256, help="per virtual node")
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--val-interval", type=int, default=25)
    parser.add_argument("--strategies", default="ddp,diloco,sparta")
    parser.add_argument("--model", choices=["small", "official"], default="small")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--run-tag", default="local")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    strategy_names = [name.strip() for name in args.strategies.split(",") if name.strip()]
    unknown = [name for name in strategy_names if name not in STRATEGY_MAP]
    if unknown:
        raise SystemExit(f"unknown strategies: {unknown}; choose from {sorted(STRATEGY_MAP)}")

    minibatch_size = largest_divisor(args.batch_size, args.minibatch_size)
    val_size = max(args.batch_size, args.val_size)
    val_size = (val_size // args.batch_size) * args.batch_size
    if val_size == 0:
        val_size = args.batch_size

    device = None if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    train_ds, val_ds = build_datasets(args.data_dir)
    optim_spec = OptimSpec(torch.optim.AdamW, lr=3e-4, weight_decay=1e-4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_names: list[tuple[str, str]] = []

    print(
        f"EXO Gym: nodes={args.num_nodes} steps={args.max_steps} "
        f"batch/rank={args.batch_size} minibatch={minibatch_size} "
        f"global_batch={args.batch_size * args.num_nodes} "
        f"val_size={val_size} device={args.device} model={args.model}"
    )

    for name in strategy_names:
        strategy_cls = STRATEGY_MAP[name]
        backbone = SmallCNN() if args.model == "small" else OfficialCNN()
        model = ModelWrapper(backbone)

        # Trainer owns the device choice. fit() does not take a device argument
        # in the current EXO Gym main branch.
        trainer = Trainer(model, train_ds, val_ds, device=device)

        strategy = strategy_cls(
            optim_spec=optim_spec,
            H=20,
            p=0.05,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs={"warmup_steps": 100, "cosine_anneal": True},
        )

        run_name = (
            f"exogym_{args.run_tag}_n{args.num_nodes}_{name}_s{args.seed}_{timestamp}"
        )
        run_names.append((f"{name} (N={args.num_nodes})", run_name))

        print(f"\n=== {name.upper()} N={args.num_nodes} run={run_name} ===")
        trainer.fit(
            num_epochs=5,
            max_steps=args.max_steps,
            strategy=strategy,
            num_nodes=args.num_nodes,
            batch_size=args.batch_size,
            minibatch_size=minibatch_size,
            val_size=val_size,
            val_interval=args.val_interval,
            seed=args.seed,
            run_name=run_name,
            wandb_project="exogym-mnist" if args.wandb else None,
        )

        # Print the last validation point so the terminal has a quick summary.
        csv_path = Path("logs") / run_name / "validation.csv"
        if csv_path.exists():
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                last = rows[-1]
                print(
                    f"[{name}] last step={last.get('step')} "
                    f"global_loss={last.get('global_loss') or last.get('local_loss')}"
                )

    summary = {
        "num_nodes": args.num_nodes,
        "max_steps": args.max_steps,
        "batch_size_per_rank": args.batch_size,
        "minibatch_size": minibatch_size,
        "global_batch": args.batch_size * args.num_nodes,
        "val_size": val_size,
        "val_interval": args.val_interval,
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "strategies": strategy_names,
        "runs": [{"label": label, "run_name": run_name} for label, run_name in run_names],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"exogym_{args.run_tag}_n{args.num_nodes}_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved run summary to {summary_path}")

    if args.plot:
        plot_runs(
            run_names,
            output_dir / f"exogym_{args.run_tag}_n{args.num_nodes}_{timestamp}.png",
        )


if __name__ == "__main__":
    main()
