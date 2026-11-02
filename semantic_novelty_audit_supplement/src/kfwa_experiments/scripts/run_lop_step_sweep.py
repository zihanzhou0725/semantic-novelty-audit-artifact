from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import get_task_loaders
from kfwa.models import build_model
from kfwa.training import run_epoch, train_model
from kfwa.utils import as_serializable, count_parameters, get_device, save_json, set_seed

from run_experiment import add_args, apply_quick, config_dict, write_history_csv


def parse_steps(raw: str) -> list[int]:
    """Parse a comma-separated positive integer step list."""

    steps = [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not steps:
        raise argparse.ArgumentTypeError("step list must not be empty.")
    if any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("all steps must be positive.")
    return steps


def metric_row(prefix: str, metrics: Any) -> dict[str, float]:
    """Return metric dict with a stable prefix."""

    return metrics.as_dict(prefix=prefix)


def set_lop_steps(model: torch.nn.Module, steps: int) -> None:
    """Override LOP-RI test-time oscillator steps on a loaded model."""

    if not hasattr(model, "oscillator_steps"):
        raise TypeError("run_lop_step_sweep expects a LOP-RI model with oscillator_steps.")
    setattr(model, "oscillator_steps", int(steps))


def build_bundle(args: argparse.Namespace):
    """Build a fresh deterministic data bundle for one training run."""

    return get_task_loaders(
        task=args.task,
        source=args.source,
        data_root=args.data_root,
        download=args.download,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        num_contexts=args.num_contexts,
        holdout_contexts=args.holdout_contexts,
        context_encoding=args.context_encoding,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        quick=args.quick,
        num_symbols=args.num_symbols,
        support_min_k=args.support_min_k,
        support_max_k=args.support_max_k,
        rule_families=args.rule_families,
        holdout_rule_frac=args.holdout_rule_frac,
        query_out_of_support_prob=args.query_out_of_support_prob,
        support_noise=args.support_noise,
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV using the union of observed keys."""

    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def add_sweep_args(parser: argparse.ArgumentParser) -> None:
    """Add train/test step-sweep arguments."""

    parser.add_argument("--train-steps", type=parse_steps, default=parse_steps("8,16,32,64,128"))
    parser.add_argument("--test-steps", type=parse_steps, default=parse_steps("8,16,32,64,128"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train LOP-RI across train steps and evaluate test-time step sweeps.")
    add_args(parser)
    add_sweep_args(parser)
    args = parser.parse_args(argv)
    apply_quick(args)

    device = get_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.out_dir) / f"{args.task}__{args.model}__seed{args.seed}__lop_step_sweep__{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=False)
    save_json(
        sweep_dir / "sweep_config.json",
        {
            **config_dict(args),
            "train_steps": args.train_steps,
            "test_steps": args.test_steps,
            "sweep_dir": str(sweep_dir),
            "device": str(device),
        },
    )

    rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []

    for train_steps in args.train_steps:
        run_args = deepcopy(args)
        run_args.lop_steps = int(train_steps)
        set_seed(run_args.seed)
        bundle = build_bundle(run_args)
        model = build_model(run_args, context_dim=bundle.context_dim, target_dim=bundle.target_dim)
        set_lop_steps(model, train_steps)
        parameters = count_parameters(model)

        train_dir = sweep_dir / f"train_steps_{train_steps}"
        train_dir.mkdir()
        save_json(
            train_dir / "config.json",
            {**config_dict(run_args), "dataset": as_serializable(bundle.metadata), "parameters": parameters},
        )

        print(
            f"=== train_steps={train_steps} | model={args.model} | seed={args.seed} | "
            f"device={device} | params={parameters} ===",
            flush=True,
        )
        result = train_model(
            model=model,
            train_loader=bundle.train,
            val_loader=bundle.val,
            test_loader=bundle.test,
            device=device,
            epochs=run_args.epochs,
            lr=run_args.lr,
            weight_decay=run_args.weight_decay,
            patience=run_args.patience,
            checkpoint_path=train_dir / "best_model.pt",
            show_progress=not run_args.no_progress,
            selection_metric=run_args.selection_metric,
        )
        write_history_csv(train_dir / "history.csv", result.history)
        train_metrics = {
            "task": run_args.task,
            "model": run_args.model,
            "seed": run_args.seed,
            "train_steps": train_steps,
            "run_dir": str(train_dir),
            "device": str(device),
            "parameters": parameters,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "best_selection_metric": result.best_selection_metric,
            "best_selection_score": result.best_selection_score,
            **result.test_metrics.as_dict(prefix="same_step_test_"),
        }
        save_json(train_dir / "metrics.json", train_metrics)
        train_rows.append(train_metrics)

        state = torch.load(train_dir / "best_model.pt", map_location=device)
        model.load_state_dict(state)
        model.to(device)
        for test_steps in args.test_steps:
            set_lop_steps(model, test_steps)
            print(f"--- eval train_steps={train_steps} test_steps={test_steps} ---", flush=True)
            metrics = run_epoch(
                model=model,
                loader=bundle.test,
                device=device,
                optimizer=None,
                phase=f"test_s{test_steps}",
                show_progress=not run_args.no_progress,
            )
            row = {
                "task": run_args.task,
                "model": run_args.model,
                "seed": run_args.seed,
                "train_steps": train_steps,
                "test_steps": test_steps,
                "run_dir": str(train_dir),
                "device": str(device),
                "parameters": parameters,
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "best_selection_metric": result.best_selection_metric,
                "best_selection_score": result.best_selection_score,
                **metric_row("test_", metrics),
            }
            rows.append(row)
            print(
                f"train_steps={train_steps} test_steps={test_steps} "
                f"acc={row['test_accuracy']:.4f} "
                f"unseen={row.get('test_unseen_accuracy', float('nan')):.4f} "
                f"out_support={row.get('test_out_support_accuracy', float('nan')):.4f} "
                f"ece={row['test_ece']:.4f}",
                flush=True,
            )

        write_rows(sweep_dir / "step_sweep.csv", rows)
        write_rows(sweep_dir / "train_summary.csv", train_rows)

    write_rows(sweep_dir / "step_sweep.csv", rows)
    write_rows(sweep_dir / "train_summary.csv", train_rows)
    print(f"Wrote {len(rows)} rows to {sweep_dir / 'step_sweep.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
