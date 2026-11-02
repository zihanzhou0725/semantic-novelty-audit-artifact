from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import (
    CONTEXT_ENCODINGS,
    IMAGE_SOURCES,
    SYMBOLIC_SPLIT_MODES,
    TASKS,
    get_task_loaders,
)
from kfwa.models import MODEL_CHOICES, build_model
from kfwa.training import train_model
from kfwa.utils import as_serializable, count_parameters, get_device, make_run_dir, save_json, set_seed


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments."""

    parser.add_argument("--task", choices=TASKS, default="context_permutation")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="kfwa_experiments/runs")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--source", choices=IMAGE_SOURCES, default="mnist")
    parser.add_argument("--data-root", default="kfwa_experiments/data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-contexts", type=int, default=8)
    parser.add_argument("--holdout-contexts", type=int, default=2)
    parser.add_argument("--context-encoding", choices=CONTEXT_ENCODINGS, default="perm_matrix")
    parser.add_argument("--train-limit", type=int, default=20_000)
    parser.add_argument("--val-limit", type=int, default=5_000)
    parser.add_argument("--test-limit", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--selection-metric",
        choices=[
            "val_loss",
            "val_ce",
            "val_accuracy",
            "val_seen_accuracy",
            "val_unseen_accuracy",
            "val_seen_function_accuracy",
            "val_unseen_function_accuracy",
            "val_in_support_accuracy",
            "val_out_support_accuracy",
            "val_ece",
            "val_brier",
        ],
        default="val_unseen_accuracy",
        help="Validation metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=3)
    parser.add_argument("--cnn-channels", type=int, default=64)

    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--public-latent-count", type=int, default=32)
    parser.add_argument("--matched-model-dim", type=int, default=92)
    parser.add_argument("--matched-ff-dim", type=int, default=192)
    parser.add_argument("--patch-size", type=int, default=4)

    parser.add_argument("--oscillator-dim", type=int, default=16)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--kuramoto-steps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--rule-fourier-bands", type=int, default=4)
    parser.add_argument("--support-aux-weight", type=float, default=0.1)
    parser.add_argument("--adapter-gate-bias", type=float, default=-4.0)
    parser.add_argument("--lop-oscillators", type=int, default=64)
    parser.add_argument("--lop-steps", type=int, default=8)
    parser.add_argument("--lop-dt", type=float, default=0.1)
    parser.add_argument("--lop-harmonics", type=int, default=4)
    parser.add_argument("--lop-clusters", type=int, default=4)
    parser.add_argument("--lop-coupling-dim", type=int, default=32)
    parser.add_argument("--lop-ssr-support-aux-weight", type=float, default=0.1)
    parser.add_argument("--lop-ssr-partial-sync-weight", type=float, default=0.01)
    parser.add_argument("--lop-ssr-usage-entropy-weight", type=float, default=0.01)

    parser.add_argument("--num-symbols", type=int, default=10)
    parser.add_argument("--support-min-k", type=int, default=2)
    parser.add_argument("--support-max-k", type=int, default=6)
    parser.add_argument("--rule-families", default="affine,block,parity")
    parser.add_argument(
        "--symbolic-split-mode",
        choices=SYMBOLIC_SPLIT_MODES,
        default="strict_function",
    )
    parser.add_argument(
        "--holdout-rule-frac",
        "--holdout-function-frac",
        dest="holdout_rule_frac",
        type=float,
        default=0.25,
        help="Fraction of unique complete functions held out from training.",
    )
    parser.add_argument("--query-out-of-support-prob", type=float, default=0.5)
    parser.add_argument("--support-noise", type=float, default=0.0)


def apply_quick(args: argparse.Namespace) -> None:
    """Shrink settings for interface smoke tests."""

    if not args.quick:
        return
    args.source = "synthetic"
    args.train_limit = min(args.train_limit, 256)
    args.val_limit = min(args.val_limit, 64)
    args.test_limit = min(args.test_limit, 128)
    args.num_contexts = min(args.num_contexts, 4)
    args.holdout_contexts = min(args.holdout_contexts, 1)
    args.batch_size = min(args.batch_size, 64)
    args.epochs = min(args.epochs, 3)
    args.patience = min(args.patience, 2)
    args.hidden_dim = min(args.hidden_dim, 128)
    args.cnn_channels = min(args.cnn_channels, 32)
    args.model_dim = min(args.model_dim, 32)
    args.ff_dim = min(args.ff_dim, 64)
    args.matched_model_dim = min(args.matched_model_dim, 48)
    args.matched_ff_dim = min(args.matched_ff_dim, 96)
    args.layers = min(args.layers, 1)
    args.heads = min(args.heads, 4)
    args.oscillator_dim = min(args.oscillator_dim, 8)
    args.adapter_rank = min(args.adapter_rank, 4)
    args.top_k = min(args.top_k, 8)
    args.kuramoto_steps = min(args.kuramoto_steps, 1)
    args.rule_fourier_bands = min(args.rule_fourier_bands, 2)
    args.lop_oscillators = min(args.lop_oscillators, 16)
    args.lop_steps = min(args.lop_steps, 2)
    args.lop_harmonics = min(args.lop_harmonics, 2)
    args.lop_clusters = min(args.lop_clusters, 2)
    args.lop_coupling_dim = min(args.lop_coupling_dim, 16)
    args.support_max_k = min(args.support_max_k, 4)
    args.support_min_k = min(args.support_min_k, args.support_max_k)


def write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    """Write epoch history to CSV."""

    if not history:
        return
    keys: list[str] = []
    for row in history:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def config_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Return JSON-safe config."""

    return {key: as_serializable(value) for key, value in vars(args).items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one KFWA benchmark experiment.")
    add_args(parser)
    args = parser.parse_args(argv)
    apply_quick(args)

    set_seed(args.seed)
    device = get_device(args.device)
    bundle = get_task_loaders(
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
        symbolic_split_mode=args.symbolic_split_mode,
        holdout_rule_frac=args.holdout_rule_frac,
        query_out_of_support_prob=args.query_out_of_support_prob,
        support_noise=args.support_noise,
    )
    model = build_model(args, context_dim=bundle.context_dim, target_dim=bundle.target_dim)
    parameters = count_parameters(model)
    model_metadata = {
        "attribution_id": getattr(model, "attribution_id", None),
        "readout_type": getattr(model, "readout_type", None),
        "uses_explicit_rule_features": getattr(model, "uses_explicit_rule_features", None),
        "rule_feature_ablation": getattr(model, "rule_feature_ablation", None),
        "uses_kuramoto": getattr(model, "uses_kuramoto", None),
        "predictor_type": getattr(model, "predictor_type", None),
        "effective_model_dim": getattr(model, "effective_model_dim", args.model_dim),
        "effective_ff_dim": getattr(model, "effective_ff_dim", args.ff_dim),
        "population_size": getattr(model, "population_size", None),
        "public_architecture": getattr(model, "public_architecture", None),
        "support_order_invariant": getattr(
            model,
            "support_order_invariant",
            None,
        ),
    }
    run_dir = make_run_dir(args.out_dir, args.task, args.model, args.seed)
    save_json(
        run_dir / "config.json",
        {
            **config_dict(args),
            "dataset": as_serializable(bundle.metadata),
            "model_metadata": as_serializable(model_metadata),
        },
    )

    result = train_model(
        model=model,
        train_loader=bundle.train,
        val_loader=bundle.val,
        test_loader=bundle.test,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_path=run_dir / "best_model.pt",
        show_progress=not args.no_progress,
        selection_metric=args.selection_metric,
    )
    write_history_csv(run_dir / "history.csv", result.history)
    metrics = {
        "task": args.task,
        "model": args.model,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "device": str(device),
        "parameters": parameters,
        "num_symbols": args.num_symbols,
        "support_min_k": args.support_min_k,
        "support_max_k": args.support_max_k,
        "rule_families": args.rule_families,
        "query_out_of_support_prob": args.query_out_of_support_prob,
        "support_noise": args.support_noise,
        "holdout_function_frac": args.holdout_rule_frac,
        "symbolic_split_mode": bundle.metadata.get("symbolic_split_mode"),
        "split_unit": bundle.metadata.get("split_unit"),
        "function_sampling": bundle.metadata.get("function_sampling"),
        "num_functions": bundle.metadata.get("num_functions"),
        "seen_functions": bundle.metadata.get("seen_functions"),
        "unseen_functions": bundle.metadata.get("unseen_functions"),
        "function_split_checksum": bundle.metadata.get("function_split_checksum"),
        "function_split_overlap": bundle.metadata.get("function_split_overlap"),
        "identity_split_checksum": bundle.metadata.get("identity_split_checksum"),
        "leaked_unseen_identities": bundle.metadata.get(
            "leaked_unseen_identities"
        ),
        "leaked_unseen_identity_fraction": bundle.metadata.get(
            "leaked_unseen_identity_fraction"
        ),
        **model_metadata,
        "model_dim": model_metadata["effective_model_dim"],
        "layers": args.layers,
        "ff_dim": model_metadata["effective_ff_dim"],
        "oscillator_dim": args.oscillator_dim,
        "kuramoto_steps": args.kuramoto_steps,
        "lop_oscillators": model_metadata["population_size"]
        if model_metadata["population_size"] is not None
        else args.lop_oscillators,
        "lop_steps": 0 if model_metadata["uses_kuramoto"] is False else args.lop_steps,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "best_selection_metric": result.best_selection_metric,
        "best_selection_score": result.best_selection_score,
        **result.test_metrics.as_dict(prefix="test_"),
    }
    save_json(run_dir / "metrics.json", metrics)
    print(
        f"{args.task} | {args.model} | seed={args.seed} | "
        f"device={device} | "
        f"best_epoch={result.best_epoch} | "
        f"test_acc={metrics['test_accuracy']:.4f} | "
        f"seen={metrics.get('test_seen_accuracy', float('nan')):.4f} | "
        f"unseen={metrics.get('test_unseen_accuracy', float('nan')):.4f} | "
        f"unseen_function={metrics.get('test_unseen_function_accuracy', float('nan')):.4f} | "
        f"out_support={metrics.get('test_out_support_accuracy', float('nan')):.4f} | "
        f"ece={metrics['test_ece']:.4f} | params={parameters}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
