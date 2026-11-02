from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", default="kfwa_experiments/runs_symbolic_rule_mapping_20label_k4_12_20seeds")
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=50_000)
    parser.add_argument("--val-limit", type=int, default=10_000)
    parser.add_argument("--test-limit", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-symbols", type=int, default=20)
    parser.add_argument("--support-min-k", type=int, default=4)
    parser.add_argument("--support-max-k", type=int, default=12)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--oscillator-dim", type=int, default=8)
    parser.add_argument("--kuramoto-steps", type=int, default=2)
    parser.add_argument("--lop-steps", type=int, default=16)
    parser.add_argument("--no-progress", action="store_true")


def run(cmd: list[str]) -> None:
    print("\n==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 20-label symbolic rule mapping suite.")
    add_args(parser)
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    run_task_seeds = script_dir / "run_task_seeds.py"
    aggregate = script_dir / "aggregate_results.py"

    common = [
        "--task",
        "symbolic_rule_mapping",
        "--seeds",
        args.seeds,
        "--out-dir",
        args.out_dir,
        "--device",
        args.device,
        "--num-symbols",
        str(args.num_symbols),
        "--support-min-k",
        str(args.support_min_k),
        "--support-max-k",
        str(args.support_max_k),
        "--rule-families",
        "affine,block,parity",
        "--query-out-of-support-prob",
        "1.0",
        "--selection-metric",
        "val_out_support_accuracy",
        "--train-limit",
        str(args.train_limit),
        "--val-limit",
        str(args.val_limit),
        "--test-limit",
        str(args.test_limit),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--model-dim",
        str(args.model_dim),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--ff-dim",
        str(args.ff_dim),
        "--oscillator-dim",
        str(args.oscillator_dim),
        "--kuramoto-steps",
        str(args.kuramoto_steps),
        "--lop-steps",
        str(args.lop_steps),
    ]
    if args.no_progress:
        common.append("--no-progress")

    run([sys.executable, str(run_task_seeds), "--models", "symbolic_transformer", *common])
    run([sys.executable, str(run_task_seeds), "--models", "symbolic_kfwa_no_adapter", *common])
    run(
        [
            sys.executable,
            str(run_task_seeds),
            "--models",
            "symbolic_kfwa_lop_features_no_kuramoto",
            *common,
            "--lop-oscillators",
            "64",
        ]
    )
    run(
        [
            sys.executable,
            str(run_task_seeds),
            "--models",
            "symbolic_kfwa_lop_features_no_kuramoto",
            *common,
            "--lop-oscillators",
            "32",
        ]
    )
    run(
        [
            sys.executable,
            str(aggregate),
            "--runs-dir",
            args.out_dir,
            "--out",
            str(Path(args.out_dir) / "summary.csv"),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
