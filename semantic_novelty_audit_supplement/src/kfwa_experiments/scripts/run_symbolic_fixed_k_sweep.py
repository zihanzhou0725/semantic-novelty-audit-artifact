from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", maxsplit=1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", default="kfwa_experiments/runs_symbolic_fixed_k_lop_sweep_20label_60ep_10seeds")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--k-values", default="2,4,6,8,10,12")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--models",
        default="symbolic_kfwa_no_adapter,symbolic_kfwa_lop_features_no_kuramoto",
        help="Comma-separated symbolic models. LOP hyperparameter variants are only applied to LOP-feature models.",
    )
    parser.add_argument("--lop-oscillators-list", default="32,64")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=50_000)
    parser.add_argument("--val-limit", type=int, default=10_000)
    parser.add_argument("--test-limit", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--num-symbols", type=int, default=20)
    parser.add_argument("--rule-families", default="affine,block,parity")
    parser.add_argument(
        "--symbolic-split-mode",
        choices=["strict_function", "legacy_identity"],
        default="strict_function",
    )
    parser.add_argument("--query-out-of-support-prob", type=float, default=1.0)
    parser.add_argument("--support-noise", type=float, default=0.0)
    parser.add_argument(
        "--holdout-rule-frac",
        "--holdout-function-frac",
        dest="holdout_rule_frac",
        type=float,
        default=0.25,
        help="Fraction of unique complete functions held out from training.",
    )

    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--public-latent-count", type=int, default=32)
    parser.add_argument("--matched-model-dim", type=int, default=92)
    parser.add_argument("--matched-ff-dim", type=int, default=192)
    parser.add_argument("--oscillator-dim", type=int, default=8)
    parser.add_argument("--kuramoto-steps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selection-metric", default="val_unseen_function_accuracy")

    parser.add_argument("--rule-fourier-bands", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--lop-steps", type=int, default=16)
    parser.add_argument("--lop-dt", type=float, default=0.1)
    parser.add_argument("--lop-harmonics", type=int, default=4)
    parser.add_argument("--lop-clusters", type=int, default=4)
    parser.add_argument("--lop-coupling-dim", type=int, default=32)
    parser.add_argument("--no-progress", action="store_true")


def uses_population_size(model: str) -> bool:
    return "lop_features" in model or model in {
        "symbolic_transformer_latent_pool",
        "symbolic_transformer_lop",
        "symbolic_transformer_lop_rule_features",
    }


def run(cmd: list[str], dry_run: bool) -> None:
    print("\n==>", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_dir.rglob("metrics.json")):
        row = load_json(metrics_path)
        row.setdefault("run_dir", str(metrics_path.parent))
        row["metrics_path"] = str(metrics_path)
        row["variant"] = variant_name(row)
        rows.append(row)
    return rows


def variant_name(row: dict[str, Any]) -> str:
    model = str(row.get("model", ""))
    if uses_population_size(model):
        return f"{model}_m{row.get('lop_oscillators', '')}"
    return model


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    preferred = [
        "variant",
        "model",
        "attribution_id",
        "readout_type",
        "uses_explicit_rule_features",
        "rule_feature_ablation",
        "uses_kuramoto",
        "seed",
        "num_symbols",
        "support_min_k",
        "support_max_k",
        "num_functions",
        "seen_functions",
        "unseen_functions",
        "function_split_checksum",
        "function_split_overlap",
        "model_dim",
        "ff_dim",
        "population_size",
        "lop_oscillators",
        "lop_steps",
        "test_accuracy",
        "test_out_support_accuracy",
        "test_unseen_function_accuracy",
        "test_seen_function_accuracy",
        "test_unseen_accuracy",
        "test_seen_accuracy",
        "test_ece",
        "test_brier",
        "best_epoch",
        "parameters",
        "run_dir",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        k = int(row["support_min_k"])
        groups.setdefault((k, str(row["variant"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (k, variant), group in sorted(groups.items()):
        acc = [float(row["test_accuracy"]) for row in group]
        out = [float(row.get("test_out_support_accuracy", row["test_accuracy"])) for row in group]
        unseen = [float(row.get("test_unseen_accuracy", row["test_accuracy"])) for row in group]
        unseen_function = [
            float(row.get("test_unseen_function_accuracy", row.get("test_unseen_accuracy", row["test_accuracy"])))
            for row in group
        ]
        ece = [float(row["test_ece"]) for row in group]
        brier = [float(row["test_brier"]) for row in group]
        summary.append(
            {
                "support_k": k,
                "variant": variant,
                "model": group[0]["model"],
                "attribution_id": group[0].get("attribution_id", ""),
                "readout_type": group[0].get("readout_type", ""),
                "uses_explicit_rule_features": group[0].get("uses_explicit_rule_features", ""),
                "rule_feature_ablation": group[0].get("rule_feature_ablation", ""),
                "uses_kuramoto": group[0].get("uses_kuramoto", ""),
                "model_dim": group[0].get("model_dim", ""),
                "ff_dim": group[0].get("ff_dim", ""),
                "population_size": group[0].get("population_size", ""),
                "lop_oscillators": group[0].get("lop_oscillators", ""),
                "lop_steps": group[0].get("lop_steps", ""),
                "n": len(group),
                "test_accuracy_mean": mean(acc),
                "test_accuracy_sd": sd(acc),
                "test_out_support_accuracy_mean": mean(out),
                "test_out_support_accuracy_sd": sd(out),
                "test_unseen_accuracy_mean": mean(unseen),
                "test_unseen_accuracy_sd": sd(unseen),
                "test_unseen_function_accuracy_mean": mean(unseen_function),
                "test_unseen_function_accuracy_sd": sd(unseen_function),
                "test_ece_mean": mean(ece),
                "test_ece_sd": sd(ece),
                "test_brier_mean": mean(brier),
                "test_brier_sd": sd(brier),
                "parameters": group[0].get("parameters", ""),
            }
        )
    return summary


def best_by_k(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in summary:
        grouped.setdefault(int(row["support_k"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for k, group in sorted(grouped.items()):
        metric = "test_unseen_function_accuracy_mean"
        best = max(group, key=lambda row: float(row[metric]))
        baseline = next(
            (
                row
                for row in group
                if row.get("attribution_id") == "A0" or row["variant"] == "symbolic_transformer"
            ),
            None,
        )
        if baseline is None:
            baseline = next((row for row in group if row["variant"] == "symbolic_kfwa_no_adapter"), None)
        delta = ""
        if baseline is not None:
            delta = float(best[metric]) - float(baseline[metric])
        rows.append(
            {
                "support_k": k,
                "selection_metric": metric,
                "best_variant": best["variant"],
                "best_model": best["model"],
                "best_attribution_id": best.get("attribution_id", ""),
                "best_lop_oscillators": best.get("lop_oscillators", ""),
                "best_unseen_function_accuracy_mean": best[metric],
                "best_unseen_function_accuracy_sd": best["test_unseen_function_accuracy_sd"],
                "baseline_variant": baseline["variant"] if baseline else "",
                "baseline_unseen_function_accuracy_mean": baseline[metric] if baseline else "",
                "best_minus_baseline": delta,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fixed-K symbolic 20-label LOP/no-adapter sweep.")
    add_args(parser)
    args = parser.parse_args(argv)

    script = Path(__file__).with_name("run_experiment.py")
    out_dir = Path(args.out_dir)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    k_values = parse_int_list(args.k_values)
    seeds = parse_seeds(args.seeds)
    lop_oscillators = parse_int_list(args.lop_oscillators_list)

    run_count = 0
    for k in k_values:
        for model in models:
            oscillator_variants = lop_oscillators if uses_population_size(model) else [None]
            for lop_m in oscillator_variants:
                for seed in seeds:
                    cmd = [
                        sys.executable,
                        str(script),
                        "--task",
                        "symbolic_rule_mapping",
                        "--model",
                        model,
                        "--seed",
                        str(seed),
                        "--out-dir",
                        str(out_dir),
                        "--device",
                        args.device,
                        "--num-symbols",
                        str(args.num_symbols),
                        "--support-min-k",
                        str(k),
                        "--support-max-k",
                        str(k),
                        "--rule-families",
                        args.rule_families,
                        "--symbolic-split-mode",
                        args.symbolic_split_mode,
                        "--query-out-of-support-prob",
                        str(args.query_out_of_support_prob),
                        "--support-noise",
                        str(args.support_noise),
                        "--holdout-function-frac",
                        str(args.holdout_rule_frac),
                        "--selection-metric",
                        args.selection_metric,
                        "--train-limit",
                        str(args.train_limit),
                        "--val-limit",
                        str(args.val_limit),
                        "--test-limit",
                        str(args.test_limit),
                        "--batch-size",
                        str(args.batch_size),
                        "--num-workers",
                        str(args.num_workers),
                        "--epochs",
                        str(args.epochs),
                        "--patience",
                        str(args.patience),
                        "--lr",
                        str(args.lr),
                        "--weight-decay",
                        str(args.weight_decay),
                        "--dropout",
                        str(args.dropout),
                        "--model-dim",
                        str(args.model_dim),
                        "--layers",
                        str(args.layers),
                        "--heads",
                        str(args.heads),
                        "--ff-dim",
                        str(args.ff_dim),
                        "--public-latent-count",
                        str(args.public_latent_count),
                        "--matched-model-dim",
                        str(args.matched_model_dim),
                        "--matched-ff-dim",
                        str(args.matched_ff_dim),
                        "--oscillator-dim",
                        str(args.oscillator_dim),
                        "--kuramoto-steps",
                        str(args.kuramoto_steps),
                        "--dt",
                        str(args.dt),
                        "--adapter-rank",
                        str(args.adapter_rank),
                        "--top-k",
                        str(args.top_k),
                        "--rule-fourier-bands",
                        str(args.rule_fourier_bands),
                        "--lop-steps",
                        str(args.lop_steps),
                        "--lop-dt",
                        str(args.lop_dt),
                        "--lop-harmonics",
                        str(args.lop_harmonics),
                        "--lop-clusters",
                        str(args.lop_clusters),
                        "--lop-coupling-dim",
                        str(args.lop_coupling_dim),
                    ]
                    if lop_m is not None:
                        cmd.extend(["--lop-oscillators", str(lop_m)])
                    if args.no_progress:
                        cmd.append("--no-progress")
                    run_count += 1
                    run(cmd, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\nDry run complete: {run_count} commands would be executed.")
        return 0

    rows = collect_rows(out_dir)
    summary = aggregate(rows)
    best = best_by_k(summary)
    write_csv(out_dir / "summary.csv", rows)
    write_csv(out_dir / "group_summary.csv", summary)
    write_csv(out_dir / "best_by_k.csv", best)
    print(f"\nWrote {len(rows)} rows to {out_dir / 'summary.csv'}")
    print(f"Wrote {len(summary)} rows to {out_dir / 'group_summary.csv'}")
    print(f"Wrote {len(best)} rows to {out_dir / 'best_by_k.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
