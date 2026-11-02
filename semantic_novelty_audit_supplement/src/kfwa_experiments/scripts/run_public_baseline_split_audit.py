from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_symbolic_a7_feature_ablations import student_t_quantile


PUBLIC_MODELS = (
    "symbolic_deepsets",
    "symbolic_relation_network",
    "symbolic_set_transformer",
    "symbolic_hopfield",
    "symbolic_perceiver",
    "symbolic_transformer",
)
PROTOCOLS = ("legacy_identity", "strict_function")


def parse_values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for value in parse_values(raw):
        if "-" in value:
            start, end = value.split("-", maxsplit=1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(value))
    return seeds


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_matches(
    config: dict[str, Any],
    *,
    model: str,
    seed: int,
    protocol: str,
    args: argparse.Namespace,
) -> bool:
    expected = {
        "task": "symbolic_rule_mapping",
        "model": model,
        "seed": seed,
        "symbolic_split_mode": protocol,
        "num_symbols": 20,
        "support_min_k": 8,
        "support_max_k": 8,
        "rule_families": "affine,block,parity",
        "query_out_of_support_prob": 1.0,
        "support_noise": 0.1,
        "holdout_rule_frac": 0.25,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "model_dim": 64,
        "layers": 2,
        "heads": 4,
        "ff_dim": 128,
        "hidden_dim": 256,
        "public_latent_count": 32,
        "dropout": 0.05,
        "lr": 0.001,
        "weight_decay": 0.0001,
    }
    return all(config.get(key) == value for key, value in expected.items())


def completed_run(
    root: Path,
    *,
    model: str,
    seed: int,
    protocol: str,
    args: argparse.Namespace,
) -> Path | None:
    matches: list[Path] = []
    for config_path in sorted(root.glob("*/config.json")):
        config = read_json(config_path)
        if not run_matches(
            config,
            model=model,
            seed=seed,
            protocol=protocol,
            args=args,
        ):
            continue
        if (
            (config_path.parent / "metrics.json").exists()
            and (config_path.parent / "best_model.pt").exists()
        ):
            matches.append(config_path.parent)
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple completed runs for {protocol}/{model}/seed{seed}: "
            f"{matches}"
        )
    return matches[0] if matches else None


def metric_mean_sd(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def paired_interval(values: list[float]) -> tuple[float, float, float]:
    center = mean(values)
    if len(values) < 2:
        return center, center, center
    half_width = (
        student_t_quantile(0.975, len(values) - 1)
        * stdev(values)
        / math.sqrt(len(values))
    )
    return center, center - half_width, center + half_width


def collect_rows(
    out_dir: Path,
    *,
    models: list[str],
    seeds: list[int],
    protocols: list[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol in protocols:
        protocol_root = out_dir / protocol
        for model in models:
            for seed in seeds:
                run_dir = completed_run(
                    protocol_root,
                    model=model,
                    seed=seed,
                    protocol=protocol,
                    args=args,
                )
                if run_dir is None:
                    raise FileNotFoundError(
                        f"Missing completed run: {protocol}/{model}/seed{seed}"
                    )
                config = read_json(run_dir / "config.json")
                metrics = read_json(run_dir / "metrics.json")
                primary_key = (
                    "test_unseen_accuracy"
                    if protocol == "legacy_identity"
                    else "test_unseen_function_accuracy"
                )
                rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "seed": seed,
                        "parameters": int(metrics["parameters"]),
                        "primary_accuracy": float(metrics[primary_key]),
                        "nominal_unseen_identity_accuracy": (
                            float(metrics["test_unseen_accuracy"])
                            if protocol == "legacy_identity"
                            else ""
                        ),
                        "semantic_unseen_function_accuracy": float(
                            metrics["test_unseen_function_accuracy"]
                        ),
                        "overall_accuracy": float(metrics["test_accuracy"]),
                        "ce": float(metrics["test_ce"]),
                        "brier": float(metrics["test_brier"]),
                        "ece": float(metrics["test_ece"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "split_unit": config["dataset"]["split_unit"],
                        "function_sampling": config["dataset"][
                            "function_sampling"
                        ],
                        "function_split_overlap": int(
                            config["dataset"]["function_split_overlap"]
                        ),
                        "leaked_unseen_identities": int(
                            config["dataset"]["leaked_unseen_identities"]
                        ),
                        "leaked_unseen_identity_fraction": float(
                            config["dataset"][
                                "leaked_unseen_identity_fraction"
                            ]
                        ),
                        "split_checksum": (
                            config["dataset"]["identity_split_checksum"]
                            if protocol == "legacy_identity"
                            else config["dataset"]["function_split_checksum"]
                        ),
                        "run_dir": str(run_dir),
                    }
                )
    return rows


def aggregate(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    seeds: list[int],
    protocols: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    by_key = {
        (str(row["protocol"]), str(row["model"]), int(row["seed"])): row
        for row in rows
    }
    groups: list[dict[str, Any]] = []
    for protocol in protocols:
        for model in models:
            group = [
                by_key[(protocol, model, seed)]
                for seed in seeds
            ]
            primary_mean, primary_sd = metric_mean_sd(
                group,
                "primary_accuracy",
            )
            semantic_mean, semantic_sd = metric_mean_sd(
                group,
                "semantic_unseen_function_accuracy",
            )
            ce_mean, ce_sd = metric_mean_sd(group, "ce")
            brier_mean, brier_sd = metric_mean_sd(group, "brier")
            groups.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "n": len(group),
                    "parameters": group[0]["parameters"],
                    "primary_accuracy_mean": primary_mean,
                    "primary_accuracy_sd": primary_sd,
                    "semantic_unseen_function_accuracy_mean": semantic_mean,
                    "semantic_unseen_function_accuracy_sd": semantic_sd,
                    "ce_mean": ce_mean,
                    "ce_sd": ce_sd,
                    "brier_mean": brier_mean,
                    "brier_sd": brier_sd,
                }
            )

    comparisons: list[dict[str, Any]] = []
    positive_models = 0
    model_differences_by_seed: dict[int, list[float]] = {
        seed: [] for seed in seeds
    }
    for model in models:
        primary_differences = [
            float(
                by_key[("legacy_identity", model, seed)][
                    "primary_accuracy"
                ]
            )
            - float(
                by_key[("strict_function", model, seed)][
                    "primary_accuracy"
                ]
            )
            for seed in seeds
        ]
        semantic_differences = [
            float(
                by_key[("legacy_identity", model, seed)][
                    "semantic_unseen_function_accuracy"
                ]
            )
            - float(
                by_key[("strict_function", model, seed)][
                    "semantic_unseen_function_accuracy"
                ]
            )
            for seed in seeds
        ]
        for seed, difference in zip(seeds, primary_differences):
            model_differences_by_seed[seed].append(difference)
        center, low, high = paired_interval(primary_differences)
        semantic_center, semantic_low, semantic_high = paired_interval(
            semantic_differences
        )
        positive_models += int(center > 0.0)
        comparisons.append(
            {
                "model": model,
                "n": len(primary_differences),
                "legacy_nominal_minus_strict_mean": center,
                "legacy_nominal_minus_strict_ci_low": low,
                "legacy_nominal_minus_strict_ci_high": high,
                "legacy_semantic_minus_strict_mean": semantic_center,
                "legacy_semantic_minus_strict_ci_low": semantic_low,
                "legacy_semantic_minus_strict_ci_high": semantic_high,
                "wins": sum(value > 0.0 for value in primary_differences),
                "losses": sum(value < 0.0 for value in primary_differences),
                "ties": sum(value == 0.0 for value in primary_differences),
            }
        )

    pooled_by_seed = [
        mean(model_differences_by_seed[seed]) for seed in seeds
    ]
    pooled_center, pooled_low, pooled_high = paired_interval(pooled_by_seed)
    comparisons.append(
        {
            "model": "architecture_macro",
            "n": len(pooled_by_seed),
            "legacy_nominal_minus_strict_mean": pooled_center,
            "legacy_nominal_minus_strict_ci_low": pooled_low,
            "legacy_nominal_minus_strict_ci_high": pooled_high,
            "legacy_semantic_minus_strict_mean": "",
            "legacy_semantic_minus_strict_ci_low": "",
            "legacy_semantic_minus_strict_ci_high": "",
            "wins": sum(value > 0.0 for value in pooled_by_seed),
            "losses": sum(value < 0.0 for value in pooled_by_seed),
            "ties": sum(value == 0.0 for value in pooled_by_seed),
        }
    )

    group_by_key = {
        (str(row["protocol"]), str(row["model"])): row for row in groups
    }
    ranks: dict[str, dict[str, int]] = {}
    ranking_rows: list[dict[str, Any]] = []
    for protocol in protocols:
        ordered = sorted(
            models,
            key=lambda model: float(
                group_by_key[(protocol, model)][
                    "primary_accuracy_mean"
                ]
            ),
            reverse=True,
        )
        ranks[protocol] = {
            model: rank for rank, model in enumerate(ordered, start=1)
        }
    for model in models:
        legacy_rank = ranks["legacy_identity"][model]
        strict_rank = ranks["strict_function"][model]
        ranking_rows.append(
            {
                "model": model,
                "legacy_rank": legacy_rank,
                "strict_rank": strict_rank,
                "strict_minus_legacy_rank": strict_rank - legacy_rank,
            }
        )
    n_models = len(models)
    squared_rank_difference = sum(
        (
            ranks["legacy_identity"][model]
            - ranks["strict_function"][model]
        )
        ** 2
        for model in models
    )
    spearman = 1.0 - (
        6.0
        * squared_rank_difference
        / (n_models * (n_models * n_models - 1))
    )
    max_rank_move = max(
        abs(int(row["strict_minus_legacy_rank"])) for row in ranking_rows
    )
    legacy_leakage_by_seed = {
        seed: {
            "leaked_unseen_identities": int(
                by_key[("legacy_identity", models[0], seed)][
                    "leaked_unseen_identities"
                ]
            ),
            "leaked_unseen_identity_fraction": float(
                by_key[("legacy_identity", models[0], seed)][
                    "leaked_unseen_identity_fraction"
                ]
            ),
        }
        for seed in seeds
    }
    gates = {
        "legacy_leakage_by_seed": legacy_leakage_by_seed,
        "legacy_leaked_unseen_identities_mean": mean(
            value["leaked_unseen_identities"]
            for value in legacy_leakage_by_seed.values()
        ),
        "legacy_leaked_unseen_identity_fraction_mean": mean(
            value["leaked_unseen_identity_fraction"]
            for value in legacy_leakage_by_seed.values()
        ),
        "positive_architectures": positive_models,
        "architecture_macro_difference": pooled_center,
        "architecture_macro_ci": [pooled_low, pooled_high],
        "cross_architecture_inflation_supported": (
            len(seeds) >= 5 and positive_models >= 4 and pooled_low > 0.0
        ),
        "rank_spearman": spearman,
        "maximum_absolute_rank_change": max_rank_move,
        "ranking_instability_supported": (
            len(seeds) >= 5 and (max_rank_move >= 2 or spearman < 0.9)
        ),
    }
    return groups, comparisons, ranking_rows, gates


def analyze(
    out_dir: Path,
    *,
    models: list[str],
    seeds: list[int],
    protocols: list[str],
    args: argparse.Namespace,
) -> None:
    if set(protocols) != set(PROTOCOLS):
        raise ValueError("Analysis requires both registered protocols.")
    rows = collect_rows(
        out_dir,
        models=models,
        seeds=seeds,
        protocols=protocols,
        args=args,
    )
    groups, comparisons, rankings, gates = aggregate(
        rows,
        models=models,
        seeds=seeds,
        protocols=protocols,
    )
    analysis_dir = out_dir / "analysis"
    write_csv(analysis_dir / "seed_results.csv", rows)
    write_csv(analysis_dir / "group_summary.csv", groups)
    write_csv(analysis_dir / "paired_protocol_comparisons.csv", comparisons)
    write_csv(analysis_dir / "protocol_rankings.csv", rankings)
    write_json(analysis_dir / "claim_gates.json", gates)
    print(f"Wrote audit analysis to {analysis_dir}")
    print(json.dumps(gates, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the matched public-baseline legacy/strict split audit."
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "kfwa_experiments/"
            "runs_public_baseline_split_audit_k8_5seeds"
        ),
    )
    parser.add_argument("--models", default=",".join(PUBLIC_MODELS))
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-limit", type=int, default=20_000)
    parser.add_argument("--val-limit", type=int, default=5_000)
    parser.add_argument("--test-limit", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()

    models = parse_values(args.models)
    unknown_models = sorted(set(models) - set(PUBLIC_MODELS))
    if unknown_models:
        raise ValueError(f"Unregistered public models: {unknown_models}")
    seeds = parse_seeds(args.seeds)
    protocols = parse_values(args.protocols)
    unknown_protocols = sorted(set(protocols) - set(PROTOCOLS))
    if unknown_protocols:
        raise ValueError(f"Unregistered protocols: {unknown_protocols}")

    out_dir = Path(args.out_dir)
    run_script = Path(__file__).with_name("run_experiment.py")
    commands: list[list[str]] = []
    skipped: list[str] = []
    if not args.analysis_only:
        for protocol in protocols:
            protocol_root = out_dir / protocol
            selection_metric = (
                "val_unseen_accuracy"
                if protocol == "legacy_identity"
                else "val_unseen_function_accuracy"
            )
            for model in models:
                for seed in seeds:
                    existing = completed_run(
                        protocol_root,
                        model=model,
                        seed=seed,
                        protocol=protocol,
                        args=args,
                    )
                    if existing is not None:
                        skipped.append(str(existing))
                        continue
                    commands.append(
                        [
                            sys.executable,
                            str(run_script),
                            "--task",
                            "symbolic_rule_mapping",
                            "--model",
                            model,
                            "--seed",
                            str(seed),
                            "--out-dir",
                            str(protocol_root),
                            "--device",
                            args.device,
                            "--num-symbols",
                            "20",
                            "--support-min-k",
                            "8",
                            "--support-max-k",
                            "8",
                            "--rule-families",
                            "affine,block,parity",
                            "--symbolic-split-mode",
                            protocol,
                            "--query-out-of-support-prob",
                            "1.0",
                            "--support-noise",
                            "0.1",
                            "--holdout-function-frac",
                            "0.25",
                            "--selection-metric",
                            selection_metric,
                            "--train-limit",
                            str(args.train_limit),
                            "--val-limit",
                            str(args.val_limit),
                            "--test-limit",
                            str(args.test_limit),
                            "--batch-size",
                            str(args.batch_size),
                            "--num-workers",
                            "0",
                            "--epochs",
                            str(args.epochs),
                            "--patience",
                            str(args.patience),
                            "--lr",
                            "0.001",
                            "--weight-decay",
                            "0.0001",
                            "--dropout",
                            "0.05",
                            "--hidden-dim",
                            "256",
                            "--model-dim",
                            "64",
                            "--layers",
                            "2",
                            "--heads",
                            "4",
                            "--ff-dim",
                            "128",
                            "--public-latent-count",
                            "32",
                            "--no-progress",
                        ]
                    )

    manifest = {
        "models": models,
        "seeds": seeds,
        "protocols": protocols,
        "device": args.device,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "scheduled_commands": len(commands),
        "skipped_completed_runs": len(skipped),
    }
    write_json(out_dir / "sweep_config.json", manifest)

    for index, command in enumerate(commands, start=1):
        print(
            f"[{index}/{len(commands)}] {' '.join(command)}",
            flush=True,
        )
        if not args.dry_run:
            subprocess.run(command, check=True)

    if args.dry_run:
        print(
            f"Dry run: {len(commands)} commands, "
            f"{len(skipped)} completed runs skipped."
        )
        return 0

    analyze(
        out_dir,
        models=models,
        seeds=seeds,
        protocols=protocols,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
