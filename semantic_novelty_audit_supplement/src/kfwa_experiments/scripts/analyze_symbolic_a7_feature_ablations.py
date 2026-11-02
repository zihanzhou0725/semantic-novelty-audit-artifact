from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import mpmath
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import get_symbolic_rule_mapping_loaders
from kfwa.models import build_model
from kfwa.training import move_batch, unpack_model_output
from kfwa.utils import get_device


VARIANTS = {
    "full": "symbolic_transformer_rule_features",
    "no_fourier": "symbolic_transformer_rule_features_no_fourier",
    "no_parity": "symbolic_transformer_rule_features_no_parity",
    "no_block": "symbolic_transformer_rule_features_no_block",
    "no_scaled": "symbolic_transformer_rule_features_no_scaled",
    "no_delta": "symbolic_transformer_rule_features_no_delta",
}


def parse_ints(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", maxsplit=1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(chunk))
    return values


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def discover_run(root: Path, model_name: str, seed: int) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for config_path in sorted(root.glob("*/config.json")):
        config = read_json(config_path)
        if config.get("model") != model_name or int(config.get("seed", -1)) != seed:
            continue
        checkpoint = config_path.parent / "best_model.pt"
        metrics = config_path.parent / "metrics.json"
        if not checkpoint.exists() or not metrics.exists():
            raise FileNotFoundError(f"Incomplete run: {config_path.parent}")
        matches.append(
            {
                "run_dir": config_path.parent,
                "checkpoint": checkpoint,
                "config": config,
                "metrics": read_json(metrics),
            }
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one model={model_name}, seed={seed} run under {root}; "
            f"found {len(matches)}."
        )
    return matches[0]


def protocol_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config["task"],
        config["seed"],
        config["num_symbols"],
        config["support_min_k"],
        config["support_max_k"],
        config["rule_families"],
        config["holdout_rule_frac"],
        config["query_out_of_support_prob"],
        config["support_noise"],
        config["test_limit"],
        config["dataset"]["function_split_checksum"],
    )


def load_model(run: dict[str, Any], device: torch.device) -> torch.nn.Module:
    config = run["config"]
    model = build_model(
        Namespace(**config),
        context_dim=0,
        target_dim=int(config["num_symbols"]),
    )
    state = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def empty_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "correct": 0,
        "ce_sum": 0.0,
        "brier_sum": 0.0,
        "function_ids": set(),
    }


def update_stats(
    stats: dict[str, Any],
    *,
    correct: bool,
    ce: float,
    brier: float,
    function_id: int,
) -> None:
    stats["count"] += 1
    stats["correct"] += int(correct)
    stats["ce_sum"] += ce
    stats["brier_sum"] += brier
    stats["function_ids"].add(function_id)


def evaluate_seed(
    runs: dict[str, dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    signatures = {variant: protocol_signature(run["config"]) for variant, run in runs.items()}
    if len(set(signatures.values())) != 1:
        raise AssertionError(f"Variant protocol mismatch: {signatures}")
    config = runs["full"]["config"]
    seed = int(config["seed"])
    support_k = int(config["support_min_k"])
    if support_k != int(config["support_max_k"]):
        raise ValueError("Feature ablation analysis requires fixed support K.")
    bundle = get_symbolic_rule_mapping_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        train_limit=1,
        val_limit=1,
        test_limit=int(config["test_limit"]),
        num_symbols=int(config["num_symbols"]),
        support_min_k=support_k,
        support_max_k=support_k,
        rule_families=str(config["rule_families"]),
        holdout_rule_frac=float(config["holdout_rule_frac"]),
        query_out_of_support_prob=float(config["query_out_of_support_prob"]),
        support_noise=float(config["support_noise"]),
    )
    checksum = str(bundle.metadata["function_split_checksum"])
    if checksum != str(config["dataset"]["function_split_checksum"]):
        raise AssertionError("Regenerated function split checksum mismatch.")
    models = {variant: load_model(run, device) for variant, run in runs.items()}
    family_names = list(bundle.metadata["rule_families"])
    totals: dict[tuple[str, str, str], dict[str, Any]] = {}

    with torch.no_grad():
        for batch, target in bundle.test:
            batch = move_batch(batch, device)
            target = target.to(device)
            one_hot = torch.nn.functional.one_hot(
                target.long(),
                num_classes=int(config["num_symbols"]),
            ).float()
            predictions: dict[str, dict[str, torch.Tensor]] = {}
            for variant, model in models.items():
                logits, _ = unpack_model_output(model(batch))
                probs = logits.softmax(dim=-1)
                target_probability = probs.gather(1, target.unsqueeze(1)).squeeze(1)
                predictions[variant] = {
                    "correct": probs.argmax(dim=1).eq(target),
                    "ce": -target_probability.clamp_min(1.0e-12).log(),
                    "brier": (probs - one_hot).square().sum(dim=1),
                }

            for index in range(target.shape[0]):
                family = family_names[int(batch["rule_family"][index].cpu())]
                seen_group = (
                    "seen" if bool(batch["seen_function"][index].cpu()) else "unseen"
                )
                function_id = int(batch["function_id"][index].cpu())
                strata = (
                    ("overall", "all"),
                    ("seen_function", seen_group),
                    ("rule_family", family),
                    ("family_x_seen", f"{family}|{seen_group}"),
                )
                for variant, values in predictions.items():
                    for stratum_type, stratum_value in strata:
                        key = (variant, stratum_type, stratum_value)
                        stats = totals.setdefault(key, empty_stats())
                        update_stats(
                            stats,
                            correct=bool(values["correct"][index].cpu()),
                            ce=float(values["ce"][index].cpu()),
                            brier=float(values["brier"][index].cpu()),
                            function_id=function_id,
                        )

    output: list[dict[str, Any]] = []
    for (variant, stratum_type, stratum_value), stats in sorted(totals.items()):
        count = int(stats["count"])
        output.append(
            {
                "variant": variant,
                "model": VARIANTS[variant],
                "seed": seed,
                "support_k": support_k,
                "support_noise": float(config["support_noise"]),
                "split_checksum": checksum,
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "episode_count": count,
                "function_count": len(stats["function_ids"]),
                "accuracy": stats["correct"] / float(count),
                "ce": stats["ce_sum"] / float(count),
                "brier": stats["brier_sum"] / float(count),
                "run_dir": str(runs[variant]["run_dir"]),
            }
        )
    return output


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * float(
        mpmath.betainc(
            degrees_of_freedom / 2.0,
            0.5,
            0,
            x,
            regularized=True,
        )
    )
    return 1.0 - tail if value > 0.0 else tail


def student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    lower = 0.0
    upper = 100.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def exact_sign_flip_p(differences: list[float]) -> float:
    observed = abs(sum(differences))
    extreme = 0
    for mask in range(1 << len(differences)):
        signed_sum = sum(
            (1.0 if (mask >> index) & 1 else -1.0) * difference
            for index, difference in enumerate(differences)
        )
        extreme += int(abs(signed_sum) >= observed - 1.0e-15)
    return extreme / float(1 << len(differences))


def aggregate_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["variant"]),
            str(row["stratum_type"]),
            str(row["stratum_value"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (variant, stratum_type, stratum_value), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "variant": variant,
            "model": VARIANTS[variant],
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "n": len(group),
            "episode_count_total": sum(int(row["episode_count"]) for row in group),
        }
        for metric in ("accuracy", "ce", "brier"):
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["variant"]),
            int(row["seed"]),
            str(row["stratum_type"]),
            str(row["stratum_value"]),
        ): row
        for row in rows
    }
    strata = sorted(
        {
            (str(row["stratum_type"]), str(row["stratum_value"]))
            for row in rows
        }
    )
    seeds = sorted({int(row["seed"]) for row in rows})
    output: list[dict[str, Any]] = []
    for ablation in VARIANTS:
        if ablation == "full":
            continue
        for stratum_type, stratum_value in strata:
            for metric in ("accuracy", "ce", "brier"):
                differences = [
                    float(lookup[("full", seed, stratum_type, stratum_value)][metric])
                    - float(lookup[(ablation, seed, stratum_type, stratum_value)][metric])
                    for seed in seeds
                ]
                average = mean(differences)
                sample_sd = stdev(differences) if len(differences) > 1 else 0.0
                standard_error = sample_sd / math.sqrt(len(differences))
                critical = (
                    student_t_quantile(0.975, len(differences) - 1)
                    if len(differences) > 1
                    else 0.0
                )
                output.append(
                    {
                        "ablation": ablation,
                        "ablated_model": VARIANTS[ablation],
                        "stratum_type": stratum_type,
                        "stratum_value": stratum_value,
                        "metric": metric,
                        "difference_definition": "full_minus_ablation",
                        "n": len(differences),
                        "mean_difference": average,
                        "sd_difference": sample_sd,
                        "ci95_low": average - critical * standard_error,
                        "ci95_high": average + critical * standard_error,
                        "paired_dz": average / sample_sd if sample_sd > 0.0 else float("inf"),
                        "full_wins": sum(difference > 0.0 for difference in differences),
                        "ablation_wins": sum(difference < 0.0 for difference in differences),
                        "ties": sum(difference == 0.0 for difference in differences),
                        "exact_sign_flip_p": exact_sign_flip_p(differences),
                        "seed_differences": "|".join(
                            f"{difference:.12g}" for difference in differences
                        ),
                    }
                )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-evaluate full A7 and leave-one-feature-group-out checkpoints."
    )
    parser.add_argument(
        "--full-runs-dir",
        default=(
            "kfwa_experiments/"
            "runs_function_disjoint_attribution_canary_k8_seed0_20k_60ep"
        ),
    )
    parser.add_argument(
        "--ablation-runs-dir",
        default="kfwa_experiments/runs_symbolic_a7_feature_ablation_k8_5seeds",
    )
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    full_root = Path(args.full_runs_dir)
    ablation_root = Path(args.ablation_runs_dir)
    device = get_device(args.device)
    rows: list[dict[str, Any]] = []
    for seed in parse_ints(args.seeds):
        runs = {
            variant: discover_run(
                full_root if variant == "full" else ablation_root,
                model_name,
                seed,
            )
            for variant, model_name in VARIANTS.items()
        }
        seed_rows = evaluate_seed(
            runs,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        rows.extend(seed_rows)
        print(f"seed={seed} strata={len(seed_rows)}", flush=True)

    group_rows = aggregate_strata(rows)
    comparison_rows = paired_comparisons(rows)
    write_csv(ablation_root / "stratified_seed_metrics.csv", rows)
    write_csv(ablation_root / "stratified_group_summary.csv", group_rows)
    write_csv(ablation_root / "paired_comparisons.csv", comparison_rows)
    print(f"Wrote {len(rows)} rows to {ablation_root / 'stratified_seed_metrics.csv'}")
    print(
        f"Wrote {len(group_rows)} rows to "
        f"{ablation_root / 'stratified_group_summary.csv'}"
    )
    print(
        f"Wrote {len(comparison_rows)} rows to "
        f"{ablation_root / 'paired_comparisons.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
