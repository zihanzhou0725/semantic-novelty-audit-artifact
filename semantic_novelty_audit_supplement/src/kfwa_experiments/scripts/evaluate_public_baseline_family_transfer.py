from __future__ import annotations

import argparse
import hashlib
import math
import sys
from argparse import Namespace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import (
    SymbolicRuleMappingDataset,
    build_symbolic_family_transfer_evaluation,
)
from kfwa.metrics import brier_score, expected_calibration_error
from kfwa.models import build_model
from kfwa.symbolic import SymbolicBayesianPredictor
from kfwa.training import move_batch, unpack_model_output
from kfwa.utils import get_device
from run_public_baseline_split_audit import (
    PUBLIC_MODELS,
    paired_interval,
    parse_seeds,
    parse_values,
    read_json,
    write_csv,
    write_json,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_run(root: Path, model_name: str, seed: int) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for config_path in sorted(root.glob("*/config.json")):
        config = read_json(config_path)
        if config.get("model") != model_name or int(config.get("seed", -1)) != seed:
            continue
        if config.get("symbolic_split_mode") != "strict_function":
            continue
        checkpoint = config_path.parent / "best_model.pt"
        metrics_path = config_path.parent / "metrics.json"
        if checkpoint.exists() and metrics_path.exists():
            matches.append(
                {
                    "run_dir": config_path.parent,
                    "checkpoint": checkpoint,
                    "config": config,
                    "metrics": read_json(metrics_path),
                }
            )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one strict model={model_name}, seed={seed} run under "
            f"{root}; found {len(matches)}."
        )
    return matches[0]


def protocol_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    dataset = config["dataset"]
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
        config["train_limit"],
        config["val_limit"],
        config["test_limit"],
        dataset["symbolic_split_mode"],
        dataset["function_split_checksum"],
    )


def load_model(
    run: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    config = run["config"]
    model = build_model(
        Namespace(**config),
        context_dim=0,
        target_dim=int(config["num_symbols"]),
    )
    state = torch.load(
        run["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def masked_family_probs(
    predictor: SymbolicBayesianPredictor,
    batch: dict[str, torch.Tensor],
    eligible_mask: torch.Tensor,
) -> torch.Tensor:
    log_likelihood = predictor.support_log_likelihood(batch)
    log_likelihood = log_likelihood.masked_fill(
        ~eligible_mask.unsqueeze(0),
        float("-inf"),
    )
    posterior = torch.softmax(log_likelihood, dim=1)
    return predictor.predictive_probs(batch, posterior)


def summarize_predictions(
    probabilities: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    target_probability = probabilities.gather(
        1,
        target.unsqueeze(1),
    ).squeeze(1)
    return {
        "accuracy": float(
            probabilities.argmax(dim=1).eq(target).float().mean().item()
        ),
        "ce": float(
            -target_probability.clamp_min(1.0e-12).log().mean().item()
        ),
        "brier": brier_score(
            probabilities,
            target,
            num_classes=probabilities.shape[1],
        ),
        "ece": expected_calibration_error(probabilities, target),
    }


def evaluate_family(
    *,
    config: dict[str, Any],
    models: dict[str, torch.nn.Module],
    seed: int,
    family: str,
    family_index: int,
    device: torch.device,
    batch_size: int,
    test_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_families = str(config["rule_families"])
    rules, eligible_ids, metadata = build_symbolic_family_transfer_evaluation(
        num_symbols=int(config["num_symbols"]),
        family=family,
        excluded_families=training_families,
    )
    training_function_ids = {
        int(function_id)
        for function_id in config["dataset"]["seen_function_ids"]
    }
    dataset = SymbolicRuleMappingDataset(
        size=test_limit,
        rules=rules,
        active_function_ids=eligible_ids,
        seen_function_ids=training_function_ids.intersection(eligible_ids),
        min_support_k=int(config["support_min_k"]),
        max_support_k=int(config["support_max_k"]),
        query_out_of_support_prob=float(
            config["query_out_of_support_prob"]
        ),
        support_noise=float(config["support_noise"]),
        seed=seed + 40_000 + family_index * 10_000,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    bayes = SymbolicBayesianPredictor(
        num_symbols=int(config["num_symbols"]),
        rule_families=family,
        target_dim=int(config["num_symbols"]),
        support_noise=float(config["support_noise"]),
    ).to(device)
    bayes.eval()
    eligible_set = set(eligible_ids)
    eligible_mask = torch.tensor(
        [
            int(function_id) in eligible_set
            for function_id in bayes.function_ids
        ],
        dtype=torch.bool,
        device=device,
    )
    if int(eligible_mask.sum()) != len(eligible_ids):
        raise AssertionError(f"Incomplete Bayes mask for {family}.")

    logits_by_model: dict[str, list[torch.Tensor]] = {
        model_name: [] for model_name in models
    }
    bayes_probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch, target in loader:
            batch = move_batch(batch, device)
            target = target.to(device)
            targets.append(target.cpu())
            bayes_probabilities.append(
                masked_family_probs(
                    bayes,
                    batch,
                    eligible_mask,
                ).cpu()
            )
            for model_name, model in models.items():
                logits, _ = unpack_model_output(model(batch))
                logits_by_model[model_name].append(logits.cpu())

    target = torch.cat(targets)
    rows: list[dict[str, Any]] = []
    bayes_metrics = summarize_predictions(
        torch.cat(bayes_probabilities),
        target,
    )
    rows.append(
        {
            "seed": seed,
            "family": family,
            "model": "family_conditioned_bayes",
            "n": len(target),
            **bayes_metrics,
        }
    )
    for model_name in models:
        logits = torch.cat(logits_by_model[model_name])
        metrics = summarize_predictions(logits.softmax(dim=-1), target)
        rows.append(
            {
                "seed": seed,
                "family": family,
                "model": model_name,
                "n": len(target),
                **metrics,
            }
        )
    return rows, metadata


def add_macro_rows(
    rows: list[dict[str, Any]],
    families: list[str],
) -> list[dict[str, Any]]:
    output = list(rows)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["seed"]), str(row["model"])), []).append(
            row
        )
    for (seed, model), group in sorted(groups.items()):
        if sorted(str(row["family"]) for row in group) != sorted(families):
            raise AssertionError(f"Incomplete family macro for {model}/seed{seed}.")
        output.append(
            {
                "seed": seed,
                "family": "heldout_macro",
                "model": model,
                "n": sum(int(row["n"]) for row in group),
                "accuracy": mean(float(row["accuracy"]) for row in group),
                "ce": mean(float(row["ce"]) for row in group),
                "brier": mean(float(row["brier"]) for row in group),
                "ece": mean(float(row["ece"]) for row in group),
            }
        )
    return output


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["family"]), str(row["model"])),
            [],
        ).append(row)
    output: list[dict[str, Any]] = []
    for (family, model), group in sorted(groups.items()):
        result: dict[str, Any] = {
            "family": family,
            "model": model,
            "seeds": len(group),
            "episodes_per_seed": group[0]["n"],
        }
        for metric in ("accuracy", "ce", "brier", "ece"):
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = mean(values)
            result[f"{metric}_sd"] = (
                stdev(values) if len(values) > 1 else 0.0
            )
        output.append(result)
    return output


def compare_to_transformer(
    rows: list[dict[str, Any]],
    models: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    macro = {
        (str(row["model"]), int(row["seed"])): row
        for row in rows
        if row["family"] == "heldout_macro"
    }
    output: list[dict[str, Any]] = []
    for model in models:
        if model == "symbolic_transformer":
            continue
        row: dict[str, Any] = {"model": model, "reference": "symbolic_transformer"}
        for metric in ("accuracy", "ce", "brier"):
            differences = [
                float(macro[(model, seed)][metric])
                - float(macro[("symbolic_transformer", seed)][metric])
                for seed in seeds
            ]
            center, low, high = paired_interval(differences)
            row[f"{metric}_difference_mean"] = center
            row[f"{metric}_difference_ci_low"] = low
            row[f"{metric}_difference_ci_high"] = high
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate strict public baselines on held-out families."
    )
    parser.add_argument(
        "--runs-dir",
        default=(
            "kfwa_experiments/"
            "runs_public_baseline_split_audit_k8_5seeds/strict_function"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "kfwa_experiments/"
            "runs_public_baseline_split_audit_k8_5seeds/family_transfer"
        ),
    )
    parser.add_argument("--models", default=",".join(PUBLIC_MODELS))
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--families", default="quadratic,piecewise_affine")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-limit-per-family", type=int, default=5_000)
    args = parser.parse_args()

    models_to_evaluate = parse_values(args.models)
    seeds = parse_seeds(args.seeds)
    families = parse_values(args.families)
    runs_root = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    device = get_device(args.device)

    all_rows: list[dict[str, Any]] = []
    family_metadata: dict[str, dict[str, Any]] = {}
    checkpoint_manifest: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        runs = {
            model_name: discover_run(runs_root, model_name, seed)
            for model_name in models_to_evaluate
        }
        signatures = {
            model_name: protocol_signature(run["config"])
            for model_name, run in runs.items()
        }
        if len(set(signatures.values())) != 1:
            raise AssertionError(
                f"Strict checkpoint protocol mismatch for seed {seed}."
            )
        config = runs["symbolic_transformer"]["config"]
        models = {
            model_name: load_model(run, device)
            for model_name, run in runs.items()
        }
        checkpoint_manifest[str(seed)] = {
            model_name: {
                "run_dir": str(run["run_dir"]),
                "sha256": file_sha256(run["checkpoint"]),
            }
            for model_name, run in runs.items()
        }
        for family_index, family in enumerate(families):
            rows, metadata = evaluate_family(
                config=config,
                models=models,
                seed=seed,
                family=family,
                family_index=family_index,
                device=device,
                batch_size=args.batch_size,
                test_limit=args.test_limit_per_family,
            )
            all_rows.extend(rows)
            if family in family_metadata and family_metadata[family] != metadata:
                raise AssertionError(
                    f"Family metadata changed across seeds for {family}."
                )
            family_metadata[family] = metadata
            summary = {
                row["model"]: 100.0 * float(row["accuracy"])
                for row in rows
            }
            print(f"seed={seed} family={family} accuracy={summary}")

    seed_rows = add_macro_rows(all_rows, families)
    group_rows = aggregate_rows(seed_rows)
    comparisons = compare_to_transformer(
        seed_rows,
        models=models_to_evaluate,
        seeds=seeds,
    )
    write_csv(out_dir / "seed_summary.csv", seed_rows)
    write_csv(out_dir / "group_summary.csv", group_rows)
    write_csv(out_dir / "paired_vs_transformer.csv", comparisons)
    write_json(
        out_dir / "manifest.json",
        {
            "protocol": "frozen_strict_public_baseline_family_transfer",
            "training_families": ["affine", "block", "parity"],
            "heldout_families": families,
            "models": models_to_evaluate,
            "seeds": seeds,
            "test_limit_per_family": args.test_limit_per_family,
            "device": str(device),
            "family_metadata": family_metadata,
            "checkpoints": checkpoint_manifest,
        },
    )
    print(f"Wrote family-transfer results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
