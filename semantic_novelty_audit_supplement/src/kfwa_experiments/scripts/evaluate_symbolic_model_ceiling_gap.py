from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import (
    SymbolicRule,
    build_symbolic_rules,
    get_symbolic_rule_mapping_loaders,
    symbolic_function_representatives,
)
from kfwa.models import build_model
from kfwa.symbolic import SymbolicBayesianPredictor
from kfwa.training import move_batch, unpack_model_output
from kfwa.utils import get_device


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def categorical_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1.0e-12).log()).sum(dim=-1)


def confidence_bin(confidence: float) -> str:
    if confidence < 0.50:
        return "[0.00,0.50)"
    if confidence < 0.70:
        return "[0.50,0.70)"
    if confidence < 0.85:
        return "[0.70,0.85)"
    if confidence < 0.95:
        return "[0.85,0.95)"
    if confidence < 0.99:
        return "[0.95,0.99)"
    return "[0.99,1.00]"


def effective_count_bin(value: float) -> str:
    if value < 1.5:
        return "[1,1.5)"
    if value < 2.5:
        return "[1.5,2.5)"
    if value < 5.0:
        return "[2.5,5)"
    if value < 10.0:
        return "[5,10)"
    return "[10,+inf)"


def candidate_count_bin(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 2:
        return "2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    return "11+"


def parameter_regions(rule: SymbolicRule, num_symbols: int) -> tuple[str, str]:
    if rule.family == "affine":
        multiplier, offset = rule.params
        if multiplier == 1:
            multiplier_region = "identity"
        elif multiplier == num_symbols - 1:
            multiplier_region = "reflection"
        else:
            multiplier_region = "other"
        return (
            f"affine|multiplier={multiplier_region}",
            f"affine|offset={'zero' if offset == 0 else 'nonzero'}",
        )
    first_shift, second_shift, group_flip = rule.params
    relation = "same" if first_shift == second_shift else "different"
    if rule.family == "block":
        return (
            f"block|swap={group_flip}",
            f"block|shift_relation={relation}",
        )
    if rule.family == "parity":
        return (
            f"parity|flip={group_flip}",
            f"parity|shift_relation={relation}",
        )
    return (f"{rule.family}|primary=other", f"{rule.family}|relation=other")


def discover_run(
    runs_dir: Path,
    *,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for config_path in sorted(runs_dir.glob("*/config.json")):
        config = read_json(config_path)
        if config.get("model") != model_name or int(config.get("seed", -1)) != seed:
            continue
        run_dir = config_path.parent
        checkpoint_path = run_dir / "best_model.pt"
        metrics_path = run_dir / "metrics.json"
        if not checkpoint_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Incomplete run directory: {run_dir}")
        matches.append(
            {
                "run_dir": run_dir,
                "checkpoint_path": checkpoint_path,
                "config": config,
                "metrics": read_json(metrics_path),
            }
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one run for model={model_name}, seed={seed}; found {len(matches)}."
        )
    return matches[0]


def protocol_value(config: dict[str, Any], key: str) -> Any:
    if key == "function_split_checksum":
        return config["dataset"][key]
    return config[key]


def validate_paired_protocol(
    baseline_run: dict[str, Any],
    feature_run: dict[str, Any],
) -> None:
    keys = (
        "task",
        "seed",
        "num_symbols",
        "support_min_k",
        "support_max_k",
        "rule_families",
        "holdout_rule_frac",
        "query_out_of_support_prob",
        "support_noise",
        "test_limit",
        "function_split_checksum",
    )
    for key in keys:
        baseline = protocol_value(baseline_run["config"], key)
        feature = protocol_value(feature_run["config"], key)
        if baseline != feature:
            raise AssertionError(f"Paired protocol mismatch for {key}: {baseline!r} != {feature!r}")
    if baseline_run["config"]["task"] != "symbolic_rule_mapping":
        raise ValueError("This evaluator only supports symbolic_rule_mapping checkpoints.")


def load_model(run: dict[str, Any], device: torch.device) -> torch.nn.Module:
    config = run["config"]
    model = build_model(
        Namespace(**config),
        context_dim=0,
        target_dim=int(config["num_symbols"]),
    )
    state = torch.load(
        run["checkpoint_path"],
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def prediction_tensors(
    probs: torch.Tensor,
    *,
    target: torch.Tensor,
    one_hot_target: torch.Tensor,
    bayes_probs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    confidence, prediction = probs.max(dim=1)
    target_probability = probs.gather(1, target.unsqueeze(1)).squeeze(1)
    posterior_expected_accuracy = bayes_probs.gather(
        1,
        prediction.unsqueeze(1),
    ).squeeze(1)
    return {
        "prediction": prediction,
        "correct": prediction.eq(target),
        "confidence": confidence,
        "target_probability": target_probability,
        "ce": -target_probability.clamp_min(1.0e-12).log(),
        "brier": (probs - one_hot_target).square().sum(dim=1),
        "posterior_expected_accuracy": posterior_expected_accuracy,
    }


def outcome_name(a0_correct: bool, a7_correct: bool) -> str:
    if a0_correct and a7_correct:
        return "both_correct"
    if not a0_correct and a7_correct:
        return "a7_recovers_a0"
    if a0_correct and not a7_correct:
        return "a7_regresses_from_a0"
    return "both_wrong"


def evaluate_seed(
    *,
    baseline_run: dict[str, Any],
    feature_run: dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    test_limit: int,
    high_confidence_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_paired_protocol(baseline_run, feature_run)
    config = baseline_run["config"]
    seed = int(config["seed"])
    support_k = int(config["support_min_k"])
    if support_k != int(config["support_max_k"]):
        raise ValueError("Model-ceiling diagnostics require fixed support K.")
    effective_test_limit = test_limit or int(config["test_limit"])
    bundle = get_symbolic_rule_mapping_loaders(
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        train_limit=1,
        val_limit=1,
        test_limit=effective_test_limit,
        num_symbols=int(config["num_symbols"]),
        support_min_k=support_k,
        support_max_k=support_k,
        rule_families=str(config["rule_families"]),
        holdout_rule_frac=float(config["holdout_rule_frac"]),
        query_out_of_support_prob=float(config["query_out_of_support_prob"]),
        support_noise=float(config["support_noise"]),
    )
    expected_checksum = str(config["dataset"]["function_split_checksum"])
    actual_checksum = str(bundle.metadata["function_split_checksum"])
    if actual_checksum != expected_checksum:
        raise AssertionError(
            f"Regenerated split checksum mismatch: {actual_checksum} != {expected_checksum}"
        )

    a0 = load_model(baseline_run, device)
    a7 = load_model(feature_run, device)
    bayes = SymbolicBayesianPredictor(
        num_symbols=int(config["num_symbols"]),
        rule_families=str(config["rule_families"]),
        target_dim=int(config["num_symbols"]),
        support_noise=float(config["support_noise"]),
    ).to(device)
    bayes.eval()
    representatives = symbolic_function_representatives(
        build_symbolic_rules(
            num_symbols=int(config["num_symbols"]),
            families=str(config["rule_families"]),
        )
    )
    family_names = list(bundle.metadata["rule_families"])

    rows: list[dict[str, Any]] = []
    episode_index = 0
    with torch.no_grad():
        for batch, target in bundle.test:
            batch = move_batch(batch, device)
            target = target.to(device)
            function_ids = batch["function_id"].long()
            function_indices = torch.searchsorted(bayes.function_ids, function_ids)
            if not torch.equal(bayes.function_ids[function_indices], function_ids):
                raise AssertionError("A dataset function_id is absent from the Bayes table.")

            true_tables = bayes.rule_table[function_indices]
            clean_support = true_tables.gather(1, batch["support_digits"].long())
            corruption_count = (
                clean_support.ne(batch["support_labels"].long())
                & batch["support_mask"].bool()
            ).sum(dim=1)
            clean_target = true_tables.gather(
                1,
                batch["query_digit"].long().unsqueeze(1),
            ).squeeze(1)
            if not torch.equal(clean_target, target):
                raise AssertionError("Dataset targets disagree with the Bayes function table.")

            function_posterior = bayes.posterior_function_probs(batch)
            bayes_probs = bayes.predictive_probs(batch, function_posterior)
            a0_logits, _ = unpack_model_output(a0(batch))
            a7_logits, _ = unpack_model_output(a7(batch))
            a0_probs = a0_logits.softmax(dim=-1)
            a7_probs = a7_logits.softmax(dim=-1)
            one_hot_target = torch.nn.functional.one_hot(
                target.long(),
                num_classes=int(config["num_symbols"]),
            ).float()
            diagnostics = {
                "bayes": prediction_tensors(
                    bayes_probs,
                    target=target,
                    one_hot_target=one_hot_target,
                    bayes_probs=bayes_probs,
                ),
                "a0": prediction_tensors(
                    a0_probs,
                    target=target,
                    one_hot_target=one_hot_target,
                    bayes_probs=bayes_probs,
                ),
                "a7": prediction_tensors(
                    a7_probs,
                    target=target,
                    one_hot_target=one_hot_target,
                    bayes_probs=bayes_probs,
                ),
            }
            bayes_confidence = diagnostics["bayes"]["confidence"]
            for name in ("a0", "a7"):
                if torch.any(
                    diagnostics[name]["posterior_expected_accuracy"]
                    > bayes_confidence + 1.0e-6
                ):
                    raise AssertionError(f"{name} decision exceeds Bayes conditional value.")

            function_entropy = categorical_entropy(function_posterior)
            function_effective_count = function_entropy.exp()
            predictive_entropy = categorical_entropy(bayes_probs)
            support_log_likelihood = bayes.support_log_likelihood(batch)
            ml_mask = support_log_likelihood.eq(
                support_log_likelihood.max(dim=1, keepdim=True).values
            )
            true_function_posterior = function_posterior.gather(
                1,
                function_indices.unsqueeze(1),
            ).squeeze(1)

            for row_index in range(target.shape[0]):
                function_id = int(function_ids[row_index].cpu())
                rule = representatives[function_id]
                family_id = int(batch["rule_family"][row_index].cpu())
                if rule.family != family_names[family_id]:
                    raise AssertionError("Dataset family metadata disagrees with canonical rule.")
                primary_region, relation_region = parameter_regions(
                    rule,
                    int(config["num_symbols"]),
                )
                confidence = float(bayes_confidence[row_index].cpu())
                high_confidence = confidence >= high_confidence_threshold
                a0_correct = bool(diagnostics["a0"]["correct"][row_index].cpu())
                a7_correct = bool(diagnostics["a7"]["correct"][row_index].cpu())
                candidate_count = int(ml_mask[row_index].sum().cpu())
                effective_count = float(function_effective_count[row_index].cpu())
                row: dict[str, Any] = {
                    "episode_id": (
                        f"noise={float(config['support_noise']):.4f}|"
                        f"K={support_k}|seed={seed}|index={episode_index}"
                    ),
                    "seed": seed,
                    "episode_index": episode_index,
                    "support_k": support_k,
                    "support_noise": float(config["support_noise"]),
                    "split_checksum": actual_checksum,
                    "function_id": function_id,
                    "canonical_rule_id": rule.rule_id,
                    "rule_family_id": family_id,
                    "rule_family": rule.family,
                    "rule_params": "|".join(str(value) for value in rule.params),
                    "parameter_primary_region": primary_region,
                    "parameter_relation_region": relation_region,
                    "seen_function": bool(batch["seen_function"][row_index].cpu()),
                    "seen_group": (
                        "seen"
                        if bool(batch["seen_function"][row_index].cpu())
                        else "unseen"
                    ),
                    "query_digit": int(batch["query_digit"][row_index].cpu()),
                    "target": int(target[row_index].cpu()),
                    "query_in_support": bool(batch["query_in_support"][row_index].cpu()),
                    "corruption_count": int(corruption_count[row_index].cpu()),
                    "function_posterior_entropy": float(function_entropy[row_index].cpu()),
                    "function_effective_count": effective_count,
                    "function_effective_count_bin": effective_count_bin(effective_count),
                    "true_function_posterior": float(true_function_posterior[row_index].cpu()),
                    "ml_candidate_count": candidate_count,
                    "ml_candidate_count_bin": candidate_count_bin(candidate_count),
                    "predictive_entropy": float(predictive_entropy[row_index].cpu()),
                    "bayes_confidence": confidence,
                    "ambiguity_bin": confidence_bin(confidence),
                    "bayes_solvability": (
                        "high_confidence" if high_confidence else "ambiguous"
                    ),
                    "bayes_high_confidence": high_confidence,
                    "a0_a7_outcome": outcome_name(a0_correct, a7_correct),
                }
                for name, values in diagnostics.items():
                    for metric_name, tensor in values.items():
                        value = tensor[row_index].cpu()
                        if metric_name == "prediction":
                            row[f"{name}_{metric_name}"] = int(value)
                        elif metric_name == "correct":
                            row[f"{name}_{metric_name}"] = int(bool(value))
                        else:
                            row[f"{name}_{metric_name}"] = float(value)
                row["a0_decision_regret"] = (
                    row["bayes_posterior_expected_accuracy"]
                    - row["a0_posterior_expected_accuracy"]
                )
                row["a7_decision_regret"] = (
                    row["bayes_posterior_expected_accuracy"]
                    - row["a7_posterior_expected_accuracy"]
                )
                row["a7_minus_a0_correct"] = row["a7_correct"] - row["a0_correct"]
                row["a7_minus_a0_expected_accuracy"] = (
                    row["a7_posterior_expected_accuracy"]
                    - row["a0_posterior_expected_accuracy"]
                )
                row["a7_ce_reduction"] = row["a0_ce"] - row["a7_ce"]
                row["a7_brier_reduction"] = row["a0_brier"] - row["a7_brier"]
                row["a0_excess_ce"] = row["a0_ce"] - row["bayes_ce"]
                row["a7_excess_ce"] = row["a7_ce"] - row["bayes_ce"]
                row["a0_excess_brier"] = row["a0_brier"] - row["bayes_brier"]
                row["a7_excess_brier"] = row["a7_brier"] - row["bayes_brier"]
                row["a0_error_on_bayes_high_confidence"] = int(
                    high_confidence and not a0_correct
                )
                row["a7_error_on_bayes_high_confidence"] = int(
                    high_confidence and not a7_correct
                )
                rows.append(row)
                episode_index += 1

    run_info = {
        "seed": seed,
        "support_k": support_k,
        "support_noise": float(config["support_noise"]),
        "split_checksum": actual_checksum,
        "test_limit": effective_test_limit,
        "checkpoint_test_limit": int(config["test_limit"]),
        "baseline_run_dir": str(baseline_run["run_dir"]),
        "feature_run_dir": str(feature_run["run_dir"]),
        "baseline_checkpoint_sha256": file_sha256(baseline_run["checkpoint_path"]),
        "feature_checkpoint_sha256": file_sha256(feature_run["checkpoint_path"]),
        "baseline_metrics": baseline_run["metrics"],
        "feature_metrics": feature_run["metrics"],
    }
    return rows, run_info


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / float(denominator) if denominator else float("nan")


def summarize_episode_group(
    rows: list[dict[str, Any]],
    *,
    support_noise: float,
    support_k: int,
    stratum_type: str,
    stratum_value: str,
    total_count: int,
) -> dict[str, Any]:
    def avg(key: str) -> float:
        return mean(float(row[key]) for row in rows)

    high_confidence_rows = [row for row in rows if bool(row["bayes_high_confidence"])]
    a0_errors = sum(1 for row in rows if not bool(row["a0_correct"]))
    a0_correct = len(rows) - a0_errors
    a7_wins = sum(
        1
        for row in rows
        if not bool(row["a0_correct"]) and bool(row["a7_correct"])
    )
    a7_losses = sum(
        1
        for row in rows
        if bool(row["a0_correct"]) and not bool(row["a7_correct"])
    )
    a0_high_confidence_errors = sum(
        int(row["a0_error_on_bayes_high_confidence"]) for row in rows
    )
    a7_high_confidence_errors = sum(
        int(row["a7_error_on_bayes_high_confidence"]) for row in rows
    )
    return {
        "support_noise": support_noise,
        "support_k": support_k,
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "episode_count": len(rows),
        "fraction": len(rows) / float(total_count),
        "seed_count": len({int(row["seed"]) for row in rows}),
        "function_count": len({int(row["function_id"]) for row in rows}),
        "mean_corruption_count": avg("corruption_count"),
        "mean_function_effective_count": avg("function_effective_count"),
        "mean_bayes_confidence": avg("bayes_confidence"),
        "bayes_accuracy": avg("bayes_correct"),
        "a0_accuracy": avg("a0_correct"),
        "a7_accuracy": avg("a7_correct"),
        "a0_realized_gap_to_bayes": avg("bayes_correct") - avg("a0_correct"),
        "a7_realized_gap_to_bayes": avg("bayes_correct") - avg("a7_correct"),
        "a7_minus_a0_accuracy": avg("a7_minus_a0_correct"),
        "bayes_ce": avg("bayes_ce"),
        "a0_ce": avg("a0_ce"),
        "a7_ce": avg("a7_ce"),
        "a0_excess_ce": avg("a0_excess_ce"),
        "a7_excess_ce": avg("a7_excess_ce"),
        "a7_ce_reduction": avg("a7_ce_reduction"),
        "bayes_brier": avg("bayes_brier"),
        "a0_brier": avg("a0_brier"),
        "a7_brier": avg("a7_brier"),
        "a0_excess_brier": avg("a0_excess_brier"),
        "a7_excess_brier": avg("a7_excess_brier"),
        "a7_brier_reduction": avg("a7_brier_reduction"),
        "bayes_posterior_expected_accuracy": avg(
            "bayes_posterior_expected_accuracy"
        ),
        "a0_posterior_expected_accuracy": avg("a0_posterior_expected_accuracy"),
        "a7_posterior_expected_accuracy": avg("a7_posterior_expected_accuracy"),
        "a0_decision_regret": avg("a0_decision_regret"),
        "a7_decision_regret": avg("a7_decision_regret"),
        "a7_minus_a0_expected_accuracy": avg("a7_minus_a0_expected_accuracy"),
        "a7_win_count": a7_wins,
        "a7_loss_count": a7_losses,
        "a7_net_win_count": a7_wins - a7_losses,
        "a7_recovery_rate_of_a0_errors": safe_rate(a7_wins, a0_errors),
        "a7_regression_rate_of_a0_correct": safe_rate(a7_losses, a0_correct),
        "bayes_high_confidence_count": len(high_confidence_rows),
        "a0_error_on_bayes_high_confidence_count": a0_high_confidence_errors,
        "a7_error_on_bayes_high_confidence_count": a7_high_confidence_errors,
        "a0_error_rate_on_bayes_high_confidence": safe_rate(
            a0_high_confidence_errors,
            len(high_confidence_rows),
        ),
        "a7_error_rate_on_bayes_high_confidence": safe_rate(
            a7_high_confidence_errors,
            len(high_confidence_rows),
        ),
    }


def stratify_episodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    base_groups: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["support_noise"]), int(row["support_k"]))
        base_groups.setdefault(key, []).append(row)

    strata = (
        ("overall", lambda row: "all"),
        ("rule_family", lambda row: str(row["rule_family"])),
        ("seen_function", lambda row: str(row["seen_group"])),
        ("corruption_count", lambda row: str(row["corruption_count"])),
        ("ambiguity", lambda row: str(row["ambiguity_bin"])),
        (
            "function_effective_count",
            lambda row: str(row["function_effective_count_bin"]),
        ),
        ("ml_candidate_count", lambda row: str(row["ml_candidate_count_bin"])),
        ("bayes_solvability", lambda row: str(row["bayes_solvability"])),
        (
            "family_parameter_primary",
            lambda row: str(row["parameter_primary_region"]),
        ),
        (
            "family_parameter_relation",
            lambda row: str(row["parameter_relation_region"]),
        ),
        ("a0_a7_outcome", lambda row: str(row["a0_a7_outcome"])),
        (
            "family_x_corruption",
            lambda row: f"{row['rule_family']}|corruption={row['corruption_count']}",
        ),
    )
    for (support_noise, support_k), base_rows in sorted(base_groups.items()):
        for stratum_type, value_fn in strata:
            groups: dict[str, list[dict[str, Any]]] = {}
            for row in base_rows:
                groups.setdefault(value_fn(row), []).append(row)
            for stratum_value, group in sorted(groups.items()):
                output.append(
                    summarize_episode_group(
                        group,
                        support_noise=support_noise,
                        support_k=support_k,
                        stratum_type=stratum_type,
                        stratum_value=stratum_value,
                        total_count=len(base_rows),
                    )
                )
    return output


def summarize_seeds(
    rows: list[dict[str, Any]],
    run_info_by_seed: dict[int, dict[str, Any]],
    *,
    compare_stored_metrics: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in sorted(run_info_by_seed):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        if not seed_rows:
            raise AssertionError(f"No episode rows found for seed {seed}.")
        info = run_info_by_seed[seed]
        summary = summarize_episode_group(
            seed_rows,
            support_noise=float(seed_rows[0]["support_noise"]),
            support_k=int(seed_rows[0]["support_k"]),
            stratum_type="seed",
            stratum_value=str(seed),
            total_count=len(seed_rows),
        )
        unseen = [row for row in seed_rows if row["seen_group"] == "unseen"]
        summary.update(
            {
                "seed": seed,
                "split_checksum": info["split_checksum"],
                "a0_unseen_function_accuracy": mean(
                    float(row["a0_correct"]) for row in unseen
                ),
                "a7_unseen_function_accuracy": mean(
                    float(row["a7_correct"]) for row in unseen
                ),
                "a7_minus_a0_unseen_function_accuracy": mean(
                    float(row["a7_correct"]) - float(row["a0_correct"])
                    for row in unseen
                ),
                "baseline_run_dir": info["baseline_run_dir"],
                "feature_run_dir": info["feature_run_dir"],
                "baseline_checkpoint_sha256": info["baseline_checkpoint_sha256"],
                "feature_checkpoint_sha256": info["feature_checkpoint_sha256"],
            }
        )
        for prefix, stored_key in (
            ("a0", "baseline_metrics"),
            ("a7", "feature_metrics"),
        ):
            stored = info[stored_key]
            for metric_name in ("accuracy", "ce", "brier"):
                stored_value = float(stored[f"test_{metric_name}"])
                regenerated_value = float(summary[f"{prefix}_{metric_name}"])
                summary[f"{prefix}_stored_{metric_name}"] = stored_value
                summary[f"{prefix}_{metric_name}_reproduction_error"] = (
                    regenerated_value - stored_value
                )
                if compare_stored_metrics and abs(regenerated_value - stored_value) > 1.0e-5:
                    raise AssertionError(
                        f"{prefix} seed={seed} test_{metric_name} reproduction error "
                        f"{regenerated_value - stored_value:.3g} exceeds tolerance."
                    )
            stored_unseen = float(stored["test_unseen_function_accuracy"])
            regenerated_unseen = float(
                summary[f"{prefix}_unseen_function_accuracy"]
            )
            summary[f"{prefix}_stored_unseen_function_accuracy"] = stored_unseen
            summary[f"{prefix}_unseen_function_accuracy_reproduction_error"] = (
                regenerated_unseen - stored_unseen
            )
            if compare_stored_metrics and abs(regenerated_unseen - stored_unseen) > 1.0e-5:
                raise AssertionError(
                    f"{prefix} seed={seed} unseen accuracy reproduction error "
                    f"{regenerated_unseen - stored_unseen:.3g} exceeds tolerance."
                )
        output.append(summary)
    return output


def aggregate_seed_summary(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "bayes_accuracy",
        "a0_accuracy",
        "a7_accuracy",
        "a0_unseen_function_accuracy",
        "a7_unseen_function_accuracy",
        "a7_minus_a0_accuracy",
        "a7_minus_a0_unseen_function_accuracy",
        "a0_ce",
        "a7_ce",
        "a7_ce_reduction",
        "a0_brier",
        "a7_brier",
        "a7_brier_reduction",
        "a0_decision_regret",
        "a7_decision_regret",
        "a0_error_rate_on_bayes_high_confidence",
        "a7_error_rate_on_bayes_high_confidence",
    )
    summary: dict[str, Any] = {
        "support_noise": seed_rows[0]["support_noise"],
        "support_k": seed_rows[0]["support_k"],
        "n": len(seed_rows),
    }
    for metric_name in metrics:
        values = [float(row[metric_name]) for row in seed_rows]
        summary[f"{metric_name}_mean"] = mean(values)
        summary[f"{metric_name}_sd"] = stdev(values) if len(values) > 1 else 0.0
    return [summary]


def validate_ceiling_alignment(
    rows: list[dict[str, Any]],
    ceiling_path: Path,
) -> None:
    episode_ids = {row["episode_id"] for row in rows}
    with ceiling_path.open(newline="", encoding="utf-8") as handle:
        ceiling_rows = {
            row["episode_id"]: row
            for row in csv.DictReader(handle)
            if row["episode_id"] in episode_ids
        }
    if len(ceiling_rows) != len(rows):
        raise AssertionError(
            f"Ceiling alignment found {len(ceiling_rows)} of {len(rows)} episode ids."
        )
    exact_fields = (
        "seed",
        "episode_index",
        "support_k",
        "split_checksum",
        "function_id",
        "rule_family",
        "seen_group",
        "query_digit",
        "target",
        "corruption_count",
        "ml_candidate_count",
        "bayes_prediction",
        "bayes_correct",
    )
    float_fields = (
        "support_noise",
        "function_posterior_entropy",
        "function_effective_count",
        "true_function_posterior",
        "predictive_entropy",
        "bayes_confidence",
        "bayes_ce",
        "bayes_brier",
        "bayes_posterior_expected_accuracy",
    )
    for row in rows:
        ceiling = ceiling_rows[row["episode_id"]]
        for field in exact_fields:
            if str(row[field]) != ceiling[field]:
                raise AssertionError(
                    f"Ceiling mismatch for {row['episode_id']} field={field}: "
                    f"{row[field]!r} != {ceiling[field]!r}"
                )
        for field in float_fields:
            if abs(float(row[field]) - float(ceiling[field])) > 1.0e-6:
                raise AssertionError(
                    f"Ceiling mismatch for {row['episode_id']} field={field}."
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align trained symbolic models with the exact Bayes ceiling per episode."
    )
    parser.add_argument(
        "--runs-dir",
        default=(
            "kfwa_experiments/"
            "runs_function_disjoint_attribution_canary_k8_seed0_20k_60ep"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="kfwa_experiments/runs_symbolic_model_ceiling_gap_k8_5seeds",
    )
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--baseline-model", default="symbolic_transformer")
    parser.add_argument(
        "--feature-model",
        default="symbolic_transformer_rule_features",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--test-limit",
        type=int,
        default=0,
        help="Override checkpoint test_limit; zero uses the stored value.",
    )
    parser.add_argument("--bayes-high-confidence", type=float, default=0.99)
    parser.add_argument(
        "--ceiling-episodes",
        default=(
            "kfwa_experiments/"
            "runs_symbolic_ceiling_strict_k6_k8_5seeds/episodes.csv"
        ),
    )
    parser.add_argument("--skip-ceiling-alignment", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    seeds = parse_ints(args.seeds)
    device = get_device(args.device)
    rows: list[dict[str, Any]] = []
    run_info_by_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        baseline_run = discover_run(
            runs_dir,
            model_name=args.baseline_model,
            seed=seed,
        )
        feature_run = discover_run(
            runs_dir,
            model_name=args.feature_model,
            seed=seed,
        )
        seed_rows, run_info = evaluate_seed(
            baseline_run=baseline_run,
            feature_run=feature_run,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            test_limit=args.test_limit,
            high_confidence_threshold=args.bayes_high_confidence,
        )
        rows.extend(seed_rows)
        run_info_by_seed[seed] = run_info
        print(
            f"seed={seed} episodes={len(seed_rows)} "
            f"A0={mean(float(row['a0_correct']) for row in seed_rows):.4f} "
            f"A7={mean(float(row['a7_correct']) for row in seed_rows):.4f} "
            f"Bayes={mean(float(row['bayes_correct']) for row in seed_rows):.4f}",
            flush=True,
        )

    expected_rows = sum(int(info["test_limit"]) for info in run_info_by_seed.values())
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(rows)}.")
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise AssertionError("Episode ids are not unique.")

    ceiling_path = Path(args.ceiling_episodes)
    if not args.skip_ceiling_alignment:
        if not ceiling_path.exists():
            raise FileNotFoundError(f"Missing ceiling episode report: {ceiling_path}")
        validate_ceiling_alignment(rows, ceiling_path)
        print(f"Aligned {len(rows)} episodes with {ceiling_path}", flush=True)

    # metrics.json has no test_examples in historical runs, so the stored config
    # determines whether exact checkpoint metric reproduction is expected.
    using_full_test = all(
        int(info["test_limit"]) == int(info["checkpoint_test_limit"])
        for info in run_info_by_seed.values()
    )
    seed_rows = summarize_seeds(
        rows,
        run_info_by_seed,
        compare_stored_metrics=using_full_test,
    )
    stratified_rows = stratify_episodes(rows)
    group_rows = aggregate_seed_summary(seed_rows)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "episodes.csv", rows)
    write_csv(out_dir / "seed_summary.csv", seed_rows)
    write_csv(out_dir / "group_summary.csv", group_rows)
    write_csv(out_dir / "stratified_summary.csv", stratified_rows)
    manifest = {
        "runs_dir": str(runs_dir),
        "baseline_model": args.baseline_model,
        "feature_model": args.feature_model,
        "seeds": seeds,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "test_limit_override": args.test_limit,
        "bayes_high_confidence": args.bayes_high_confidence,
        "ceiling_episodes": str(ceiling_path),
        "ceiling_episodes_sha256": (
            file_sha256(ceiling_path)
            if ceiling_path.exists() and not args.skip_ceiling_alignment
            else None
        ),
        "episode_count": len(rows),
        "stratified_row_count": len(stratified_rows),
        "runs": [
            {
                key: value
                for key, value in run_info_by_seed[seed].items()
                if key not in {"baseline_metrics", "feature_metrics"}
            }
            for seed in seeds
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"Wrote {len(rows)} rows to {out_dir / 'episodes.csv'}")
    print(f"Wrote {len(seed_rows)} rows to {out_dir / 'seed_summary.csv'}")
    print(f"Wrote {len(group_rows)} rows to {out_dir / 'group_summary.csv'}")
    print(
        f"Wrote {len(stratified_rows)} rows to "
        f"{out_dir / 'stratified_summary.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
