from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import get_symbolic_rule_mapping_loaders
from kfwa.symbolic import (
    SymbolicBayesianPredictor,
    SymbolicMaximumLikelihoodPredictor,
    SymbolicRuleOracle,
)
from kfwa.training import move_batch, run_epoch
from kfwa.utils import get_device


PREDICTORS = {
    "bayes": SymbolicBayesianPredictor,
    "ml": SymbolicMaximumLikelihoodPredictor,
    "legacy": SymbolicRuleOracle,
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


def parse_floats(raw: str) -> list[float]:
    return [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


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


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (float(row["support_noise"]), int(row["support_k"]), str(row["predictor"]))
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    metric_names = (
        "test_accuracy",
        "test_seen_function_accuracy",
        "test_unseen_function_accuracy",
        "test_posterior_expected_accuracy",
        "test_posterior_expected_regret",
        "test_ce",
        "test_brier",
        "test_ece",
    )
    for (noise, support_k, predictor), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "support_noise": noise,
            "support_k": support_k,
            "predictor": predictor,
            "n": len(group),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in group]
            summary[f"{metric_name}_mean"] = mean(values)
            summary[f"{metric_name}_sd"] = stdev(values) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def posterior_expected_accuracy(
    predictor: torch.nn.Module,
    bayesian_reference: SymbolicBayesianPredictor,
    loader: Any,
    device: torch.device,
) -> float:
    """Evaluate a predictor decision under the exact conditional label posterior."""

    predictor.eval()
    bayesian_reference.eval()
    expected_correct = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch, _ in loader:
            batch = move_batch(batch, device)
            posterior_predictive = bayesian_reference(batch).softmax(dim=-1)
            decisions = predictor(batch).argmax(dim=-1, keepdim=True)
            expected_correct += float(posterior_predictive.gather(1, decisions).sum().cpu())
            sample_count += decisions.shape[0]
    return expected_correct / max(sample_count, 1)


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


def collect_episode_diagnostics(
    bayes: SymbolicBayesianPredictor,
    ml: SymbolicMaximumLikelihoodPredictor,
    loader: Any,
    device: torch.device,
    *,
    seed: int,
    support_k: int,
    support_noise: float,
    family_names: list[str],
    split_checksum: str,
) -> list[dict[str, Any]]:
    """Collect exact posterior and realized-corruption diagnostics per episode."""

    bayes.eval()
    ml.eval()
    rows: list[dict[str, Any]] = []
    episode_index = 0
    with torch.no_grad():
        for batch, target in loader:
            batch = move_batch(batch, device)
            target = target.to(device)
            function_indices = torch.searchsorted(bayes.function_ids, batch["function_id"].long())
            if not torch.equal(bayes.function_ids[function_indices], batch["function_id"].long()):
                raise AssertionError("A dataset function_id is absent from the oracle function table.")

            true_tables = bayes.rule_table[function_indices]
            clean_support = true_tables.gather(1, batch["support_digits"].long())
            corruption_mask = clean_support.ne(batch["support_labels"].long()) & batch["support_mask"].bool()
            corruption_count = corruption_mask.sum(dim=1)
            clean_target = true_tables.gather(1, batch["query_digit"].long().unsqueeze(1)).squeeze(1)
            if not torch.equal(clean_target, target):
                raise AssertionError("Dataset targets disagree with the oracle function table.")

            function_posterior = bayes.posterior_function_probs(batch)
            bayes_probs = bayes.predictive_probs(batch, function_posterior)
            ml_mask = ml.maximum_likelihood_mask(batch)
            ml_probs = ml(batch).softmax(dim=-1)
            bayes_confidence, bayes_prediction = bayes_probs.max(dim=1)
            ml_confidence, ml_prediction = ml_probs.max(dim=1)
            ml_expected_accuracy = bayes_probs.gather(1, ml_prediction.unsqueeze(1)).squeeze(1)
            if torch.any(ml_expected_accuracy > bayes_confidence + 1.0e-6):
                raise AssertionError("An ML decision exceeds the Bayes conditional decision value.")

            function_entropy = categorical_entropy(function_posterior)
            function_effective_count = function_entropy.exp()
            predictive_entropy = categorical_entropy(bayes_probs)
            match_counts = bayes.support_match_counts(batch)
            max_match_count = match_counts.max(dim=1).values
            true_function_posterior = function_posterior.gather(
                1,
                function_indices.unsqueeze(1),
            ).squeeze(1)
            one_hot_target = torch.nn.functional.one_hot(
                target.long(),
                num_classes=bayes.num_symbols,
            ).float()
            bayes_brier = (bayes_probs - one_hot_target).square().sum(dim=1)
            ml_brier = (ml_probs - one_hot_target).square().sum(dim=1)
            bayes_target_prob = bayes_probs.gather(1, target.unsqueeze(1)).squeeze(1)
            ml_target_prob = ml_probs.gather(1, target.unsqueeze(1)).squeeze(1)

            for row_index in range(target.shape[0]):
                family_id = int(batch["rule_family"][row_index].cpu())
                if not 0 <= family_id < len(family_names):
                    raise AssertionError(f"Unknown rule family id: {family_id}")
                expected_count = float(function_effective_count[row_index].cpu())
                candidate_count = int(ml_mask[row_index].sum().cpu())
                confidence = float(bayes_confidence[row_index].cpu())
                seen_function = bool(batch["seen_function"][row_index].cpu())
                corruptions = int(corruption_count[row_index].cpu())
                rows.append(
                    {
                        "episode_id": (
                            f"noise={support_noise:.4f}|K={support_k}|seed={seed}|index={episode_index}"
                        ),
                        "seed": seed,
                        "episode_index": episode_index,
                        "support_k": support_k,
                        "support_noise": support_noise,
                        "split_checksum": split_checksum,
                        "function_id": int(batch["function_id"][row_index].cpu()),
                        "rule_family_id": family_id,
                        "rule_family": family_names[family_id],
                        "seen_function": seen_function,
                        "seen_group": "seen" if seen_function else "unseen",
                        "query_digit": int(batch["query_digit"][row_index].cpu()),
                        "target": int(target[row_index].cpu()),
                        "query_in_support": bool(batch["query_in_support"][row_index].cpu()),
                        "corruption_count": corruptions,
                        "corruption_rate": corruptions / float(support_k),
                        "function_posterior_entropy": float(function_entropy[row_index].cpu()),
                        "function_posterior_entropy_normalized": float(
                            function_entropy[row_index].cpu()
                        )
                        / math.log(float(function_posterior.shape[1])),
                        "function_effective_count": expected_count,
                        "function_effective_count_bin": effective_count_bin(expected_count),
                        "true_function_posterior": float(true_function_posterior[row_index].cpu()),
                        "ml_candidate_count": candidate_count,
                        "ml_candidate_count_bin": candidate_count_bin(candidate_count),
                        "max_support_match_count": int(max_match_count[row_index].cpu()),
                        "predictive_entropy": float(predictive_entropy[row_index].cpu()),
                        "predictive_entropy_normalized": float(predictive_entropy[row_index].cpu())
                        / math.log(float(bayes.num_symbols)),
                        "bayes_confidence": confidence,
                        "ambiguity_bin": confidence_bin(confidence),
                        "bayes_prediction": int(bayes_prediction[row_index].cpu()),
                        "bayes_correct": int(bayes_prediction[row_index].eq(target[row_index]).cpu()),
                        "bayes_target_probability": float(bayes_target_prob[row_index].cpu()),
                        "bayes_ce": float(-bayes_target_prob[row_index].clamp_min(1.0e-12).log().cpu()),
                        "bayes_brier": float(bayes_brier[row_index].cpu()),
                        "bayes_posterior_expected_accuracy": confidence,
                        "ml_prediction": int(ml_prediction[row_index].cpu()),
                        "ml_correct": int(ml_prediction[row_index].eq(target[row_index]).cpu()),
                        "ml_confidence": float(ml_confidence[row_index].cpu()),
                        "ml_target_probability": float(ml_target_prob[row_index].cpu()),
                        "ml_ce": float(-ml_target_prob[row_index].clamp_min(1.0e-12).log().cpu()),
                        "ml_brier": float(ml_brier[row_index].cpu()),
                        "ml_posterior_expected_accuracy": float(
                            ml_expected_accuracy[row_index].cpu()
                        ),
                        "ml_posterior_expected_regret": float(
                            (bayes_confidence[row_index] - ml_expected_accuracy[row_index]).cpu()
                        ),
                    }
                )
                episode_index += 1
    return rows


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
        "mean_function_posterior_entropy": avg("function_posterior_entropy"),
        "mean_function_effective_count": avg("function_effective_count"),
        "mean_true_function_posterior": avg("true_function_posterior"),
        "mean_ml_candidate_count": avg("ml_candidate_count"),
        "mean_predictive_entropy": avg("predictive_entropy"),
        "mean_bayes_confidence": avg("bayes_confidence"),
        "bayes_accuracy": avg("bayes_correct"),
        "bayes_ce": avg("bayes_ce"),
        "bayes_brier": avg("bayes_brier"),
        "bayes_posterior_expected_accuracy": avg("bayes_posterior_expected_accuracy"),
        "ml_accuracy": avg("ml_correct"),
        "ml_ce": avg("ml_ce"),
        "ml_brier": avg("ml_brier"),
        "ml_posterior_expected_accuracy": avg("ml_posterior_expected_accuracy"),
        "ml_posterior_expected_regret": avg("ml_posterior_expected_regret"),
    }


def stratify_episodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    base_groups: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for row in rows:
        base_groups.setdefault((float(row["support_noise"]), int(row["support_k"])), []).append(row)

    strata = (
        ("overall", lambda row: "all"),
        ("rule_family", lambda row: str(row["rule_family"])),
        ("seen_function", lambda row: str(row["seen_group"])),
        ("corruption_count", lambda row: str(row["corruption_count"])),
        ("ambiguity", lambda row: str(row["ambiguity_bin"])),
        ("function_effective_count", lambda row: str(row["function_effective_count_bin"])),
        ("ml_candidate_count", lambda row: str(row["ml_candidate_count_bin"])),
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate exact symbolic information ceilings without training.")
    parser.add_argument("--out-dir", default="kfwa_experiments/runs_symbolic_ceiling")
    parser.add_argument("--predictors", default="bayes,ml,legacy")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--k-values", default="6,8")
    parser.add_argument("--support-noises", default="0.10")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-symbols", type=int, default=20)
    parser.add_argument("--rule-families", default="affine,block,parity")
    parser.add_argument("--holdout-function-frac", type=float, default=0.25)
    parser.add_argument("--query-out-of-support-prob", type=float, default=1.0)
    parser.add_argument("--test-limit", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--write-episode-report", action="store_true")
    args = parser.parse_args()

    predictor_names = [name.strip() for name in args.predictors.split(",") if name.strip()]
    unknown = sorted(set(predictor_names) - set(PREDICTORS))
    if unknown:
        raise ValueError(f"Unknown predictors: {unknown}")
    seeds = parse_ints(args.seeds)
    k_values = parse_ints(args.k_values)
    support_noises = parse_floats(args.support_noises)
    device = get_device(args.device)
    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for support_noise in support_noises:
        for support_k in k_values:
            for seed in seeds:
                bundle = get_symbolic_rule_mapping_loaders(
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    seed=seed,
                    train_limit=1,
                    val_limit=1,
                    test_limit=args.test_limit,
                    num_symbols=args.num_symbols,
                    support_min_k=support_k,
                    support_max_k=support_k,
                    rule_families=args.rule_families,
                    holdout_rule_frac=args.holdout_function_frac,
                    query_out_of_support_prob=args.query_out_of_support_prob,
                    support_noise=support_noise,
                )
                bayesian_reference = SymbolicBayesianPredictor(
                    num_symbols=args.num_symbols,
                    rule_families=args.rule_families,
                    target_dim=args.num_symbols,
                    support_noise=support_noise,
                ).to(device)
                maximum_likelihood_reference = SymbolicMaximumLikelihoodPredictor(
                    num_symbols=args.num_symbols,
                    rule_families=args.rule_families,
                    target_dim=args.num_symbols,
                    support_noise=support_noise,
                ).to(device)
                if args.write_episode_report:
                    episode_rows.extend(
                        collect_episode_diagnostics(
                            bayesian_reference,
                            maximum_likelihood_reference,
                            bundle.test,
                            device,
                            seed=seed,
                            support_k=support_k,
                            support_noise=support_noise,
                            family_names=list(bundle.metadata["rule_families"]),
                            split_checksum=str(bundle.metadata["function_split_checksum"]),
                        )
                    )
                bayes_expected_accuracy = posterior_expected_accuracy(
                    bayesian_reference,
                    bayesian_reference,
                    bundle.test,
                    device,
                )
                for predictor_name in predictor_names:
                    predictor = PREDICTORS[predictor_name](
                        num_symbols=args.num_symbols,
                        rule_families=args.rule_families,
                        target_dim=args.num_symbols,
                        support_noise=support_noise,
                    ).to(device)
                    metrics = run_epoch(
                        predictor,
                        bundle.test,
                        device=device,
                        optimizer=None,
                        phase="test",
                        show_progress=False,
                    )
                    expected_accuracy = posterior_expected_accuracy(
                        predictor,
                        bayesian_reference,
                        bundle.test,
                        device,
                    )
                    row = {
                        "predictor": predictor_name,
                        "seed": seed,
                        "support_k": support_k,
                        "support_noise": support_noise,
                        "num_symbols": args.num_symbols,
                        "rule_families": args.rule_families,
                        "num_functions": bundle.metadata["num_functions"],
                        "function_split_checksum": bundle.metadata["function_split_checksum"],
                        "function_split_overlap": bundle.metadata["function_split_overlap"],
                        "test_posterior_expected_accuracy": expected_accuracy,
                        "test_posterior_expected_regret": bayes_expected_accuracy - expected_accuracy,
                        **metrics.as_dict(prefix="test_"),
                    }
                    rows.append(row)
                    print(
                        f"noise={support_noise:.2f} K={support_k} seed={seed} "
                        f"predictor={predictor_name} "
                        f"accuracy={metrics.accuracy:.4f} "
                        f"unseen_function={metrics.unseen_function_accuracy:.4f} "
                        f"posterior_expected={expected_accuracy:.4f} "
                        f"ce={metrics.ce:.4f}"
                    )

    out_dir = Path(args.out_dir)
    summaries = aggregate(rows)
    write_csv(out_dir / "summary.csv", rows)
    write_csv(out_dir / "group_summary.csv", summaries)
    if args.write_episode_report:
        expected_episode_count = (
            len(support_noises) * len(k_values) * len(seeds) * args.test_limit
        )
        if len(episode_rows) != expected_episode_count:
            raise AssertionError(
                f"Expected {expected_episode_count} episode rows, found {len(episode_rows)}."
            )
        stratified_rows = stratify_episodes(episode_rows)
        write_csv(out_dir / "episodes.csv", episode_rows)
        write_csv(out_dir / "stratified_summary.csv", stratified_rows)
        print(f"Wrote {len(episode_rows)} rows to {out_dir / 'episodes.csv'}")
        print(f"Wrote {len(stratified_rows)} rows to {out_dir / 'stratified_summary.csv'}")
    print(f"Wrote {len(rows)} rows to {out_dir / 'summary.csv'}")
    print(f"Wrote {len(summaries)} rows to {out_dir / 'group_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
