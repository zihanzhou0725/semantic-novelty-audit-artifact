from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import get_task_loaders, parse_rule_families
from kfwa.models import build_model
from kfwa.training import move_batch, unpack_model_output
from kfwa.utils import count_parameters, get_device, set_seed

from run_experiment import add_args

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_GROUPS = [
    (
        "m64",
        [
            "kfwa_experiments/runs_main_result_confirm_10seeds/summary.csv",
            "kfwa_experiments/runs_main_result_confirm_seeds10_19/summary.csv",
        ],
    ),
    (
        "m32",
        [
            "kfwa_experiments/runs_lop_oscillator_sweep_m32_5seeds/summary.csv",
            "kfwa_experiments/runs_lop_oscillator_sweep_m32_seeds5_19/summary.csv",
        ],
    ),
]

METRIC_COLUMNS = [
    "confidence",
    "margin",
    "order_r1",
    "order_r2",
    "order_r3",
    "order_r4",
    "order_radius_mean",
    "cluster_entropy",
    "cluster_dominant_mass",
    "cluster_mass_std",
    "lop_feature_norm",
]


@dataclass(frozen=True)
class RunSpec:
    group: str
    seed: int
    run_dir: Path
    summary_metrics: dict[str, str]


def add_diagnostic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="kfwa_experiments/mechanism_diagnostics")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--max-runs-per-group", type=int, default=0)
    parser.add_argument("--model", default="symbolic_kfwa_lop_features_no_kuramoto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--query-out-of-support-prob",
        type=float,
        default=None,
        help="Optional diagnostic-only override for the dataset query-out-of-support probability.",
    )
    parser.add_argument(
        "--group",
        action="append",
        help="Group spec in the form label:path1,path2. Defaults compare m64 and m32 confirmed runs.",
    )


def parse_group_specs(raw_groups: list[str] | None) -> list[tuple[str, list[str]]]:
    if not raw_groups:
        return DEFAULT_GROUPS
    parsed: list[tuple[str, list[str]]] = []
    for raw in raw_groups:
        if ":" not in raw:
            raise ValueError(f"Invalid group spec {raw!r}; expected label:path1,path2.")
        label, paths = raw.split(":", 1)
        csv_paths = [chunk.strip() for chunk in paths.split(",") if chunk.strip()]
        if not label.strip() or not csv_paths:
            raise ValueError(f"Invalid group spec {raw!r}; label and paths are required.")
        parsed.append((label.strip(), csv_paths))
    return parsed


def load_summary_runs(groups: list[tuple[str, list[str]]], model_name: str, max_runs_per_group: int) -> list[RunSpec]:
    runs: list[RunSpec] = []
    seen: set[tuple[str, int, str]] = set()
    for group, paths in groups:
        group_runs: list[RunSpec] = []
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row.get("model") != model_name:
                        continue
                    run_dir = Path(row["run_dir"])
                    seed = int(row["seed"])
                    key = (group, seed, str(run_dir))
                    if key in seen:
                        continue
                    seen.add(key)
                    group_runs.append(RunSpec(group=group, seed=seed, run_dir=run_dir, summary_metrics=row))
        group_runs.sort(key=lambda spec: spec.seed)
        if max_runs_per_group > 0:
            group_runs = group_runs[:max_runs_per_group]
        runs.extend(group_runs)
    return runs


def default_experiment_args(model: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_args(parser)
    return parser.parse_args(["--model", model])


def args_from_config(config: dict[str, Any]) -> argparse.Namespace:
    args = default_experiment_args(str(config["model"]))
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def build_loader_from_config(config: dict[str, Any], batch_size: int):
    return get_task_loaders(
        task=str(config.get("task", "symbolic_rule_mapping")),
        source=str(config.get("source", "mnist")),
        data_root=str(config.get("data_root", "kfwa_experiments/data")),
        download=False,
        batch_size=batch_size,
        num_workers=0,
        seed=int(config.get("seed", 0)),
        num_contexts=int(config.get("num_contexts", 8)),
        holdout_contexts=int(config.get("holdout_contexts", 2)),
        context_encoding=str(config.get("context_encoding", "perm_matrix")),
        train_limit=int(config.get("train_limit", 50_000)),
        val_limit=int(config.get("val_limit", 10_000)),
        test_limit=int(config.get("test_limit", 10_000)),
        quick=bool(config.get("quick", False)),
        num_symbols=int(config.get("num_symbols", 10)),
        support_min_k=int(config.get("support_min_k", 2)),
        support_max_k=int(config.get("support_max_k", 6)),
        rule_families=str(config.get("rule_families", "affine,block,parity")),
        holdout_rule_frac=float(config.get("holdout_rule_frac", 0.25)),
        query_out_of_support_prob=float(config.get("query_out_of_support_prob", 0.5)),
        support_noise=float(config.get("support_noise", 0.0)),
    )


def normalized_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    entropy = -(probs.clamp_min(1.0e-8) * probs.clamp_min(1.0e-8).log()).sum(dim=dim)
    denom = math.log(float(probs.shape[dim])) if probs.shape[dim] > 1 else 1.0
    return entropy / max(denom, 1.0e-8)


def extract_cluster_metrics(model: torch.nn.Module, theta: torch.Tensor) -> dict[str, torch.Tensor]:
    lop = getattr(model, "lop", None)
    if lop is None or getattr(lop, "cluster_encoder", None) is None:
        size = theta.shape[0]
        nan = theta.new_full((size,), float("nan"))
        return {
            "cluster_entropy": nan,
            "cluster_dominant_mass": nan,
            "cluster_mass_std": nan,
        }
    cluster_temp = lop.cluster_log_temp.exp()
    anchors = lop.cluster_anchors.view(1, 1, -1)
    cluster_logits = cluster_temp * torch.cos(theta.unsqueeze(-1) - anchors)
    cluster_mass = torch.softmax(cluster_logits, dim=-1).mean(dim=1)
    return {
        "cluster_entropy": normalized_entropy(cluster_mass, dim=-1),
        "cluster_dominant_mass": cluster_mass.max(dim=-1).values,
        "cluster_mass_std": cluster_mass.std(dim=-1, unbiased=False),
    }


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def finite_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = safe_float(row.get(metric))
        if math.isfinite(value):
            values.append(value)
    return values


def summarize_group(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)

    output: list[dict[str, Any]] = []
    for key_values, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        row: dict[str, Any] = {key: value for key, value in zip(keys, key_values)}
        row["n"] = len(group_rows)
        row["accuracy"] = mean(float(item["correct"]) for item in group_rows)
        for metric in METRIC_COLUMNS:
            values = finite_values(group_rows, metric)
            row[f"{metric}_mean"] = mean(values) if values else float("nan")
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output.append(row)
    return output


def cohen_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = stdev(a) ** 2
    var_b = stdev(b) ** 2
    pooled = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / max(len(a) + len(b) - 2, 1))
    if pooled <= 1.0e-12:
        return float("nan")
    return (mean(a) - mean(b)) / pooled


def add_binary_contrasts(
    rows: list[dict[str, Any]],
    output: list[dict[str, Any]],
    group_key: str,
    split_key: str,
    positive_value: Any,
    negative_value: Any,
    contrast_name: str,
) -> None:
    group_values = sorted({row[group_key] for row in rows})
    for group in group_values:
        group_rows = [row for row in rows if row[group_key] == group]
        pos = [row for row in group_rows if row.get(split_key) == positive_value]
        neg = [row for row in group_rows if row.get(split_key) == negative_value]
        contrast_row: dict[str, Any] = {
            "contrast": contrast_name,
            group_key: group,
            "positive_label": positive_value,
            "negative_label": negative_value,
            "n_positive": len(pos),
            "n_negative": len(neg),
        }
        for metric in ["correct", *METRIC_COLUMNS]:
            pos_values = finite_values(pos, metric)
            neg_values = finite_values(neg, metric)
            contrast_row[f"{metric}_positive_mean"] = mean(pos_values) if pos_values else float("nan")
            contrast_row[f"{metric}_negative_mean"] = mean(neg_values) if neg_values else float("nan")
            contrast_row[f"{metric}_delta"] = (
                contrast_row[f"{metric}_positive_mean"] - contrast_row[f"{metric}_negative_mean"]
                if pos_values and neg_values
                else float("nan")
            )
            contrast_row[f"{metric}_cohen_d"] = cohen_d(pos_values, neg_values)
        output.append(contrast_row)


def add_variant_contrast(rows: list[dict[str, Any]], output: list[dict[str, Any]], a_label: str, b_label: str) -> None:
    a_rows = [row for row in rows if row["group"] == a_label]
    b_rows = [row for row in rows if row["group"] == b_label]
    if not a_rows or not b_rows:
        return
    contrast_row: dict[str, Any] = {
        "contrast": f"{a_label}_vs_{b_label}",
        "positive_label": a_label,
        "negative_label": b_label,
        "n_positive": len(a_rows),
        "n_negative": len(b_rows),
    }
    for metric in ["correct", *METRIC_COLUMNS]:
        a_values = finite_values(a_rows, metric)
        b_values = finite_values(b_rows, metric)
        contrast_row[f"{metric}_positive_mean"] = mean(a_values) if a_values else float("nan")
        contrast_row[f"{metric}_negative_mean"] = mean(b_values) if b_values else float("nan")
        contrast_row[f"{metric}_delta"] = (
            contrast_row[f"{metric}_positive_mean"] - contrast_row[f"{metric}_negative_mean"]
            if a_values and b_values
            else float("nan")
        )
        contrast_row[f"{metric}_cohen_d"] = cohen_d(a_values, b_values)
    output.append(contrast_row)


def feature_structure(rows: list[dict[str, Any]], vectors: list[np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        buckets[(str(row["group"]), str(row["rule_family_name"]))].append(idx)

    family_rows: list[dict[str, Any]] = []
    centroids: dict[tuple[str, str], np.ndarray] = {}
    within: dict[tuple[str, str], float] = {}
    for key, indices in sorted(buckets.items()):
        matrix = np.stack([vectors[idx] for idx in indices], axis=0)
        centroid = matrix.mean(axis=0)
        rms = float(np.sqrt(((matrix - centroid) ** 2).sum(axis=1).mean()))
        centroids[key] = centroid
        within[key] = rms
        group, family = key
        group_rows = [rows[idx] for idx in indices]
        family_rows.append(
            {
                "group": group,
                "rule_family_name": family,
                "n": len(indices),
                "accuracy": mean(float(item["correct"]) for item in group_rows),
                "centroid_norm": float(np.linalg.norm(centroid)),
                "within_rms": rms,
            }
        )

    pair_rows: list[dict[str, Any]] = []
    groups = sorted({group for group, _ in centroids})
    for group in groups:
        families = sorted(family for candidate_group, family in centroids if candidate_group == group)
        for i, family_a in enumerate(families):
            for family_b in families[i + 1 :]:
                key_a = (group, family_a)
                key_b = (group, family_b)
                distance = float(np.linalg.norm(centroids[key_a] - centroids[key_b]))
                denom = 0.5 * (within[key_a] + within[key_b])
                pair_rows.append(
                    {
                        "group": group,
                        "family_a": family_a,
                        "family_b": family_b,
                        "centroid_distance": distance,
                        "mean_within_rms": denom,
                        "separation_ratio": distance / denom if denom > 1.0e-12 else float("nan"),
                    }
                )
    return family_rows, pair_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2, sort_keys=True)


def iter_limited(loader: Iterable[Any], max_batches: int):
    for batch_index, item in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        yield batch_index, item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose LOP synchrony features from trained symbolic checkpoints.")
    add_diagnostic_args(parser)
    args = parser.parse_args(argv)

    groups = parse_group_specs(args.group)
    runs = load_summary_runs(groups, model_name=args.model, max_runs_per_group=args.max_runs_per_group)
    if not runs:
        raise ValueError("No runs matched the requested model and group summaries.")

    device = get_device(args.device)
    out_root = Path(args.out_dir)
    run_dir = out_root / datetime.now().strftime("symbolic_lop_mechanisms_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    sample_rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    run_rows: list[dict[str, Any]] = []

    run_iter = runs
    if not args.no_progress and tqdm is not None:
        run_iter = tqdm(runs, desc="diagnostic runs", unit="run", dynamic_ncols=True, file=sys.stdout)

    for spec in run_iter:
        config_path = spec.run_dir / "config.json"
        checkpoint_path = spec.run_dir / "best_model.pt"
        if not config_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing config or checkpoint under {spec.run_dir}")

        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config = deepcopy(config)
        config["batch_size"] = args.batch_size
        if args.query_out_of_support_prob is not None:
            config["query_out_of_support_prob"] = args.query_out_of_support_prob
        model_args = args_from_config(config)
        set_seed(int(config.get("seed", spec.seed)))
        bundle = build_loader_from_config(config, batch_size=args.batch_size)
        model = build_model(model_args, context_dim=bundle.context_dim, target_dim=bundle.target_dim)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        families = parse_rule_families(str(config.get("rule_families", "affine,block,parity")))
        run_total = 0
        run_correct = 0
        with torch.inference_mode():
            for batch_index, (batch_cpu, target_cpu) in iter_limited(bundle.test, args.max_batches):
                batch = move_batch(batch_cpu, device)
                target = target_cpu.to(device)
                logits, _ = unpack_model_output(model(batch))
                probs = torch.softmax(logits, dim=-1)
                top_values, top_indices = probs.topk(k=2, dim=-1)
                pred = top_indices[:, 0]
                correct = pred.eq(target)

                theta = getattr(model, "last_theta", None)
                order = getattr(model.lop, "last_order_parameters", None)
                lop_features = getattr(model, "last_lop_features", None)
                if theta is None or order is None or lop_features is None:
                    raise RuntimeError("Model did not expose expected LOP diagnostic tensors.")

                radii = order[:, 2 * model.lop.harmonics : 3 * model.lop.harmonics]
                cluster_metrics = extract_cluster_metrics(model, theta)
                lop_norm = lop_features.norm(dim=-1)
                batch_size = int(target.shape[0])
                run_total += batch_size
                run_correct += int(correct.sum().item())

                for i in range(batch_size):
                    family_id = int(batch["rule_family"][i].detach().cpu().item())
                    family_name = families[family_id] if 0 <= family_id < len(families) else str(family_id)
                    row = {
                        "group": spec.group,
                        "seed": spec.seed,
                        "batch_index": batch_index,
                        "sample_index": i,
                        "run_dir": str(spec.run_dir),
                        "parameters": int(spec.summary_metrics.get("parameters", count_parameters(model))),
                        "target": int(target[i].detach().cpu().item()),
                        "pred": int(pred[i].detach().cpu().item()),
                        "correct": int(correct[i].detach().cpu().item()),
                        "confidence": float(top_values[i, 0].detach().cpu().item()),
                        "margin": float((top_values[i, 0] - top_values[i, 1]).detach().cpu().item()),
                        "rule_id": int(batch["rule_id"][i].detach().cpu().item()),
                        "rule_family": family_id,
                        "rule_family_name": family_name,
                        "seen_rule": int(batch["seen_context"][i].detach().cpu().item()),
                        "query_in_support": int(batch["query_in_support"][i].detach().cpu().item()),
                        "support_k": int(batch["support_k"][i].detach().cpu().item()),
                        "order_radius_mean": float(radii[i].mean().detach().cpu().item()),
                        "lop_feature_norm": float(lop_norm[i].detach().cpu().item()),
                    }
                    for harmonic_idx in range(model.lop.harmonics):
                        row[f"order_r{harmonic_idx + 1}"] = float(radii[i, harmonic_idx].detach().cpu().item())
                    for key, value in cluster_metrics.items():
                        row[key] = float(value[i].detach().cpu().item())
                    sample_rows.append(row)
                    vectors.append(lop_features[i].detach().cpu().numpy().astype(np.float32))

        run_rows.append(
            {
                "group": spec.group,
                "seed": spec.seed,
                "run_dir": str(spec.run_dir),
                "parameters": count_parameters(model),
                "diagnostic_samples": run_total,
                "diagnostic_accuracy": run_correct / max(run_total, 1),
                "summary_test_accuracy": spec.summary_metrics.get("test_accuracy", ""),
                "summary_test_seen_accuracy": spec.summary_metrics.get("test_seen_accuracy", ""),
                "summary_test_unseen_accuracy": spec.summary_metrics.get("test_unseen_accuracy", ""),
                "summary_test_out_support_accuracy": spec.summary_metrics.get("test_out_support_accuracy", ""),
            }
        )
        del model

    group_summary_rows: list[dict[str, Any]] = []
    for keys in [
        ["group"],
        ["group", "correct"],
        ["group", "seen_rule"],
        ["group", "query_in_support"],
        ["group", "rule_family_name"],
        ["group", "rule_family_name", "seen_rule"],
    ]:
        for row in summarize_group(sample_rows, keys):
            row["summary_level"] = "+".join(keys)
            group_summary_rows.append(row)

    contrast_rows: list[dict[str, Any]] = []
    add_binary_contrasts(sample_rows, contrast_rows, "group", "correct", 1, 0, "correct_vs_wrong")
    add_binary_contrasts(sample_rows, contrast_rows, "group", "seen_rule", 0, 1, "unseen_vs_seen")
    add_binary_contrasts(sample_rows, contrast_rows, "group", "query_in_support", 0, 1, "out_support_vs_in_support")
    add_variant_contrast(sample_rows, contrast_rows, "m32", "m64")

    family_rows, family_pair_rows = feature_structure(sample_rows, vectors)

    write_csv(run_dir / "diagnostics.csv", sample_rows)
    write_csv(run_dir / "run_summary.csv", run_rows)
    write_csv(run_dir / "group_summary.csv", group_summary_rows)
    write_csv(run_dir / "contrast_summary.csv", contrast_rows)
    write_csv(run_dir / "family_feature_summary.csv", family_rows)
    write_csv(run_dir / "family_pairwise_separation.csv", family_pair_rows)
    write_json(
        run_dir / "config.json",
        {
            "device": str(device),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "max_runs_per_group": args.max_runs_per_group,
            "model": args.model,
            "query_out_of_support_prob_override": args.query_out_of_support_prob,
            "groups": groups,
            "num_runs": len(runs),
            "num_samples": len(sample_rows),
        },
    )

    print(f"Wrote diagnostics to {run_dir / 'diagnostics.csv'}")
    print(f"Wrote run summary to {run_dir / 'run_summary.csv'}")
    print(f"Wrote group summary to {run_dir / 'group_summary.csv'}")
    print(f"Wrote contrasts to {run_dir / 'contrast_summary.csv'}")
    print(f"Wrote family separation to {run_dir / 'family_pairwise_separation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
