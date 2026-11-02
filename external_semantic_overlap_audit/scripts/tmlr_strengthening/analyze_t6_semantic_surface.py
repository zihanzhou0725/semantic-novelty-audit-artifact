#!/usr/bin/env python3
"""Analyze the frozen semantic-versus-surface CLUTRR intervention."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import t  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v3.training_common import load_npz  # noqa: E402
from toolkit.tmlr_strengthening.artifacts import (  # noqa: E402
    run_info,
    write_csv,
    write_json,
    write_macros,
    write_sha256sums,
)
from toolkit.paper_style import PALETTE, apply, figsize  # noqa: E402


ARMS = ("C0", "C1", "C2")
SEEDS = tuple(range(4))
BOOTSTRAP_SEED = 0
BOOTSTRAP_REPLICATES = 2000
RUN_ROOT = (
    ROOT
    / "runs"
    / "tmlr-strengthening-v1"
    / "t6_semantic_vs_surface"
)
if not RUN_ROOT.exists():
    RUN_ROOT = ROOT / "artifacts" / "tmlr_t6_run_outputs"
INPUT_DIR = (
    ROOT
    / "manifests"
    / "tmlr-strengthening-v1"
    / "t6_semantic_vs_surface"
    / "execution-inputs"
)
PROTOCOL_PATH = (
    ROOT
    / "protocols"
    / "tmlr-strengthening-v1"
    / "t6_semantic_vs_surface"
    / "protocol.json"
)
OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "tmlr-strengthening-v1"
    / "t6_semantic_vs_surface"
)


def _cluster_interval(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_REPLICATES, len(values)),
    )
    means = values[draws].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def _t_interval(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    half_width = float(
        t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return mean - half_width, mean + half_width


def _run_dir(arm: str, seed: int) -> Path:
    return RUN_ROOT / f"arm{arm}_seed{seed}"


def main() -> None:
    apply()
    input_paths = [PROTOCOL_PATH, INPUT_DIR / "decoy_allocation.csv"]
    run_scores = []
    pattern_scores: dict[tuple[str, int], np.ndarray] = {}
    pattern_semantic_ids: dict[int, str] = {}
    for arm in ARMS:
        for seed in SEEDS:
            directory = _run_dir(arm, seed)
            metrics_path = directory / "metrics.json"
            predictions_path = directory / "eval_predictions.npz"
            if not metrics_path.exists() or not predictions_path.exists():
                raise FileNotFoundError(f"Incomplete T6 run: {directory}")
            metrics = json.loads(metrics_path.read_text(encoding="ascii"))
            predictions = load_npz(predictions_path)
            correct = (
                predictions["predictions"].astype(np.int64)
                == predictions["labels"].astype(np.int64)
            )
            pattern_indices = predictions["pattern_indices"].astype(int)
            per_pattern = np.asarray(
                [
                    correct[pattern_indices == index].mean()
                    for index in range(40)
                ],
                dtype=float,
            )
            pattern_scores[(arm, seed)] = per_pattern
            for index in range(40):
                semantic_values = np.unique(
                    predictions["semantic_ids"][
                        pattern_indices == index
                    ]
                )
                if len(semantic_values) != 1:
                    raise ValueError("Pattern semantic IDs are not unique.")
                semantic_id = semantic_values[0].decode("ascii")
                previous = pattern_semantic_ids.setdefault(index, semantic_id)
                if previous != semantic_id:
                    raise ValueError("Pattern order differs across runs.")
            run_scores.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "accuracy": float(correct.mean()),
                    "cross_entropy": metrics["cross_entropy"],
                    "seconds": metrics["seconds"],
                    "device": metrics["device"],
                }
            )
            input_paths.extend([metrics_path, predictions_path])

    with (INPUT_DIR / "decoy_allocation.csv").open(
        newline="",
        encoding="ascii",
    ) as handle:
        allocation = list(csv.DictReader(handle))
    allocation_by_pattern = {
        int(row["pattern_index"]): row for row in allocation
    }
    write_csv(OUTPUT_DIR / "decoy_allocation.csv", allocation)

    pattern_arm_rows = []
    means_by_arm: dict[str, np.ndarray] = {}
    for arm in ARMS:
        matrix = np.stack(
            [pattern_scores[(arm, seed)] for seed in SEEDS],
            axis=0,
        )
        means_by_arm[arm] = matrix.mean(axis=0)
        for pattern_index in range(40):
            pattern_arm_rows.append(
                {
                    "pattern_index": pattern_index,
                    "semantic_id": pattern_semantic_ids[pattern_index],
                    "tier": allocation_by_pattern[pattern_index]["tier"],
                    "decoy_count": allocation_by_pattern[pattern_index][
                        "decoy_count"
                    ],
                    "arm": arm,
                    "mean_accuracy": float(
                        means_by_arm[arm][pattern_index]
                    ),
                    **{
                        f"seed_{seed}_accuracy": float(
                            matrix[seed, pattern_index]
                        )
                        for seed in SEEDS
                    },
                }
            )
    write_csv(OUTPUT_DIR / "run_scores.csv", run_scores)
    write_csv(OUTPUT_DIR / "pattern_arm_table.csv", pattern_arm_rows)

    run_accuracy = {
        (str(row["arm"]), int(row["seed"])): float(row["accuracy"])
        for row in run_scores
    }
    contrast_definitions = {
        "C1-C0": ("C1", "C0"),
        "C2-C0": ("C2", "C0"),
        "C1-C2": ("C1", "C2"),
    }
    contrasts = {}
    for name, (left, right) in contrast_definitions.items():
        pattern_differences = (
            means_by_arm[left] - means_by_arm[right]
        )
        seed_differences = np.asarray(
            [
                run_accuracy[(left, seed)]
                - run_accuracy[(right, seed)]
                for seed in SEEDS
            ]
        )
        cluster_interval = _cluster_interval(pattern_differences)
        seed_interval = _t_interval(seed_differences)
        contrasts[name] = {
            "effect_pp": 100.0 * float(pattern_differences.mean()),
            "pattern_cluster_ci_pp": [
                100.0 * cluster_interval[0],
                100.0 * cluster_interval[1],
            ],
            "seed_effect_pp": 100.0 * float(seed_differences.mean()),
            "seed_t_ci_pp": [
                100.0 * seed_interval[0],
                100.0 * seed_interval[1],
            ],
            "seed_effects_pp": (100.0 * seed_differences).tolist(),
        }

    semantic = contrasts["C1-C0"]["pattern_cluster_ci_pp"]
    decoy = contrasts["C2-C0"]["pattern_cluster_ci_pp"]
    semantic_minus_decoy = contrasts["C1-C2"]["pattern_cluster_ci_pp"]
    if semantic[0] > 0.0 and (
        decoy[0] <= 0.0 <= decoy[1]
        or semantic_minus_decoy[0] > 0.0
    ):
        interpretation = (
            "semantic identity, not surface similarity, drives the gain"
        )
    elif decoy[0] > 0.0:
        interpretation = (
            "surface-similar decoys also help; the gain is partly "
            "lexical/compositional"
        )
    else:
        interpretation = (
            "the prespecified directional conditions are not met"
        )

    tier_groups: dict[str, list[int]] = defaultdict(list)
    for row in allocation:
        tier_groups[str(row["tier"])].append(int(row["pattern_index"]))
    tier_results = {}
    decoy_difference = means_by_arm["C2"] - means_by_arm["C0"]
    for tier, indices in sorted(tier_groups.items()):
        values = decoy_difference[np.asarray(indices)]
        interval = _cluster_interval(values)
        tier_results[tier] = {
            "pattern_count": len(indices),
            "effect_pp": 100.0 * float(values.mean()),
            "pattern_cluster_ci_pp": [
                100.0 * interval[0],
                100.0 * interval[1],
            ],
        }

    arm_summary = {}
    for arm in ARMS:
        values = means_by_arm[arm]
        interval = _cluster_interval(values)
        arm_summary[arm] = {
            "accuracy": float(values.mean()),
            "pattern_cluster_ci": list(interval),
        }
    timing_path = RUN_ROOT / "armC0_seed0_timing" / "metrics.json"
    if not timing_path.exists():
        timing_path = RUN_ROOT / "timing" / "metrics.json"
    timing = (
        json.loads(timing_path.read_text(encoding="ascii"))
        if timing_path.exists()
        else None
    )
    if timing_path.exists():
        input_paths.append(timing_path)
    summary = {
        "arm_summary": arm_summary,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "contrasts": contrasts,
        "interpretation": interpretation,
        "protocol_sha256": __import__("hashlib").sha256(
            PROTOCOL_PATH.read_bytes()
        ).hexdigest(),
        "tier_results": tier_results,
        "timing_run": timing,
    }
    write_json(OUTPUT_DIR / "summary.json", summary)

    positions = np.arange(len(ARMS))
    arm_means = np.asarray(
        [arm_summary[arm]["accuracy"] for arm in ARMS]
    )
    intervals = np.asarray(
        [arm_summary[arm]["pattern_cluster_ci"] for arm in ARMS]
    )
    errors = np.vstack(
        [arm_means - intervals[:, 0], intervals[:, 1] - arm_means]
    )
    figure, axis = plt.subplots(figsize=figsize("panel_tall"))
    axis.bar(
        positions,
        arm_means,
        color=[PALETTE["mid"], PALETTE["blue"], PALETTE["orange"]],
        width=0.58,
        alpha=0.9,
    )
    axis.errorbar(
        positions,
        arm_means,
        yerr=errors,
        fmt="none",
        color=PALETTE["charcoal"],
        capsize=3,
        linewidth=1.0,
    )
    axis.set(
        xticks=positions,
        xticklabels=["No exposure", "Semantic", "Surface decoy"],
        ylabel="Exact-match accuracy",
        ylim=(0.0, 1.0),
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7)
    axis.yaxis.label.set_size(8)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "fig_semantic_vs_surface.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    write_macros(
        OUTPUT_DIR / "macros.tex",
        {
            "TSixCZeroAcc": (
                f"{100.0 * arm_summary['C0']['accuracy']:.1f}"
            ),
            "TSixCOneAcc": (
                f"{100.0 * arm_summary['C1']['accuracy']:.1f}"
            ),
            "TSixCTwoAcc": (
                f"{100.0 * arm_summary['C2']['accuracy']:.1f}"
            ),
            "TSixSemGain": f"{contrasts['C1-C0']['effect_pp']:.1f}",
            "TSixDecoyGain": f"{contrasts['C2-C0']['effect_pp']:.1f}",
            "TSixSemDecoyGain": (
                f"{contrasts['C1-C2']['effect_pp']:.1f}"
            ),
            "TSixSemLo": (
                f"{contrasts['C1-C0']['pattern_cluster_ci_pp'][0]:.1f}"
            ),
            "TSixSemHi": (
                f"{contrasts['C1-C0']['pattern_cluster_ci_pp'][1]:.1f}"
            ),
            "TSixDecoyLo": (
                f"{contrasts['C2-C0']['pattern_cluster_ci_pp'][0]:.1f}"
            ),
            "TSixDecoyHi": (
                f"{contrasts['C2-C0']['pattern_cluster_ci_pp'][1]:.1f}"
            ),
            "TSixSemDecoyLo": (
                f"{contrasts['C1-C2']['pattern_cluster_ci_pp'][0]:.1f}"
            ),
            "TSixSemDecoyHi": (
                f"{contrasts['C1-C2']['pattern_cluster_ci_pp'][1]:.1f}"
            ),
            "TSixSemSeedLo": (
                f"{contrasts['C1-C0']['seed_t_ci_pp'][0]:.1f}"
            ),
            "TSixSemSeedHi": (
                f"{contrasts['C1-C0']['seed_t_ci_pp'][1]:.1f}"
            ),
            "TSixDecoySeedLo": (
                f"{0.0 if abs(contrasts['C2-C0']['seed_t_ci_pp'][0]) < 0.05 else contrasts['C2-C0']['seed_t_ci_pp'][0]:.1f}"
            ),
            "TSixDecoySeedHi": (
                f"{contrasts['C2-C0']['seed_t_ci_pp'][1]:.1f}"
            ),
            "TSixSemDecoySeedLo": (
                f"{contrasts['C1-C2']['seed_t_ci_pp'][0]:.1f}"
            ),
            "TSixSemDecoySeedHi": (
                f"{contrasts['C1-C2']['seed_t_ci_pp'][1]:.1f}"
            ),
            "TSixProtocolHash": summary["protocol_sha256"][:12],
            "TSixTimingSeconds": (
                f"{float(timing['seconds']):.2f}" if timing else "NA"
            ),
            "TSixPatterns": "40",
        },
    )
    write_json(
        OUTPUT_DIR / "run_info.json",
        run_info(
            seed=list(SEEDS),
            script_path=(
                "scripts/tmlr_strengthening/"
                "analyze_t6_semantic_surface.py"
            ),
            input_paths=input_paths,
            extra={
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            },
        ),
    )
    write_sha256sums(OUTPUT_DIR)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
