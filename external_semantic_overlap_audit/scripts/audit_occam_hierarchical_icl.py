#!/usr/bin/env python3
"""Audit the frozen In-Context Occam's Razor P3 candidate."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "semantic-novelty-audit-occam-hierarchical-icl-v1"
P3_STATUS_SCHEMA_VERSION = "quality-enhancement-p3-breadth-status-v1"
CANDIDATE_ID = "7b9ebd45a3528701"
CODE_REVISION = "3b8fe0d69dd9bb56c39d933525bf0234feefcc39"
PAPER_SHA256 = "8c92e590ba749a947c22b13705a751b832495f21ae0d9ac03a6ec829872f945c"
ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ROOT / "external_semantic_overlap_audit"
PROTOCOL_DIR = EXTERNAL_ROOT / "protocols" / "quality-enhancement-v1"
P3_PROTOCOL_DIR = PROTOCOL_DIR / "p3"
CANDIDATE_MANIFEST = P3_PROTOCOL_DIR / "candidate_manifest.json"
CODE_ROOT = EXTERNAL_ROOT / "upstream" / "p3_icl_bayes_occam"
PAPER_PATH = EXTERNAL_ROOT / "upstream" / "p3_occam_arxiv_2506.19351v1.pdf"
OUTPUT_DIR = (
    EXTERNAL_ROOT
    / "analysis"
    / "quality-enhancement-v1"
    / "p3"
    / "occam_hierarchical_icl"
)
P3_ANALYSIS_DIR = OUTPUT_DIR.parent
P3_STATUS_PATH = P3_ANALYSIS_DIR / "p3_breadth_status.json"
P3_CHECKSUM_PATH = P3_ANALYSIS_DIR / "SHA256SUMS"
TENSOR_AUDIT_PATH = (
    P3_ANALYSIS_DIR / "tensor_logic_fb15k237" / "audit_report.json"
)

VOCAB_SIZE = 3
MAX_TRANSITIONS = 3
EVAL_TASK_SEEDS = tuple(12574 * index + 5 for index in range(10))
EVAL_SEQUENCES_PER_TASK = 100
PAPER_BATCH_SIZE = 32
PAPER_SEQUENCES_SEEN = 800_000
PAPER_STEPS = PAPER_SEQUENCES_SEEN // PAPER_BATCH_SIZE
MARKOV_EXPERIMENTS = (
    {
        "experiment": "figure_2_order_1_3",
        "figure": 2,
        "orders": (1, 3),
        "sequence_length": 300,
        "layers": 6,
        "heads": 6,
        "embedding_dimension": 192,
    },
    {
        "experiment": "figure_3_order_1_2",
        "figure": 3,
        "orders": (1, 2),
        "sequence_length": 200,
        "layers": 2,
        "heads": 20,
        "embedding_dimension": 320,
    },
)


def fail(message: str) -> None:
    raise SystemExit(f"Occam P3 audit failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(CODE_ROOT), *args],
        text=True,
    ).strip()


def validate_frozen_inputs() -> dict[str, Any]:
    for line in (P3_PROTOCOL_DIR / "SHA256SUMS").read_text(
        encoding="ascii"
    ).splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        path = P3_PROTOCOL_DIR / filename
        require(path.is_file(), f"missing frozen P3 file: {filename}")
        require(sha256_file(path) == digest, f"frozen P3 file changed: {filename}")

    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="ascii"))
    require(
        manifest["status"]
        == "frozen_before_candidate_dataset_download_or_semantic_extraction",
        "candidate manifest status changed",
    )
    candidates = [
        item
        for item in manifest["candidates"]
        if item["candidate_id"] == CANDIDATE_ID
    ]
    require(len(candidates) == 1, "Occam candidate is not uniquely frozen")
    candidate = candidates[0]
    require(candidate["audit_order"] == 2, "Occam audit order changed")
    require(
        candidate["code_revision"]["revision"] == CODE_REVISION,
        "Occam code revision changed",
    )
    require(
        candidate["paper_revision"] == "arXiv:2506.19351v1; COLM 2025",
        "Occam paper revision changed",
    )

    require(CODE_ROOT.is_dir(), "pinned code checkout is missing")
    require(git_output("rev-parse", "HEAD") == CODE_REVISION, "wrong code revision")
    require(
        subprocess.run(
            ["git", "-C", str(CODE_ROOT), "diff", "--quiet", "HEAD", "--"],
            check=False,
        ).returncode
        == 0,
        "tracked code checkout is dirty",
    )
    require(PAPER_PATH.is_file(), "paper PDF is missing")
    require(sha256_file(PAPER_PATH) == PAPER_SHA256, "paper PDF hash changed")
    return candidate


def source_inventory_and_invariants() -> dict[str, Any]:
    tracked_files = git_output("ls-tree", "-r", "--name-only", "HEAD").splitlines()
    require(
        tracked_files
        == [
            "README.md",
            "eval.py",
            "gpt.py",
            "kl_div_plot.py",
            "kl_div_plot_new.py",
            "main.py",
            "minigpt_utils.py",
            "model.py",
            "model_utils.py",
            "train_gpt.py",
            "utils.py",
            "utils_data.py",
        ],
        "pinned repository inventory changed",
    )

    main_text = (CODE_ROOT / "main.py").read_text(encoding="utf-8")
    train_text = (CODE_ROOT / "train_gpt.py").read_text(encoding="utf-8")
    data_text = (CODE_ROOT / "utils_data.py").read_text(encoding="utf-8")
    plot_text = (CODE_ROOT / "kl_div_plot_new.py").read_text(encoding="utf-8")
    required_fragments = {
        "main.py": (
            "parser.add_argument('--steps', type=int, default=100",
            "parser.add_argument('--bs', type=int, default=16",
            "torch.manual_seed(model_seed)",
        ),
        "train_gpt.py": (
            "num_tm_eval = 10",
            "idx = np.random.choice(len(order_list), batch_size)",
            "seed = base_seed + step + i",
            "scheduler = torch.optim.lr_scheduler.MultiStepLR",
        ),
        "utils_data.py": (
            "self.rng = np.random.default_rng(self.seed)",
            "sequence = list(np.random.choice(self.states, size=self.order))",
            (
                "next_state = np.random.choice("
                "self.states, p=self.transition_matrix[state_index])"
            ),
            "seed = 12574*j + 5",
            "num_sequences=100",
        ),
        "kl_div_plot_new.py": (
            "sequences_seen1 = (steps1 * batch_size) / 1000",
            "sequences_seen2 = (steps2 * batch_size) / 1000",
        ),
    }
    texts = {
        "main.py": main_text,
        "train_gpt.py": train_text,
        "utils_data.py": data_text,
        "kl_div_plot_new.py": plot_text,
    }
    for filename, fragments in required_fragments.items():
        for fragment in fragments:
            require(
                fragment in texts[filename],
                f"source invariant missing from {filename}: {fragment}",
            )

    active_np_seed_calls: list[str] = []
    for filename in ("main.py", "train_gpt.py", "utils_data.py"):
        tree = ast.parse(texts[filename], filename=filename)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr != "seed":
                continue
            owner = function.value
            if (
                isinstance(owner, ast.Attribute)
                and owner.attr == "random"
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "np"
            ):
                active_np_seed_calls.append(f"{filename}:{node.lineno}")
    require(not active_np_seed_calls, "unexpected active NumPy seed call")

    tracked_names = set(tracked_files)
    missing_local_modules = {
        "train.py": "train.py" not in tracked_names,
        "linear_probe.py": "linear_probe.py" not in tracked_names,
    }
    require(all(missing_local_modules.values()), "missing eval modules appeared")
    linear_implementation_files = [
        filename
        for filename in tracked_files
        if "regress" in filename.lower() or "linear" in filename.lower()
    ]
    require(
        not linear_implementation_files,
        "linear-regression implementation file appeared",
    )

    eval_text = (CODE_ROOT / "eval.py").read_text(encoding="utf-8")
    structural_eval_failures = {
        "missing_train_module": "from train import" in eval_text
        and "train.py" not in tracked_names,
        "missing_linear_probe_module": "from linear_probe import" in eval_text
        and "linear_probe.py" not in tracked_names,
        "missing_train_gpt_exports": all(
            name in eval_text and f"def {name}" not in train_text
            for name in ("train_gpt_model", "train_transformer_gpt_hf")
        ),
        "markov_classes_imported_from_wrong_module": (
            "from utils import MarkovChainGenerator" in eval_text
            and "class MarkovChainGenerator" not in (CODE_ROOT / "utils.py").read_text(
                encoding="utf-8"
            )
        ),
    }
    require(
        all(structural_eval_failures.values()),
        "expected structural eval failure changed",
    )
    return {
        "active_numpy_seed_calls": active_np_seed_calls,
        "linear_implementation_files": linear_implementation_files,
        "missing_local_modules": missing_local_modules,
        "structural_eval_failures": structural_eval_failures,
        "tracked_files": tracked_files,
    }


def transition_matrix(order: int, seed: int) -> np.ndarray:
    """Byte-for-byte numerical reproduction of the published generator."""

    rng = np.random.Generator(np.random.PCG64(seed))
    matrix = np.zeros((VOCAB_SIZE**order, VOCAB_SIZE), dtype=np.float64)
    for state in range(VOCAB_SIZE**order):
        transition_indices = rng.choice(
            VOCAB_SIZE,
            MAX_TRANSITIONS,
            replace=False,
        )
        probabilities = rng.dirichlet(np.ones(MAX_TRANSITIONS))
        matrix[state, transition_indices] = probabilities
    return matrix


def task_signature(order: int, seed: int) -> str:
    matrix = transition_matrix(order, seed)
    payload = {
        "dtype": "float64",
        "max_transitions": MAX_TRANSITIONS,
        "order": order,
        "rows_float_hex": [
            [float(value).hex() for value in row]
            for row in matrix.tolist()
        ],
        "shape": list(matrix.shape),
        "vocab_size": VOCAB_SIZE,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def seed_occurrences(seed: int, *, steps: int, batch_size: int) -> int:
    first_step = max(0, seed - (batch_size - 1))
    last_step = min(steps - 1, seed)
    return max(0, last_step - first_step + 1)


def training_task_census(*, steps: int, batch_size: int) -> dict[str, Any]:
    counts = [
        seed_occurrences(
            seed,
            steps=steps,
            batch_size=batch_size,
        )
        for seed in range(steps + batch_size - 1)
    ]
    require(sum(counts) == steps * batch_size, "training occurrence census")
    distinct_seed_ids = len(counts)
    max_distinct_order_seed_tasks = sum(min(count, 2) for count in counts)
    min_duplicate_episodes = steps * batch_size - max_distinct_order_seed_tasks
    expected_distinct_order_seed_tasks = sum(
        2.0 * (1.0 - 0.5**count) for count in counts
    )
    return {
        "batch_size": batch_size,
        "distinct_seed_ids": distinct_seed_ids,
        "episode_count": steps * batch_size,
        "expected_distinct_order_seed_tasks_under_iid_uniform_orders": (
            expected_distinct_order_seed_tasks
        ),
        "maximum_distinct_order_seed_tasks": max_distinct_order_seed_tasks,
        "minimum_distinct_order_seed_tasks": distinct_seed_ids,
        "minimum_duplicate_episode_count": min_duplicate_episodes,
        "minimum_duplicate_episode_rate": min_duplicate_episodes
        / (steps * batch_size),
        "occurrence_count_distribution": {
            str(count): frequency
            for count, frequency in sorted(Counter(counts).items())
        },
        "steps": steps,
    }


def markov_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in MARKOV_EXPERIMENTS:
        for task_index, seed in enumerate(EVAL_TASK_SEEDS):
            occurrences = seed_occurrences(
                seed,
                steps=PAPER_STEPS,
                batch_size=PAPER_BATCH_SIZE,
            )
            for order in experiment["orders"]:
                if occurrences:
                    exposure_status = (
                        "possible_exact_same_order_exposure_identity_unresolved"
                    )
                    exposure_probability = 1.0 - 0.5**occurrences
                else:
                    exposure_status = "outside_nominal_training_seed_support"
                    exposure_probability = 0.0
                rows.append(
                    {
                        "experiment": experiment["experiment"],
                        "figure": experiment["figure"],
                        "order": order,
                        "eval_task_index": task_index,
                        "eval_seed": seed,
                        "eval_task_signature_sha256": task_signature(order, seed),
                        "eval_sequences": EVAL_SEQUENCES_PER_TASK,
                        "nominal_train_seed_occurrences_all_orders": occurrences,
                        "nominal_train_seed_support": occurrences > 0,
                        "same_order_exposure_status": exposure_status,
                        "same_order_exposure_probability_under_iid_uniform_orders": (
                            f"{exposure_probability:.12f}"
                        ),
                    }
                )
    return rows


def summarize_markov_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_experiment: dict[str, Any] = {}
    for experiment in MARKOV_EXPERIMENTS:
        experiment_rows = [
            row
            for row in rows
            if row["experiment"] == experiment["experiment"]
        ]
        supported_seeds = sorted(
            {
                int(row["eval_seed"])
                for row in experiment_rows
                if row["nominal_train_seed_support"]
            }
        )
        supported_rows = [
            row for row in experiment_rows if row["nominal_train_seed_support"]
        ]
        guaranteed_collisions = len(supported_seeds)
        possible_collisions = len(supported_rows)
        task_count = len(experiment_rows)
        by_experiment[experiment["experiment"]] = {
            "eval_latent_task_count": task_count,
            "eval_sequence_count": sum(
                int(row["eval_sequences"]) for row in experiment_rows
            ),
            "exact_train_eval_collision_count_lower_bound": (
                guaranteed_collisions
            ),
            "exact_train_eval_collision_count_upper_bound": possible_collisions,
            "exact_train_eval_collision_rate_lower_bound": (
                guaranteed_collisions / task_count
            ),
            "exact_train_eval_collision_rate_upper_bound": (
                possible_collisions / task_count
            ),
            "eval_sequences_from_exposed_task_parameters_lower_bound": (
                guaranteed_collisions * EVAL_SEQUENCES_PER_TASK
            ),
            "eval_sequences_from_exposed_task_parameters_upper_bound": (
                possible_collisions * EVAL_SEQUENCES_PER_TASK
            ),
            "supported_eval_seeds": supported_seeds,
            "unresolved_identity_reason": (
                "np.random controls order assignment but is never seeded or "
                "persisted; for every supported seed, pigeonhole guarantees "
                "at least one of the two evaluated orders occurred in training"
            ),
        }
    unique_signatures = {
        str(row["eval_task_signature_sha256"]) for row in rows
    }
    return {
        "all_rows": len(rows),
        "by_experiment": by_experiment,
        "cross_experiment_duplicate_rows": len(rows) - len(unique_signatures),
        "unique_eval_latent_task_signatures": len(unique_signatures),
    }


def artifact_census(tracked_files: list[str]) -> dict[str, bool]:
    lowered = [name.lower() for name in tracked_files]
    return {
        "checkpoint": any(
            name.endswith((".pt", ".pth", ".ckpt", ".safetensors"))
            for name in lowered
        ),
        "environment_or_requirements": any(
            Path(name).name
            in {
                "environment.yml",
                "pyproject.toml",
                "requirements.txt",
                "setup.py",
            }
            for name in lowered
        ),
        "machine_readable_results": any(
            name.endswith((".csv", ".json", ".jsonl", ".npz"))
            for name in lowered
        ),
        "per_item_predictions": any(
            "prediction" in name or "output" in name for name in lowered
        ),
        "run_configuration": any(
            "config" in name or "command" in name for name in lowered
        ),
        "training_log": any("log" in Path(name).name for name in lowered),
    }


def build_report(
    candidate: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    rows = markov_rows()
    markov_summary = summarize_markov_rows(rows)
    train_census = training_task_census(
        steps=PAPER_STEPS,
        batch_size=PAPER_BATCH_SIZE,
    )
    artifacts = artifact_census(inventory["tracked_files"])
    require(not any(artifacts.values()), "unexpected reproducibility artifact found")

    experiments: list[dict[str, Any]] = []
    for experiment in MARKOV_EXPERIMENTS:
        experiments.append(
            {
                **experiment,
                "batch_size": PAPER_BATCH_SIZE,
                "nominal_sequences_seen": PAPER_SEQUENCES_SEEN,
                "nominal_steps": PAPER_STEPS,
                "relative_position_encoding": True,
                "source": (
                    "paper Figure x-axis and Appendix C; total steps are not "
                    "published as a command or machine-readable run config"
                ),
            }
        )

    return {
        "audit_date": "2026-10-09",
        "behavioral_reproducibility": {
            "artifact_census": artifacts,
            "checkpoint_revision": candidate["checkpoint_revision_if_any"],
            "decision": (
                "no_go_missing_checkpoints_predictions_logs_run_configs_and_"
                "executable_evaluation"
            ),
            "evaluation_entrypoint_structural_failures": inventory[
                "structural_eval_failures"
            ],
            "paper_results_status": (
                "aggregate_plots_reported_not_recomputed_or_per_item_verified"
            ),
        },
        "candidate_id": candidate["candidate_id"],
        "claim_gate": {
            "audit_completeness": (
                "incomplete_frozen_scope_linear_regression_generator_absent"
            ),
            "candidate_contribution_to_p3": {
                "complete_public_semantic_audit": False,
                "credible_behavioral_evidence": False,
            },
            "decision": (
                "protocol_defect_detected_candidate_ineligible_for_p3_breadth"
            ),
            "eligibility_at_audit": False,
            "eligibility_failure": (
                "training_and_evaluation_generator_is_inspectable failed for "
                "the required linear-regression scope"
            ),
            "markov_unseen_task_claim": (
                "failed_with_sound_10_to_20_percent_exact_collision_bound_per_"
                "main_experiment"
            ),
        },
        "claim_under_audit": candidate["audited_claim"],
        "code_revision": CODE_REVISION,
        "input_sha256": {
            relative(CANDIDATE_MANIFEST): sha256_file(CANDIDATE_MANIFEST),
            relative(PAPER_PATH): sha256_file(PAPER_PATH),
            relative(Path(__file__).resolve()): sha256_file(
                Path(__file__).resolve()
            ),
            **{
                relative(CODE_ROOT / filename): sha256_file(CODE_ROOT / filename)
                for filename in (
                    "README.md",
                    "eval.py",
                    "main.py",
                    "train_gpt.py",
                    "utils.py",
                    "utils_data.py",
                )
            },
        },
        "linear_regression_audit": {
            "paper_scope": {
                "dimensions": [10, 20],
                "embedding_dimension": 256,
                "evaluation_contexts_per_reported_aggregate": 2000,
                "heads": 8,
                "layers": 12,
                "sequence_length": 39,
                "simple_task": "w=[w_(d/2),0]",
                "bootstrap_trials": 100,
            },
            "published_code_status": (
                "unavailable_no_generator_training_evaluation_or_curriculum_code"
            ),
            "semantic_numerator": 0,
            "semantic_numerator_interpretation": (
                "zero verified collisions is not evidence of novelty"
            ),
            "unresolved_denominator": (
                "all published linear-regression evaluation contexts"
            ),
            "unresolved_reason": (
                "the pinned repository has no linear-regression implementation, "
                "task seeds, coefficient vectors, manifests, checkpoints, or "
                "per-context predictions"
            ),
        },
        "markov_audit": {
            "eval_protocol": {
                "eval_task_seed_formula": "12574*j+5 for j=0..9",
                "eval_task_seeds": list(EVAL_TASK_SEEDS),
                "sequences_per_eval_task": EVAL_SEQUENCES_PER_TASK,
                "transition_parameter_rng": (
                    "NumPy PCG64 initialized from each task seed"
                ),
                "sequence_rng": (
                    "global np.random state; no active np.random.seed call"
                ),
            },
            "experiments": experiments,
            "fresh_task_claim_audit": {
                "decision": "failed",
                "reason": (
                    "training seed is step+i, so adjacent batches reuse the "
                    "same transition matrix whenever the same order is selected"
                ),
                "training_task_census_per_main_experiment": train_census,
            },
            "semantic_signature": (
                "SHA256 of canonical JSON containing order, vocabulary size, "
                "max transitions, shape, and exact float.hex transition tensor"
            ),
            "summary": markov_summary,
            "unresolved_denominator": {
                "actual_same_order_collision_identity": (
                    "all four supported order-specific rows per experiment"
                ),
                "reason": (
                    "order assignment uses unseeded NumPy global RNG and no "
                    "training task manifest or log was published"
                ),
                "sound_bound": (
                    "2 to 4 of 20 eval latent tasks per main experiment were "
                    "exactly exposed during training"
                ),
            },
        },
        "paper_revision": candidate["paper_revision"],
        "reproducibility_risks": [
            (
                "main.py defaults to 100 steps, batch 16, two layers, two heads, "
                "and dimension 16 rather than either paper main setting"
            ),
            (
                "the repository publishes no command, run configuration, "
                "dependency lock, checkpoint, training log, or predictions"
            ),
            (
                "global NumPy RNG controls task-category assignments and all "
                "sequence draws but is never seeded by the entrypoint"
            ),
            (
                "eval.py imports missing local modules and symbols, so the "
                "published evaluation entrypoint is structurally non-executable"
            ),
            (
                "the candidate manifest states that linear-regression episodes "
                "are generated by the pinned revision, but that implementation "
                "is absent"
            ),
        ],
        "repository_inventory": inventory,
        "schema_version": SCHEMA_VERSION,
        "status": "complete_with_frozen_scope_failure",
        "title": candidate["title"],
    }


def build_report_cn(report: dict[str, Any]) -> str:
    train = report["markov_audit"]["fresh_task_claim_audit"][
        "training_task_census_per_main_experiment"
    ]
    figure_2 = report["markov_audit"]["summary"]["by_experiment"][
        "figure_2_order_1_3"
    ]
    figure_3 = report["markov_audit"]["summary"]["by_experiment"][
        "figure_3_order_1_2"
    ]
    return "\n".join(
        [
            "# In-Context Occam's Razor 层级 ICL 语义审计",
            "",
            f"范围：冻结候选 `{CANDIDATE_ID}`，代码 revision `{CODE_REVISION}`，"
            "论文 `arXiv:2506.19351v1`。按冻结协议，Markov chain 与 linear "
            "regression 都是必审范围，PCFG/GPT-4 不能替代。",
            "",
            "## 结论",
            "",
            "- **Markov protocol defect detected**：论文称 inference chains 为 "
            '"unseen"，但公开代码让 evaluation 与 training 共享 latent-task seed '
            "空间。",
            "- **Fresh-task premise failed**：训练 seed 定义为 `step+i`，相邻 "
            "mini-batches 大量复用同一个 transition tensor。",
            "- **Frozen scope incomplete**：pinned repository 完全没有 linear-"
            "regression generator、训练、评估或 curriculum 实现。",
            "- 因此该候选不能计为 P3 的 complete public semantic audit，也没有 "
            "credible behavioral evidence。",
            "",
            "## Markov 精确碰撞界",
            "",
            "论文 Fig. 2 与 Fig. 3 的横轴均到 800k sequences；Appendix C 指定 "
            "batch size 32，对应 nominal 25,000 steps。公开评估代码为每个 order "
            "固定 10 个 transition matrices，每个 matrix 生成 100 条序列。",
            "",
            "- Fig. 2（orders 1/3）："
            f"{figure_2['exact_train_eval_collision_count_lower_bound']}"
            f"-{figure_2['exact_train_eval_collision_count_upper_bound']}/"
            f"{figure_2['eval_latent_task_count']} 个 eval latent tasks 与训练精确重合；"
            f"对应 {figure_2['eval_sequences_from_exposed_task_parameters_lower_bound']}"
            f"-{figure_2['eval_sequences_from_exposed_task_parameters_upper_bound']}/"
            f"{figure_2['eval_sequence_count']} 条 eval sequences。",
            "- Fig. 3（orders 1/2）："
            f"{figure_3['exact_train_eval_collision_count_lower_bound']}"
            f"-{figure_3['exact_train_eval_collision_count_upper_bound']}/"
            f"{figure_3['eval_latent_task_count']} 个 eval latent tasks 与训练精确重合；"
            f"对应 {figure_3['eval_sequences_from_exposed_task_parameters_lower_bound']}"
            f"-{figure_3['eval_sequences_from_exposed_task_parameters_upper_bound']}/"
            f"{figure_3['eval_sequence_count']} 条 eval sequences。",
            "- 碰撞 eval seeds 为 `5` 与 `12579`。具体落在哪个 order 无法恢复，"
            "因为 order assignment 使用未设 seed 的全局 `np.random`，且没有发布 "
            "task manifest；但每个 seed 在两个训练 order 中至少落入一个，所以"
            "上述 10% 下界是确定的。",
            "",
            "## 训练任务复用",
            "",
            f"- 每个主实验生成 {train['episode_count']:,} 个训练 episodes，只有 "
            f"{train['distinct_seed_ids']:,} 个不同 seed ids。",
            f"- 即使按最有利情况让每个 seed 同时覆盖两个 orders，也至多有 "
            f"{train['maximum_distinct_order_seed_tasks']:,} 个不同 latent tasks。",
            f"- 因而至少 {train['minimum_duplicate_episode_count']:,}/"
            f"{train['episode_count']:,}（"
            f"{100 * train['minimum_duplicate_episode_rate']:.4f}%）训练 episodes "
            "复用了已出现的 `(order, transition tensor)`；这与“fresh tasks at "
            'each optimization iteration”不一致。',
            "",
            "## Linear regression 未解析分母",
            "",
            "- 论文范围为 `d=10/20`、`T=39`、12 layers、8 heads、dim 256，"
            "每个报告聚合使用 2,000 个独立 regressors 与 100 次 bootstrap。",
            "- pinned repository 的 12 个 tracked files 中没有 regression "
            "generator、coefficient vectors、task seeds、run config、checkpoint 或"
            "逐 context prediction。",
            "- 所有 linear-regression evaluation contexts 均属于 unresolved "
            "denominator；`0` 个已验证碰撞不能解释为语义新颖。",
            "",
            "## 行为证据与 P3 Gate",
            "",
            "- `eval.py` 静态引用缺失的 `train.py`、`linear_probe.py` 及不存在的 "
            "exports；公开 evaluation entrypoint 在 pinned revision 上结构性不可执行。",
            "- 仓库没有 checkpoint、训练日志、机器可读结果或逐条 predictions，"
            "因此论文 aggregate plots 未达到 P2-strength reproducibility。",
            "- P3 当前为 `1/2` 个 complete audits、`0/1` 个 credible behavioral "
            "benchmarks，Breadth Gate = **No-Go**。两个冻结候选均保留报告，不存在"
            "结果依赖排除。",
            "",
        ]
    )


def build_p3_status(occam_report: dict[str, Any]) -> dict[str, Any]:
    tensor_report = json.loads(TENSOR_AUDIT_PATH.read_text(encoding="ascii"))
    tensor_contribution = tensor_report["claim_gate"][
        "candidate_contribution_to_p3"
    ]
    occam_contribution = occam_report["claim_gate"][
        "candidate_contribution_to_p3"
    ]
    candidates = [
        {
            "candidate_id": tensor_report["candidate_id"],
            "complete_public_semantic_audit": tensor_contribution[
                "complete_public_semantic_audit"
            ],
            "credible_behavioral_evidence": tensor_contribution[
                "credible_behavioral_evidence"
            ],
            "decision": tensor_report["claim_gate"]["decision"],
            "reported_without_result_dependent_exclusion": True,
        },
        {
            "candidate_id": occam_report["candidate_id"],
            "complete_public_semantic_audit": occam_contribution[
                "complete_public_semantic_audit"
            ],
            "credible_behavioral_evidence": occam_contribution[
                "credible_behavioral_evidence"
            ],
            "decision": occam_report["claim_gate"]["decision"],
            "reported_without_result_dependent_exclusion": True,
        },
    ]
    complete_count = sum(
        int(item["complete_public_semantic_audit"]) for item in candidates
    )
    behavioral_count = sum(
        int(item["credible_behavioral_evidence"]) for item in candidates
    )
    return {
        "audit_date": "2026-10-09",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "decision": "no_go",
        "gate": "p3_breadth_go",
        "input_sha256": {
            relative(CANDIDATE_MANIFEST): sha256_file(CANDIDATE_MANIFEST),
            relative(TENSOR_AUDIT_PATH): sha256_file(TENSOR_AUDIT_PATH),
            relative(OUTPUT_DIR / "audit_report.json"): sha256_file(
                OUTPUT_DIR / "audit_report.json"
            ),
        },
        "observed": {
            "complete_public_semantic_audits": complete_count,
            "credible_behavioral_evidence_benchmarks": behavioral_count,
            "result_dependent_candidate_exclusions": 0,
        },
        "reason": (
            "complete audit count is 1<2 and credible behavioral benchmark "
            "count is 0<1"
        ),
        "requirements": {
            "complete_public_semantic_audits_at_least": 2,
            "credible_behavioral_evidence_benchmarks_at_least": 1,
            "explicit_unresolved_denominator_required": True,
            "result_dependent_candidate_exclusion_allowed": False,
            "sound_numerator_required": True,
        },
        "schema_version": P3_STATUS_SCHEMA_VERSION,
        "status": "complete",
    }


def write_checksums(paths: list[Path], destination: Path, base: Path) -> None:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(base).as_posix()}"
        for path in paths
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    candidate = validate_frozen_inputs()
    inventory = source_inventory_and_invariants()
    rows = markov_rows()
    report = build_report(candidate, inventory)
    require(
        report["markov_audit"]["summary"]["all_rows"] == len(rows) == 40,
        "Markov manifest row count",
    )
    require(
        report["markov_audit"]["summary"]["unique_eval_latent_task_signatures"]
        == 30,
        "Markov unique signature count",
    )
    for summary in report["markov_audit"]["summary"][
        "by_experiment"
    ].values():
        require(
            summary["exact_train_eval_collision_count_lower_bound"] == 2,
            "Markov collision lower bound",
        )
        require(
            summary["exact_train_eval_collision_count_upper_bound"] == 4,
            "Markov collision upper bound",
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT_DIR / "audit_report.json"
    manifest_path = OUTPUT_DIR / "markov_task_manifest.csv"
    report_path = OUTPUT_DIR / "report_cn.md"
    checksum_path = OUTPUT_DIR / "SHA256SUMS"
    write_json(audit_path, report)
    write_csv(manifest_path, rows)
    report_path.write_text(build_report_cn(report), encoding="utf-8")
    write_checksums(
        [audit_path, manifest_path, report_path],
        checksum_path,
        OUTPUT_DIR,
    )

    p3_status = build_p3_status(report)
    write_json(P3_STATUS_PATH, p3_status)
    write_checksums(
        [
            P3_STATUS_PATH,
            TENSOR_AUDIT_PATH,
            audit_path,
        ],
        P3_CHECKSUM_PATH,
        P3_ANALYSIS_DIR,
    )

    print(
        json.dumps(
            {
                "candidate_status": report["status"],
                "markov_rows": len(rows),
                "p3_breadth_decision": p3_status["decision"],
                "p3_complete_audits": p3_status["observed"][
                    "complete_public_semantic_audits"
                ],
                "p3_credible_behavioral_benchmarks": p3_status["observed"][
                    "credible_behavioral_evidence_benchmarks"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
