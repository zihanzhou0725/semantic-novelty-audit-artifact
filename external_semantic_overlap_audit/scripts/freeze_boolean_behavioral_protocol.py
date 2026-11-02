#!/usr/bin/env python3
"""Freeze the preregistered Boolean ICL behavioral experiment protocol.

This script creates immutable evaluation-function manifests and deterministic
training-function stream hashes. It does not train or evaluate a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "boolean-behavioral-v1"
FREEZE_DATE = "2026-10-07"
MASTER_SEED = 20261007
MODEL_SEEDS = (0, 1, 2)
TRAIN_STEPS = 20_000
TRAIN_BATCH_SIZE = 32
TRAIN_FUNCTION_DRAWS = TRAIN_STEPS * TRAIN_BATCH_SIZE
EVAL_FUNCTION_DRAWS = 1_280
N_POINTS = 40
TAIL_START_ZERO_INDEXED = 20
PUBLIC_REPOSITORY_REVISION = "a25a2cb0c62ad541cd30b0672df136230d878d9c"
PUBLIC_SOURCE_HASHES = {
    "args.py": "64759e100b1b07e55e333028d17297a73716ff3684be8753a619620474fec1be",
    "bool_tasks.py": "5bda5cdb00bce9ad7710faa643d822c99eb09ba20a5de1a26d9da3788269ca60",
    "eval.py": "fbd87507b71f3f191678e4d36221bb967daee0d015a749d77762b5274f1698cd",
    "models.py": "397e25e2b35a737900195037b7aecd4b183b41313856e969bdf7f99f600a05bc",
    "train.py": "7b36b99ec1fe91a0de8b18cbf4e1a841be1bd52662592a4d11ee27a31c7ff86c",
}

TASKS = {
    "conjunction_d28": {
        "public_task_name": "conjunction",
        "n_dims": 28,
        "semantic_id": "base3 literal vector: 0=absent, 1=positive, 2=negative",
        "distribution": "independent digits with probabilities [0.70, 0.15, 0.15]",
    },
    "majority_d28": {
        "public_task_name": "majority",
        "n_dims": 28,
        "semantic_id": "bit mask of the active-variable subset",
        "distribution": "independent active bits with probability 0.30",
    },
    "sparse_disjunction_d50_k4_positive": {
        "public_task_name": "sparse_disjunction",
        "n_dims": 50,
        "k": 4,
        "semantic_id": "bit mask of the four active positive literals",
        "distribution": "uniform four-element subset of 50 variables",
    },
}

UINT64_CARDINALITY = 1 << 64
THRESHOLD_70 = int(Decimal("0.70") * UINT64_CARDINALITY)
THRESHOLD_85 = int(Decimal("0.85") * UINT64_CARDINALITY)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(namespace: str) -> int:
    payload = f"{SCHEMA_VERSION}:{MASTER_SEED}:{namespace}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def ids_sha256(ids: np.ndarray) -> str:
    values = np.asarray(ids, dtype="<u8")
    return sha256_bytes(values.tobytes(order="C"))


def semantic_id(task_name: str, encoded: int) -> str:
    return f"{task_name}:v1:{encoded:016x}"


def _sample_conjunction_ids(
    rng: np.random.Generator, count: int, n_dims: int
) -> np.ndarray:
    raw = rng.bit_generator.random_raw((count, n_dims))
    digits = np.where(raw < THRESHOLD_70, 0, np.where(raw < THRESHOLD_85, 1, 2))
    powers = np.asarray([3**index for index in range(n_dims)], dtype=np.uint64)
    return digits.astype(np.uint64) @ powers


def _sample_majority_ids(
    rng: np.random.Generator, count: int, n_dims: int
) -> np.ndarray:
    raw = rng.bit_generator.random_raw((count, n_dims))
    active = raw >= THRESHOLD_70
    powers = np.left_shift(np.uint64(1), np.arange(n_dims, dtype=np.uint64))
    return active.astype(np.uint64) @ powers


def _sample_sparse_subset_ids(
    rng: np.random.Generator, count: int, n_dims: int, k: int
) -> np.ndarray:
    result = np.empty(count, dtype=np.uint64)
    chunk_size = 32_768
    powers = np.left_shift(np.uint64(1), np.arange(n_dims, dtype=np.uint64))
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        raw = rng.bit_generator.random_raw((stop - start, n_dims))
        active = np.argpartition(raw, kth=k - 1, axis=1)[:, :k]
        result[start:stop] = np.bitwise_or.reduce(powers[active], axis=1)
    return result


def sample_ids_from_rng(
    task_name: str, count: int, rng: np.random.Generator
) -> np.ndarray:
    task = TASKS[task_name]
    if task_name == "conjunction_d28":
        return _sample_conjunction_ids(rng, count, task["n_dims"])
    if task_name == "majority_d28":
        return _sample_majority_ids(rng, count, task["n_dims"])
    if task_name == "sparse_disjunction_d50_k4_positive":
        return _sample_sparse_subset_ids(
            rng, count, task["n_dims"], task["k"]
        )
    raise ValueError(f"Unknown task: {task_name}")


def sample_ids(task_name: str, count: int, seed: int) -> np.ndarray:
    return sample_ids_from_rng(task_name, count, np.random.Generator(np.random.PCG64(seed)))


def decode_parameter(task_name: str, encoded: int) -> dict:
    if task_name == "conjunction_d28":
        literals = []
        remainder = encoded
        for _ in range(TASKS[task_name]["n_dims"]):
            digit = remainder % 3
            literals.append((0, 1, -1)[digit])
            remainder //= 3
        return {"literals": literals}

    n_dims = TASKS[task_name]["n_dims"]
    active = [index for index in range(n_dims) if encoded & (1 << index)]
    return {"active_indices": active}


def semantic_probability(task_name: str, encoded: int) -> float:
    if task_name == "conjunction_d28":
        literals = decode_parameter(task_name, encoded)["literals"]
        active = sum(value != 0 for value in literals)
        return 0.70 ** (28 - active) * 0.15**active
    if task_name == "majority_d28":
        active = encoded.bit_count()
        return 0.70 ** (28 - active) * 0.30**active
    if task_name == "sparse_disjunction_d50_k4_positive":
        return 1.0 / math.comb(50, 4)
    raise ValueError(f"Unknown task: {task_name}")


def expected_exposure_probability(task_name: str, draws: int) -> float:
    def hit(probability: float) -> float:
        return -math.expm1(draws * math.log1p(-probability))

    if task_name == "conjunction_d28":
        return sum(
            math.comb(28, active)
            * 2**active
            * 0.70 ** (28 - active)
            * 0.15**active
            * hit(0.70 ** (28 - active) * 0.15**active)
            for active in range(29)
        )
    if task_name == "majority_d28":
        return sum(
            math.comb(28, active)
            * 0.70 ** (28 - active)
            * 0.30**active
            * hit(0.70 ** (28 - active) * 0.30**active)
            for active in range(29)
        )
    if task_name == "sparse_disjunction_d50_k4_positive":
        probability = 1.0 / math.comb(50, 4)
        return hit(probability)
    raise ValueError(f"Unknown task: {task_name}")


def replace_excluded_slots(
    task_name: str,
    original: np.ndarray,
    excluded: np.ndarray,
    replacement_seed: int,
) -> tuple[np.ndarray, int]:
    strict = original.copy()
    excluded_set = set(int(value) for value in excluded)
    pending = np.flatnonzero(np.isin(strict, excluded))
    replacement_draws = 0
    rng = np.random.Generator(np.random.PCG64(replacement_seed))

    while len(pending):
        candidates = sample_ids_from_rng(task_name, len(pending), rng)
        replacement_draws += len(candidates)
        accepted = np.fromiter(
            (int(value) not in excluded_set for value in candidates),
            dtype=bool,
            count=len(candidates),
        )
        strict[pending[accepted]] = candidates[accepted]
        pending = pending[~accepted]

    return strict, replacement_draws


def build_preregistration() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "freeze_date": FREEZE_DATE,
        "status": "frozen_before_training",
        "training_started": False,
        "scope": (
            "Controlled behavioral test of semantic exposure under the released "
            "Boolean ICL generators; not a reproduction of unreleased paper checkpoints."
        ),
        "source": {
            "repository": "https://github.com/satwik77/incontext-bool",
            "revision": PUBLIC_REPOSITORY_REVISION,
            "source_sha256": PUBLIC_SOURCE_HASHES,
        },
        "semantic_identity": {
            task_name: {
                **task,
                "proof_status": (
                    "injective for the released task family; conjunction and majority "
                    "validated by exhaustive truth tables at d=6"
                ),
            }
            for task_name, task in TASKS.items()
        },
        "split": {
            "master_seed": MASTER_SEED,
            "bit_generator": "numpy.random.PCG64 raw uint64 stream",
            "evaluation_function_draws": EVAL_FUNCTION_DRAWS,
            "evaluation_sampling": "i.i.d. with replacement from the released distribution",
            "strict_exclusion": (
                "Every unique evaluation semantic ID is excluded from every strict "
                "training-function draw."
            ),
            "paired_slot_rule": (
                "Original and strict use the same primary function at every training "
                "slot. Strict replaces only slots whose primary semantic ID is excluded, "
                "using a separately seeded conditional stream."
            ),
        },
        "model": {
            "family": "san",
            "implementation": "src.models.TransformerModel using GPT2Config",
            "n_embd": 256,
            "n_layer": 6,
            "n_head": 8,
            "n_positions": 150,
            "dropout": 0.0,
            "precision": "float32",
        },
        "training": {
            "model_seeds": list(MODEL_SEEDS),
            "optimizer": "Adam",
            "learning_rate": 0.0001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "loss": "released sigmoid plus binary cross entropy",
            "steps": TRAIN_STEPS,
            "batch_size": TRAIN_BATCH_SIZE,
            "function_draws_per_run": TRAIN_FUNCTION_DRAWS,
            "n_points": N_POINTS,
            "checkpoint_selection": "fixed final checkpoint; no validation selection",
            "protocols": ["original", "strict_function_disjoint"],
            "core_run_count": len(TASKS) * len(MODEL_SEEDS) * 2,
        },
        "evaluation": {
            "same_manifest_for_every_run": True,
            "same_prompts_for_paired_protocols": True,
            "n_points": N_POINTS,
            "primary_positions_zero_indexed": list(
                range(TAIL_START_ZERO_INDEXED, N_POINTS)
            ),
            "unit_of_analysis": "evaluation episode, clustered by semantic ID",
        },
        "metrics": {
            "primary": {
                "name": "tail_accuracy",
                "definition": "mean sign accuracy over zero-indexed positions 20 through 39",
                "effect": "100 * (original - strict_function_disjoint)",
                "unit": "percentage points",
            },
            "secondary": [
                "final_position_accuracy",
                "all_positions_1_through_39_accuracy",
                "tail_binary_cross_entropy",
                "original_accuracy_stratified_by_actual_training_exposure",
            ],
            "uncertainty": (
                "Report each model-seed paired delta and a paired cluster bootstrap "
                "over semantic IDs within each seed. Do not treat positions as "
                "independent observations."
            ),
        },
        "gate": {
            "core_go": (
                "At least one preregistered task has mean original-minus-strict "
                "tail-accuracy >= 2.0 pp and all three seed deltas are positive."
            ),
            "strong_pattern": (
                "Sparse Disjunction is >= 5.0 pp, Conjunction absolute delta is <= "
                "2.0 pp, and task effects increase from low to high expected exposure."
            ),
            "no_go": (
                "No task passes core_go. Report the null/negative result and do not "
                "claim observed neural reliance on semantic exposure."
            ),
            "reporting_rule": "Report all tasks and seeds regardless of gate outcome.",
        },
        "conditional_expansion": {
            "transformer_seeds_3_and_4": (
                "Run only for tasks passing core_go after the frozen three-seed analysis."
            ),
            "ranking_test": (
                "Run the released LSTM on Sparse Disjunction only if the Transformer "
                "Sparse Disjunction result passes core_go. A ranking-change claim "
                "requires the model ordering to reverse between protocols; otherwise "
                "the claim is forbidden."
            ),
        },
        "claim_boundary": {
            "allowed_if_core_go": [
                "The tested Transformer benefits from prior exposure to evaluation semantics.",
                "For a passing task, the original protocol exceeds the strict protocol by the reported paired delta.",
            ],
            "forbidden_without_additional_evidence": [
                "The original paper model memorizes evaluation functions.",
                "All model families rely on memorization.",
                "Model rankings change.",
                "Every benchmark with semantic overlap has inflated accuracy.",
            ],
        },
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "protocols" / SCHEMA_VERSION
    output_dir.mkdir(parents=True, exist_ok=True)

    source_root = root / "upstream" / "boolean_icl_public" / "src"
    for name, expected in PUBLIC_SOURCE_HASHES.items():
        observed = sha256_file(source_root / name)
        if observed != expected:
            raise SystemExit(
                f"Public source checksum mismatch for {name}: {observed} != {expected}"
            )

    preregistration_path = output_dir / "preregistration.json"
    eval_path = output_dir / "eval_functions.jsonl"
    run_path = output_dir / "run_manifest.csv"
    split_path = output_dir / "split_manifest.json"
    checksums_path = output_dir / "SHA256SUMS"

    preregistration = build_preregistration()
    write_json(preregistration_path, preregistration)

    eval_by_task = {}
    eval_summaries = {}
    with eval_path.open("w") as handle:
        for task_name in TASKS:
            eval_seed = derive_seed(f"eval-functions:{task_name}")
            eval_ids = sample_ids(task_name, EVAL_FUNCTION_DRAWS, eval_seed)
            eval_by_task[task_name] = eval_ids
            unique_ids = np.unique(eval_ids)
            excluded_mass = sum(
                semantic_probability(task_name, int(value)) for value in unique_ids
            )
            eval_summaries[task_name] = {
                "function_seed": eval_seed,
                "draw_count": len(eval_ids),
                "unique_semantic_id_count": len(unique_ids),
                "ordered_semantic_ids_sha256": ids_sha256(eval_ids),
                "sorted_unique_semantic_ids_sha256": ids_sha256(unique_ids),
                "strict_excluded_probability_mass": excluded_mass,
                "expected_original_exposure_probability": (
                    expected_exposure_probability(task_name, TRAIN_FUNCTION_DRAWS)
                ),
            }
            for index, encoded in enumerate(eval_ids):
                encoded_int = int(encoded)
                row = {
                    "task": task_name,
                    "eval_index": index,
                    "semantic_id": semantic_id(task_name, encoded_int),
                    "encoded_semantic_id": encoded_int,
                    "parameter": decode_parameter(task_name, encoded_int),
                    "numpy_prompt_seed": derive_seed(
                        f"eval-numpy-prompt:{task_name}:{index}"
                    ),
                    "torch_prompt_seed": derive_seed(
                        f"eval-torch-prompt:{task_name}:{index}"
                    ),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    fieldnames = [
        "run_id",
        "task",
        "protocol",
        "model_seed",
        "train_function_seed",
        "replacement_seed",
        "train_numpy_input_seed",
        "train_torch_input_seed",
        "function_draw_count",
        "function_stream_sha256",
        "paired_unchanged_slot_count",
        "replaced_slot_count",
        "replacement_candidate_draw_count",
        "eval_draw_exposure_count",
        "eval_draw_exposure_rate",
        "eval_unique_exposure_count",
        "eval_unique_semantic_id_count",
    ]
    run_rows = []
    task_run_summaries = {}
    for task_name in TASKS:
        eval_ids = eval_by_task[task_name]
        eval_unique = np.unique(eval_ids)
        task_run_summaries[task_name] = []
        for model_seed in MODEL_SEEDS:
            function_seed = derive_seed(
                f"train-functions:{task_name}:model-seed-{model_seed}"
            )
            replacement_seed = derive_seed(
                f"strict-replacements:{task_name}:model-seed-{model_seed}"
            )
            numpy_input_seed = derive_seed(
                f"train-numpy-inputs:{task_name}:model-seed-{model_seed}"
            )
            torch_input_seed = derive_seed(
                f"train-torch-inputs:{task_name}:model-seed-{model_seed}"
            )
            original = sample_ids(
                task_name, TRAIN_FUNCTION_DRAWS, function_seed
            )
            original_unique = np.unique(original)
            collision_mask = np.isin(original, eval_unique)
            strict, replacement_draws = replace_excluded_slots(
                task_name, original, eval_unique, replacement_seed
            )
            if np.any(np.isin(strict, eval_unique)):
                raise AssertionError(f"Strict stream overlaps eval IDs: {task_name}")

            eval_exposed = np.isin(eval_ids, original_unique)
            unique_exposed = np.isin(eval_unique, original_unique)
            common = {
                "task": task_name,
                "model_seed": model_seed,
                "train_function_seed": function_seed,
                "replacement_seed": replacement_seed,
                "train_numpy_input_seed": numpy_input_seed,
                "train_torch_input_seed": torch_input_seed,
                "function_draw_count": TRAIN_FUNCTION_DRAWS,
                "eval_draw_exposure_count": int(eval_exposed.sum()),
                "eval_draw_exposure_rate": float(eval_exposed.mean()),
                "eval_unique_exposure_count": int(unique_exposed.sum()),
                "eval_unique_semantic_id_count": len(eval_unique),
            }
            original_row = {
                "run_id": f"{task_name}__original__seed{model_seed}",
                "protocol": "original",
                "function_stream_sha256": ids_sha256(original),
                "paired_unchanged_slot_count": TRAIN_FUNCTION_DRAWS,
                "replaced_slot_count": 0,
                "replacement_candidate_draw_count": 0,
                **common,
            }
            strict_row = {
                "run_id": f"{task_name}__strict__seed{model_seed}",
                "protocol": "strict_function_disjoint",
                "function_stream_sha256": ids_sha256(strict),
                "paired_unchanged_slot_count": int((~collision_mask).sum()),
                "replaced_slot_count": int(collision_mask.sum()),
                "replacement_candidate_draw_count": replacement_draws,
                **common,
            }
            run_rows.extend((original_row, strict_row))
            task_run_summaries[task_name].append(
                {
                    "model_seed": model_seed,
                    "original_function_stream_sha256": original_row[
                        "function_stream_sha256"
                    ],
                    "strict_function_stream_sha256": strict_row[
                        "function_stream_sha256"
                    ],
                    "original_train_slots_matching_eval": int(collision_mask.sum()),
                    "strict_replacement_candidate_draws": replacement_draws,
                    "eval_draw_exposure_count_in_original": int(eval_exposed.sum()),
                    "eval_draw_exposure_rate_in_original": float(eval_exposed.mean()),
                    "strict_eval_overlap_count": 0,
                }
            )

    with run_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "freeze_date": FREEZE_DATE,
        "protocol_generator_sha256": sha256_file(Path(__file__).resolve()),
        "training_adapter_sha256": sha256_file(
            root / "scripts" / "run_boolean_behavioral.py"
        ),
        "behavioral_requirements_sha256": sha256_file(
            root / "behavioral_requirements.txt"
        ),
        "generator_runtime": {
            "numpy_version": np.__version__,
            "bit_generator": "PCG64",
            "sampling_primitive": "bit_generator.random_raw",
        },
        "preregistration_sha256": sha256_file(preregistration_path),
        "eval_functions_file": eval_path.name,
        "eval_functions_sha256": sha256_file(eval_path),
        "run_manifest_file": run_path.name,
        "run_manifest_sha256": sha256_file(run_path),
        "task_eval_summaries": eval_summaries,
        "task_training_stream_summaries": task_run_summaries,
        "validation_invariants": {
            "core_run_count": len(run_rows),
            "strict_eval_overlap_count": 0,
            "original_strict_noncollision_slots_identical_by_construction": True,
            "all_eval_rows_have_explicit_semantic_id_and_prompt_seeds": True,
        },
    }
    write_json(split_path, split_manifest)

    checksum_paths = [
        preregistration_path,
        eval_path,
        run_path,
        split_path,
    ]
    readme_path = output_dir / "README.md"
    if readme_path.exists():
        checksum_paths.append(readme_path)
    checksums_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths)
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "core_runs": len(run_rows),
                "task_eval_summaries": eval_summaries,
                "sha256s": {
                    path.name: sha256_file(path) for path in checksum_paths
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("No model training or inference was performed.")


if __name__ == "__main__":
    main()
