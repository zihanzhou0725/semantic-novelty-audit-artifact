#!/usr/bin/env python3
"""Zero-training truth-table overlap audit for the ICLR 2024 Boolean ICL tasks."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


OPENREVIEW_ID = "ekeyCgeRfC"
SUPPLEMENTARY_SHA256 = "fb3e9f4eca33acba5504cf23cbaaac1a1b8c735edf5871e17731d94604dacfe8"
ARXIV_SOURCE_SHA256 = "0836940c35396696d3e9771c7fb5217b6085491268ec2ec9c11e0151c599a0c5"
CAMERA_READY_PAPER_SHA256 = "5af0615bf431cc637d93dae8f1f60337dbfd44b4b610a8c2832a18716c9b617f"
PUBLIC_REPOSITORY_REVISION = "a25a2cb0c62ad541cd30b0672df136230d878d9c"
PUBLIC_BOOL_TASKS_SHA256 = "5bda5cdb00bce9ad7710faa643d822c99eb09ba20a5de1a26d9da3788269ca60"


def hit_probability(probability: float, draws: int) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    return -math.expm1(draws * math.log1p(-probability))


def categorical_vector_overlap(
    n_dims: int, zero_probability: float, signed_probability: float, draws: int
) -> float:
    """Expected overlap when every coordinate is 0/+1/-1 independently."""

    overlap = 0.0
    for active in range(n_dims + 1):
        function_probability = (
            zero_probability ** (n_dims - active) * signed_probability**active
        )
        function_count = math.comb(n_dims, active) * 2**active
        overlap += (
            function_count
            * function_probability
            * hit_probability(function_probability, draws)
        )
    return overlap


def subset_overlap(n_dims: int, inclusion_probability: float, draws: int) -> float:
    """Expected overlap for a Bernoulli-distributed subset of variables."""

    overlap = 0.0
    for active in range(n_dims + 1):
        function_probability = (
            (1.0 - inclusion_probability) ** (n_dims - active)
            * inclusion_probability**active
        )
        overlap += (
            math.comb(n_dims, active)
            * function_probability
            * hit_probability(function_probability, draws)
        )
    return overlap


def uniform_overlap(function_count: int, draws: int) -> float:
    return hit_probability(1.0 / function_count, draws)


def ordered_three_clause_parameter_overlap(n_dims: int, draws: int) -> float:
    """Exact representation overlap for the released 3-DNF/3-CNF sampler."""

    positions = 3 * n_dims
    overlap = 0.0
    for active in range(positions + 1):
        representation_probability = 0.8 ** (positions - active) * 0.1**active
        representation_count = math.comb(positions, active) * 2**active
        overlap += (
            representation_count
            * representation_probability
            * hit_probability(representation_probability, draws)
        )
    return overlap


def three_clause_constant_mass(n_dims: int) -> float:
    """Mass of the constant truth table caused by at least one empty clause."""

    empty_clause_probability = 0.8**n_dims
    return 1.0 - (1.0 - empty_clause_probability) ** 3


def threshold_same_weight_semantic_overlap(n_dims: int, draws: int) -> dict[str, float]:
    """Sound lower bound for released SparseThreshold truth-table overlap.

    It merges threshold values that induce the same truth table for a fixed weight
    vector and merges all constant-positive/all-constant-negative functions. It
    intentionally does not claim completeness across different nonconstant weights.
    """

    thresholds = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
    constant_mass = {"positive": 0.0, "negative": 0.0}
    nonconstant_overlap = 0.0

    for active in range(n_dims + 1):
        weight_probability = 0.7 ** (n_dims - active) * 0.15**active
        weight_count = math.comb(n_dims, active) * 2**active
        attainable = tuple(range(-active, active + 1, 2))
        groups: dict[tuple[str, int | None], int] = {}
        for threshold in thresholds:
            positive_values = [value for value in attainable if value > threshold]
            if not positive_values:
                key = ("constant_negative", None)
            elif len(positive_values) == len(attainable):
                key = ("constant_positive", None)
            else:
                key = ("cut", min(positive_values))
            groups[key] = groups.get(key, 0) + 1

        for (kind, _), multiplicity in groups.items():
            class_probability_per_weight = weight_probability * multiplicity / len(thresholds)
            if kind == "constant_positive":
                constant_mass["positive"] += weight_count * class_probability_per_weight
            elif kind == "constant_negative":
                constant_mass["negative"] += weight_count * class_probability_per_weight
            else:
                nonconstant_overlap += (
                    weight_count
                    * class_probability_per_weight
                    * hit_probability(class_probability_per_weight, draws)
                )

    constant_overlap = sum(
        mass * hit_probability(mass, draws) for mass in constant_mass.values()
    )
    return {
        "overlap_lower_bound": constant_overlap + nonconstant_overlap,
        "constant_positive_mass": constant_mass["positive"],
        "constant_negative_mass": constant_mass["negative"],
    }


def threshold_exact_parameter_overlap(n_dims: int, draws: int) -> float:
    overlap = 0.0
    threshold_count = 6
    for active in range(n_dims + 1):
        weight_probability = 0.7 ** (n_dims - active) * 0.15**active
        parameter_probability = weight_probability / threshold_count
        parameter_count = math.comb(n_dims, active) * 2**active * threshold_count
        overlap += (
            parameter_count
            * parameter_probability
            * hit_probability(parameter_probability, draws)
        )
    return overlap


def integer_halfspace_scaling_overlap(n_dims: int, draws: int) -> float:
    """Overlap lower bound after merging positive integer multiples of weights."""

    parameter_count = 7**n_dims
    primitive_max_one = 3**n_dims - 1
    singleton_classes = parameter_count - 3 * primitive_max_one
    return (
        primitive_max_one
        * (3.0 / parameter_count)
        * hit_probability(3.0 / parameter_count, draws)
        + singleton_classes
        * (1.0 / parameter_count)
        * hit_probability(1.0 / parameter_count, draws)
    )


def truth_table(kind: str, parameter: tuple[int, ...]) -> tuple[int, ...]:
    table = []
    for point in itertools.product((-1, 1), repeat=len(parameter)):
        if kind == "conjunction":
            active = [i for i, value in enumerate(parameter) if value]
            output = all(point[i] == parameter[i] for i in active)
        elif kind == "majority":
            active = [i for i, value in enumerate(parameter) if value]
            output = sum(point[i] for i in active) >= 1
        elif kind == "parity":
            active = [i for i, value in enumerate(parameter) if value]
            output = sum(point[i] == 1 for i in active) % 2 == 1
        else:
            raise ValueError(kind)
        table.append(1 if output else -1)
    return tuple(table)


def exhaustive_small_dimension_checks(n_dims: int = 6) -> dict:
    checks = {}
    parameter_spaces = {
        "conjunction": itertools.product((-1, 0, 1), repeat=n_dims),
        "majority": itertools.product((0, 1), repeat=n_dims),
        "parity": itertools.product((0, 1), repeat=n_dims),
    }
    for kind, parameters in parameter_spaces.items():
        params = list(parameters)
        tables = {truth_table(kind, parameter) for parameter in params}
        checks[kind] = {
            "n_dims": n_dims,
            "parameter_count": len(params),
            "unique_truth_tables": len(tables),
            "parameter_to_truth_table_is_injective": len(params) == len(tables),
        }
    return checks


def build_results(draws: int, eval_examples: int) -> dict:
    conjunction = categorical_vector_overlap(28, 0.7, 0.15, draws)
    majority = subset_overlap(28, 0.3, draws)
    threshold_semantic = threshold_same_weight_semantic_overlap(28, draws)
    constant_mass = three_clause_constant_mass(28)
    constant_lower_bound = constant_mass * hit_probability(constant_mass, draws)

    tasks = {
        "conjunction_d28": {
            "status": "exact",
            "expected_test_function_overlap": conjunction,
            "expected_overlapping_eval_examples": eval_examples * conjunction,
            "basis": "Released sampler; every {-1,0,+1} literal vector has a unique truth table.",
        },
        "disjunction_d28": {
            "status": "exact",
            "expected_test_function_overlap": conjunction,
            "expected_overlapping_eval_examples": eval_examples * conjunction,
            "basis": "Same parameter distribution and injectivity as conjunction.",
        },
        "majority_d28": {
            "status": "exact",
            "expected_test_function_overlap": majority,
            "expected_overlapping_eval_examples": eval_examples * majority,
            "basis": "Released Bernoulli(0.3) subset sampler; subsets have unique truth tables.",
        },
        "sparse_disjunction_paper_stated_count_d50_k3": {
            "status": "overlap_implied_by_paper_stated_count_under_uniform_sampling",
            "paper_stated_function_count": math.comb(50, 3) * 3**3,
            "expected_test_function_overlap": uniform_overlap(
                math.comb(50, 3) * 3**3, draws
            ),
        },
        "sparse_disjunction_signed_at_most_k_d50_k3": {
            "status": "exact_for_the_paper_class_definition_under_uniform_sampling",
            "function_count": sum(math.comb(50, k) * 2**k for k in range(4)),
            "expected_test_function_overlap": uniform_overlap(
                sum(math.comb(50, k) * 2**k for k in range(4)), draws
            ),
        },
        "sparse_disjunction_released_code_d50_k4_positive_only": {
            "status": "exact_for_released_code",
            "function_count": math.comb(50, 4),
            "expected_test_function_overlap": uniform_overlap(math.comb(50, 4), draws),
        },
        "three_dnf_and_three_cnf_d28": {
            "status": "sound_lower_bound",
            "expected_test_function_overlap_lower_bound": constant_lower_bound,
            "constant_truth_table_probability": constant_mass,
            "exact_ordered_parameter_overlap": ordered_three_clause_parameter_overlap(
                28, draws
            ),
            "basis": (
                "An empty conjunction makes a 3-DNF constantly positive; an empty "
                "disjunction makes a 3-CNF constantly negative. Clause permutation, "
                "absorption, and other Boolean identities can only increase overlap."
            ),
        },
        "threshold_released_code_d28": {
            "status": "sound_lower_bound",
            "exact_parameter_overlap": threshold_exact_parameter_overlap(28, draws),
            **threshold_semantic,
            "basis": (
                "Merges identical parameters, same-weight threshold aliases, and "
                "constant truth tables. Cross-weight threshold representations are omitted."
            ),
        },
        "integer_halfspace_released_code_d20": {
            "status": "sound_lower_bound",
            "expected_test_function_overlap_lower_bound": integer_halfspace_scaling_overlap(
                20, draws
            ),
            "basis": (
                "Merges only positive integer multiples w, 2w, and 3w. Other equivalent "
                "linear-threshold representations are omitted."
            ),
        },
        "parity_d10_k2": {
            "status": "exact",
            "function_count": math.comb(10, 2),
            "expected_test_function_overlap": uniform_overlap(math.comb(10, 2), draws),
        },
        "parity_d20_k3": {
            "status": "exact_for_paper_task_not_released_sampler",
            "function_count": math.comb(20, 3),
            "expected_test_function_overlap": uniform_overlap(math.comb(20, 3), draws),
        },
        "parity_d20_all": {
            "status": "exact",
            "function_count": 2**20,
            "expected_test_function_overlap": uniform_overlap(2**20, draws),
            "expected_overlapping_eval_examples": eval_examples
            * uniform_overlap(2**20, draws),
        },
    }
    return {
        "training_function_draws": draws,
        "evaluation_examples": eval_examples,
        "overlap_definition": (
            "Probability that an independently sampled test truth table was sampled at "
            "least once among N training functions: sum_f p(f)[1-(1-p(f))^N]."
        ),
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument(
        "--supplementary",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "upstream/boolean_icl_supplementary.zip",
    )
    parser.add_argument(
        "--arxiv-source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "upstream/boolean_icl_arxiv_source.tar",
    )
    parser.add_argument(
        "--camera-ready-paper",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "upstream/boolean_icl_iclr2024_paper.pdf",
    )
    parser.add_argument(
        "--public-bool-tasks",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "upstream/boolean_icl_public/src/bool_tasks.py",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path, expected in (
        (args.supplementary, SUPPLEMENTARY_SHA256),
        (args.arxiv_source, ARXIV_SOURCE_SHA256),
        (args.camera_ready_paper, CAMERA_READY_PAPER_SHA256),
        (args.public_bool_tasks, PUBLIC_BOOL_TASKS_SHA256),
    ):
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise SystemExit(f"Checksum mismatch for {path}: {observed} != {expected}")

    budgets = {
        "public_readme_conjunction_demo_15k_x_64": build_results(15_000 * 64, 1280),
        "paper_explicit_conjunction_example_20k_x_64": build_results(
            20_000 * 64, 1280
        ),
        "paper_conjunction_upper_scenario_40k_x_64": build_results(
            40_000 * 64, 1280
        ),
        "parity_500k_x_released_default_batch64_scenario": build_results(
            500_000 * 64, 1280
        ),
    }
    result = {
        "source": {
            "openreview_id": OPENREVIEW_ID,
            "supplementary_sha256": SUPPLEMENTARY_SHA256,
            "arxiv_id": "2310.03016",
            "arxiv_source_sha256": ARXIV_SOURCE_SHA256,
            "camera_ready_paper_sha256": CAMERA_READY_PAPER_SHA256,
            "official_repository": "https://github.com/satwik77/incontext-bool",
            "official_repository_revision": PUBLIC_REPOSITORY_REVISION,
            "public_bool_tasks_sha256": PUBLIC_BOOL_TASKS_SHA256,
        },
        "generator_version_finding": (
            "The author-designated public generator is "
            "satwik77/incontext-bool@a25a2cb. Its core task implementations are "
            "functionally equivalent to the camera-ready supplementary archive. "
            "No released branch can reproduce several conflicting main-table task "
            "definitions, so the private generator/config actually used for those "
            "rows is not uniquely recoverable."
        ),
        "budget_scope_warning": (
            "The 1.28M-function budget is explicitly documented only as a "
            "Conjunction example. Exact per-task main-table function draws are not "
            "recoverable because task-specific final steps, batch sizes, logs, and "
            "checkpoints were not released. Other budgets below are comparison scenarios."
        ),
        "protocol_finding": (
            "There is no function-disjoint train/test split. Training and evaluation draw "
            "fresh functions independently from the same discrete distribution."
        ),
        "small_dimension_truth_table_checks": exhaustive_small_dimension_checks(),
        "budgets": budgets,
        "released_artifact_mismatches": [
            {
                "paper": (
                    "Sparse Disjunction uses at most k=3 signed literals, but states "
                    "C(50,3)*3^3=529,200 functions and calls this at least one million. "
                    "The unique truth-table count is sum_{j=0}^3 C(50,j)*2^j=161,801."
                ),
                "code": "SparseDisjunction hard-codes k=4 and only positive weights.",
            },
            {
                "paper": "Parity-(20,3) is a main task.",
                "code": "SparseParity hard-codes k=2; no k argument is consumed.",
            },
            {
                "paper": "0-1 Threshold samples weights uniformly and b in {-3,...,3}.",
                "code": (
                    "SparseThreshold samples weights with probabilities "
                    "[0.7,0.15,0.15] and six half-integer thresholds -2.5,...,2.5."
                ),
            },
            {
                "paper": "Main task names include 0-1 Threshold and Integer Halfspace.",
                "code": (
                    "The released task registry exposes sparse_thres and int_halfspace; "
                    "bundled configs use halfspace/sparse_halfspace names not in the registry."
                ),
            },
        ],
        "interpretation": (
            "Function repetition is not the same as repeated prompts because input sequences "
            "are resampled. It does, however, invalidate a function-disjoint generalization "
            "interpretation for classes with material overlap."
        ),
    }

    output_path = args.output_dir / "boolean_icl_overlap_results.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nResults: {output_path}")


if __name__ == "__main__":
    main()
