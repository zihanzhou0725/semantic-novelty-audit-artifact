#!/usr/bin/env python3
"""Run focused solver properties and write overlapping-certificate artifacts."""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.semantic_audit.bounds import (  # noqa: E402
    ExposureConstraint,
    assignment_satisfies_constraints,
    solve_overlapping_certificate_bounds,
    verify_dual_certificate,
)
from toolkit.tmlr_strengthening.artifacts import (  # noqa: E402
    run_info,
    write_json,
    write_macros,
    write_sha256sums,
)


OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "tmlr-strengthening-v1"
    / "t9_overlapping_certificates"
)
BOUNDS_PATH = ROOT / "toolkit" / "semantic_audit" / "bounds.py"
SEED = 0
MILP_CASES = 100
DUAL_CASES = 100


def _constraint(
    index: int,
    indices: tuple[int, ...],
    lower: int,
    upper: int,
) -> ExposureConstraint:
    return ExposureConstraint(
        name=f"random_{index}",
        item_indices=indices,
        lower=lower,
        upper=upper,
        witness_type="property_test",
    )


def _random_case(
    rng: random.Random,
) -> tuple[int, list[ExposureConstraint], list[float]]:
    item_count = rng.randint(3, 8)
    constraints = []
    used_groups: set[tuple[int, ...]] = set()
    for index in range(rng.randint(0, 4)):
        size = rng.randint(1, item_count)
        group = tuple(sorted(rng.sample(range(item_count), size)))
        if group in used_groups:
            continue
        used_groups.add(group)
        lower = rng.randint(0, size)
        upper = rng.randint(lower, size)
        constraints.append(_constraint(index, group, lower, upper))
    weights = [float(rng.randint(1, 3)) for _ in range(item_count)]
    return item_count, constraints, weights


def _exhaustive(
    item_count: int,
    constraints: list[ExposureConstraint],
    weights: list[float],
) -> tuple[float, float] | None:
    values = [
        sum(weight * bit for weight, bit in zip(weights, assignment))
        for assignment in itertools.product((0, 1), repeat=item_count)
        if assignment_satisfies_constraints(assignment, constraints)
    ]
    return (min(values), max(values)) if values else None


def _matrix(
    item_count: int,
    constraints: list[ExposureConstraint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.zeros((len(constraints), item_count), dtype=float)
    for row_index, constraint in enumerate(constraints):
        matrix[row_index, list(constraint.item_indices)] = 1.0
    return (
        matrix,
        np.asarray([item.lower for item in constraints], dtype=float),
        np.asarray([item.upper for item in constraints], dtype=float),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    milp_passed = 0
    dual_passed = 0
    attempts = 0
    while milp_passed < MILP_CASES or dual_passed < DUAL_CASES:
        attempts += 1
        if attempts > 1000:
            raise RuntimeError("Property generation did not converge.")
        item_count, constraints, weights = _random_case(rng)
        expected = _exhaustive(item_count, constraints, weights)
        exact = solve_overlapping_certificate_bounds(
            item_count=item_count,
            constraints=constraints,
            weights=weights,
            method="milp",
        )
        if milp_passed < MILP_CASES:
            if expected is None:
                if exact["feasible"]:
                    raise AssertionError("MILP accepted an infeasible case.")
            elif (
                not exact["feasible"]
                or abs(float(exact["lower"]) - expected[0]) > 1e-8
                or abs(float(exact["upper"]) - expected[1]) > 1e-8
            ):
                raise AssertionError("MILP differs from exhaustive search.")
            milp_passed += 1

        if dual_passed < DUAL_CASES:
            relaxed = solve_overlapping_certificate_bounds(
                item_count=item_count,
                constraints=constraints,
                weights=weights,
                method="lp",
            )
            if not relaxed["feasible"]:
                continue
            if expected is not None and (
                float(relaxed["lower"]) > expected[0] + 1e-8
                or float(relaxed["upper"]) < expected[1] - 1e-8
            ):
                raise AssertionError("LP interval does not contain MILP interval.")
            matrix, lower, upper = _matrix(item_count, constraints)
            for sense in ("lower", "upper"):
                certificate = relaxed["lp_dual_certificate"][sense]
                if not verify_dual_certificate(
                    matrix,
                    lower,
                    upper,
                    np.asarray(weights),
                    certificate["alpha"],
                    certificate["beta"],
                    certificate["gamma"],
                    float(relaxed[sense]),
                    sense,
                ):
                    raise AssertionError("Generated dual certificate failed.")
            dual_passed += 1

    odd_constraints = [
        _constraint(0, (0, 1), 1, 1),
        _constraint(1, (1, 2), 1, 1),
        _constraint(2, (0, 2), 1, 1),
    ]
    odd_cycle = {
        "constraints": [item.to_dict() for item in odd_constraints],
        "milp": solve_overlapping_certificate_bounds(
            item_count=3,
            constraints=odd_constraints,
            method="milp",
        ),
        "lp": solve_overlapping_certificate_bounds(
            item_count=3,
            constraints=odd_constraints,
            method="lp",
        ),
    }
    summary = {
        "seed": SEED,
        "milp_cases": MILP_CASES,
        "milp_passed": milp_passed,
        "dual_cases": DUAL_CASES,
        "dual_passed": dual_passed,
        "odd_cycle_integer_feasible": odd_cycle["milp"]["feasible"],
        "odd_cycle_lp_interval": [
            odd_cycle["lp"]["lower"],
            odd_cycle["lp"]["upper"],
        ],
    }
    write_json(OUTPUT_DIR / "property_tests_summary.json", summary)
    write_json(OUTPUT_DIR / "odd_cycle_example.json", odd_cycle)
    write_macros(
        OUTPUT_DIR / "macros.tex",
        {
            "TNineMilpCases": str(MILP_CASES),
            "TNineDualCases": str(DUAL_CASES),
            "TNineOddLower": f"{odd_cycle['lp']['lower']:.1f}",
            "TNineOddUpper": f"{odd_cycle['lp']['upper']:.1f}",
        },
    )
    write_json(
        OUTPUT_DIR / "run_info.json",
        run_info(
            seed=SEED,
            script_path=(
                "scripts/tmlr_strengthening/"
                "run_t9_property_summary.py"
            ),
            input_paths=[BOUNDS_PATH],
        ),
    )
    write_sha256sums(OUTPUT_DIR)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
