"""Focused solver tests for overlapping exposure certificates."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from toolkit.semantic_audit.bounds import (
    ExposureConstraint,
    assignment_satisfies_constraints,
    solve_disjoint_certificate_bounds,
    solve_overlapping_certificate_bounds,
    verify_dual_certificate,
)


def constraint(
    name: str,
    indices: tuple[int, ...],
    lower: int,
    upper: int,
) -> ExposureConstraint:
    return ExposureConstraint(
        name=name,
        item_indices=indices,
        lower=lower,
        upper=upper,
        witness_type="test",
    )


def exhaustive_bounds(
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


class OverlappingBoundsTest(unittest.TestCase):
    def test_weighted_milp_matches_exhaustive_solution(self) -> None:
        constraints = [
            constraint("left", (0, 1, 2), 1, 2),
            constraint("right", (1, 2, 3), 1, 1),
        ]
        weights = [1.0, 2.0, 3.0, 4.0]
        expected = exhaustive_bounds(4, constraints, weights)
        result = solve_overlapping_certificate_bounds(
            item_count=4,
            constraints=constraints,
            weights=weights,
            method="milp",
        )
        self.assertTrue(result["feasible"])
        self.assertTrue(result["exact"])
        self.assertEqual((result["lower"], result["upper"]), expected)

    def test_odd_cycle_separates_integer_and_lp_feasibility(self) -> None:
        constraints = [
            constraint("ab", (0, 1), 1, 1),
            constraint("bc", (1, 2), 1, 1),
            constraint("ac", (0, 2), 1, 1),
        ]
        integer = solve_overlapping_certificate_bounds(
            item_count=3,
            constraints=constraints,
            method="milp",
        )
        relaxed = solve_overlapping_certificate_bounds(
            item_count=3,
            constraints=constraints,
            method="lp",
        )
        self.assertFalse(integer["feasible"])
        self.assertTrue(relaxed["feasible"])
        self.assertAlmostEqual(relaxed["lower"], 1.5)
        self.assertAlmostEqual(relaxed["upper"], 1.5)

    def test_lp_dual_certificates_verify_and_tampering_fails(self) -> None:
        constraints = [
            constraint("left", (0, 1), 1, 2),
            constraint("right", (1, 2), 1, 2),
        ]
        result = solve_overlapping_certificate_bounds(
            item_count=3,
            constraints=constraints,
            weights=[1.0, 2.0, 1.0],
            method="lp",
        )
        matrix = np.asarray(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
        lower = np.asarray([1.0, 1.0])
        upper = np.asarray([2.0, 2.0])
        weights = np.asarray([1.0, 2.0, 1.0])
        for sense in ("lower", "upper"):
            certificate_data = result["lp_dual_certificate"][sense]
            self.assertTrue(
                verify_dual_certificate(
                    matrix,
                    lower,
                    upper,
                    weights,
                    certificate_data["alpha"],
                    certificate_data["beta"],
                    certificate_data["gamma"],
                    result[sense],
                    sense,
                )
            )
        damaged = dict(result["lp_dual_certificate"]["lower"])
        damaged["beta"] = list(damaged["beta"])
        damaged["beta"][0] += 1.0
        self.assertFalse(
            verify_dual_certificate(
                matrix,
                lower,
                upper,
                weights,
                damaged["alpha"],
                damaged["beta"],
                damaged["gamma"],
                result["lower"],
                "lower",
            )
        )

    def test_disjoint_case_remains_backward_compatible(self) -> None:
        constraints = [
            constraint("a", (0, 1), 1, 2),
            constraint("b", (2,), 0, 0),
        ]
        old = solve_disjoint_certificate_bounds(
            item_count=4,
            constraints=constraints,
        )
        new = solve_overlapping_certificate_bounds(
            item_count=4,
            constraints=constraints,
            method="milp",
        )
        self.assertEqual(
            (new["lower"], new["upper"]),
            (
                old["exposure_count_lower_bound"],
                old["exposure_count_upper_bound"],
            ),
        )


if __name__ == "__main__":
    unittest.main()
