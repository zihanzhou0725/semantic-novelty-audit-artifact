#!/usr/bin/env python3
"""Falsification canaries for the P5 certified-exposure proposition."""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path


EXTERNAL_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_ROOT = EXTERNAL_ROOT / "toolkit"
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from semantic_audit.bounds import (  # noqa: E402
    ExposureConstraint,
    assignment_satisfies_constraints,
    grouped_seed_constraint,
    singleton_constraint,
    solve_disjoint_certificate_bounds,
)
from semantic_audit.core import AuditError  # noqa: E402


class ExposureBoundCanaries(unittest.TestCase):
    def test_exact_function_interval(self) -> None:
        constraints = [
            singleton_constraint(
                name="seen",
                item_index=0,
                exposed=True,
                witness_type="complete_truth_table",
            ),
            singleton_constraint(
                name="unseen",
                item_index=1,
                exposed=False,
                witness_type="complete_truth_table",
            ),
        ]
        result = solve_disjoint_certificate_bounds(
            item_count=2,
            constraints=constraints,
        )
        self.assertEqual(
            (
                result["exposure_count_lower_bound"],
                result["exposure_count_upper_bound"],
            ),
            (1, 1),
        )

    def test_sound_alias_interval(self) -> None:
        result = solve_disjoint_certificate_bounds(
            item_count=2,
            constraints=[
                singleton_constraint(
                    name="witnessed_alias",
                    item_index=0,
                    exposed=True,
                    witness_type="executable_alias",
                )
            ],
        )
        self.assertEqual(
            (
                result["exposure_count_lower_bound"],
                result["exposure_count_upper_bound"],
            ),
            (1, 2),
        )

    def test_unresolved_interval(self) -> None:
        result = solve_disjoint_certificate_bounds(
            item_count=3,
            constraints=[],
        )
        self.assertEqual(
            (
                result["exposure_count_lower_bound"],
                result["exposure_count_upper_bound"],
            ),
            (0, 3),
        )

    def test_grouped_seed_pigeonhole_interval(self) -> None:
        constraint = grouped_seed_constraint(
            name="seed_5",
            item_indices=(0, 1),
            training_episode_count=6,
        )
        self.assertEqual((constraint.lower, constraint.upper), (1, 2))
        result = solve_disjoint_certificate_bounds(
            item_count=2,
            constraints=[constraint],
        )
        self.assertEqual(
            (
                result["exposure_count_lower_bound"],
                result["exposure_count_upper_bound"],
            ),
            (1, 2),
        )

    def test_overlapping_certificate_groups_rejected(self) -> None:
        constraints = [
            ExposureConstraint(
                name="first",
                item_indices=(0, 1),
                lower=1,
                upper=2,
                witness_type="first",
            ),
            ExposureConstraint(
                name="second",
                item_indices=(1, 2),
                lower=1,
                upper=2,
                witness_type="second",
            ),
        ]
        with self.assertRaises(AuditError):
            solve_disjoint_certificate_bounds(
                item_count=3,
                constraints=constraints,
            )

    def test_assignment_falsifies_invalid_certificate(self) -> None:
        constraint = ExposureConstraint(
            name="at_least_one",
            item_indices=(0, 1),
            lower=1,
            upper=2,
            witness_type="grouped_seed_pigeonhole",
        )
        self.assertFalse(
            assignment_satisfies_constraints(
                [False, False],
                [constraint],
            )
        )
        self.assertTrue(
            assignment_satisfies_constraints(
                [True, False],
                [constraint],
            )
        )

    def test_closed_form_matches_exhaustive_binary_search(self) -> None:
        constraints = [
            ExposureConstraint(
                name="group_a",
                item_indices=(0, 1),
                lower=1,
                upper=2,
                witness_type="grouped",
            ),
            ExposureConstraint(
                name="group_b",
                item_indices=(2,),
                lower=0,
                upper=0,
                witness_type="exact_negative",
            ),
        ]
        result = solve_disjoint_certificate_bounds(
            item_count=4,
            constraints=constraints,
        )
        feasible_totals = [
            sum(assignment)
            for assignment in itertools.product((0, 1), repeat=4)
            if assignment_satisfies_constraints(assignment, constraints)
        ]
        self.assertEqual(
            (
                result["exposure_count_lower_bound"],
                result["exposure_count_upper_bound"],
            ),
            (min(feasible_totals), max(feasible_totals)),
        )


if __name__ == "__main__":
    unittest.main()
