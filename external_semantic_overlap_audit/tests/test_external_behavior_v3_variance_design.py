"""Synthetic tests for V3 variance decomposition and V4 design gating."""

from __future__ import annotations

import math
import unittest

import numpy as np

from toolkit.external_behavior_v3.variance_design import (
    all_candidate_pairs,
    bootstrap_decomposition,
    precision_projection,
    random_effects_anova,
    sampling_distribution,
    v4_gate,
)


class VarianceDesignTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sizes = np.full((40, 34), 6, dtype=np.int16)
        self.sizes[:, -1] = 2
        self.pairs = np.arange(40, dtype=np.int16).reshape(20, 2)

    def test_sampling_subsets_preserve_constant_effect(self) -> None:
        differences = np.broadcast_to(
            self.sizes[None, :, :],
            (8, 40, 34),
        ).copy()
        for mask in range(8):
            values = sampling_distribution(
                differences,
                self.sizes,
                self.pairs,
                mask=mask,
                replicates=73,
                seed=100 + mask,
            )
            np.testing.assert_array_equal(values, np.full(73, 100.0))

    def test_shapley_contributions_add_to_full_variance(self) -> None:
        rng = np.random.default_rng(31)
        differences = rng.integers(
            -2,
            3,
            size=(8, 40, 34),
            dtype=np.int16,
        )
        _, summary, shapley = bootstrap_decomposition(
            differences,
            self.sizes,
            self.pairs,
        )
        full = next(row for row in summary if row["mask"] == 7)
        self.assertAlmostEqual(
            sum(row["shapley_variance_pp2"] for row in shapley),
            full["variance_pp2"],
            places=10,
        )

    def test_random_effects_projection_recovers_base_width(self) -> None:
        seeds = np.linspace(-2.0, 2.0, 8)[:, None]
        pairs = np.linspace(-10.0, 10.0, 20)[None, :]
        interaction = (
            np.arange(160, dtype=np.float64).reshape(8, 20) % 3
        )
        anova, rows = random_effects_anova(
            12.0 + seeds + pairs + interaction
        )
        projections, summary = precision_projection(anova, 9.25, 139)
        base = next(
            row
            for row in projections
            if row["seed_count"] == 8 and row["pair_count"] == 20
        )
        self.assertAlmostEqual(
            base["projected_interval_width_pp"],
            9.25,
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(
            math.isfinite(
                summary[
                    "projected_width_at_infinite_seeds_and_max_pairs_pp"
                ]
            )
        )

    def test_capacity_and_gate(self) -> None:
        self.assertEqual(len(all_candidate_pairs()), 159)
        projection = {
            "minimum_pairs_at_infinite_seeds": 600,
            "projected_width_at_infinite_seeds_and_max_pairs_pp": 4.0,
        }
        gate = v4_gate(projection, 139)
        self.assertEqual(
            gate["decision"],
            "strict_v3_precision_replication_no_go",
        )
        self.assertEqual(
            gate["recommendation"],
            "do_not_release_same_gate_v4_training",
        )
        self.assertFalse(gate["v4_training_released"])


if __name__ == "__main__":
    unittest.main()
