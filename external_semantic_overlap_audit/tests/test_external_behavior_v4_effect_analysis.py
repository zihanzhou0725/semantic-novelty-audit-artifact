"""Synthetic tests for directional V4 effect analysis."""

from __future__ import annotations

import unittest

import numpy as np

from toolkit.external_behavior_v4.effect_analysis import (
    classify_gate,
    exposed_arms,
    hierarchical_bootstrap,
    patterns,
)


class DirectionalV4EffectAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sizes = np.full((80, 34), 6, dtype=np.int16)
        self.sizes[:, -1] = 2
        self.pairs = np.arange(80, dtype=np.int16).reshape(40, 2)

    def test_bootstrap_preserves_constant_effect(self) -> None:
        differences = np.broadcast_to(
            self.sizes[None, :, :],
            (8, 80, 34),
        ).copy()
        effects, hashes = hierarchical_bootstrap(
            differences,
            self.sizes,
            self.pairs,
            replicates=131,
            seed=17,
            batch_size=13,
        )
        np.testing.assert_array_equal(effects, np.full(131, 100.0))
        self.assertEqual(
            set(hashes),
            {
                "group_draws_sha256",
                "pair_draws_sha256",
                "seed_draws_sha256",
            },
        )

    def test_bootstrap_is_batch_size_independent(self) -> None:
        rng = np.random.default_rng(9)
        differences = rng.integers(
            -2,
            3,
            size=(8, 80, 34),
            dtype=np.int16,
        )
        first, first_hashes = hierarchical_bootstrap(
            differences,
            self.sizes,
            self.pairs,
            replicates=97,
            seed=21,
            batch_size=7,
        )
        second, second_hashes = hierarchical_bootstrap(
            differences,
            self.sizes,
            self.pairs,
            replicates=97,
            seed=21,
            batch_size=32,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_hashes, second_hashes)

    def test_directional_gate(self) -> None:
        self.assertEqual(
            classify_gate(1.2, 0.1, 2.0)["classification"],
            "directional_positive_go",
        )
        self.assertEqual(
            classify_gate(0.9, 0.1, 1.8)["classification"],
            "directional_no_go",
        )
        self.assertEqual(
            classify_gate(2.0, -0.1, 4.0)["classification"],
            "directional_no_go",
        )

    def test_pattern_and_exposure_alignment(self) -> None:
        records, pair_indices = patterns()
        exposure = exposed_arms(records)
        self.assertEqual(len(records), 80)
        self.assertEqual(pair_indices.shape, (40, 2))
        self.assertEqual(exposure.shape, (8, 80))
        self.assertEqual(set(exposure.ravel().tolist()), {0, 1})
        self.assertEqual(
            [row["pattern_index"] for row in records],
            list(range(80)),
        )


if __name__ == "__main__":
    unittest.main()
