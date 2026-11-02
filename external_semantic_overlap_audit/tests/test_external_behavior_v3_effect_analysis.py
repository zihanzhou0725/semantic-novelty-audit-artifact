"""Synthetic tests for the frozen CLUTRR V3 effect analysis."""

from __future__ import annotations

import unittest

import numpy as np

from toolkit.external_behavior_v3.effect_analysis import (
    classification_metrics,
    classify_gate,
    hierarchical_bootstrap,
)


class EffectAnalysisTest(unittest.TestCase):
    def test_bootstrap_preserves_constant_cluster_effect(self) -> None:
        sizes = np.full((40, 34), 6, dtype=np.int16)
        sizes[:, -1] = 2
        differences = np.broadcast_to(
            sizes[None, :, :],
            (8, 40, 34),
        ).copy()
        pairs = np.arange(40, dtype=np.int16).reshape(20, 2)
        effects, hashes = hierarchical_bootstrap(
            differences,
            sizes,
            pairs,
            replicates=257,
            seed=11,
            batch_size=17,
        )
        np.testing.assert_array_equal(effects, np.full(257, 100.0))
        self.assertEqual(
            set(hashes),
            {
                "group_draws_sha256",
                "pair_draws_sha256",
                "seed_draws_sha256",
            },
        )

    def test_bootstrap_is_independent_of_batch_size(self) -> None:
        rng = np.random.default_rng(7)
        sizes = np.full((40, 34), 6, dtype=np.int16)
        sizes[:, -1] = 2
        differences = rng.integers(
            -2,
            3,
            size=(8, 40, 34),
            dtype=np.int16,
        )
        pairs = np.arange(40, dtype=np.int16).reshape(20, 2)
        first, first_hashes = hierarchical_bootstrap(
            differences,
            sizes,
            pairs,
            replicates=129,
            seed=19,
            batch_size=7,
        )
        second, second_hashes = hierarchical_bootstrap(
            differences,
            sizes,
            pairs,
            replicates=129,
            seed=19,
            batch_size=64,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_hashes, second_hashes)

    def test_gate_precedence(self) -> None:
        self.assertEqual(
            classify_gate(2.0, 0.2, 2.3)["classification"],
            "imprecise_no_go",
        )
        self.assertEqual(
            classify_gate(1.1, 0.1, 1.5)["classification"],
            "positive_behavioral_go",
        )
        self.assertEqual(
            classify_gate(-1.1, -1.5, -0.1)["classification"],
            "negative_behavioral_go",
        )
        self.assertEqual(
            classify_gate(0.0, -0.4, 0.4)["classification"],
            "null_equivalence_go",
        )
        self.assertEqual(
            classify_gate(0.6, -0.1, 0.9)["classification"],
            "inconclusive_no_go",
        )

    def test_classification_metrics(self) -> None:
        logits = np.asarray(
            [[8.0, 0.0], [0.0, 8.0], [0.0, 8.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 1, 0], dtype=np.int64)
        metrics = classification_metrics(logits, labels)
        self.assertAlmostEqual(metrics["accuracy"], 2.0 / 3.0)
        self.assertGreater(metrics["cross_entropy"], 0.0)
        self.assertGreater(metrics["brier_score"], 0.0)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
