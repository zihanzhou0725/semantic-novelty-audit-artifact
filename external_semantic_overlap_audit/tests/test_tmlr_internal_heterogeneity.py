"""Tests for the internal finite-function heterogeneity analysis."""

from __future__ import annotations

import unittest
from pathlib import Path

from toolkit.tmlr_strengthening.internal_heterogeneity import analyze_internal


ROOT = Path(__file__).resolve().parents[1]
FUNCTION_EFFECTS = (
    ROOT
    / "analysis"
    / "quality-enhancement-v1"
    / "p1"
    / "function_effects.csv"
)


class InternalHeterogeneityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.table, cls.summary, cls.groups = analyze_internal(FUNCTION_EFFECTS)

    def test_real_data_descriptive_anchors(self) -> None:
        self.assertEqual(len(self.table), 40)
        self.assertAlmostEqual(
            self.summary["effect_mean_pp"],
            1.832,
            delta=0.005,
        )
        self.assertAlmostEqual(
            self.summary["effect_sd_pp"],
            0.970,
            delta=0.005,
        )
        self.assertAlmostEqual(
            self.summary["effect_median_pp"],
            1.80,
            delta=0.01,
        )
        self.assertAlmostEqual(
            self.summary["effect_min_pp"],
            0.133,
            delta=0.005,
        )
        self.assertAlmostEqual(
            self.summary["effect_max_pp"],
            4.48,
            delta=0.005,
        )
        self.assertEqual(self.summary["positive_effect_count"], 40)

    def test_real_data_headroom_and_normalized_gain_anchors(self) -> None:
        self.assertAlmostEqual(
            self.summary["effect_baseline_spearman"],
            -0.727,
            delta=0.005,
        )
        self.assertAlmostEqual(
            self.summary["baseline_mean"],
            0.7679,
            delta=0.0005,
        )
        self.assertAlmostEqual(
            self.summary["norm_gain_mean"],
            0.0767,
            delta=0.0005,
        )
        self.assertAlmostEqual(
            self.summary["norm_gain_median"],
            0.0768,
            delta=0.0005,
        )

    def test_group_anchors(self) -> None:
        self.assertAlmostEqual(
            self.groups["multiplier"]["1"]["effect_mean_pp"],
            1.507,
            delta=0.005,
        )
        self.assertAlmostEqual(
            self.groups["multiplier"]["11"]["effect_mean_pp"],
            2.157,
            delta=0.005,
        )
        affine_parity = self.groups["alias_signature"]["affine+parity"]
        self.assertEqual(affine_parity["count"], 38)
        self.assertAlmostEqual(
            affine_parity["effect_mean_pp"],
            1.874,
            delta=0.005,
        )


if __name__ == "__main__":
    unittest.main()
