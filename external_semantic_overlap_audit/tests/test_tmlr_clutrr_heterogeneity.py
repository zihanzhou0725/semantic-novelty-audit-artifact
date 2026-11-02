"""Tests for the exploratory CLUTRR pattern-level heterogeneity analysis."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from toolkit.tmlr_strengthening.clutrr_heterogeneity import (
    analyze_clutrr,
    decode_semantic_id,
    eta_squared,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "external-behavior-v4" / "effect-analysis-v1"
CLASSES_PATH = (
    ROOT
    / "manifests"
    / "external-behavior-v4"
    / "execution-v1"
    / "classes.json"
)


class ClutrrHeterogeneityTest(unittest.TestCase):
    def test_eta_squared_matches_hand_calculation(self) -> None:
        values = np.asarray([1.0, 3.0, 5.0, 7.0])
        groups = ["a", "a", "b", "b"]
        self.assertAlmostEqual(eta_squared(values, groups), 0.8)

    def test_semantic_hash_decodes_known_edges(self) -> None:
        target = "aunt"
        edges = ("father", "daughter", "aunt")
        payload = json.dumps(
            [target, *edges],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        semantic_id = hashlib.sha256(payload).hexdigest()
        decoded = decode_semantic_id(
            semantic_id,
            target,
            ["aunt", "daughter", "father"],
        )
        self.assertEqual(decoded, edges)

    def test_real_data_matches_registered_anchors(self) -> None:
        table, summary, _ = analyze_clutrr(
            ANALYSIS_DIR / "pattern_effects.csv",
            ANALYSIS_DIR / "seed_pattern_effects.csv",
            CLASSES_PATH,
        )
        self.assertEqual(len(table), 80)
        self.assertEqual(summary["decoded_count"], 80)
        self.assertAlmostEqual(summary["effect_mean_pp"], 23.339, delta=0.01)
        self.assertAlmostEqual(
            summary["effect_baseline_spearman"],
            -0.847,
            delta=0.005,
        )
        self.assertEqual(summary["positive_effect_count"], 76)


if __name__ == "__main__":
    unittest.main()
