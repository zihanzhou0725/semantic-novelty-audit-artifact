#!/usr/bin/env python3
"""Checks for the additional CLUTRR V4 cluster sensitivity."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "analyze_external_behavior_v4_cluster_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location("cluster_sensitivity", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load cluster-sensitivity implementation.")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ClusterSensitivityTests(unittest.TestCase):
    def test_frozen_result(self) -> None:
        results = MODULE.analyze(MODULE.read_rows())
        self.assertEqual(results["grid_validation"]["observation_count"], 640)
        self.assertEqual(results["grid_validation"]["semantic_id_count"], 80)
        self.assertTrue(
            math.isclose(
                results["effect_pp"],
                23.3390625,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                results["standard_error_pp"],
                2.6186837958668105,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        interval = results["confidence_interval_95"]
        self.assertTrue(
            math.isclose(
                interval["lower_pp"],
                17.146859289040293,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                interval["upper_pp"],
                29.53126571095971,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def test_two_way_identity(self) -> None:
        results = MODULE.analyze(MODULE.read_rows())
        components = results["variance_components"]
        expected = (
            components["model_seed"]["variance_pp2"]
            + components["pattern_pair"]["variance_pp2"]
            - components["seed_by_pair_intersection"]["variance_pp2"]
        )
        self.assertTrue(
            math.isclose(
                components["two_way_variance_pp2"],
                expected,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        self.assertGreater(components["two_way_variance_pp2"], 0.0)


if __name__ == "__main__":
    unittest.main()
