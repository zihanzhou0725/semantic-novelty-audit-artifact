"""Synthetic tests for prospective directional V4 design."""

from __future__ import annotations

import unittest

from toolkit.external_behavior_v4.directional_design import (
    DESIGN_EFFECT_PP,
    analytic_power,
    calibrated_standard_error,
    remaining_pairs,
    select_block_gate,
    selected_pair_rows,
    wilson_interval,
)


class DirectionalV4DesignTest(unittest.TestCase):
    def test_wilson_interval_contains_observed_proportion(self) -> None:
        lower, upper = wilson_interval(900, 1000)
        self.assertLess(lower, 0.9)
        self.assertGreater(upper, 0.9)

    def test_analytic_power_increases_with_effect_and_pairs(self) -> None:
        se_20 = calibrated_standard_error(20)
        se_40 = calibrated_standard_error(40)
        self.assertLess(se_40, se_20)
        self.assertGreater(
            analytic_power(DESIGN_EFFECT_PP, se_40),
            analytic_power(DESIGN_EFFECT_PP, se_20),
        )

    def test_gate_selects_minimum_passing_block(self) -> None:
        rows = [
            {
                "block_count": 1,
                "design_scenario": True,
                "monte_carlo_power": 0.89,
                "pair_count": 20,
                "run_count": 16,
                "wilson_95_lower": 0.88,
            },
            {
                "block_count": 2,
                "design_scenario": True,
                "monte_carlo_power": 0.94,
                "pair_count": 40,
                "run_count": 32,
                "wilson_95_lower": 0.93,
            },
            {
                "block_count": 3,
                "design_scenario": True,
                "monte_carlo_power": 0.98,
                "pair_count": 60,
                "run_count": 48,
                "wilson_95_lower": 0.97,
            },
        ]
        gate = select_block_gate(rows)
        self.assertEqual(gate["decision"], "directional_v4_design_go")
        self.assertEqual(gate["selected_block_count"], 2)
        self.assertFalse(gate["training_released"])

    def test_gate_handles_no_go(self) -> None:
        gate = select_block_gate(
            [
                {
                    "block_count": 1,
                    "design_scenario": True,
                    "monte_carlo_power": 0.5,
                    "pair_count": 20,
                    "run_count": 16,
                    "wilson_95_lower": 0.49,
                }
            ]
        )
        self.assertEqual(gate["decision"], "directional_v4_design_no_go")
        self.assertIsNone(gate["selected_pair_count"])

    def test_pair_prefix_is_v3_disjoint(self) -> None:
        candidates = remaining_pairs()
        selected = selected_pair_rows(40)
        self.assertEqual(len(candidates), 139)
        self.assertEqual(len(selected), 40)
        self.assertEqual(
            [row["pair_sha256"] for row in selected],
            [row["pair_sha256"] for row in candidates[:40]],
        )
        self.assertEqual(
            {row["block_index"] for row in selected},
            {0, 1},
        )


if __name__ == "__main__":
    unittest.main()
