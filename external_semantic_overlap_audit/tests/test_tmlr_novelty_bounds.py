"""Property and real-data tests for novelty-conditioned score bounds."""

from __future__ import annotations

import csv
import itertools
import random
import unittest
from pathlib import Path

from toolkit.tmlr_strengthening.novelty_bounds import (
    binary_closed_form,
    novelty_score_interval,
)


ROOT = Path(__file__).resolve().parents[1]
SLR_ITEMS = ROOT / "artifacts" / "slr_official_output_per_item.csv"


def exhaustive_interval(scores: list[float], upper_exposed: int) -> tuple[float, float]:
    means = []
    item_count = len(scores)
    for size in range(item_count - upper_exposed, item_count + 1):
        for subset in itertools.combinations(scores, size):
            means.append(sum(subset) / size)
    return min(means), max(means)


class NoveltyBoundsTest(unittest.TestCase):
    def test_interval_matches_exhaustive_enumeration(self) -> None:
        rng = random.Random(0)
        for case_index in range(300):
            item_count = rng.randint(2, 9)
            upper_exposed = rng.randint(0, item_count - 1)
            if case_index % 2:
                scores = [
                    rng.choice((0.0, 0.5, 1.0))
                    for _ in range(item_count)
                ]
            else:
                scores = [rng.random() for _ in range(item_count)]
            expected = exhaustive_interval(scores, upper_exposed)
            actual = novelty_score_interval(scores, upper_exposed)
            self.assertAlmostEqual(actual[0], expected[0], delta=1e-12)
            self.assertAlmostEqual(actual[1], expected[1], delta=1e-12)

    def test_binary_closed_form_matches_general_interval(self) -> None:
        for item_count in range(2, 15):
            for correct_count in range(item_count + 1):
                scores = [1.0] * correct_count + [0.0] * (
                    item_count - correct_count
                )
                for upper_exposed in range(item_count):
                    self.assertEqual(
                        binary_closed_form(
                            item_count,
                            correct_count,
                            upper_exposed,
                        ),
                        novelty_score_interval(scores, upper_exposed),
                    )

    def test_interval_widens_monotonically(self) -> None:
        scores = [0.0, 0.1, 0.4, 0.8, 1.0]
        intervals = [
            novelty_score_interval(scores, upper_exposed)
            for upper_exposed in range(len(scores))
        ]
        self.assertEqual(
            [lower for lower, _ in intervals],
            sorted((lower for lower, _ in intervals), reverse=True),
        )
        self.assertEqual(
            [upper for _, upper in intervals],
            sorted(upper for _, upper in intervals),
        )

    def test_slr_anchors(self) -> None:
        with SLR_ITEMS.open(newline="", encoding="utf-8") as handle:
            unresolved = [
                row
                for row in csv.DictReader(handle)
                if row["overlap_class"] == "unresolved_or_novel"
            ]
        self.assertEqual(len(unresolved), 548)
        exact = [
            float(row["is_correct"].lower() in ("1", "true"))
            for row in unresolved
        ]
        partial = [float(row["partial_score"]) for row in unresolved]
        exact_anchors = {
            0.00: (0.1898, 0.1898),
            0.05: (0.1478, 0.1996),
            0.10: (0.0994, 0.2110),
            0.20: (0.0000, 0.2374),
            0.30: (0.0000, 0.2708),
            0.50: (0.0000, 0.3796),
            0.70: (0.0000, 0.6341),
            0.90: (0.0000, 1.0000),
        }
        partial_anchors = {
            0.00: (0.7622, 0.7622),
            0.05: (0.7499, 0.7850),
            0.10: (0.7356, 0.8045),
            0.20: (0.7029, 0.8405),
            0.30: (0.6703, 0.8728),
            0.50: (0.5965, 0.9278),
        }
        for rho, expected in exact_anchors.items():
            upper_exposed = int(rho * len(exact) + 0.5)
            actual = novelty_score_interval(exact, upper_exposed)
            self.assertEqual(tuple(round(value, 4) for value in actual), expected)
        for rho, expected in partial_anchors.items():
            upper_exposed = int(rho * len(partial) + 0.5)
            actual = novelty_score_interval(partial, upper_exposed)
            self.assertEqual(tuple(round(value, 4) for value in actual), expected)

    def test_invalid_upper_exposed_is_rejected(self) -> None:
        for upper_exposed in (-1, 2):
            with self.assertRaisesRegex(ValueError, "0 <= U < M"):
                novelty_score_interval([0.0, 1.0], upper_exposed)


if __name__ == "__main__":
    unittest.main()
