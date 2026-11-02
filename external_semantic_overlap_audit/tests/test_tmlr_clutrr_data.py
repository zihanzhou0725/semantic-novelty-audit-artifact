"""Integrity checks for the semantic-versus-surface CLUTRR allocation."""

from __future__ import annotations

from collections import Counter
import unittest

from toolkit.tmlr_strengthening.clutrr_tmlr_data import (
    TRAIN_ROWS,
    build_t6_design,
    shared_edge_count,
)


class SemanticSurfaceAllocationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = build_t6_design()

    def test_decoys_satisfy_similarity_and_exclusion_rules(self) -> None:
        common_ids = set(self.design["common_surface_ids"])
        forbidden_semantic_ids = set(self.design["forbidden_semantic_ids"])
        for record in self.design["allocation"]:
            for row in record["decoy_rows"]:
                self.assertEqual(
                    row["target_relation"],
                    record["target_relation"],
                )
                self.assertNotEqual(
                    row["semantic_id"],
                    record["semantic_id"],
                )
                self.assertNotIn(row["semantic_id"], forbidden_semantic_ids)
                self.assertNotIn(row["surface_id"], common_ids)
                overlap = shared_edge_count(
                    record["edge_types"],
                    row["edge_types"],
                )
                self.assertGreaterEqual(overlap, 1)
                if record["tier"] == 2:
                    self.assertGreaterEqual(overlap, 2)

    def test_decoys_are_globally_unique(self) -> None:
        row_ids = [
            row["surface_id"]
            for record in self.design["allocation"]
            for row in record["decoy_rows"]
        ]
        self.assertEqual(len(row_ids), len(set(row_ids)))

    def test_arms_have_equal_size_and_label_counts(self) -> None:
        arms = self.design["arm_rows"]
        self.assertEqual(set(arms), {"C0", "C1", "C2"})
        counts = {
            arm: Counter(row["target_relation"] for row in rows)
            for arm, rows in arms.items()
        }
        for rows in arms.values():
            self.assertEqual(len(rows), TRAIN_ROWS)
        self.assertEqual(counts["C0"], counts["C1"])
        self.assertEqual(counts["C0"], counts["C2"])

    def test_each_arm_uses_unique_surface_rows(self) -> None:
        for arm, rows in self.design["arm_rows"].items():
            row_ids = [row["surface_id"] for row in rows]
            self.assertEqual(
                len(row_ids),
                len(set(row_ids)),
                msg=f"duplicate surface row in {arm}",
            )


if __name__ == "__main__":
    unittest.main()
