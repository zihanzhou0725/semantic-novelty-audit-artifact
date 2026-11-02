"""Synthetic tests for directional V4 materialization."""

from __future__ import annotations

import unittest

from toolkit.external_behavior_v4.materialize import (
    BLOCK_COUNT,
    COMMON_BASE_ROWS,
    MODEL_SEEDS,
    PAIRS_PER_BLOCK,
    TRAIN_ROWS,
    allocate_blocks,
    arm_assignment_bit,
    complete_groups,
    load_selected_pairs,
    v3_target_ids,
)


def synthetic_rows(semantic_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_index in range(45):
        group_id = f"{semantic_id}_group_{group_index:02d}"
        for permutation_index in range(6):
            rows.append(
                {
                    "permutation_index": permutation_index,
                    "semantic_id": semantic_id,
                    "source_group_id": group_id,
                    "surface_id": (
                        f"{semantic_id}_{group_index}_{permutation_index}"
                    ),
                    "target_relation": "sister",
                }
            )
    return rows


class V4MaterializationTest(unittest.TestCase):
    def test_allocation_uses_disjoint_group_ranges(self) -> None:
        candidates = synthetic_rows("member0") + synthetic_rows("member1")
        pair = {
            "block_index": 0,
            "block_pair_index": 0,
            "global_v4_pair_index": 0,
            "member_0": "member0",
            "member_1": "member1",
            "target_relation": "sister",
        }
        targets, evaluation, exposure = allocate_blocks(
            candidates,
            [pair],
        )
        self.assertEqual(len(targets), 2)
        self.assertEqual(len(evaluation[0]), 400)
        self.assertEqual(len(exposure[0]), 128)
        for target in targets:
            self.assertEqual(len(target["evaluation_row_ids"]), 200)
            self.assertEqual(len(target["exposure_row_ids"]), 64)
            self.assertFalse(
                set(target["evaluation_source_group_ids"])
                & set(target["exposure_source_group_ids"])
            )

    def test_complete_groups_rejects_incomplete_group(self) -> None:
        rows = synthetic_rows("member0")
        rows.pop()
        groups = complete_groups(rows)
        self.assertEqual(len(groups["member0"]), 44)

    def test_arm_assignment_is_deterministic(self) -> None:
        first = [
            arm_assignment_bit(block, seed, pair)
            for block in range(BLOCK_COUNT)
            for seed in MODEL_SEEDS
            for pair in range(PAIRS_PER_BLOCK)
        ]
        second = [
            arm_assignment_bit(block, seed, pair)
            for block in range(BLOCK_COUNT)
            for seed in MODEL_SEEDS
            for pair in range(PAIRS_PER_BLOCK)
        ]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {0, 1})

    def test_selected_targets_are_v3_disjoint(self) -> None:
        pairs = load_selected_pairs()
        selected = {
            pair[member]
            for pair in pairs
            for member in ("member_0", "member_1")
        }
        self.assertEqual(len(pairs), 40)
        self.assertEqual(len(selected), 80)
        self.assertFalse(selected & v3_target_ids())

    def test_row_arithmetic(self) -> None:
        self.assertEqual(COMMON_BASE_ROWS + 20 * 64, TRAIN_ROWS)


if __name__ == "__main__":
    unittest.main()
