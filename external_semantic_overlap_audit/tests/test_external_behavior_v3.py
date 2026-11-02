from __future__ import annotations

import hashlib
import json
import unittest

from toolkit.external_behavior.common import arm_assignment_bit
from toolkit.external_behavior_v3.design import (
    PERMUTATIONS,
    allocate_targets,
    expand_source_row,
    parse_sentences,
    select_pairs,
)


def source_row() -> dict[str, object]:
    story = (
        "entity_1 is entity_0's sister. "
        "entity_1 has a daughter called entity_2. "
        "entity_2 has a sister named entity_3."
    )
    query = "fill in the blanks : entity_3 is entity_0 's _."
    target = "neice"
    edge_types = ["sister", "daughter", "sister"]
    canonical = json.dumps(
        [target, *edge_types],
        separators=(",", ":"),
    )
    semantic_id = hashlib.sha256(canonical.encode()).hexdigest()
    normalized_input = f"{story} <sep> {query}"
    surface_id = hashlib.sha256(
        json.dumps(
            [normalized_input, target],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "edge_types": edge_types,
        "normalized_input": normalized_input,
        "normalized_query": query,
        "normalized_story": story,
        "query_edge": [0, 3],
        "row_order_sha256": "0" * 64,
        "semantic_id": semantic_id,
        "semantic_serialization": canonical,
        "story_edges": [[0, 1], [1, 2], [2, 3]],
        "surface_id": surface_id,
        "target_relation": target,
    }


class ExternalBehaviorV3Test(unittest.TestCase):
    def test_parser_and_permutations(self) -> None:
        source = source_row()
        sentences = parse_sentences(str(source["normalized_story"]))
        self.assertEqual(len(sentences), 3)
        expanded = expand_source_row(source)
        self.assertEqual(len(expanded), 6)
        self.assertEqual(
            [tuple(row["permutation"]) for row in expanded],
            list(PERMUTATIONS),
        )
        self.assertEqual(
            len({row["normalized_input"] for row in expanded}),
            6,
        )
        self.assertEqual(
            {row["semantic_id"] for row in expanded},
            {source["semantic_id"]},
        )
        self.assertEqual(
            {row["source_group_id"] for row in expanded},
            {source["surface_id"]},
        )

    def test_source_group_allocation_is_disjoint(self) -> None:
        base = expand_source_row(source_row())
        semantic_id = str(base[0]["semantic_id"])
        groups = {}
        for group_index in range(45):
            group_id = f"{group_index:064x}"
            groups[group_id] = [
                {
                    **row,
                    "source_group_id": group_id,
                    "surface_id": hashlib.sha256(
                        f"{group_id}:{row['permutation_index']}".encode()
                    ).hexdigest(),
                }
                for row in base
            ]
        pairs = [
            {
                "member_0": semantic_id,
                "member_1": semantic_id,
                "pair_index": 0,
                "target_relation": "neice",
            }
        ]
        targets, evaluation, exposure = allocate_targets(
            {semantic_id: groups},
            pairs,
        )
        self.assertEqual(len(targets), 2)
        self.assertEqual(len(evaluation), 400)
        self.assertEqual(len(exposure), 128)
        for target in targets:
            self.assertEqual(len(target["evaluation_row_ids"]), 200)
            self.assertEqual(len(target["exposure_row_ids"]), 64)
            self.assertEqual(
                len(target["evaluation_source_group_ids"]),
                34,
            )
            self.assertEqual(
                len(target["exposure_source_group_ids"]),
                11,
            )
            self.assertFalse(
                set(target["evaluation_source_group_ids"])
                & set(target["exposure_source_group_ids"])
            )

    def test_pair_selection_is_same_label_and_deterministic(self) -> None:
        eligible = [
            {
                "complete_source_groups": 45,
                "semantic_id": f"{index:064x}",
                "target_relation": "aunt",
                "unique_expanded_rows": 270,
            }
            for index in range(40)
        ]
        pairs = select_pairs(eligible)
        self.assertEqual(len(pairs), 20)
        self.assertEqual(
            [pair["pair_sha256"] for pair in pairs],
            sorted(pair["pair_sha256"] for pair in pairs),
        )
        self.assertTrue(
            all(pair["target_relation"] == "aunt" for pair in pairs)
        )

    def test_arm_assignment_is_complementary(self) -> None:
        for seed in range(8):
            for pair_index in range(20):
                bit = arm_assignment_bit(seed, pair_index)
                self.assertIn(bit, (0, 1))
                member_a = bit
                member_b = 1 - bit
                self.assertNotEqual(member_a, member_b)


if __name__ == "__main__":
    unittest.main()
