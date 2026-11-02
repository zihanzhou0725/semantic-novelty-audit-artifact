from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import yaml

from toolkit.external_behavior.clutrr import (
    normalize_text,
    reconstruct_semantics,
    select_pairs,
)
from toolkit.external_behavior.common import derive_seed
from toolkit.external_behavior.mathematics import canonical_polynomial
from toolkit.external_behavior.model import (
    CLS_ID,
    SEP_ID,
    encode_normalized_input,
    inspect_architecture,
)


class ExternalBehaviorMaterializationTest(unittest.TestCase):
    def test_seed_derivation_matches_preregistration(self) -> None:
        namespace = "clutrr:generator:python"
        expected = int.from_bytes(
            hashlib.sha256(
                f"external-behavior-v1:{namespace}".encode("ascii")
            ).digest()[:8],
            "big",
        ) & ((1 << 63) - 1)
        self.assertEqual(derive_seed(namespace), expected)

    def test_clutrr_normalization_and_reconstruction(self) -> None:
        story = "[Alice] is [Bob]'s mother.  [Bob] is [Carol]'s brother."
        self.assertEqual(
            normalize_text(story, ["Alice", "Bob", "Carol"]),
            "entity_0 is entity_1's mother. entity_1 is entity_2's brother.",
        )
        row = {
            "edge_types": ["mother", "brother", "son"],
            "query_edge": [0, 3],
            "story_edges": [[0, 1], [1, 2], [2, 3]],
            "target_relation": "aunt",
        }
        canonical, semantic_id = reconstruct_semantics(row)
        self.assertEqual(canonical, '["aunt","mother","brother","son"]')
        self.assertEqual(
            semantic_id,
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def test_pair_selection_is_same_label_and_hash_ordered(self) -> None:
        eligible = [
            {
                "available_rows": 264,
                "semantic_id": value,
                "target_relation": label,
            }
            for label, value in (
                ("a", "0" * 64),
                ("a", "1" * 64),
                ("b", "2" * 64),
                ("b", "3" * 64),
            )
        ]
        pairs = select_pairs(eligible)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(
            all(pair["member_0"] < pair["member_1"] for pair in pairs)
        )
        self.assertEqual(
            [pair["pair_sha256"] for pair in pairs],
            sorted(pair["pair_sha256"] for pair in pairs),
        )

    def test_math_signature_is_alpha_invariant(self) -> None:
        first = canonical_polynomial(
            "polynomials__expand",
            "Expand (x + 1)*(x - 1).",
        )
        second = canonical_polynomial(
            "polynomials__expand",
            "Expand (z - 1)*(z + 1).",
        )
        self.assertEqual(first, second)

    def test_registered_clutrr_surface_capacity_warning(self) -> None:
        root = Path(__file__).resolve().parents[1]
        store = root / "upstream" / "external_behavior_v1_clutrr" / "clutrr" / "store"
        relations = yaml.safe_load(
            (store / "relations_store.yaml").read_text(encoding="ascii")
        )
        questions = yaml.safe_load(
            (store / "question_store.yaml").read_text(encoding="ascii")
        )
        maximum_templates = max(
            len(value["p"])
            for relation in relations.values()
            for value in relation.values()
        )
        maximum_unique_surfaces = (
            maximum_templates**3 * len(questions["relational"])
        )
        self.assertEqual(maximum_unique_surfaces, 162)
        self.assertLess(maximum_unique_surfaces, 200 + 64)

    def test_tokenizer_and_model_match_registered_shape(self) -> None:
        tokens = encode_normalized_input("entity_0 parent. <sep> relation?")
        self.assertEqual(tokens[0], CLS_ID)
        self.assertEqual(tokens.count(SEP_ID), 1)
        inspection = inspect_architecture(23)
        self.assertTrue(inspection["checks"]["layers"])
        self.assertTrue(inspection["checks"]["attention_heads"])
        self.assertEqual(inspection["num_classes"], 23)


if __name__ == "__main__":
    unittest.main()
