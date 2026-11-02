from __future__ import annotations

import hashlib
import json
import unittest

from toolkit.external_behavior.mathematics import canonical_polynomial
from toolkit.external_behavior_v2.collision_census import (
    analyze_module,
    verify_symbolic_zero,
)


def row(
    module: str,
    regime: str,
    index: int,
    question: str,
    answer: str,
) -> dict[str, object]:
    canonical, semantic_id = canonical_polynomial(module, question)
    row_id = hashlib.sha256(
        json.dumps(
            [regime, module, index, question, answer],
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return {
        "answer": answer,
        "canonical_srepr": canonical,
        "module": module,
        "question": question,
        "regime": regime,
        "row_id": row_id,
        "row_index": index,
        "semantic_id": semantic_id,
    }


class MathematicsCollisionCensusTest(unittest.TestCase):
    def test_alias_verbatim_no_match_and_multiplicity(self) -> None:
        module = "polynomials__expand"
        train = [
            row(module, "train", 0, "Expand (x + 1)*(x - 1).", "x**2 - 1"),
            row(module, "train", 1, "Expand x**2 - 1.", "x**2 - 1"),
            row(module, "train", 2, "Expand (x + 1)*(x - 1).", "x**2 - 1"),
        ]
        evaluation = [
            row(module, "interpolate", 0, "Expand (z - 1)*(z + 1).", "z**2 - 1"),
            row(module, "interpolate", 1, "Expand (x + 1)*(x - 1).", "x**2 - 1"),
            row(module, "interpolate", 2, "Expand z + 2.", "z + 2"),
        ]
        result = analyze_module(module, train, evaluation)
        summary = result["summary"]
        self.assertEqual(summary["canonical_semantic_overlap_row_count"], 2)
        self.assertEqual(summary["verbatim_question_overlap_row_count"], 1)
        self.assertEqual(summary["interpolate_row_denominator"], 3)
        self.assertEqual(summary["semantic_collision_witness_count"], 2)
        self.assertEqual(result["audit_rows"][0]["train_semantic_match_count"], 3)
        self.assertEqual(
            result["audit_rows"][0]["semantic_train_witness_row_index"],
            0,
        )
        self.assertFalse(result["audit_rows"][2]["semantic_overlap"])
        self.assertEqual(len(result["witnesses"]), 2)
        self.assertTrue(
            all(item["verifier_exact_zero"] for item in result["witnesses"])
        )

    def test_modules_are_not_pooled(self) -> None:
        expand_train = [
            row(
                "polynomials__expand",
                "train",
                0,
                "Expand x + 1.",
                "x + 1",
            )
        ]
        collect_eval = [
            row(
                "polynomials__collect",
                "interpolate",
                0,
                "Collect the terms in z + 1.",
                "z + 1",
            )
        ]
        result = analyze_module(
            "polynomials__collect",
            [],
            collect_eval,
        )
        self.assertEqual(
            result["summary"]["canonical_semantic_overlap_row_count"],
            0,
        )
        self.assertNotEqual(
            expand_train[0]["semantic_id"],
            collect_eval[0]["semantic_id"],
        )

    def test_symbolic_verifier_accepts_alpha_alias(self) -> None:
        self.assertTrue(
            verify_symbolic_zero(
                "polynomials__expand",
                "Expand (x + 1)*(x - 1).",
                "Expand z**2 - 1.",
            )
        )

    def test_symbolic_verifier_rejects_non_equivalence(self) -> None:
        self.assertFalse(
            verify_symbolic_zero(
                "polynomials__expand",
                "Expand x + 1.",
                "Expand z + 2.",
            )
        )


if __name__ == "__main__":
    unittest.main()
