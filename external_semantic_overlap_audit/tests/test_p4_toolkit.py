#!/usr/bin/env python3
"""Unit canaries and adapter integration tests for P4."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXTERNAL_ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_ROOT = EXTERNAL_ROOT / "toolkit"
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from semantic_audit.adapters import build_adapter_output  # noqa: E402
from semantic_audit.core import (  # noqa: E402
    TOOL_VERSION,
    AuditError,
    build_output,
    make_assessment,
    make_overlap_summary,
    semantic_id,
)
from semantic_audit.schema import validate_audit_output  # noqa: E402


class EvidenceCanaries(unittest.TestCase):
    def test_exact_function_canary(self) -> None:
        affine_surface = {"family": "affine", "params": [1, 0]}
        lookup_surface = {"family": "lookup", "values": [0, 1, 2, 3]}
        canonical_outputs = [0, 1, 2, 3]
        affine_id = semantic_id("function_outputs_v1", canonical_outputs)
        lookup_id = semantic_id("function_outputs_v1", canonical_outputs)

        self.assertNotEqual(affine_surface, lookup_surface)
        self.assertEqual(affine_id, lookup_id)
        assessment = make_assessment(
            claim_scope="exact_function",
            equivalence_witness_type="complete_truth_table",
            completeness_status="exact",
            overlap=True,
            witness_count=4,
            details={"domain_size": 4},
        )
        self.assertIs(assessment["overlap"], True)

    def test_sound_alias_canary(self) -> None:
        assessment = make_assessment(
            claim_scope="sound_alias",
            equivalence_witness_type="executable_subset_witness",
            completeness_status="sound_lower_bound",
            overlap=True,
            witness_count=1,
            details={"unsearched_aliases_remain": True},
        )
        self.assertEqual(
            assessment["completeness_status"],
            "sound_lower_bound",
        )
        with self.assertRaises(AuditError):
            make_assessment(
                claim_scope="sound_alias",
                equivalence_witness_type="invalid_negative_lower_bound",
                completeness_status="sound_lower_bound",
                overlap=False,
                witness_count=0,
                details={},
            )

    def test_unresolved_canary(self) -> None:
        assessment = make_assessment(
            claim_scope="unresolved",
            equivalence_witness_type="search_without_completeness_proof",
            completeness_status="unresolved",
            overlap=None,
            witness_count=0,
            details={"novelty_not_established": True},
        )
        self.assertIsNone(assessment["overlap"])
        with self.assertRaises(AuditError):
            make_assessment(
                claim_scope="unresolved",
                equivalence_witness_type="invalid_novelty_claim",
                completeness_status="unresolved",
                overlap=False,
                witness_count=0,
                details={},
            )


class SchemaCanaries(unittest.TestCase):
    def fixture(self) -> dict:
        assessment = make_assessment(
            claim_scope="exact_function",
            equivalence_witness_type="complete_truth_table",
            completeness_status="exact",
            overlap=True,
            witness_count=1,
            details={},
        )
        return build_output(
            adapter={
                "id": "canary_v1",
                "semantic_representation": "finite truth table",
                "version": TOOL_VERSION,
            },
            benchmark={
                "candidate_id": "canary",
                "name": "Canary",
                "public": True,
                "scope": "unit test",
            },
            observation_contract={
                "admissible_inputs_and_contexts": (
                    "Two-input finite truth tables in the canary fixture."
                ),
                "observable_outputs": "Complete output-vector equality.",
                "termination_and_errors": (
                    "Finite evaluation terminates; malformed fixtures fail."
                ),
                "effects": "The canary functions are effect-free.",
                "nondeterminism": "The canary functions are deterministic.",
                "resource_observations": (
                    "Runtime and memory are excluded from semantic identity."
                ),
                "semantic_unit_adequacy": {
                    "advertised_claim": "Exact finite-function identity.",
                    "must_match_rule": (
                        "Representations with equal complete output vectors "
                        "must share an identity."
                    ),
                    "must_differ_rule": (
                        "Different complete output vectors must remain "
                        "distinct."
                    ),
                    "sensitivity_policy": (
                        "No alternative unit is admitted for this canary."
                    ),
                },
            },
            source_revisions_and_hashes={
                "code_revision": "canary",
                "dataset_revision": "canary",
                "files": [{"path": "canary", "sha256": "0" * 64}],
                "paper_revision": "canary",
            },
            identity_records={
                "evaluation": {
                    "completeness_status": "exact",
                    "definition": "one canary",
                    "record_count": 1,
                    "records": [
                        {
                            "assessments": [assessment],
                            "record_id": "canary:0",
                            "semantic_ids": {
                                "function": semantic_id(
                                    "function_outputs_v1",
                                    [0, 1],
                                )
                            },
                            "split": "test",
                            "surface_identity": {"name": "identity"},
                        }
                    ],
                    "records_materialization": "full",
                },
                "train": {
                    "completeness_status": "exact",
                    "definition": "one canary",
                    "record_count": 1,
                    "records": [],
                    "records_materialization": "summary_only",
                },
            },
            semantic_ids={
                "canonicalization": "canonical output vector",
                "namespaces": ["function"],
            },
            witness_types=["complete_truth_table"],
            completeness_status="exact",
            overlap_summaries=[
                make_overlap_summary(
                    claim_scope="exact_function",
                    denominator=1,
                    witnessed_overlap_count=1,
                    overlap_lower_bound=1,
                    overlap_upper_bound=1,
                    unresolved_count=0,
                    unit="functions",
                )
            ],
            behavioral_metadata=None,
            claim_gate={
                "allowed_wording": ["Exact overlap established."],
                "decision": "passed",
                "forbidden_wording": ["Unresolved means novel."],
            },
        )

    def test_schema_accepts_valid_output(self) -> None:
        validate_audit_output(self.fixture())

    def test_schema_rejects_missing_required_field(self) -> None:
        output = self.fixture()
        del output["semantic_ids"]
        with self.assertRaises(AuditError):
            validate_audit_output(output)

    def test_schema_rejects_missing_observation_contract(self) -> None:
        output = self.fixture()
        del output["observation_contract"]
        with self.assertRaises(AuditError):
            validate_audit_output(output)

    def test_schema_rejects_unresolved_boolean_decision(self) -> None:
        output = self.fixture()
        assessment = output["train_and_evaluation_identity_records"][
            "evaluation"
        ]["records"][0]["assessments"][0]
        assessment["completeness_status"] = "unresolved"
        with self.assertRaises(AuditError):
            validate_audit_output(output)

    def test_schema_rejects_full_record_count_mismatch(self) -> None:
        output = self.fixture()
        output["train_and_evaluation_identity_records"]["evaluation"][
            "record_count"
        ] = 2
        with self.assertRaises(AuditError):
            validate_audit_output(output)


class PublicAdapterTests(unittest.TestCase):
    def test_tensor_logic_adapter(self) -> None:
        output = build_adapter_output("tensor-logic", EXTERNAL_ROOT)
        validate_audit_output(output)
        summaries = {
            row["claim_scope"]: row
            for row in output["overlap_counts_and_denominators"]
        }
        self.assertEqual(
            summaries["endpoint_conditioned_two_hop_semantics"][
                "overlap_lower_bound"
            ],
            484,
        )
        self.assertEqual(
            summaries["ordered_relation_composition_schema"][
                "overlap_lower_bound"
            ],
            999,
        )

    def test_occam_markov_adapter(self) -> None:
        output = build_adapter_output("occam-markov", EXTERNAL_ROOT)
        validate_audit_output(output)
        self.assertEqual(
            [
                (
                    row["overlap_lower_bound"],
                    row["overlap_upper_bound"],
                    row["denominator"],
                )
                for row in output["overlap_counts_and_denominators"]
            ],
            [(2, 4, 20), (2, 4, 20)],
        )

    def test_unknown_adapter_rejected(self) -> None:
        with self.assertRaises(AuditError):
            build_adapter_output("missing", EXTERNAL_ROOT)


if __name__ == "__main__":
    unittest.main()
