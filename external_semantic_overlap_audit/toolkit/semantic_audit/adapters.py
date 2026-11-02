"""Public-benchmark adapters for the common semantic-audit evidence model."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .core import (
    TOOL_VERSION,
    AuditError,
    build_output,
    make_assessment,
    make_overlap_summary,
    parse_bool,
    sha256_file,
)


TENSOR_CANDIDATE_ID = "7b1fa5fbdb1da4ee"
OCCAM_CANDIDATE_ID = "7b9ebd45a3528701"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise AuditError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _canonical_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _candidate(
    external_root: Path,
    candidate_id: str,
) -> tuple[dict[str, Any], Path]:
    manifest_path = (
        external_root
        / "protocols"
        / "quality-enhancement-v1"
        / "p3"
        / "candidate_manifest.json"
    )
    manifest = _read_json(manifest_path)
    matches = [
        item
        for item in manifest["candidates"]
        if item["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise AuditError(f"Candidate is not uniquely frozen: {candidate_id}")
    return matches[0], manifest_path


def _source_metadata(
    *,
    external_root: Path,
    candidate: dict[str, Any],
    p3_report: dict[str, Any],
    extra_paths: list[Path],
) -> dict[str, Any]:
    project_root = external_root.parent
    expected_hashes = dict(p3_report["input_sha256"])
    for path in extra_paths:
        expected_hashes[_relative(path, project_root)] = sha256_file(path)

    files: list[dict[str, str]] = []
    for relative_path, expected_hash in sorted(expected_hashes.items()):
        path = project_root / relative_path
        if not path.is_file():
            raise AuditError(f"Missing pinned source: {relative_path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise AuditError(f"Pinned source hash changed: {relative_path}")
        files.append({"path": relative_path, "sha256": actual_hash})

    return {
        "code_revision": _canonical_text(candidate["code_revision"]),
        "dataset_revision": _canonical_text(candidate["dataset_revision"]),
        "files": files,
        "paper_revision": candidate["paper_revision"],
    }


def _tensor_logic_record(row: dict[str, str]) -> dict[str, Any]:
    endpoint_witness = parse_bool(row["endpoint_path_witness_retained"])
    schema_witness = parse_bool(
        row["relation_schema_seen_in_scrubbed_train"]
    )
    direct_edge_overlap = parse_bool(row["direct_edge_in_scrubbed_train"])
    endpoint_count = int(row["executable_witness_count"])
    schema_count = int(row["relation_schema_training_witness_count"])

    endpoint_assessment = make_assessment(
        claim_scope="endpoint_conditioned_two_hop_semantics",
        equivalence_witness_type="executable_retained_two_hop_path",
        completeness_status="exact" if endpoint_witness else "unresolved",
        overlap=True if endpoint_witness else None,
        witness_count=endpoint_count,
        details={
            "intermediates": json.loads(
                row["executable_witness_intermediates_json"]
            ),
            "selected_path_retained": parse_bool(
                row["selected_path_retained"]
            ),
        },
    )
    schema_assessment = make_assessment(
        claim_scope="ordered_relation_composition_schema",
        equivalence_witness_type="training_relation_schema_instance",
        completeness_status=(
            "sound_lower_bound" if schema_witness else "unresolved"
        ),
        overlap=True if schema_witness else None,
        witness_count=schema_count,
        details={
            "schema_seen_in_other_eval_split": parse_bool(
                row["relation_schema_seen_in_other_eval_split"]
            )
        },
    )
    direct_assessment = make_assessment(
        claim_scope="direct_triple_identity",
        equivalence_witness_type="scrubbed_train_membership_census",
        completeness_status="exact",
        overlap=direct_edge_overlap,
        witness_count=int(direct_edge_overlap),
        details={
            "direct_edge_in_original_train": parse_bool(
                row["direct_edge_in_original_train"]
            )
        },
    )
    return {
        "assessments": [
            direct_assessment,
            endpoint_assessment,
            schema_assessment,
        ],
        "record_id": f"test:{int(row['eval_index'])}",
        "semantic_ids": {
            "endpoint_path": row["endpoint_path_signature_sha256"],
            "full_query": row["full_query_signature_sha256"],
            "relation_schema": row["relation_schema_signature_sha256"],
        },
        "split": "test",
        "surface_identity": {
            "head": row["head"],
            "relation_1": row["relation_1"],
            "relation_2": row["relation_2"],
            "selected_middle": row["selected_middle"],
            "tail": row["tail"],
            "target_relation": row["target_relation"],
        },
    }


def build_tensor_logic_output(external_root: Path) -> dict[str, Any]:
    candidate, manifest_path = _candidate(
        external_root,
        TENSOR_CANDIDATE_ID,
    )
    analysis_dir = (
        external_root
        / "analysis"
        / "quality-enhancement-v1"
        / "p3"
        / "tensor_logic_fb15k237"
    )
    report_path = analysis_dir / "audit_report.json"
    rows_path = analysis_dir / "eval_paths.csv"
    p3_report = _read_json(report_path)
    rows = [
        row for row in _read_csv(rows_path) if row["split"] == "test"
    ]
    if len(rows) != 1000:
        raise AuditError("Tensor Logic test denominator changed.")
    records = [_tensor_logic_record(row) for row in rows]

    endpoint_count = sum(
        assessment["overlap"] is True
        for record in records
        for assessment in record["assessments"]
        if assessment["claim_scope"]
        == "endpoint_conditioned_two_hop_semantics"
    )
    schema_count = sum(
        assessment["overlap"] is True
        for record in records
        for assessment in record["assessments"]
        if assessment["claim_scope"]
        == "ordered_relation_composition_schema"
    )
    direct_count = sum(
        assessment["overlap"] is True
        for record in records
        for assessment in record["assessments"]
        if assessment["claim_scope"] == "direct_triple_identity"
    )
    if (endpoint_count, schema_count, direct_count) != (484, 999, 0):
        raise AuditError("Tensor Logic common-schema census changed.")

    source_metadata = _source_metadata(
        external_root=external_root,
        candidate=candidate,
        p3_report=p3_report,
        extra_paths=[manifest_path, report_path, rows_path],
    )
    return build_output(
        adapter={
            "id": "tensor_logic_fb15k237_v1",
            "semantic_representation": (
                "knowledge-graph direct triples, endpoint-conditioned ordered "
                "two-hop paths, and relation-composition schemas"
            ),
            "version": TOOL_VERSION,
        },
        benchmark={
            "candidate_id": candidate["candidate_id"],
            "name": candidate["title"],
            "public": True,
            "scope": candidate["scope"],
        },
        observation_contract={
            "admissible_inputs_and_contexts": (
                "Frozen FB15k-237 validation/test queries evaluated against "
                "the final scrubbed training graph."
            ),
            "observable_outputs": (
                "Direct-triple membership, retained endpoint-conditioned "
                "two-hop paths, and ordered relation-schema presence."
            ),
            "termination_and_errors": (
                "Finite graph enumeration must terminate; missing, malformed, "
                "or hash-mismatched source artifacts fail the adapter."
            ),
            "effects": "The graph queries are effect-free.",
            "nondeterminism": (
                "No runtime nondeterminism remains after the graph and query "
                "manifests are frozen."
            ),
            "resource_observations": (
                "Runtime and memory are excluded from semantic identity."
            ),
            "semantic_unit_adequacy": {
                "advertised_claim": (
                    "Direct-edge, endpoint-path, and relation-schema novelty "
                    "are separate claim-specific objects."
                ),
                "must_match_rule": (
                    "Realizations with the same frozen claim-specific graph "
                    "identity must share a semantic ID."
                ),
                "must_differ_rule": (
                    "Different direct triples, endpoint-conditioned paths, or "
                    "ordered schemas remain distinct within their namespace."
                ),
                "sensitivity_policy": (
                    "Report endpoint-path and relation-schema overlap "
                    "separately; neither substitutes for the other."
                ),
            },
        },
        source_revisions_and_hashes=source_metadata,
        identity_records={
            "evaluation": {
                "completeness_status": "exact",
                "definition": (
                    "All 1,000 frozen test paths generated by Experiment 3B"
                ),
                "record_count": len(records),
                "records": records,
                "records_materialization": "full",
            },
            "train": {
                "completeness_status": "exact",
                "definition": (
                    "The 270,115 triples remaining after removing 2,000 "
                    "direct validation/test links from FB15k-237 train"
                ),
                "record_count": int(
                    p3_report["dataset"]["scrubbed_train_triples"]
                ),
                "records": [],
                "records_materialization": "summary_only",
            },
        },
        semantic_ids={
            "canonicalization": p3_report["semantic_definitions"][
                "full_query_signature"
            ],
            "namespaces": [
                "endpoint_path",
                "full_query",
                "relation_schema",
            ],
        },
        witness_types=[
            "executable_retained_two_hop_path",
            "scrubbed_train_membership_census",
            "training_relation_schema_instance",
        ],
        completeness_status="sound_lower_bound",
        overlap_summaries=[
            make_overlap_summary(
                claim_scope="direct_triple_identity",
                denominator=1000,
                witnessed_overlap_count=direct_count,
                overlap_lower_bound=direct_count,
                overlap_upper_bound=direct_count,
                unresolved_count=0,
                unit="test_paths",
            ),
            make_overlap_summary(
                claim_scope="endpoint_conditioned_two_hop_semantics",
                denominator=1000,
                witnessed_overlap_count=endpoint_count,
                overlap_lower_bound=endpoint_count,
                overlap_upper_bound=1000,
                unresolved_count=1000 - endpoint_count,
                unit="test_paths",
            ),
            make_overlap_summary(
                claim_scope="ordered_relation_composition_schema",
                denominator=1000,
                witnessed_overlap_count=schema_count,
                overlap_lower_bound=schema_count,
                overlap_upper_bound=1000,
                unresolved_count=1000 - schema_count,
                unit="test_paths",
            ),
        ],
        behavioral_metadata={
            "artifact_status": (
                "aggregate metrics reported; no checkpoint, per-item "
                "predictions, or training logs"
            ),
            "design": "reported single-seed training, not a controlled intervention",
            "estimand": "MRR and Hits@K on the generated compositional split",
            "reproducibility_gate": "failed",
        },
        claim_gate={
            "allowed_wording": [
                (
                    "Direct-edge withholding succeeded for all 1,000 test "
                    "paths."
                ),
                (
                    "Retained endpoint-conditioned two-hop witnesses establish "
                    "a 484/1,000 semantic-overlap lower bound."
                ),
                (
                    "Relation-composition schemas were witnessed for "
                    "999/1,000 test paths."
                ),
            ],
            "decision": "protocol_defect_detected_behavioral_no_go",
            "forbidden_wording": [
                "Every test path remains executable after direct-edge removal.",
                "No retained witness proves semantic novelty.",
                "The public artifacts establish a behavioral effect.",
            ],
        },
    )


def _occam_record(row: dict[str, str]) -> dict[str, Any]:
    in_support = parse_bool(row["nominal_train_seed_support"])
    occurrences = int(row["nominal_train_seed_occurrences_all_orders"])
    if in_support:
        assessment = make_assessment(
            claim_scope="exact_latent_transition_task_exposure",
            equivalence_witness_type="unseeded_order_assignment_candidate",
            completeness_status="unresolved",
            overlap=None,
            witness_count=occurrences,
            details={
                "eval_seed_in_training_seed_support": True,
                "pigeonhole_group": (
                    f"{row['experiment']}:seed:{row['eval_seed']}"
                ),
            },
        )
    else:
        assessment = make_assessment(
            claim_scope="exact_latent_transition_task_exposure",
            equivalence_witness_type="deterministic_seed_range_exclusion",
            completeness_status="exact",
            overlap=False,
            witness_count=0,
            details={"eval_seed_in_training_seed_support": False},
        )
    return {
        "assessments": [assessment],
        "record_id": (
            f"{row['experiment']}:order:{int(row['order'])}:"
            f"task:{int(row['eval_task_index'])}"
        ),
        "semantic_ids": {
            "markov_transition_tensor": row[
                "eval_task_signature_sha256"
            ]
        },
        "split": row["experiment"],
        "surface_identity": {
            "eval_seed": int(row["eval_seed"]),
            "eval_task_index": int(row["eval_task_index"]),
            "figure": int(row["figure"]),
            "order": int(row["order"]),
            "sequence_count": int(row["eval_sequences"]),
        },
    }


def _occam_bounds(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    by_experiment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_experiment[row["experiment"]].append(row)

    result: dict[str, dict[str, int]] = {}
    for experiment, experiment_rows in sorted(by_experiment.items()):
        supported = [
            row
            for row in experiment_rows
            if parse_bool(row["nominal_train_seed_support"])
        ]
        supported_by_seed = Counter(
            int(row["eval_seed"]) for row in supported
        )
        if set(supported_by_seed.values()) != {2}:
            raise AuditError("Occam supported seed does not cover two orders.")
        lower = len(supported_by_seed)
        upper = len(supported)
        result[experiment] = {
            "denominator": len(experiment_rows),
            "lower": lower,
            "unresolved": upper,
            "upper": upper,
        }
    return result


def build_occam_markov_output(external_root: Path) -> dict[str, Any]:
    candidate, manifest_path = _candidate(
        external_root,
        OCCAM_CANDIDATE_ID,
    )
    analysis_dir = (
        external_root
        / "analysis"
        / "quality-enhancement-v1"
        / "p3"
        / "occam_hierarchical_icl"
    )
    report_path = analysis_dir / "audit_report.json"
    rows_path = analysis_dir / "markov_task_manifest.csv"
    p3_report = _read_json(report_path)
    rows = _read_csv(rows_path)
    if len(rows) != 40:
        raise AuditError("Occam Markov evaluation denominator changed.")
    records = [_occam_record(row) for row in rows]
    bounds = _occam_bounds(rows)
    if bounds != {
        "figure_2_order_1_3": {
            "denominator": 20,
            "lower": 2,
            "unresolved": 4,
            "upper": 4,
        },
        "figure_3_order_1_2": {
            "denominator": 20,
            "lower": 2,
            "unresolved": 4,
            "upper": 4,
        },
    }:
        raise AuditError("Occam Markov collision bounds changed.")

    summaries: list[dict[str, Any]] = []
    for experiment, values in bounds.items():
        summaries.append(
            make_overlap_summary(
                claim_scope=(
                    f"exact_latent_transition_task_exposure:{experiment}"
                ),
                denominator=values["denominator"],
                witnessed_overlap_count=values["lower"],
                overlap_lower_bound=values["lower"],
                overlap_upper_bound=values["upper"],
                unresolved_count=values["unresolved"],
                unit="eval_latent_tasks",
            )
        )

    source_metadata = _source_metadata(
        external_root=external_root,
        candidate=candidate,
        p3_report=p3_report,
        extra_paths=[manifest_path, report_path, rows_path],
    )
    return build_output(
        adapter={
            "id": "occam_markov_transition_v1",
            "semantic_representation": (
                "Markov order plus exact float64 transition tensor generated "
                "from a deterministic task seed"
            ),
            "version": TOOL_VERSION,
        },
        benchmark={
            "candidate_id": candidate["candidate_id"],
            "name": candidate["title"],
            "public": True,
            "scope": (
                "Published Figure 2 and Figure 3 Markov-chain experiments; "
                "linear regression is explicitly outside this adapter"
            ),
        },
        observation_contract={
            "admissible_inputs_and_contexts": (
                "Frozen order-specific Markov evaluation tasks and the nominal "
                "online training seed support for the two main experiments."
            ),
            "observable_outputs": (
                "Exact float64 transition-tensor equality within each Markov "
                "order and grouped-seed exposure constraints."
            ),
            "termination_and_errors": (
                "Finite seed and signature censuses must terminate; missing "
                "or hash-mismatched source artifacts fail the adapter."
            ),
            "effects": "Transition-task generation is effect-free.",
            "nondeterminism": (
                "Unpersisted NumPy order assignments are treated as unresolved "
                "rather than reconstructed."
            ),
            "resource_observations": (
                "Runtime and memory are excluded from semantic identity."
            ),
            "semantic_unit_adequacy": {
                "advertised_claim": (
                    "Novelty concerns an order-specific transition task, not a "
                    "seed string or an aggregate Markov family."
                ),
                "must_match_rule": (
                    "Tasks with the same order and exact transition tensor "
                    "must share a semantic ID."
                ),
                "must_differ_rule": (
                    "Different transition tensors or different orders remain "
                    "distinct tasks."
                ),
                "sensitivity_policy": (
                    "Report exact task identities and grouped-seed bounds "
                    "separately when order assignments are unavailable."
                ),
            },
        },
        source_revisions_and_hashes=source_metadata,
        identity_records={
            "evaluation": {
                "completeness_status": "exact",
                "definition": (
                    "All 40 order-specific latent tasks from two main Markov "
                    "experiments, with exact transition-tensor hashes"
                ),
                "record_count": len(records),
                "records": records,
                "records_materialization": "full",
            },
            "train": {
                "completeness_status": "unresolved",
                "definition": (
                    "Two nominal 800,000-episode runs; seed=step+i is known, "
                    "but unseeded order assignments were not persisted"
                ),
                "record_count": 1_600_000,
                "records": [],
                "records_materialization": "summary_only",
            },
        },
        semantic_ids={
            "canonicalization": p3_report["markov_audit"][
                "semantic_signature"
            ],
            "namespaces": ["markov_transition_tensor"],
        },
        witness_types=[
            "deterministic_seed_range_exclusion",
            "pigeonhole_seed_group_bound",
            "unseeded_order_assignment_candidate",
        ],
        completeness_status="sound_lower_bound",
        overlap_summaries=summaries,
        behavioral_metadata={
            "artifact_status": (
                "aggregate plots reported; no checkpoints, run configs, "
                "training logs, or per-task predictions"
            ),
            "design": (
                "online task generation with unseeded category assignments"
            ),
            "estimand": (
                "KL divergence between model output and n-gram baselines"
            ),
            "reproducibility_gate": "failed",
        },
        claim_gate={
            "allowed_wording": [
                (
                    "Under the nominal 800k-sequence budget, exact latent-task "
                    "exposure is bounded between 2/20 and 4/20 per main "
                    "Markov experiment."
                ),
                (
                    "The identities of exposed order-specific tasks remain "
                    "unresolved because NumPy order assignments were unseeded."
                ),
            ],
            "decision": "markov_protocol_defect_detected",
            "forbidden_wording": [
                "All inference Markov chains were unseen during training.",
                "The Markov adapter completes the missing regression audit.",
                "Exact exposure caused the reported model behavior.",
            ],
        },
    )


ADAPTERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "occam-markov": build_occam_markov_output,
    "tensor-logic": build_tensor_logic_output,
}


def build_adapter_output(
    adapter_id: str,
    external_root: Path,
) -> dict[str, Any]:
    try:
        adapter = ADAPTERS[adapter_id]
    except KeyError as error:
        choices = ", ".join(sorted(ADAPTERS))
        raise AuditError(
            f"Unknown adapter {adapter_id!r}; choose from {choices}"
        ) from error
    return adapter(external_root.resolve())
