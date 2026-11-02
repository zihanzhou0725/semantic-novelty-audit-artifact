"""Core evidence model for the versioned semantic-audit toolkit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "semantic-audit-output-v1"
TOOL_VERSION = "1.0.0"
COMPLETENESS_VALUES = ("exact", "sound_lower_bound", "unresolved")


class AuditError(ValueError):
    """Raised when evidence violates a toolkit invariant."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_id(namespace: str, canonical_semantics: Any) -> str:
    payload = {
        "canonical_semantics": canonical_semantics,
        "namespace": namespace,
    }
    return sha256_bytes(canonical_json_bytes(payload))


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise AuditError(f"Expected CSV boolean, got {value!r}")


def make_assessment(
    *,
    claim_scope: str,
    equivalence_witness_type: str,
    completeness_status: str,
    overlap: bool | None,
    witness_count: int | None,
    details: dict[str, Any],
) -> dict[str, Any]:
    if completeness_status not in COMPLETENESS_VALUES:
        raise AuditError(f"Unknown completeness status: {completeness_status}")
    if completeness_status == "exact" and not isinstance(overlap, bool):
        raise AuditError("Exact evidence requires a Boolean overlap decision.")
    if completeness_status == "sound_lower_bound" and overlap is not True:
        raise AuditError("Sound lower-bound evidence can only establish overlap.")
    if completeness_status == "unresolved" and overlap is not None:
        raise AuditError("Unresolved evidence cannot assert overlap or novelty.")
    if witness_count is not None and witness_count < 0:
        raise AuditError("Witness count cannot be negative.")
    return {
        "claim_scope": claim_scope,
        "completeness_status": completeness_status,
        "details": details,
        "equivalence_witness_type": equivalence_witness_type,
        "overlap": overlap,
        "witness_count": witness_count,
    }


def make_overlap_summary(
    *,
    claim_scope: str,
    denominator: int,
    witnessed_overlap_count: int,
    overlap_lower_bound: int,
    overlap_upper_bound: int,
    unresolved_count: int,
    unit: str,
) -> dict[str, Any]:
    values = (
        denominator,
        witnessed_overlap_count,
        overlap_lower_bound,
        overlap_upper_bound,
        unresolved_count,
    )
    if any(value < 0 for value in values):
        raise AuditError("Overlap summary counts cannot be negative.")
    if not (
        witnessed_overlap_count <= overlap_upper_bound <= denominator
        and overlap_lower_bound <= overlap_upper_bound
        and unresolved_count <= denominator
    ):
        raise AuditError("Incoherent overlap summary bounds.")
    return {
        "claim_scope": claim_scope,
        "denominator": denominator,
        "overlap_lower_bound": overlap_lower_bound,
        "overlap_upper_bound": overlap_upper_bound,
        "unit": unit,
        "unresolved_count": unresolved_count,
        "witnessed_overlap_count": witnessed_overlap_count,
    }


def build_output(
    *,
    adapter: dict[str, Any],
    benchmark: dict[str, Any],
    observation_contract: dict[str, Any],
    source_revisions_and_hashes: dict[str, Any],
    identity_records: dict[str, Any],
    semantic_ids: dict[str, Any],
    witness_types: list[str],
    completeness_status: str,
    overlap_summaries: list[dict[str, Any]],
    behavioral_metadata: dict[str, Any] | None,
    claim_gate: dict[str, Any],
) -> dict[str, Any]:
    if completeness_status not in COMPLETENESS_VALUES:
        raise AuditError(f"Unknown output completeness: {completeness_status}")
    return {
        "adapter": adapter,
        "behavioral_estimand_metadata_if_present": behavioral_metadata,
        "benchmark": benchmark,
        "claim_gate_and_allowed_wording": claim_gate,
        "completeness_status": completeness_status,
        "equivalence_witness_type": sorted(set(witness_types)),
        "observation_contract": observation_contract,
        "overlap_counts_and_denominators": overlap_summaries,
        "schema_version": SCHEMA_VERSION,
        "semantic_ids": semantic_ids,
        "source_revisions_and_hashes": source_revisions_and_hashes,
        "tool_version": TOOL_VERSION,
        "train_and_evaluation_identity_records": identity_records,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
