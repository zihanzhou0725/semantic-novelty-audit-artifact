"""Portable semantic-audit toolkit."""

from .adapters import ADAPTERS, build_adapter_output
from .bounds import (
    ExposureConstraint,
    assignment_satisfies_constraints,
    grouped_seed_constraint,
    singleton_constraint,
    solve_disjoint_certificate_bounds,
)
from .core import (
    AuditError,
    SCHEMA_VERSION,
    TOOL_VERSION,
    canonical_json_bytes,
    semantic_id,
    sha256_file,
    write_json,
)
from .schema import validate_audit_output


__all__ = [
    "ADAPTERS",
    "AuditError",
    "ExposureConstraint",
    "SCHEMA_VERSION",
    "TOOL_VERSION",
    "assignment_satisfies_constraints",
    "build_adapter_output",
    "canonical_json_bytes",
    "grouped_seed_constraint",
    "semantic_id",
    "sha256_file",
    "singleton_constraint",
    "solve_disjoint_certificate_bounds",
    "validate_audit_output",
    "write_json",
]
