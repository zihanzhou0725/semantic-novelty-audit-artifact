"""Dependency-free validation for the toolkit's JSON Schema subset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .core import AuditError, canonical_json_bytes


DEFAULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "audit-output-v1.schema.json"
)


def _resolve_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise AuditError(f"Only local JSON Schema refs are supported: {reference}")
    value: Any = root
    for component in reference[2:].split("/"):
        key = component.replace("~1", "/").replace("~0", "~")
        value = value[key]
    if not isinstance(value, dict):
        raise AuditError(f"Schema ref does not resolve to an object: {reference}")
    return value


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    raise AuditError(f"Unsupported JSON Schema type: {expected}")


def _validate(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        _validate(value, _resolve_ref(root, schema["$ref"]), root, path)
        return

    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = (
            expected_type if isinstance(expected_type, list) else [expected_type]
        )
        if not any(_matches_type(value, item) for item in expected_types):
            raise AuditError(
                f"{path}: expected type {expected_types}, "
                f"got {type(value).__name__}"
            )

    if "const" in schema and value != schema["const"]:
        raise AuditError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise AuditError(f"{path}: value {value!r} is outside enum")

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise AuditError(f"{path}: missing required keys {missing}")
        properties = schema.get("properties", {})
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in properties:
                _validate(child, properties[key], root, child_path)
                continue
            additional = schema.get("additionalProperties", True)
            if additional is False:
                raise AuditError(f"{child_path}: additional property forbidden")
            if isinstance(additional, dict):
                _validate(child, additional, root, child_path)
        minimum = schema.get("minProperties")
        if minimum is not None and len(value) < minimum:
            raise AuditError(f"{path}: fewer than {minimum} properties")

    if isinstance(value, list):
        minimum = schema.get("minItems")
        if minimum is not None and len(value) < minimum:
            raise AuditError(f"{path}: fewer than {minimum} items")
        if schema.get("uniqueItems"):
            encoded = [canonical_json_bytes(item) for item in value]
            if len(encoded) != len(set(encoded)):
                raise AuditError(f"{path}: array items are not unique")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, child in enumerate(value):
                _validate(child, item_schema, root, f"{path}[{index}]")

    if isinstance(value, str):
        minimum = schema.get("minLength")
        if minimum is not None and len(value) < minimum:
            raise AuditError(f"{path}: string shorter than {minimum}")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            raise AuditError(f"{path}: string does not match {pattern!r}")

    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and "minimum" in schema
        and value < schema["minimum"]
    ):
        raise AuditError(f"{path}: value below minimum {schema['minimum']}")


def load_schema(path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise AuditError("Audit schema must be a JSON object.")
    return value


def validate_audit_output(
    output: dict[str, Any],
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> None:
    schema = load_schema(schema_path)
    _validate(output, schema, schema, "$")

    identities = output["train_and_evaluation_identity_records"]
    for split_name in ("train", "evaluation"):
        collection = identities[split_name]
        materialization = collection["records_materialization"]
        records = collection["records"]
        if materialization == "full" and collection["record_count"] != len(records):
            raise AuditError(
                f"$.train_and_evaluation_identity_records.{split_name}: "
                "full record count mismatch"
            )
        if materialization == "summary_only" and records:
            raise AuditError(
                f"$.train_and_evaluation_identity_records.{split_name}: "
                "summary-only collection must not contain records"
            )

    for record in identities["evaluation"]["records"]:
        for assessment in record["assessments"]:
            status = assessment["completeness_status"]
            overlap = assessment["overlap"]
            if status == "exact" and not isinstance(overlap, bool):
                raise AuditError("Exact assessment lacks a Boolean decision.")
            if status == "sound_lower_bound" and overlap is not True:
                raise AuditError("Sound lower bound does not establish overlap.")
            if status == "unresolved" and overlap is not None:
                raise AuditError("Unresolved assessment asserts a decision.")

    for summary in output["overlap_counts_and_denominators"]:
        denominator = summary["denominator"]
        lower = summary["overlap_lower_bound"]
        upper = summary["overlap_upper_bound"]
        witnessed = summary["witnessed_overlap_count"]
        unresolved = summary["unresolved_count"]
        if not (
            0 <= lower <= upper <= denominator
            and 0 <= witnessed <= upper
            and 0 <= unresolved <= denominator
        ):
            raise AuditError("Overlap summary bounds are incoherent.")
