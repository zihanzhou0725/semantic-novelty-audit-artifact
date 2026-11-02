"""Independent materialization validator for external-behavior-v2."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sympy

from toolkit.external_behavior.common import iter_jsonl

from .common import (
    DATA_DIR,
    MANIFEST_DIR,
    MATHEMATICS_ROOT,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    V1_MANIFEST_DIR,
    read_json,
    sha256_file,
    verify_source_manifest,
    write_json,
)
from .materialize import validate_implementation_freeze
from .mathematics import (
    MAX_ASSERTION_DROPS,
    RECOVERABLE_LINE,
    RECOVERABLE_RELATIVE_PATH,
)


EXPECTED_STREAMS = {
    ("train", "polynomials__expand"): 100_000,
    ("train", "polynomials__collect"): 100_000,
    ("interpolate", "polynomials__expand"): 10_000,
    ("interpolate", "polynomials__collect"): 10_000,
}
V1_COMPLETE_FILES = {
    "mathematics_train_polynomials__collect.jsonl":
        "9607ee4372974c6141da16f4548345eb67f783cb29d5b561e752038510bc556e",
    "mathematics_interpolate_polynomials__expand.jsonl":
        "3d86b69cefd2f1ea37e67463f9dd166c216cc7b76a3284e5d8b6dc99a1745a9c",
    "mathematics_interpolate_polynomials__collect.jsonl":
        "55c8c06aa61cdfaa13c22a463fa818c74a44573090456ae3b82db9f70bedab05",
}
V1_PREFIX_ROWS = 67_937
V1_PREFIX_SHA256 = (
    "9a492db08770a7f2266cc54a3b29ae7a088a531045d7e5b127b0c41db08abc6b"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="ascii").splitlines()]


def _checksum_status(directory: Path) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for line in _read_lines(directory / "SHA256SUMS"):
        expected, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256_file(path) != expected:
            failures.append(f"hash:{relative}")
    return not failures, failures


def _independent_signature(
    module_name: str,
    question: str,
) -> tuple[str, str]:
    prefixes = {
        "polynomials__expand": "Expand ",
        "polynomials__collect": "Collect the terms in ",
    }
    prefix = prefixes[module_name]
    if not question.startswith(prefix) or not question.endswith("."):
        raise ValueError("Question does not match the frozen parser.")
    expression = sympy.sympify(question[len(prefix) : -1], evaluate=False)
    source_symbols = sorted(expression.free_symbols, key=str)
    normalized_symbols = [
        sympy.Symbol(f"x{index}") for index in range(len(source_symbols))
    ]
    normalized = expression.subs(
        dict(zip(source_symbols, normalized_symbols)),
        simultaneous=True,
    )
    polynomial = sympy.Poly(sympy.expand(normalized), *normalized_symbols)
    canonical = sympy.srepr(
        sympy.Tuple(
            *[
                sympy.Tuple(
                    sympy.Tuple(
                        *[sympy.Integer(power) for power in monomial]
                    ),
                    sympy.Rational(coefficient),
                )
                for monomial, coefficient in polynomial.terms()
            ]
        )
    )
    semantic_id = hashlib.sha256(
        f"{module_name}\x1f{canonical}".encode("ascii")
    ).hexdigest()
    return canonical, semantic_id


def _row_id(
    regime: str,
    module_name: str,
    row_index: int,
    question: str,
    answer: str,
) -> str:
    payload = json.dumps(
        [regime, module_name, row_index, question, answer],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _prefix_hash(path: Path, line_count: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as handle:
        while observed < line_count:
            line = handle.readline()
            if not line:
                break
            digest.update(line)
            observed += 1
    return digest.hexdigest(), observed


def _stage_checklist(checks: list[dict[str, Any]]) -> str:
    values = {item["name"]: item["pass"] for item in checks}
    entries = [
        ("Freeze V2 implementation hashes.", "implementation_freeze"),
        ("Freeze Python and dependencies.", "environment_lock"),
        ("Verify pinned source revision and hashes.", "source_validation"),
        ("Generate all 220,000 retained rows.", "row_counts"),
        ("Validate every anomaly against the registered assertion.", "anomalies"),
        ("Keep assertion drops below the per-stream cap.", "anomaly_cap"),
        ("Match all three V1-complete streams byte-for-byte.", "complete_compatibility"),
        ("Match the V1 train/expand 67,937-row prefix.", "prefix_compatibility"),
        ("Independently regenerate all semantic IDs.", "signatures"),
        ("Verify contiguous row indices and row IDs.", "row_integrity"),
        ("Freeze and validate all artifact hashes.", "data_hashes"),
        ("Keep overlap join and aggregate release sealed.", "aggregate_blindness"),
    ]
    lines = [
        "# External Behavioral Validation v2 Materialization Gate",
        "",
    ]
    for label, name in entries:
        lines.append(f"- [{'x' if values.get(name, False) else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_materialization() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    implementation = validate_implementation_freeze()
    check("implementation_freeze", implementation["status"] == "frozen")
    current_lock = sorted(
        line.strip()
        for line in subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
        ).splitlines()
        if line.strip()
    )
    frozen_lock = sorted(_read_lines(MANIFEST_DIR / "requirements.lock"))
    environment = read_json(MANIFEST_DIR / "environment.json")
    check(
        "environment_lock",
        current_lock == frozen_lock
        and environment["executable"] == sys.executable,
    )
    source = verify_source_manifest()
    check("source_validation", source["status"] == "pass")
    data_hash_ok, data_hash_failures = _checksum_status(DATA_DIR)
    check("data_hashes", data_hash_ok, data_hash_failures)

    materialization = read_json(
        MANIFEST_DIR / "materialization_manifest.json"
    )
    stream_report = {
        (stream["regime"], stream["module"]): stream
        for stream in materialization["mathematics"]["streams"]
    }
    observed_counts: dict[str, int] = {}
    signature_failures = 0
    row_integrity_failures = 0
    anomaly_failures: list[str] = []
    anomaly_counts: dict[str, int] = {}
    anomaly_cap_ok = True

    for (regime, module_name), expected_count in EXPECTED_STREAMS.items():
        stem = f"mathematics_{regime}_{module_name}"
        path = DATA_DIR / f"{stem}.jsonl"
        count = 0
        for count, row in enumerate(iter_jsonl(path), start=1):
            try:
                canonical, semantic_id = _independent_signature(
                    module_name,
                    row["question"],
                )
                if (
                    canonical != row["canonical_srepr"]
                    or semantic_id != row["semantic_id"]
                ):
                    signature_failures += 1
                if (
                    row["row_index"] != count - 1
                    or row["regime"] != regime
                    or row["module"] != module_name
                    or row["row_id"]
                    != _row_id(
                        regime,
                        module_name,
                        count - 1,
                        row["question"],
                        row["answer"],
                    )
                ):
                    row_integrity_failures += 1
            except Exception:
                signature_failures += 1
            if count % 20_000 == 0:
                print(f"Validate v2 {regime}/{module_name} {count}", flush=True)
        observed_counts[f"{regime}:{module_name}"] = count
        if count != expected_count:
            row_integrity_failures += 1

        anomaly_path = DATA_DIR / f"{stem}.anomalies.jsonl"
        anomalies = list(iter_jsonl(anomaly_path))
        anomaly_counts[f"{regime}:{module_name}"] = len(anomalies)
        if len(anomalies) > MAX_ASSERTION_DROPS:
            anomaly_cap_ok = False
        for index, anomaly in enumerate(anomalies):
            if (
                anomaly["anomaly_index"] != index
                or anomaly["exception_class"] != "AssertionError"
                or anomaly["exception_message"] != ""
                or anomaly["regime"] != regime
                or anomaly["module"] != module_name
                or anomaly["terminal_frame_relative_path"]
                != RECOVERABLE_RELATIVE_PATH
                or anomaly["terminal_frame_line"] != RECOVERABLE_LINE
                or not 0
                <= anomaly["next_retained_row_index"]
                < expected_count
            ):
                anomaly_failures.append(f"{regime}:{module_name}:{index}")
        report = stream_report[(regime, module_name)]
        if (
            report["generator_assertion_drops"] != len(anomalies)
            or report["manifest_sha256"] != sha256_file(path)
            or report["anomaly_log_sha256"] != sha256_file(anomaly_path)
        ):
            anomaly_failures.append(f"report:{regime}:{module_name}")

    check(
        "row_counts",
        sum(observed_counts.values()) == 220_000
        and all(
            observed_counts[f"{regime}:{module}"] == expected
            for (regime, module), expected in EXPECTED_STREAMS.items()
        ),
        observed_counts,
    )
    check("signatures", signature_failures == 0, signature_failures)
    check(
        "row_integrity",
        row_integrity_failures == 0,
        row_integrity_failures,
    )
    check("anomalies", not anomaly_failures, {
        "counts": anomaly_counts,
        "failures": anomaly_failures,
    })
    check("anomaly_cap", anomaly_cap_ok, anomaly_counts)

    compatibility_failures: list[str] = []
    v1_data = V1_MANIFEST_DIR / ".data.build"
    for filename, expected_hash in V1_COMPLETE_FILES.items():
        v1_path = v1_data / filename
        v2_path = DATA_DIR / filename
        if (
            sha256_file(v1_path) != expected_hash
            or sha256_file(v2_path) != expected_hash
        ):
            compatibility_failures.append(filename)
    check(
        "complete_compatibility",
        not compatibility_failures,
        compatibility_failures,
    )
    prefix_hash, prefix_rows = _prefix_hash(
        DATA_DIR / "mathematics_train_polynomials__expand.jsonl",
        V1_PREFIX_ROWS,
    )
    check(
        "prefix_compatibility",
        prefix_rows == V1_PREFIX_ROWS
        and prefix_hash == V1_PREFIX_SHA256
        and sha256_file(
            v1_data / "mathematics_train_polynomials__expand.jsonl.tmp"
        )
        == V1_PREFIX_SHA256,
        {"rows": prefix_rows, "sha256": prefix_hash},
    )

    overlap_named_files = [
        path.name
        for path in DATA_DIR.iterdir()
        if "overlap" in path.name.lower() or "collision" in path.name.lower()
    ]
    check(
        "aggregate_blindness",
        materialization["aggregate_overlap_released"] is False
        and materialization["join_or_collision_census_executed"] is False
        and not overlap_named_files,
        overlap_named_files,
    )
    run_root = PROJECT_ROOT / "runs" / PROTOCOL_ID
    check(
        "no_behavioral_training",
        not run_root.exists() or not any(run_root.iterdir()),
    )

    failed = [item for item in checks if not item["pass"]]
    report = {
        "aggregate_overlap_released": False,
        "anomaly_counts": anomaly_counts,
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": [item["name"] for item in failed],
        "failed_count": len(failed),
        "materialization_gate": "pass" if not failed else "no_go",
        "protocol_id": PROTOCOL_ID,
        "row_counts": observed_counts,
        "schema_version": "external-behavior-v2-validation-v1",
        "status": "pass" if not failed else "no_go",
        "validated_at_utc": _now(),
    }
    checklist_path = MANIFEST_DIR / "materialization_checklist.md"
    checklist_path.write_text(_stage_checklist(checks), encoding="ascii")
    report_path = MANIFEST_DIR / "materialization_validation_report.json"
    write_json(report_path, report)
    freeze_paths = [
        MANIFEST_DIR / "IMPLEMENTATION_SHA256SUMS",
        MANIFEST_DIR / "environment.json",
        MANIFEST_DIR / "implementation_manifest.json",
        MANIFEST_DIR / "materialization_manifest.json",
        checklist_path,
        report_path,
        MANIFEST_DIR / "requirements.lock",
        DATA_DIR / "SHA256SUMS",
    ]
    (MANIFEST_DIR / "MATERIALIZATION_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  "
            f"{path.relative_to(MANIFEST_DIR).as_posix()}\n"
            for path in freeze_paths
        ),
        encoding="ascii",
    )
    return report
