"""Independent materialization-gate validator for external-behavior-v1."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sympy

from .clutrr import (
    CANDIDATE_CAP,
    COMMON_BASE_ROWS,
    EVALUATION_ROWS_PER_PATTERN,
    EXPOSURE_ROWS_PER_PATTERN,
    PAIR_COUNT,
    ROWS_PER_ARM,
    TARGET_PATTERN_COUNT,
    eligible_patterns,
    normalize_text,
    reconstruct_semantics,
    select_pairs,
)
from .common import (
    ARMS,
    DATA_DIR,
    MANIFEST_DIR,
    MODEL_SEEDS,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    arm_assignment_bit,
    iter_jsonl,
    read_json,
    sha256_file,
    verify_source_manifest,
    write_json,
)
from .materialize import validate_implementation_freeze
from .model import encode_normalized_input, inspect_architecture


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="ascii").splitlines()]


def _validate_checksum_file(directory: Path) -> tuple[bool, list[str]]:
    failures: list[str] = []
    path = directory / "SHA256SUMS"
    if not path.is_file():
        return False, ["missing SHA256SUMS"]
    for line in _read_lines(path):
        expected, relative = line.split("  ", 1)
        target = directory / relative
        if not target.is_file():
            failures.append(f"missing:{relative}")
        elif sha256_file(target) != expected:
            failures.append(f"hash:{relative}")
    return not failures, failures


def _independent_math_signature(
    module_name: str,
    question: str,
) -> tuple[str, str]:
    prefixes = {
        "polynomials__expand": "Expand ",
        "polynomials__collect": "Collect the terms in ",
    }
    prefix = prefixes[module_name]
    if not question.startswith(prefix) or question[-1:] != ".":
        raise ValueError("Question does not match its registered parser.")
    expression = sympy.sympify(question[len(prefix) : -1], evaluate=False)
    source_symbols = sorted(expression.free_symbols, key=lambda value: str(value))
    normalized_symbols = [
        sympy.Symbol(f"x{index}") for index in range(len(source_symbols))
    ]
    normalized = expression.subs(
        dict(zip(source_symbols, normalized_symbols)),
        simultaneous=True,
    )
    polynomial = sympy.Poly(sympy.expand(normalized), *normalized_symbols)
    terms = []
    for monomial, coefficient in polynomial.terms():
        terms.append(
            sympy.Tuple(
                sympy.Tuple(*[sympy.Integer(value) for value in monomial]),
                sympy.Rational(coefficient),
            )
        )
    canonical = sympy.srepr(sympy.Tuple(*terms))
    semantic_id = hashlib.sha256(
        f"{module_name}\x1f{canonical}".encode("ascii")
    ).hexdigest()
    return canonical, semantic_id


def _markdown_checklist(checks: list[dict[str, Any]]) -> str:
    by_name = {item["name"]: item["pass"] for item in checks}
    entries = [
        ("Freeze adapter and model implementation hashes.", "implementation_hashes"),
        ("Freeze Python and dependency environment.", "environment_lock"),
        ("Verify pinned upstream source hashes.", "source_hashes"),
        ("Generate exactly 120,000 CLUTRR candidate rows.", "clutrr_raw_count"),
        ("Reconstruct every CLUTRR semantic signature.", "clutrr_reconstruction"),
        ("Verify at least 20 eligible same-label pattern pairs.", "clutrr_pair_capacity"),
        ("Freeze exactly 40 target patterns and 20 pairs.", "clutrr_targets"),
        ("Freeze 8,000 evaluation rows.", "clutrr_evaluation"),
        ("Freeze 64 exposure rows per target pattern.", "clutrr_exposure"),
        ("Freeze 22,720 common base rows without target IDs.", "clutrr_common_base"),
        ("Verify A/B size and label equality for all seeds.", "clutrr_arms"),
        ("Verify evaluation/training surface disjointness.", "clutrr_disjoint"),
        ("Verify no sequence truncation at 512 bytes.", "clutrr_lengths"),
        ("Freeze all CLUTRR manifests and SHA-256 values.", "data_hashes"),
        ("Generate 220,000 retained Mathematics examples.", "math_count"),
        ("Verify 100% Mathematics parser coverage.", "math_parser"),
        ("Verify deterministic Mathematics signature regeneration.", "math_signatures"),
        ("Freeze Mathematics manifests before aggregate release.", "math_blind_freeze"),
    ]
    lines = [
        "# External Behavioral Validation Materialization Gate",
        "",
        "This stage report supplements the immutable preregistration checklist.",
        "",
    ]
    for label, name in entries:
        mark = "x" if by_name.get(name, False) else " "
        lines.append(f"- [{mark}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_materialization() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    implementation = validate_implementation_freeze()
    check("implementation_hashes", True)
    current_freeze = sorted(
        line.strip()
        for line in subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
        ).splitlines()
        if line.strip()
    )
    locked_freeze = sorted(_read_lines(MANIFEST_DIR / "requirements.lock"))
    environment = read_json(MANIFEST_DIR / "environment.json")
    check(
        "environment_lock",
        current_freeze == locked_freeze
        and environment["executable"] == sys.executable,
        {
            "executable": sys.executable,
            "lock_matches": current_freeze == locked_freeze,
        },
    )
    source_validation = verify_source_manifest()
    check("source_hashes", source_validation["status"] == "pass")
    hash_status, hash_failures = _validate_checksum_file(DATA_DIR)
    check("data_hashes", hash_status, hash_failures)

    raw_path = DATA_DIR / "clutrr_candidates_raw.jsonl"
    raw_count = 0
    reconstruction_failures = 0
    normalization_failures = 0
    sequence_failures = 0
    for expected_index, row in enumerate(iter_jsonl(raw_path)):
        raw_count += 1
        if row["candidate_index"] != expected_index:
            reconstruction_failures += 1
        try:
            canonical, semantic_id = reconstruct_semantics(row)
            if (
                canonical != row["semantic_serialization"]
                or semantic_id != row["semantic_id"]
            ):
                reconstruction_failures += 1
            story = normalize_text(row["clean_story"], row["entity_names"])
            query = normalize_text(row["text_query"], row["entity_names"])
            if story != row["normalized_story"] or query != row["normalized_query"]:
                normalization_failures += 1
            encoded = encode_normalized_input(row["normalized_input"])
            if len(encoded) != row["sequence_length"]:
                sequence_failures += 1
        except Exception:
            reconstruction_failures += 1
    check("clutrr_raw_count", raw_count == CANDIDATE_CAP, raw_count)
    check(
        "clutrr_reconstruction",
        reconstruction_failures == 0 and normalization_failures == 0,
        {
            "normalization_failures": normalization_failures,
            "reconstruction_failures": reconstruction_failures,
        },
    )

    deduplicated = list(
        iter_jsonl(DATA_DIR / "clutrr_candidates_deduplicated.jsonl")
    )
    surface_ids = [row["surface_id"] for row in deduplicated]
    check(
        "clutrr_deduplication",
        len(surface_ids) == len(set(surface_ids))
        and deduplicated
        == sorted(
            deduplicated,
            key=lambda row: (row["row_order_sha256"], row["surface_id"]),
        ),
        len(deduplicated),
    )
    _, eligible = eligible_patterns(deduplicated)
    expected_pairs = select_pairs(eligible)
    observed_eligible = list(
        iter_jsonl(DATA_DIR / "clutrr_eligible_patterns.jsonl")
    )
    observed_pairs = list(iter_jsonl(DATA_DIR / "clutrr_pairs.jsonl"))
    check(
        "clutrr_pair_capacity",
        len(expected_pairs) == PAIR_COUNT,
        {
            "eligible_patterns": len(eligible),
            "same_label_pairs": len(expected_pairs),
        },
    )
    targets = list(iter_jsonl(DATA_DIR / "clutrr_targets.jsonl"))
    check(
        "clutrr_targets",
        observed_eligible == eligible
        and observed_pairs == expected_pairs
        and len(observed_pairs) == PAIR_COUNT
        and len(targets) == TARGET_PATTERN_COUNT
        and len({row["semantic_id"] for row in targets})
        == TARGET_PATTERN_COUNT,
        {"pairs": len(observed_pairs), "targets": len(targets)},
    )

    evaluation = list(iter_jsonl(DATA_DIR / "clutrr_evaluation.jsonl"))
    exposure = list(iter_jsonl(DATA_DIR / "clutrr_exposure.jsonl"))
    common = list(iter_jsonl(DATA_DIR / "clutrr_common_base.jsonl"))
    eval_counts = Counter(row["semantic_id"] for row in evaluation)
    exposure_counts = Counter(row["semantic_id"] for row in exposure)
    target_ids = {row["semantic_id"] for row in targets}
    check(
        "clutrr_evaluation",
        len(evaluation) == TARGET_PATTERN_COUNT * EVALUATION_ROWS_PER_PATTERN
        and set(eval_counts) == target_ids
        and all(
            value == EVALUATION_ROWS_PER_PATTERN
            for value in eval_counts.values()
        ),
        len(evaluation),
    )
    check(
        "clutrr_exposure",
        len(exposure) == TARGET_PATTERN_COUNT * EXPOSURE_ROWS_PER_PATTERN
        and set(exposure_counts) == target_ids
        and all(
            value == EXPOSURE_ROWS_PER_PATTERN
            for value in exposure_counts.values()
        ),
        len(exposure),
    )
    check(
        "clutrr_common_base",
        len(common) == COMMON_BASE_ROWS
        and not ({row["semantic_id"] for row in common} & target_ids),
        len(common),
    )

    eval_surfaces = {row["surface_id"] for row in evaluation}
    training_surfaces = {
        row["surface_id"] for row in exposure + common
    }
    check(
        "clutrr_disjoint",
        not (eval_surfaces & training_surfaces),
        len(eval_surfaces & training_surfaces),
    )
    allocated = evaluation + exposure + common
    max_length = max(
        (len(encode_normalized_input(row["normalized_input"])) for row in allocated),
        default=0,
    )
    check(
        "clutrr_lengths",
        sequence_failures == 0 and max_length <= 512,
        {"candidate_length_failures": sequence_failures, "maximum": max_length},
    )

    label_by_surface = {
        row["surface_id"]: row["target_relation"] for row in deduplicated
    }
    arm_failures: list[str] = []
    arm_rows: dict[tuple[int, str], list[str]] = {}
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            path = DATA_DIR / f"clutrr_train_seed{seed}_arm{arm}.txt"
            if not path.is_file():
                arm_failures.append(f"missing:{seed}:{arm}")
                continue
            ids = _read_lines(path)
            arm_rows[(seed, arm)] = ids
            if len(ids) != ROWS_PER_ARM:
                arm_failures.append(f"size:{seed}:{arm}:{len(ids)}")
            if any(value not in label_by_surface for value in ids):
                arm_failures.append(f"unknown_surface:{seed}:{arm}")
        if (seed, "A") not in arm_rows or (seed, "B") not in arm_rows:
            continue
        labels_a = Counter(label_by_surface[value] for value in arm_rows[(seed, "A")])
        labels_b = Counter(label_by_surface[value] for value in arm_rows[(seed, "B")])
        if labels_a != labels_b:
            arm_failures.append(f"label_balance:{seed}")
        if (
            arm_rows[(seed, "A")][:COMMON_BASE_ROWS]
            != arm_rows[(seed, "B")][:COMMON_BASE_ROWS]
        ):
            arm_failures.append(f"common_base:{seed}")
        sets = {
            arm: Counter(
                row["semantic_id"]
                for row in exposure
                if row["surface_id"] in set(arm_rows[(seed, arm)])
            )
            for arm in ARMS
        }
        for pair in observed_pairs:
            pair_index = int(pair["pair_index"])
            bit = arm_assignment_bit(seed, pair_index)
            expected_members = {"A": bit, "B": 1 - bit}
            for arm in ARMS:
                semantic_id = pair[f"member_{expected_members[arm]}"]
                other_id = pair[f"member_{1 - expected_members[arm]}"]
                if (
                    sets[arm][semantic_id] != EXPOSURE_ROWS_PER_PATTERN
                    or sets[arm][other_id] != 0
                ):
                    arm_failures.append(
                        f"strict_exposure:{seed}:{arm}:{pair_index}"
                    )
    check("clutrr_arms", not arm_failures, arm_failures[:100])

    architecture = inspect_architecture(
        len(implementation["model_output_classes"])
    )
    check(
        "model_architecture",
        architecture == implementation["architecture_inspection"],
        architecture,
    )

    expected_math = {
        ("train", "polynomials__expand"): 100_000,
        ("train", "polynomials__collect"): 100_000,
        ("interpolate", "polynomials__expand"): 10_000,
        ("interpolate", "polynomials__collect"): 10_000,
    }
    math_counts: dict[str, int] = {}
    math_parser_failures = 0
    math_signature_failures = 0
    for (regime, module_name), expected_count in expected_math.items():
        path = DATA_DIR / f"mathematics_{regime}_{module_name}.jsonl"
        count = 0
        for row in iter_jsonl(path):
            count += 1
            try:
                canonical, semantic_id = _independent_math_signature(
                    module_name,
                    row["question"],
                )
                if (
                    canonical != row["canonical_srepr"]
                    or semantic_id != row["semantic_id"]
                    or row["row_index"] != count - 1
                    or row["regime"] != regime
                    or row["module"] != module_name
                ):
                    math_signature_failures += 1
            except Exception:
                math_parser_failures += 1
        math_counts[f"{regime}:{module_name}"] = count
        if count != expected_count:
            math_signature_failures += 1
    check("math_count", sum(math_counts.values()) == 220_000, math_counts)
    check("math_parser", math_parser_failures == 0, math_parser_failures)
    check(
        "math_signatures",
        math_signature_failures == 0,
        math_signature_failures,
    )
    math_report = read_json(DATA_DIR / "mathematics_materialization_report.json")
    check(
        "math_blind_freeze",
        math_report["aggregate_overlap_released"] is False
        and math_report["join_or_collision_census_executed"] is False
        and sha256_file(DATA_DIR / "SHA256SUMS")
        == read_json(MANIFEST_DIR / "materialization_manifest.json")[
            "data_checksum_manifest_sha256"
        ],
    )

    run_root = PROJECT_ROOT / "runs" / PROTOCOL_ID
    check(
        "training_not_started",
        not run_root.exists() or not any(run_root.iterdir()),
    )
    clutrr_required = {
        "clutrr_raw_count",
        "clutrr_reconstruction",
        "clutrr_pair_capacity",
        "clutrr_targets",
        "clutrr_evaluation",
        "clutrr_exposure",
        "clutrr_common_base",
        "clutrr_arms",
        "clutrr_disjoint",
        "clutrr_lengths",
        "model_architecture",
        "data_hashes",
    }
    math_required = {
        "math_count",
        "math_parser",
        "math_signatures",
        "math_blind_freeze",
        "source_hashes",
        "environment_lock",
    }
    by_name = {item["name"]: item["pass"] for item in checks}
    clutrr_pass = all(by_name[name] for name in clutrr_required)
    math_pass = all(by_name[name] for name in math_required)
    report = {
        "aggregate_mathematics_overlap_released": False,
        "check_count": len(checks),
        "checks": checks,
        "clutrr_design_gate": "pass" if clutrr_pass else "no_go",
        "failed_checks": [
            item["name"] for item in checks if not item["pass"]
        ],
        "failed_count": sum(not item["pass"] for item in checks),
        "mathematics_materialization_gate": "pass" if math_pass else "no_go",
        "protocol_id": PROTOCOL_ID,
        "schema_version": "external-behavior-materialization-validation-v1",
        "status": "pass" if clutrr_pass and math_pass else "no_go",
        "training_started": False,
        "validated_at_utc": _now(),
    }
    checklist_path = MANIFEST_DIR / "materialization_checklist.md"
    checklist_path.write_text(_markdown_checklist(checks), encoding="ascii")
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
