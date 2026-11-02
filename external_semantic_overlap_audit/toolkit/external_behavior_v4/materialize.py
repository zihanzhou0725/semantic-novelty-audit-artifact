"""Materialize and validate the frozen directional V4 block design."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from toolkit.external_behavior.common import (
    derive_seed,
    iter_jsonl,
    write_jsonl,
    write_lines,
)
from toolkit.external_behavior_v3.design import allocate_common_base
from toolkit.external_behavior_v3.effect_analysis import (
    DATA_DIR as V3_DATA_DIR,
    PROJECT_ROOT,
    read_csv,
    read_json,
    relative,
    sha256_file,
    write_csv,
    write_json,
)
from toolkit.external_behavior_v4.directional_design import (
    ANALYSIS_DIR as DIRECTIONAL_ANALYSIS_DIR,
    DESIGN_ID,
    MANIFEST_DIR as DIRECTIONAL_MANIFEST_DIR,
    PROTOCOL_DIR as DIRECTIONAL_PROTOCOL_DIR,
    verify_inputs as verify_directional_inputs,
    verify_protocol as verify_directional_protocol,
)


MATERIALIZATION_ID = "external-behavior-v4-materialization-v1"
PROTOCOL_DIR = (
    PROJECT_ROOT
    / "protocols"
    / "external-behavior-v4"
    / "materialization-v1"
)
MANIFEST_DIR = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "materialization-v1"
)
DATA_DIR = MANIFEST_DIR / "data"
RUN_ROOT = (
    PROJECT_ROOT
    / "runs"
    / "external-behavior-v4"
    / "directional-v1"
)
BLOCK_COUNT = 2
PAIRS_PER_BLOCK = 20
TARGETS_PER_BLOCK = 40
MODEL_SEEDS = tuple(range(8))
ARMS = ("A", "B")
EVALUATION_GROUPS = 34
EXPOSURE_GROUPS = 11
EVALUATION_ROWS_PER_TARGET = 200
EXPOSURE_ROWS_PER_TARGET = 64
COMMON_BASE_ROWS = 22_720
TRAIN_ROWS = 24_000
FIXED_PADDING_LENGTH = 185

IMPLEMENTATION_PATHS = (
    "scripts/freeze_external_behavior_v4_materialization.py",
    "scripts/materialize_external_behavior_v4.py",
    "scripts/validate_external_behavior_v4_materialization.py",
    "tests/test_external_behavior_v4_materialization.py",
    "toolkit/external_behavior/common.py",
    "toolkit/external_behavior_v3/design.py",
    "toolkit/external_behavior_v4/materialize.py",
    "protocols/external-behavior-v4/materialization-v1/validate.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checksum_entries(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line:
            digest, name = line.split("  ", 1)
            entries.append((digest, path.parent / name))
    return entries


def _input_paths() -> list[Path]:
    return sorted(
        [
            DIRECTIONAL_PROTOCOL_DIR / "SHA256SUMS",
            DIRECTIONAL_PROTOCOL_DIR / "input_manifest.json",
            DIRECTIONAL_PROTOCOL_DIR / "preregistration.json",
            DIRECTIONAL_ANALYSIS_DIR / "ANALYSIS_SHA256SUMS",
            DIRECTIONAL_ANALYSIS_DIR / "analysis_validation_report.json",
            DIRECTIONAL_ANALYSIS_DIR / "block_gate_report.json",
            DIRECTIONAL_MANIFEST_DIR / "DESIGN_SHA256SUMS",
            DIRECTIONAL_MANIFEST_DIR / "design_manifest.json",
            DIRECTIONAL_MANIFEST_DIR / "selected_pairs.csv",
            V3_DATA_DIR / "SHA256SUMS",
            V3_DATA_DIR / "clutrr_v3_candidates.jsonl",
            V3_DATA_DIR / "clutrr_v3_eligible_patterns.jsonl",
            V3_DATA_DIR / "clutrr_v3_pairs.jsonl",
        ],
        key=relative,
    )


def _implementation_hashes() -> dict[str, str]:
    missing = [
        name
        for name in IMPLEMENTATION_PATHS
        if not (PROJECT_ROOT / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing materialization code: {missing}")
    return {
        name: sha256_file(PROJECT_ROOT / name)
        for name in IMPLEMENTATION_PATHS
    }


def verify_protocol() -> None:
    for expected, path in _checksum_entries(PROTOCOL_DIR / "SHA256SUMS"):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(
                f"Materialization protocol checksum mismatch: {path}"
            )
    report = read_json(PROTOCOL_DIR / "validation_report.json")
    if (
        report.get("status") != "pass"
        or report.get("stage") != "pre_materialization"
        or report.get("failed_count") != 0
    ):
        raise RuntimeError("Pre-materialization validation did not pass.")


def verify_inputs() -> None:
    manifest = read_json(PROTOCOL_DIR / "input_manifest.json")
    for name, expected in manifest["input_sha256"].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen materialization input changed: {name}")
    for name, expected in manifest["implementation_sha256"].items():
        path = PROJECT_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen materialization code changed: {name}")


def validate_protocol(*, require_outputs_absent: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    prereg = read_json(PROTOCOL_DIR / "preregistration.json")
    design = read_json(DIRECTIONAL_MANIFEST_DIR / "design_manifest.json")
    design_validation = read_json(
        DIRECTIONAL_ANALYSIS_DIR / "analysis_validation_report.json"
    )
    check(
        "directional_design_gate",
        design_validation.get("status") == "pass"
        and design["status"] == "design_go_pending_materialization_gate"
        and design["training_released"] is False,
    )
    check(
        "block_arithmetic",
        prereg["frozen_design"]["block_count"] == BLOCK_COUNT
        and prereg["frozen_design"]["pairs_per_block"]
        == PAIRS_PER_BLOCK
        and prereg["frozen_design"]["run_count"] == 32,
    )
    check(
        "row_arithmetic",
        prereg["allocation"]["common_base_rows"] == COMMON_BASE_ROWS
        and prereg["allocation"]["training_rows_per_run"] == TRAIN_ROWS
        and prereg["allocation"]["evaluation_rows_per_target"]
        == EVALUATION_ROWS_PER_TARGET
        and prereg["allocation"]["exposure_rows_per_target"]
        == EXPOSURE_ROWS_PER_TARGET,
    )
    check(
        "source_group_policy",
        prereg["allocation"]["evaluation_source_groups"]
        == EVALUATION_GROUPS
        and prereg["allocation"]["exposure_source_groups"]
        == EXPOSURE_GROUPS,
    )
    check(
        "training_closed",
        prereg["gate"]["training_started_before_materialization"]
        is False
        and prereg["gate"]["success_action"]
        == "eligible_for_execution_protocol_freeze_not_training",
    )
    manifest_path = PROTOCOL_DIR / "input_manifest.json"
    receipt_path = PROTOCOL_DIR / "freeze_receipt.json"
    check(
        "freeze_artifacts",
        manifest_path.is_file() and receipt_path.is_file(),
    )
    input_failures: list[str] = []
    code_failures: list[str] = []
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        for name, expected in manifest["input_sha256"].items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                input_failures.append(name)
        for name, expected in manifest["implementation_sha256"].items():
            path = PROJECT_ROOT / name
            if not path.is_file() or sha256_file(path) != expected:
                code_failures.append(name)
    check("input_hashes", not input_failures, input_failures)
    check("implementation_hashes", not code_failures, code_failures)
    artifacts = (
        list(MANIFEST_DIR.rglob("*")) if MANIFEST_DIR.exists() else []
    )
    if require_outputs_absent:
        check(
            "outputs_absent",
            not artifacts and not RUN_ROOT.exists(),
            {
                "manifest_artifacts": [relative(path) for path in artifacts],
                "run_root_exists": RUN_ROOT.exists(),
            },
        )
    failed = [item for item in checks if not item["pass"]]
    document_names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "input_manifest.json",
        "preregistration.json",
        "validate.py",
    )
    return {
        "check_count": len(checks),
        "checks": checks,
        "document_sha256": {
            name: sha256_file(PROTOCOL_DIR / name)
            for name in document_names
        },
        "failed_count": len(failed),
        "materialization_id": MATERIALIZATION_ID,
        "schema_version": (
            "external-behavior-v4-materialization-protocol-validation-v1"
        ),
        "stage": (
            "pre_materialization"
            if require_outputs_absent
            else "post_materialization_recheck"
        ),
        "status": "pass" if not failed else "fail",
        "validated_at_utc": _now(),
    }


def freeze_protocol() -> dict[str, Any]:
    if (PROTOCOL_DIR / "input_manifest.json").exists():
        raise RuntimeError("Materialization protocol is already frozen.")
    if MANIFEST_DIR.exists() or RUN_ROOT.exists():
        raise RuntimeError("Materialization outputs or runs already exist.")
    verify_directional_protocol()
    verify_directional_inputs()
    design_validation = read_json(
        DIRECTIONAL_ANALYSIS_DIR / "analysis_validation_report.json"
    )
    if design_validation.get("status") != "pass":
        raise RuntimeError("Directional design integrity did not pass.")
    paths = _input_paths()
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing materialization inputs: {missing}")
    manifest = {
        "frozen_at_utc": _now(),
        "implementation_sha256": _implementation_hashes(),
        "input_sha256": {
            relative(path): sha256_file(path) for path in paths
        },
        "materialization_id": MATERIALIZATION_ID,
        "schema_version": (
            "external-behavior-v4-materialization-input-manifest-v1"
        ),
        "status": "frozen_before_materialization",
    }
    write_json(PROTOCOL_DIR / "input_manifest.json", manifest)
    receipt = {
        "frozen_at_utc": _now(),
        "materialization_id": MATERIALIZATION_ID,
        "schema_version": (
            "external-behavior-v4-materialization-freeze-receipt-v1"
        ),
        "state_at_freeze": {
            "materialization_outputs_absent": True,
            "run_root_absent": True,
            "training_started": False,
            "v4_design_gate_passed": True,
        },
        "status": "frozen_before_materialization",
    }
    write_json(PROTOCOL_DIR / "freeze_receipt.json", receipt)
    report = validate_protocol(require_outputs_absent=True)
    if report["status"] != "pass":
        raise RuntimeError(
            f"Pre-materialization protocol validation failed: {report}"
        )
    write_json(PROTOCOL_DIR / "validation_report.json", report)
    names = (
        "README.md",
        "checklist.md",
        "freeze_receipt.json",
        "input_manifest.json",
        "preregistration.json",
        "validate.py",
        "validation_report.json",
    )
    (PROTOCOL_DIR / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(PROTOCOL_DIR / name)}  {name}\n"
            for name in names
        ),
        encoding="ascii",
    )
    return {
        "input_manifest_sha256": sha256_file(
            PROTOCOL_DIR / "input_manifest.json"
        ),
        "materialization_id": MATERIALIZATION_ID,
        "status": receipt["status"],
        "validation_status": report["status"],
    }


def load_selected_pairs() -> list[dict[str, Any]]:
    rows = read_csv(DIRECTIONAL_MANIFEST_DIR / "selected_pairs.csv")
    pairs: list[dict[str, Any]] = []
    for row in rows:
        pairs.append(
            {
                "block_index": int(row["block_index"]),
                "block_pair_index": int(row["block_pair_index"]),
                "global_v4_pair_index": int(row["global_v4_pair_index"]),
                "member_0": row["member_0"],
                "member_0_complete_source_groups": int(
                    row["member_0_complete_source_groups"]
                ),
                "member_1": row["member_1"],
                "member_1_complete_source_groups": int(
                    row["member_1_complete_source_groups"]
                ),
                "pair_sha256": row["pair_sha256"],
                "target_relation": row["target_relation"],
            }
        )
    pairs.sort(key=lambda row: row["global_v4_pair_index"])
    if (
        len(pairs) != BLOCK_COUNT * PAIRS_PER_BLOCK
        or [row["global_v4_pair_index"] for row in pairs]
        != list(range(40))
    ):
        raise RuntimeError("Frozen selected-pair manifest changed.")
    return pairs


def load_candidates() -> list[dict[str, Any]]:
    return list(iter_jsonl(V3_DATA_DIR / "clutrr_v3_candidates.jsonl"))


def complete_groups(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[tuple[str, list[dict[str, Any]]]]]:
    grouped: dict[
        str,
        dict[str, list[dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["semantic_id"]][row["source_group_id"]].append(row)
    result: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    for semantic_id, source_groups in grouped.items():
        complete: list[tuple[str, list[dict[str, Any]]]] = []
        for group_id, group_rows in sorted(source_groups.items()):
            ordered = sorted(
                group_rows,
                key=lambda row: row["permutation_index"],
            )
            if (
                len(ordered) == 6
                and [row["permutation_index"] for row in ordered]
                == list(range(6))
            ):
                complete.append((group_id, ordered))
        result[semantic_id] = complete
    return result


def allocate_blocks(
    candidates: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
]:
    selected_ids = {
        pair[member]
        for pair in pairs
        for member in ("member_0", "member_1")
    }
    selected_rows = [
        row for row in candidates if row["semantic_id"] in selected_ids
    ]
    groups = complete_groups(selected_rows)
    targets: list[dict[str, Any]] = []
    evaluation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    exposure: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        block = pair["block_index"]
        block_pair = pair["block_pair_index"]
        for member in (0, 1):
            semantic_id = pair[f"member_{member}"]
            source_groups = groups.get(semantic_id, [])
            if len(source_groups) < 45:
                raise RuntimeError(
                    f"Selected target lacks 45 groups: {semantic_id}"
                )
            eval_groups = source_groups[:34]
            exposure_groups = source_groups[34:45]
            eval_rows = [
                row
                for _, group_rows in eval_groups[:33]
                for row in group_rows
            ] + eval_groups[33][1][:2]
            exposure_rows = [
                row
                for _, group_rows in exposure_groups[:10]
                for row in group_rows
            ] + exposure_groups[10][1][:4]
            target = {
                "block_index": block,
                "block_pair_index": block_pair,
                "evaluation_row_ids": [
                    row["surface_id"] for row in eval_rows
                ],
                "evaluation_source_group_ids": [
                    group_id for group_id, _ in eval_groups
                ],
                "exposure_row_ids": [
                    row["surface_id"] for row in exposure_rows
                ],
                "exposure_source_group_ids": [
                    group_id for group_id, _ in exposure_groups
                ],
                "global_v4_pair_index": pair["global_v4_pair_index"],
                "member_index": member,
                "semantic_id": semantic_id,
                "target_relation": pair["target_relation"],
            }
            targets.append(target)
            evaluation[block].extend(
                {
                    **row,
                    "allocation_index": allocation_index,
                    "block_index": block,
                    "block_pair_index": block_pair,
                    "global_v4_pair_index": pair[
                        "global_v4_pair_index"
                    ],
                    "pair_member": member,
                }
                for allocation_index, row in enumerate(eval_rows)
            )
            exposure[block].extend(
                {
                    **row,
                    "allocation_index": allocation_index,
                    "block_index": block,
                    "block_pair_index": block_pair,
                    "global_v4_pair_index": pair[
                        "global_v4_pair_index"
                    ],
                    "pair_member": member,
                }
                for allocation_index, row in enumerate(exposure_rows)
            )
    targets.sort(
        key=lambda row: (
            row["block_index"],
            row["block_pair_index"],
            row["member_index"],
        )
    )
    for block in range(BLOCK_COUNT):
        ordering = lambda row: (
            row["block_pair_index"],
            row["pair_member"],
            row["allocation_index"],
        )
        evaluation[block].sort(key=ordering)
        exposure[block].sort(key=ordering)
    return targets, dict(evaluation), dict(exposure)


def v3_target_ids() -> set[str]:
    result: set[str] = set()
    for row in iter_jsonl(V3_DATA_DIR / "clutrr_v3_pairs.jsonl"):
        result.add(row["member_0"])
        result.add(row["member_1"])
    return result


def arm_assignment_bit(
    block_index: int,
    model_seed: int,
    block_pair_index: int,
) -> int:
    namespace = (
        f"clutrr:v4:arm:{block_index}:{model_seed}:{block_pair_index}"
    )
    return derive_seed(namespace) & 1


def materialize_runs(
    common_base: list[dict[str, Any]],
    exposure: dict[int, list[dict[str, Any]]],
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    common_ids = [row["surface_id"] for row in common_base]
    common_labels = [row["target_relation"] for row in common_base]
    run_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for block in range(BLOCK_COUNT):
        block_pairs = [
            pair for pair in pairs if pair["block_index"] == block
        ]
        exposure_by_member: dict[
            tuple[int, int], list[dict[str, Any]]
        ] = defaultdict(list)
        for row in exposure[block]:
            exposure_by_member[
                (row["block_pair_index"], row["pair_member"])
            ].append(row)
        for rows in exposure_by_member.values():
            rows.sort(key=lambda row: row["allocation_index"])
        eval_path = DATA_DIR / f"evaluation_block{block}.jsonl"
        for model_seed in MODEL_SEEDS:
            labels_by_arm: dict[str, list[str]] = {}
            assignments_by_arm: dict[str, dict[str, int]] = {}
            for arm in ARMS:
                row_ids = list(common_ids)
                labels = list(common_labels)
                assignments: dict[str, int] = {}
                for pair in block_pairs:
                    pair_index = pair["block_pair_index"]
                    bit = arm_assignment_bit(
                        block,
                        model_seed,
                        pair_index,
                    )
                    member = bit if arm == "A" else 1 - bit
                    assignments[str(pair_index)] = member
                    selected = exposure_by_member[(pair_index, member)]
                    row_ids.extend(row["surface_id"] for row in selected)
                    labels.extend(
                        row["target_relation"] for row in selected
                    )
                filename = (
                    f"train_block{block}_seed{model_seed}_arm{arm}.txt"
                )
                path = DATA_DIR / filename
                write_lines(path, row_ids)
                run_id = f"block{block}_seed{model_seed}_arm{arm}"
                labels_by_arm[arm] = labels
                assignments_by_arm[arm] = assignments
                label_counts = dict(sorted(Counter(labels).items()))
                run_rows.append(
                    {
                        "arm": arm,
                        "block_index": block,
                        "common_base_sha256": sha256_file(
                            DATA_DIR / "common_base.jsonl"
                        ),
                        "evaluation_manifest": eval_path.name,
                        "evaluation_manifest_sha256": sha256_file(
                            eval_path
                        ),
                        "exposed_pair_members": json.dumps(
                            assignments,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "model_seed": model_seed,
                        "rows": len(row_ids),
                        "run_id": run_id,
                        "train_manifest": filename,
                        "train_manifest_sha256": sha256_file(path),
                    }
                )
                summaries.append(
                    {
                        "arm": arm,
                        "block_index": block,
                        "exposed_pair_members": assignments,
                        "label_counts": label_counts,
                        "model_seed": model_seed,
                        "rows": len(row_ids),
                        "run_id": run_id,
                    }
                )
            if Counter(labels_by_arm["A"]) != Counter(
                labels_by_arm["B"]
            ):
                raise RuntimeError("Paired V4 label counts differ.")
            for pair_index in range(PAIRS_PER_BLOCK):
                if (
                    assignments_by_arm["A"][str(pair_index)]
                    == assignments_by_arm["B"][str(pair_index)]
                ):
                    raise RuntimeError("V4 pair exposure is not complementary.")
    return run_rows, summaries


def _write_data_checksums() -> None:
    paths = sorted(
        (
            path
            for path in DATA_DIR.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        ),
        key=lambda path: path.name,
    )
    (DATA_DIR / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in paths
        ),
        encoding="ascii",
    )


def materialize() -> dict[str, Any]:
    verify_protocol()
    verify_inputs()
    if MANIFEST_DIR.exists() or RUN_ROOT.exists():
        raise RuntimeError("V4 materialization or training already exists.")
    pairs = load_selected_pairs()
    candidates = load_candidates()
    targets, evaluation, exposure = allocate_blocks(candidates, pairs)
    selected_target_ids = {row["semantic_id"] for row in targets}
    excluded_ids = selected_target_ids | v3_target_ids()
    common_base = allocate_common_base(candidates, excluded_ids)
    if len(common_base) != COMMON_BASE_ROWS:
        raise RuntimeError("Could not allocate the frozen common base.")

    DATA_DIR.mkdir(parents=True)
    write_jsonl(DATA_DIR / "targets.jsonl", targets)
    write_jsonl(DATA_DIR / "common_base.jsonl", common_base)
    for block in range(BLOCK_COUNT):
        write_jsonl(
            DATA_DIR / f"evaluation_block{block}.jsonl",
            evaluation[block],
        )
        write_jsonl(
            DATA_DIR / f"exposure_block{block}.jsonl",
            exposure[block],
        )
    run_rows, summaries = materialize_runs(
        common_base,
        exposure,
        pairs,
    )
    write_csv(MANIFEST_DIR / "run_manifest.csv", run_rows)
    write_json(MANIFEST_DIR / "arm_summary.json", {"runs": summaries})
    allocated_rows = (
        common_base
        + [row for block in range(BLOCK_COUNT) for row in evaluation[block]]
        + [row for block in range(BLOCK_COUNT) for row in exposure[block]]
    )
    report = {
        "block_count": BLOCK_COUNT,
        "common_base_rows": len(common_base),
        "evaluation_rows_by_block": {
            str(block): len(evaluation[block])
            for block in range(BLOCK_COUNT)
        },
        "exposure_rows_by_block": {
            str(block): len(exposure[block])
            for block in range(BLOCK_COUNT)
        },
        "materialization_id": MATERIALIZATION_ID,
        "maximum_allocated_sequence_length": max(
            row["sequence_length"] for row in allocated_rows
        ),
        "run_manifest_count": len(run_rows),
        "schema_version": (
            "external-behavior-v4-materialization-report-v1"
        ),
        "selected_pair_count": len(pairs),
        "selected_target_count": len(targets),
        "status": "materialized_pending_independent_validation",
        "training_started": False,
        "v3_target_overlap_count": len(
            selected_target_ids & v3_target_ids()
        ),
    }
    write_json(MANIFEST_DIR / "materialization_report.json", report)
    _write_data_checksums()
    return report


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="ascii").splitlines()


def _checklist(checks: list[dict[str, Any]]) -> str:
    status = {item["name"]: item["pass"] for item in checks}
    labels = (
        ("protocol", "Validate the frozen materialization protocol."),
        ("inputs", "Validate all frozen inputs and implementation hashes."),
        ("pair_manifest", "Validate 40 selected pairs in two blocks."),
        ("target_allocations", "Rebuild all 80 target allocations."),
        ("evaluation", "Validate 8,000 evaluation rows per block."),
        ("exposure", "Validate 2,560 exposure rows per block."),
        ("source_group_separation", "Validate evaluation/exposure group separation."),
        ("common_base", "Rebuild the shared 22,720-row common base."),
        ("run_count", "Validate all 32 run manifests."),
        ("training_rows", "Validate 24,000 exact rows per run."),
        ("complementary_exposure", "Validate paired complementary exposure."),
        ("label_balance", "Validate exact A/B label balance."),
        ("fixed_length", "Validate every allocated sequence fits length 185."),
        ("target_independence", "Validate zero V3/V4 target overlap."),
        ("training_absent", "Confirm no V4 training has started."),
        ("artifact_hashes", "Validate all materialized artifact hashes."),
    )
    lines = ["# Directional V4 Materialization Gate", ""]
    for name, label in labels:
        lines.append(f"- [{'x' if status.get(name) else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_materialization() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    try:
        verify_protocol()
        check("protocol", True)
    except Exception as error:
        check("protocol", False, repr(error))
    try:
        verify_inputs()
        check("inputs", True)
    except Exception as error:
        check("inputs", False, repr(error))

    pairs = load_selected_pairs()
    pair_ok = (
        len(pairs) == 40
        and Counter(row["block_index"] for row in pairs)
        == Counter({0: 20, 1: 20})
        and all(
            row["block_pair_index"] == index % 20
            for index, row in enumerate(pairs)
        )
    )
    check("pair_manifest", pair_ok)
    candidates = load_candidates()
    expected_targets, expected_eval, expected_exposure = allocate_blocks(
        candidates,
        pairs,
    )
    observed_targets = list(iter_jsonl(DATA_DIR / "targets.jsonl"))
    check(
        "target_allocations",
        observed_targets == expected_targets and len(observed_targets) == 80,
    )
    evaluation_ok = True
    exposure_ok = True
    separation_ok = True
    for block in range(BLOCK_COUNT):
        observed_eval = list(
            iter_jsonl(DATA_DIR / f"evaluation_block{block}.jsonl")
        )
        observed_exposure = list(
            iter_jsonl(DATA_DIR / f"exposure_block{block}.jsonl")
        )
        evaluation_ok &= (
            observed_eval == expected_eval[block]
            and len(observed_eval) == 8_000
        )
        exposure_ok &= (
            observed_exposure == expected_exposure[block]
            and len(observed_exposure) == 2_560
        )
        eval_groups = {
            (row["block_pair_index"], row["pair_member"]): set()
            for row in observed_eval
        }
        exposure_groups = {
            (row["block_pair_index"], row["pair_member"]): set()
            for row in observed_exposure
        }
        for row in observed_eval:
            eval_groups[
                (row["block_pair_index"], row["pair_member"])
            ].add(row["source_group_id"])
        for row in observed_exposure:
            exposure_groups[
                (row["block_pair_index"], row["pair_member"])
            ].add(row["source_group_id"])
        for key in eval_groups:
            separation_ok &= (
                len(eval_groups[key]) == 34
                and len(exposure_groups[key]) == 11
                and not eval_groups[key] & exposure_groups[key]
            )
    check("evaluation", evaluation_ok)
    check("exposure", exposure_ok)
    check("source_group_separation", separation_ok)

    target_ids = {row["semantic_id"] for row in expected_targets}
    excluded_ids = target_ids | v3_target_ids()
    expected_common = allocate_common_base(candidates, excluded_ids)
    observed_common = list(
        iter_jsonl(DATA_DIR / "common_base.jsonl")
    )
    common_ok = (
        observed_common == expected_common
        and len(observed_common) == COMMON_BASE_ROWS
        and not {
            row["semantic_id"] for row in observed_common
        }
        & excluded_ids
    )
    check("common_base", common_ok)

    run_rows = read_csv(MANIFEST_DIR / "run_manifest.csv")
    check(
        "run_count",
        len(run_rows) == 32
        and len({row["run_id"] for row in run_rows}) == 32,
    )
    exposure_ids = {
        row["surface_id"]
        for block in range(BLOCK_COUNT)
        for row in expected_exposure[block]
    }
    exposure_semantic_by_id = {
        row["surface_id"]: row["semantic_id"]
        for block in range(BLOCK_COUNT)
        for row in expected_exposure[block]
    }
    pair_semantics = {
        (
            row["block_index"],
            row["block_pair_index"],
            member,
        ): row[f"member_{member}"]
        for row in pairs
        for member in (0, 1)
    }
    common_ids = [row["surface_id"] for row in observed_common]
    common_labels = [row["target_relation"] for row in observed_common]
    training_failures: list[str] = []
    complement_failures: list[str] = []
    label_failures: list[str] = []
    rows_by_key = {
        (
            int(row["block_index"]),
            int(row["model_seed"]),
            row["arm"],
        ): row
        for row in run_rows
    }
    for block in range(BLOCK_COUNT):
        for model_seed in MODEL_SEEDS:
            labels_by_arm: dict[str, list[str]] = {}
            assignments: dict[str, dict[str, int]] = {}
            for arm in ARMS:
                row = rows_by_key.get((block, model_seed, arm))
                if row is None:
                    training_failures.append(
                        f"missing:{block}:{model_seed}:{arm}"
                    )
                    continue
                path = DATA_DIR / row["train_manifest"]
                ids = _read_lines(path)
                if (
                    len(ids) != TRAIN_ROWS
                    or int(row["rows"]) != TRAIN_ROWS
                    or ids[:COMMON_BASE_ROWS] != common_ids
                    or sha256_file(path) != row["train_manifest_sha256"]
                    or row["common_base_sha256"]
                    != sha256_file(DATA_DIR / "common_base.jsonl")
                ):
                    training_failures.append(row["run_id"])
                assignment = json.loads(row["exposed_pair_members"])
                assignments[arm] = assignment
                expected_exposure_ids: list[str] = []
                labels = list(common_labels)
                for pair_index in range(PAIRS_PER_BLOCK):
                    member = int(assignment[str(pair_index)])
                    expected_bit = arm_assignment_bit(
                        block,
                        model_seed,
                        pair_index,
                    )
                    expected_member = (
                        expected_bit if arm == "A" else 1 - expected_bit
                    )
                    if member != expected_member:
                        complement_failures.append(
                            f"assignment:{row['run_id']}:{pair_index}"
                        )
                    semantic_id = pair_semantics[
                        (block, pair_index, member)
                    ]
                    selected = [
                        item
                        for item in expected_exposure[block]
                        if item["block_pair_index"] == pair_index
                        and item["pair_member"] == member
                    ]
                    expected_exposure_ids.extend(
                        item["surface_id"] for item in selected
                    )
                    labels.extend(
                        item["target_relation"] for item in selected
                    )
                    observed_count = sum(
                        exposure_semantic_by_id.get(item) == semantic_id
                        for item in ids
                        if item in exposure_ids
                    )
                    if observed_count != 64:
                        complement_failures.append(
                            f"dose:{row['run_id']}:{pair_index}"
                        )
                if ids[COMMON_BASE_ROWS:] != expected_exposure_ids:
                    training_failures.append(
                        f"exposure_order:{row['run_id']}"
                    )
                labels_by_arm[arm] = labels
            if set(assignments) == set(ARMS):
                for pair_index in range(PAIRS_PER_BLOCK):
                    if (
                        assignments["A"][str(pair_index)]
                        == assignments["B"][str(pair_index)]
                    ):
                        complement_failures.append(
                            f"same:{block}:{model_seed}:{pair_index}"
                        )
            if set(labels_by_arm) == set(ARMS) and Counter(
                labels_by_arm["A"]
            ) != Counter(labels_by_arm["B"]):
                label_failures.append(f"{block}:{model_seed}")
    check("training_rows", not training_failures, training_failures[:50])
    check(
        "complementary_exposure",
        not complement_failures,
        complement_failures[:50],
    )
    check("label_balance", not label_failures, label_failures)

    allocated = (
        observed_common
        + [
            row
            for block in range(BLOCK_COUNT)
            for row in expected_eval[block]
        ]
        + [
            row
            for block in range(BLOCK_COUNT)
            for row in expected_exposure[block]
        ]
    )
    maximum_length = max(row["sequence_length"] for row in allocated)
    check("fixed_length", maximum_length <= FIXED_PADDING_LENGTH, maximum_length)
    check(
        "target_independence",
        not target_ids & v3_target_ids(),
        len(target_ids & v3_target_ids()),
    )
    check(
        "training_absent",
        not RUN_ROOT.exists(),
        str(RUN_ROOT),
    )
    checksum_failures: list[str] = []
    for expected, path in _checksum_entries(DATA_DIR / "SHA256SUMS"):
        if not path.is_file() or sha256_file(path) != expected:
            checksum_failures.append(path.name)
    for row in run_rows:
        path = DATA_DIR / row["evaluation_manifest"]
        if sha256_file(path) != row["evaluation_manifest_sha256"]:
            checksum_failures.append(
                f"evaluation:{row['run_id']}"
            )
    check(
        "artifact_hashes",
        not checksum_failures,
        checksum_failures,
    )
    failed = [item for item in checks if not item["pass"]]
    passed = not failed
    report = {
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": [item["name"] for item in failed],
        "failed_count": len(failed),
        "materialization_gate": "go" if passed else "no_go",
        "materialization_id": MATERIALIZATION_ID,
        "maximum_allocated_sequence_length": maximum_length,
        "schema_version": (
            "external-behavior-v4-materialization-validation-v1"
        ),
        "status": "pass" if passed else "no_go",
        "training_started": False,
        "validated_at_utc": _now(),
    }
    write_json(MANIFEST_DIR / "materialization_validation_report.json", report)
    (MANIFEST_DIR / "materialization_checklist.md").write_text(
        _checklist(checks),
        encoding="ascii",
    )
    frozen_paths = [
        DATA_DIR / "SHA256SUMS",
        MANIFEST_DIR / "arm_summary.json",
        MANIFEST_DIR / "run_manifest.csv",
        MANIFEST_DIR / "materialization_report.json",
        MANIFEST_DIR / "materialization_checklist.md",
        MANIFEST_DIR / "materialization_validation_report.json",
    ]
    (MANIFEST_DIR / "MATERIALIZATION_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {relative(path)}\n"
            for path in frozen_paths
        ),
        encoding="ascii",
    )
    return report
