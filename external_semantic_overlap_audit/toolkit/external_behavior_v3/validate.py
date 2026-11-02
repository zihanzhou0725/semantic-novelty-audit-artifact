"""Independent design-gate validator for external-behavior-v3."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from toolkit.external_behavior.clutrr import reconstruct_semantics
from toolkit.external_behavior.common import (
    ARMS,
    MODEL_SEEDS,
    arm_assignment_bit,
    iter_jsonl,
    sha256_bytes,
)
from toolkit.external_behavior.model import encode_parts, inspect_architecture

from .common import (
    DATA_DIR,
    MANIFEST_DIR,
    PROJECT_ROOT,
    PROTOCOL_ID,
    SOURCE_ROWS,
    read_json,
    sha256_file,
    verify_frozen_inputs,
    write_json,
)
from .design import (
    COMMON_BASE_ROWS,
    EVALUATION_GROUPS_PER_PATTERN,
    EVALUATION_ROWS_PER_PATTERN,
    EXPANDED_ROW_COUNT,
    EXPOSURE_GROUPS_PER_PATTERN,
    EXPOSURE_ROWS_PER_PATTERN,
    PAIR_COUNT,
    REQUIRED_GROUPS_PER_PATTERN,
    ROWS_PER_ARM,
    TARGET_PATTERN_COUNT,
)
from .materialize import validate_implementation_freeze


PERMUTATIONS = tuple(itertools.permutations(range(3)))


def _sentences(story: str) -> tuple[str, str, str]:
    if not story.endswith("."):
        raise ValueError("Story lacks terminal period.")
    parts = story[:-1].split(". ")
    if len(parts) != 3 or any(not part or "." in part for part in parts):
        raise ValueError("Story is not exactly three atomic sentences.")
    return tuple(f"{part}." for part in parts)  # type: ignore[return-value]


def _surface_id(normalized_input: str, target_relation: str) -> str:
    payload = json.dumps(
        [normalized_input, target_relation],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _expected_expanded(
    source: dict[str, Any],
    permutation_index: int,
    permutation: tuple[int, int, int],
) -> dict[str, Any]:
    canonical, semantic_id = reconstruct_semantics(source)
    sentences = _sentences(source["normalized_story"])
    story = " ".join(sentences[index] for index in permutation)
    normalized_input = f"{story} <sep> {source['normalized_query']}"
    surface_id = _surface_id(
        normalized_input,
        source["target_relation"],
    )
    return {
        "candidate_row_id": sha256_bytes(
            json.dumps(
                [
                    source["surface_id"],
                    permutation_index,
                    surface_id,
                    semantic_id,
                ],
                separators=(",", ":"),
            ).encode("ascii")
        ),
        "edge_types": source["edge_types"],
        "normalized_input": normalized_input,
        "normalized_query": source["normalized_query"],
        "normalized_story": story,
        "permutation": list(permutation),
        "permutation_index": permutation_index,
        "query_edge": source["query_edge"],
        "row_order_sha256": sha256_bytes(
            (
                normalized_input
                + canonical
                + source["target_relation"]
            ).encode("utf-8")
        ),
        "semantic_id": semantic_id,
        "semantic_serialization": canonical,
        "sequence_length": len(encode_parts(story, source["normalized_query"])),
        "source_group_id": source["surface_id"],
        "source_row_order_sha256": source["row_order_sha256"],
        "story_edges": source["story_edges"],
        "surface_id": surface_id,
        "target_relation": source["target_relation"],
    }


def _checksum_status(directory: Path) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for line in (directory / "SHA256SUMS").read_text(
        encoding="ascii"
    ).splitlines():
        expected, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256_file(path) != expected:
            failures.append(f"hash:{relative}")
    return not failures, failures


def _patterns(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    list[dict[str, Any]],
]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[row["semantic_id"]][row["source_group_id"]].append(row)
        labels[row["semantic_id"]].add(row["target_relation"])
    eligible: list[dict[str, Any]] = []
    for semantic_id, groups in grouped.items():
        for group_rows in groups.values():
            group_rows.sort(key=lambda row: row["permutation_index"])
        complete = sum(
            len(group_rows) == 6
            and [row["permutation_index"] for row in group_rows]
            == list(range(6))
            for group_rows in groups.values()
        )
        if complete >= REQUIRED_GROUPS_PER_PATTERN:
            eligible.append(
                {
                    "complete_source_groups": complete,
                    "semantic_id": semantic_id,
                    "target_relation": next(iter(labels[semantic_id])),
                    "unique_expanded_rows": sum(
                        len(group_rows) for group_rows in groups.values()
                    ),
                }
            )
    eligible.sort(key=lambda row: row["semantic_id"])
    return grouped, eligible


def _pairs(eligible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    counts = {
        row["semantic_id"]: row["complete_source_groups"]
        for row in eligible
    }
    for row in eligible:
        by_label[row["target_relation"]].append(row["semantic_id"])
    candidates: list[dict[str, Any]] = []
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        for offset in range(0, len(ids) - 1, 2):
            members = ids[offset : offset + 2]
            candidates.append(
                {
                    "member_0": members[0],
                    "member_0_complete_source_groups": counts[members[0]],
                    "member_1": members[1],
                    "member_1_complete_source_groups": counts[members[1]],
                    "pair_sha256": sha256_bytes(
                        "".join(members).encode("ascii")
                    ),
                    "target_relation": label,
                }
            )
    candidates.sort(key=lambda row: row["pair_sha256"])
    selected = candidates[:PAIR_COUNT]
    for index, pair in enumerate(selected):
        pair["pair_index"] = index
    return selected


def _target_allocations(
    grouped: dict[str, dict[str, list[dict[str, Any]]]],
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    exposure: list[dict[str, Any]] = []
    for pair in pairs:
        for member in (0, 1):
            semantic_id = pair[f"member_{member}"]
            groups = [
                (group_id, rows)
                for group_id, rows in sorted(grouped[semantic_id].items())
                if len(rows) == 6
                and [row["permutation_index"] for row in rows]
                == list(range(6))
            ]
            eval_groups = groups[:EVALUATION_GROUPS_PER_PATTERN]
            exp_groups = groups[
                EVALUATION_GROUPS_PER_PATTERN :
                EVALUATION_GROUPS_PER_PATTERN
                + EXPOSURE_GROUPS_PER_PATTERN
            ]
            eval_rows = [
                row
                for _, rows in eval_groups[:33]
                for row in rows
            ] + eval_groups[33][1][:2]
            exp_rows = [
                row
                for _, rows in exp_groups[:10]
                for row in rows
            ] + exp_groups[10][1][:4]
            targets.append(
                {
                    "evaluation_row_ids": [
                        row["surface_id"] for row in eval_rows
                    ],
                    "evaluation_source_group_ids": [
                        group_id for group_id, _ in eval_groups
                    ],
                    "exposure_row_ids": [
                        row["surface_id"] for row in exp_rows
                    ],
                    "exposure_source_group_ids": [
                        group_id for group_id, _ in exp_groups
                    ],
                    "member_index": member,
                    "pair_index": pair["pair_index"],
                    "semantic_id": semantic_id,
                    "target_relation": pair["target_relation"],
                }
            )
            evaluation.extend(
                {
                    **row,
                    "allocation_index": index,
                    "pair_index": pair["pair_index"],
                    "pair_member": member,
                }
                for index, row in enumerate(eval_rows)
            )
            exposure.extend(
                {
                    **row,
                    "allocation_index": index,
                    "pair_index": pair["pair_index"],
                    "pair_member": member,
                }
                for index, row in enumerate(exp_rows)
            )
    targets.sort(key=lambda row: (row["pair_index"], row["member_index"]))
    evaluation.sort(
        key=lambda row: (
            row["pair_index"],
            row["pair_member"],
            row["allocation_index"],
        )
    )
    exposure.sort(
        key=lambda row: (
            row["pair_index"],
            row["pair_member"],
            row["allocation_index"],
        )
    )
    return targets, evaluation, exposure


def _common_base(
    rows: list[dict[str, Any]],
    target_ids: set[str],
) -> list[dict[str, Any]]:
    queues: dict[str, dict[str, deque[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for row in sorted(
        rows,
        key=lambda item: (
            item["target_relation"],
            item["semantic_id"],
            item["source_group_id"],
            item["permutation_index"],
        ),
    ):
        if row["semantic_id"] not in target_ids:
            queues[row["target_relation"]][row["semantic_id"]].append(row)
    labels = sorted(queues)
    ids = {label: sorted(queues[label]) for label in labels}
    positions = {label: 0 for label in labels}
    output: list[dict[str, Any]] = []
    while len(output) < COMMON_BASE_ROWS:
        progress = False
        for label in labels:
            for _ in range(len(ids[label])):
                position = positions[label] % len(ids[label])
                semantic_id = ids[label][position]
                positions[label] = (position + 1) % len(ids[label])
                if queues[label][semantic_id]:
                    output.append(queues[label][semantic_id].popleft())
                    progress = True
                    break
            if len(output) == COMMON_BASE_ROWS:
                break
        if not progress:
            break
    return output


def _checklist(checks: list[dict[str, Any]]) -> str:
    status = {item["name"]: item["pass"] for item in checks}
    entries = [
        ("Validate frozen inputs and implementation.", "frozen_inputs"),
        ("Regenerate all 204,870 raw permutations.", "raw_expansion"),
        ("Validate global surface deduplication.", "deduplication"),
        ("Verify at least 20 eligible same-label pairs.", "pair_capacity"),
        ("Freeze exactly 40 targets and 20 pairs.", "targets"),
        ("Freeze 8,000 evaluation rows.", "evaluation"),
        ("Freeze 2,560 exposure rows.", "exposure"),
        ("Verify evaluation/exposure source-group disjointness.", "group_disjointness"),
        ("Freeze deterministic 22,720-row common base.", "common_base"),
        ("Validate all 16 paired arm manifests.", "arms"),
        ("Verify all allocated inputs fit 512 bytes.", "sequence_lengths"),
        ("Validate frozen byte Transformer architecture.", "model_architecture"),
        ("Validate all data hashes.", "data_hashes"),
        ("Confirm model training has not started.", "training_not_started"),
    ]
    lines = ["# External Behavioral Validation v3 Design Gate", ""]
    for label, name in entries:
        lines.append(f"- [{'x' if status.get(name, False) else ' '}] {label}")
    lines.append("")
    return "\n".join(lines)


def validate_design() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    inputs = verify_frozen_inputs()
    implementation = validate_implementation_freeze()
    check(
        "frozen_inputs",
        inputs["status"] == "pass" and implementation["status"] == "frozen",
    )
    current_lock = sorted(
        line.strip()
        for line in subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
        ).splitlines()
        if line.strip()
    )
    frozen_lock = sorted(
        line.strip()
        for line in (MANIFEST_DIR / "requirements.lock")
        .read_text(encoding="ascii")
        .splitlines()
        if line.strip()
    )
    check("environment_lock", current_lock == frozen_lock)
    hashes_ok, hash_failures = _checksum_status(DATA_DIR)
    check("data_hashes", hashes_ok, hash_failures)

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    raw_failures = 0
    raw_count = 0
    raw_iterator: Iterator[dict[str, Any]] = iter_jsonl(
        DATA_DIR / "clutrr_v3_expanded_raw.jsonl"
    )
    for source in iter_jsonl(SOURCE_ROWS):
        for permutation_index, permutation in enumerate(PERMUTATIONS):
            try:
                observed = next(raw_iterator)
            except StopIteration:
                raw_failures += 1
                break
            expected = _expected_expanded(
                source,
                permutation_index,
                permutation,
            )
            raw_count += 1
            if observed != expected:
                raw_failures += 1
            key = (
                observed["normalized_input"],
                observed["target_relation"],
            )
            prior = unique.get(key)
            if prior is not None and prior["semantic_id"] != observed["semantic_id"]:
                raw_failures += 1
            unique.setdefault(key, observed)
    try:
        next(raw_iterator)
        raw_failures += 1
    except StopIteration:
        pass
    check(
        "raw_expansion",
        raw_count == EXPANDED_ROW_COUNT and raw_failures == 0,
        {"failures": raw_failures, "rows": raw_count},
    )

    expected_candidates = sorted(
        unique.values(),
        key=lambda row: (
            row["row_order_sha256"],
            row["source_group_id"],
            row["permutation_index"],
        ),
    )
    observed_candidates = list(
        iter_jsonl(DATA_DIR / "clutrr_v3_candidates.jsonl")
    )
    check(
        "deduplication",
        observed_candidates == expected_candidates,
        len(observed_candidates),
    )
    grouped, eligible = _patterns(expected_candidates)
    observed_eligible = list(
        iter_jsonl(DATA_DIR / "clutrr_v3_eligible_patterns.jsonl")
    )
    pairs = _pairs(eligible)
    observed_pairs = list(iter_jsonl(DATA_DIR / "clutrr_v3_pairs.jsonl"))
    check(
        "pair_capacity",
        len(pairs) == PAIR_COUNT,
        {"eligible_patterns": len(eligible), "pairs": len(pairs)},
    )
    expected_targets, expected_eval, expected_exposure = _target_allocations(
        grouped,
        pairs,
    )
    targets = list(iter_jsonl(DATA_DIR / "clutrr_v3_targets.jsonl"))
    evaluation = list(
        iter_jsonl(DATA_DIR / "clutrr_v3_evaluation.jsonl")
    )
    exposure = list(iter_jsonl(DATA_DIR / "clutrr_v3_exposure.jsonl"))
    check(
        "targets",
        observed_eligible == eligible
        and observed_pairs == pairs
        and targets == expected_targets
        and len(targets) == TARGET_PATTERN_COUNT,
        {"pairs": len(observed_pairs), "targets": len(targets)},
    )
    check(
        "evaluation",
        evaluation == expected_eval
        and len(evaluation)
        == TARGET_PATTERN_COUNT * EVALUATION_ROWS_PER_PATTERN
        and len({row["source_group_id"] for row in evaluation})
        == TARGET_PATTERN_COUNT * EVALUATION_GROUPS_PER_PATTERN,
        len(evaluation),
    )
    check(
        "exposure",
        exposure == expected_exposure
        and len(exposure)
        == TARGET_PATTERN_COUNT * EXPOSURE_ROWS_PER_PATTERN
        and len({row["source_group_id"] for row in exposure})
        == TARGET_PATTERN_COUNT * EXPOSURE_GROUPS_PER_PATTERN,
        len(exposure),
    )
    eval_groups = {row["source_group_id"] for row in evaluation}
    exposure_groups = {row["source_group_id"] for row in exposure}
    check(
        "group_disjointness",
        not (eval_groups & exposure_groups),
        len(eval_groups & exposure_groups),
    )

    target_ids = {row["semantic_id"] for row in targets}
    expected_common = _common_base(expected_candidates, target_ids)
    common = list(iter_jsonl(DATA_DIR / "clutrr_v3_common_base.jsonl"))
    check(
        "common_base",
        common == expected_common
        and len(common) == COMMON_BASE_ROWS
        and not ({row["semantic_id"] for row in common} & target_ids),
        len(common),
    )

    labels_by_surface = {
        row["surface_id"]: row["target_relation"]
        for row in expected_candidates
    }
    exposure_by_member: dict[
        tuple[int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in expected_exposure:
        exposure_by_member[
            (row["pair_index"], row["pair_member"])
        ].append(row)
    for rows in exposure_by_member.values():
        rows.sort(key=lambda row: row["allocation_index"])
    arm_failures: list[str] = []
    for seed in MODEL_SEEDS:
        expected_ids: dict[str, list[str]] = {}
        for arm in ARMS:
            ids = [row["surface_id"] for row in expected_common]
            for pair in pairs:
                bit = arm_assignment_bit(seed, pair["pair_index"])
                member = bit if arm == "A" else 1 - bit
                ids.extend(
                    row["surface_id"]
                    for row in exposure_by_member[
                        (pair["pair_index"], member)
                    ]
                )
            expected_ids[arm] = ids
            path = DATA_DIR / f"clutrr_train_seed{seed}_arm{arm}.txt"
            observed_ids = path.read_text(encoding="ascii").splitlines()
            if observed_ids != ids or len(ids) != ROWS_PER_ARM:
                arm_failures.append(f"rows:{seed}:{arm}")
        if Counter(
            labels_by_surface[row_id] for row_id in expected_ids["A"]
        ) != Counter(
            labels_by_surface[row_id] for row_id in expected_ids["B"]
        ):
            arm_failures.append(f"labels:{seed}")
    check("arms", not arm_failures, arm_failures)

    maximum_length = max(
        row["sequence_length"]
        for row in evaluation + exposure + common
    )
    check(
        "sequence_lengths",
        maximum_length <= 512,
        maximum_length,
    )
    architecture = inspect_architecture(23)
    materialization = read_json(MANIFEST_DIR / "materialization_manifest.json")
    check(
        "model_architecture",
        architecture
        == materialization["design_report"]["model_architecture"],
        architecture,
    )
    run_root = PROJECT_ROOT / "runs" / PROTOCOL_ID
    check(
        "training_not_started",
        not run_root.exists() or not any(run_root.iterdir()),
    )

    required = {
        "frozen_inputs",
        "environment_lock",
        "data_hashes",
        "raw_expansion",
        "deduplication",
        "pair_capacity",
        "targets",
        "evaluation",
        "exposure",
        "group_disjointness",
        "common_base",
        "arms",
        "sequence_lengths",
        "model_architecture",
        "training_not_started",
    }
    by_name = {item["name"]: item["pass"] for item in checks}
    gate_pass = all(by_name[name] for name in required)
    report = {
        "check_count": len(checks),
        "checks": checks,
        "design_gate": "pass" if gate_pass else "no_go",
        "failed_checks": [
            item["name"] for item in checks if not item["pass"]
        ],
        "failed_count": sum(not item["pass"] for item in checks),
        "model_training_started": False,
        "protocol_id": PROTOCOL_ID,
        "schema_version": "external-behavior-v3-design-validation-v1",
        "status": "pass" if gate_pass else "no_go",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checklist_path = MANIFEST_DIR / "design_checklist.md"
    checklist_path.write_text(_checklist(checks), encoding="ascii")
    report_path = MANIFEST_DIR / "design_validation_report.json"
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
    (MANIFEST_DIR / "DESIGN_SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  "
            f"{path.relative_to(MANIFEST_DIR).as_posix()}\n"
            for path in freeze_paths
        ),
        encoding="ascii",
    )
    return report
