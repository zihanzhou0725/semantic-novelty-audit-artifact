"""Materialize the CLUTRR V3 sentence-permutation design."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable

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
    SOURCE_ROWS,
    sha256_file,
    verify_frozen_inputs,
    write_json,
    write_jsonl,
    write_lines,
)


PERMUTATIONS = tuple(itertools.permutations(range(3)))
EXPANDED_ROW_COUNT = 204_870
TARGET_PATTERN_COUNT = 40
PAIR_COUNT = 20
EVALUATION_ROWS_PER_PATTERN = 200
EXPOSURE_ROWS_PER_PATTERN = 64
EVALUATION_GROUPS_PER_PATTERN = 34
EXPOSURE_GROUPS_PER_PATTERN = 11
REQUIRED_GROUPS_PER_PATTERN = 45
COMMON_BASE_ROWS = 22_720
ROWS_PER_ARM = 24_000


def parse_sentences(normalized_story: str) -> tuple[str, str, str]:
    if not normalized_story.endswith("."):
        raise ValueError("Normalized story must end in a period.")
    parts = normalized_story[:-1].split(". ")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError("Normalized story must contain exactly three sentences.")
    if any("." in part for part in parts):
        raise ValueError("Internal sentence periods are forbidden.")
    return tuple(f"{part}." for part in parts)  # type: ignore[return-value]


def _surface_id(normalized_input: str, target_relation: str) -> str:
    return sha256_bytes(
        json.dumps(
            [normalized_input, target_relation],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )


def expand_source_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    canonical, semantic_id = reconstruct_semantics(row)
    if (
        canonical != row["semantic_serialization"]
        or semantic_id != row["semantic_id"]
    ):
        raise ValueError("Frozen source semantic reconstruction failed.")
    sentences = parse_sentences(row["normalized_story"])
    expected_input = (
        f"{row['normalized_story']} <sep> {row['normalized_query']}"
    )
    if expected_input != row["normalized_input"]:
        raise ValueError("Frozen source normalized input is inconsistent.")

    expanded: list[dict[str, Any]] = []
    for permutation_index, permutation in enumerate(PERMUTATIONS):
        story = " ".join(sentences[index] for index in permutation)
        normalized_input = f"{story} <sep> {row['normalized_query']}"
        surface_id = _surface_id(
            normalized_input,
            row["target_relation"],
        )
        order_hash = sha256_bytes(
            (
                normalized_input
                + row["semantic_serialization"]
                + row["target_relation"]
            ).encode("utf-8")
        )
        expanded.append(
            {
                "candidate_row_id": sha256_bytes(
                    json.dumps(
                        [
                            row["surface_id"],
                            permutation_index,
                            surface_id,
                            semantic_id,
                        ],
                        separators=(",", ":"),
                    ).encode("ascii")
                ),
                "edge_types": row["edge_types"],
                "normalized_input": normalized_input,
                "normalized_query": row["normalized_query"],
                "normalized_story": story,
                "permutation": list(permutation),
                "permutation_index": permutation_index,
                "query_edge": row["query_edge"],
                "row_order_sha256": order_hash,
                "semantic_id": semantic_id,
                "semantic_serialization": canonical,
                "sequence_length": len(
                    encode_parts(story, row["normalized_query"])
                ),
                "source_group_id": row["surface_id"],
                "source_row_order_sha256": row["row_order_sha256"],
                "story_edges": row["story_edges"],
                "surface_id": surface_id,
                "target_relation": row["target_relation"],
            }
        )
    if len({item["normalized_input"] for item in expanded}) != 6:
        raise ValueError("Source group does not have six unique permutations.")
    return expanded


def materialize_candidates(
    raw_path: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique_surfaces: dict[tuple[str, str], dict[str, Any]] = {}
    semantic_by_surface: dict[tuple[str, str], str] = {}
    source_rows = 0
    expanded_rows = 0
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii") as handle:
        for source_rows, source in enumerate(
            iter_jsonl(SOURCE_ROWS),
            start=1,
        ):
            for expanded in expand_source_row(source):
                key = (
                    expanded["normalized_input"],
                    expanded["target_relation"],
                )
                prior = semantic_by_surface.get(key)
                if prior is not None and prior != expanded["semantic_id"]:
                    raise ValueError(
                        "Expanded surface has conflicting semantic IDs."
                    )
                semantic_by_surface[key] = expanded["semantic_id"]
                unique_surfaces.setdefault(key, expanded)
                handle.write(
                    json.dumps(
                        expanded,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                expanded_rows += 1
            if progress is not None and (
                source_rows % 5_000 == 0 or source_rows == 34_145
            ):
                progress(source_rows, 34_145)
    temporary.replace(raw_path)
    if expanded_rows != EXPANDED_ROW_COUNT:
        raise ValueError(
            f"Expanded row count changed: {expanded_rows}."
        )
    deduplicated = sorted(
        unique_surfaces.values(),
        key=lambda row: (
            row["row_order_sha256"],
            row["source_group_id"],
            row["permutation_index"],
        ),
    )
    return deduplicated, {
        "deduplicated_rows": len(deduplicated),
        "expanded_rows": expanded_rows,
        "raw_manifest_sha256": sha256_file(raw_path),
        "source_rows": source_rows,
    }


def pattern_groups(
    rows: Iterable[dict[str, Any]],
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
    for semantic_id, source_groups in grouped.items():
        if len(labels[semantic_id]) != 1:
            raise ValueError("Semantic pattern has multiple target labels.")
        complete_groups = 0
        for group_rows in source_groups.values():
            group_rows.sort(key=lambda row: row["permutation_index"])
            if (
                len(group_rows) == 6
                and [row["permutation_index"] for row in group_rows]
                == list(range(6))
            ):
                complete_groups += 1
        if complete_groups >= REQUIRED_GROUPS_PER_PATTERN:
            eligible.append(
                {
                    "complete_source_groups": complete_groups,
                    "semantic_id": semantic_id,
                    "target_relation": next(iter(labels[semantic_id])),
                    "unique_expanded_rows": sum(
                        len(group_rows)
                        for group_rows in source_groups.values()
                    ),
                }
            )
    eligible.sort(key=lambda row: row["semantic_id"])
    return grouped, eligible


def select_pairs(eligible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    group_counts = {
        row["semantic_id"]: row["complete_source_groups"]
        for row in eligible
    }
    for row in eligible:
        by_label[row["target_relation"]].append(row["semantic_id"])
    pairs: list[dict[str, Any]] = []
    for label in sorted(by_label):
        semantic_ids = sorted(by_label[label])
        for offset in range(0, len(semantic_ids) - 1, 2):
            members = semantic_ids[offset : offset + 2]
            pairs.append(
                {
                    "member_0": members[0],
                    "member_0_complete_source_groups": group_counts[
                        members[0]
                    ],
                    "member_1": members[1],
                    "member_1_complete_source_groups": group_counts[
                        members[1]
                    ],
                    "pair_sha256": sha256_bytes(
                        "".join(members).encode("ascii")
                    ),
                    "target_relation": label,
                }
            )
    pairs.sort(key=lambda row: row["pair_sha256"])
    selected = pairs[:PAIR_COUNT]
    for pair_index, pair in enumerate(selected):
        pair["pair_index"] = pair_index
    return selected


def _complete_groups(
    source_groups: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (group_id, rows)
        for group_id, rows in sorted(source_groups.items())
        if len(rows) == 6
        and [row["permutation_index"] for row in rows] == list(range(6))
    ]


def allocate_targets(
    grouped: dict[str, dict[str, list[dict[str, Any]]]],
    pairs: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    targets: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    exposure: list[dict[str, Any]] = []
    for pair in pairs:
        for member_index in (0, 1):
            semantic_id = pair[f"member_{member_index}"]
            groups = _complete_groups(grouped[semantic_id])
            if len(groups) < REQUIRED_GROUPS_PER_PATTERN:
                raise ValueError("Selected target lacks complete source groups.")
            eval_groups = groups[:EVALUATION_GROUPS_PER_PATTERN]
            exposure_groups = groups[
                EVALUATION_GROUPS_PER_PATTERN :
                EVALUATION_GROUPS_PER_PATTERN
                + EXPOSURE_GROUPS_PER_PATTERN
            ]
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
            if (
                len(eval_rows) != EVALUATION_ROWS_PER_PATTERN
                or len(exposure_rows) != EXPOSURE_ROWS_PER_PATTERN
            ):
                raise AssertionError("Target allocation arithmetic changed.")
            eval_group_ids = [group_id for group_id, _ in eval_groups]
            exposure_group_ids = [
                group_id for group_id, _ in exposure_groups
            ]
            if set(eval_group_ids) & set(exposure_group_ids):
                raise AssertionError("Evaluation/exposure source groups overlap.")
            target = {
                "evaluation_row_ids": [
                    row["surface_id"] for row in eval_rows
                ],
                "evaluation_source_group_ids": eval_group_ids,
                "exposure_row_ids": [
                    row["surface_id"] for row in exposure_rows
                ],
                "exposure_source_group_ids": exposure_group_ids,
                "member_index": member_index,
                "pair_index": pair["pair_index"],
                "semantic_id": semantic_id,
                "target_relation": pair["target_relation"],
            }
            targets.append(target)
            evaluation.extend(
                {
                    **row,
                    "allocation_index": allocation_index,
                    "pair_index": pair["pair_index"],
                    "pair_member": member_index,
                }
                for allocation_index, row in enumerate(eval_rows)
            )
            exposure.extend(
                {
                    **row,
                    "allocation_index": allocation_index,
                    "pair_index": pair["pair_index"],
                    "pair_member": member_index,
                }
                for allocation_index, row in enumerate(exposure_rows)
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


def allocate_common_base(
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
    semantic_ids = {
        label: sorted(queues[label]) for label in labels
    }
    positions = {label: 0 for label in labels}
    output: list[dict[str, Any]] = []
    while len(output) < COMMON_BASE_ROWS:
        made_progress = False
        for label in labels:
            ids = semantic_ids[label]
            for _ in range(len(ids)):
                position = positions[label] % len(ids)
                semantic_id = ids[position]
                positions[label] = (position + 1) % len(ids)
                if queues[label][semantic_id]:
                    output.append(queues[label][semantic_id].popleft())
                    made_progress = True
                    break
            if len(output) == COMMON_BASE_ROWS:
                break
        if not made_progress:
            break
    return output


def materialize_arm_manifests(
    data_dir: Path,
    common_base: list[dict[str, Any]],
    exposure: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        len(common_base) != COMMON_BASE_ROWS
        or len(exposure) != TARGET_PATTERN_COUNT * EXPOSURE_ROWS_PER_PATTERN
        or len(pairs) != PAIR_COUNT
    ):
        return []
    exposure_by_member: dict[
        tuple[int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in exposure:
        exposure_by_member[
            (row["pair_index"], row["pair_member"])
        ].append(row)
    for rows in exposure_by_member.values():
        rows.sort(key=lambda row: row["allocation_index"])

    common_ids = [row["surface_id"] for row in common_base]
    common_labels = [row["target_relation"] for row in common_base]
    summaries: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        arm_assignments: dict[str, dict[str, int]] = {}
        arm_labels: dict[str, list[str]] = {}
        for arm in ARMS:
            row_ids = list(common_ids)
            labels = list(common_labels)
            arm_assignments[arm] = {}
            for pair in pairs:
                pair_index = pair["pair_index"]
                bit = arm_assignment_bit(seed, pair_index)
                member = bit if arm == "A" else 1 - bit
                arm_assignments[arm][str(pair_index)] = member
                selected = exposure_by_member[(pair_index, member)]
                row_ids.extend(row["surface_id"] for row in selected)
                labels.extend(row["target_relation"] for row in selected)
            path = data_dir / f"clutrr_train_seed{seed}_arm{arm}.txt"
            write_lines(path, row_ids)
            arm_labels[arm] = labels
            summaries.append(
                {
                    "arm": arm,
                    "exposed_pair_members": arm_assignments[arm],
                    "label_counts": dict(sorted(Counter(labels).items())),
                    "manifest": path.name,
                    "manifest_sha256": sha256_file(path),
                    "model_seed": seed,
                    "rows": len(row_ids),
                }
            )
        if Counter(arm_labels["A"]) != Counter(arm_labels["B"]):
            raise AssertionError("A/B label counts differ.")
        for pair in pairs:
            index = str(pair["pair_index"])
            if arm_assignments["A"][index] == arm_assignments["B"][index]:
                raise AssertionError("Pair exposure is not complementary.")
    return summaries


def _progress(completed: int, total: int) -> None:
    print(f"CLUTRR V3 source expansion {completed}/{total}", flush=True)


def materialize_design(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    inputs = verify_frozen_inputs()
    data_dir.mkdir(parents=True, exist_ok=False)
    raw_path = data_dir / "clutrr_v3_expanded_raw.jsonl"
    candidates, generation = materialize_candidates(
        raw_path,
        progress=_progress,
    )
    candidates_path = data_dir / "clutrr_v3_candidates.jsonl"
    write_jsonl(candidates_path, candidates)
    grouped, eligible = pattern_groups(candidates)
    pairs = select_pairs(eligible)
    targets, evaluation, exposure = allocate_targets(grouped, pairs)
    target_ids = {row["semantic_id"] for row in targets}
    common_base = allocate_common_base(candidates, target_ids)

    outputs = {
        "eligible": data_dir / "clutrr_v3_eligible_patterns.jsonl",
        "pairs": data_dir / "clutrr_v3_pairs.jsonl",
        "targets": data_dir / "clutrr_v3_targets.jsonl",
        "evaluation": data_dir / "clutrr_v3_evaluation.jsonl",
        "exposure": data_dir / "clutrr_v3_exposure.jsonl",
        "common_base": data_dir / "clutrr_v3_common_base.jsonl",
    }
    write_jsonl(outputs["eligible"], eligible)
    write_jsonl(outputs["pairs"], pairs)
    write_jsonl(outputs["targets"], targets)
    write_jsonl(outputs["evaluation"], evaluation)
    write_jsonl(outputs["exposure"], exposure)
    write_jsonl(outputs["common_base"], common_base)
    arms = materialize_arm_manifests(
        data_dir,
        common_base,
        exposure,
        pairs,
    )
    write_json(data_dir / "clutrr_v3_arm_summary.json", arms)
    architecture = inspect_architecture(23)
    allocated = evaluation + exposure + common_base
    maximum_length = max(
        (row["sequence_length"] for row in allocated),
        default=0,
    )
    design_ready = (
        generation["expanded_rows"] == EXPANDED_ROW_COUNT
        and len(pairs) == PAIR_COUNT
        and len(targets) == TARGET_PATTERN_COUNT
        and len(evaluation)
        == TARGET_PATTERN_COUNT * EVALUATION_ROWS_PER_PATTERN
        and len(exposure)
        == TARGET_PATTERN_COUNT * EXPOSURE_ROWS_PER_PATTERN
        and len(common_base) == COMMON_BASE_ROWS
        and len(arms) == len(MODEL_SEEDS) * len(ARMS)
        and maximum_length <= 512
    )
    report = {
        "arm_manifest_count": len(arms),
        "candidate_generation": generation,
        "common_base_rows": len(common_base),
        "design_ready_for_validation": design_ready,
        "eligible_pattern_count": len(eligible),
        "evaluation_rows": len(evaluation),
        "evaluation_source_groups": len(
            {row["source_group_id"] for row in evaluation}
        ),
        "exposure_rows": len(exposure),
        "exposure_source_groups": len(
            {row["source_group_id"] for row in exposure}
        ),
        "input_manifest_sha256": inputs["input_manifest_sha256"],
        "maximum_allocated_sequence_length": maximum_length,
        "model_architecture": architecture,
        "pair_count": len(pairs),
        "status": "materialized_pending_independent_validation",
        "target_count": len(targets),
    }
    write_json(data_dir / "clutrr_v3_design_report.json", report)
    return report
