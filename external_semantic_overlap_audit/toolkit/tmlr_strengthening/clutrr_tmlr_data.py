"""Deterministic CLUTRR data allocation for semantic-versus-surface exposure."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

from toolkit.external_behavior.common import iter_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v3"
    / "data"
    / "clutrr_v3_candidates.jsonl"
)
VTHREE_TARGETS_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v3"
    / "data"
    / "clutrr_v3_targets.jsonl"
)
VFOUR_TARGETS_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "materialization-v1"
    / "data"
    / "targets.jsonl"
)
COMMON_BASE_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "materialization-v1"
    / "data"
    / "common_base.jsonl"
)
EVALUATION_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "materialization-v1"
    / "data"
    / "evaluation_block0.jsonl"
)
SELECTED_PAIRS_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "external-behavior-v4"
    / "directional-design-v1"
    / "selected_pairs.csv"
)

TARGET_COUNT = 40
SLOTS_PER_TARGET = 48
COMMON_ROWS = 22_720
TRAIN_ROWS = COMMON_ROWS + TARGET_COUNT * SLOTS_PER_TARGET
EVAL_ROWS = TARGET_COUNT * 200
ARMS = ("C0", "C1", "C2")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _semantic_id(target_relation: str, edges: Iterable[str]) -> str:
    payload = json.dumps(
        [target_relation, *edges],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def shared_edge_count(left: Iterable[str], right: Iterable[str]) -> int:
    """Return multiset edge overlap."""

    return sum((Counter(left) & Counter(right)).values())


def _complete_groups(
    rows: list[dict[str, object]],
) -> list[tuple[str, list[dict[str, object]]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group_id"])].append(row)
    complete = []
    for group_id, group_rows in sorted(grouped.items()):
        ordered = sorted(
            group_rows,
            key=lambda row: int(row["permutation_index"]),
        )
        if len(ordered) == 6 and [
            int(row["permutation_index"]) for row in ordered
        ] == list(range(6)):
            complete.append((group_id, ordered))
    return complete


def _round_robin(
    rows: list[dict[str, object]],
    limit: int,
) -> list[dict[str, object]]:
    by_semantic: dict[str, deque[dict[str, object]]] = {}
    for row in rows:
        semantic_id = str(row["semantic_id"])
        by_semantic.setdefault(semantic_id, deque()).append(row)
    selected: list[dict[str, object]] = []
    while len(selected) < limit and by_semantic:
        exhausted = []
        for semantic_id in list(by_semantic):
            selected.append(by_semantic[semantic_id].popleft())
            if not by_semantic[semantic_id]:
                exhausted.append(semantic_id)
            if len(selected) == limit:
                break
        for semantic_id in exhausted:
            del by_semantic[semantic_id]
    return selected


def _take_fillers(
    pool: list[dict[str, object]],
    count: int,
    used: set[str],
) -> list[dict[str, object]]:
    if count == 0:
        return []
    selected = []
    for row in pool:
        surface_id = str(row["surface_id"])
        if surface_id in used:
            continue
        selected.append(row)
        used.add(surface_id)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Filler pool has fewer than {count} unused rows; "
        f"selected {len(selected)}."
    )


def _load_pairs() -> list[dict[str, str]]:
    with SELECTED_PAIRS_PATH.open(newline="", encoding="ascii") as handle:
        pairs = list(csv.DictReader(handle))
    if len(pairs) != 40:
        raise ValueError("Expected 40 selected V4 pattern pairs.")
    return pairs


def build_t6_design() -> dict[str, object]:
    """Build all three deterministic arms and their allocation evidence."""

    required_paths = (
        CANDIDATES_PATH,
        VTHREE_TARGETS_PATH,
        VFOUR_TARGETS_PATH,
        COMMON_BASE_PATH,
        EVALUATION_PATH,
        SELECTED_PAIRS_PATH,
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing T6 inputs: {missing}")

    candidates = list(iter_jsonl(CANDIDATES_PATH))
    common_base = list(iter_jsonl(COMMON_BASE_PATH))
    vthree_targets = list(iter_jsonl(VTHREE_TARGETS_PATH))
    vfour_targets = list(iter_jsonl(VFOUR_TARGETS_PATH))
    frozen_evaluation = list(iter_jsonl(EVALUATION_PATH))
    pairs = _load_pairs()
    if len(common_base) != COMMON_ROWS:
        raise ValueError("V4 common base row count changed.")
    if len(vthree_targets) != 40 or len(vfour_targets) != 80:
        raise ValueError("Frozen V3 or V4 target count changed.")

    pair_target_ids = {
        pair[f"member_{member}"]
        for pair in pairs
        for member in (0, 1)
    }
    vfour_target_ids = {
        str(row["semantic_id"]) for row in vfour_targets
    }
    if pair_target_ids != vfour_target_ids:
        raise ValueError("Selected-pair and materialized target IDs differ.")
    vthree_target_ids = {
        str(row["semantic_id"]) for row in vthree_targets
    }
    forbidden_semantic_ids = vfour_target_ids | vthree_target_ids
    common_surface_ids = {
        str(row["surface_id"]) for row in common_base
    }

    rows_by_semantic: dict[str, list[dict[str, object]]] = defaultdict(list)
    rows_by_relation: dict[str, list[dict[str, object]]] = defaultdict(list)
    semantic_definitions: dict[str, tuple[str, tuple[str, ...]]] = {}
    surface_ids: set[str] = set()
    for row in candidates:
        semantic_id = str(row["semantic_id"])
        surface_id = str(row["surface_id"])
        if surface_id in surface_ids:
            raise ValueError(f"Duplicate candidate surface_id: {surface_id}")
        surface_ids.add(surface_id)
        target_relation = str(row["target_relation"])
        edges = tuple(str(edge) for edge in row["edge_types"])
        definition = (target_relation, edges)
        previous = semantic_definitions.setdefault(semantic_id, definition)
        if previous != definition:
            raise ValueError(f"Inconsistent semantic definition: {semantic_id}")
        rows_by_semantic[semantic_id].append(row)
        rows_by_relation[target_relation].append(row)
    for semantic_id, (target_relation, edges) in semantic_definitions.items():
        if _semantic_id(target_relation, edges) != semantic_id:
            raise ValueError(f"Semantic hash mismatch: {semantic_id}")

    filler_pools: dict[str, list[dict[str, object]]] = {}
    for relation, rows in rows_by_relation.items():
        filler_pools[relation] = sorted(
            [
                row
                for row in rows
                if row["semantic_id"] not in forbidden_semantic_ids
                and row["surface_id"] not in common_surface_ids
            ],
            key=lambda row: _digest(
                f"tmlrs:filler:{row['surface_id']}"
            ),
        )

    targets = sorted(
        [
            row
            for row in vfour_targets
            if int(row["block_index"]) == 0
        ],
        key=lambda row: (
            int(row["block_pair_index"]),
            int(row["member_index"]),
        ),
    )
    if len(targets) != TARGET_COUNT:
        raise ValueError("Expected 40 block-0 T6 targets.")
    evaluation_by_semantic: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frozen_evaluation:
        evaluation_by_semantic[str(row["semantic_id"])].append(row)
    if len(frozen_evaluation) != EVAL_ROWS:
        raise ValueError("Expected 8,000 block-0 evaluation rows.")

    target_material: list[dict[str, object]] = []
    for target in targets:
        semantic_id = str(target["semantic_id"])
        groups = _complete_groups(rows_by_semantic[semantic_id])
        if len(groups) < 42:
            raise ValueError(f"Target lacks 42 complete groups: {semantic_id}")
        evaluation = [
            row for _, group_rows in groups[:33] for row in group_rows
        ] + groups[33][1][:2]
        observed_ids = [str(row["surface_id"]) for row in evaluation]
        expected_ids = [str(value) for value in target["evaluation_row_ids"]]
        if observed_ids != expected_ids:
            raise ValueError(f"V4 evaluation rows changed: {semantic_id}")
        materialized_ids = [
            str(row["surface_id"])
            for row in evaluation_by_semantic[semantic_id]
        ]
        if observed_ids != materialized_ids:
            raise ValueError(
                f"V4 evaluation manifest order changed: {semantic_id}"
            )
        exposure = [
            row
            for _, group_rows in groups[34:42]
            for row in group_rows
        ]
        if len(exposure) != SLOTS_PER_TARGET:
            raise ValueError(f"Exposure row count changed: {semantic_id}")
        definition = semantic_definitions[semantic_id]
        target_material.append(
            {
                **target,
                "edge_types": list(definition[1]),
                "evaluation_rows": evaluation,
                "exposure_rows": exposure,
            }
        )

    used_decoys: set[str] = set()
    decoy_material: list[dict[str, object]] = []
    for target in target_material:
        semantic_id = str(target["semantic_id"])
        relation = str(target["target_relation"])
        target_edges = list(target["edge_types"])
        eligible = [
            row
            for row in rows_by_relation[relation]
            if row["semantic_id"] not in forbidden_semantic_ids
            and row["surface_id"] not in common_surface_ids
            and row["surface_id"] not in used_decoys
        ]
        tier_two = sorted(
            [
                row
                for row in eligible
                if shared_edge_count(target_edges, row["edge_types"]) >= 2
            ],
            key=lambda row: _digest(
                f"tmlrs:t6:decoy:{semantic_id}:{row['surface_id']}"
            ),
        )
        selected = _round_robin(tier_two, SLOTS_PER_TARGET)
        tier = 2
        if len(selected) < SLOTS_PER_TARGET:
            tier_one = sorted(
                [
                    row
                    for row in eligible
                    if shared_edge_count(target_edges, row["edge_types"])
                    >= 1
                ],
                key=lambda row: _digest(
                    f"tmlrs:t6:decoy:{semantic_id}:{row['surface_id']}"
                ),
            )
            selected = _round_robin(tier_one, SLOTS_PER_TARGET)
            tier = 1
        for row in selected:
            used_decoys.add(str(row["surface_id"]))
        decoy_material.append(
            {
                **target,
                "tier": tier,
                "decoy_rows": selected,
                "included_main": len(selected) >= 24,
            }
        )

    used_fillers = {"C0": set(), "C2": set(used_decoys)}
    arm_slots: dict[str, list[dict[str, object]]] = {
        arm: [] for arm in ARMS
    }
    allocation: list[dict[str, object]] = []
    for target in decoy_material:
        relation = str(target["target_relation"])
        c0_rows = _take_fillers(
            filler_pools[relation],
            SLOTS_PER_TARGET,
            used_fillers["C0"],
        )
        c1_rows = list(target["exposure_rows"])
        decoys = list(target["decoy_rows"])
        c2_fillers = _take_fillers(
            filler_pools[relation],
            SLOTS_PER_TARGET - len(decoys),
            used_fillers["C2"],
        )
        c2_rows = decoys + c2_fillers
        arm_slots["C0"].extend(c0_rows)
        arm_slots["C1"].extend(c1_rows)
        arm_slots["C2"].extend(c2_rows)
        allocation.append(
            {
                "pattern_index": len(allocation),
                "semantic_id": target["semantic_id"],
                "target_relation": relation,
                "edge_types": list(target["edge_types"]),
                "tier": target["tier"],
                "decoy_count": len(decoys),
                "decoy_semantic_count": len(
                    {str(row["semantic_id"]) for row in decoys}
                ),
                "included_main": target["included_main"],
                "decoy_rows": decoys,
                "filler_rows_c0": c0_rows,
                "filler_rows_c2": c2_fillers,
                "exposure_rows": c1_rows,
            }
        )

    arm_rows = {
        arm: common_base + arm_slots[arm]
        for arm in ARMS
    }
    label_counts = {
        arm: Counter(str(row["target_relation"]) for row in rows)
        for arm, rows in arm_rows.items()
    }
    if any(len(rows) != TRAIN_ROWS for rows in arm_rows.values()):
        raise ValueError("T6 arm row count mismatch.")
    if not (
        label_counts["C0"] == label_counts["C1"] == label_counts["C2"]
    ):
        raise ValueError("T6 arm label distributions differ.")
    for arm, rows in arm_rows.items():
        ids = [str(row["surface_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"T6 arm {arm} contains duplicate surface rows.")

    evaluation_rows = [
        row
        for target in target_material
        for row in target["evaluation_rows"]
    ]
    return {
        "allocation": allocation,
        "arm_rows": arm_rows,
        "common_surface_ids": sorted(common_surface_ids),
        "evaluation_rows": evaluation_rows,
        "forbidden_semantic_ids": sorted(forbidden_semantic_ids),
        "label_counts": {
            arm: dict(sorted(counts.items()))
            for arm, counts in label_counts.items()
        },
        "target_count": TARGET_COUNT,
        "train_rows": TRAIN_ROWS,
        "evaluation_row_count": len(evaluation_rows),
    }
