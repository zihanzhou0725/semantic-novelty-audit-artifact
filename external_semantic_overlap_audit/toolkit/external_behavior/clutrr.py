"""CLUTRR adapter and deterministic paired-design materializer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml

from .common import (
    ARMS,
    CLUTRR_ROOT,
    MODEL_SEEDS,
    arm_assignment_bit,
    derive_seed,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
    write_lines,
)
from .model import encode_parts, normalized_input


CANDIDATE_CAP = 120_000
EVALUATION_ROWS_PER_PATTERN = 200
EXPOSURE_ROWS_PER_PATTERN = 64
TARGET_PATTERN_COUNT = 40
PAIR_COUNT = 20
COMMON_BASE_ROWS = 22_720
ROWS_PER_ARM = 24_000


def _ensure_upstream_imports() -> tuple[Any, Any, Any, Any]:
    sys.dont_write_bytecode = True
    root = str(CLUTRR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from clutrr.actors.ancestry import Ancestry
    from clutrr.relations.builder import RelationBuilder
    from clutrr.relations.templator import TemplatorSynthetic
    from clutrr.store.store import Store

    return Ancestry, RelationBuilder, TemplatorSynthetic, Store


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        attribute_store="attribute_store.json",
        boundary=True,
        combination_length=1,
        data_type="train",
        equal=False,
        holdout=False,
        max_child=4,
        max_levels=3,
        memory=0,
        min_child=4,
        noise_attributes=False,
        noise_disconnected=False,
        noise_irrelevant=False,
        noise_support=False,
        num_rows=CANDIDATE_CAP,
        output="external_behavior_v1",
        p_marry=1.0,
        question=0,
        question_store="question_store.yaml",
        relation_length=3,
        relations_store="relations_store.yaml",
        rules_store="rules_store.yaml",
        template_file="unused_for_synthetic_templates",
        template_split=False,
        unique_test_pattern=False,
        use_mturk_template=False,
    )


def _load_store(args: argparse.Namespace) -> Any:
    _, _, _, Store = _ensure_upstream_imports()
    original_load = yaml.load

    def compatible_load(stream: Any, *values: Any, **kwargs: Any) -> Any:
        if not values and "Loader" not in kwargs:
            kwargs["Loader"] = yaml.FullLoader
        return original_load(stream, *values, **kwargs)

    yaml.load = compatible_load
    try:
        return Store(args)
    finally:
        yaml.load = original_load


def relation_classes(store: Any) -> list[str]:
    return sorted(
        {
            gender_value["rel"]
            for relation_value in store.relations_store.values()
            for gender_value in relation_value.values()
        }
    )


def normalize_text(text: str, entity_names: list[str]) -> str:
    value = unicodedata.normalize("NFKC", text)
    for index, name in enumerate(entity_names):
        value = value.replace(f"[{name}]", f"entity_{index}")
    return " ".join(value.lower().split())


def semantic_serialization(
    target_relation: str,
    edge_types: list[str],
) -> str:
    return json.dumps(
        [target_relation, *edge_types],
        separators=(",", ":"),
        ensure_ascii=True,
    )


def reconstruct_semantics(row: dict[str, Any]) -> tuple[str, str]:
    edges = [tuple(int(value) for value in edge) for edge in row["story_edges"]]
    edge_types = [str(value) for value in row["edge_types"]]
    query = tuple(int(value) for value in row["query_edge"])
    if len(edges) != 3 or len(edge_types) != 3:
        raise ValueError("CLUTRR row does not contain a length-three clean path.")
    if any(edges[index][1] != edges[index + 1][0] for index in range(2)):
        raise ValueError("Stored CLUTRR story edges are not an ordered path.")
    if query != (edges[0][0], edges[-1][1]):
        raise ValueError("Stored CLUTRR query does not match the clean path.")
    canonical = semantic_serialization(row["target_relation"], edge_types)
    return canonical, sha256_bytes(canonical.encode("utf-8"))


def _surface_id(normalized: str, target_relation: str) -> str:
    return sha256_bytes(
        json.dumps(
            [normalized, target_relation],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )


def _render_puzzle(
    puzzle: Any,
    store: Any,
    templator_class: Any,
    candidate_index: int,
) -> dict[str, Any]:
    synthetic_templates: dict[str, list[str]] = {}
    for relation_value in store.relations_store.values():
        for gender_value in relation_value.values():
            synthetic_templates[gender_value["rel"]] = gender_value["p"]
    query_templates = {
        gender_value["rel"]: store.question_store["relational"]
        for relation_value in store.relations_store.values()
        for gender_value in relation_value.values()
    }

    story_templator = templator_class(
        templates=synthetic_templates,
        family=puzzle.anc.family_data,
    )
    story_text = puzzle.generate_text(
        stype="story",
        combination_length=1,
        templator=story_templator,
    )
    clean_story = " ".join(story_text)
    query_templator = templator_class(
        templates=query_templates,
        family=puzzle.anc.family_data,
    )
    text_query = " ".join(
        puzzle.generate_text(
            stype="query",
            combination_length=1,
            templator=query_templator,
        )
    ).replace("?.", "?")

    puzzle.convert_node_ids(stype="story")
    story_edges = [
        list(edge) for edge in puzzle.get_sorted_story_edges(stype="story")
    ]
    edge_types = list(puzzle.get_story_relations(stype="story"))
    query_edge = list(puzzle.get_sorted_query_edge())
    entity_names = [
        puzzle.anc.family_data[node_id].name
        for node_id in puzzle.story_sort_dict
    ]
    normalized_story = normalize_text(clean_story, entity_names)
    normalized_query = normalize_text(text_query, entity_names)
    model_input = normalized_input(normalized_story, normalized_query)
    target_relation = puzzle.target_edge_rel
    canonical = semantic_serialization(target_relation, edge_types)
    semantic_id = sha256_bytes(canonical.encode("utf-8"))
    order_hash = sha256_bytes(
        (model_input + canonical + target_relation).encode("utf-8")
    )
    surface_id = _surface_id(model_input, target_relation)
    row = {
        "candidate_index": candidate_index,
        "candidate_row_id": sha256_bytes(
            json.dumps(
                [candidate_index, surface_id, semantic_id],
                separators=(",", ":"),
            ).encode("ascii")
        ),
        "clean_story": clean_story,
        "edge_types": edge_types,
        "entity_names": entity_names,
        "normalized_input": model_input,
        "normalized_query": normalized_query,
        "normalized_story": normalized_story,
        "query_edge": query_edge,
        "row_order_sha256": order_hash,
        "semantic_id": semantic_id,
        "semantic_serialization": canonical,
        "sequence_length": len(encode_parts(normalized_story, normalized_query)),
        "story_edges": story_edges,
        "surface_id": surface_id,
        "target_relation": target_relation,
        "text_query": text_query,
    }
    reconstructed, reconstructed_id = reconstruct_semantics(row)
    if (reconstructed, reconstructed_id) != (canonical, semantic_id):
        raise AssertionError("CLUTRR semantic reconstruction failed.")
    return row


def generate_candidates(
    output_path: Path,
    *,
    candidate_cap: int = CANDIDATE_CAP,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    Ancestry, RelationBuilder, TemplatorSynthetic, _ = _ensure_upstream_imports()
    args = _args()
    args.num_rows = candidate_cap
    python_seed = derive_seed("clutrr:generator:python")
    numpy_registered_seed = derive_seed("clutrr:generator:numpy")
    numpy_seed = numpy_registered_seed % (2**32)
    random.seed(python_seed)
    np.random.seed(numpy_seed)
    store = _load_store(args)
    ancestry = Ancestry(args, store)
    builder = RelationBuilder(args, store, ancestry)

    unique_surfaces: dict[tuple[str, str], dict[str, Any]] = {}
    semantic_by_surface: dict[tuple[str, str], str] = {}
    generated = 0
    build_iterations = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii") as handle:
        while generated < candidate_cap:
            build_iterations += 1
            if not builder.build():
                builder.reset_puzzle()
                builder.anc.next_flip()
                continue
            builder.add_facts()
            builder.generate_puzzles()
            for puzzle in builder.puzzles.values():
                row = _render_puzzle(
                    puzzle,
                    store,
                    TemplatorSynthetic,
                    generated,
                )
                key = (row["normalized_input"], row["target_relation"])
                prior_semantic = semantic_by_surface.get(key)
                if prior_semantic is not None and prior_semantic != row["semantic_id"]:
                    raise ValueError(
                        "Identical normalized CLUTRR surface has conflicting "
                        "semantic signatures."
                    )
                semantic_by_surface[key] = row["semantic_id"]
                if key not in unique_surfaces:
                    unique_surfaces[key] = copy.deepcopy(row)
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                generated += 1
                if progress is not None and (
                    generated % 10_000 == 0 or generated == candidate_cap
                ):
                    progress(generated, candidate_cap)
                if generated == candidate_cap:
                    break
            builder.reset_puzzle()
            builder.anc.next_flip()
    temporary.replace(output_path)

    deduplicated = sorted(
        unique_surfaces.values(),
        key=lambda row: (row["row_order_sha256"], row["surface_id"]),
    )
    return deduplicated, {
        "build_iterations": build_iterations,
        "candidate_cap": candidate_cap,
        "generated_rows": generated,
        "generator_configuration": {
            "combination_length": args.combination_length,
            "max_child": args.max_child,
            "max_levels": args.max_levels,
            "min_child": args.min_child,
            "noise_facts": "none",
            "p_marry": args.p_marry,
            "relation_length": args.relation_length,
            "task": 1,
            "templates": "synthetic",
        },
        "numpy_registered_seed": numpy_registered_seed,
        "numpy_seed_modulo_2_32": numpy_seed,
        "python_seed": python_seed,
        "raw_manifest_sha256": sha256_file(output_path),
        "relation_classes": relation_classes(store),
        "unique_surface_rows": len(deduplicated),
    }


def eligible_patterns(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["semantic_id"]].append(row)
    eligible: list[dict[str, Any]] = []
    required = EVALUATION_ROWS_PER_PATTERN + EXPOSURE_ROWS_PER_PATTERN
    for semantic_id, pattern_rows in grouped.items():
        labels = {row["target_relation"] for row in pattern_rows}
        if len(labels) != 1:
            raise ValueError("A CLUTRR semantic pattern has multiple targets.")
        pattern_rows.sort(
            key=lambda row: (row["row_order_sha256"], row["surface_id"])
        )
        if len(pattern_rows) >= required:
            eligible.append(
                {
                    "available_rows": len(pattern_rows),
                    "semantic_id": semantic_id,
                    "target_relation": next(iter(labels)),
                }
            )
    eligible.sort(key=lambda row: row["semantic_id"])
    return grouped, eligible


def select_pairs(eligible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    availability = {
        row["semantic_id"]: int(row["available_rows"]) for row in eligible
    }
    for row in eligible:
        by_label[row["target_relation"]].append(row["semantic_id"])
    candidates: list[dict[str, Any]] = []
    for target_relation in sorted(by_label):
        semantic_ids = sorted(by_label[target_relation])
        for offset in range(0, len(semantic_ids) - 1, 2):
            members = semantic_ids[offset : offset + 2]
            pair_hash = sha256_bytes("".join(members).encode("ascii"))
            candidates.append(
                {
                    "member_0": members[0],
                    "member_0_available_rows": availability[members[0]],
                    "member_1": members[1],
                    "member_1_available_rows": availability[members[1]],
                    "pair_sha256": pair_hash,
                    "target_relation": target_relation,
                }
            )
    candidates.sort(key=lambda row: row["pair_sha256"])
    selected = candidates[:PAIR_COUNT]
    for pair_index, row in enumerate(selected):
        row["pair_index"] = pair_index
    return selected


def allocate_targets(
    grouped: dict[str, list[dict[str, Any]]],
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
            rows = grouped[semantic_id]
            eval_rows = rows[:EVALUATION_ROWS_PER_PATTERN]
            exposure_rows = rows[
                EVALUATION_ROWS_PER_PATTERN :
                EVALUATION_ROWS_PER_PATTERN + EXPOSURE_ROWS_PER_PATTERN
            ]
            targets.append(
                {
                    "available_rows": len(rows),
                    "evaluation_row_ids": [
                        row["surface_id"] for row in eval_rows
                    ],
                    "exposure_row_ids": [
                        row["surface_id"] for row in exposure_rows
                    ],
                    "member_index": member_index,
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
                    "pair_member": member_index,
                }
                for index, row in enumerate(eval_rows)
            )
            exposure.extend(
                {
                    **row,
                    "allocation_index": index,
                    "pair_index": pair["pair_index"],
                    "pair_member": member_index,
                }
                for index, row in enumerate(exposure_rows)
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
    by_label_pattern: dict[str, dict[str, deque[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(deque)
    )
    for row in rows:
        if row["semantic_id"] in target_ids:
            continue
        by_label_pattern[row["target_relation"]][row["semantic_id"]].append(row)
    labels = sorted(by_label_pattern)
    pattern_ids = {
        label: sorted(by_label_pattern[label]) for label in labels
    }
    pattern_positions = {label: 0 for label in labels}
    output: list[dict[str, Any]] = []
    while len(output) < COMMON_BASE_ROWS:
        made_progress = False
        for label in labels:
            ids = pattern_ids[label]
            if not ids:
                continue
            attempts = 0
            while attempts < len(ids):
                position = pattern_positions[label] % len(ids)
                semantic_id = ids[position]
                pattern_positions[label] = (position + 1) % len(ids)
                attempts += 1
                queue = by_label_pattern[label][semantic_id]
                if queue:
                    output.append(queue.popleft())
                    made_progress = True
                    break
            if len(output) == COMMON_BASE_ROWS:
                break
        if not made_progress:
            break
    return output


def _exposure_by_member(
    exposure: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in exposure:
        grouped[(row["pair_index"], row["pair_member"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["allocation_index"])
    return grouped


def materialize_arm_manifests(
    data_dir: Path,
    common_base: list[dict[str, Any]],
    exposure: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        len(common_base) != COMMON_BASE_ROWS
        or len(pairs) != PAIR_COUNT
        or len(exposure) != TARGET_PATTERN_COUNT * EXPOSURE_ROWS_PER_PATTERN
    ):
        return []
    exposures = _exposure_by_member(exposure)
    summaries: list[dict[str, Any]] = []
    common_ids = [row["surface_id"] for row in common_base]
    common_labels = [row["target_relation"] for row in common_base]
    for model_seed in MODEL_SEEDS:
        paired_arm_ids: dict[str, list[str]] = {}
        paired_arm_labels: dict[str, list[str]] = {}
        assignments: dict[str, dict[str, int]] = {}
        for arm in ARMS:
            row_ids = list(common_ids)
            labels = list(common_labels)
            assignments[arm] = {}
            for pair in pairs:
                pair_index = int(pair["pair_index"])
                bit = arm_assignment_bit(model_seed, pair_index)
                member = bit if arm == "A" else 1 - bit
                assignments[arm][str(pair_index)] = member
                selected = exposures[(pair_index, member)]
                row_ids.extend(row["surface_id"] for row in selected)
                labels.extend(row["target_relation"] for row in selected)
            paired_arm_ids[arm] = row_ids
            paired_arm_labels[arm] = labels
            path = data_dir / f"clutrr_train_seed{model_seed}_arm{arm}.txt"
            write_lines(path, row_ids)
            summaries.append(
                {
                    "arm": arm,
                    "exposed_pair_members": assignments[arm],
                    "label_counts": dict(sorted(Counter(labels).items())),
                    "manifest": path.name,
                    "manifest_sha256": sha256_file(path),
                    "model_seed": model_seed,
                    "rows": len(row_ids),
                }
            )
        if Counter(paired_arm_labels["A"]) != Counter(paired_arm_labels["B"]):
            raise AssertionError("CLUTRR A/B label counts differ.")
        if len(paired_arm_ids["A"]) != len(paired_arm_ids["B"]):
            raise AssertionError("CLUTRR A/B row counts differ.")
        for pair in pairs:
            index = str(pair["pair_index"])
            if assignments["A"][index] == assignments["B"][index]:
                raise AssertionError("CLUTRR pair assignment is not complementary.")
    return summaries


def materialize_clutrr(
    data_dir: Path,
    *,
    candidate_cap: int = CANDIDATE_CAP,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "clutrr_candidates_raw.jsonl"
    rows, generation = generate_candidates(
        raw_path,
        candidate_cap=candidate_cap,
        progress=progress,
    )
    deduplicated_path = data_dir / "clutrr_candidates_deduplicated.jsonl"
    write_jsonl(deduplicated_path, rows)
    grouped, eligible = eligible_patterns(rows)
    pairs = select_pairs(eligible)
    targets, evaluation, exposure = allocate_targets(grouped, pairs)
    target_ids = {row["semantic_id"] for row in targets}
    common_base = allocate_common_base(rows, target_ids)

    paths = {
        "eligible_patterns": data_dir / "clutrr_eligible_patterns.jsonl",
        "pairs": data_dir / "clutrr_pairs.jsonl",
        "targets": data_dir / "clutrr_targets.jsonl",
        "evaluation": data_dir / "clutrr_evaluation.jsonl",
        "exposure": data_dir / "clutrr_exposure.jsonl",
        "common_base": data_dir / "clutrr_common_base.jsonl",
    }
    write_jsonl(paths["eligible_patterns"], eligible)
    write_jsonl(paths["pairs"], pairs)
    write_jsonl(paths["targets"], targets)
    write_jsonl(paths["evaluation"], evaluation)
    write_jsonl(paths["exposure"], exposure)
    write_jsonl(paths["common_base"], common_base)
    arms = materialize_arm_manifests(
        data_dir,
        common_base,
        exposure,
        pairs,
    )
    arm_summary_path = data_dir / "clutrr_arm_summary.json"
    write_json(arm_summary_path, arms)

    allocated = evaluation + exposure + common_base
    maximum_sequence_length = max(
        (int(row["sequence_length"]) for row in allocated),
        default=0,
    )
    selected_pair_count = len(pairs)
    selected_target_count = len(targets)
    design_ready = (
        generation["generated_rows"] == candidate_cap
        and selected_pair_count == PAIR_COUNT
        and selected_target_count == TARGET_PATTERN_COUNT
        and len(evaluation)
        == TARGET_PATTERN_COUNT * EVALUATION_ROWS_PER_PATTERN
        and len(exposure)
        == TARGET_PATTERN_COUNT * EXPOSURE_ROWS_PER_PATTERN
        and len(common_base) == COMMON_BASE_ROWS
        and len(arms) == len(MODEL_SEEDS) * len(ARMS)
        and maximum_sequence_length <= 512
    )
    files = [
        raw_path,
        deduplicated_path,
        *paths.values(),
        arm_summary_path,
        *sorted(data_dir.glob("clutrr_train_seed*_arm*.txt")),
    ]
    report = {
        "arm_manifest_count": len(arms),
        "candidate_generation": generation,
        "common_base_rows": len(common_base),
        "deduplicated_manifest_sha256": sha256_file(deduplicated_path),
        "design_ready_for_validation": design_ready,
        "eligible_pattern_count": len(eligible),
        "evaluation_rows": len(evaluation),
        "exposure_rows": len(exposure),
        "files_sha256": {
            path.name: sha256_file(path) for path in sorted(files)
        },
        "maximum_allocated_sequence_length": maximum_sequence_length,
        "pair_count": selected_pair_count,
        "status": "materialized",
        "target_count": selected_target_count,
    }
    write_json(data_dir / "clutrr_materialization_report.json", report)
    return report
