#!/usr/bin/env python3
"""Audit Tensor Logic Experiment 3B's FB15k-237 compositional split."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from quality_p1_common import EXTERNAL_ROOT, PROTOCOL_DIR, ROOT, sha256_file, write_json


SCHEMA_VERSION = "semantic-novelty-audit-tensor-logic-fb15k237-v1"
CODE_REVISION = "9b497c27992843bfbfdd99801dab6881cebb465e"
DATA_REVISION = "2e440e0f9c687314d5ff67ead68ce985dc446e3a"
CODE_ROOT = EXTERNAL_ROOT / "upstream" / "p3_tensor_logic"
DATA_ROOT = (
    EXTERNAL_ROOT
    / "upstream"
    / "p3_knowledge_graph_embedding"
    / "data"
    / "FB15k-237"
)
CANDIDATE_MANIFEST = (
    PROTOCOL_DIR / "p3" / "candidate_manifest.json"
)
OUTPUT_DIR = (
    EXTERNAL_ROOT
    / "analysis"
    / "quality-enhancement-v1"
    / "p3"
    / "tensor_logic_fb15k237"
)


def fail(message: str) -> None:
    raise SystemExit(f"Tensor Logic audit failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def semantic_id(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
    ).strip()


def git_status(path: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--short"],
        cwd=path,
        text=True,
    ).strip()


def load_triples(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            require(len(parts) == 3, f"{path.name}:{line_number} is not a triple")
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def extract_composition_paths(
    train_triples: list[tuple[str, str, str]],
    *,
    num_val: int = 1000,
    num_test: int = 1000,
    seed: int = 42,
) -> tuple[
    list[tuple[str, str, str, str, str, str]],
    list[tuple[str, str, str, str, str, str]],
    set[tuple[str, str, str]],
]:
    """Byte-for-byte algorithmic reproduction of tensor_kgraph.py."""

    rng = np.random.RandomState(seed)
    forward: dict[str, list[tuple[str, str]]] = defaultdict(list)
    direct_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    for head, relation, tail in train_triples:
        forward[head].append((relation, tail))
        direct_map[(head, tail)].add(relation)

    nodes = list(forward)
    rng.shuffle(nodes)
    paths: list[tuple[str, str, str, str, str, str]] = []
    unique_pairs: set[tuple[str, str]] = set()
    links_to_remove: set[tuple[str, str, str]] = set()
    target_total = num_val + num_test
    for head in nodes:
        if len(paths) >= target_total:
            break
        for relation_1, middle in forward[head]:
            if middle not in forward:
                continue
            for relation_2, tail in forward[middle]:
                if head == middle or middle == tail or head == tail:
                    continue
                if (head, tail) in unique_pairs:
                    continue
                direct_relations = direct_map.get((head, tail))
                if not direct_relations or len(direct_relations) != 1:
                    continue
                target_relation = next(iter(direct_relations))
                unique_pairs.add((head, tail))
                paths.append(
                    (
                        head,
                        relation_1,
                        middle,
                        relation_2,
                        tail,
                        target_relation,
                    )
                )
                links_to_remove.add((head, target_relation, tail))
                if len(paths) >= target_total:
                    break
            if len(paths) >= target_total:
                break
    rng.shuffle(paths)
    return (
        paths[:num_val],
        paths[num_val : num_val + num_test],
        links_to_remove,
    )


def build_forward(
    triples: set[tuple[str, str, str]],
) -> dict[str, dict[str, set[str]]]:
    forward: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for head, relation, tail in triples:
        forward[head][relation].add(tail)
    return forward


def build_tail_map(
    triples: list[tuple[str, str, str]] | set[tuple[str, str, str]],
) -> dict[tuple[str, str], set[str]]:
    values: dict[tuple[str, str], set[str]] = defaultdict(set)
    for head, relation, tail in triples:
        values[(head, relation)].add(tail)
    return values


def schema_witness_counts(
    forward: dict[str, dict[str, set[str]]],
) -> Counter[tuple[str, str, str]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for head, outgoing in forward.items():
        for relation_1, middles in outgoing.items():
            for middle in middles:
                for relation_2, tails in forward.get(middle, {}).items():
                    for tail in tails:
                        for target_relation, direct_tails in outgoing.items():
                            if tail in direct_tails:
                                counts[
                                    (
                                        relation_1,
                                        relation_2,
                                        target_relation,
                                    )
                                ] += 1
    return counts


def executable_intermediates(
    forward: dict[str, dict[str, set[str]]],
    head: str,
    relation_1: str,
    relation_2: str,
    tail: str,
) -> list[str]:
    return sorted(
        middle
        for middle in forward.get(head, {}).get(relation_1, set())
        if tail in forward.get(middle, {}).get(relation_2, set())
    )


def validate_candidate_manifest() -> dict[str, Any]:
    p3_checksums = PROTOCOL_DIR / "p3" / "SHA256SUMS"
    for line in p3_checksums.read_text(encoding="ascii").splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        path = PROTOCOL_DIR / "p3" / filename
        require(path.is_file(), f"missing P3 artifact: {filename}")
        require(sha256_file(path) == digest, f"P3 artifact changed: {filename}")
    manifest = json.loads(CANDIDATE_MANIFEST.read_text(encoding="ascii"))
    require(
        manifest["status"]
        == "frozen_before_candidate_dataset_download_or_semantic_extraction",
        "candidate manifest was not frozen before audit",
    )
    require(
        manifest["candidate_semantic_extraction_started"] is False,
        "candidate manifest records earlier semantic extraction",
    )
    candidates = [
        candidate
        for candidate in manifest["candidates"]
        if candidate["candidate_id"] == "7b1fa5fbdb1da4ee"
    ]
    require(len(candidates) == 1, "Tensor Logic candidate is not uniquely frozen")
    candidate = candidates[0]
    require(
        candidate["code_revision"]["revision"] == CODE_REVISION,
        "Tensor Logic code revision changed",
    )
    require(
        candidate["dataset_revision"]["source_revision"] == DATA_REVISION,
        "FB15k-237 source revision changed",
    )
    require(candidate["audit_order"] == 1, "Tensor Logic audit order changed")
    return candidate


def path_rows(
    split: str,
    paths: list[tuple[str, str, str, str, str, str]],
    original_train: set[tuple[str, str, str]],
    scrubbed_train: set[tuple[str, str, str]],
    scrubbed_forward: dict[str, dict[str, set[str]]],
    schema_counts: Counter[tuple[str, str, str]],
    filter_all: dict[tuple[str, str], set[str]],
    filter_scrubbed: dict[tuple[str, str], set[str]],
    other_split_schemas: set[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        (
            head,
            relation_1,
            selected_middle,
            relation_2,
            tail,
            target_relation,
        ) = path
        first_edge = (head, relation_1, selected_middle)
        second_edge = (selected_middle, relation_2, tail)
        direct_edge = (head, target_relation, tail)
        first_retained = first_edge in scrubbed_train
        second_retained = second_edge in scrubbed_train
        intermediates = executable_intermediates(
            scrubbed_forward,
            head,
            relation_1,
            relation_2,
            tail,
        )
        schema = (relation_1, relation_2, target_relation)
        endpoint_signature = {
            "head": head,
            "ordered_relations": [relation_1, relation_2],
            "tail": tail,
        }
        full_query_signature = {
            **endpoint_signature,
            "target_relation": target_relation,
        }
        all_other_tails = filter_all.get((head, target_relation), set()) - {
            tail
        }
        scrubbed_other_tails = filter_scrubbed.get(
            (head, target_relation), set()
        ) - {tail}
        extra_filter_tails = all_other_tails - scrubbed_other_tails
        rows.append(
            {
                "split": split,
                "eval_index": index,
                "head": head,
                "relation_1": relation_1,
                "selected_middle": selected_middle,
                "relation_2": relation_2,
                "tail": tail,
                "target_relation": target_relation,
                "direct_edge_in_original_train": direct_edge in original_train,
                "direct_edge_in_scrubbed_train": direct_edge in scrubbed_train,
                "selected_first_edge_retained": first_retained,
                "selected_second_edge_retained": second_retained,
                "selected_path_retained": first_retained and second_retained,
                "executable_witness_count": len(intermediates),
                "executable_witness_intermediates_json": canonical_json(
                    intermediates
                ),
                "endpoint_path_witness_retained": bool(intermediates),
                "endpoint_path_signature_sha256": semantic_id(
                    endpoint_signature
                ),
                "full_query_signature_sha256": semantic_id(
                    full_query_signature
                ),
                "relation_schema_signature_sha256": semantic_id(
                    {
                        "ordered_relations": [relation_1, relation_2],
                        "target_relation": target_relation,
                    }
                ),
                "relation_schema_training_witness_count": schema_counts[
                    schema
                ],
                "relation_schema_seen_in_scrubbed_train": (
                    schema_counts[schema] > 0
                ),
                "relation_schema_seen_in_other_eval_split": (
                    schema in other_split_schemas
                ),
                "same_relation_reverse_edge_retained": (
                    (tail, target_relation, head) in scrubbed_train
                ),
                "scrubbed_filter_other_tail_count": len(
                    scrubbed_other_tails
                ),
                "published_filter_other_tail_count": len(all_other_tails),
                "filter_only_extra_tail_count": len(extra_filter_tails),
                "semantic_status": (
                    "witnessed_endpoint_path_in_scrubbed_train"
                    if intermediates
                    else "no_retained_exact_two_hop_witness_not_proven_novel"
                ),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chosen_edge_counts = Counter(
        int(row["selected_first_edge_retained"])
        + int(row["selected_second_edge_retained"])
        for row in rows
    )
    return {
        "rows": len(rows),
        "direct_edges_in_original_train": sum(
            row["direct_edge_in_original_train"] for row in rows
        ),
        "direct_edges_in_scrubbed_train": sum(
            row["direct_edge_in_scrubbed_train"] for row in rows
        ),
        "selected_path_retained": sum(
            row["selected_path_retained"] for row in rows
        ),
        "selected_path_retained_edge_count": {
            str(count): chosen_edge_counts[count] for count in (0, 1, 2)
        },
        "endpoint_path_witness_retained": sum(
            row["endpoint_path_witness_retained"] for row in rows
        ),
        "no_retained_exact_two_hop_witness": sum(
            not row["endpoint_path_witness_retained"] for row in rows
        ),
        "relation_schema_seen_in_scrubbed_train": sum(
            row["relation_schema_seen_in_scrubbed_train"] for row in rows
        ),
        "relation_schema_seen_in_other_eval_split": sum(
            row["relation_schema_seen_in_other_eval_split"] for row in rows
        ),
        "same_relation_reverse_edge_retained": sum(
            row["same_relation_reverse_edge_retained"] for row in rows
        ),
        "rows_with_filter_only_extra_tails": sum(
            row["filter_only_extra_tail_count"] > 0 for row in rows
        ),
        "filter_only_extra_tail_total": sum(
            row["filter_only_extra_tail_count"] for row in rows
        ),
    }


def build_report(report: dict[str, Any]) -> str:
    test = report["split_audit"]["test"]
    validation = report["split_audit"]["validation"]
    return "\n".join(
        [
            "# Tensor Logic FB15k-237 Compositional Split 语义审计",
            "",
            "范围：冻结的 Experiment 3B，代码 revision "
            f"`{CODE_REVISION}`，数据 revision `{DATA_REVISION}`。",
            "",
            "## 结论",
            "",
            "- **Protocol defect detected**：direct-edge withholding 成立，"
            "但“每条评估样本均保留可执行两跳路径”的前提失败。",
            "- 本结果只审计 split 与语义 witness；未重跑 H200 训练，"
            "不对论文 MRR 作替代估计。",
            "",
            "## Test split",
            "",
            f"- Direct edge 成功删除：{test['rows'] - test['direct_edges_in_scrubbed_train']}/"
            f"{test['rows']}。",
            f"- 指定中间点的两条边均保留：{test['selected_path_retained']}/"
            f"{test['rows']}。",
            f"- 允许任意替代中间点后，精确 `(a,r1,r2,c)` witness："
            f"{test['endpoint_path_witness_retained']}/{test['rows']}。",
            f"- 无任何保留两跳 witness："
            f"{test['no_retained_exact_two_hop_witness']}/{test['rows']}；"
            "这些行不能称为 novel，只能称为 unresolved。",
            f"- `(r1,r2 -> r_direct)` schema 在 scrubbed train 中出现："
            f"{test['relation_schema_seen_in_scrubbed_train']}/{test['rows']}。",
            "",
            "指定 witness 被级联删除的分布：",
            "",
            f"- 0/2 edges retained："
            f"{test['selected_path_retained_edge_count']['0']}",
            f"- 1/2 edges retained："
            f"{test['selected_path_retained_edge_count']['1']}",
            f"- 2/2 edges retained："
            f"{test['selected_path_retained_edge_count']['2']}",
            "",
            "## Validation split",
            "",
            f"- 指定路径完整保留：{validation['selected_path_retained']}/"
            f"{validation['rows']}。",
            f"- 任意精确两跳 witness 保留："
            f"{validation['endpoint_path_witness_retained']}/"
            f"{validation['rows']}。",
            "",
            "## 原因",
            "",
            "生成器先选出 2,000 个 endpoint pairs，再统一删除这 2,000 条"
            " direct edges。代码没有重新验证待删除边是否同时是其他评估路径"
            " 的第一跳或第二跳，因此删除操作破坏了大量先前选中的 witness。",
            "",
            "## 行为证据边界",
            "",
            "- 仓库没有发布 composition checkpoint、逐条 ranks/predictions、"
            "训练日志或机器可读结果 manifest。",
            "- 公开脚本固定一个训练 seed，无法从仓库产物核验论文中"
            "“all random seeds”表述。",
            "- 因此该候选可计为一个完成的 protocol-level semantic audit，"
            "但不能计为 P3 所需的 credible behavioral evidence。",
            "",
        ]
    )


def main() -> int:
    candidate = validate_candidate_manifest()
    require(git_head(CODE_ROOT) == CODE_REVISION, "code checkout revision")
    require(git_head(DATA_ROOT.parents[1]) == DATA_REVISION, "data checkout revision")
    require(git_status(CODE_ROOT) == "", "code checkout is dirty")
    require(git_status(DATA_ROOT.parents[1]) == "", "data checkout is dirty")

    code_path = CODE_ROOT / "tensor_kgraph.py"
    code_text = code_path.read_text(encoding="utf-8")
    for required_text in (
        '"num_val_paths": 1000',
        '"num_test_paths": 1000',
        '"seed_paths": 42',
        "extract_composition_from_train_only(",
        "train_comp = [tr for tr in data[\"train\"] if tr not in links_to_remove_train]",
        "filter_tail=filter_tail_all",
    ):
        require(required_text in code_text, f"code invariant missing: {required_text}")

    train = load_triples(DATA_ROOT / "train.txt")
    validation_triples = load_triples(DATA_ROOT / "valid.txt")
    test_triples = load_triples(DATA_ROOT / "test.txt")
    require(
        (len(train), len(validation_triples), len(test_triples))
        == (272115, 17535, 20466),
        "FB15k-237 split counts changed",
    )
    require(len(set(train)) == len(train), "duplicate train triples")
    entities = {
        entity
        for triples in (train, validation_triples, test_triples)
        for head, _, tail in triples
        for entity in (head, tail)
    }
    relations = {
        relation
        for triples in (train, validation_triples, test_triples)
        for _, relation, _ in triples
    }
    require(len(entities) == 14541, "entity census changed")
    require(len(relations) == 237, "relation census changed")

    validation_paths, test_paths, removed_links = extract_composition_paths(
        train
    )
    require(len(validation_paths) == 1000, "validation path count")
    require(len(test_paths) == 1000, "test path count")
    require(len(removed_links) == 2000, "removed direct-link count")
    original_train = set(train)
    scrubbed_train = original_train - removed_links
    require(len(scrubbed_train) == 270115, "scrubbed train count")
    require(not (scrubbed_train & removed_links), "direct links remain")

    scrubbed_forward = build_forward(scrubbed_train)
    schema_counts = schema_witness_counts(scrubbed_forward)
    validation_schemas = {
        (relation_1, relation_2, target_relation)
        for _, relation_1, _, relation_2, _, target_relation in validation_paths
    }
    test_schemas = {
        (relation_1, relation_2, target_relation)
        for _, relation_1, _, relation_2, _, target_relation in test_paths
    }
    filter_all = build_tail_map(
        train + validation_triples + test_triples
    )
    filter_scrubbed = build_tail_map(scrubbed_train)
    validation_rows = path_rows(
        "validation",
        validation_paths,
        original_train,
        scrubbed_train,
        scrubbed_forward,
        schema_counts,
        filter_all,
        filter_scrubbed,
        test_schemas,
    )
    test_rows = path_rows(
        "test",
        test_paths,
        original_train,
        scrubbed_train,
        scrubbed_forward,
        schema_counts,
        filter_all,
        filter_scrubbed,
        validation_schemas,
    )
    validation_summary = summarize_rows(validation_rows)
    test_summary = summarize_rows(test_rows)

    require(
        test_summary["direct_edges_in_scrubbed_train"] == 0,
        "test direct-edge withholding failed",
    )
    require(
        test_summary["selected_path_retained"] == 379,
        "test selected-witness census changed",
    )
    require(
        test_summary["endpoint_path_witness_retained"] == 484,
        "test endpoint-witness census changed",
    )
    require(
        test_summary["relation_schema_seen_in_scrubbed_train"] == 999,
        "test schema-exposure census changed",
    )
    require(
        validation_summary["selected_path_retained"] == 336,
        "validation selected-witness census changed",
    )
    require(
        validation_summary["endpoint_path_witness_retained"] == 441,
        "validation endpoint-witness census changed",
    )

    repo_names = {path.name for path in CODE_ROOT.iterdir()}
    published_behavioral_artifacts = {
        "composition_checkpoint": "best_model_comp.pth" in repo_names,
        "per_item_ranks_or_predictions": any(
            "rank" in name.lower() or "prediction" in name.lower()
            for name in repo_names
        ),
        "training_log": any("log" in name.lower() for name in repo_names),
        "machine_readable_result_manifest": any(
            name.endswith(".json") or name.endswith(".csv")
            for name in repo_names
        ),
    }
    require(
        not any(published_behavioral_artifacts.values()),
        "unexpected behavioral artifact found; audit protocol needs amendment",
    )

    report = {
        "audit_date": "2026-10-09",
        "behavioral_reproducibility": {
            "artifact_census": published_behavioral_artifacts,
            "checkpoint_revision": candidate["checkpoint_revision_if_any"],
            "code_training_seed_count": 1,
            "code_training_seed": 13579,
            "decision": "no_go_missing_checkpoint_predictions_and_logs",
            "published_aggregate_metrics": {
                "hits_at_1": 0.2400,
                "hits_at_3": 0.3690,
                "hits_at_10": 0.5220,
                "mrr": 0.3346,
                "source": "Report.pdf Table 4",
                "status": "reported_not_recomputed",
            },
        },
        "candidate_id": candidate["candidate_id"],
        "claim_gate": {
            "audit_completeness": "exact_for_all_1000_generated_test_paths",
            "candidate_contribution_to_p3": {
                "complete_public_semantic_audit": True,
                "credible_behavioral_evidence": False,
            },
            "decision": "protocol_defect_detected_behavioral_no_go",
            "direct_edge_withholding": "passed",
            "every_test_case_has_retained_two_hop_path": "failed",
            "relation_schema_novelty": "failed_999_of_1000_seen",
        },
        "claim_under_audit": candidate["audited_claim"],
        "dataset": {
            "entities": len(entities),
            "relations": len(relations),
            "scrubbed_train_triples": len(scrubbed_train),
            "test_triples": len(test_triples),
            "train_triples": len(train),
            "validation_triples": len(validation_triples),
        },
        "generator": {
            "direct_links_removed": len(removed_links),
            "num_test_paths": len(test_paths),
            "num_validation_paths": len(validation_paths),
            "path_seed": 42,
            "selection": (
                "first 2000 unique endpoint pairs reached after shuffling "
                "training heads, followed by a second path shuffle"
            ),
        },
        "input_sha256": {
            relative(CANDIDATE_MANIFEST): sha256_file(CANDIDATE_MANIFEST),
            relative(code_path): sha256_file(code_path),
            relative(CODE_ROOT / "Report.pdf"): sha256_file(
                CODE_ROOT / "Report.pdf"
            ),
            relative(DATA_ROOT / "train.txt"): sha256_file(
                DATA_ROOT / "train.txt"
            ),
            relative(DATA_ROOT / "valid.txt"): sha256_file(
                DATA_ROOT / "valid.txt"
            ),
            relative(DATA_ROOT / "test.txt"): sha256_file(
                DATA_ROOT / "test.txt"
            ),
        },
        "reproducibility_risks": [
            (
                "tensor_kgraph.py downloads FB15k-237 from the mutable master "
                "branch without a commit or file hash"
            ),
            (
                "the repository does not persist generated path manifests, "
                "checkpoints, ranks, predictions, or training logs"
            ),
            (
                "composition filtering uses original train plus canonical "
                "validation/test truth maps rather than only scrubbed training facts"
            ),
            (
                "the public script executes one training seed although the paper "
                "states that compositional generalization emerged across random seeds"
            ),
        ],
        "schema_version": SCHEMA_VERSION,
        "semantic_definitions": {
            "endpoint_path_signature": (
                "SHA256 canonical JSON of head, ordered [r1,r2], and tail"
            ),
            "full_query_signature": (
                "endpoint path signature fields plus target direct relation"
            ),
            "relation_schema_signature": (
                "SHA256 canonical JSON of ordered [r1,r2] and target relation"
            ),
            "unresolved_rule": (
                "absence of a retained exact two-hop witness does not establish "
                "semantic novelty"
            ),
        },
        "split_audit": {
            "relation_schema_training_witness_instances": int(
                sum(schema_counts.values())
            ),
            "relation_schema_training_witness_types": len(schema_counts),
            "test": test_summary,
            "test_unique_relation_schemas": len(test_schemas),
            "validation": validation_summary,
            "validation_test_relation_schema_intersection": len(
                validation_schemas & test_schemas
            ),
            "validation_unique_relation_schemas": len(validation_schemas),
        },
        "status": "complete",
        "title": candidate["title"],
    }

    output_files = (
        "audit_report.json",
        "eval_paths.csv",
        "report_cn.md",
        "summary_by_split.csv",
    )
    for filename in (*output_files, "SHA256SUMS"):
        require(
            not (OUTPUT_DIR / filename).exists(),
            f"audit output already exists: {filename}",
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "audit_report.json", report)
    all_rows = validation_rows + test_rows
    with (OUTPUT_DIR / "eval_paths.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary_rows = [
        {"split": "validation", **validation_summary},
        {"split": "test", **test_summary},
    ]
    with (OUTPUT_DIR / "summary_by_split.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        for row in summary_rows:
            row = {
                key: (
                    canonical_json(value)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(row)
    (OUTPUT_DIR / "report_cn.md").write_text(
        build_report(report),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256_file(OUTPUT_DIR / filename)}  {filename}"
            for filename in output_files
        )
        + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
