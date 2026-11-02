#!/usr/bin/env python3
"""Evaluate the outcome-free P2 reproducibility and identification gates."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from quality_p1_common import (
    EXTERNAL_ROOT,
    PROTOCOL_DIR,
    ROOT,
    sha256_file,
    write_csv,
    write_json,
)


SCHEMA_VERSION = "quality-enhancement-p2-identification-v1"
P2_PROTOCOL_DIR = PROTOCOL_DIR / "p2"
ANALYSIS_DIR = (
    EXTERNAL_ROOT / "analysis" / "quality-enhancement-v1" / "p2"
)
TEST_PARQUET = (
    EXTERNAL_ROOT
    / "upstream"
    / "slr_dataset"
    / "v1-All"
    / "test-00000-of-00001.parquet"
)
TOKEN_WITNESSES = (
    EXTERNAL_ROOT / "artifacts" / "slr_token_overlap_witnesses.csv"
)
SEMANTIC_WITNESSES = (
    EXTERNAL_ROOT / "artifacts" / "slr_semantic_overlap_witnesses.csv"
)
AUDIT_SCRIPT = EXTERNAL_ROOT / "scripts" / "audit_slr.py"


def fail(message: str) -> None:
    raise SystemExit(f"P2 identification evaluation failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def contains_sequence(tokens: list[str], sequence: tuple[str, ...]) -> bool:
    width = len(sequence)
    return any(
        tuple(tokens[index : index + width]) == sequence
        for index in range(len(tokens) - width + 1)
    )


def load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "quality_p2_frozen_audit_slr",
        AUDIT_SCRIPT,
    )
    require(spec is not None and spec.loader is not None, "cannot load audit_slr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def logical_structure(
    rule: str,
    tokens: list[str],
    audit_module: Any,
) -> tuple[str, int]:
    body_markers = [
        index
        for index in range(len(tokens) - 1)
        if tokens[index : index + 2] == [":", "-"]
    ]
    require(len(body_markers) == 1, "test rule has no unique body marker")
    body = tokens[body_markers[0] + 2 :]
    identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    callable_names = {
        body[index]
        for index in range(len(body) - 1)
        if identifier.fullmatch(body[index]) and body[index + 1] == "("
    }
    atom_count = sum(
        bool(identifier.fullmatch(body[index]))
        and index + 1 < len(body)
        and body[index + 1] == "("
        for index in range(len(body))
    )
    if callable_names & {"findall", "bagof", "setof", "forall"}:
        structure = "aggregate_or_quantified"
    elif "not" in callable_names or contains_sequence(body, ("\\", "+")):
        structure = "negated"
    elif ";" in body:
        structure = "disjunctive"
    elif any(
        contains_sequence(body, operator)
        for operator in (
            ("is",),
            (">",),
            ("<",),
            (">", "="),
            ("=", "<"),
            ("=", ":", "="),
            ("=", "\\", "="),
        )
    ):
        structure = "arithmetic_or_comparison"
    elif audit_module.parse_flat_rule(rule) is not None:
        structure = "flat_positive"
    else:
        structure = "other_complex"
    return structure, atom_count


def read_witnesses(
    path: Path,
    witness_type: str,
    dataset_rows: list[dict[str, Any]],
) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[int, str] = {}
    for row in rows:
        row_index = int(row["test_row"])
        require(0 <= row_index < len(dataset_rows), f"{witness_type} test row")
        dataset_row = dataset_rows[row_index]
        test_id = int(row["test_id"])
        require(test_id == int(dataset_row["id"]), f"{witness_type} test ID")
        require(
            int(row["test_level"]) == int(dataset_row["curriculum level"]),
            f"{witness_type} curriculum level",
        )
        require(
            row["test_tier"] == dataset_row["curriculum tier"],
            f"{witness_type} curriculum tier",
        )
        require(
            row["test_sampling"] == dataset_row["rule sampling"],
            f"{witness_type} sampling source",
        )
        require(
            row["test_rule"] == dataset_row["ground-truth rule"],
            f"{witness_type} rule text",
        )
        require(test_id not in result, f"duplicate {witness_type} test ID")
        result[test_id] = witness_type
    return result


def build_report(report: dict[str, Any]) -> str:
    support = report["common_support"]["exact_curriculum_precheck"]
    source = report["reproducibility_precondition"]
    return "\n".join(
        [
            "# P2 SLR 调整关联识别前置检查",
            "",
            "协议：`quality-enhancement-p2-identification-v1`",
            "",
            "## 决策",
            "",
            "- **Identification No-Go before outcome fit**",
            f"- Witnessed 保留：{support['witnessed_retained']}/"
            f"{support['witnessed_total']} = "
            f"{100.0 * support['witnessed_retained_fraction']:.2f}%",
            f"- No-witness 保留：{support['no_witness_retained']}/"
            f"{support['no_witness_total']} = "
            f"{100.0 * support['no_witness_retained_fraction']:.2f}%",
            "- 注册门槛：每组至少保留 80%，且每组至少 100 行。",
            "",
            "Witnessed 组在 curriculum exact-level precheck 已低于 80%，"
            "因此不得放宽 strata、改变 covariate、拟合 propensity/outcome GLM，"
            "也不得执行 10,000 次 bootstrap。",
            "",
            "## Prediction source",
            "",
            f"- 合格 confirmatory source 数："
            f"{source['eligible_confirmatory_source_count']}",
            "- 当前 repository-hosted output 与 1,000 个测试 ID/rule 对齐，"
            "但未绑定 checkpoint revision/完整 decoding config，且未复现 ACL "
            "2026 Table 2，因此仅可作 exploratory source。",
            "",
            "## 可报告结论",
            "",
            "- SLR-Bench witnessed semantic-overlap 下界仍为 452/1000（45.2%）。",
            "- P2 没有 adjusted association 点估计、置信区间或方向性结论。",
            "- `no registered witness` 不等同于 semantic novelty。",
            "",
        ]
    )


def main() -> int:
    source_manifest_path = P2_PROTOCOL_DIR / "source_manifest.json"
    source_checksums = P2_PROTOCOL_DIR / "SHA256SUMS"
    require(source_manifest_path.is_file(), "P2 sources are not frozen")
    require(source_checksums.is_file(), "P2 source checksum manifest is missing")
    checksum_line = source_checksums.read_text(encoding="ascii").strip()
    expected_hash, filename = checksum_line.split("  ", maxsplit=1)
    require(filename == "source_manifest.json", "unexpected P2 checksum entry")
    require(
        sha256_file(source_manifest_path) == expected_hash,
        "P2 source manifest checksum mismatch",
    )
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="ascii")
    )
    require(
        source_manifest["status"]
        == "frozen_before_new_confirmatory_prediction_overlap_or_correctness_join",
        "P2 source manifest status",
    )
    require(
        source_manifest[
            "new_prediction_overlap_or_correctness_join_performed"
        ]
        is False,
        "source freeze records a new outcome join",
    )

    preregistration_path = PROTOCOL_DIR / "preregistration.json"
    preregistration = json.loads(preregistration_path.read_text(encoding="ascii"))
    require(
        source_manifest["parent_preregistration_sha256"]
        == sha256_file(preregistration_path),
        "P2 sources refer to another preregistration",
    )
    p2 = preregistration["p2"]

    dataset_rows = pq.read_table(
        TEST_PARQUET,
        columns=[
            "id",
            "ground-truth rule",
            "curriculum level",
            "curriculum tier",
            "rule sampling",
        ],
    ).to_pylist()
    require(len(dataset_rows) == 1000, "test split is not 1,000 rows")
    require(
        len({int(row["id"]) for row in dataset_rows}) == 1000,
        "test IDs are not unique",
    )
    audit_module = load_audit_module()

    structure_counts: Counter[str] = Counter()
    token_lengths: list[int] = []
    atom_counts: list[int] = []
    missing_covariates = 0
    for row in dataset_rows:
        if (
            row["id"] is None
            or row["curriculum level"] is None
            or row["rule sampling"] is None
            or row["ground-truth rule"] is None
        ):
            missing_covariates += 1
            continue
        token_value = audit_module.token_key(row["ground-truth rule"])
        require(token_value is not None, "test rule tokenization failed")
        tokens = list(token_value)
        structure, atom_count = logical_structure(
            row["ground-truth rule"],
            tokens,
            audit_module,
        )
        structure_counts[structure] += 1
        token_lengths.append(len(tokens))
        atom_counts.append(atom_count)
    require(missing_covariates == 0, "registered covariates are missing")

    expected_census = p2["analysis"]["covariate_test_census"]
    observed_census = {
        "atom_count_range": [min(atom_counts), max(atom_counts)],
        "logical_structure_counts": dict(sorted(structure_counts.items())),
        "missing_covariate_rows": missing_covariates,
        "rows": len(dataset_rows),
        "token_length_range": [min(token_lengths), max(token_lengths)],
    }
    require(
        observed_census["rows"] == expected_census["rows"],
        "covariate row count changed",
    )
    require(
        observed_census["logical_structure_counts"]
        == expected_census["logical_structure_counts"],
        "logical-structure census changed",
    )
    require(
        observed_census["token_length_range"]
        == expected_census["token_length_range"],
        "token-length range changed",
    )
    require(
        observed_census["atom_count_range"]
        == expected_census["atom_count_range"],
        "atom-count range changed",
    )

    token_ids = read_witnesses(
        TOKEN_WITNESSES,
        "token_identity",
        dataset_rows,
    )
    semantic_ids = read_witnesses(
        SEMANTIC_WITNESSES,
        "flat_semantic_alias",
        dataset_rows,
    )
    require(len(token_ids) == 70, "token witness count changed")
    require(len(semantic_ids) == 382, "semantic witness count changed")
    require(
        not (set(token_ids) & set(semantic_ids)),
        "token and semantic witness sets overlap",
    )
    witnessed_ids = set(token_ids) | set(semantic_ids)
    require(len(witnessed_ids) == 452, "witness union count changed")

    level_counts: Counter[tuple[int, bool]] = Counter(
        (
            int(row["curriculum level"]),
            int(row["id"]) in witnessed_ids,
        )
        for row in dataset_rows
    )
    levels = sorted({level for level, _ in level_counts})
    supported_levels = [
        level
        for level in levels
        if level_counts[(level, True)] > 0
        and level_counts[(level, False)] > 0
    ]
    witnessed_retained = sum(
        level_counts[(level, True)] for level in supported_levels
    )
    no_witness_retained = sum(
        level_counts[(level, False)] for level in supported_levels
    )
    witnessed_total = len(witnessed_ids)
    no_witness_total = len(dataset_rows) - witnessed_total
    witnessed_fraction = witnessed_retained / witnessed_total
    no_witness_fraction = no_witness_retained / no_witness_total
    minimum_fraction = p2["common_support"][
        "minimum_retained_fraction_per_group"
    ]
    minimum_rows = p2["common_support"]["minimum_group_rows_after_trim"]

    level_rows: list[dict[str, Any]] = []
    unsupported_levels: dict[str, str] = {}
    for level in levels:
        witnessed = level_counts[(level, True)]
        no_witness = level_counts[(level, False)]
        has_both = witnessed > 0 and no_witness > 0
        if has_both:
            reason = "both_groups"
        elif witnessed > 0:
            reason = "witness_only"
            unsupported_levels[str(level)] = reason
        else:
            reason = "no_witness_only"
            unsupported_levels[str(level)] = reason
        level_rows.append(
            {
                "curriculum_level": level,
                "witnessed_rows": witnessed,
                "no_witness_rows": no_witness,
                "has_both_groups": has_both,
                "retained_by_exact_level_precheck": has_both,
                "support_class": reason,
            }
        )

    expected_support = p2["common_support"]["exact_curriculum_precheck"]
    require(
        supported_levels == expected_support["levels_with_both_groups"],
        "supported curriculum levels changed",
    )
    require(
        unsupported_levels == expected_support["unsupported_levels"],
        "unsupported curriculum levels changed",
    )
    require(
        witnessed_retained == expected_support["witnessed_retained"] == 352
        and witnessed_total == expected_support["witnessed_total"] == 452,
        "witnessed retention changed",
    )
    require(
        no_witness_retained == expected_support["no_witness_retained"] == 498
        and no_witness_total == expected_support["no_witness_total"] == 548,
        "no-witness retention changed",
    )

    fraction_gate = (
        witnessed_fraction >= minimum_fraction
        and no_witness_fraction >= minimum_fraction
    )
    row_count_gate = (
        witnessed_retained >= minimum_rows
        and no_witness_retained >= minimum_rows
    )
    require(not fraction_gate, "registered exact-level precheck unexpectedly passed")
    require(
        source_manifest["selection_decision"][
            "eligible_confirmatory_source_count"
        ]
        == 0,
        "an eligible confirmatory prediction source unexpectedly exists",
    )

    report = {
        "adjusted_association": {
            "bootstrap_attempts": 0,
            "bootstrap_successful_replicates": 0,
            "confidence_interval_pp": None,
            "estimate_pp": None,
            "outcome_model_fit_run": False,
            "propensity_model_fit_run": False,
            "status": "not_estimated_due_identification_no_go",
        },
        "behavioral_outcomes_loaded": False,
        "claim_boundary": {
            "allowed": [
                "The witnessed semantic-overlap lower bound is 452/1000 (45.2%).",
                "P2 v1 is Identification No-Go before outcome fit.",
            ],
            "forbidden": [
                "An adjusted overlap-correctness association under P2 v1.",
                "No registered witness means semantic novelty.",
                "A causal, inflation, or memorization claim from P2.",
            ],
        },
        "common_support": {
            "exact_curriculum_precheck": {
                "fraction_gate_passed": fraction_gate,
                "minimum_group_rows": minimum_rows,
                "minimum_retained_fraction": minimum_fraction,
                "no_witness_retained": no_witness_retained,
                "no_witness_retained_fraction": no_witness_fraction,
                "no_witness_total": no_witness_total,
                "row_count_gate_passed": row_count_gate,
                "supported_levels": supported_levels,
                "unsupported_levels": unsupported_levels,
                "witnessed_retained": witnessed_retained,
                "witnessed_retained_fraction": witnessed_fraction,
                "witnessed_total": witnessed_total,
            },
            "propensity_trim": {
                "bounds": p2["common_support"][
                    "propensity_bounds_inclusive"
                ],
                "status": "not_run_due_exact_level_precheck_failure",
            },
        },
        "covariates": observed_census,
        "decision": "identification_no_go_before_outcome_fit",
        "failure_reasons": [
            "no_reproducible_confirmatory_prediction_source",
            "witnessed_exact_curriculum_retention_below_80_percent",
        ],
        "input_sha256": {
            relative(AUDIT_SCRIPT): sha256_file(AUDIT_SCRIPT),
            relative(SEMANTIC_WITNESSES): sha256_file(SEMANTIC_WITNESSES),
            relative(TEST_PARQUET): sha256_file(TEST_PARQUET),
            relative(TOKEN_WITNESSES): sha256_file(TOKEN_WITNESSES),
            relative(preregistration_path): sha256_file(preregistration_path),
            relative(source_manifest_path): sha256_file(source_manifest_path),
        },
        "model_predictions_loaded": False,
        "reproducibility_precondition": {
            "eligible_confirmatory_source_count": 0,
            "repository_output_test_alignment": "100_percent",
            "status": "failed",
            "source_manifest_sha256": sha256_file(source_manifest_path),
        },
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "witness_census": {
            "flat_semantic_alias_rows": len(semantic_ids),
            "no_registered_witness_rows": no_witness_total,
            "test_rows": len(dataset_rows),
            "token_identity_rows": len(token_ids),
            "witness_intersection_rows": 0,
            "witnessed_overlap_rows": witnessed_total,
        },
    }

    output_names = (
        "common_support_by_level.csv",
        "identification_report.json",
        "report_cn.md",
    )
    for name in (*output_names, "SHA256SUMS"):
        require(
            not (ANALYSIS_DIR / name).exists(),
            f"P2 output already exists: {name}",
        )
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ANALYSIS_DIR / "common_support_by_level.csv", level_rows)
    write_json(ANALYSIS_DIR / "identification_report.json", report)
    (ANALYSIS_DIR / "report_cn.md").write_text(
        build_report(report),
        encoding="utf-8",
    )
    checksum_lines = [
        f"{sha256_file(ANALYSIS_DIR / name)}  {name}" for name in output_names
    ]
    (ANALYSIS_DIR / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="ascii",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
