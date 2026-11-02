"""Deterministic train/interpolate collision census."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import sympy

from toolkit.external_behavior.common import iter_jsonl

from .collision_common import (
    ANALYSIS_ID,
    MODULES,
    PROTOCOL_DIR,
    sha256_file,
    stream_path,
    verify_inputs,
    write_json,
    write_jsonl,
)


def _parse_normalized(module_name: str, question: str) -> sympy.Expr:
    prefixes = {
        "polynomials__expand": "Expand ",
        "polynomials__collect": "Collect the terms in ",
    }
    prefix = prefixes[module_name]
    if not question.startswith(prefix) or not question.endswith("."):
        raise ValueError("Question does not match its frozen module parser.")
    expression = sympy.sympify(question[len(prefix) : -1], evaluate=False)
    symbols = sorted(expression.free_symbols, key=str)
    normalized_symbols = [
        sympy.Symbol(f"x{index}") for index in range(len(symbols))
    ]
    return expression.xreplace(dict(zip(symbols, normalized_symbols)))


def verify_symbolic_zero(
    module_name: str,
    train_question: str,
    interpolate_question: str,
) -> bool:
    train = _parse_normalized(module_name, train_question)
    interpolate = _parse_normalized(module_name, interpolate_question)
    if len(train.free_symbols) != len(interpolate.free_symbols):
        return False
    difference = sympy.simplify(sympy.expand(train - interpolate))
    return difference == sympy.Integer(0)


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    module_name: str,
    regime: str,
) -> None:
    for expected_index, row in enumerate(rows):
        if (
            row["module"] != module_name
            or row["regime"] != regime
            or row["row_index"] != expected_index
        ):
            raise ValueError(
                f"Non-contiguous or mis-scoped row: {module_name}/{regime}/"
                f"{expected_index}"
            )


def analyze_module(
    module_name: str,
    train_rows: Iterable[dict[str, Any]],
    interpolate_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    train = list(train_rows)
    evaluation = list(interpolate_rows)
    _validate_rows(train, module_name=module_name, regime="train")
    _validate_rows(
        evaluation,
        module_name=module_name,
        regime="interpolate",
    )

    train_semantic_counts = Counter(row["semantic_id"] for row in train)
    train_question_counts = Counter(row["question"] for row in train)
    train_semantic_first: dict[str, dict[str, Any]] = {}
    train_question_first: dict[str, dict[str, Any]] = {}
    for row in train:
        train_semantic_first.setdefault(row["semantic_id"], row)
        train_question_first.setdefault(row["question"], row)

    eval_semantic_counts = Counter(row["semantic_id"] for row in evaluation)
    audit_rows: list[dict[str, Any]] = []
    witnesses: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    semantic_overlap_rows = 0
    verbatim_overlap_rows = 0

    for semantic_id in sorted(eval_semantic_counts):
        train_count = train_semantic_counts[semantic_id]
        semantic_rows.append(
            {
                "interpolate_multiplicity": eval_semantic_counts[semantic_id],
                "module": module_name,
                "overlap": train_count > 0,
                "semantic_id": semantic_id,
                "train_multiplicity": train_count,
            }
        )

    for row in evaluation:
        semantic_id = row["semantic_id"]
        semantic_count = train_semantic_counts[semantic_id]
        verbatim_count = train_question_counts[row["question"]]
        semantic_overlap = semantic_count > 0
        verbatim_overlap = verbatim_count > 0
        semantic_train = (
            train_semantic_first[semantic_id] if semantic_overlap else None
        )
        verbatim_train = (
            train_question_first[row["question"]] if verbatim_overlap else None
        )
        if semantic_overlap:
            semantic_overlap_rows += 1
        if verbatim_overlap:
            verbatim_overlap_rows += 1
        audit_rows.append(
            {
                "interpolate_row_id": row["row_id"],
                "interpolate_row_index": row["row_index"],
                "interpolate_semantic_multiplicity": eval_semantic_counts[
                    semantic_id
                ],
                "module": module_name,
                "semantic_id": semantic_id,
                "semantic_overlap": semantic_overlap,
                "semantic_train_witness_row_id": (
                    semantic_train["row_id"] if semantic_train else None
                ),
                "semantic_train_witness_row_index": (
                    semantic_train["row_index"] if semantic_train else None
                ),
                "train_semantic_match_count": semantic_count,
                "train_verbatim_match_count": verbatim_count,
                "unresolved": False,
                "verbatim_overlap": verbatim_overlap,
                "verbatim_train_witness_row_id": (
                    verbatim_train["row_id"] if verbatim_train else None
                ),
                "verbatim_train_witness_row_index": (
                    verbatim_train["row_index"] if verbatim_train else None
                ),
            }
        )
        if not semantic_overlap:
            continue
        exact_zero = verify_symbolic_zero(
            module_name,
            semantic_train["question"],
            row["question"],
        )
        if not exact_zero:
            raise ValueError(
                f"Semantic witness failed symbolic verification: "
                f"{module_name}/{row['row_index']}"
            )
        witnesses.append(
            {
                "interpolate_answer": row["answer"],
                "interpolate_question": row["question"],
                "interpolate_row_id": row["row_id"],
                "interpolate_row_index": row["row_index"],
                "interpolate_semantic_multiplicity": eval_semantic_counts[
                    semantic_id
                ],
                "module": module_name,
                "semantic_id": semantic_id,
                "semantic_overlap": True,
                "semantic_train_witness_answer": semantic_train["answer"],
                "semantic_train_witness_question": semantic_train["question"],
                "semantic_train_witness_row_id": semantic_train["row_id"],
                "semantic_train_witness_row_index": semantic_train["row_index"],
                "train_semantic_match_count": semantic_count,
                "train_verbatim_match_count": verbatim_count,
                "verbatim_overlap": verbatim_overlap,
                "verbatim_train_witness_row_id": (
                    verbatim_train["row_id"] if verbatim_train else None
                ),
                "verbatim_train_witness_row_index": (
                    verbatim_train["row_index"] if verbatim_train else None
                ),
                "verifier_exact_zero": exact_zero,
            }
        )

    unique_overlap = sum(row["overlap"] for row in semantic_rows)
    pair_distribution = Counter(
        (row["train_multiplicity"], row["interpolate_multiplicity"])
        for row in semantic_rows
    )
    row_distribution = Counter()
    for row in semantic_rows:
        row_distribution[row["train_multiplicity"]] += row[
            "interpolate_multiplicity"
        ]
    summary = {
        "canonical_semantic_overlap_row_count": semantic_overlap_rows,
        "canonical_semantic_overlap_row_prevalence": (
            semantic_overlap_rows / len(evaluation)
        ),
        "interpolate_row_denominator": len(evaluation),
        "interpolate_unique_semantic_id_denominator": len(semantic_rows),
        "module": module_name,
        "overlapping_unique_semantic_id_count": unique_overlap,
        "overlapping_unique_semantic_id_prevalence": (
            unique_overlap / len(semantic_rows)
        ),
        "semantic_collision_witness_count": len(witnesses),
        "unresolved_row_count": 0,
        "verbatim_question_overlap_row_count": verbatim_overlap_rows,
        "verbatim_question_overlap_row_prevalence": (
            verbatim_overlap_rows / len(evaluation)
        ),
    }
    return {
        "audit_rows": audit_rows,
        "multiplicity_distribution": [
            {
                "interpolate_multiplicity": interpolate_count,
                "train_multiplicity": train_count,
                "unique_semantic_id_count": count,
            }
            for (train_count, interpolate_count), count in sorted(
                pair_distribution.items()
            )
        ],
        "row_weighted_train_multiplicity_distribution": [
            {
                "interpolate_row_count": count,
                "train_multiplicity": train_count,
            }
            for train_count, count in sorted(row_distribution.items())
        ],
        "semantic_rows": semantic_rows,
        "summary": summary,
        "witnesses": witnesses,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_census(output_dir: Path) -> dict[str, Any]:
    input_validation = verify_inputs()
    output_dir.mkdir(parents=True, exist_ok=False)
    all_audit_rows: list[dict[str, Any]] = []
    all_witnesses: list[dict[str, Any]] = []
    all_semantic_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    multiplicity: dict[str, Any] = {}

    for module_name in MODULES:
        result = analyze_module(
            module_name,
            iter_jsonl(stream_path("train", module_name)),
            iter_jsonl(stream_path("interpolate", module_name)),
        )
        all_audit_rows.extend(result["audit_rows"])
        all_witnesses.extend(result["witnesses"])
        all_semantic_rows.extend(result["semantic_rows"])
        summaries.append(result["summary"])
        multiplicity[module_name] = {
            "semantic_id_distribution": result["multiplicity_distribution"],
            "row_weighted_train_multiplicity_distribution": result[
                "row_weighted_train_multiplicity_distribution"
            ],
        }

    overall = {
        "canonical_semantic_overlap_row_count": sum(
            row["canonical_semantic_overlap_row_count"] for row in summaries
        ),
        "interpolate_row_denominator": sum(
            row["interpolate_row_denominator"] for row in summaries
        ),
        "interpolate_unique_semantic_id_denominator": sum(
            row["interpolate_unique_semantic_id_denominator"]
            for row in summaries
        ),
        "overlapping_unique_semantic_id_count": sum(
            row["overlapping_unique_semantic_id_count"] for row in summaries
        ),
        "unresolved_row_count": sum(
            row["unresolved_row_count"] for row in summaries
        ),
        "verbatim_question_overlap_row_count": sum(
            row["verbatim_question_overlap_row_count"] for row in summaries
        ),
    }
    overall["canonical_semantic_overlap_row_prevalence"] = (
        overall["canonical_semantic_overlap_row_count"]
        / overall["interpolate_row_denominator"]
    )
    overall["overlapping_unique_semantic_id_prevalence"] = (
        overall["overlapping_unique_semantic_id_count"]
        / overall["interpolate_unique_semantic_id_denominator"]
    )
    overall["verbatim_question_overlap_row_prevalence"] = (
        overall["verbatim_question_overlap_row_count"]
        / overall["interpolate_row_denominator"]
    )

    write_jsonl(output_dir / "evaluation_overlap_rows.jsonl", all_audit_rows)
    write_jsonl(
        output_dir / "semantic_collision_witnesses.jsonl",
        all_witnesses,
    )
    _write_csv(output_dir / "semantic_multiplicity.csv", all_semantic_rows)
    _write_csv(output_dir / "summary_by_module.csv", summaries)
    results = {
        "aggregate_release_status": "pending_independent_validation",
        "analysis_id": ANALYSIS_ID,
        "behavioral_claim_allowed": False,
        "by_module": summaries,
        "claim_scope": (
            "Regenerated train/interpolate streams for exactly "
            "polynomials__expand and polynomials__collect."
        ),
        "input_manifest_sha256": input_validation["input_manifest_sha256"],
        "multiplicity_distributions": multiplicity,
        "overall": overall,
        "schema_version": "mathematics-collision-census-results-v1",
    }
    write_json(output_dir / "results.json", results)
    write_json(
        output_dir / "analysis_inputs.json",
        {
            "analysis_preregistration_sha256": sha256_file(
                PROTOCOL_DIR / "preregistration.json"
            ),
            "input_manifest_sha256": input_validation[
                "input_manifest_sha256"
            ],
            "inputs": input_validation["checks"],
        },
    )
    return results
