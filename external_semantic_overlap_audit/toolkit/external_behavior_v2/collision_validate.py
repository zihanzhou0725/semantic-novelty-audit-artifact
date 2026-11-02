"""Independent verifier for Mathematics collision-census outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sympy

from toolkit.external_behavior.common import iter_jsonl

from .collision_common import (
    ANALYSIS_ID,
    MODULES,
    PROTOCOL_DIR,
    read_json,
    stream_path,
    verify_inputs,
    write_json,
)


def _independent_details(
    module_name: str,
    question: str,
) -> tuple[sympy.Expr, str, str]:
    prefixes = {
        "polynomials__expand": "Expand ",
        "polynomials__collect": "Collect the terms in ",
    }
    prefix = prefixes[module_name]
    if not question.startswith(prefix) or not question.endswith("."):
        raise ValueError("Question does not match registered parser.")
    expression = sympy.sympify(question[len(prefix) : -1], evaluate=False)
    source_symbols = sorted(expression.free_symbols, key=str)
    symbols = [sympy.Symbol(f"x{index}") for index in range(len(source_symbols))]
    normalized = expression.subs(
        dict(zip(source_symbols, symbols)),
        simultaneous=True,
    )
    polynomial = sympy.Poly(sympy.expand(normalized), *symbols)
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
    return normalized, canonical, semantic_id


def _exact_zero(
    train_expression: sympy.Expr,
    eval_expression: sympy.Expr,
) -> bool:
    if len(train_expression.free_symbols) != len(eval_expression.free_symbols):
        return False
    return (
        sympy.simplify(sympy.expand(train_expression - eval_expression))
        == sympy.Integer(0)
    )


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def validate_census(
    output_dir: Path,
    *,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"detail": detail, "name": name, "pass": bool(condition)})

    input_validation = verify_inputs()
    check("input_hashes", input_validation["status"] == "pass")
    results = read_json(output_dir / "results.json")
    audit_rows = list(iter_jsonl(output_dir / "evaluation_overlap_rows.jsonl"))
    witness_rows = list(
        iter_jsonl(output_dir / "semantic_collision_witnesses.jsonl")
    )
    witnesses = {
        (row["module"], row["interpolate_row_index"]): row
        for row in witness_rows
    }
    check(
        "witness_keys_unique",
        len(witnesses) == len(witness_rows),
        len(witness_rows),
    )

    expected_audit: list[dict[str, Any]] = []
    expected_semantic_rows: list[dict[str, Any]] = []
    expected_summaries: list[dict[str, Any]] = []
    expected_multiplicity: dict[str, Any] = {}
    symbolic_failures: list[str] = []
    witness_failures: list[str] = []

    for module_name in MODULES:
        train = list(iter_jsonl(stream_path("train", module_name)))
        evaluation = list(iter_jsonl(stream_path("interpolate", module_name)))
        train_semantic_counts = Counter(row["semantic_id"] for row in train)
        train_question_counts = Counter(row["question"] for row in train)
        eval_semantic_counts = Counter(row["semantic_id"] for row in evaluation)
        train_semantic_first: dict[str, dict[str, Any]] = {}
        train_question_first: dict[str, dict[str, Any]] = {}
        for row in train:
            train_semantic_first.setdefault(row["semantic_id"], row)
            train_question_first.setdefault(row["question"], row)

        module_semantic_rows: list[dict[str, Any]] = []
        for semantic_id in sorted(eval_semantic_counts):
            module_semantic_rows.append(
                {
                    "interpolate_multiplicity": eval_semantic_counts[
                        semantic_id
                    ],
                    "module": module_name,
                    "overlap": train_semantic_counts[semantic_id] > 0,
                    "semantic_id": semantic_id,
                    "train_multiplicity": train_semantic_counts[semantic_id],
                }
            )
        expected_semantic_rows.extend(module_semantic_rows)

        semantic_overlap_rows = 0
        verbatim_overlap_rows = 0
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
                train_question_first[row["question"]]
                if verbatim_overlap
                else None
            )
            expected_audit.append(
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
            if semantic_overlap:
                semantic_overlap_rows += 1
                witness = witnesses.get((module_name, row["row_index"]))
                if witness is None:
                    witness_failures.append(
                        f"missing:{module_name}:{row['row_index']}"
                    )
                    continue
                try:
                    train_expression, train_canonical, train_id = (
                        _independent_details(
                            module_name,
                            semantic_train["question"],
                        )
                    )
                    eval_expression, eval_canonical, eval_id = (
                        _independent_details(module_name, row["question"])
                    )
                    exact_zero = _exact_zero(
                        train_expression,
                        eval_expression,
                    )
                    if (
                        train_canonical != semantic_train["canonical_srepr"]
                        or eval_canonical != row["canonical_srepr"]
                        or train_id != semantic_train["semantic_id"]
                        or eval_id != row["semantic_id"]
                        or not exact_zero
                    ):
                        symbolic_failures.append(
                            f"{module_name}:{row['row_index']}"
                        )
                    if (
                        witness["semantic_train_witness_row_index"]
                        != semantic_train["row_index"]
                        or witness["semantic_train_witness_row_id"]
                        != semantic_train["row_id"]
                        or witness["semantic_train_witness_question"]
                        != semantic_train["question"]
                        or witness["semantic_train_witness_answer"]
                        != semantic_train["answer"]
                        or witness["interpolate_row_id"] != row["row_id"]
                        or witness["interpolate_question"] != row["question"]
                        or witness["interpolate_answer"] != row["answer"]
                        or witness["semantic_id"] != semantic_id
                        or witness["train_semantic_match_count"]
                        != semantic_count
                        or witness["train_verbatim_match_count"]
                        != verbatim_count
                        or witness["interpolate_semantic_multiplicity"]
                        != eval_semantic_counts[semantic_id]
                        or witness["verbatim_overlap"] is not verbatim_overlap
                        or witness["verbatim_train_witness_row_index"]
                        != (
                            verbatim_train["row_index"]
                            if verbatim_train
                            else None
                        )
                        or witness["verbatim_train_witness_row_id"]
                        != (
                            verbatim_train["row_id"]
                            if verbatim_train
                            else None
                        )
                        or witness["verifier_exact_zero"] is not True
                    ):
                        witness_failures.append(
                            f"content:{module_name}:{row['row_index']}"
                        )
                except Exception:
                    symbolic_failures.append(
                        f"exception:{module_name}:{row['row_index']}"
                    )
            if verbatim_overlap:
                verbatim_overlap_rows += 1

        unique_overlap = sum(row["overlap"] for row in module_semantic_rows)
        summary = {
            "canonical_semantic_overlap_row_count": semantic_overlap_rows,
            "canonical_semantic_overlap_row_prevalence": (
                semantic_overlap_rows / len(evaluation)
            ),
            "interpolate_row_denominator": len(evaluation),
            "interpolate_unique_semantic_id_denominator": len(
                module_semantic_rows
            ),
            "module": module_name,
            "overlapping_unique_semantic_id_count": unique_overlap,
            "overlapping_unique_semantic_id_prevalence": (
                unique_overlap / len(module_semantic_rows)
            ),
            "semantic_collision_witness_count": semantic_overlap_rows,
            "unresolved_row_count": 0,
            "verbatim_question_overlap_row_count": verbatim_overlap_rows,
            "verbatim_question_overlap_row_prevalence": (
                verbatim_overlap_rows / len(evaluation)
            ),
        }
        expected_summaries.append(summary)
        pair_distribution = Counter(
            (row["train_multiplicity"], row["interpolate_multiplicity"])
            for row in module_semantic_rows
        )
        row_distribution = Counter()
        for semantic_row in module_semantic_rows:
            row_distribution[semantic_row["train_multiplicity"]] += (
                semantic_row["interpolate_multiplicity"]
            )
        expected_multiplicity[module_name] = {
            "row_weighted_train_multiplicity_distribution": [
                {
                    "interpolate_row_count": count,
                    "train_multiplicity": train_count,
                }
                for train_count, count in sorted(row_distribution.items())
            ],
            "semantic_id_distribution": [
                {
                    "interpolate_multiplicity": interpolate_count,
                    "train_multiplicity": train_count,
                    "unique_semantic_id_count": count,
                }
                for (train_count, interpolate_count), count in sorted(
                    pair_distribution.items()
                )
            ],
        }

    check(
        "evaluation_rows",
        audit_rows == expected_audit and len(audit_rows) == 20000,
        len(audit_rows),
    )
    expected_witness_count = sum(
        row["canonical_semantic_overlap_row_count"]
        for row in expected_summaries
    )
    check(
        "witness_count",
        len(witness_rows) == expected_witness_count,
        {
            "expected": expected_witness_count,
            "observed": len(witness_rows),
        },
    )
    check(
        "symbolic_verification",
        not symbolic_failures,
        symbolic_failures[:100],
    )
    check(
        "witness_selection",
        not witness_failures,
        witness_failures[:100],
    )

    semantic_csv = _load_csv(output_dir / "semantic_multiplicity.csv")
    parsed_semantic_csv = [
        {
            "interpolate_multiplicity": int(row["interpolate_multiplicity"]),
            "module": row["module"],
            "overlap": row["overlap"] == "True",
            "semantic_id": row["semantic_id"],
            "train_multiplicity": int(row["train_multiplicity"]),
        }
        for row in semantic_csv
    ]
    check(
        "semantic_multiplicity_table",
        parsed_semantic_csv == expected_semantic_rows,
        len(parsed_semantic_csv),
    )
    summary_csv = _load_csv(output_dir / "summary_by_module.csv")
    parsed_summary_csv = [
        {
            "canonical_semantic_overlap_row_count": int(
                row["canonical_semantic_overlap_row_count"]
            ),
            "canonical_semantic_overlap_row_prevalence": float(
                row["canonical_semantic_overlap_row_prevalence"]
            ),
            "interpolate_row_denominator": int(
                row["interpolate_row_denominator"]
            ),
            "interpolate_unique_semantic_id_denominator": int(
                row["interpolate_unique_semantic_id_denominator"]
            ),
            "module": row["module"],
            "overlapping_unique_semantic_id_count": int(
                row["overlapping_unique_semantic_id_count"]
            ),
            "overlapping_unique_semantic_id_prevalence": float(
                row["overlapping_unique_semantic_id_prevalence"]
            ),
            "semantic_collision_witness_count": int(
                row["semantic_collision_witness_count"]
            ),
            "unresolved_row_count": int(row["unresolved_row_count"]),
            "verbatim_question_overlap_row_count": int(
                row["verbatim_question_overlap_row_count"]
            ),
            "verbatim_question_overlap_row_prevalence": float(
                row["verbatim_question_overlap_row_prevalence"]
            ),
        }
        for row in summary_csv
    ]
    check(
        "summary_csv",
        parsed_summary_csv == expected_summaries,
    )
    check(
        "module_summaries",
        results["by_module"] == expected_summaries,
    )
    check(
        "multiplicity_distributions",
        results["multiplicity_distributions"] == expected_multiplicity,
    )

    overall = {
        "canonical_semantic_overlap_row_count": sum(
            row["canonical_semantic_overlap_row_count"]
            for row in expected_summaries
        ),
        "interpolate_row_denominator": 20000,
        "interpolate_unique_semantic_id_denominator": sum(
            row["interpolate_unique_semantic_id_denominator"]
            for row in expected_summaries
        ),
        "overlapping_unique_semantic_id_count": sum(
            row["overlapping_unique_semantic_id_count"]
            for row in expected_summaries
        ),
        "unresolved_row_count": 0,
        "verbatim_question_overlap_row_count": sum(
            row["verbatim_question_overlap_row_count"]
            for row in expected_summaries
        ),
    }
    overall["canonical_semantic_overlap_row_prevalence"] = (
        overall["canonical_semantic_overlap_row_count"] / 20000
    )
    overall["overlapping_unique_semantic_id_prevalence"] = (
        overall["overlapping_unique_semantic_id_count"]
        / overall["interpolate_unique_semantic_id_denominator"]
    )
    overall["verbatim_question_overlap_row_prevalence"] = (
        overall["verbatim_question_overlap_row_count"] / 20000
    )
    check("overall_aggregate", results["overall"] == overall)
    check(
        "conservation",
        sum(
            item["interpolate_row_count"]
            for module in expected_multiplicity.values()
            for item in module[
                "row_weighted_train_multiplicity_distribution"
            ]
        )
        == 20000
        and sum(
            row["interpolate_unique_semantic_id_denominator"]
            for row in expected_summaries
        )
        == len(expected_semantic_rows),
    )

    failed = [item for item in checks if not item["pass"]]
    report = {
        "aggregate_release_authorized": not failed,
        "analysis_id": ANALYSIS_ID,
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": [item["name"] for item in failed],
        "failed_count": len(failed),
        "independently_recomputed_overall": overall,
        "schema_version": "mathematics-collision-census-validation-v1",
        "status": "pass" if not failed else "no_go",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not failed and write_artifacts:
        results["aggregate_release_status"] = (
            "validated_for_atomic_release"
        )
        write_json(output_dir / "results.json", results)
    if write_artifacts:
        write_json(output_dir / "validation_report.json", report)
    return report
