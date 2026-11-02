#!/usr/bin/env python3
"""Score official SLR model outputs and stratify by semantic-overlap evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


EXPECTED_OUTPUTS_SHA256 = "ecd979d5c1f5ed54e11197830b2afb8e4924075a97babf80b188bea630b5a15b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_rule(text: str, target_predicate: str = "eastbound") -> str:
    text = re.sub(r"%.*?(?=\n|$)", "", text)
    text = re.sub(r"\n\s*", " ", text)
    pattern = re.compile(rf"({re.escape(target_predicate)}\([^()]*\)\s*:-.*?\.)")
    rules = pattern.findall(text)
    return rules[-1].strip() if rules else ""


def fix_validation_program(
    validation_program: str,
    positive_predicate: str = "eastbound",
    negative_predicate: str = "westbound",
) -> str:
    validation_program = re.sub(
        rf"\b{re.escape(positive_predicate)}\b", "pos", validation_program
    )
    validation_program = re.sub(
        rf"\b{re.escape(negative_predicate)}\b", "neg", validation_program
    )
    return (
        validation_program.replace("(train", "(mytrain")
        .replace("(car", "(mycar")
        .replace(", car", ", mycar")
    )


def build_program(output: dict) -> tuple[str, int, bool]:
    reference = output["reference"]
    config = reference.get(
        "evaluation_config",
        {"positive_predicate": "eastbound", "negative_predicate": "westbound"},
    )
    positive = config.get("positive_predicate", "eastbound")
    negative = config.get("negative_predicate", "westbound")
    validation = reference["validation_program"]
    rule = extract_rule(output["model_completion"], positive)
    if not rule or positive not in rule:
        return "", 0, False

    positive_examples = re.findall(rf"{re.escape(positive)}\(([^)]+)\)", validation)
    negative_examples = re.findall(rf"{re.escape(negative)}\(([^)]+)\)", validation)
    total_examples = len(positive_examples) + len(negative_examples)
    if total_examples == 0:
        return "", 0, False

    validation = fix_validation_program(validation, positive, negative)
    validation = "\n".join(sorted(validation.splitlines()))
    judge = f"""
check(X1) :- pos(X1), {positive}(X1).
check(X1) :- neg(X1), \\+ {positive}(X1).
check_count(Count) :-
    (setof((X1), ((pos(X1); neg(X1)), check(X1)), CorrectExamples) ->
        length(CorrectExamples, Count)
    ;
        Count = 0
    ).
"""
    return f"{validation}\n\n{judge}\n\n{rule}\n", total_examples, True


def score_one(
    output: dict,
    *,
    node: Path,
    launcher: Path,
    wasm: Path,
    temp_dir: Path,
    timeout: int,
) -> dict:
    program, total_examples, syntax_extracted = build_program(output)
    base = {
        "problem_id": int(output["problem_id"]),
        "syntax_extracted": syntax_extracted,
        "correct_examples": 0,
        "total_examples": total_examples,
        "partial_score": 0.0,
        "is_correct": False,
        "runtime_error": "",
    }
    if not syntax_extracted:
        base["runtime_error"] = "No target rule extracted"
        return base

    path = temp_dir / f"task_{int(output['problem_id']):04d}.pl"
    path.write_text(program)
    env = os.environ.copy()
    env["NODE_NO_WARNINGS"] = "1"
    command = [
        str(node),
        str(launcher),
        str(wasm),
        path.name,
        "-q",
        "-g",
        "check_count(Count),write(Count),nl,halt",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=temp_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        match = re.search(r"(?m)^\s*(\d+)\s*$", result.stdout)
        if match is None:
            base["runtime_error"] = (result.stderr or result.stdout).strip()[:1000]
            return base
        correct_examples = int(match.group(1))
        partial_score = correct_examples / total_examples
        base.update(
            {
                "correct_examples": correct_examples,
                "partial_score": partial_score,
                "is_correct": correct_examples == total_examples,
                "runtime_error": result.stderr.strip()[:1000],
            }
        )
        return base
    except subprocess.TimeoutExpired:
        base["runtime_error"] = "Prolog timeout"
        return base
    finally:
        path.unlink(missing_ok=True)


def read_ids(path: Path, field: str = "test_id") -> set[int]:
    with path.open(newline="") as handle:
        return {int(row[field]) for row in csv.DictReader(handle)}


def summarize(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    result = {}
    for group, items in sorted(groups.items()):
        n = len(items)
        result[group] = {
            "n": n,
            "accuracy": sum(item["is_correct"] for item in items) / n,
            "partial_score": sum(item["partial_score"] for item in items) / n,
            "syntax_extracted": sum(item["syntax_extracted"] for item in items) / n,
            "runtime_error_count": sum(bool(item["runtime_error"]) for item in items),
            "completion_tokens_mean": sum(item["completion_tokens"] for item in items) / n,
        }
    return result


def summarize_joint(rows: list[dict], outer_key: str, inner_key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[outer_key])].append(row)
    return {
        group: summarize(items, inner_key)
        for group, items in sorted(groups.items())
    }


def compare_overlap_within_level(rows: list[dict]) -> dict:
    duplicate_by_level: dict[str, list[dict]] = defaultdict(list)
    unresolved_by_level: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        level = str(row["curriculum_level"])
        if row["overlap_class"] == "unresolved_or_novel":
            unresolved_by_level[level].append(row)
        else:
            duplicate_by_level[level].append(row)

    common_levels = sorted(
        set(duplicate_by_level) & set(unresolved_by_level), key=int
    )
    per_level = {}
    deltas = []
    weighted_deltas = []
    for level in common_levels:
        duplicate_rows = duplicate_by_level[level]
        unresolved_rows = unresolved_by_level[level]
        duplicate_accuracy = sum(
            row["is_correct"] for row in duplicate_rows
        ) / len(duplicate_rows)
        unresolved_accuracy = sum(
            row["is_correct"] for row in unresolved_rows
        ) / len(unresolved_rows)
        delta = duplicate_accuracy - unresolved_accuracy
        matched_weight = min(len(duplicate_rows), len(unresolved_rows))
        deltas.append(delta)
        weighted_deltas.append((delta, matched_weight))
        per_level[level] = {
            "duplicate_n": len(duplicate_rows),
            "duplicate_accuracy": duplicate_accuracy,
            "unresolved_n": len(unresolved_rows),
            "unresolved_accuracy": unresolved_accuracy,
            "accuracy_delta": delta,
            "min_group_size_weight": matched_weight,
        }

    return {
        "comparison": "union(token_duplicate, flat_semantic_duplicate) - unresolved_or_novel",
        "common_levels": [int(level) for level in common_levels],
        "per_level": per_level,
        "unweighted_mean_accuracy_delta": sum(deltas) / len(deltas),
        "min_group_size_weighted_accuracy_delta": (
            sum(delta * weight for delta, weight in weighted_deltas)
            / sum(weight for _, weight in weighted_deltas)
        ),
        "min_group_size_weight_total": sum(
            weight for _, weight in weighted_deltas
        ),
        "interpretation": (
            "Descriptive only. Curriculum level is controlled by stratification, "
            "but overlap was not randomized and residual rule-complexity confounding "
            "remains."
        ),
    }


def paper_crosscheck(rows: list[dict], by_tier: dict) -> dict:
    reported = {
        "acl_2026_final_table_2_llama3_1_8b_it_slr": {
            "basic": 0.96,
            "easy": 0.54,
            "medium": 0.18,
            "hard": 0.05,
        },
        "arxiv_v2_table_2_llama3_1_8b_it_fft": {
            "basic": 0.92,
            "easy": 0.77,
            "medium": 0.17,
            "hard": 0.02,
        },
    }
    comparisons = {}
    for paper_version, reported_by_tier in reported.items():
        comparisons[paper_version] = {
            "reported_by_tier": reported_by_tier,
            "artifact_by_tier": {
                tier: by_tier[tier]["accuracy"] for tier in reported_by_tier
            },
            "artifact_minus_reported_pp": {
                tier: 100 * (by_tier[tier]["accuracy"] - accuracy)
                for tier, accuracy in reported_by_tier.items()
            },
            "reported_macro_accuracy": (
                sum(reported_by_tier.values()) / len(reported_by_tier)
            ),
        }

    basic = [row for row in rows if row["curriculum_tier"] == "basic"]
    overall_errors = sum(bool(row["runtime_error"]) for row in rows)
    basic_errors = sum(bool(row["runtime_error"]) for row in basic)
    overall_correct = sum(row["is_correct"] for row in rows)
    basic_correct = sum(row["is_correct"] for row in basic)
    return {
        "behavioral_use_status": "exploratory_only_not_reported_main_table_behavior",
        "comparisons": comparisons,
        "runtime_error_upper_bounds": {
            "overall_accuracy_if_every_runtime_error_were_correct": (
                overall_correct + overall_errors
            )
            / len(rows),
            "basic_accuracy_if_every_runtime_error_were_correct": (
                basic_correct + basic_errors
            )
            / len(basic),
        },
        "conclusion": (
            "The repository-hosted artifact aligns exactly with v1-All test IDs and "
            "ground-truth rules, but its tier accuracies do not reproduce either "
            "published Table 2. In particular, even treating every Basic-tier runtime "
            "error as correct gives an upper bound below the final paper's 96%. Do not "
            "use this artifact to estimate the reported model's seen/unseen effect."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-outputs",
        type=Path,
        default=root / "upstream/llama31_8b_slr_model_outputs.json",
    )
    parser.add_argument(
        "--test-parquet",
        type=Path,
        default=root / "upstream/slr_dataset/v1-All/test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--semantic-witnesses",
        type=Path,
        default=root / "artifacts/slr_semantic_overlap_witnesses.csv",
    )
    parser.add_argument(
        "--token-witnesses",
        type=Path,
        default=root / "artifacts/slr_token_overlap_witnesses.csv",
    )
    parser.add_argument(
        "--node",
        type=Path,
        default=Path("<LOCAL_TOOLS>/bin/node/bin/node"),
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        default=root / "scripts/run_trealla_wasi.js",
    )
    parser.add_argument(
        "--wasm",
        type=Path,
        default=root
        / "upstream/prolog_runtime/wasm/tpl-wasm-wasi/tpl.wasm",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "artifacts"
    )
    args = parser.parse_args()

    observed_hash = sha256(args.model_outputs)
    if observed_hash != EXPECTED_OUTPUTS_SHA256:
        raise SystemExit(
            f"Official output checksum mismatch: {observed_hash} != "
            f"{EXPECTED_OUTPUTS_SHA256}"
        )
    for path in (
        args.test_parquet,
        args.semantic_witnesses,
        args.token_witnesses,
        args.node,
        args.launcher,
        args.wasm,
    ):
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")

    outputs = json.loads(args.model_outputs.read_text())
    dataset_rows = pq.read_table(
        args.test_parquet,
        columns=[
            "id",
            "ground-truth rule",
            "curriculum level",
            "curriculum tier",
            "rule sampling",
        ],
    ).to_pylist()
    if len(outputs) != len(dataset_rows):
        raise SystemExit("Official outputs and test split have different row counts")
    for output, dataset_row in zip(outputs, dataset_rows):
        if output["problem_id"] != dataset_row["id"]:
            raise SystemExit("Official outputs do not align with test IDs")
        if output["ground_truth"] != dataset_row["ground-truth rule"]:
            raise SystemExit("Official outputs do not align with ground-truth rules")

    if args.limit is not None:
        outputs = outputs[: args.limit]
        dataset_rows = dataset_rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_ids = read_ids(args.semantic_witnesses)
    token_ids = read_ids(args.token_witnesses)

    with tempfile.TemporaryDirectory(dir=root) as temp_name:
        temp_dir = Path(temp_name)
        score_kwargs = {
            "node": args.node.resolve(),
            "launcher": args.launcher.resolve(),
            "wasm": args.wasm.resolve(),
            "temp_dir": temp_dir,
            "timeout": args.timeout,
        }
        scores: dict[int, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(score_one, output, **score_kwargs): output["problem_id"]
                for output in outputs
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                score = future.result()
                scores[score["problem_id"]] = score
                if completed % 100 == 0:
                    print(f"scored {completed}/{len(futures)}", flush=True)

    rows = []
    for output, dataset_row in zip(outputs, dataset_rows):
        problem_id = int(output["problem_id"])
        if problem_id in token_ids:
            overlap_class = "token_duplicate"
        elif problem_id in semantic_ids:
            overlap_class = "flat_semantic_duplicate"
        else:
            overlap_class = "unresolved_or_novel"
        rows.append(
            {
                **scores[problem_id],
                "overlap_class": overlap_class,
                "curriculum_level": dataset_row["curriculum level"],
                "curriculum_tier": dataset_row["curriculum tier"],
                "rule_sampling": dataset_row["rule sampling"],
                "prompt_tokens": output["prompt_tokens"],
                "completion_tokens": output["completion_tokens"],
                "model_completion": output["model_completion"],
                "ground_truth": output["ground_truth"],
            }
        )

    by_tier = summarize(rows, "curriculum_tier")
    result = {
        "source": {
            "model_repository": "AIML-TUDA/Llama-3.1-8B-SLR",
            "model_outputs_sha256": observed_hash,
            "output_rows": len(rows),
            "test_alignment": "problem_id and ground-truth rule match 100%",
            "prolog_runtime": "Trealla Prolog WASI v3.12.6",
            "prolog_runtime_sha256": sha256(args.wasm),
            "judge_basis": (
                "The extensional check program and rule extraction mirror the official "
                "AIML-TUDA VerifiableRewardsForScalableLogicalReasoning metric; "
                "execution uses Trealla rather than SWI-Prolog."
            ),
        },
        "overall": summarize([{**row, "_all": "all"} for row in rows], "_all")["all"],
        "by_overlap_class": summarize(rows, "overlap_class"),
        "by_tier": by_tier,
        "by_level": summarize(rows, "curriculum_level"),
        "by_overlap_and_tier": summarize_joint(
            rows, "overlap_class", "curriculum_tier"
        ),
        "by_overlap_and_level": summarize_joint(
            rows, "overlap_class", "curriculum_level"
        ),
        "within_level_overlap_comparison": compare_overlap_within_level(rows),
        "paper_crosscheck": paper_crosscheck(rows, by_tier),
        "error_types": dict(
            Counter(row["runtime_error"][:120] for row in rows if row["runtime_error"])
        ),
    }
    suffix = f"_first_{args.limit}" if args.limit is not None else ""
    json_path = args.output_dir / f"slr_official_output_stratification{suffix}.json"
    csv_path = args.output_dir / f"slr_official_output_per_item{suffix}.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Per-item: {csv_path}")


if __name__ == "__main__":
    main()
