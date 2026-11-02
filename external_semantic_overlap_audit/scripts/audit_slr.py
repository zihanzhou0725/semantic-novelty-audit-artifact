#!/usr/bin/env python3
"""Audit SLR-Bench train/test rule overlap without running any model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


DATASET_REVISION = "cecc0aa2602943ead28a4ea74c7a8f3c91264cbf"
SOURCE_REVISION = "3b46979ccdf9bb1c624809cfc140fe7c5af0f778"
RULE_COLUMN = "ground-truth rule"
META_COLUMNS = ["id", RULE_COLUMN, "curriculum level", "curriculum tier", "rule sampling"]
ATOM_RE = re.compile(
    r"^([a-z_][A-Za-z0-9_]*)\s*\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)$"
)
RULE_RE = re.compile(
    r"^\s*eastbound\(\s*([A-Z][A-Za-z0-9_]*)\s*\)\s*:-\s*(.*?)\s*\.\s*$",
    re.DOTALL,
)
TOKEN_RE = re.compile(
    r"'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"|[A-Za-z_][A-Za-z0-9_]*|"
    r"\d+(?:\.\d+)?|\S"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(pq.read_table(path, columns=META_COLUMNS).to_pylist())
    return rows


def split_top_level(text: str) -> list[str] | None:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if depth != 0:
        return None
    parts.append(text[start:].strip())
    return parts


def parse_flat_rule(rule: str) -> list[frozenset[tuple[str, str]]] | None:
    """Parse the positive, flat star-shaped rules emitted by generate_random_rule.

    Each existential car variable is represented by its set of unary properties.
    The shared has_car(Train, CarN) edge is implicit in every profile.
    """

    match = RULE_RE.fullmatch(rule)
    if not match:
        return None
    head_variable, body = match.groups()
    parts = split_top_level(body)
    if not parts:
        return None

    declared: set[str] = set()
    profiles: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for part in parts:
        atom = ATOM_RE.fullmatch(part)
        if atom is None:
            return None
        predicate, first, second = (item.strip() for item in atom.groups())
        if predicate == "has_car" and first == head_variable and second[:1].isupper():
            declared.add(second)
        elif first[:1].isupper() and not second[:1].isupper():
            profiles[first].add((predicate, second))
        else:
            return None

    if not set(profiles).issubset(declared):
        return None
    return [frozenset(profiles[variable]) for variable in declared]


def profile_key(
    profiles: list[frozenset[tuple[str, str]]],
    *,
    collapse_duplicates: bool,
    remove_subsumed: bool,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    normalized = list(profiles)
    if collapse_duplicates:
        normalized = list(set(normalized))
    if remove_subsumed:
        unique = set(normalized)
        # exists x:Q(x) entails exists y:P(y) whenever P is a strict subset of Q.
        normalized = [profile for profile in unique if not any(profile < other for other in unique)]
    return tuple(sorted(tuple(sorted(profile)) for profile in normalized))


def token_key(rule: str) -> tuple[str, ...] | None:
    """Return the Prolog token stream while ignoring only whitespace."""

    tokens: list[str] = []
    cursor = 0
    for match in TOKEN_RE.finditer(rule):
        if rule[cursor : match.start()].strip():
            return None
        tokens.append(match.group())
        cursor = match.end()
    if rule[cursor:].strip():
        return None
    return tuple(tokens)


def build_index(
    rows: list[dict],
    *,
    collapse_duplicates: bool,
    remove_subsumed: bool,
) -> tuple[dict[tuple, list[int]], int]:
    index: dict[tuple, list[int]] = defaultdict(list)
    parsed = 0
    for row_index, row in enumerate(rows):
        profiles = parse_flat_rule(row[RULE_COLUMN])
        if profiles is None:
            continue
        parsed += 1
        index[
            profile_key(
                profiles,
                collapse_duplicates=collapse_duplicates,
                remove_subsumed=remove_subsumed,
            )
        ].append(row_index)
    return index, parsed


def overlap_rows(
    train_rows: list[dict],
    test_rows: list[dict],
    *,
    collapse_duplicates: bool,
    remove_subsumed: bool,
) -> tuple[list[tuple[int, int, tuple]], int, int]:
    train_index, train_parsed = build_index(
        train_rows,
        collapse_duplicates=collapse_duplicates,
        remove_subsumed=remove_subsumed,
    )
    hits: list[tuple[int, int, tuple]] = []
    test_parsed = 0
    for test_index, row in enumerate(test_rows):
        profiles = parse_flat_rule(row[RULE_COLUMN])
        if profiles is None:
            continue
        test_parsed += 1
        key = profile_key(
            profiles,
            collapse_duplicates=collapse_duplicates,
            remove_subsumed=remove_subsumed,
        )
        if key in train_index:
            hits.append((test_index, train_index[key][0], key))
    return hits, train_parsed, test_parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "upstream/slr_dataset/v1-All",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = sorted(args.dataset_dir.glob("train-*.parquet"))
    test_paths = sorted(args.dataset_dir.glob("test-*.parquet"))
    if not train_paths or not test_paths:
        raise SystemExit(f"Missing Parquet split files under {args.dataset_dir}")

    train_rows = read_rows(train_paths)
    test_rows = read_rows(test_paths)
    train_rules = [row[RULE_COLUMN] for row in train_rows]
    test_rules = [row[RULE_COLUMN] for row in test_rows]
    raw_train = set(train_rules)
    token_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for train_index, rule in enumerate(train_rules):
        key = token_key(rule)
        if key is not None:
            token_index[key].append(train_index)
    token_hits = []
    for test_index, rule in enumerate(test_rules):
        key = token_key(rule)
        if key is not None and key in token_index:
            token_hits.append((test_index, token_index[key][0], key))

    variants = {
        "alpha_and_atom_order": (False, False),
        "plus_existential_idempotence": (True, False),
        "plus_existential_subsumption": (True, True),
    }
    variant_results: dict[str, dict] = {}
    final_hits: list[tuple[int, int, tuple]] = []
    for name, (collapse_duplicates, remove_subsumed) in variants.items():
        hits, train_parsed, test_parsed = overlap_rows(
            train_rows,
            test_rows,
            collapse_duplicates=collapse_duplicates,
            remove_subsumed=remove_subsumed,
        )
        variant_results[name] = {
            "train_parseable_rows": train_parsed,
            "test_parseable_rows": test_parsed,
            "test_overlap_rows": len(hits),
            "test_overlap_rate_all": len(hits) / len(test_rows),
            "test_overlap_rate_parseable": len(hits) / test_parsed,
            "overlap_by_level": dict(
                sorted(Counter(test_rows[i]["curriculum level"] for i, _, _ in hits).items())
            ),
            "overlap_by_sampling": dict(
                sorted(Counter(test_rows[i]["rule sampling"] for i, _, _ in hits).items())
            ),
        }
        if name == "plus_existential_subsumption":
            final_hits = hits

    structural_test_indices = {test_index for test_index, _, _ in final_hits}
    token_test_indices = {test_index for test_index, _, _ in token_hits}
    semantic_union = structural_test_indices | token_test_indices
    result = {
        "source": {
            "dataset": "AIML-TUDA/SLR-Bench",
            "configuration": "v1-All",
            "dataset_revision": DATASET_REVISION,
            "source_repository": "ml-research/ScalableLogicalReasoning",
            "source_revision": SOURCE_REVISION,
            "files": {path.name: sha256(path) for path in train_paths + test_paths},
        },
        "rows": {"train": len(train_rows), "test": len(test_rows)},
        "raw_identity": {
            "train_unique_rule_strings": len(set(train_rules)),
            "test_unique_rule_strings": len(set(test_rules)),
            "test_rows_found_verbatim_in_train": sum(rule in raw_train for rule in test_rules),
            "test_rows_with_identical_prolog_tokens_in_train": len(token_hits),
            "token_overlap_by_level": dict(
                sorted(Counter(test_rows[i]["curriculum level"] for i, _, _ in token_hits).items())
            ),
            "token_overlap_by_sampling": dict(
                sorted(Counter(test_rows[i]["rule sampling"] for i, _, _ in token_hits).items())
            ),
        },
        "sound_flat_rule_equivalence_lower_bound": variant_results,
        "sound_semantic_overlap_union": {
            "test_overlap_rows": len(semantic_union),
            "test_overlap_rate": len(semantic_union) / len(test_rows),
            "token_identity_rows": len(token_test_indices),
            "flat_rule_equivalence_rows": len(structural_test_indices),
            "intersection_rows": len(token_test_indices & structural_test_indices),
            "overlap_by_level": dict(
                sorted(Counter(test_rows[i]["curriculum level"] for i in semantic_union).items())
            ),
            "overlap_by_sampling": dict(
                sorted(Counter(test_rows[i]["rule sampling"] for i in semantic_union).items())
            ),
        },
        "scope": (
            "Token identity ignores only whitespace. The structural proof covers flat "
            "positive rules and is sound under ordinary Prolog semantics. Aggregate, "
            "negation, arithmetic, and non-token-identical disjunctive LLM-guided rules "
            "are not classified; therefore the union remains a lower bound."
        ),
    }
    result_path = args.output_dir / "slr_overlap_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    witness_path = args.output_dir / "slr_semantic_overlap_witnesses.csv"
    with witness_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "test_row",
                "test_id",
                "test_level",
                "test_tier",
                "test_sampling",
                "train_row",
                "train_id",
                "semantic_key",
                "test_rule",
                "train_rule",
            ],
        )
        writer.writeheader()
        for test_index, train_index, key in final_hits:
            test_row = test_rows[test_index]
            train_row = train_rows[train_index]
            writer.writerow(
                {
                    "test_row": test_index,
                    "test_id": test_row["id"],
                    "test_level": test_row["curriculum level"],
                    "test_tier": test_row["curriculum tier"],
                    "test_sampling": test_row["rule sampling"],
                    "train_row": train_index,
                    "train_id": train_row["id"],
                    "semantic_key": json.dumps(key),
                    "test_rule": test_row[RULE_COLUMN],
                    "train_rule": train_row[RULE_COLUMN],
                }
            )

    token_witness_path = args.output_dir / "slr_token_overlap_witnesses.csv"
    with token_witness_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "test_row",
                "test_id",
                "test_level",
                "test_tier",
                "test_sampling",
                "train_row",
                "train_id",
                "test_rule",
                "train_rule",
            ],
        )
        writer.writeheader()
        for test_index, train_index, _ in token_hits:
            test_row = test_rows[test_index]
            train_row = train_rows[train_index]
            writer.writerow(
                {
                    "test_row": test_index,
                    "test_id": test_row["id"],
                    "test_level": test_row["curriculum level"],
                    "test_tier": test_row["curriculum tier"],
                    "test_sampling": test_row["rule sampling"],
                    "train_row": train_index,
                    "train_id": train_row["id"],
                    "test_rule": test_row[RULE_COLUMN],
                    "train_rule": train_row[RULE_COLUMN],
                }
            )

    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"\nSound semantic-overlap lower bound: {len(semantic_union)}/"
        f"{len(test_rows)} ({len(semantic_union) / len(test_rows):.2%})"
    )
    print(f"Witnesses: {witness_path}")
    print(f"Token witnesses: {token_witness_path}")


if __name__ == "__main__":
    main()
