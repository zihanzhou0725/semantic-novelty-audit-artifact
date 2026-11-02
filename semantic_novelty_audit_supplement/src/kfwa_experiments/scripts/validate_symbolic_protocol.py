from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import (  # noqa: E402
    build_symbolic_rules,
    get_symbolic_rule_mapping_loaders,
    group_symbolic_rules_by_function,
    split_symbolic_functions,
    symbolic_function_representatives,
    symbolic_function_split_checksum,
)


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", maxsplit=1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(chunk))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def assert_same_episode(
    left: tuple[dict[str, torch.Tensor], torch.Tensor],
    right: tuple[dict[str, torch.Tensor], torch.Tensor],
) -> None:
    left_batch, left_target = left
    right_batch, right_target = right
    if left_batch.keys() != right_batch.keys():
        raise AssertionError("Deterministic datasets returned different batch fields.")
    for key in left_batch:
        if not torch.equal(left_batch[key], right_batch[key]):
            raise AssertionError(f"Episode field {key!r} is not deterministic.")
    if not torch.equal(left_target, right_target):
        raise AssertionError("Episode target is not deterministic.")


def validate_setting(
    *,
    families: str,
    num_symbols: int,
    seeds: list[int],
    holdout_function_frac: float,
    sample_count: int,
    support_k: int,
    support_noise: float,
    noise_tolerance: float,
) -> dict[str, object]:
    rules = build_symbolic_rules(num_symbols=num_symbols, families=families)
    groups = group_symbolic_rules_by_function(rules)
    representatives = symbolic_function_representatives(rules)
    reversed_families = ",".join(reversed(families.split(",")))
    reordered_rules = build_symbolic_rules(num_symbols=num_symbols, families=reversed_families)
    ids_by_outputs = {rule.outputs: rule.function_id for rule in rules}
    reordered_ids_by_outputs = {rule.outputs: rule.function_id for rule in reordered_rules}
    if ids_by_outputs != reordered_ids_by_outputs:
        raise AssertionError("Function ids depend on rule-family enumeration order.")
    duplicate_groups = sum(len(aliases) > 1 for aliases in groups.values())
    split_checksums: set[str] = set()
    reference_holdout_family_counts: dict[str, int] | None = None
    corrupted = 0
    observed = 0

    for seed in seeds:
        seen, unseen = split_symbolic_functions(
            rules,
            holdout_rule_frac=holdout_function_frac,
            seed=seed + 7919,
        )
        seen_outputs = {representatives[function_id].outputs for function_id in seen}
        unseen_outputs = {representatives[function_id].outputs for function_id in unseen}
        if not seen_outputs.isdisjoint(unseen_outputs):
            raise AssertionError(f"Function leakage detected for seed={seed}.")
        holdout_family_counts = dict(Counter(representatives[function_id].family for function_id in unseen))
        if reference_holdout_family_counts is None:
            reference_holdout_family_counts = holdout_family_counts
        elif holdout_family_counts != reference_holdout_family_counts:
            raise AssertionError(
                f"Canonical-family holdout counts changed for seed={seed}: "
                f"{holdout_family_counts} != {reference_holdout_family_counts}."
            )

        checksum = symbolic_function_split_checksum(rules, seen, unseen)
        repeated_checksum = symbolic_function_split_checksum(rules, seen, unseen)
        if checksum != repeated_checksum:
            raise AssertionError(f"Non-deterministic split checksum for seed={seed}.")
        split_checksums.add(checksum)

        bundle_a = get_symbolic_rule_mapping_loaders(
            batch_size=min(sample_count, 128),
            seed=seed,
            train_limit=sample_count,
            val_limit=sample_count,
            test_limit=sample_count,
            num_symbols=num_symbols,
            support_min_k=support_k,
            support_max_k=support_k,
            rule_families=families,
            holdout_rule_frac=holdout_function_frac,
            query_out_of_support_prob=1.0,
            support_noise=support_noise,
        )
        bundle_b = get_symbolic_rule_mapping_loaders(
            batch_size=min(sample_count, 128),
            seed=seed,
            train_limit=sample_count,
            val_limit=sample_count,
            test_limit=sample_count,
            num_symbols=num_symbols,
            support_min_k=support_k,
            support_max_k=support_k,
            rule_families=families,
            holdout_rule_frac=holdout_function_frac,
            query_out_of_support_prob=1.0,
            support_noise=support_noise,
        )
        metadata = bundle_a.metadata
        if metadata["function_split_overlap"] != 0:
            raise AssertionError(f"Non-zero metadata overlap for seed={seed}.")
        if metadata["function_split_checksum"] != checksum:
            raise AssertionError(f"Loader checksum mismatch for seed={seed}.")

        compare_count = min(sample_count, 32)
        for index in range(compare_count):
            assert_same_episode(bundle_a.test.dataset[index], bundle_b.test.dataset[index])

        for index in range(sample_count):
            batch, _ = bundle_a.test.dataset[index]
            if bool(batch["query_in_support"].item()):
                raise AssertionError(f"Out-of-support violation for seed={seed}, index={index}.")
            support_mask = batch["support_mask"].bool()
            support_digits = batch["support_digits"][support_mask].tolist()
            if len(support_digits) != len(set(support_digits)):
                raise AssertionError(f"Duplicate support digit for seed={seed}, index={index}.")
            if bool(batch["seen_function"].item()) != bool(batch["seen_context"].item()):
                raise AssertionError(f"Legacy seen_context alias mismatch for seed={seed}, index={index}.")

            function_id = int(batch["function_id"].item())
            clean_outputs = representatives[function_id].outputs
            support_labels = batch["support_labels"][support_mask].tolist()
            for digit, label in zip(support_digits, support_labels):
                corrupted += int(label != clean_outputs[digit])
                observed += 1

    empirical_noise = corrupted / max(observed, 1)
    if abs(empirical_noise - support_noise) > noise_tolerance:
        raise AssertionError(
            f"Empirical noise {empirical_noise:.4f} differs from configured "
            f"{support_noise:.4f} by more than {noise_tolerance:.4f}."
        )

    return {
        "families": families,
        "identities": len(rules),
        "functions": len(groups),
        "duplicate_groups": duplicate_groups,
        "seeds": len(seeds),
        "unique_split_checksums": len(split_checksums),
        "function_overlap": 0,
        "holdout_family_counts": reference_holdout_family_counts,
        "empirical_noise": empirical_noise,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the function-disjoint symbolic protocol.")
    parser.add_argument(
        "--family-settings",
        default="affine,block,parity;affine,block,parity,piecewise_affine",
        help="Semicolon-separated rule-family settings.",
    )
    parser.add_argument("--num-symbols", type=int, default=20)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--holdout-function-frac", type=float, default=0.25)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--support-k", type=int, default=4)
    parser.add_argument("--support-noise", type=float, default=0.10)
    parser.add_argument("--noise-tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)

    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive.")
    seeds = parse_seeds(args.seeds)
    family_settings = [value.strip() for value in args.family_settings.split(";") if value.strip()]
    if not family_settings:
        raise ValueError("At least one family setting is required.")

    for families in family_settings:
        summary = validate_setting(
            families=families,
            num_symbols=args.num_symbols,
            seeds=seeds,
            holdout_function_frac=args.holdout_function_frac,
            sample_count=args.sample_count,
            support_k=args.support_k,
            support_noise=args.support_noise,
            noise_tolerance=args.noise_tolerance,
        )
        print(
            f"{summary['families']}: identities={summary['identities']} "
            f"functions={summary['functions']} duplicate_groups={summary['duplicate_groups']} "
            f"seeds={summary['seeds']} overlap={summary['function_overlap']} "
            f"holdout_by_family={summary['holdout_family_counts']} noise={summary['empirical_noise']:.4f}"
        )
    print("SYMBOLIC_PROTOCOL_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
