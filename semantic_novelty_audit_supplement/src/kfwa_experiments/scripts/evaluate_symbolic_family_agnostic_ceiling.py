from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterator, TextIO

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_symbolic_a7_feature_ablations import student_t_quantile
from evaluate_symbolic_model_ceiling_gap import (
    categorical_entropy,
    file_sha256,
    parse_ints,
    write_csv,
    write_json,
)
from kfwa.datasets import (
    SymbolicRuleMappingDataset,
    build_symbolic_family_transfer_evaluation,
    build_symbolic_rules,
    symbolic_function_representatives,
    symbolic_function_set_checksum,
)
from kfwa.symbolic import SymbolicBayesianPredictor
from kfwa.training import move_batch
from kfwa.utils import get_device


FAMILIES = (
    "affine",
    "block",
    "parity",
    "quadratic",
    "piecewise_affine",
    "composition",
)
KNOWN_FAMILIES = ("affine", "block", "parity")
HELDOUT_FAMILIES = ("quadratic", "piecewise_affine", "composition")
ORACLES = (
    "family_conditioned",
    "family_balanced",
    "uniform_function",
)
ORACLE_METRICS = (
    "correct",
    "posterior_expected_accuracy",
    "ce",
    "brier",
    "confidence",
    "predictive_entropy",
    "function_posterior_entropy",
    "function_effective_count",
)


@dataclass
class OracleUniverse:
    predictor: SymbolicBayesianPredictor
    family_ids: dict[str, list[int]]
    family_masks: torch.Tensor
    component_prior: torch.Tensor
    family_balanced_prior: torch.Tensor
    uniform_function_prior: torch.Tensor
    metadata: dict[str, Any]


@dataclass
class ReferenceEpisodes:
    handle: TextIO
    rows: Iterator[dict[str, str]]
    path: Path
    row_count: int = 0

    def check(self, row: dict[str, Any]) -> None:
        try:
            reference = next(self.rows)
        except StopIteration as error:
            raise AssertionError("Registered reference ended early.") from error
        expected = {
            "episode_id": str(row["episode_id"]),
            "seed": str(row["seed"]),
            "family": str(row["family"]),
            "episode_index": str(row["episode_index"]),
            "function_id": str(row["function_id"]),
            "query_digit": str(row["query_digit"]),
            "target": str(row["target"]),
        }
        observed = {key: reference[key] for key in expected}
        if observed != expected:
            raise AssertionError(
                f"Registered episode mismatch at row {self.row_count}: "
                f"expected={expected}, observed={observed}"
            )
        self.row_count += 1

    def finish(self) -> None:
        try:
            extra = next(self.rows)
        except StopIteration:
            self.handle.close()
            return
        self.handle.close()
        raise AssertionError(f"Registered reference has an extra row: {extra}")


def family_exclusions(family: str) -> tuple[str, ...] | None:
    if family == "composition":
        return FAMILIES[:-1]
    if family in {"quadratic", "piecewise_affine"}:
        return KNOWN_FAMILIES
    return None


def prior_checksum(function_ids: torch.Tensor, prior: torch.Tensor) -> str:
    payload = [
        [int(function_id), float(weight)]
        for function_id, weight in zip(
            function_ids.detach().cpu().tolist(),
            prior.detach().cpu().double().tolist(),
            strict=True,
        )
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_oracle_universe(
    *,
    num_symbols: int,
    support_noise: float,
    device: torch.device,
) -> OracleUniverse:
    family_ids: dict[str, list[int]] = {}
    family_metadata: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        _, eligible_ids, metadata = build_symbolic_family_transfer_evaluation(
            num_symbols=num_symbols,
            family=family,
            excluded_families=family_exclusions(family),
        )
        family_ids[family] = eligible_ids
        family_metadata[family] = metadata

    predictor = SymbolicBayesianPredictor(
        num_symbols=num_symbols,
        rule_families=",".join(FAMILIES),
        target_dim=num_symbols,
        support_noise=support_noise,
    ).to(device)
    predictor.eval()
    function_ids = predictor.function_ids
    function_id_list = [int(value) for value in function_ids.detach().cpu()]
    function_index = {
        function_id: index for index, function_id in enumerate(function_id_list)
    }
    eligible_union = set().union(*(set(ids) for ids in family_ids.values()))
    if set(function_id_list) != eligible_union:
        missing = sorted(eligible_union - set(function_id_list))
        extra = sorted(set(function_id_list) - eligible_union)
        raise AssertionError(
            f"Union oracle mismatch: missing={missing}, extra={extra}."
        )

    family_masks = torch.zeros(
        len(FAMILIES),
        len(function_id_list),
        dtype=torch.bool,
        device=device,
    )
    component_prior = torch.zeros(
        len(FAMILIES),
        len(function_id_list),
        dtype=torch.float64,
        device=device,
    )
    family_probability = 1.0 / float(len(FAMILIES))
    for family_index, family in enumerate(FAMILIES):
        ids = family_ids[family]
        indices = torch.tensor(
            [function_index[function_id] for function_id in ids],
            dtype=torch.long,
            device=device,
        )
        family_masks[family_index, indices] = True
        component_prior[family_index, indices] = (
            family_probability / float(len(ids))
        )

    family_balanced_prior = component_prior.sum(dim=0)
    uniform_function_prior = torch.full(
        (len(function_id_list),),
        1.0 / float(len(function_id_list)),
        dtype=torch.float64,
        device=device,
    )
    if not torch.isclose(
        component_prior.sum(),
        torch.tensor(1.0, dtype=torch.float64, device=device),
        atol=1.0e-12,
    ):
        raise AssertionError("Family-component prior does not sum to one.")
    if not torch.isclose(
        family_balanced_prior.sum(),
        torch.tensor(1.0, dtype=torch.float64, device=device),
        atol=1.0e-12,
    ):
        raise AssertionError("Family-balanced prior does not sum to one.")
    if torch.any(family_balanced_prior <= 0.0):
        raise AssertionError("Every union function must have positive prior mass.")

    multiplicities = family_masks.sum(dim=0)
    multiplicity_histogram = {
        str(value): int(multiplicities.eq(value).sum().cpu())
        for value in sorted(set(int(item) for item in multiplicities.cpu()))
    }
    pairwise_overlap: dict[str, int] = {}
    for left_index, left in enumerate(FAMILIES):
        for right_index in range(left_index + 1, len(FAMILIES)):
            right = FAMILIES[right_index]
            overlap = len(set(family_ids[left]) & set(family_ids[right]))
            pairwise_overlap[f"{left}|{right}"] = overlap

    all_rules = build_symbolic_rules(
        num_symbols=num_symbols,
        families=",".join(FAMILIES),
    )
    metadata = {
        "families": list(FAMILIES),
        "family_metadata": family_metadata,
        "component_entry_count": sum(len(ids) for ids in family_ids.values()),
        "unique_function_count": len(function_id_list),
        "duplicate_component_count": (
            sum(len(ids) for ids in family_ids.values()) - len(function_id_list)
        ),
        "multiplicity_histogram": multiplicity_histogram,
        "pairwise_overlap": pairwise_overlap,
        "candidate_function_checksum": symbolic_function_set_checksum(
            all_rules,
            function_id_list,
        ),
        "family_balanced_prior_checksum": prior_checksum(
            function_ids,
            family_balanced_prior,
        ),
        "uniform_function_prior_checksum": prior_checksum(
            function_ids,
            uniform_function_prior,
        ),
    }
    return OracleUniverse(
        predictor=predictor,
        family_ids=family_ids,
        family_masks=family_masks,
        component_prior=component_prior,
        family_balanced_prior=family_balanced_prior,
        uniform_function_prior=uniform_function_prior,
        metadata=metadata,
    )


def posterior_from_prior(
    log_likelihood: torch.Tensor,
    prior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prior.ndim != 1 or prior.shape[0] != log_likelihood.shape[1]:
        raise ValueError("Prior shape must match the candidate-function axis.")
    if torch.any(prior < 0.0) or not torch.isfinite(prior).all():
        raise ValueError("Prior probabilities must be finite and nonnegative.")
    log_prior = prior.log().to(dtype=log_likelihood.dtype)
    scores = log_likelihood + log_prior.unsqueeze(0)
    log_evidence = torch.logsumexp(scores, dim=1)
    if not torch.isfinite(log_evidence).all():
        raise AssertionError("Oracle evidence must be finite.")
    posterior = torch.softmax(scores, dim=1)
    return posterior, log_evidence


def family_conditioned_posterior(
    log_likelihood: torch.Tensor,
    family_mask: torch.Tensor,
) -> torch.Tensor:
    scores = log_likelihood.masked_fill(
        ~family_mask.unsqueeze(0),
        float("-inf"),
    )
    if not torch.isfinite(scores).any(dim=1).all():
        raise AssertionError("Family-conditioned candidate set is empty.")
    return torch.softmax(scores, dim=1)


def family_component_posterior(
    log_likelihood: torch.Tensor,
    component_prior: torch.Tensor,
    balanced_log_evidence: torch.Tensor,
) -> torch.Tensor:
    log_component_prior = component_prior.log().to(dtype=log_likelihood.dtype)
    log_component_evidence = torch.logsumexp(
        log_likelihood.unsqueeze(1) + log_component_prior.unsqueeze(0),
        dim=2,
    )
    component_log_normalizer = torch.logsumexp(
        log_component_evidence,
        dim=1,
    )
    if not torch.allclose(
        component_log_normalizer,
        balanced_log_evidence,
        atol=2.0e-5,
        rtol=2.0e-5,
    ):
        raise AssertionError(
            "Family-component and deduplicated-function evidence disagree."
        )
    return torch.softmax(log_component_evidence, dim=1)


def oracle_diagnostics(
    *,
    predictor: SymbolicBayesianPredictor,
    batch: dict[str, torch.Tensor],
    posterior: torch.Tensor,
    target: torch.Tensor,
    one_hot_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    probs = predictor.predictive_probs(batch, posterior)
    confidence, prediction = probs.max(dim=1)
    target_probability = probs.gather(1, target.unsqueeze(1)).squeeze(1)
    function_entropy = categorical_entropy(posterior)
    return {
        "prediction": prediction,
        "correct": prediction.eq(target),
        "confidence": confidence,
        "target_probability": target_probability,
        "ce": -target_probability.clamp_min(1.0e-12).log(),
        "brier": (probs - one_hot_target).square().sum(dim=1),
        "posterior_expected_accuracy": confidence,
        "predictive_entropy": categorical_entropy(probs),
        "function_posterior_entropy": function_entropy,
        "function_effective_count": function_entropy.exp(),
    }


def open_reference(path: Path | None) -> ReferenceEpisodes | None:
    if path is None:
        return None
    handle = path.open("r", newline="", encoding="utf-8")
    return ReferenceEpisodes(
        handle=handle,
        rows=iter(csv.DictReader(handle)),
        path=path,
    )


def evaluate_family(
    *,
    universe: OracleUniverse,
    family: str,
    family_index: int,
    seed: int,
    num_symbols: int,
    support_k: int,
    support_noise: float,
    query_out_of_support_prob: float,
    test_limit: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    reference: ReferenceEpisodes | None,
) -> list[dict[str, Any]]:
    rules, eligible_ids, _ = build_symbolic_family_transfer_evaluation(
        num_symbols=num_symbols,
        family=family,
        excluded_families=family_exclusions(family),
    )
    dataset_seed = seed + 40_000 + family_index * 10_000
    dataset = SymbolicRuleMappingDataset(
        size=test_limit,
        rules=rules,
        active_function_ids=eligible_ids,
        seen_function_ids=set(),
        min_support_k=support_k,
        max_support_k=support_k,
        query_out_of_support_prob=query_out_of_support_prob,
        support_noise=support_noise,
        seed=dataset_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    predictor = universe.predictor
    family_mask = universe.family_masks[family_index]
    rows: list[dict[str, Any]] = []
    episode_index = 0
    with torch.no_grad():
        for batch, target in loader:
            batch = move_batch(batch, device)
            target = target.to(device)
            function_ids = batch["function_id"].long()
            function_indices = torch.searchsorted(
                predictor.function_ids,
                function_ids,
            )
            if not torch.equal(
                predictor.function_ids[function_indices],
                function_ids,
            ):
                raise AssertionError(f"{family} function is absent from union.")
            if not family_mask[function_indices].all():
                raise AssertionError(
                    f"{family} sampled outside its eligible candidate set."
                )
            true_tables = predictor.rule_table[function_indices]
            clean_support = true_tables.gather(
                1,
                batch["support_digits"].long(),
            )
            corruption_count = (
                clean_support.ne(batch["support_labels"].long())
                & batch["support_mask"].bool()
            ).sum(dim=1)
            clean_target = true_tables.gather(
                1,
                batch["query_digit"].long().unsqueeze(1),
            ).squeeze(1)
            if not torch.equal(clean_target, target):
                raise AssertionError(f"{family} target disagrees with union table.")

            log_likelihood = predictor.support_log_likelihood(batch)
            conditioned_posterior = family_conditioned_posterior(
                log_likelihood,
                family_mask,
            )
            balanced_posterior, balanced_log_evidence = posterior_from_prior(
                log_likelihood,
                universe.family_balanced_prior,
            )
            uniform_posterior, _ = posterior_from_prior(
                log_likelihood,
                universe.uniform_function_prior,
            )
            family_posterior = family_component_posterior(
                log_likelihood,
                universe.component_prior,
                balanced_log_evidence,
            )
            one_hot_target = torch.nn.functional.one_hot(
                target.long(),
                num_classes=num_symbols,
            ).float()
            diagnostics = {
                "family_conditioned": oracle_diagnostics(
                    predictor=predictor,
                    batch=batch,
                    posterior=conditioned_posterior,
                    target=target,
                    one_hot_target=one_hot_target,
                ),
                "family_balanced": oracle_diagnostics(
                    predictor=predictor,
                    batch=batch,
                    posterior=balanced_posterior,
                    target=target,
                    one_hot_target=one_hot_target,
                ),
                "uniform_function": oracle_diagnostics(
                    predictor=predictor,
                    batch=batch,
                    posterior=uniform_posterior,
                    target=target,
                    one_hot_target=one_hot_target,
                ),
            }
            family_confidence, family_prediction = family_posterior.max(dim=1)
            family_entropy = categorical_entropy(family_posterior)
            true_family_probability = family_posterior[:, family_index]
            true_function_probability = balanced_posterior.gather(
                1,
                function_indices.unsqueeze(1),
            ).squeeze(1)

            for index in range(target.shape[0]):
                row: dict[str, Any] = {
                    "episode_id": (
                        f"family={family}|seed={seed}|index={episode_index}"
                    ),
                    "seed": seed,
                    "family": family,
                    "family_status": (
                        "known" if family in KNOWN_FAMILIES else "heldout"
                    ),
                    "episode_index": episode_index,
                    "dataset_seed": dataset_seed,
                    "support_k": support_k,
                    "support_noise": support_noise,
                    "function_id": int(function_ids[index].cpu()),
                    "query_digit": int(batch["query_digit"][index].cpu()),
                    "target": int(target[index].cpu()),
                    "corruption_count": int(corruption_count[index].cpu()),
                    "true_function_posterior": float(
                        true_function_probability[index].cpu()
                    ),
                    "true_family_posterior": float(
                        true_family_probability[index].cpu()
                    ),
                    "family_prediction": int(family_prediction[index].cpu()),
                    "family_prediction_name": FAMILIES[
                        int(family_prediction[index].cpu())
                    ],
                    "family_correct": int(
                        int(family_prediction[index].cpu()) == family_index
                    ),
                    "family_confidence": float(
                        family_confidence[index].cpu()
                    ),
                    "family_posterior_entropy": float(
                        family_entropy[index].cpu()
                    ),
                }
                for prefix, values in diagnostics.items():
                    for metric, tensor in values.items():
                        value = tensor[index].cpu()
                        if metric == "prediction":
                            row[f"{prefix}_{metric}"] = int(value)
                        elif metric == "correct":
                            row[f"{prefix}_{metric}"] = int(bool(value))
                        else:
                            row[f"{prefix}_{metric}"] = float(value)
                if reference is not None:
                    reference.check(row)
                rows.append(row)
                episode_index += 1
    if len(rows) != test_limit:
        raise AssertionError(f"{family} generated {len(rows)} != {test_limit}.")
    return rows


def subset_average(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def seed_scopes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    category_families = {
        "family_known": set(KNOWN_FAMILIES),
        "development_ood": {"quadratic", "piecewise_affine"},
        "untouched_composition": {"composition"},
        "all_heldout": set(HELDOUT_FAMILIES),
        "all_families": set(FAMILIES),
    }
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        scopes: list[tuple[str, str, list[dict[str, Any]]]] = [
            (
                "family",
                family,
                [row for row in seed_rows if row["family"] == family],
            )
            for family in FAMILIES
        ]
        scopes.extend(
            (
                "category",
                scope,
                [
                    row
                    for row in seed_rows
                    if row["family"] in selected_families
                ],
            )
            for scope, selected_families in category_families.items()
        )
        for scope_type, scope, group in scopes:
            summary: dict[str, Any] = {
                "scope_type": scope_type,
                "scope": scope,
                "seed": seed,
                "episode_count": len(group),
            }
            for prefix in ORACLES:
                for metric in ORACLE_METRICS:
                    summary[f"{prefix}_{metric}"] = subset_average(
                        group,
                        f"{prefix}_{metric}",
                    )
            for metric in (
                "true_function_posterior",
                "true_family_posterior",
                "family_correct",
                "family_confidence",
                "family_posterior_entropy",
            ):
                summary[metric] = subset_average(group, metric)
            summary["conditioned_minus_balanced_expected_accuracy"] = (
                float(
                    summary[
                        "family_conditioned_posterior_expected_accuracy"
                    ]
                )
                - float(
                    summary[
                        "family_balanced_posterior_expected_accuracy"
                    ]
                )
            )
            summary["balanced_minus_uniform_expected_accuracy"] = (
                float(
                    summary[
                        "family_balanced_posterior_expected_accuracy"
                    ]
                )
                - float(
                    summary[
                        "uniform_function_posterior_expected_accuracy"
                    ]
                )
            )
            output.append(summary)
    return output


def aggregate_seed_scopes(
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    identifiers = {"scope_type", "scope", "seed", "episode_count"}
    output: list[dict[str, Any]] = []
    scopes = sorted(
        {(str(row["scope_type"]), str(row["scope"])) for row in seed_rows}
    )
    for scope_type, scope in scopes:
        group = [
            row
            for row in seed_rows
            if row["scope_type"] == scope_type and row["scope"] == scope
        ]
        summary: dict[str, Any] = {
            "scope_type": scope_type,
            "scope": scope,
            "n": len(group),
            "episodes_per_seed": int(group[0]["episode_count"]),
        }
        for metric in [key for key in group[0] if key not in identifiers]:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_sd"] = (
                stdev(values) if len(values) > 1 else 0.0
            )
        output.append(summary)
    return output


def comparison_rows(
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = (
        "conditioned_minus_balanced_expected_accuracy",
        "balanced_minus_uniform_expected_accuracy",
    )
    scopes = sorted(
        {(str(row["scope_type"]), str(row["scope"])) for row in seed_rows}
    )
    for scope_type, scope in scopes:
        group = sorted(
            [
                row
                for row in seed_rows
                if row["scope_type"] == scope_type
                and row["scope"] == scope
            ],
            key=lambda row: int(row["seed"]),
        )
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            average = mean(values)
            sample_sd = stdev(values) if len(values) > 1 else 0.0
            standard_error = sample_sd / math.sqrt(len(values))
            critical = (
                student_t_quantile(0.975, len(values) - 1)
                if len(values) > 1
                else 0.0
            )
            output.append(
                {
                    "scope_type": scope_type,
                    "scope": scope,
                    "metric": metric,
                    "n": len(values),
                    "mean": average,
                    "sd": sample_sd,
                    "ci95_low": average - critical * standard_error,
                    "ci95_high": average + critical * standard_error,
                    "seed_values": "|".join(
                        f"{value:.12g}" for value in values
                    ),
                }
            )
    return output


def find_group(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    matches = [row for row in rows if row["scope"] == scope]
    if len(matches) != 1:
        raise AssertionError(f"Expected one aggregate row for {scope}.")
    return matches[0]


def decision_gate(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    heldout = find_group(group_rows, "all_heldout")
    heldout_family_rows = [
        find_group(group_rows, family) for family in HELDOUT_FAMILIES
    ]
    expected_key = (
        "family_balanced_posterior_expected_accuracy_mean"
    )
    conditioned_key = (
        "family_conditioned_posterior_expected_accuracy_mean"
    )
    uniform_key = "uniform_function_posterior_expected_accuracy_mean"
    heldout_expected = float(heldout[expected_key])
    minimum_family_expected = min(
        float(row[expected_key]) for row in heldout_family_rows
    )
    conditioned_gap = (
        float(heldout[conditioned_key]) - heldout_expected
    )
    prior_sensitivity = abs(
        heldout_expected - float(heldout[uniform_key])
    )
    checks = {
        "heldout_expected_accuracy_at_least_80pct": (
            heldout_expected >= 0.80
        ),
        "each_heldout_family_at_least_70pct": (
            minimum_family_expected >= 0.70
        ),
        "conditioned_gap_at_most_20pp": conditioned_gap <= 0.20,
        "prior_sensitivity_at_most_5pp": prior_sensitivity <= 0.05,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "values": {
            "all_heldout_family_balanced_expected_accuracy": (
                heldout_expected
            ),
            "minimum_heldout_family_expected_accuracy": (
                minimum_family_expected
            ),
            "all_heldout_family_conditioned_expected_accuracy": float(
                heldout[conditioned_key]
            ),
            "conditioned_minus_balanced_gap": conditioned_gap,
            "all_heldout_uniform_function_expected_accuracy": float(
                heldout[uniform_key]
            ),
            "balanced_minus_uniform_absolute_gap": prior_sensitivity,
        },
        "decision": (
            "proceed_to_calibrated_a11_pilot"
            if all(checks.values())
            else "run_support_size_identifiability_before_a11"
        ),
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_analysis(
    path: Path,
    *,
    group_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Phase 2.9 Family-Agnostic Bayesian Ceiling",
        "",
        "## Candidate Universe",
        "",
        (
            f"The six family components contain "
            f"{metadata['component_entry_count']:,} entries and "
            f"{metadata['unique_function_count']:,} unique complete mappings."
        ),
        (
            f"Cross-family aliases contribute "
            f"{metadata['duplicate_component_count']} duplicate component entries."
        ),
        "",
        "## Family Results",
        "",
        (
            "| Family | Conditioned expected | Family-agnostic expected | "
            "Family-agnostic accuracy | Uniform-prior expected | "
            "Family ID accuracy |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        row = find_group(group_rows, family)
        lines.append(
            "| "
            + " | ".join(
                [
                    family,
                    pct(
                        float(
                            row[
                                "family_conditioned_posterior_expected_accuracy_mean"
                            ]
                        )
                    ),
                    pct(
                        float(
                            row[
                                "family_balanced_posterior_expected_accuracy_mean"
                            ]
                        )
                    ),
                    pct(float(row["family_balanced_correct_mean"])),
                    pct(
                        float(
                            row[
                                "uniform_function_posterior_expected_accuracy_mean"
                            ]
                        )
                    ),
                    pct(float(row["family_correct_mean"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            (
                "| Scope | Conditioned expected | Family-agnostic expected | "
                "Gap | Uniform-prior expected |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scope in (
        "family_known",
        "development_ood",
        "untouched_composition",
        "all_heldout",
        "all_families",
    ):
        row = find_group(group_rows, scope)
        lines.append(
            "| "
            + " | ".join(
                [
                    scope,
                    pct(
                        float(
                            row[
                                "family_conditioned_posterior_expected_accuracy_mean"
                            ]
                        )
                    ),
                    pct(
                        float(
                            row[
                                "family_balanced_posterior_expected_accuracy_mean"
                            ]
                        )
                    ),
                    pct(
                        float(
                            row[
                                "conditioned_minus_balanced_expected_accuracy_mean"
                            ]
                        )
                    ),
                    pct(
                        float(
                            row[
                                "uniform_function_posterior_expected_accuracy_mean"
                            ]
                        )
                    ),
                ]
            )
            + " |"
        )
    heldout = find_group(group_rows, "all_heldout")
    parity = find_group(group_rows, "parity")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The family-agnostic held-out expected ceiling is "
                f"{pct(float(heldout['family_balanced_posterior_expected_accuracy_mean']))}, "
                "only "
                f"{pct(float(heldout['conditioned_minus_balanced_expected_accuracy_mean']))} "
                "below the family-conditioned reference. Its exact-posterior "
                f"cross-entropy is {float(heldout['family_balanced_ce_mean']):.4f}."
            ),
            (
                "Changing to a uniform prior over unique functions changes the "
                "held-out expected ceiling by only "
                f"{pct(abs(float(heldout['balanced_minus_uniform_expected_accuracy_mean'])))}."
            ),
            (
                "Parity family-ID accuracy is "
                f"{pct(float(parity['family_correct_mean']))}, while its query-label "
                "expected ceiling remains "
                f"{pct(float(parity['family_balanced_posterior_expected_accuracy_mean']))}. "
                "This is expected because 40 parity mappings are complete-map aliases "
                "of affine mappings: the latent family can be ambiguous while the "
                "query output is identical."
            ),
            (
                "Therefore the K=8 OOD failure is not explained by an "
                "information-theoretic ceiling under the registered union hypothesis "
                "class. The remaining gap is an inference and calibration problem."
            ),
        ]
    )
    lines.extend(["", "## Decision Gate", ""])
    for name, passed in gate["checks"].items():
        lines.append(f"- `{name}`: **{passed}**")
    lines.extend(
        [
            "",
            f"Phase 2.9 gate: **{'Go' if gate['passed'] else 'No-Go'}**.",
            "",
            (
                "Decision: the K=8 task is information-sufficient."
                if gate["passed"]
                else (
                    "Decision: run a support-size identifiability sweep before "
                    "drawing model-level conclusions."
                )
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the family-agnostic exact Bayesian ceiling."
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "kfwa_experiments/"
            "runs_phase2_9_family_agnostic_bayes_k8_5seeds"
        ),
    )
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--num-symbols", type=int, default=20)
    parser.add_argument("--support-k", type=int, default=8)
    parser.add_argument("--support-noise", type=float, default=0.1)
    parser.add_argument(
        "--query-out-of-support-prob",
        type=float,
        default=1.0,
    )
    parser.add_argument("--test-limit-per-family", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reference-episodes",
        default="",
        help="Optional registered episodes.csv to match exactly.",
    )
    args = parser.parse_args()

    seeds = parse_ints(args.seeds)
    device = get_device(args.device)
    universe = build_oracle_universe(
        num_symbols=args.num_symbols,
        support_noise=args.support_noise,
        device=device,
    )
    reference_path = (
        Path(args.reference_episodes)
        if str(args.reference_episodes).strip()
        else None
    )
    reference = open_reference(reference_path)
    all_rows: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            for family_index, family in enumerate(FAMILIES):
                family_rows = evaluate_family(
                    universe=universe,
                    family=family,
                    family_index=family_index,
                    seed=seed,
                    num_symbols=args.num_symbols,
                    support_k=args.support_k,
                    support_noise=args.support_noise,
                    query_out_of_support_prob=(
                        args.query_out_of_support_prob
                    ),
                    test_limit=args.test_limit_per_family,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    reference=reference,
                )
                all_rows.extend(family_rows)
                print(
                    f"seed={seed} family={family} "
                    f"conditioned={subset_average(family_rows, 'family_conditioned_correct'):.4f} "
                    f"agnostic={subset_average(family_rows, 'family_balanced_correct'):.4f} "
                    f"family_id={subset_average(family_rows, 'family_correct'):.4f}",
                    flush=True,
                )
        if reference is not None:
            reference.finish()
    except BaseException:
        if reference is not None and not reference.handle.closed:
            reference.handle.close()
        raise

    expected_rows = len(seeds) * len(FAMILIES) * args.test_limit_per_family
    if len(all_rows) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows, got {len(all_rows)}."
        )
    if len({str(row["episode_id"]) for row in all_rows}) != expected_rows:
        raise AssertionError("Duplicate Phase 2.9 episode IDs.")

    seed_rows = seed_scopes(all_rows)
    group_rows = aggregate_seed_scopes(seed_rows)
    comparisons = comparison_rows(seed_rows)
    gate = decision_gate(group_rows)
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "episodes.csv", all_rows)
    write_csv(out_dir / "seed_summary.csv", seed_rows)
    write_csv(out_dir / "group_summary.csv", group_rows)
    write_csv(out_dir / "comparisons.csv", comparisons)
    write_json(out_dir / "gate.json", gate)
    write_analysis(
        out_dir / "analysis.md",
        group_rows=group_rows,
        gate=gate,
        metadata=universe.metadata,
    )
    manifest = {
        "protocol": {
            "seeds": seeds,
            "families": list(FAMILIES),
            "num_symbols": args.num_symbols,
            "support_k": args.support_k,
            "support_noise": args.support_noise,
            "query_out_of_support_prob": (
                args.query_out_of_support_prob
            ),
            "test_limit_per_family": args.test_limit_per_family,
            "episode_count": len(all_rows),
            "device": str(device),
        },
        "candidate_universe": universe.metadata,
        "reference": (
            {
                "file": "/".join(reference_path.parts[-2:]),
                "sha256": file_sha256(reference_path),
                "matched_rows": expected_rows,
            }
            if reference_path is not None
            else None
        ),
        "outputs": {
            "episodes_sha256": file_sha256(out_dir / "episodes.csv"),
            "seed_summary_sha256": file_sha256(
                out_dir / "seed_summary.csv"
            ),
            "group_summary_sha256": file_sha256(
                out_dir / "group_summary.csv"
            ),
            "comparisons_sha256": file_sha256(
                out_dir / "comparisons.csv"
            ),
            "gate_sha256": file_sha256(out_dir / "gate.json"),
            "analysis_sha256": file_sha256(out_dir / "analysis.md"),
        },
        "implementation": {
            "evaluator_sha256": file_sha256(Path(__file__)),
            "validator_sha256": file_sha256(
                Path(__file__).with_name(
                    "validate_symbolic_family_agnostic_ceiling.py"
                )
            ),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"Wrote {len(all_rows)} rows to {out_dir / 'episodes.csv'}")
    print(f"PHASE2_9_GATE_PASSED={gate['passed']}")
    print(f"PHASE2_9_DECISION={gate['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
