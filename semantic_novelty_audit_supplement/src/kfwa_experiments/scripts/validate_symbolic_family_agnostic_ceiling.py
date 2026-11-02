from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_symbolic_family_agnostic_ceiling import (
    FAMILIES,
    build_oracle_universe,
    family_component_posterior,
    family_conditioned_posterior,
    family_exclusions,
    posterior_from_prior,
)
from kfwa.datasets import (
    SymbolicRuleMappingDataset,
    build_symbolic_family_transfer_evaluation,
)


def scalar_log_likelihoods(
    rule_table: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    support_noise: float,
    num_symbols: int,
) -> torch.Tensor:
    values: list[float] = []
    support_digits = batch["support_digits"][0]
    support_labels = batch["support_labels"][0]
    support_mask = batch["support_mask"][0]
    for outputs in rule_table:
        log_likelihood = 0.0
        for digit, label, valid in zip(
            support_digits,
            support_labels,
            support_mask,
            strict=True,
        ):
            if not bool(valid):
                continue
            if int(outputs[int(digit)]) == int(label):
                log_likelihood += math.log1p(-support_noise)
            else:
                log_likelihood += math.log(
                    support_noise / float(num_symbols - 1)
                )
        values.append(log_likelihood)
    return torch.tensor(values, dtype=torch.float64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the family-agnostic Bayesian ceiling."
    )
    parser.parse_args()

    universe = build_oracle_universe(
        num_symbols=20,
        support_noise=0.1,
        device=torch.device("cpu"),
    )
    metadata = universe.metadata
    if metadata["component_entry_count"] != 1686:
        raise AssertionError("Unexpected family-component count.")
    if metadata["unique_function_count"] != 1644:
        raise AssertionError("Unexpected unique-function count.")
    if metadata["duplicate_component_count"] != 42:
        raise AssertionError("Unexpected duplicate-component count.")
    if metadata["multiplicity_histogram"] != {
        "1": 1604,
        "2": 38,
        "3": 2,
    }:
        raise AssertionError("Unexpected cross-family multiplicities.")

    if not torch.isclose(
        universe.family_balanced_prior.sum(),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1.0e-12,
    ):
        raise AssertionError("Family-balanced prior is not normalized.")
    if not torch.isclose(
        universe.uniform_function_prior.sum(),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1.0e-12,
    ):
        raise AssertionError("Uniform-function prior is not normalized.")
    if not torch.all(universe.family_balanced_prior > 0.0):
        raise AssertionError(
            "A family-agnostic prior must retain every union function."
        )

    multiplicities = universe.family_masks.sum(dim=0)
    duplicate_index = int(multiplicities.eq(3).nonzero()[0])
    expected_duplicate_mass = sum(
        1.0 / (len(FAMILIES) * len(universe.family_ids[family]))
        for family_index, family in enumerate(FAMILIES)
        if bool(universe.family_masks[family_index, duplicate_index])
    )
    observed_duplicate_mass = float(
        universe.family_balanced_prior[duplicate_index]
    )
    if not math.isclose(
        observed_duplicate_mass,
        expected_duplicate_mass,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise AssertionError("Duplicate function prior mass was not summed.")

    family = "composition"
    family_index = FAMILIES.index(family)
    rules, eligible_ids, _ = build_symbolic_family_transfer_evaluation(
        num_symbols=20,
        family=family,
        excluded_families=family_exclusions(family),
    )
    dataset = SymbolicRuleMappingDataset(
        size=3,
        rules=rules,
        active_function_ids=eligible_ids,
        seen_function_ids=set(),
        min_support_k=8,
        max_support_k=8,
        query_out_of_support_prob=1.0,
        support_noise=0.1,
        seed=90_000,
    )
    batch, target = next(iter(DataLoader(dataset, batch_size=3)))
    predictor = universe.predictor
    vector_log_likelihood = predictor.support_log_likelihood(batch)
    scalar_log_likelihood = scalar_log_likelihoods(
        predictor.rule_table,
        {key: value[:1] for key, value in batch.items()},
        support_noise=0.1,
        num_symbols=20,
    )
    if not torch.allclose(
        vector_log_likelihood[0].double(),
        scalar_log_likelihood,
        atol=2.0e-6,
        rtol=0.0,
    ):
        raise AssertionError("Vector likelihood disagrees with scalar enumeration.")

    balanced_posterior, log_evidence = posterior_from_prior(
        vector_log_likelihood,
        universe.family_balanced_prior,
    )
    uniform_posterior, _ = posterior_from_prior(
        vector_log_likelihood,
        universe.uniform_function_prior,
    )
    conditioned_posterior = family_conditioned_posterior(
        vector_log_likelihood,
        universe.family_masks[family_index],
    )
    family_posterior = family_component_posterior(
        vector_log_likelihood,
        universe.component_prior,
        log_evidence,
    )
    for posterior in (
        balanced_posterior,
        uniform_posterior,
        conditioned_posterior,
        family_posterior,
    ):
        if not torch.isfinite(posterior).all():
            raise AssertionError("Posterior contains a non-finite value.")
        if not torch.allclose(
            posterior.sum(dim=1),
            torch.ones(posterior.shape[0]),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise AssertionError("Posterior is not normalized.")
    if torch.any(
        conditioned_posterior[
            :,
            ~universe.family_masks[family_index],
        ]
        != 0.0
    ):
        raise AssertionError(
            "Family-conditioned posterior leaked outside its family."
        )

    scalar_scores = (
        scalar_log_likelihood
        + universe.family_balanced_prior.log()
    )
    scalar_posterior = torch.softmax(scalar_scores, dim=0)
    if not torch.allclose(
        balanced_posterior[0].double(),
        scalar_posterior,
        atol=2.0e-7,
        rtol=2.0e-7,
    ):
        raise AssertionError("Vector posterior disagrees with scalar enumeration.")
    query_digit = int(batch["query_digit"][0])
    scalar_predictive = torch.zeros(20, dtype=torch.float64)
    for function_index, probability in enumerate(scalar_posterior):
        label = int(predictor.rule_table[function_index, query_digit])
        scalar_predictive[label] += probability
    vector_predictive = predictor.predictive_probs(
        {key: value[:1] for key, value in batch.items()},
        balanced_posterior[:1],
    )[0]
    if not torch.allclose(
        vector_predictive.double(),
        scalar_predictive,
        atol=2.0e-7,
        rtol=2.0e-7,
    ):
        raise AssertionError(
            "Vector posterior predictive disagrees with scalar enumeration."
        )
    if int(target[0]) != int(
        predictor.rule_table[
            torch.searchsorted(
                predictor.function_ids,
                batch["function_id"][0],
            ),
            query_digit,
        ]
    ):
        raise AssertionError("Dataset target disagrees with union rule table.")

    print(
        "family components unique multiplicity checksum\n"
        f"{metadata['component_entry_count']} "
        f"{metadata['unique_function_count']} "
        f"{metadata['multiplicity_histogram']} "
        f"{metadata['candidate_function_checksum']}"
    )
    print("SYMBOLIC_FAMILY_AGNOSTIC_CEILING_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
