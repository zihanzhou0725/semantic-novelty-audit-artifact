from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import get_symbolic_rule_mapping_loaders
from kfwa.symbolic import (
    SymbolicBayesianPredictor,
    SymbolicMaximumLikelihoodPredictor,
    SymbolicRuleOracle,
)


def brute_force_predictions(
    rule_table: torch.Tensor,
    batch: dict[str, torch.Tensor],
    support_noise: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independently compute posterior, predictive probabilities, and ML ties."""

    batch_size = batch["query_digit"].shape[0]
    function_count = rule_table.shape[0]
    num_symbols = rule_table.shape[1]
    posterior = torch.zeros(batch_size, function_count, dtype=torch.float64)
    predictive = torch.zeros(batch_size, num_symbols, dtype=torch.float64)
    ml_mask = torch.zeros(batch_size, function_count, dtype=torch.bool)

    for row in range(batch_size):
        log_likelihoods: list[float] = []
        for function_index in range(function_count):
            log_likelihood = 0.0
            possible = True
            for support_index in range(batch["support_digits"].shape[1]):
                if not bool(batch["support_mask"][row, support_index]):
                    continue
                digit = int(batch["support_digits"][row, support_index])
                observed = int(batch["support_labels"][row, support_index])
                expected = int(rule_table[function_index, digit])
                if observed == expected:
                    log_likelihood += math.log1p(-support_noise)
                elif support_noise > 0.0:
                    log_likelihood += math.log(support_noise / float(num_symbols - 1))
                else:
                    possible = False
                    break
            log_likelihoods.append(log_likelihood if possible else float("-inf"))

        log_likelihood_tensor = torch.tensor(log_likelihoods, dtype=torch.float64)
        if not torch.isfinite(log_likelihood_tensor).any():
            raise AssertionError("Generated evidence has zero likelihood under every candidate function.")
        posterior[row] = torch.softmax(log_likelihood_tensor, dim=0)
        maximum = log_likelihood_tensor.max()
        ml_mask[row] = torch.isclose(log_likelihood_tensor, maximum, atol=1.0e-12, rtol=1.0e-12)
        query_digit = int(batch["query_digit"][row])
        for function_index in range(function_count):
            query_label = int(rule_table[function_index, query_digit])
            predictive[row, query_label] += posterior[row, function_index]
    return posterior, predictive, ml_mask


def mutate_metadata(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    changed = {key: value.clone() for key, value in batch.items()}
    for key in (
        "rule_id",
        "identity_id",
        "function_id",
        "function_alias_count",
        "rule_family",
        "context_id",
        "support_k",
    ):
        changed[key] = changed[key] + 100_000
    for key in ("seen_function", "seen_context", "query_in_support"):
        changed[key] = ~changed[key]
    return changed


def mutate_padding(batch: dict[str, torch.Tensor], num_symbols: int) -> dict[str, torch.Tensor]:
    changed = {key: value.clone() for key, value in batch.items()}
    padding = ~changed["support_mask"]
    changed["support_digits"][padding] = num_symbols - 1
    changed["support_labels"][padding] = num_symbols - 2
    return changed


def validate_noise_level(
    support_noise: float,
    seed: int,
    num_symbols: int,
    rule_families: str,
) -> None:
    bundle = get_symbolic_rule_mapping_loaders(
        batch_size=16,
        num_workers=0,
        seed=seed,
        train_limit=32,
        val_limit=32,
        test_limit=32,
        num_symbols=num_symbols,
        support_min_k=4,
        support_max_k=8,
        rule_families=rule_families,
        holdout_rule_frac=0.25,
        query_out_of_support_prob=1.0,
        support_noise=support_noise,
    )
    batch, _ = next(iter(bundle.test))
    common = {
        "num_symbols": num_symbols,
        "rule_families": rule_families,
        "target_dim": num_symbols,
        "support_noise": support_noise,
    }
    bayes = SymbolicBayesianPredictor(**common).eval()
    ml = SymbolicMaximumLikelihoodPredictor(**common).eval()
    legacy = SymbolicRuleOracle(**common).eval()

    if not torch.equal(bayes.rule_table, ml.rule_table) or not torch.equal(bayes.rule_table, legacy.rule_table):
        raise AssertionError("Oracle function tables differ.")
    if bayes.rule_table.unique(dim=0).shape[0] != bayes.rule_table.shape[0]:
        raise AssertionError("Oracle table contains duplicate complete functions.")

    expected_posterior, expected_predictive, expected_ml_mask = brute_force_predictions(
        bayes.rule_table,
        batch,
        support_noise,
    )
    actual_posterior = bayes.posterior_function_probs(batch).double()
    actual_predictive = bayes(batch).softmax(dim=-1).double()
    torch.testing.assert_close(actual_posterior, expected_posterior, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(actual_predictive, expected_predictive, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(
        actual_posterior.sum(dim=1),
        torch.ones(actual_posterior.shape[0], dtype=torch.float64),
        atol=1.0e-5,
        rtol=0.0,
    )

    actual_ml_mask = ml.maximum_likelihood_mask(batch)
    if not torch.equal(actual_ml_mask, expected_ml_mask):
        raise AssertionError("Maximum-likelihood candidate set disagrees with brute force.")

    current_noise_threshold = float(num_symbols - 1) / float(num_symbols)
    if support_noise < current_noise_threshold:
        legacy_mask = legacy.support_match_counts(batch)
        legacy_mask = legacy_mask.eq(legacy_mask.max(dim=1, keepdim=True).values)
        if not torch.equal(legacy_mask, actual_ml_mask):
            raise AssertionError("Legacy max-match oracle is not equivalent to ML below the noise threshold.")
        if not torch.equal(legacy(batch), ml(batch)):
            raise AssertionError("Legacy oracle and ML logits differ under the current noise model.")

    for predictor in (bayes, ml, legacy):
        reference = predictor(batch)
        if not torch.equal(reference, predictor(mutate_metadata(batch))):
            raise AssertionError(f"{predictor.predictor_type} reads evaluation-only metadata.")
        torch.testing.assert_close(
            reference,
            predictor(mutate_padding(batch, num_symbols)),
            atol=0.0,
            rtol=0.0,
        )

    print(
        f"noise={support_noise:.2f} functions={bayes.rule_table.shape[0]} "
        f"bayes_bruteforce=ok ml_bruteforce=ok legacy_equals_ml=ok"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate exact symbolic Bayesian and ML predictors.")
    parser.add_argument("--num-symbols", type=int, default=20)
    parser.add_argument("--rule-families", default="affine,block,parity")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for support_noise in (0.0, 0.1, 0.3):
        validate_noise_level(
            support_noise=support_noise,
            seed=args.seed,
            num_symbols=args.num_symbols,
            rule_families=args.rule_families,
        )
    print("SYMBOLIC_ORACLES_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
