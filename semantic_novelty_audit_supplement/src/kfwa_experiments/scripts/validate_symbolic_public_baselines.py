from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.datasets import (
    build_symbolic_rules,
    get_symbolic_rule_mapping_loaders,
    split_symbolic_identities,
)
from kfwa.models import build_model


PUBLIC_MODELS = (
    "symbolic_deepsets",
    "symbolic_relation_network",
    "symbolic_set_transformer",
    "symbolic_hopfield",
    "symbolic_perceiver",
    "symbolic_transformer",
)
INVARIANT_MODELS = set(PUBLIC_MODELS) - {"symbolic_transformer"}


def model_args(model: str) -> Namespace:
    return Namespace(
        model=model,
        num_symbols=20,
        support_max_k=8,
        model_dim=64,
        hidden_dim=256,
        layers=2,
        heads=4,
        ff_dim=128,
        public_latent_count=32,
        dropout=0.05,
    )


def make_batch() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(2027)
    support_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True, False],
            [True, True, True, True, True, True, False, False],
            [True, True, True, True, True, True, True, True],
        ]
    )
    batch_size, support_size = support_mask.shape
    batch = {
        "query_digit": torch.randint(0, 20, (batch_size,), generator=generator),
        "support_digits": torch.randint(
            0,
            20,
            (batch_size, support_size),
            generator=generator,
        ),
        "support_labels": torch.randint(
            0,
            20,
            (batch_size, support_size),
            generator=generator,
        ),
        "support_mask": support_mask,
        "rule_id": torch.arange(batch_size),
        "identity_id": torch.arange(batch_size),
        "function_id": torch.arange(batch_size),
        "rule_family": torch.arange(batch_size),
        "context_id": torch.arange(batch_size),
        "seen_identity": torch.ones(batch_size, dtype=torch.bool),
        "seen_function": torch.ones(batch_size, dtype=torch.bool),
        "seen_context": torch.ones(batch_size, dtype=torch.bool),
    }
    target = torch.randint(0, 20, (batch_size,), generator=generator)
    return batch, target


def permute_support(
    batch: dict[str, torch.Tensor],
    permutation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    output = {key: value.clone() for key, value in batch.items()}
    for key in ("support_digits", "support_labels", "support_mask"):
        output[key] = output[key][:, permutation]
    return output


def validate_splits() -> None:
    strict_default = get_symbolic_rule_mapping_loaders(
        seed=0,
        train_limit=16,
        val_limit=16,
        test_limit=16,
        num_symbols=20,
        support_min_k=8,
        support_max_k=8,
        rule_families="affine,block,parity",
        support_noise=0.1,
        query_out_of_support_prob=1.0,
    )
    strict_explicit = get_symbolic_rule_mapping_loaders(
        seed=0,
        train_limit=16,
        val_limit=16,
        test_limit=16,
        num_symbols=20,
        support_min_k=8,
        support_max_k=8,
        rule_families="affine,block,parity",
        support_noise=0.1,
        query_out_of_support_prob=1.0,
        symbolic_split_mode="strict_function",
    )
    assert strict_default.metadata["function_split_checksum"] == (
        strict_explicit.metadata["function_split_checksum"]
    )
    assert strict_default.metadata["function_split_overlap"] == 0
    assert strict_default.metadata["seen_functions"] == 388
    assert strict_default.metadata["unseen_functions"] == 130

    legacy = get_symbolic_rule_mapping_loaders(
        seed=0,
        train_limit=16,
        val_limit=16,
        test_limit=16,
        num_symbols=20,
        support_min_k=8,
        support_max_k=8,
        rule_families="affine,block,parity",
        support_noise=0.1,
        query_out_of_support_prob=1.0,
        symbolic_split_mode="legacy_identity",
    )
    assert legacy.metadata["split_unit"] == "parameter_identity"
    assert legacy.metadata["seen_rules"] == 420
    assert legacy.metadata["unseen_rules"] == 140
    assert legacy.metadata["function_split_overlap"] > 0

    rules = build_symbolic_rules(20, "affine,block,parity")
    leaked_counts: list[int] = []
    for seed in range(20):
        seen, unseen = split_symbolic_identities(
            rules,
            holdout_rule_frac=0.25,
            seed=seed + 7919,
        )
        seen_functions = {rules[rule_id].function_id for rule_id in seen}
        leaked_counts.append(
            sum(
                rules[rule_id].function_id in seen_functions
                for rule_id in unseen
            )
        )
    assert sum(leaked_counts) / len(leaked_counts) == 14.8
    assert sum(leaked_counts) / (len(leaked_counts) * 140) == (
        0.10571428571428572
    )


def validate_models() -> None:
    batch, target = make_batch()
    permutation = torch.tensor([3, 0, 7, 2, 6, 1, 5, 4])
    for model_name in PUBLIC_MODELS:
        model = build_model(
            model_args(model_name),
            context_dim=0,
            target_dim=20,
        )
        model.eval()
        with torch.no_grad():
            reference = model(batch)
            assert reference.shape == (target.shape[0], 20)
            assert torch.isfinite(reference).all()

            metadata_mutation = {
                key: value.clone() for key, value in batch.items()
            }
            for key in (
                "rule_id",
                "identity_id",
                "function_id",
                "rule_family",
                "context_id",
            ):
                metadata_mutation[key] = metadata_mutation[key] + 101
            for key in ("seen_identity", "seen_function", "seen_context"):
                metadata_mutation[key] = ~metadata_mutation[key]
            assert torch.allclose(
                reference,
                model(metadata_mutation),
                atol=0.0,
                rtol=0.0,
            )

            padding_mutation = {
                key: value.clone() for key, value in batch.items()
            }
            padded = ~padding_mutation["support_mask"]
            padding_mutation["support_digits"][padded] = 19
            padding_mutation["support_labels"][padded] = 18
            assert torch.allclose(
                reference,
                model(padding_mutation),
                atol=1.0e-6,
                rtol=1.0e-6,
            )

            if model_name in INVARIANT_MODELS:
                permuted = permute_support(batch, permutation)
                assert torch.allclose(
                    reference,
                    model(permuted),
                    atol=1.0e-5,
                    rtol=1.0e-5,
                )

        model.train()
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)

        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"{model_name}: parameters={parameters} "
            f"order_invariant={model_name in INVARIANT_MODELS}"
        )


def main() -> int:
    validate_splits()
    validate_models()
    print("public baseline and split validation: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
