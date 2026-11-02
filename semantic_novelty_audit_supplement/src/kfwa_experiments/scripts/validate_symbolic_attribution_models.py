from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kfwa.models import build_model
from kfwa.symbolic import (
    DigitFourierRuleFeatures,
    GenericLatentAttentionReadout,
    HarmonicPhaseClusterReadout,
)
from kfwa.utils import count_parameters, set_seed


ATTRIBUTION_MODELS = {
    "A0": "symbolic_transformer",
    "A2": "symbolic_transformer_latent_pool",
    "A4": "symbolic_transformer_lop",
    "A6": "symbolic_transformer_lop_rule_features",
    "A7": "symbolic_transformer_rule_features",
    "A7-no-fourier": "symbolic_transformer_rule_features_no_fourier",
    "A7-no-parity": "symbolic_transformer_rule_features_no_parity",
    "A7-no-block": "symbolic_transformer_rule_features_no_block",
    "A7-no-scaled": "symbolic_transformer_rule_features_no_scaled",
    "A7-no-delta": "symbolic_transformer_rule_features_no_delta",
    "A8": "symbolic_transformer_param_matched",
}


def model_args(model: str, max_support_k: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        num_symbols=20,
        support_max_k=max_support_k,
        model_dim=64,
        layers=2,
        heads=4,
        ff_dim=128,
        dropout=0.05,
        rule_fourier_bands=4,
        lop_oscillators=32,
        lop_harmonics=4,
        lop_clusters=4,
        matched_model_dim=92,
        matched_ff_dim=192,
    )


def build_batch(max_support_k: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if max_support_k < 6:
        raise ValueError("max_support_k must be at least 6 for this validation.")
    batch_size = 3
    support_digits = torch.zeros(batch_size, max_support_k, dtype=torch.long)
    support_labels = torch.zeros(batch_size, max_support_k, dtype=torch.long)
    support_mask = torch.zeros(batch_size, max_support_k, dtype=torch.bool)
    lengths = [4, 6, max_support_k]
    for row, length in enumerate(lengths):
        digits = (torch.arange(length) * (row + 2) + row).remainder(20)
        labels = (3 * digits + row + 1).remainder(20)
        support_digits[row, :length] = digits
        support_labels[row, :length] = labels
        support_mask[row, :length] = True
    batch = {
        "query_digit": torch.tensor([11, 13, 17], dtype=torch.long),
        "support_digits": support_digits,
        "support_labels": support_labels,
        "support_mask": support_mask,
        "support_k": torch.tensor(lengths, dtype=torch.long),
        "rule_id": torch.tensor([1, 2, 3], dtype=torch.long),
        "identity_id": torch.tensor([1, 2, 3], dtype=torch.long),
        "function_id": torch.tensor([101, 102, 103], dtype=torch.long),
        "function_alias_count": torch.ones(batch_size, dtype=torch.long),
        "rule_family": torch.tensor([0, 1, 2], dtype=torch.long),
        "context_id": torch.tensor([0, 1, 2], dtype=torch.long),
        "seen_function": torch.tensor([True, False, True]),
        "seen_context": torch.tensor([True, False, True]),
        "query_in_support": torch.tensor([False, False, False]),
    }
    targets = torch.tensor([5, 7, 9], dtype=torch.long)
    return batch, targets


def changed_metadata(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
        changed[key] = changed[key] + 10_000
    for key in ("seen_function", "seen_context", "query_in_support"):
        changed[key] = ~changed[key]
    return changed


def changed_padding(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    changed = {key: value.clone() for key, value in batch.items()}
    padding = ~changed["support_mask"]
    changed["support_digits"][padding] = 19
    changed["support_labels"][padding] = 17
    return changed


def changed_evidence(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    changed = {key: value.clone() for key, value in batch.items()}
    changed["support_labels"][0, 0] = (changed["support_labels"][0, 0] + 7).remainder(20)
    return changed


def validate_model(
    attribution_id: str,
    model_name: str,
    max_support_k: int,
    batch: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> torch.nn.Module:
    set_seed(1234)
    model = build_model(model_args(model_name, max_support_k), context_dim=0, target_dim=20)
    if getattr(model, "attribution_id", None) != attribution_id:
        raise AssertionError(f"{model_name} has the wrong attribution_id.")

    model.eval()
    with torch.no_grad():
        logits = model(batch)
        metadata_logits = model(changed_metadata(batch))
        padding_logits = model(changed_padding(batch))
        evidence_logits = model(changed_evidence(batch))
    if logits.shape != (targets.shape[0], 20) or not torch.isfinite(logits).all():
        raise AssertionError(f"{model_name} produced invalid logits.")
    if not torch.equal(logits, metadata_logits):
        raise AssertionError(f"{model_name} reads evaluation-only metadata.")
    if not torch.allclose(logits, padding_logits, atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError(f"{model_name} is sensitive to padded support values.")
    if torch.allclose(logits, evidence_logits, atol=1.0e-7, rtol=1.0e-7):
        raise AssertionError(f"{model_name} is insensitive to valid support evidence.")

    model.train()
    loss = F.cross_entropy(model(batch), targets)
    loss.backward()
    gradients = [param.grad for param in model.parameters() if param.requires_grad and param.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError(f"{model_name} has missing or non-finite gradients.")
    if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
        raise AssertionError(f"{model_name} has no non-zero gradients.")
    model.eval()
    with torch.no_grad():
        model(batch)
    return model


def validate_architecture(models: dict[str, torch.nn.Module]) -> None:
    for attribution_id in ("A0", "A2", "A4", "A8"):
        feature_modules = [
            module
            for module in models[attribution_id].modules()
            if isinstance(module, DigitFourierRuleFeatures)
        ]
        if feature_modules:
            raise AssertionError(f"{attribution_id} must not instantiate explicit rule features.")
    explicit_feature_ids = (
        "A6",
        "A7",
        "A7-no-fourier",
        "A7-no-parity",
        "A7-no-block",
        "A7-no-scaled",
        "A7-no-delta",
    )
    for attribution_id in explicit_feature_ids:
        if sum(
            isinstance(module, DigitFourierRuleFeatures)
            for module in models[attribution_id].modules()
        ) != 1:
            raise AssertionError(f"{attribution_id} must contain exactly one explicit rule-feature module.")

    if not isinstance(models["A2"].readout, GenericLatentAttentionReadout):
        raise AssertionError("A2 is not using generic latent attention pooling.")
    if not isinstance(models["A4"].readout, HarmonicPhaseClusterReadout):
        raise AssertionError("A4 is not using the harmonic phase-cluster readout.")
    if not isinstance(models["A6"].readout, HarmonicPhaseClusterReadout):
        raise AssertionError("A6 is not using the harmonic phase-cluster readout.")
    if getattr(models["A4"], "uses_kuramoto", None) is not False:
        raise AssertionError("A4 must not use Kuramoto dynamics.")
    if getattr(models["A6"], "uses_kuramoto", None) is not False:
        raise AssertionError("A6 must not use Kuramoto dynamics.")

    a0_shapes = {
        name: tuple(param.shape)
        for name, param in models["A0"].named_parameters()
        if not name.startswith("head.")
    }
    for attribution_id in ("A2", "A4"):
        trunk_shapes = {
            name: tuple(param.shape)
            for name, param in models[attribution_id].transformer.named_parameters()
            if not name.startswith("head.")
        }
        if trunk_shapes != a0_shapes:
            raise AssertionError(f"{attribution_id} does not share the A0 Transformer architecture.")
        if any(param.requires_grad for param in models[attribution_id].transformer.head.parameters()):
            raise AssertionError(f"{attribution_id} left the unused Transformer head trainable.")
    a7_shapes = {
        name: tuple(param.shape)
        for name, param in models["A7"].named_parameters()
        if not name.startswith("head.")
    }
    a6_trunk_shapes = {
        name: tuple(param.shape)
        for name, param in models["A6"].transformer.named_parameters()
        if not name.startswith("head.")
    }
    if a6_trunk_shapes != a7_shapes:
        raise AssertionError("A6 does not share the A7 explicit-feature Transformer architecture.")

    counts = {key: count_parameters(model) for key, model in models.items()}
    for attribution_id in explicit_feature_ids[2:]:
        if counts[attribution_id] != counts["A7"]:
            raise AssertionError(f"{attribution_id} is not parameter-matched to full A7.")
    if abs(counts["A2"] - counts["A4"]) / counts["A4"] > 0.01:
        raise AssertionError("A2 and A4 differ by more than 1% in trainable parameters.")
    if abs(counts["A8"] - counts["A4"]) / counts["A4"] > 0.01:
        raise AssertionError("A8 and A4 differ by more than 1% in trainable parameters.")

    a2_attention = models["A2"].readout.last_attention
    if a2_attention is None or not torch.allclose(
        a2_attention.sum(dim=-1),
        torch.ones_like(a2_attention.sum(dim=-1)),
        atol=1.0e-6,
    ):
        raise AssertionError("A2 attention weights are unavailable or not normalized.")
    a4_readout = models["A4"].readout
    if (
        a4_readout.last_theta is None
        or a4_readout.last_order_parameters is None
        or a4_readout.last_cluster_mass is None
    ):
        raise AssertionError("A4 phase diagnostics were not populated.")


def validate_feature_masks(
    models: dict[str, torch.nn.Module],
    batch: dict[str, torch.Tensor],
) -> None:
    full_module = models["A7"].rule_features
    full_features = full_module.token_features(batch)
    digit_dim = full_module.digit_dim
    fourier_dim = 2 * full_module.fourier_bands
    parity_index = fourier_dim
    block_index = fourier_dim + 1
    scaled_index = fourier_dim + 2
    expected_indices = {
        "A7-no-fourier": [
            group * digit_dim + index
            for group in range(3)
            for index in range(fourier_dim)
        ],
        "A7-no-parity": [
            *(group * digit_dim + parity_index for group in range(3)),
            3 * digit_dim,
        ],
        "A7-no-block": [
            *(group * digit_dim + block_index for group in range(3)),
            3 * digit_dim + 1,
        ],
        "A7-no-scaled": [
            group * digit_dim + scaled_index for group in range(3)
        ],
        "A7-no-delta": list(range(2 * digit_dim, 3 * digit_dim)),
    }
    for attribution_id, indices in expected_indices.items():
        expected = full_features.clone()
        expected[..., indices] = 0.0
        actual = models[attribution_id].rule_features.token_features(batch)
        if not torch.equal(actual, expected):
            raise AssertionError(f"{attribution_id} does not mask exactly its feature group.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate A0/A2/A4/A7/A8 symbolic attribution models.")
    parser.add_argument("--max-support-k", type=int, default=8)
    args = parser.parse_args()

    batch, targets = build_batch(args.max_support_k)
    models = {
        attribution_id: validate_model(
            attribution_id,
            model_name,
            args.max_support_k,
            batch,
            targets,
        )
        for attribution_id, model_name in ATTRIBUTION_MODELS.items()
    }
    validate_architecture(models)
    validate_feature_masks(models, batch)

    print("id  model                                        params  explicit  readout")
    for attribution_id, model_name in ATTRIBUTION_MODELS.items():
        model = models[attribution_id]
        print(
            f"{attribution_id:<3} {model_name:<44} "
            f"{count_parameters(model):>7}  "
            f"{str(model.uses_explicit_rule_features):<8}  "
            f"{model.readout_type}"
        )
    print("SYMBOLIC_ATTRIBUTION_MODELS_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
