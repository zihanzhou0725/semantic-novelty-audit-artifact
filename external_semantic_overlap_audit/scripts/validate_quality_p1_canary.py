#!/usr/bin/env python3
"""Run non-training P1 canaries before manifests are frozen."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quality_p1_common import (
    MODELS,
    P1_SCHEMA_VERSION,
    ROOT,
    SUPPORT_SIZE,
    build_eval_arrays,
    build_train_arrays,
    model_args,
    set_deterministic_state,
)


KFWA_ROOT = (
    ROOT
    / "semantic_novelty_audit_supplement"
    / "src"
    / "kfwa_experiments"
)
if str(KFWA_ROOT) not in sys.path:
    sys.path.insert(0, str(KFWA_ROOT))

from kfwa.models import build_model  # noqa: E402


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("ascii"))
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def make_batch(
    arrays: dict[str, np.ndarray],
    arm: str,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    count = len(indices)
    family = torch.from_numpy(
        arrays[f"family_id_{arm}"][indices].astype(np.int64)
    )
    batch = {
        "query_digit": torch.from_numpy(
            arrays["query_digit"][indices].astype(np.int64)
        ),
        "support_digits": torch.from_numpy(
            arrays["support_digits"][indices].astype(np.int64)
        ),
        "support_labels": torch.from_numpy(
            arrays[f"support_labels_{arm}"][indices].astype(np.int64)
        ),
        "support_mask": torch.ones((count, SUPPORT_SIZE), dtype=torch.bool),
        "support_k": torch.full((count,), SUPPORT_SIZE, dtype=torch.long),
        "rule_id": torch.from_numpy(
            arrays[f"identity_id_{arm}"][indices].astype(np.int64)
        ),
        "identity_id": torch.from_numpy(
            arrays[f"identity_id_{arm}"][indices].astype(np.int64)
        ),
        "function_id": torch.from_numpy(
            arrays[f"function_id_{arm}"][indices].astype(np.int64)
        ),
        "function_alias_count": torch.from_numpy(
            arrays[f"function_alias_count_{arm}"][indices].astype(np.int64)
        ),
        "rule_family": family,
        "context_id": family,
        "seen_identity": torch.ones(count, dtype=torch.bool),
        "seen_function": torch.ones(count, dtype=torch.bool),
        "seen_context": torch.ones(count, dtype=torch.bool),
        "query_in_support": torch.zeros(count, dtype=torch.bool),
    }
    target = torch.from_numpy(
        arrays[f"target_{arm}"][indices].astype(np.int64)
    )
    return (
        {key: value.to(device) for key, value in batch.items()},
        target.to(device),
    )


def validate_array_canaries(
    train: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
) -> dict[str, Any]:
    if train["support_digits"].shape != (20_000, 8):
        raise AssertionError("Unexpected P1 train support shape.")
    if evaluation["support_digits"].shape != (10_000, 8):
        raise AssertionError("Unexpected P1 eval support shape.")
    base = train["slot_kind"] == 0
    target = ~base
    if int(base.sum()) != 19_200 or int(target.sum()) != 800:
        raise AssertionError("Unexpected base/target slot counts.")
    for field in (
        "function_id",
        "identity_id",
        "family_id",
        "function_alias_count",
        "support_labels",
        "target",
    ):
        if not np.array_equal(
            train[f"{field}_A"][base],
            train[f"{field}_B"][base],
        ):
            raise AssertionError(f"Base pairing failed for {field}.")
    if np.any(train["function_id_A"][target] == train["function_id_B"][target]):
        raise AssertionError("A target pair exposes the same function in both arms.")
    if not np.all(train["family_id_A"][target] == 2) or not np.all(
        train["family_id_B"][target] == 2
    ):
        raise AssertionError("Target exposure is not represented by parity aliases.")
    if not np.all(evaluation["family_id"] == 0):
        raise AssertionError("Evaluation identities are not affine.")
    return {
        "base_slots": int(base.sum()),
        "target_slots": int(target.sum()),
        "train_rows": len(train["query_digit"]),
        "eval_rows": len(evaluation["query_digit"]),
    }


def run_canaries() -> dict[str, Any]:
    train = build_train_arrays(0)
    evaluation = build_eval_arrays()
    array_report = validate_array_canaries(train, evaluation)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    batch_indices = np.arange(64, dtype=np.int64)
    model_reports: dict[str, Any] = {}
    for model_name in MODELS:
        set_deterministic_state(0)
        first = build_model(model_args(model_name), context_dim=0, target_dim=20)
        first_hash = state_sha256(first)
        set_deterministic_state(0)
        second = build_model(model_args(model_name), context_dim=0, target_dim=20)
        second_hash = state_sha256(second)
        if first_hash != second_hash:
            raise AssertionError(f"Initialization is not stable for {model_name}.")

        first.to(device)
        first.train()
        batch, target = make_batch(train, "A", batch_indices, device)
        logits = first(batch)
        if isinstance(logits, dict):
            logits = logits["logits"]
        elif isinstance(logits, tuple):
            logits = logits[0]
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in first.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(
            bool(torch.isfinite(gradient).all().item())
            for gradient in gradients
        ):
            raise AssertionError(f"Invalid gradients for {model_name}.")
        model_reports[model_name] = {
            "initial_state_sha256": first_hash,
            "loss": float(loss.detach().cpu().item()),
            "parameters": sum(
                parameter.numel()
                for parameter in first.parameters()
                if parameter.requires_grad
            ),
        }
        del first, second, batch, target, logits, loss, gradients
        if device.type == "mps":
            torch.mps.empty_cache()

    return {
        "array_canaries": array_report,
        "device": str(device),
        "models": model_reports,
        "platform": platform.platform(),
        "schema_version": P1_SCHEMA_VERSION,
        "status": "passed",
        "torch_version": torch.__version__,
        "training_updates_performed": 0,
    }


def main() -> int:
    report = run_canaries()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
