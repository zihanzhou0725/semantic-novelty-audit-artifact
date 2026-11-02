#!/usr/bin/env python3
"""Materialize T6 inputs and freeze its prospective training protocol."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior.common import write_jsonl  # noqa: E402
from toolkit.external_behavior_v3.training_common import (  # noqa: E402
    classes,
    load_npz,
)
from toolkit.tmlr_strengthening.artifacts import (  # noqa: E402
    sha256_file,
    write_csv,
    write_json,
    write_sha256sums,
)
from toolkit.tmlr_strengthening.clutrr_tmlr_data import (  # noqa: E402
    CANDIDATES_PATH,
    COMMON_BASE_PATH,
    EVALUATION_PATH,
    SELECTED_PAIRS_PATH,
    VFOUR_TARGETS_PATH,
    VTHREE_TARGETS_PATH,
    build_t6_design,
)
from toolkit.tmlr_strengthening.clutrr_tmlr_train import (  # noqa: E402
    BETAS,
    EPOCHS,
    EPSILON,
    EVAL_BATCH_SIZE,
    FIXED_LENGTH,
    GRADIENT_CLIP,
    LEARNING_RATE,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
    build_execution_inputs,
    initialize_model,
)


NAMESPACE = "tmlrs:t6"
SEEDS = list(range(4))
MANIFEST_DIR = (
    ROOT
    / "manifests"
    / "tmlr-strengthening-v1"
    / "t6_semantic_vs_surface"
)
INPUT_DIR = MANIFEST_DIR / "execution-inputs"
PROTOCOL_DIR = (
    ROOT
    / "protocols"
    / "tmlr-strengthening-v1"
    / "t6_semantic_vs_surface"
)
PROTOCOL_PATH = PROTOCOL_DIR / "protocol.json"
DATA_MODULE = (
    ROOT
    / "toolkit"
    / "tmlr_strengthening"
    / "clutrr_tmlr_data.py"
)
TRAIN_MODULE = (
    ROOT
    / "toolkit"
    / "tmlr_strengthening"
    / "clutrr_tmlr_train.py"
)


def main() -> None:
    if PROTOCOL_PATH.exists():
        raise FileExistsError(
            f"Frozen protocol already exists: {PROTOCOL_PATH}"
        )
    design = build_t6_design()
    manifest = build_execution_inputs(design, INPUT_DIR, classes())

    allocation_rows = []
    decoy_rows = []
    for record in design["allocation"]:
        allocation_rows.append(
            {
                "pattern_index": record["pattern_index"],
                "semantic_id": record["semantic_id"],
                "target_relation": record["target_relation"],
                "edge_types": "|".join(record["edge_types"]),
                "tier": record["tier"],
                "decoy_count": record["decoy_count"],
                "decoy_semantic_count": record["decoy_semantic_count"],
                "included_main": record["included_main"],
            }
        )
        for row in record["decoy_rows"]:
            decoy_rows.append(
                {
                    "pattern_index": record["pattern_index"],
                    "pattern_semantic_id": record["semantic_id"],
                    "target_relation": record["target_relation"],
                    "tier": record["tier"],
                    "surface_id": row["surface_id"],
                    "semantic_id": row["semantic_id"],
                    "edge_types": row["edge_types"],
                }
            )
    write_csv(INPUT_DIR / "decoy_allocation.csv", allocation_rows)
    write_jsonl(INPUT_DIR / "decoy_rows.jsonl", decoy_rows)
    for arm, rows in design["arm_rows"].items():
        (INPUT_DIR / f"arm_{arm}_surface_ids.txt").write_text(
            "\n".join(str(row["surface_id"]) for row in rows) + "\n",
            encoding="ascii",
        )
    (INPUT_DIR / "evaluation_surface_ids.txt").write_text(
        "\n".join(
            str(row["surface_id"]) for row in design["evaluation_rows"]
        )
        + "\n",
        encoding="ascii",
    )

    cache = load_npz(INPUT_DIR / "tensor_cache.npz")
    model, initial_hash = initialize_model(0, NAMESPACE)
    model.eval()
    with torch.no_grad():
        smoke_logits = model(
            torch.from_numpy(cache["token_ids"][:2].astype("int64")),
            torch.from_numpy(cache["attention_mask"][:2]),
        )
    if smoke_logits.shape != (2, 23) or not torch.isfinite(
        smoke_logits
    ).all():
        raise RuntimeError("T6 forward-pass smoke check failed.")

    input_artifacts = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(INPUT_DIR.iterdir())
        if path.is_file()
        and path.name not in {"execution_inputs_manifest.json", "SHA256SUMS"}
    }
    manifest["artifacts"] = input_artifacts
    manifest["smoke_initial_model_sha256"] = initial_hash
    write_json(INPUT_DIR / "execution_inputs_manifest.json", manifest)
    write_sha256sums(INPUT_DIR)
    protocol_artifacts = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(INPUT_DIR.iterdir())
        if path.is_file()
    }

    protocol = {
        "analysis": {
            "bootstrap": {
                "cluster": "pattern",
                "replicates": 2000,
                "seed": 0,
            },
            "contrasts": ["C1-C0", "C2-C0", "C1-C2"],
            "decision_rules": [
                {
                    "condition": (
                        "C1-C0 cluster CI is above zero and either C2-C0 "
                        "contains zero or C1-C2 is above zero"
                    ),
                    "interpretation": (
                        "semantic identity, not surface similarity, drives "
                        "the gain"
                    ),
                },
                {
                    "condition": "C2-C0 cluster CI is above zero",
                    "interpretation": (
                        "surface-similar decoys also help; the gain is "
                        "partly lexical/compositional"
                    ),
                },
                {
                    "condition": "otherwise",
                    "interpretation": "describe the contrasts without a directional claim",
                },
            ],
            "seed_interval": "paired seed-level t interval",
            "stratification": "decoy edge-overlap tier",
        },
        "arms": {
            "C0": (
                "No target-pattern exposure: 48 relation-matched filler "
                "rows per pattern."
            ),
            "C1": (
                "Semantic exposure: eight complete source groups (48 rows) "
                "per target pattern."
            ),
            "C2": (
                "Surface-similar decoys: relation matched, semantically "
                "different, and sharing at least two edge types when "
                "available; deterministic one-edge fallback and filler "
                "rules are retained."
            ),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "evaluation": "V4 block 0, 40 patterns, 8,000 rows",
            "main_analysis_minimum_decoys": 24,
            "slots_per_pattern": 48,
            "train_rows": 24_640,
        },
        "exploratory_status": "exploratory (not preregistered)",
        "input_artifacts": protocol_artifacts,
        "input_sources": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                CANDIDATES_PATH,
                COMMON_BASE_PATH,
                EVALUATION_PATH,
                SELECTED_PAIRS_PATH,
                VFOUR_TARGETS_PATH,
                VTHREE_TARGETS_PATH,
            )
        },
        "implementation": {
            str(DATA_MODULE.relative_to(ROOT)): sha256_file(DATA_MODULE),
            str(TRAIN_MODULE.relative_to(ROOT)): sha256_file(TRAIN_MODULE),
        },
        "namespace": NAMESPACE,
        "primary_metric": "exact-match accuracy",
        "purpose": (
            "Distinguish exposure to the same semantic pattern from "
            "surface-similar exposure with a different semantic identity."
        ),
        "schema_version": "tmlr-strengthening-t6-protocol-v1",
        "seeds": SEEDS,
        "training": {
            "batch_size": TRAIN_BATCH_SIZE,
            "betas": list(BETAS),
            "epochs": EPOCHS,
            "epsilon": EPSILON,
            "evaluation_batch_size": EVAL_BATCH_SIZE,
            "fixed_length": FIXED_LENGTH,
            "gradient_clip": GRADIENT_CLIP,
            "learning_rate": LEARNING_RATE,
            "model": "ByteTransformerClassifier(23)",
            "optimizer": "AdamW",
            "precision": "float32",
            "weight_decay": WEIGHT_DECAY,
        },
    }
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=False)
    write_json(PROTOCOL_PATH, protocol)
    write_sha256sums(PROTOCOL_DIR)
    print(
        json.dumps(
            {
                "protocol": str(PROTOCOL_PATH.relative_to(ROOT)),
                "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "tier_counts": {
                    str(tier): sum(
                        int(row["tier"]) == tier
                        for row in allocation_rows
                    )
                    for tier in (1, 2)
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
