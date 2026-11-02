#!/usr/bin/env python3
"""Validate the frozen T6 protocol and execute one independent run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.tmlr_strengthening.artifacts import sha256_file  # noqa: E402
from toolkit.tmlr_strengthening.clutrr_tmlr_train import (  # noqa: E402
    EPOCHS,
    train_and_eval,
)


PROTOCOL_PATH = (
    ROOT
    / "protocols"
    / "tmlr-strengthening-v1"
    / "t6_semantic_vs_surface"
    / "protocol.json"
)
INPUT_DIR = (
    ROOT
    / "manifests"
    / "tmlr-strengthening-v1"
    / "t6_semantic_vs_surface"
    / "execution-inputs"
)
RUN_ROOT = (
    ROOT
    / "runs"
    / "tmlr-strengthening-v1"
    / "t6_semantic_vs_surface"
)


def validate_protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="ascii"))
    for relative, expected in {
        **protocol["implementation"],
        **protocol["input_artifacts"],
        **protocol["input_sources"],
    }.items():
        path = ROOT / relative
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen T6 input changed: {relative}")
    if (
        protocol["namespace"] != "tmlrs:t6"
        or protocol["seeds"] != list(range(4))
        or protocol["training"]["epochs"] != EPOCHS
    ):
        raise RuntimeError("Frozen T6 protocol fields changed.")
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("C0", "C1", "C2"), required=True)
    parser.add_argument("--seed", type=int, choices=range(4), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--timing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = validate_protocol()
    epochs = 1 if args.timing else EPOCHS
    suffix = "_timing" if args.timing else ""
    output_dir = RUN_ROOT / f"arm{args.arm}_seed{args.seed}{suffix}"
    metrics = train_and_eval(
        input_dir=INPUT_DIR,
        arm=args.arm,
        seed=args.seed,
        namespace=str(protocol["namespace"]),
        output_dir=output_dir,
        epochs=epochs,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

