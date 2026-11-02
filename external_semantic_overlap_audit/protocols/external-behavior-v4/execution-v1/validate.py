#!/usr/bin/env python3
"""Validate the frozen directional V4 execution protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v4.training_freeze import validate_protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("pre-freeze", "post-freeze"),
        default="post-freeze",
    )
    args = parser.parse_args()
    report = validate_protocol(
        require_execution_absent=args.stage == "pre-freeze"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
