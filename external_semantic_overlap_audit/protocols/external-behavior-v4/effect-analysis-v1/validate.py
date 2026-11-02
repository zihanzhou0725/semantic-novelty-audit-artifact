#!/usr/bin/env python3
"""Validate the directional V4 effect-analysis protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v4.effect_analysis import validate_protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("pre-unblinding", "post-unblinding"),
        default="post-unblinding",
    )
    args = parser.parse_args()
    report = validate_protocol(
        require_outcomes_absent=args.stage == "pre-unblinding"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
