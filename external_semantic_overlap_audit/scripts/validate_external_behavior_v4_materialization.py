#!/usr/bin/env python3
"""Validate directional V4 materialized data and run manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v4.materialize import (
    validate_materialization,
)


if __name__ == "__main__":
    report = validate_materialization()
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
