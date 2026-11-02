#!/usr/bin/env python3
"""Validate the V3 variance decomposition and V4 design gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v3.variance_design import (
    validate_variance_design,
)


if __name__ == "__main__":
    report = validate_variance_design()
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
