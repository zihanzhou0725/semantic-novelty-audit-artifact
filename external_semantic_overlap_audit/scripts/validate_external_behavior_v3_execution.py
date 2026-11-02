#!/usr/bin/env python3
"""Validate all frozen CLUTRR V3 training runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v3.train_integrity import validate_execution


if __name__ == "__main__":
    report = validate_execution()
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)
