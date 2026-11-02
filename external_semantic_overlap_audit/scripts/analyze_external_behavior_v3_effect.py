#!/usr/bin/env python3
"""Run the frozen CLUTRR V3 exposure-effect analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v3.effect_analysis import run_analysis


if __name__ == "__main__":
    print(json.dumps(run_analysis(), indent=2, sort_keys=True))
