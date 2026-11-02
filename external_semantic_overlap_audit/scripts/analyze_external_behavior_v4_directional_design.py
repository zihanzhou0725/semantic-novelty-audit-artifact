#!/usr/bin/env python3
"""Run directional V4 power simulation and block selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v4.directional_design import run_design


if __name__ == "__main__":
    print(json.dumps(run_design(), indent=2, sort_keys=True))
