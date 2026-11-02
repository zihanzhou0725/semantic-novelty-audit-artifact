#!/usr/bin/env python3
"""Freeze and validate the directional V4 execution bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v4.training_freeze import freeze_all


if __name__ == "__main__":
    print(json.dumps(freeze_all(), indent=2, sort_keys=True))
