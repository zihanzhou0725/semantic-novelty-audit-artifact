#!/usr/bin/env python3
"""Freeze CLUTRR V3 execution implementation and manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v3.training_freeze import freeze_execution


if __name__ == "__main__":
    print(json.dumps(freeze_execution(), sort_keys=True))
