#!/usr/bin/env python3
"""Freeze external-behavior-v3 implementation and environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v3.freeze import freeze_implementation


if __name__ == "__main__":
    print(json.dumps(freeze_implementation(), sort_keys=True))
