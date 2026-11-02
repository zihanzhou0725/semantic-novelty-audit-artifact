#!/usr/bin/env python3
"""Materialize external-behavior-v3 design artifacts without training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v3.materialize import materialize


if __name__ == "__main__":
    print(json.dumps(materialize(), sort_keys=True))
