#!/usr/bin/env python3
"""Freeze the post-unblinding V3 variance and V4 design protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v3.variance_design import freeze_protocol


if __name__ == "__main__":
    print(json.dumps(freeze_protocol(), indent=2, sort_keys=True))
