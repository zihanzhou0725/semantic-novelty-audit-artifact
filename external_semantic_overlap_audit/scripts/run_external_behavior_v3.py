#!/usr/bin/env python3
"""Run one frozen CLUTRR V3 arm."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v3.train_run import main


if __name__ == "__main__":
    raise SystemExit(main())
