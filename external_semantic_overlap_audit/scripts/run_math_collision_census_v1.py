#!/usr/bin/env python3
"""Run, validate, and release Mathematics collision aggregates."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v2.collision_run import run_and_release


if __name__ == "__main__":
    print(json.dumps(run_and_release(), sort_keys=True))
