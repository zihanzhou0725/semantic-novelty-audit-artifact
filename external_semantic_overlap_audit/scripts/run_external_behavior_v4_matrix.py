#!/usr/bin/env python3
"""Execute the 32-run frozen directional V4 matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v4.train_driver import run_matrix


if __name__ == "__main__":
    print(json.dumps(run_matrix(), sort_keys=True))
