#!/usr/bin/env python3
"""Revalidate released Mathematics collision-census artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.external_behavior_v2.collision_common import ANALYSIS_DIR
from toolkit.external_behavior_v2.collision_run import validate_release_bundle
from toolkit.external_behavior_v2.collision_validate import validate_census


if __name__ == "__main__":
    report = validate_census(ANALYSIS_DIR, write_artifacts=False)
    bundle = validate_release_bundle()
    output = {
        "bundle": bundle,
        "independent_revalidation": report,
        "status": (
            "pass"
            if report["status"] == "pass" and bundle["status"] == "pass"
            else "fail"
        ),
    }
    print(json.dumps(output, sort_keys=True))
    raise SystemExit(0 if output["status"] == "pass" else 1)
