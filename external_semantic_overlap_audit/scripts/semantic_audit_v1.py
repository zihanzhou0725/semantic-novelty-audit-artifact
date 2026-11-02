#!/usr/bin/env python3
"""Versioned entrypoint for the portable semantic-audit toolkit."""

from __future__ import annotations

import sys
from pathlib import Path


TOOLKIT_ROOT = Path(__file__).resolve().parents[1] / "toolkit"
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from semantic_audit.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
