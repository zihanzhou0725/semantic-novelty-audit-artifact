#!/usr/bin/env python3
"""Copy regenerated artifacts into this manuscript, independent of bundle layout."""

import importlib.util
import sys
from pathlib import Path

paper = Path(__file__).resolve().parents[1]
audit = Path(sys.argv[1]).resolve()
source = audit / "scripts/tmlr_strengthening/sync_paper_artifacts.py"
spec = importlib.util.spec_from_file_location("strengthening_sync", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.PAPER_DIR = paper
module.main()
