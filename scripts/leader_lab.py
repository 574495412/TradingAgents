#!/usr/bin/env python3
"""Standalone entry: no TradingAgents application bootstrap or account credentials."""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    # Reuse this tool's dedicated environment for old macOS Python or requested ML.
    dedicated = Path.home() / ".cache/leader-lab/venv/bin/python"
    if sys.version_info < (3, 11) or (
        "learn" in sys.argv[1:] and importlib.util.find_spec("sklearn") is None
    ):
        if dedicated.exists() and Path(sys.executable) != dedicated:
            os.execv(str(dedicated), [str(dedicated), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise SystemExit(
            "Use Python3.11+ and install research_tools/leader_lab/requirements.txt in a research environment"
        )
    from research_tools.leader_lab.__main__ import main

    raise SystemExit(main())
