#!/usr/bin/env python3
"""Standalone Damorne research entry; uses the isolated research Python environment."""

import argparse
import importlib.util
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    dedicated = Path.home() / ".cache/leader-lab/venv/bin/python"
    if sys.version_info < (3, 11) or importlib.util.find_spec("pandas") is None:
        if dedicated.exists() and Path(sys.executable) != dedicated:
            os.execv(str(dedicated), [str(dedicated), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise SystemExit("Install research_tools/leader_lab/requirements.txt in Python3.11+")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "study",
            "audit",
            "paper",
            "resting-target",
            "mechanisms",
            "payoff",
            "risk-ablation",
            "derivatives",
            "funding-budget",
            "external-contracts",
            "carry-cost",
            "carry-capture",
            "carry-account",
        ],
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = {
        "study": "damorne_research",
        "audit": "damorne_order_audit",
        "paper": "damorne_paper",
        "resting-target": "damorne_limit",
        "mechanisms": "damorne_mechanisms",
        "payoff": "damorne_payoff",
        "risk-ablation": "damorne_risk_ablation",
        "derivatives": "damorne_derivatives",
        "funding-budget": "damorne_funding_budget",
        "external-contracts": "external_contract_screen",
        "carry-cost": "cross_venue_carry_screen",
        "carry-capture": "cross_venue_capture",
        "carry-account": "cross_venue_carry_account",
    }[args.stage]
    sys.argv = [f"leader_lab.{module}", *args.arguments]
    runpy.run_module(f"research_tools.leader_lab.{module}", run_name="__main__")
