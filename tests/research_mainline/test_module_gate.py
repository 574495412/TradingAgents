"""Regressions for the user's module/portfolio objective distinction."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from research_tools.regime_program.module_gate import assess_module


def records():
    protocol = json.loads(Path("docs/research/regime_program/targets.json").read_text())[
        "module_acceptance"
    ]
    cases = [
        {
            "slippage_bps": cost,
            "capital": 10000,
            "start": "2025-01-01",
            "end": "2026-01-01",
            "risk_halted": False,
            "audit_passed": True,
            "metrics": {
                "cagr": 0.02,
                "sharpe": 0.4,
                "net_return": 0.02,
                "closed_net_pnl": 200.0,
                "closed_trades": 40,
                "max_gross_exposure": 0.8,
            },
        }
        for cost in (3, 10)
    ]
    return cases, protocol


def verdict(cases, protocol, **kwargs):
    return assess_module(
        cases,
        protocol,
        **{
            "domain_registered": True,
            "sleeve_capacity_passed": True,
            "cluster_groups": 4,
            **kwargs,
        },
    )


def test_positive_specialist_need_not_reach_20percent_or_sharpe_one():
    cases, protocol = records()
    v = verdict(cases, protocol)
    assert v["eligible_for_portfolio_test"] and not v["production_eligible"]
    assert v["status"] == "research_qualified_for_portfolio_test"


@pytest.mark.parametrize("field,value", [("net_return", -0.01), ("closed_net_pnl", -10)])
def test_floating_losses_or_closed_losses_cannot_be_hidden(field, value):
    cases, protocol = records()
    cases[1]["metrics"][field] = value
    assert verdict(cases, protocol)["status"] == "cost_sensitive_current_domain"


def test_failed_hedges_do_not_inflate_specialist_evidence():
    cases, protocol = records()
    for r in cases:
        r["metrics"].update(closed_trades=100, matched_closed_pairs=5)
    assert verdict(cases, protocol)["status"] == "evidence_insufficient"
    cases[0]["metrics"]["matched_closed_pairs"] = 100
    cases[1]["metrics"]["matched_closed_pairs"] = 100
    assert verdict(cases, protocol, cluster_groups=1)["status"] == "evidence_insufficient"


def test_posthoc_domains_unmatched_accounts_and_missing_audit_do_not_qualify():
    cases, protocol = records()
    assert verdict(cases, protocol, domain_registered=False)["status"] == "invalid_evidence"
    changed = deepcopy(cases)
    changed[1]["capital"] = 100000
    assert verdict(changed, protocol)["status"] == "invalid_evidence"
    changed = deepcopy(cases)
    changed[0]["audit_passed"] = False
    assert verdict(changed, protocol)["status"] == "invalid_evidence"


def test_risk_and_allocated_sleeve_capacity_are_still_required():
    cases, protocol = records()
    assert (
        verdict(cases, protocol, sleeve_capacity_passed=None)["status"]
        == "allocated_sleeve_capacity_unverified"
    )
    cases[0]["risk_halted"] = True
    assert verdict(cases, protocol)["status"] == "risk_failed_current_domain"


def test_portfolio_annual_hurdle_cannot_silently_reappear_in_module_gate():
    cases, protocol = records()
    protocol["minimum_cagr"] = 0.20
    with pytest.raises(ValueError, match="Portfolio annual hurdle"):
        verdict(cases, protocol)


def test_empty_opportunity_set_or_unknown_metrics_are_not_economic_rejections():
    cases, protocol = records()
    for row in cases:
        row["metrics"].update(closed_trades=0, closed_net_pnl=0, net_return=0)
    assert verdict(cases, protocol)["status"] == "evidence_insufficient"
    cases, protocol = records()
    cases[0]["metrics"]["net_return"] = float("nan")
    assert verdict(cases, protocol)["status"] == "evidence_insufficient"
