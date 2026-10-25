"""New-sizing followups cannot be replaced by old controls or future winners."""

import json
from pathlib import Path

import pytest

from research_tools.regime_program.run import select


def test_new_profile_budget_and_two_cost_economic_gates_remain_distinct():
    plan = json.loads(
        (
            Path(__file__).resolve().parents[2] / "research_tools/regime_program/sizing_plan.json"
        ).read_text()
    )
    performance = {
        "trend": 0.5,
        "breakout": 0.4,
        "mean_reversion": 0.3,
        "trend_risk16": 0.25,
        "trend_conviction": 0.22,
        "trend_risk16_buffer10": 0.15,
        "breakout_risk16": 0.1,
        "reversion_risk16": 0.05,
        "equal_mix": 0,
        "cash": 0,
    }
    rows = []
    for candidate in plan["candidates"]:
        for cost in (3, 10):
            value = performance[candidate]
            rows.append(
                {
                    "candidate": candidate,
                    "slippage_bps": cost,
                    "risk_halted": False,
                    "metrics": {
                        "net_return": value,
                        "cagr": value,
                        "max_drawdown": 0.1,
                        "sharpe": 1.2,
                        "closed_trades": 120,
                        "profit_factor": 1.3,
                        "closed_net_pnl": 1,
                    },
                }
            )
    selection = select(rows, plan)
    assert selection["best_static"] == "trend"
    assert selection["research_followups"] == [
        "trend_risk16",
        "trend_conviction",
        "trend_risk16_buffer10",
    ]
    assert "trend_risk16_buffer10" not in selection["admitted_candidates"]
    # A later result cannot be silently included in a two-cost screen.
    with pytest.raises(ValueError, match="Incomplete"):
        select(rows + [rows[0]], plan)
