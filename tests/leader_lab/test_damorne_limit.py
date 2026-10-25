import json

import numpy as np
import pandas as pd

from research_tools.leader_lab.damorne_limit import account
from research_tools.leader_lab.damorne_research import make_candidates


def inputs():
    c = make_candidates(
        {"train_small": {"gross_price_return": {"0.5": 0.009}}}, np.zeros((20, 8)), np.zeros(20)
    )[0]
    c.update(small_stop=0.02, max_additions=0)
    f = pd.DataFrame(
        {
            "open": 100.0,
            "mark": 100.0,
            "high": 100.0,
            "low": 100.0,
            "funding": 0.0,
            "return5": 0.002,
            "return15": 0.005,
            "return60": 0.02,
            "range_location": 0.8,
            "atr": 0.001,
            "volume_ratio": 1.0,
            "flow": 0.2,
            "quote60": 2_000_000.0,
        },
        index=np.arange(10) * 60000,
    )
    meta = {
        "XUSDT": {
            "step": 0.01,
            "min_qty": 0.01,
            "max_qty": 10000,
            "min_notional": 5,
            "price_tick": 0.01,
        }
    }
    return c, f, meta


def test_target_activation_and_conservative_exit_clock():
    c, f, meta = inputs()
    f.loc[:, "high"] = 102
    summary, state = account(c, {"XUSDT": f}, meta, 0, 4 * 60000)
    assert summary["complete_positions"] == 1
    assert state["closed"][0]["opened_ms"] == 120000
    assert state["closed"][0]["closed_ms"] == 239999
    assert state["closed"][0]["net_pnl"] > 0


def test_same_bar_both_targets_uses_stop_first():
    c, f, meta = inputs()
    f.loc[180000, ["high", "low"]] = [103, 95]
    summary, state = account(c, {"XUSDT": f}, meta, 0, 4 * 60000)
    assert summary["complete_positions"] == 1
    assert state["closed"][0]["net_pnl"] < 0


def test_intrabar_future_cannot_change_same_minute_decision():
    c, f, meta = inputs()
    altered = f.copy()
    altered.loc[180000, "high"] = 103
    _, base = account(c, {"XUSDT": f}, meta, 0, 5 * 60000)
    _, changed = account(c, {"XUSDT": altered}, meta, 0, 5 * 60000)
    assert [x for x in base["decisions"] if x["decision_ms"] <= 180000] == [
        x for x in changed["decisions"] if x["decision_ms"] <= 180000
    ]
    _, prefix = account(c, {"XUSDT": altered}, meta, 0, 10 * 60000, break_ms=5 * 60000)
    resumed, tail = account(
        c, {"XUSDT": altered}, meta, 5 * 60000, 10 * 60000, state=json.loads(json.dumps(prefix))
    )
    whole, final = account(c, {"XUSDT": altered}, meta, 0, 10 * 60000)
    assert whole == resumed and tail == final
