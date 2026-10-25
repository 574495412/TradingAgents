import numpy as np
import pandas as pd
import pytest

from research_tools.leader_lab.cross_venue_carry_screen import known_binance_funding, known_window
from research_tools.leader_lab.external_contract_screen import (
    at_decision_clock,
    completed_bars,
    economic_gate,
    rule_features,
)


def market(n):
    x = np.arange(n)
    close = 100 + 3 * np.sin(x / 40) + x * 0.0005
    return pd.DataFrame(
        {"open": close, "close": close, "high": close + 0.1, "low": close - 0.1}, index=x * 60000
    )


def test_partial_higher_timeframe_is_unavailable_until_fully_closed():
    frame = market(14)
    bars = completed_bars(frame, 5)
    assert len(bars) == 2
    mapped = at_decision_clock(bars.close, frame.index, 5)
    assert mapped.iloc[:5].isna().all()
    assert mapped.iloc[5] == frame.close.iloc[4]
    assert mapped.iloc[10] == frame.close.iloc[9]


def test_future_partial_bar_cannot_rewrite_any_existing_signal():
    pytest.importorskip("pandas_ta")
    frame = market(6000)
    changed = frame.copy()
    cut = 60 * 64 + 17
    changed.loc[cut * 60000 :, ["close", "high", "low"]] = [900.0, 1000.0, 800.0]
    a, b = rule_features(frame), rule_features(changed)
    assert (a.to_numpy() != 0).any()
    pd.testing.assert_frame_equal(a.iloc[: cut + 1], b.iloc[: cut + 1])
    prefix = rule_features(frame.iloc[:cut])
    pd.testing.assert_frame_equal(a.iloc[:cut], prefix)


def test_positive_open_pnl_cannot_hide_losing_closed_sleeve():
    score = {
        "sleeves": {
            "small": {"net_pnl": 5, "terminal_open_net_pnl": 6, "complete_positions": 30},
            "large": {"net_pnl": 10, "terminal_open_net_pnl": 0, "complete_positions": 15},
        },
        "profit_factor": 1.2,
        "minute_observed_mark_open_drawdown": 0.01,
    }
    assert economic_gate(score, 30, 15) is False
    score["sleeves"]["small"]["terminal_open_net_pnl"] = 0
    assert economic_gate(score, 30, 15) is True


def test_original_late_settlement_is_not_known_at_rounded_hour():
    hour = 3600000
    rows = [{"time": hour - 300001}, {"time": hour - 300000}, {"time": hour + 139}]
    known = known_window(rows, hour, "time")
    assert known == rows[:2]
    assert known_window([rows[-1]], hour + 300000, "time") == []
    assert known_window([rows[-1]], hour + 300139, "time") == [rows[-1]]


@pytest.mark.parametrize("cadence", [4, 8])
def test_binance_cadence_is_known_from_past_and_missing_settlement_is_rejected(cadence):
    hour = 3600000
    rows = [{"fundingTime": i * hour + 17} for i in range(0, 96, cadence)]
    known = known_binance_funding(rows, 73 * hour)
    assert len(known) == 24 // cadence
    missing = [x for x in rows if x != known[-1]]
    assert known_binance_funding(missing, 73 * hour) is None
    future = [*rows, {"fundingTime": 100 * hour}]
    assert known_binance_funding(future, 73 * hour) == known
