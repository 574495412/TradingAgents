"""Period and state attribution retain one continuous account's economic meaning."""

import numpy as np
import pandas as pd

from research_tools.regime_program.reporting import cluster_uncertainty, partitions, trade_stats


def test_state_contributions_reconcile_without_constructing_a_fake_regime_nav():
    index = pd.date_range("2025-01-01", periods=577, freq="5min", tz="UTC")
    equity = 10000 + np.sin(np.arange(len(index)) / 50) * 100
    curve = pd.DataFrame({"date": index, "equity": equity, "gross_exposure": 4000})
    states = pd.DataFrame(
        {"state": ["trend_up", "range", "trend_down"]},
        index=pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
    )
    events = pd.Series(False, index=pd.date_range("2025-01-01", periods=49, freq="h", tz="UTC"))
    events.iloc[10:20] = True
    trades = [
        {
            "entry_time": "2025-01-01T00:05:00+00:00",
            "exit_time": "2025-01-02T01:00:00+00:00",
            "net_pnl": 25,
            "symbol": "BTCUSDT",
        }
    ]
    result = partitions(curve, trades, states, events, 10000)
    for group in ("regime", "covered_fomc"):
        rows = [x for x in result["state_contributions"] if x["group"] == group]
        assert abs(sum(x["net_pnl_contribution"] for x in rows) - (equity[-1] - equity[0])) < 1e-8
        assert sum(x["lifecycle_by_entry_state"]["complete_lifecycles"] for x in rows) == 1
        assert all("cagr" not in x for x in rows)


def test_lifecycle_win_interval_and_cluster_shortfalls_remain_unknown():
    assert trade_stats([])["win_rate_wilson95"] is None
    stats = trade_stats([{"net_pnl": 1}, {"net_pnl": -2}])
    assert stats["win_rate"] == 0.5 and stats["profit_factor"] == 0.5
    assert stats["win_rate_wilson95"][0] < 0.5 < stats["win_rate_wilson95"][1]
    few = [{"symbol": "BTCUSDT", "entry_time": "2025-01-01", "net_pnl": 2}]
    assert not cluster_uncertainty(few)["by_symbol"]["sufficient"]
