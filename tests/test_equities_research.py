"""Independent economic examples and temporal perturbations for stock research."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.equities.__main__ import choose, freeze
from tradingagents.research.equities.data import (
    LongbridgeQuotes,
    daily_from_hourly,
    sessions,
    validate,
)
from tradingagents.research.equities.engine import account, signals


def sample():
    days = pd.date_range("2024-01-01", periods=5)
    calendar = pd.DataFrame(
        {
            "open": days.tz_localize("UTC") + pd.Timedelta(hours=9),
            "close": days.tz_localize("UTC") + pd.Timedelta(hours=16),
        },
        index=days,
    )
    frame = pd.DataFrame(
        {
            "open": [10, 20, 25, 30, 40],
            "high": [12, 22, 27, 33, 42],
            "low": [8, 18, 23, 28, 38],
            "close": [11, 21, 26, 31, 41],
            "volume": 10000,
            "dividend": 0.0,
            "split": 0.0,
        },
        index=days,
    )
    spec = {
        "initial_cash": 1000,
        "minimum_fee": 1,
        "dividend_haircut": 0,
        "symbols": [{"symbol": "S.US", "lot_size": 10}],
    }
    decision = pd.DataFrame(
        {"enter": [True, False, False, False, False], "exit": [False, False, True, False, False]},
        index=days,
    )
    return frame, calendar, spec, decision


def run(frame, calendar, spec, decision, **kwargs):
    return account(
        {"S.US": frame},
        calendar,
        {"S.US": decision},
        spec,
        start="2024-01-01",
        end="2024-01-06",
        cost_bps=0,
        **kwargs,
    )


def test_next_open_lots_minimum_cost_and_unrealized_identity():
    frame, calendar, spec, decision = sample()
    result = run(frame, calendar, spec, decision)
    trade = result["trades"][0]
    # 40 shares at next day's 20, sell two sessions later at 30; two $1 fees.
    assert trade["quantity"] == 40
    assert trade["entry_price"] == 20
    assert trade["exit_price"] == 30
    assert trade["net_pnl"] == 398
    assert result["metrics"]["ending_equity"] == 1398
    assert trade["entry_ms"] == int(calendar.iloc[1].open.timestamp() * 1000)


def test_dividend_entitlement_precedes_ex_date_open_sale():
    frame, calendar, spec, decision = sample()
    frame.iloc[3, frame.columns.get_loc("dividend")] = 2
    spec["dividend_haircut"] = 0.30
    result = run(frame, calendar, spec, decision)
    assert result["trades"][0]["dividends"] == pytest.approx(56)
    assert result["metrics"]["cash"] == 1398
    assert result["metrics"]["dividend_receivables"] == pytest.approx(56)
    assert result["metrics"]["ending_equity"] == pytest.approx(1454)


def test_ex_date_buyer_gets_no_dividend_and_no_synthetic_terminal_fill():
    frame, calendar, spec, decision = sample()
    frame.iloc[1, frame.columns.get_loc("dividend")] = 2
    decision["exit"] = False
    result = run(frame, calendar, spec, decision)
    assert result["trades"] == []
    assert result["metrics"]["win_rate"] is None
    assert result["metrics"]["dividend_receivables"] == 0
    assert result["metrics"]["ending_equity"] == 1839  # $199 cash + 40 * $41
    assert result["metrics"]["unrealized_pnl"] == 839


def test_daily_low_risk_retains_floating_loss():
    frame, calendar, spec, decision = sample()
    frame.iloc[2, frame.columns.get_loc("low")] = 1
    result = run(frame, calendar, spec, decision)
    assert result["metrics"]["max_drawdown"] > 0.7
    assert result["metrics"]["close_max_drawdown"] < 0.1


def test_unknown_calendar_mark_is_rejected():
    frame, calendar, spec, decision = sample()
    with pytest.raises(ValueError, match="missing account marks"):
        run(frame.iloc[:-1], calendar, spec, decision)


def test_quality_checks_reject_missing_duplicate_split_and_nan():
    frame, calendar, _, _ = sample()
    validate(frame, calendar)
    for bad in (frame.iloc[:-1], pd.concat([frame, frame.iloc[-1:]])):
        with pytest.raises(ValueError):
            validate(bad, calendar)
    frame.iloc[1, frame.columns.get_loc("split")] = 2
    with pytest.raises(ValueError, match="split detected"):
        validate(frame, calendar)
    frame.iloc[1, frame.columns.get_loc("split")] = np.nan
    with pytest.raises(ValueError, match="unknown"):
        validate(frame, calendar)


def test_calendar_holidays_half_day_and_dst():
    us = sessions("US", "2024-03-08", "2024-03-12")
    assert us.iloc[0].open.hour == 14
    assert us.iloc[1].open.hour == 13
    us_half = sessions("US", "2024-11-28", "2024-11-30")
    assert len(us_half) == 1
    assert us_half.iloc[0].close.hour == 18
    hk = sessions("HK", "2024-02-09", "2024-02-15")
    assert pd.Timestamp("2024-02-12") not in hk.index
    assert hk.loc["2024-02-09", "close"].hour == 4
    weather = sessions("HK", "2023-09-01", "2023-10-10")
    assert pd.Timestamp("2023-09-01") not in weather.index
    assert pd.Timestamp("2023-09-08") not in weather.index
    assert weather.loc["2023-10-09", "open"].hour == 6


def test_hourly_aggregation_checks_exact_grid_with_mixed_timestamp_units():
    calendar = sessions("HK", "2025-10-24", "2025-10-25")
    index = pd.date_range("2025-10-24 01:30", periods=7, freq="1h", tz="UTC").as_unit("us")
    frame = pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 14, 15, 16],
            "high": 20,
            "low": 9,
            "close": 17,
            "volume": 100,
            "dividend": 0,
            "split": 0,
        },
        index=index,
    )
    actual = daily_from_hourly(frame, calendar, "2025-10-24")
    assert actual == {
        "open": 10,
        "high": 20,
        "low": 9,
        "close": 17,
        "volume": 700,
        "dividend": 0,
        "split": 0,
    }
    with pytest.raises(ValueError, match="complete vendor session grid"):
        daily_from_hourly(frame.iloc[:-1], calendar, "2025-10-24")


@pytest.mark.parametrize("candidate", ["trend_20_60", "rsi2_recovery", "buy_hold", "cash"])
def test_signals_do_not_change_when_future_prices_change(candidate):
    rng = np.random.default_rng(42)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 320))))
    frame = pd.DataFrame({"close": close})
    original = signals(frame, candidate)
    altered = frame.copy()
    altered.iloc[250:] *= 50
    pd.testing.assert_frame_equal(original.iloc[:250], signals(altered, candidate).iloc[:250])
    pd.testing.assert_frame_equal(original.iloc[:250], signals(frame.iloc[:250], candidate))


def test_selection_stays_cash_on_weak_or_unknown_training_metrics():
    passed = {"net_return": 0.2, "closed_trades": 30, "profit_factor": 1.5, "max_drawdown": 0.1}
    assert choose({"a": [passed, passed]}) == "a"
    for key, value in [
        ("closed_trades", 1),
        ("profit_factor", None),
        ("max_drawdown", 0.3),
        ("net_return", -0.01),
    ]:
        assert choose({"a": [passed, {**passed, key: value}]}) == "cash"


def test_broker_chunks_no_adjust_regular_sessions_and_no_fake_actions():
    queries = []
    adapter = LongbridgeQuotes.__new__(LongbridgeQuotes)
    adapter.api = SimpleNamespace(
        Period=SimpleNamespace(Day="day"),
        AdjustType=SimpleNamespace(NoAdjust="raw"),
        TradeSessions=SimpleNamespace(Intraday="regular"),
    )

    def history(*args):
        queries.append(args)
        return [
            SimpleNamespace(
                timestamp=pd.Timestamp(args[3], tz="Asia/Hong_Kong"),
                open=1,
                high=2,
                low=1,
                close=2,
                volume=100,
            )
        ]

    adapter.context = SimpleNamespace(history_candlesticks_by_date=history)
    result = adapter.history("2800.HK", "2024-01-01", "2025-02-01")
    assert len(queries) == 3
    assert all((q[4] - q[3]).days < 180 and q[2] == "raw" and q[5] == "regular" for q in queries)
    assert queries[-1][4].isoformat() == "2025-01-31"
    assert "dividend" not in result


def test_freeze_preserves_bytes_and_cannot_overwrite(tmp_path):
    factor = tmp_path / "factor.py"
    factor.write_text("def calculate_factor(frame): return frame.close\n")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"version": 1}))
    output = freeze(plan, factor, tmp_path / "run")
    assert (output / "source/legacy_ema_daily.py").read_bytes() == factor.read_bytes()
    assert json.loads((output / "source-identities.json").read_text())["legacy_ema_daily.py"]
    with pytest.raises(FileExistsError):
        freeze(plan, factor, output)
