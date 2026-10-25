"""检验H04时钟、两腿实际库存与完整统计家族，合成收益不作盈利证据。"""

import math

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.factors.hedged_carry import (
    DAY,
    DELAY,
    HOUR,
    MINUTE,
    PERP_FEE,
    SPOT_FEE,
    cashflows,
    feature,
    samples,
)
from tradingagents.research.quant.hedged_carry_study import evaluate, fit_ols


def event(time, rate="0.01", kind="Regular"):
    """固定基础单位与原毫秒事件。"""
    return {"symbol": "BTCUSDT", "time_ms": time, "rate": rate, "mark": "100", "type": kind}


def test_feature_delay_left_open_and_special_exclusion():
    t = 2 * DAY
    left, right = t - DAY - DELAY, t - DELAY
    data = [
        event(left),
        event(left + 1, "0.02"),
        event(right, "-0.03"),
        event(right + 1, "20"),
        event(right, "40", "Special"),
    ]
    assert feature(data, t, 0, 4 * DAY)[0] == pytest.approx(0.01)
    assert math.isnan(feature(data, DAY, 0, 4 * DAY)[0])
    assert feature([], t, 0, 4 * DAY)[0] == 0


@pytest.mark.parametrize("slip", [3, 10, 11])
def test_cash_matches_direct_received_asset_ledger(slip):
    q, a, b = 1.7, 1000, 2000
    ev = [event(a, "70"), event(a + 1, "0.02"), event(b, "-0.005", "Special"), event(b + 1, "90")]
    result = cashflows(q, 100, 110, 101, 108, ev, a, b, slip)
    s = slip / 10000
    buy, sell, short, cover = 100 * (1 + s), 110 * (1 - s), 101 * (1 - s), 108 * (1 + s)
    direct = (
        q * sell * (1 - SPOT_FEE)
        - q / (1 - SPOT_FEE) * buy
        + q * (short - cover)
        - q * PERP_FEE * (short + cover)
        + q * 100 * (0.02 - 0.005)
    )
    assert result["net_pnl"] == pytest.approx(direct, abs=1e-12)
    assert result["spot_gross_quantity"] * (1 - SPOT_FEE) == pytest.approx(q)
    assert result["funding_count"] == 2
    assert result["special_count"] == 1
    assert result["last_funding_ms"] == b


def minute_frame(end):
    """完整分钟来源，缺失必须由测试显式制造。"""
    return pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "raw_open_ms": range(0, end, MINUTE),
            "spot_status": "KNOWN_RAW_MINUTE",
            "perp_status": "KNOWN_RAW_MINUTE",
            "spot_open": "100",
            "spot_close": "100",
            "perp_open": "101",
            "perp_close": "101",
        }
    )


def test_all_rows_mature_boundary_and_unknown_label_retained():
    end, cutoff = 4 * DAY, 2 * DAY
    minutes = minute_frame(end)
    clock = DAY + HOUR
    minutes.loc[minutes.raw_open_ms.eq(clock + MINUTE + DAY), "spot_status"] = "UNKNOWN"
    frame = samples(minutes, [event(DAY)], start=0, cutoff=cutoff, end=end)
    assert len(frame) == 4 * 24 * 3
    assert frame.duplicated(["symbol", "clock_ms", "slippage_bps"]).sum() == 0
    missing = frame[frame.clock_ms.eq(clock)]
    assert missing.x.notna().all()
    assert missing.y.isna().all()
    assert missing.label_status.eq("PRICE_OR_QUANTITY_UNKNOWN").all()
    assert frame[frame.clock_ms.eq(DAY)].phase.eq("training_embargo").all()
    assert frame[frame.clock_ms.eq(3 * DAY)].phase.eq("validation_immature").all()
    assert frame[frame.clock_ms.eq(0)].feature_status.eq("FEATURE_WINDOW_OUTSIDE_SOURCE").all()


def test_future_funding_only_changes_label_not_current_feature():
    end, t = 4 * DAY, 2 * DAY
    data = minute_frame(end)
    a = samples(data, [event(DAY), event(t + HOUR, "0.001")], start=0, cutoff=2 * DAY, end=end)
    b = samples(data, [event(DAY), event(t + HOUR, "0.9")], start=0, cutoff=2 * DAY, end=end)
    current = a.clock_ms.eq(t)
    assert a.loc[current, "x"].equals(b.loc[current, "x"])
    assert (a.loc[current, "funding_cash"] < b.loc[current, "funding_cash"]).all()


def test_ols_matches_lstsq_and_calendar_score_sandwich():
    rng = np.random.default_rng(21)
    days = pd.date_range("2026-08-01", periods=12).strftime("%Y-%m-%d").tolist()
    frame = pd.DataFrame({"x": rng.normal(size=240) * 0.0001, "day": np.repeat(days, 20)})
    frame["y"] = 0.0002 - 0.6 * frame.x + rng.normal(size=240) * 0.00001
    frame = frame[frame.day.ne(days[4])]
    fit = fit_ols(frame, days)
    x = np.column_stack([np.ones(len(frame)), frame.x])
    assert [fit["alpha"], fit["beta"]] == pytest.approx(np.linalg.lstsq(x, frame.y, rcond=None)[0])
    assert fit["daily_scores"][days[4]] == 0
    assert fit["valid_training_days"] == 11
    score = np.array(list(fit["daily_scores"].values()))
    kernel = np.fromfunction(lambda i, j: np.maximum(0, 1 - np.abs(i - j) / 4), (12, 12))
    expected = math.sqrt(score @ kernel @ score) / np.sum((frame.x - frame.x.mean()) ** 2)
    assert fit["beta_standard_error"] == pytest.approx(expected)
    assert fit_ols(frame.assign(x=1), days) is None


def test_full_family_keeps_unknown_cases_and_ties_without_label_selection():
    start = int(pd.Timestamp("2026-08-01", tz="UTC").timestamp() * 1000)
    cutoff, end = start + 10 * DAY, start + 20 * DAY
    rows = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        for slip in (3, 10, 11):
            for day in range(20):
                t = start + day * DAY
                rows.append(
                    {
                        "symbol": symbol,
                        "slippage_bps": slip,
                        "clock_ms": t,
                        "day": pd.Timestamp(t, unit="ms", tz="UTC").date().isoformat(),
                        "x": 1.0,
                        "y": float(day),
                        "phase": "training" if day < 10 else "validation",
                        "feature_status": "KNOWN",
                        "label_status": "KNOWN",
                    }
                )
    result, frame = evaluate(pd.DataFrame(rows), start=start, cutoff=cutoff, end=end)
    assert len(result["cases"]) == 6
    assert len(result["primary_tests"]) == 18
    assert all(t["p"] is None and t["q"] == 1 and not t["passed"] for t in result["primary_tests"])
    assert frame.prediction.isna().all()
    assert not any(c["predictive_screen_passed"] for c in result["cases"].values())
