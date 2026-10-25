import json

import numpy as np
import pandas as pd
import pytest

from research_tools.leader_lab.damorne_derivatives import EXTRA_FEATURES, extra_vectors
from research_tools.leader_lab.damorne_payoff import ECONOMICS, feature_tables, payoff_label
from research_tools.leader_lab.damorne_research import ContractBook
from research_tools.leader_lab.damorne_tail_account import _add, _open, candidates, run_account


def fixture(n=100):
    config = candidates(ECONOMICS)[0]
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 99.9,
            "close": 100.0,
            "mark": 100.0,
            "funding": 0.0,
            "quote60": 2_000_000.0,
            "atr": 0.001,
            "volume_ratio": 1.0,
        },
        index=np.arange(n) * 60000,
    )
    signals = np.zeros((n, 2, 4))
    signals[:, 0] = [0.9, 0.002, 0.9, 0.004]
    metadata = {
        "XUSDT": {
            "step": 0.01,
            "min_qty": 0.01,
            "max_qty": 100000.0,
            "min_notional": 5.0,
            "price_tick": 0.01,
        }
    }
    return config, frame, {"XUSDT": signals}, metadata


def test_limit_never_exceeds_cap_and_no_synthetic_campaign_on_no_fill():
    c, f, predictions, meta = fixture(20)
    f["low"] = 100.0
    result, state = run_account(c, {"XUSDT": f}, predictions, meta, 0, 20 * 60000, {"XUSDT": 100})
    assert not state["campaigns"] and not state["book"]["positions"]
    assert result["complete_positions"] == 0 and result["fees"] == 0
    f.loc[120000, "low"] = 99.9
    result, state = run_account(c, {"XUSDT": f}, predictions, meta, 0, 20 * 60000, {"XUSDT": 100})
    p = state["book"]["positions"][0]
    assert p["entry"] == 100
    assert state["campaigns"][0]["started_ms"] == 179999
    assert result["whole_position_cash_reconciled"]


def test_stop_first_and_only_own_net_profit_can_rearm():
    c, f, predictions, meta = fixture(30)
    f.loc[180000, ["high", "low"]] = [103, 98]
    result, state = run_account(c, {"XUSDT": f}, predictions, meta, 0, 30 * 60000, {"XUSDT": 100})
    assert result["complete_positions"] == 1 and state["book"]["closed"][0]["net_pnl"] < 0
    assert result["policy_rejections"]["own_loss_cooldown"] > 0
    assert not any(q["role"] == "own_profit_rearm" for q in state["book"]["decisions"])


def test_profit_rearm_costs_and_exact_restart():
    c, f, predictions, meta = fixture(200)
    f["high"] = 102
    result, state = run_account(c, {"XUSDT": f}, predictions, meta, 0, 200 * 60000, {"XUSDT": 100})
    assert any(q["role"] == "own_profit_rearm" for q in state["book"]["decisions"])
    assert result["trading_fee_component"] + result[
        "modeled_execution_cost_component"
    ] == pytest.approx(result["fees"])
    _, prefix = run_account(
        c, {"XUSDT": f}, predictions, meta, 0, 200 * 60000, {"XUSDT": 100}, break_ms=101 * 60000
    )
    resumed, tail = run_account(
        c,
        {"XUSDT": f},
        predictions,
        meta,
        101 * 60000,
        200 * 60000,
        {"XUSDT": 100},
        state=json.loads(json.dumps(prefix)),
    )
    assert resumed == result and tail == state
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("sign,price", [(1, 101.0), (-1, 99.0)])
def test_addition_preserves_initial_loss_budget_and_tightens_stop(sign, price):
    c, _, _, meta = fixture()
    c["allow_additions"] = True
    book = ContractBook(c, meta)
    _open(
        book,
        {"symbol": "XUSDT", "sign": sign, "price": 100.0, "sleeve": "small"},
        0,
        {"XUSDT": 100.0},
    )
    p = book.positions[("XUSDT", sign)]
    initial_stop, budget = p["stop"], p["risk_budget"]
    assert _add(
        book,
        {"symbol": "XUSDT", "sign": sign, "position_id": p["id"]},
        price,
        600000,
        {"XUSDT": price},
    )
    assert sign * (p["stop"] - initial_stop) >= 0
    net_at_stop = (
        p["gross_realized"]
        + sign * p["qty"] * (p["stop"] - p["entry"])
        + p["funding"]
        - p["fees"]
        - p["qty"] * p["stop"] * c["fee_bps"] / 10000
    )
    assert net_at_stop >= -budget - 1e-8
    assert p["risk_budget"] == budget


def test_future_range_or_current_volume_cannot_change_same_clock_orders():
    c, f, predictions, meta = fixture(12)
    changed = f.copy()
    changed.loc[60000, "quote60"] = 0
    changed.loc[180000, "high"] = 120
    _, a = run_account(c, {"XUSDT": f}, predictions, meta, 0, 4 * 60000, {"XUSDT": 100})
    _, b = run_account(c, {"XUSDT": changed}, predictions, meta, 0, 4 * 60000, {"XUSDT": 100})
    assert a["book"]["decisions"] == b["book"]["decisions"]


def test_shared_wallet_partial_reduction_is_one_whole_position():
    c, f, predictions, meta = fixture(30)
    c["allow_reductions"] = True
    f.loc[180000:, "high"] = 100.6
    f.loc[600000:, "high"] = 102
    result, state = run_account(c, {"XUSDT": f}, predictions, meta, 0, 30 * 60000, {"XUSDT": 100})
    assert result["event_counts"]["PROFIT_TRIM"] >= 1
    assert len(state["book"]["closed"]) == result["complete_positions"]
    assert result["partial_trims_count_as_wins"] is False


def test_timed_expiry_and_real_endpoint_marks():
    c, f, predictions, meta = fixture(15)
    c["small_max_minutes"] = 3
    result, state = run_account(c, {"XUSDT": f}, predictions, meta, 0, 15 * 60000, {"XUSDT": 105})
    assert state["book"]["closed"][0]["closed_ms"] == 300000
    assert state["book"]["curve"][-1]["timestamp"] == 15 * 60000
    assert result["complete_calendar_days_for_risk_metrics"] == 0


def test_label_no_fill_and_stop_before_target():
    bars = np.tile([100.0, 100.0, 100.0, 100.0, 100.0, 0.0], (400, 1))
    assert payoff_label(bars, 10, 1, 0.009, 0.007, 120, 0.01) == (0.0, False)
    bars[11] = [100, 105, 95, 100, 100, 0]
    value, filled = payoff_label(bars, 10, 1, 0.009, 0.007, 120, 0.01)
    assert filled and value < 0


def test_feature_availability_is_previous_closed_bar():
    n = 600
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "close": 100.0,
            "high": 101.0,
            "low": 99.0,
            "mark": 100.0,
            "funding": 0.0,
            "atr": 0.001,
            "volume_ratio": 1.0,
            "flow": 0.1,
            "quote_volume": 1000.0,
            "quote60": 60000.0,
            "taker_buy_quote_volume": 550.0,
        },
        index=np.arange(n) * 60000,
    )
    changed = frame.copy()
    changed.loc[500 * 60000 :, ["close", "quote60"]] = [1000.0, 1e10]
    a = feature_tables({"BTCUSDT": frame})["BTCUSDT"]
    b = feature_tables({"BTCUSDT": changed})["BTCUSDT"]
    np.testing.assert_array_equal(a[:501], b[:501])


def test_hourly_positioning_waits_five_minutes_and_never_reads_future_bar():
    minutes = np.arange(4000) * 60000
    frame = pd.DataFrame(
        {"close": 100.0, "high": 101.0, "low": 99.0, "quote_volume": 1000.0, "quote60": 60000.0},
        index=minutes,
    )
    hours = np.arange(67) * 3600000
    positions = pd.DataFrame(
        {
            "sum_open_interest": 100.0 + np.arange(67),
            "sum_open_interest_value": 10000.0 + np.arange(67),
            "count_long_short_ratio": 1.1,
        },
        index=hours,
    )
    changed = positions.copy()
    changed.loc[50 * 3600000, "count_long_short_ratio"] = 2.0
    a = extra_vectors(frame, positions, 1)
    b = extra_vectors(frame, changed, 1)
    column = EXTRA_FEATURES.index("crowding")
    np.testing.assert_array_equal(a[:3005], b[:3005])
    assert a[3005, column] != b[3005, column]
    assert a[3005, EXTRA_FEATURES.index("metric_age_minutes")] == 0
    future = frame.copy()
    future.loc[3005 * 60000 :, ["close", "high", "quote60"]] = [900.0, 1000.0, 1e10]
    np.testing.assert_array_equal(a[:3006], extra_vectors(future, positions, 1)[:3006])


def test_missing_hourly_positioning_is_excluded_instead_of_carried_forward():
    frame = pd.DataFrame(
        {"close": 100.0, "high": 101.0, "low": 99.0, "quote_volume": 1000.0, "quote60": 60000.0},
        index=np.arange(4000) * 60000,
    )
    positions = pd.DataFrame(
        {
            "sum_open_interest": 100.0,
            "sum_open_interest_value": 10000.0,
            "count_long_short_ratio": 1.1,
        },
        index=np.arange(67) * 3600000,
    )
    positions.loc[50 * 3600000] = np.nan
    a = extra_vectors(frame, positions, 1)
    crowding = EXTRA_FEATURES.index("crowding")
    assert np.isfinite(a[3004, crowding])
    assert np.isnan(a[3005:3065, crowding]).all()
    assert np.isfinite(a[3065, crowding])


@pytest.mark.parametrize("sign", [1, -1])
def test_settled_funding_tightens_owned_stop_without_hiding_gap_losses(sign):
    c, frame, predictions, meta = fixture(20)
    c["funding_adjusted_stop"] = True
    c["loss_cooldown_minutes"] = 1440
    predictions["XUSDT"][:] = 0
    predictions["XUSDT"][:, 0 if sign == 1 else 1] = [0.9, 0.002, 0.0, 0.0]
    frame["high"] = 100.1
    frame.loc[180000, "funding"] = sign * 0.003
    frame.loc[240000, ["low", "high"]] = [99.2, 100.8]
    result, state = run_account(
        c, {"XUSDT": frame}, predictions, meta, 0, 20 * 60000, {"XUSDT": 100.0}
    )
    first = state["book"]["closed"][0]
    assert result["event_counts"]["FUNDING_STOP_TIGHTEN"] == 1
    assert first["net_pnl"] >= -first["risk_budget"] - 1e-8
    gap = frame.copy()
    gap.loc[180000, "open"] = 98.0 if sign == 1 else 102.0
    gap.loc[180000, "low" if sign == 1 else "high"] = 98.0 if sign == 1 else 102.0
    _, breached = run_account(c, {"XUSDT": gap}, predictions, meta, 0, 20 * 60000, {"XUSDT": 100.0})
    closed = breached["book"]["closed"][0]
    assert closed["net_pnl"] < -closed["risk_budget"]
    assert breached["events"][-1]["action"] == "FUNDING_BUDGET_CLOSE"
