"""Settlement, cash ownership, stale-source and checkpoint counterexamples."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def engine(name):
    p = Path("../freqtrade/research/crypto") / name
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def inputs():
    times = pd.date_range("2025-01-01T00:00Z", periods=15, freq="5min")
    f = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "quote_volume": 1000000.0},
        index=times,
    )
    market = {"BTCUSDT": f}
    rates = {
        "BTCUSDT": pd.DataFrame(
            columns=["funding_rate", "mark"], index=pd.DatetimeIndex([], tz="UTC")
        )
    }
    config = {
        "capital": 10000,
        "spot_fee_bps": 10,
        "perpetual_fee_bps": 5,
        "settlement_fee_bps": 5,
        "slippage_bps": 3,
        "entry_slippage_buffer_bps": 10,
        "initial_futures_wallet_fraction": 0.7,
        "asset_spot_fraction": 0.25,
        "gross_reduction_trigger": 0.8,
        "gross_reduction_target": 0.5,
        "participation": 0.01,
        "drawdown_halt": 0.15,
        "initial_margin_ratio": 1,
        "maintenance_stress_ratio": 0.05,
        "margin_exit_ratio": 0.2,
        "rearm_hours": 24,
        "pretrade_liquidity_cap": True,
    }
    contracts = {
        "BTCUSDT": {
            "symbol": "BTCUSDT_TEST",
            "base": "BTC",
            "quote": "USDT",
            "style": "linear",
            "expiry": "2025-01-01T00:35:00Z",
            "entry_stop": "2025-01-01T00:25:00Z",
            "known_at": "2024-01-01T00:00:00Z",
            "settlement_price": 99.0,
        }
    }
    signals = [
        {"time": "2025-01-01T00:05:00Z", "assets": {"BTCUSDT": {"entry": True, "exit": False}}}
    ]
    return market, copy.deepcopy(market), copy.deepcopy(market), rates, signals, config, contracts


def replay(data, end="2025-01-01T01:00Z", **kw):
    return engine("spot_dated_account_v1.py").replay(
        *data[:6], "2025-01-01T00:05Z", end, contracts=data[6], **kw
    )


def test_unchanged_prefix_reproduces_original_two_wallet_engine():
    d = inputs()
    a = replay(d, end="2025-01-01T00:20Z", final_end="2025-01-01T01:00Z")
    b = engine("spot_perpetual_account_v2.py").replay(
        *d[:6], "2025-01-01T00:05Z", "2025-01-01T00:20Z", final_end="2025-01-01T01:00Z"
    )
    pd.testing.assert_frame_equal(a["curve"], b["curve"], check_exact=True)
    assert a["events"] == b["events"] and a["requests"] == b["requests"]


def test_settlement_and_independent_spot_unwind_have_distinct_clocks():
    r = replay(inputs())
    settle = [e for e in r["events"] if e["type"] == "settlement"]
    exits = [e for e in r["events"] if e["type"] == "exit"]
    assert len(settle) == 1 and settle[0]["time"] == "2025-01-01T00:35:00+00:00"
    assert (
        len(exits) == 1
        and exits[0]["market"] == "spot"
        and exits[0]["time"] == "2025-01-01T00:40:00+00:00"
    )
    assert not r["positions"] and not r["capacity"]["hedge_failures"]
    c = r["curve"].set_index("date").loc[pd.Timestamp("2025-01-01T00:35Z")]
    assert c.short_notional == 0 and c.spot_assets > 0


def test_post_expiry_source_prices_cannot_change_any_money_or_fill():
    d = inputs()
    ref = replay(d)
    for market in [d[1], d[2]]:
        market["BTCUSDT"].loc[
            pd.Timestamp("2025-01-01T00:35Z") :, ["open", "high", "low", "close"]
        ] = np.nan
    actual = replay(d)
    pd.testing.assert_frame_equal(actual["curve"], ref["curve"], check_exact=True)
    assert actual["events"] == ref["events"]


def test_checkpoint_at_settlement_preserves_spot_only_inventory_and_does_not_settle_twice():
    d = inputs()
    full = replay(d)
    first = replay(d, end="2025-01-01T00:35Z", final_end="2025-01-01T01:00Z")
    assert (
        first["positions"]["BTCUSDT"]["perp_q"] == 0 and first["positions"]["BTCUSDT"]["spot_q"] > 0
    )
    state = json.loads(json.dumps(first["resume_state"], allow_nan=False))
    last = engine("spot_dated_account_v1.py").replay(
        *d[:6],
        "2025-01-01T00:35Z",
        "2025-01-01T01:00Z",
        contracts=d[6],
        initial_state=state,
        final_end="2025-01-01T01:00Z",
    )
    pd.testing.assert_frame_equal(
        pd.concat([first["curve"], last["curve"].iloc[1:]], ignore_index=True),
        full["curve"],
        check_exact=True,
    )
    assert (
        first["events"] + last["events"] == full["events"]
        and last["resume_state"] == full["resume_state"]
    )


def test_future_settlement_reference_cannot_change_prefix_decisions_or_nav():
    d = inputs()
    prefix = replay(d, end="2025-01-01T00:30Z", final_end="2025-01-01T01:00Z")
    d[6]["BTCUSDT"]["settlement_price"] = 999
    altered = replay(d, end="2025-01-01T00:30Z", final_end="2025-01-01T01:00Z")
    pd.testing.assert_frame_equal(prefix["curve"], altered["curve"], check_exact=True)
    assert prefix["events"] == altered["events"] and prefix["requests"] == altered["requests"]


def test_no_spot_gain_is_available_as_future_margin():
    d = inputs()
    for frames in d[:3]:
        frames["BTCUSDT"].loc[
            pd.Timestamp("2025-01-01T00:15Z") :, ["open", "high", "low", "close"]
        ] = 500
    r = replay(d)
    assert r["risk_halted"] and r["flags"]["possible_liquidation"]


def test_reduce_only_boundary_prevents_a_new_position():
    d = inputs()
    d[4][0]["time"] = "2025-01-01T00:20Z"
    r = replay(d)
    assert not any(e["type"] == "entry" for e in r["events"])
    assert any(e.get("reason") == "reduce_only_or_expired" for e in r["requests"])


def test_dated_funding_and_other_product_checkpoint_are_rejected():
    d = inputs()
    d[3]["BTCUSDT"].loc[pd.Timestamp("2025-01-01T00:10Z")] = [0.001, 100]
    with pytest.raises(ValueError, match="cannot receive perpetual funding"):
        replay(d)
    d = inputs()
    p = replay(d, end="2025-01-01T00:20Z", final_end="2025-01-01T01:00Z")
    p["resume_state"]["engine_version"] = "segregated_spot_perpetual"
    with pytest.raises(ValueError, match="Continuation"):
        engine("spot_dated_account_v1.py").replay(
            *d[:6],
            "2025-01-01T00:20Z",
            "2025-01-01T01:00Z",
            contracts=d[6],
            initial_state=p["resume_state"],
            final_end="2025-01-01T01:00Z",
        )
