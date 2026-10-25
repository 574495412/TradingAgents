"""Adversarial accounting / recovery examples for the equity-only extension."""

import json

import numpy as np
import pandas as pd
import pytest

from factors.equity.opportunities import targets
from tradingagents.research.equities.audit_v002 import audit
from tradingagents.research.equities.continuous import replay, statistics
from tradingagents.research.equities.data import sessions
from tradingagents.research.equities.fees import charge


def fixture():
    cal = sessions("US", "2024-01-01", "2024-03-15")
    f = pd.DataFrame(
        {
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "volume": 1000000.0,
            "dividend": 0.0,
            "split": 0.0,
        },
        index=cal.index,
    )
    spec = {
        "initial_cash": 1000,
        "minimum_fee": 1,
        "commission_bps": 1,
        "dividend_haircut": 0.3,
        "symbols": [{"symbol": "SPY.US", "lot_size": 1, "instrument": "ETF"}],
    }
    weights = pd.DataFrame(0.0, index=cal.index, columns=["SPY.US"])
    weights.iloc[20:22] = 0.8
    f.iloc[23, f.columns.get_loc("dividend")] = 1.0
    return {"SPY.US": f}, cal, weights, pd.Series("up", index=cal.index), spec


def run(args, **kwargs):
    options = {
        "market": "US",
        "start": "2024-01-01",
        "end": "2024-03-15",
        "slippage_bps": 5,
        "identity": "frozen_inputs",
        "policy": "static",
    }
    options.update(kwargs)
    return replay(*args, **options)


def test_next_session_dividend_and_sale_cash_lock_with_independent_audit():
    args = fixture()
    st = run(args)
    buy, sell = st["fills"]
    assert buy["day"] == str(args[1].index[21].date())
    assert sell["day"] == str(args[1].index[23].date())
    assert buy["price"] > 10 > sell["price"]
    assert st["dividends"][0]["quantity"] == buy["quantity"]
    assert st["receivables"] == pytest.approx(buy["quantity"] * 0.7)
    by_day = {r["day"]: r for r in st["equity_curve"][1:]}
    for i in (23, 24, 25):
        assert by_day[str(args[1].index[i].date())]["unsettled"] > 0
    assert by_day[str(args[1].index[26].date())]["unsettled"] == 0
    assert audit(st, args[0], args[1])["passed"]
    m = statistics(st)
    assert m["net_pnl"] == pytest.approx(m["realized_closed_net_pnl"])
    assert m["closed_trades"] == 1
    assert m["win_rate_wilson95"] is not None


def test_json_recovery_preserves_inventory_pending_cash_peak_and_rejects_skips(tmp_path):
    args = fixture()
    for split in (22, 24):
        end = str(args[1].index[split].date())
        checkpoint = run(args, end=end)
        file = tmp_path / "checkpoint.json"
        file.write_text(json.dumps(checkpoint))
        resumed = run(args, start=end, state=json.loads(file.read_text()))
        assert resumed == run(args)
        assert checkpoint["positions"] or checkpoint["unsettled"]
        with pytest.raises(ValueError, match="very next"):
            run(args, start=str(args[1].index[split + 1].date()), state=checkpoint)
        with pytest.raises(ValueError, match="identity"):
            run(args, start=end, state=checkpoint, identity="changed")


def test_partial_rebalances_form_one_full_cycle_not_fake_trade_wins():
    args = fixture()
    args[2].iloc[21:23] = 0.4
    st = run(args)
    assert len(st["fills"]) == 3
    assert len(st["trades"]) == 1
    assert audit(st, args[0], args[1])["passed"]


def test_independent_ledger_detects_cash_and_fee_tampering():
    args = fixture()
    state = run(args)
    for mutation in ("balance", "fees", "dividend"):
        bad = json.loads(json.dumps(state))
        if mutation == "balance":
            bad["equity_curve"][-1]["cash"] += 1
        elif mutation == "fees":
            bad["fills"][0]["fee_parts"]["commission"] = 0
        else:
            bad["dividends"][0]["amount"] += 1
        with pytest.raises(AssertionError):
            audit(bad, args[0], args[1])


def test_risk_stop_is_continuous_and_survives_recovery():
    args = fixture()
    args[2].iloc[20:] = 0.8
    f = args[0]["SPY.US"]
    f.loc[f.index[24] :, ["open", "high", "low", "close"]] *= 0.5
    full = run(args, policy="finite_state")
    assert full["halted"] and full["peak"] >= 1000
    cut = str(args[1].index[26].date())
    checkpoint = run(args, policy="finite_state", end=cut)
    assert checkpoint["halted"]
    assert (
        run(args, policy="finite_state", start=cut, state=json.loads(json.dumps(checkpoint)))
        == full
    )


def test_weights_prefix_future_perturbation_and_warmup():
    dates = pd.bdate_range("2020-01-01", periods=340)
    f = pd.DataFrame({"close": 100 + np.arange(340) * 0.2 + np.sin(np.arange(340))}, index=dates)
    for policy in ("static", "risk_scaled", "finite_state", "trend_only", "recovery_only"):
        full, _ = targets({"A": f, "B": f * 1.1}, policy)
        for n in (60, 200, 275):
            prefix, _ = targets({"A": f.iloc[:n], "B": f.iloc[:n] * 1.1}, policy)
            pd.testing.assert_frame_equal(prefix, full.iloc[:n])
        future = f.copy()
        future.iloc[280:] *= 4
        changed, _ = targets({"A": future, "B": future * 1.1}, policy)
        pd.testing.assert_frame_equal(changed.iloc[:280], full.iloc[:280])
        assert full.sum(axis=1).le(0.80000001).all()


def test_fee_dates_etf_exemption_and_reject_unsupported_dates():
    spec = fixture()[-1]
    assert charge("US", "2025-05-13", "SELL", 100, 1000.0, spec)["sec31"] == 2.78
    assert charge("US", "2025-05-14", "SELL", 100, 1000.0, spec)["sec31"] == 0
    assert charge("US", "2026-04-06", "SELL", 100, 1000.0, spec)["sec31"] == 2.06
    assert charge("US", "2026-01-02", "SELL", 100, 1000.0, spec)["taf"] == 0.02
    assert charge("HK", "2025-06-27", "BUY", 1000, 10.0, spec)["settlement"] == 2
    assert charge("HK", "2025-06-30", "BUY", 1000, 10.0, spec)["settlement"] == 0.42
    assert charge("HK", "2025-06-30", "SELL", 1000, 10.0, spec)["stamp"] == 0
    with pytest.raises(ValueError):
        charge("US", "2026-10-01", "SELL", 100, 1000.0, spec)


def test_reject_negative_and_missing_targets_and_source_splits():
    args = fixture()
    args[2].iloc[21] = -0.1
    with pytest.raises(ValueError, match="weights"):
        run(args)
    args[2].iloc[21] = np.nan
    with pytest.raises(ValueError, match="weights"):
        run(args)
    args[2].iloc[21] = 0.8
    args[0]["SPY.US"].iloc[21, 6] = 2
    with pytest.raises(ValueError, match="split"):
        run(args)
