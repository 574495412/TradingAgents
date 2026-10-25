"""Independent audit rejects omitted settlement and nonregistered entry clocks."""

from pathlib import Path

import pandas as pd
import pytest

from research_tools.factor_lab.io import read, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.max_effect.account_audit import audit_account
from research_tools.regime_program.storage import write_account


def account(tmp_path):
    index = pd.date_range("2024-12-31 23:55", "2025-01-03", freq="5min", inclusive="left", tz="UTC")
    bars = {
        "BTCUSDT": pd.DataFrame(
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "quote_volume": 1e8},
            index=index,
        )
    }
    rates = {
        "BTCUSDT": pd.DataFrame(
            {"funding_rate": [0.001, 0.002], "mark": [110.0, 120.0]},
            index=pd.to_datetime(
                ["2025-01-01T00:05:00.015Z", "2025-01-02T00:00Z"], format="mixed", utc=True
            ),
        )
    }
    hours = pd.date_range("2025-01-01", "2025-01-03", freq="h", inclusive="left", tz="UTC")
    scores = pd.DataFrame(
        {"target_BTCUSDT": 0.5, "risk_scale": 1.0, "rebalance": False}, index=hours
    )
    scores.loc[hours[0], "rebalance"] = True
    scores.loc["2025-01-02":, "target_BTCUSDT"] = 0
    scores.loc["2025-01-02T00:00Z", "rebalance"] = True
    config = {
        "capital": 10000,
        "fee_bps": 5,
        "slippage_bps": 3,
        "participation": 0.01,
        "gross": 0.8,
        "max_positions": 10,
        "threshold": 0.0032,
        "stop_loss": 0.08,
        "drawdown_halt": 0.15,
    }
    engine = executor(
        str(Path(__file__).resolve().parents[3] / "freqtrade/research/crypto/calendar_portfolio.py")
    )
    a = engine.replay(bars, rates, scores, config, "2025-01-01", "2025-01-03")
    benchmark = pd.Series(
        100.0, index=pd.date_range("2025-01-01", "2025-01-03", freq="D", tz="UTC")
    )
    a["metrics"] = portfolio_metrics(a["curve"], a["trades"], a["events"], 10000, benchmark)
    row = {"id": "case"}
    write_account(
        tmp_path / "accounts/case", a, config, {"start": "2025-01-01", "end": "2025-01-03"}
    )
    return row, bars, rates, scores


def test_independent_audit_rejects_funding_omission_even_with_resealed_ledger(tmp_path):
    import json

    row, bars, rates, scores = account(tmp_path)
    assert audit_account(tmp_path, row, bars, rates, scores)["passed"]
    folder = tmp_path / "accounts/case"
    events = read(folder / "events.json")
    removed = next(e for e in events if e["type"] == "funding")
    events.remove(removed)
    (folder / "events.json").write_text(json.dumps(events))
    result = read(folder / "result.json")
    result["source_hashes"]["events.json"] = sha(folder / "events.json")
    (folder / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Missing or extra exact funding"):
        audit_account(tmp_path, row, bars, rates, scores)


def test_independent_audit_rejects_entry_without_weekly_formation(tmp_path):
    row, bars, rates, scores = account(tmp_path)
    scores["rebalance"] = False
    with pytest.raises(ValueError, match="prior registered weekly signal"):
        audit_account(tmp_path, row, bars, rates, scores)
