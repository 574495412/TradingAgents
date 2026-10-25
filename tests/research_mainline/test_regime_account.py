"""Independent fill-to-account reconstruction with actual growing campaigns."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_tools.factor_lab.io import sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.regime_program.account_audit import audit_account
from research_tools.regime_program.storage import write_account


def test_multiple_fills_funding_and_closed_campaign_are_financially_reconstructed(tmp_path):
    engine = executor(
        Path(__file__).resolve().parents[3] / "freqtrade/research/crypto/regime_portfolio.py"
    )
    index = pd.date_range("2024-12-31", "2025-01-05", freq="5min", inclusive="left", tz="UTC")
    price = 100 + np.arange(len(index)) * 0.002
    bars = {
        "BTCUSDT": pd.DataFrame(
            {
                "open": price,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "quote_volume": 1e8,
            },
            index=index,
        )
    }
    funding = {
        "BTCUSDT": pd.DataFrame(
            {"funding_rate": 0.001, "mark": 101.0},
            index=pd.date_range("2025-01-01T08:00", "2025-01-04T08:00", freq="8h", tz="UTC"),
        )
    }
    target = pd.DataFrame(
        {"target_BTCUSDT": 1.0, "risk_scale": 1.0},
        index=pd.date_range("2025-01-01", "2025-01-05", freq="h", inclusive="left", tz="UTC"),
    )
    target.loc["2025-01-02":"2025-01-02T23:00", "risk_scale"] = 0.5
    target.loc["2025-01-04":, "target_BTCUSDT"] = 0
    config = {
        "capital": 10000,
        "fee_bps": 5,
        "slippage_bps": 10,
        "participation": 0.01,
        "gross": 0.8,
        "max_positions": 1,
        "rebalance_hours": 24,
        "stop_loss": 0.08,
        "drawdown_halt": 0.15,
    }
    window = {"start": "2025-01-01", "end": "2025-01-05"}
    a = engine.replay(bars, funding, target, config, **window)
    boundary = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
    benchmark = bars["BTCUSDT"].close.reindex(boundary - pd.Timedelta(minutes=5))
    benchmark.index = boundary
    a["metrics"] = portfolio_metrics(a["curve"], a["trades"], a["events"], 10000, benchmark)
    folder = tmp_path / "accounts" / "resized"
    write_account(folder, a, config, window)
    result = audit_account(tmp_path, {"id": "resized"}, bars, funding)
    assert result["passed"] and result["complete_lifecycles"] == 1
    assert sum(e.get("addition", False) for e in a["events"]) >= 1
    assert (folder / "resume_state.json").is_file()
    # Bypass only the hash check deliberately: accounting checks must still catch
    # a wrong fee even when a publisher has recomputed the enclosing receipt.
    events = json.loads((folder / "events.json").read_text())
    events[0]["fee"] += 1
    (folder / "events.json").write_text(json.dumps(events))
    report = json.loads((folder / "result.json").read_text())
    report["source_hashes"]["events.json"] = sha(folder / "events.json")
    (folder / "result.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Fee"):
        audit_account(tmp_path, {"id": "resized"}, bars, funding)
