"""Verify pair audit rejects internally resealed accounting errors."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from research_tools.factor_lab.io import sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.regime_program.pair_account_audit import audit_account
from research_tools.regime_program.pair_storage import write_account
from research_tools.regime_program.run_pairs import assess

HELPER = Path(__file__).resolve().parents[3] / "freqtrade/tests/research/test_pair_portfolio.py"
if not HELPER.is_file():
    pytest.skip("Optional sibling crypto research executor", allow_module_level=True)
spec = importlib.util.spec_from_file_location("pair_fixture", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def evidence(tmp_path, partial=False):
    bars, rates, signals, config = helper.fixture()
    end = "2025-01-01 06:00" if partial else "2025-01-02"
    if partial:
        bars["Y"].loc["2025-01-01 04:00":, "quote_volume"] = 0
    account = helper.run(bars, rates, signals, config, end=end)
    benchmark = pd.Series(
        [100.0, 100.0], index=pd.DatetimeIndex(["2025-01-01", "2025-01-02"], tz="UTC")
    )
    m = portfolio_metrics(account["curve"], account["trades"], account["events"], 10000, benchmark)
    m["matched_closed_pairs"] = sum(t["hedge_matched"] for t in account["trades"])
    m["open_campaign_settled_net_pnl"] = sum(
        p["net_pnl"] for p in account["pair_positions"].values()
    )
    account["metrics"] = m
    folder = tmp_path / "accounts" / "test"
    write_account(folder, account, config, {"start": "2025-01-01", "end": end})
    return {"id": "test"}, bars, rates, signals, folder


def test_independent_pair_audit_detects_fee_error_even_after_hash_reseal(tmp_path):
    row, bars, rates, signals, folder = evidence(tmp_path)
    assert audit_account(tmp_path, row, bars, rates, signals)["passed"]
    events = json.loads((folder / "events.json").read_text())
    events[0]["fee"] += 0.1
    (folder / "events.json").write_text(json.dumps(events))
    result = json.loads((folder / "result.json").read_text())
    result["source_hashes"]["events.json"] = sha(folder / "events.json")
    (folder / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="price/fee"):
        audit_account(tmp_path, row, bars, rates, signals)


def test_open_pair_keeps_settlement_of_leg_that_already_closed(tmp_path):
    row, bars, rates, signals, folder = evidence(tmp_path, partial=True)
    assert audit_account(tmp_path, row, bars, rates, signals)["passed"]
    result = json.loads((folder / "result.json").read_text())
    assert list(result["open_positions"]) == ["Y"]
    pair = next(iter(result["open_pairs"].values()))
    assert pair["legs"]["X"]["q"] == 0 and pair["legs"]["X"]["net_pnl"] < 0
    pair["net_pnl"] = pair["legs"]["Y"]["net_pnl"]
    (folder / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="closed-leg settlement"):
        audit_account(tmp_path, row, bars, rates, signals)


def test_failed_hedges_cannot_fill_the_100_matched_lifecycle_gate():
    protocol = {
        "admission": {
            "minimum_cagr": 0.2,
            "maximum_drawdown": 0.15,
            "minimum_sharpe": 1,
            "minimum_complete_lifecycles": 100,
            "minimum_closed_profit_factor": 1.15,
        }
    }
    row = {
        "slippage_bps": 10,
        "risk_halted": False,
        "metrics": {
            "cagr": 0.3,
            "max_drawdown": 0.1,
            "sharpe": 1.2,
            "closed_trades": 100,
            "matched_closed_pairs": 0,
            "profit_factor": 2,
            "closed_net_pnl": 1000,
            "max_gross_exposure": 0.8,
        },
    }
    verdict = assess([row], protocol)
    assert not verdict["passed"] and verdict["reasons"] == ["slip10:matched_lifecycles"]
