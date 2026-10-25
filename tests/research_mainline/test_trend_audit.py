"""Independent auditor rejects ledger, funding and capacity falsehoods after rehashing."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "trend_auditor",
    Path(__file__).resolve().parents[2] / "tradingagents/research/quant/trend_mainline_audit.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture(root):
    folder = root / "train/Example_capital_1000/cost_0bps"
    folder.mkdir(parents=True)
    (root / "data").mkdir()
    times = pd.date_range("2024-01-01T01:00Z", "2024-01-01T02:00Z", freq="5min")
    bars = pd.DataFrame(
        {"close": 100.0, "quote_volume": 100000.0}, index=times - pd.Timedelta(minutes=5)
    )
    bars.to_parquet(root / "data/BTCUSDT_5m.parquet")
    pd.DataFrame({"funding_rate": [], "mark": []}, index=pd.DatetimeIndex([], tz="UTC")).to_parquet(
        root / "data/BTCUSDT_funding.parquet"
    )
    curve = pd.DataFrame(
        {
            "date": times,
            "equity": [1000.0] + [999.9] * 12,
            "cash": [1000.0, 999.95, 999.95] + [999.9] * 10,
        }
    )
    curve.to_parquet(folder / "equity_5m.parquet", index=False)
    events = [
        {
            "type": kind,
            "time": clock,
            "submitted_at": submitted,
            "symbol": "BTCUSDT",
            "position_id": 1,
            "quantity": 1.0,
            "price": 100.0,
            "side": 1,
            "fee": 0.05,
            "cash_delta": -0.05,
        }
        for kind, clock, submitted in [
            ("entry", "2024-01-01T01:05:00+00:00", "2024-01-01T01:00:00+00:00"),
            ("exit", "2024-01-01T01:15:00+00:00", "2024-01-01T01:10:00+00:00"),
        ]
    ]
    for name, value in [
        ("events", events),
        ("trades", [{"id": 1, "net_pnl": -0.1}]),
        ("state", {"cash": 999.9, "positions": {}, "halted": False}),
    ]:
        (folder / (name + ".json")).write_text(json.dumps(value))
    result = {
        "start": "2024-01-01T01:00",
        "end": "2024-01-01T02:00",
        "initial_equity_usdt": 1000,
        "extra_cost_bps_per_side": 0,
        "net_return": -0.0001,
        "max_drawdown": 0.0001,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "daily_sharpe": None,
        "closed_trades": 1,
        "economic_audit": {"passed": False},
    }
    plan = {
        "symbols": ["BTCUSDT"],
        "fee_per_side": 0.0005,
        "execution": {"realized_interval_participation": 0.01},
        "screening_targets": {
            "minimum_complete_positions": 30,
            "minimum_profit_factor": 1.1,
            "minimum_daily_sharpe": 0.5,
            "maximum_drawdown": 0.15,
        },
    }
    return folder, result, plan


def seal(root, folder, result):
    result["source_hashes"] = {
        p.name: AUDIT.sha(p) for p in folder.iterdir() if p.name != "result.json"
    }
    (folder / "result.json").write_text(json.dumps(result))
    return {
        "path": str(folder.relative_to(root)),
        "result_sha256": AUDIT.sha(folder / "result.json"),
    }


def test_known_two_fill_cash_account(tmp_path):
    folder, result, plan = fixture(tmp_path)
    report = AUDIT.audit_account(tmp_path, seal(tmp_path, folder, result), plan)
    assert report["passed"] and not report["economic_screen_passed"]
    assert report["maximum_nav_error_usdt"] < 1e-9


@pytest.mark.parametrize("fault", ["cash", "capacity", "missing_funding"])
def test_rehashed_semantic_corruption_still_fails(tmp_path, fault):
    folder, result, plan = fixture(tmp_path)
    events = json.loads((folder / "events.json").read_text())
    if fault == "cash":
        events[-1]["cash_delta"] = -0.04
    elif fault == "capacity":
        events[-1]["quantity"] = 100000
    else:
        pd.DataFrame(
            {"funding_rate": [0.01], "mark": [100.0]}, index=pd.DatetimeIndex(["2024-01-01T01:10Z"])
        ).to_parquet(tmp_path / "data/BTCUSDT_funding.parquet")
    (folder / "events.json").write_text(json.dumps(events))
    with pytest.raises(ValueError):
        AUDIT.audit_account(tmp_path, seal(tmp_path, folder, result), plan)


def coordinator():
    path = Path(__file__).resolve().parents[2] / "scripts/trend_mainline.py"
    spec = importlib.util.spec_from_file_location("coordinator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coordinator_preserves_prior_runs(tmp_path, monkeypatch):
    c = coordinator()
    monkeypatch.setattr(c, "ROOT", tmp_path / "TradingAgents")
    fq = tmp_path / "freqtrade"
    (fq / "user_data/crypto_quant/trend_mainline_existing").mkdir(parents=True)
    with pytest.raises(ValueError, match="preserved"):
        c.run("trend_mainline_existing", fq)


def test_coordinator_stops_after_failed_stage_and_records_it(tmp_path, monkeypatch):
    import subprocess

    c = coordinator()
    monkeypatch.setattr(c, "ROOT", tmp_path / "TradingAgents")
    fq = tmp_path / "freqtrade"
    (fq / ".venv/bin").mkdir(parents=True)
    (fq / ".venv/bin/python").touch()
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0 if len(calls) == 1 else 9)

    monkeypatch.setattr(c.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="prepare"):
        c.run("trend_mainline_failure", fq)
    assert len(calls) == 2
    receipt = tmp_path / "TradingAgents/results/research/trend_mainline_failure_workflow"
    assert json.loads((receipt / "prepare.json").read_text())["returncode"] == 9
    assert not (receipt / "accounts.json").exists()
