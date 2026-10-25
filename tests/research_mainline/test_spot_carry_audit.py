import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from research_tools.carry_feasibility.account_audit import audit_account
from research_tools.factor_lab.io import save, sha

source = Path(__file__).parents[3] / "freqtrade/tests/research/test_spot_perpetual_account.py"
spec = importlib.util.spec_from_file_location("cash_account_fixture", source)
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def write_result(folder, parts, result):
    folder.mkdir()
    result["curve"].to_parquet(folder / "equity_5m.parquet", index=False)
    for key in ["events", "trades", "positions"]:
        save(folder / (key + ".json"), result[key])
    save(folder / "signals.json", parts[4])
    nav = result["curve"].equity
    metrics = {
        "net_return": float(nav.iloc[-1] / 10000 - 1),
        "max_drawdown": float((1 - nav / nav.cummax()).max()),
        "closed_net_pnl": sum(t["net_pnl"] for t in result["trades"]),
        "fees": sum(e.get("fee", 0) for e in result["events"]),
        "funding_net": sum(e["cash_delta"] for e in result["events"] if e["type"] == "funding"),
        "sharpe": None,
    }
    save(
        folder / "result.json",
        {
            "config": parts[5],
            "window": {"start": "2025-01-01", "end": "2025-01-01 00:20"},
            "metrics": metrics,
            "flags": result["flags"],
            "source_hashes": {p.name: sha(p) for p in folder.iterdir()},
        },
    )


def test_independent_audit_rejects_consistent_ledger_that_omits_real_negative_funding(tmp_path):
    parts = list(fixture_module.fixture())
    real = {s: f.copy() for s, f in parts[3].items()}
    real["X"] = pd.DataFrame(
        {"funding_rate": [-0.01], "mark": [100.0]},
        index=pd.to_datetime(["2025-01-01T00:10Z"], utc=True),
    )
    result = fixture_module.run(parts)
    write_result(tmp_path / "forged", parts, result)
    # This forged account's ledger and NAV agree with each other but omit a real bill.
    with pytest.raises(ValueError, match="funding missing"):
        audit_account(tmp_path / "forged", parts[0], parts[1], parts[2], real)


def test_full_segregated_ledger_passes_without_trusting_engine_nav(tmp_path):
    parts = fixture_module.fixture()
    result = fixture_module.run(parts)
    write_result(tmp_path / "valid", parts, result)
    audit = audit_account(tmp_path / "valid", *parts[:4])
    assert audit["passed"] and audit["maximum_nav_error"] < 1e-8
