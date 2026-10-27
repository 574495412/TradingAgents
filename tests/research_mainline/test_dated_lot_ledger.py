"""Independent integer/cash audit of synthetic complete-day v2 paths."""

import json

import pandas as pd
import pytest
from test_dated_account_contract import engine
from test_dated_account_lots import data

from research_tools.dated_carry.account_audit_lots_v2 import audit_account
from research_tools.dated_carry.run_accounts import publish
from research_tools.factor_lab.io import sha


def account_folder(tmp_path):
    d = data()
    clocks = pd.date_range("2024-12-31T23:55Z", "2025-01-02T00:00Z", freq="5min")
    for frames in d[:3]:
        frames["BTCUSDT"] = frames["BTCUSDT"].reindex(clocks).fillna(100.0)
        frames["BTCUSDT"]["quote_volume"] = 1000000.0
    window = {"start": "2025-01-01T00:00Z", "end": "2025-01-02T00:00Z"}
    account = engine("spot_dated_account_v2.py").replay(
        *d[:6], window["start"], window["end"], contracts=d[6]
    )
    folder = tmp_path / "account"
    result = publish(folder, account, d[4], d[5], {"window": window, "contracts": d[6]}, d[0])
    return d, folder, result


def test_independent_cash_units_and_full_dust_reserve_reconcile(tmp_path):
    d, folder, result = account_folder(tmp_path)
    proof = audit_account(folder, *d[:4], d[6])
    assert proof["passed"] and proof["completed_campaigns"] == 0
    assert result["metrics"]["win_rate"] is None and result["metrics"]["closed_net_pnl"] == 0
    assert abs(proof["open_net_pnl_with_exit_reserve"] - result["metrics"]["net_pnl"]) < 1e-7


def test_dropping_owned_dust_cannot_pass_audit_even_with_updated_file_hash(tmp_path):
    d, folder, _ = account_folder(tmp_path)
    (folder / "positions.json").write_text("{}")
    r = json.loads((folder / "result.json").read_text())
    r["source_hashes"]["positions.json"] = sha(folder / "positions.json")
    (folder / "result.json").write_text(json.dumps(r))
    with pytest.raises(ValueError, match="Final open inventory"):
        audit_account(folder, *d[:4], d[6])
