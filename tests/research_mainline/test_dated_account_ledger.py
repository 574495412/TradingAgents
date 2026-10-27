"""End-to-end independent ledger checks, without market strategy trials."""

import copy
import json

import pandas as pd
import pytest
from test_dated_account_contract import engine, inputs, replay

from research_tools.dated_carry.account_audit import audit_account
from research_tools.dated_carry.run_accounts import publish
from research_tools.factor_lab.io import sha


def test_independent_ledger_accepts_settlement_and_rejects_duplicate_cash(tmp_path):
    d = inputs()
    times = pd.date_range("2024-12-31T23:55Z", "2025-01-02T00:00Z", freq="5min")
    for frames in d[:3]:
        frames["BTCUSDT"] = frames["BTCUSDT"].reindex(times).fillna(100.0)
        frames["BTCUSDT"]["quote_volume"] = 1000000.0
    window = {"start": "2025-01-01T00:00Z", "end": "2025-01-02T00:00Z"}
    account = engine("spot_dated_account_v1.py").replay(
        *d[:6], window["start"], window["end"], contracts=d[6]
    )
    folder = tmp_path / "book"
    publish(folder, account, d[4], d[5], {"window": window, "contracts": d[6]}, d[0])
    proof = audit_account(folder, *d[:4], d[6])
    assert proof["passed"] and proof["settlement_events"] == 1
    events = json.loads((folder / "events.json").read_text())
    at = next(i for i, e in enumerate(events) if e["type"] == "settlement")
    events.insert(at + 1, copy.deepcopy(events[at]))
    (folder / "events.json").write_text(json.dumps(events))
    result = json.loads((folder / "result.json").read_text())
    result["source_hashes"]["events.json"] = sha(folder / "events.json")
    (folder / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Settlement identity"):
        audit_account(folder, *d[:4], d[6])


def test_replay_does_not_mutate_retained_input_prices():
    d = inputs()
    before = copy.deepcopy(d[:3])
    replay(d)
    for original, current in zip(before, d[:3], strict=True):
        pd.testing.assert_frame_equal(original["BTCUSDT"], current["BTCUSDT"], check_exact=True)


def test_known_zero_dated_liquidity_does_not_create_an_entry():
    d = inputs()
    d[1]["BTCUSDT"].loc[pd.Timestamp("2025-01-01T00:00Z"), "quote_volume"] = 0
    result = replay(d)
    assert not any(e["type"] == "entry" for e in result["events"])
    assert result["curve"].equity.eq(10000).all()
