"""真实文件与子进程验证持仓恢复、提交尾部重放和故障隔离。"""

import asyncio
import copy
import json
import signal
import subprocess
import sys

import pytest

from tests.test_research_paper_account import MINUTE, PaperAccount, entered, frame, payment, public
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.forward_inputs import Bars
from tradingagents.research.quant.backtesting.paper_live import (
    bootstrap_expired,
    exit_wakeup,
    history_records,
    startup_identity,
    warm_history,
)
from tradingagents.research.quant.backtesting.paper_live_audit import (
    require_account_contract,
    require_original,
)
from tradingagents.research.quant.backtesting.paper_store import StudyStore, atomic, entries


def protocol():
    return {"symbols": ["A"], "simulation_end_ms": 200 * MINUTE, "fixtures_only": True}


@pytest.mark.parametrize("changed", ["slippage_bps_each_side", "symbols", "period_end_ms"])
def test_audit_binds_cost_account_label_universe_and_original_end(changed):
    frozen = protocol()
    state = PaperAccount(
        frozen["symbols"], 10, period_end_ms=frozen["simulation_end_ms"]
    ).checkpoint()
    require_account_contract(state, frozen, "10")
    state["protocol"][changed] = {
        "slippage_bps_each_side": 3,
        "symbols": ["B"],
        "period_end_ms": 201 * MINUTE,
    }[changed]
    with pytest.raises(ValueError, match="账户成本、固定全集或结束时间"):
        require_account_contract(state, frozen, "10")


def commit(store, value):
    store.source.add({"kind": "fixture_source", "received_ms": value["actual_ms"]})
    store.commit(value)


def test_directory_lock_prevents_two_account_writers(tmp_path):
    root = tmp_path / "study"
    first = StudyStore(root, protocol())
    try:
        with pytest.raises(ValueError, match="进程持有文件锁"):
            StudyStore(root, resume=True)
    finally:
        first.close()


def test_checkpoint_behind_durable_frames_replays_committed_tail(tmp_path):
    root = tmp_path / "study"
    store = StudyStore(root, protocol())
    for minute in range(100, 104):
        commit(store, frame(minute, direction=1 if minute >= 103 else 0))
    old = json.loads((root / "checkpoint.json").read_text())
    for minute in (104, 105):
        commit(store, frame(minute, direction=1))
    expected = store.states()
    store.close()
    atomic(root / "checkpoint.json", old)
    restored = StudyStore(root, resume=True)
    try:
        assert restored.states() == expected
        assert restored.replayed_committed_tail == 2
        assert all(len(a.positions) == 1 for a in restored.accounts.values())
        assert restored.protocol["simulation_end_ms"] == 200 * MINUTE
    finally:
        restored.close()


def test_torn_uncommitted_lines_are_preserved_separately_without_losing_committed_inventory(
    tmp_path,
):
    root = tmp_path / "study"
    store = StudyStore(root, protocol())
    for minute in range(100, 106):
        commit(store, frame(minute, direction=1 if minute >= 103 else 0))
    expected = store.states()
    store.close()
    for name in ("public_events.jsonl", "account_frames.jsonl"):
        with (root / name).open("ab") as target:
            target.write(b'{"sequence":')
    restored = StudyStore(root, resume=True)
    try:
        assert restored.states() == expected
        assert len(list(root.glob("*.uncommitted-*"))) == 2
    finally:
        restored.close()


def test_changed_checkpoint_or_complete_corrupt_line_cannot_be_silently_repaired(tmp_path):
    root = tmp_path / "study"
    store = StudyStore(root, protocol())
    commit(store, frame(100))
    store.close()
    value = json.loads((root / "checkpoint.json").read_text())
    value["accounts"]["3"]["cash"] = "9999"
    atomic(root / "checkpoint.json", value)
    with pytest.raises(ValueError, match="检查点不能"):
        StudyStore(root, resume=True)
    raw = root / "public_events.jsonl"
    with raw.open("ab") as target:
        target.write(b'{"sequence":100,"previous_sha256":null,"record":{}}\n')
    with pytest.raises(ValueError, match="校验链不连续"):
        StudyStore(root, resume=True)


def test_exit_only_frame_cannot_consume_pending_recheck_and_restart_explicitly_does():
    account = PaperAccount(["A"], 3)
    for minute in range(100, 105):
        account.apply(frame(minute, direction=1 if minute >= 103 else 0))
    tick = frame(105, direction=1)
    tick["phase"] = "exit_only"
    tick["evidence"] = {"A": {"ready": False}}
    account.apply(tick)
    assert "A" in account.pending and not account.positions
    valid = frame(105, direction=1, now=105 * MINUTE + 8000)
    account.apply(valid)
    assert len(account.positions) == 1
    restarted = copy.deepcopy(account)
    future = frame(106, direction=1)
    future["source_restart"] = True
    restarted.apply(future)
    assert len(restarted.positions) == 1 and not restarted.pending
    assert restarted.quality_errors[-1]["reason"] == "OBSERVATION_GAP"


def test_sigkill_releases_lock_and_new_process_state_recovers_an_open_position(tmp_path):
    root = tmp_path / "crashed"
    code = """
import os, signal, sys
from pathlib import Path
sys.path.insert(0, str(Path("tests").resolve()))
from tradingagents.research.quant.backtesting.paper_store import StudyStore
from tests.test_research_paper_store import protocol, commit
from tests.test_research_paper_account import frame
store = StudyStore(Path(sys.argv[1]), protocol())
for minute in range(100, 106):
    commit(store, frame(minute, direction=1 if minute >= 103 else 0))
os.kill(os.getpid(), signal.SIGKILL)
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(root)], capture_output=True, text=True, check=False
    )
    assert process.returncode == -signal.SIGKILL, process.stderr
    before = json.loads((root / "checkpoint.json").read_text())
    restored = StudyStore(root, resume=True)
    try:
        assert restored.states() == before["accounts"]
        assert all(len(a.positions) == 1 for a in restored.accounts.values())
        assert len(list(entries(root / "account_frames.jsonl"))) == 6
        next_frame = frame(106, direction=1)
        next_frame["source_restart"] = True
        commit(restored, next_frame)
        assert len(restored.accounts["3"].fills) == 1
    finally:
        restored.close()


def test_empty_failed_http_response_is_preserved_and_can_be_recovered(tmp_path):
    root = tmp_path / "failed_http"
    store = StudyStore(root, protocol())
    failed = received_record("", received_ms=100 * MINUTE) | {"kind": "index", "status": 0}
    store.source.add(failed)
    commit(store, frame(100))
    expected = store.states()
    store.close()
    restored = StudyStore(root, resume=True)
    try:
        assert restored.states() == expected
        assert any(record == failed for _, record, _ in entries(root / "public_events.jsonl"))
    finally:
        restored.close()


def test_nested_raw_identity_cannot_be_reused_with_an_earlier_received_clock():
    original = received_record("{}", received_ms=1000)
    from tradingagents.dataflows.research.live_capture import digest

    identities = {digest(original)}
    require_original(original, identities)
    forged = original | {"received_ms": 999}
    with pytest.raises(ValueError, match="包装不在已提交前缀"):
        require_original(forged, identities)


def test_funding_rechecks_wake_early_then_later_and_do_not_repeat_verified_old_trades():
    account, _ = entered()
    for minute in range(106, 132):
        value = frame(
            minute,
            direction=0 if 126 <= minute <= 128 else 1,
            now=minute * MINUTE + (8000 if minute == 125 else 5000),
        )
        if minute >= 126:
            value["funding"] = [payment(minute, [])]
        account.apply(value)
        if minute == 125:
            assert exit_wakeup({"3": account}, 200 * MINUTE) == account.trades[0]["exit_ms"] + 1001
        if minute == 126:
            assert (
                exit_wakeup({"3": account}, 200 * MINUTE)
                == account.trades[0]["exit_ms"] + MINUTE + 1000
            )
    assert len(account.positions) == 1 and len(account.trades) == 1
    assert (
        exit_wakeup({"3": account}, 200 * MINUTE)
        == account.positions["A"]["entry_ms"] + 20 * MINUTE
    )


def test_startup_retry_keeps_failed_bytes_and_only_retries_missing_fixed_members(monkeypatch):
    class Journal:
        def __init__(self):
            self.records = []

        def add(self, record):
            self.records.append(record)

    class Client:
        def __init__(self):
            self.calls = []

        def batch(self, jobs):
            self.calls.append([j[1] for j in jobs])
            result = []
            for _, symbol, _ in jobs:
                r = public(
                    "klines",
                    [frame(100)["marks"]["A"]["row"]],
                    100 * MINUTE + 1000,
                    100 * MINUTE + 2000,
                    symbol=symbol,
                )
                if symbol == "B" and len(self.calls) == 1:
                    r = public(
                        "klines", [], 100 * MINUTE + 1000, 100 * MINUTE + 2000, symbol=symbol
                    ) | {"status": 0}
                result.append(r)
            return result

    async def no_delay(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_delay)
    client, journal = Client(), Journal()
    jobs = [("klines", s, {}) for s in ("A", "B")]
    records = asyncio.run(history_records(client, journal, jobs, 100 * MINUTE))
    assert client.calls == [["A", "B"], ["B"]]
    assert len(records) == 3 and records[1]["status"] == 0
    assert len(journal.records) == 4


def test_exhausted_startup_retry_cannot_shrink_the_required_overlap(monkeypatch):
    class Client:
        calls = 0

        def batch(self, jobs):
            self.calls += 1
            return [
                public("klines", [], 100 * MINUTE + 1000, 100 * MINUTE + 2000, symbol=s)
                | {"status": 0}
                for _, s, _ in jobs
            ]

    class Journal:
        def add(self, _):
            return None

    async def no_delay(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_delay)
    client = Client()
    with pytest.raises(ValueError, match="没有完整交接"):
        asyncio.run(warm_history(client, Journal(), Bars(["A"]), ["A"], 100 * MINUTE, ["A"]))
    assert client.calls == 3


def test_unknown_closed_position_funding_cannot_finance_a_new_activity():
    account, _ = entered()
    for minute in range(106, 115):
        value = frame(
            minute,
            direction=0 if minute in (110, 111) else 1,
            price="102" if 107 <= minute <= 109 else "100",
        )
        account.apply(value)
    assert len(account.trades) == 1 and not account.positions
    assert account.declines[-1]["reason"] == "CLOSED_POSITION_FUNDING_COVERAGE_INCOMPLETE"


@pytest.mark.parametrize("failure", ["transient_metadata", "persistent_metadata", "clock"])
def test_startup_identity_retries_are_bounded_and_clock_failure_cannot_admit_metadata(
    monkeypatch, failure
):
    closing = 100 * MINUTE

    class Client:
        def __init__(self):
            self.calls = []

        def request(self, kind, symbol, params):
            self.calls.append(kind)
            if kind == "clock":
                return public(
                    kind,
                    {"serverTime": closing + 1500 if failure != "clock" else 0},
                    closing + 1000,
                    closing + 2000,
                )
            result = public(
                kind,
                {
                    "symbols": [
                        {
                            "symbol": "A",
                            "status": "TRADING",
                            "quoteAsset": "USDT",
                            "contractType": "TRADIFI_PERPETUAL",
                            "onboardDate": 0,
                            "deliveryDate": 0,
                        }
                    ]
                },
                closing + 1000,
                closing + 2000,
            )
            if failure == "persistent_metadata" or self.calls.count("exchange") == 1:
                result["status"] = 0
            return result

    class Journal:
        def __init__(self):
            self.records = []

        def add(self, record):
            self.records.append(record)

    async def no_delay(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_delay)
    monkeypatch.setattr(
        "tradingagents.research.quant.backtesting.paper_live.milliseconds", lambda: closing + 5000
    )
    client, journal = Client(), Journal()
    if failure == "transient_metadata":
        _, contracts = asyncio.run(
            startup_identity(client, journal, {"A": {"asset_class": "TRADIFI_PERPETUAL"}})
        )
        assert set(contracts) == {"A"} and client.calls == ["clock", "exchange", "exchange"]
    else:
        with pytest.raises(ValueError):
            asyncio.run(
                startup_identity(client, journal, {"A": {"asset_class": "TRADIFI_PERPETUAL"}})
            )
        assert client.calls == (
            ["clock"] * 3 if failure == "clock" else ["clock"] + ["exchange"] * 3
        )


def test_reconnect_has_its_own_bootstrap_budget_without_extending_the_study():
    assert bootstrap_expired({"connection_started_ms": 100_000}, 100_000, 400_000, 600_000)
    status = {"connection_started_ms": 380_000}
    assert not bootstrap_expired(status, 100_000, 400_000, 600_000)
    assert not bootstrap_expired(status, 100_000, 600_001, 600_000)
    assert status == {"connection_started_ms": 380_000}
