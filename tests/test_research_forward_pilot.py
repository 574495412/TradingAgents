"""验证活动启动、断流权限、单次触发及研究原文前缀的可恢复性。"""

import asyncio
import json

import pandas as pd

from tradingagents.dataflows.research.live_capture import digest
from tradingagents.research.quant.backtesting import forward_pilot as module
from tradingagents.research.quant.backtesting.forward_pilot import MINUTE, Activities, Journal


def frame():
    return pd.DataFrame({"ret_20_pct": [2.0], "atr_pct": [1.0]}, index=["PUBLIC"])


def evidence():
    return {"PUBLIC": {"positioning_pass": True, "basis_short_pass": True}}


def emit(tracker, clock):
    return tracker.emit(evidence(), frame(), closing=clock, decision_ms=clock + 5000)


def test_startup_activity_cannot_masquerade_as_a_new_trade():
    tracker = Activities()
    tracker.observe(["PUBLIC"], MINUTE)
    assert not emit(tracker, MINUTE)
    tracker.observe(["PUBLIC"], 2 * MINUTE)
    assert not emit(tracker, 2 * MINUTE)


def test_fresh_activity_emits_once_per_arm_with_real_decision_delay():
    tracker = Activities()
    tracker.observe([], MINUTE)
    tracker.observe(["PUBLIC"], 2 * MINUTE)
    signals = emit(tracker, 2 * MINUTE)
    assert {"parent", "basis"}.issubset({s["arm"] for s in signals})
    assert all(s["earliest_execution_ms"] == 3 * MINUTE + 5000 for s in signals)
    tracker.observe(["PUBLIC"], 3 * MINUTE)
    assert not emit(tracker, 3 * MINUTE)


def test_gap_resets_authority_and_checkpoint_preserves_consumed_activity():
    tracker = Activities()
    tracker.observe([], MINUTE)
    tracker.observe(["PUBLIC"], 2 * MINUTE)
    emit(tracker, 2 * MINUTE)
    restored = Activities.restore(json.loads(json.dumps(tracker.state())))
    for value in (tracker, restored):
        value.observe(["PUBLIC"], 3 * MINUTE)
        assert not emit(value, 3 * MINUTE)
        value.gap()
        value.observe(["PUBLIC"], 5 * MINUTE)
        assert not emit(value, 5 * MINUTE)
        value.observe([], 6 * MINUTE)
        value.observe(["PUBLIC"], 7 * MINUTE)
        assert {"parent", "basis"}.issubset({s["arm"] for s in emit(value, 7 * MINUTE)})
    assert tracker.state() == restored.state()


def test_missing_index_does_not_veto_the_positioning_only_parent():
    tracker = Activities()
    tracker.observe([], MINUTE)
    tracker.observe(["PUBLIC"], 2 * MINUTE)
    signals = tracker.emit(
        {"PUBLIC": {"positioning_pass": True, "ready": False, "reason": "INDEX_MISSING"}},
        frame(),
        closing=2 * MINUTE,
        decision_ms=2 * MINUTE + 5000,
    )
    assert "parent" in {s["arm"] for s in signals} and "basis" not in {s["arm"] for s in signals}


def test_durable_journal_decision_prefix_has_a_reproducible_hash_chain(tmp_path):
    journal = Journal(tmp_path)
    journal.add({"public": "first"})
    prefix = journal.seal()
    journal.add({"public": "second"})
    journal.close()
    rows = [
        json.loads(line) for line in (tmp_path / "public_events.jsonl").read_bytes().splitlines()
    ]
    assert rows[0]["previous_sha256"] is None
    assert rows[1]["previous_sha256"] == digest(rows[0]) == prefix["event_chain_sha256"]
    assert prefix["event_count"] == 1


def test_terminal_connection_failure_is_logged_before_a_new_generation(monkeypatch, tmp_path):
    calls = []
    original_sleep = asyncio.sleep

    async def connect(symbols, pending, journal, stop, status):
        calls.append(status["connection_generation"])
        if len(calls) == 1:
            status["connection_error"] = "ConnectionClosedError"
        else:
            status["subscription_confirmed"] = True
            stop.set()

    async def no_wait(seconds):
        await original_sleep(0)

    async def run():
        journal, status, stop = Journal(tmp_path), {"error": None}, asyncio.Event()
        await module.stream(["PUBLIC"], [], journal, stop, status)
        journal.close()
        return status

    monkeypatch.setattr(module, "connection", connect)
    monkeypatch.setattr(module.asyncio, "sleep", no_wait)
    status = asyncio.run(run())
    assert calls == [1, 2] and status["error"] is None
    assert len(status["transient_disconnects"]) == 1
    assert (
        json.loads((tmp_path / "public_events.jsonl").read_text())["record"]["kind"]
        == "websocket_disconnect"
    )


def test_keepalive_uses_sparse_empty_control_frames_without_market_observations(tmp_path):
    class Socket:
        def __init__(self):
            self.payloads = []

        async def pong(self, payload):
            self.payloads.append(payload)

    async def run():
        journal, socket, status = Journal(tmp_path), Socket(), {"connection_generation": 1}
        tick = await module.keepalive(socket, journal, status, 0, 19)
        assert tick == 0 and socket.payloads == []
        tick = await module.keepalive(socket, journal, status, tick, 20)
        tick = await module.keepalive(socket, journal, status, tick, 21)
        await module.keepalive(socket, journal, status, tick, 40)
        journal.close()
        assert socket.payloads == [b"", b""] and status["keepalive_pongs"] == 2

    asyncio.run(run())
    records = [
        json.loads(line)["record"]
        for line in (tmp_path / "public_events.jsonl").read_text().splitlines()
    ]
    assert all(
        r["kind"] == "public_websocket_keepalive" and not r["market_observation"] for r in records
    )
