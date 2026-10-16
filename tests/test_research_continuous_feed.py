"""人工传输验证双路逐点交接和有限重连，不代表市场收益。"""

import asyncio
import copy
from types import SimpleNamespace

import pytest

from tradingagents.dataflows.research.live_capture import MINUTE, canonical
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting import continuous_feed as module
from tradingagents.research.quant.backtesting.paper_live_audit import verify_handoffs
from tradingagents.research.quant.backtesting.paper_store import Chain, entries
from tradingagents.research.quant.backtesting.source_cursor import SourceCursor


class Journal:
    def __init__(self):
        self.records = []

    def add(self, record):
        self.records.append(record)


def candle(symbol, closing, generation, received, price="100"):
    return received_record(
        canonical(
            {
                "e": "kline",
                "E": closing + 100,
                "s": symbol,
                "k": {
                    "s": symbol,
                    "i": "1m",
                    "t": closing - MINUTE,
                    "T": closing - 1,
                    "o": "100",
                    "h": "102",
                    "l": "99",
                    "c": price,
                    "v": "1",
                    "q": "100",
                    "n": 1,
                    "V": "0.5",
                    "Q": "50",
                    "x": True,
                },
            }
        ),
        received_ms=received,
    ) | {"connection_generation": generation}


@pytest.mark.parametrize(
    "failure", [None, "incomplete", "different", "forged_receipt", "false_start"]
)
def test_handoff_needs_both_full_matching_minute_originals_and_real_receipts(tmp_path, failure):
    journal, symbols = Journal(), ["A", "B"]
    old, new = module.Tap(journal, symbols), module.Tap(journal, symbols)
    journal.add(
        {"kind": "public_stream_connection_started", "received_ms": 0, "generation": "session:1"}
    )
    journal.add(
        {
            "kind": "public_stream_connection_started",
            "received_ms": MINUTE,
            "generation": "session:2",
        }
    )
    for symbol in symbols:
        old.add(candle(symbol, 2 * MINUTE, "session:1", 2 * MINUTE + 200))
        if failure != "incomplete" or symbol == "A":
            new.add(
                candle(
                    symbol,
                    2 * MINUTE,
                    "session:2",
                    2 * MINUTE + 300,
                    "101" if failure == "different" else "100",
                )
            )
    for tap in (old, new):
        tap.refresh(2 * MINUTE + 2000)
    if failure == "different":
        with pytest.raises(ValueError, match="终值冲突"):
            old.overlap(new, "session:1", "session:2", 2 * MINUTE + 2000, MINUTE)
        return
    proof = old.overlap(new, "session:1", "session:2", 2 * MINUTE + 2000, MINUTE)
    if failure == "incomplete":
        assert proof is None
        return
    if failure == "forged_receipt":
        proof["members"][0]["new_received_ms"] -= 1
    elif failure == "false_start":
        proof["new_connection_started_ms"] -= 1
    journal.add(proof)
    raw = Chain(tmp_path / "raw.jsonl")
    for record in journal.records:
        raw.add(record)
    prefix = raw.seal()
    raw.close()
    cursor = SourceCursor(entries(raw.path), symbols, tmp_path / "members.db")
    try:
        if failure:
            with pytest.raises(ValueError):
                verify_handoffs(journal.records, symbols)
            with pytest.raises(ValueError):
                cursor.advance(prefix, now=2 * MINUTE + 3000, closing=2 * MINUTE)
        else:
            assert verify_handoffs(journal.records, symbols) == 1
            cursor.advance(prefix, now=2 * MINUTE + 3000, closing=2 * MINUTE)
            assert cursor.handoffs == 1
            assert (
                cursor.bars.at("A", 2 * MINUTE, 2 * MINUTE + 3000)["received_ms"]
                == 2 * MINUTE + 200
            )
    finally:
        cursor.close()


@pytest.mark.parametrize("mode", ["persistent_failure", "healthy_reconnects", "planned_overlap"])
def test_supervisor_bounds_outages_resets_after_health_and_proves_rotation(monkeypatch, mode):
    elapsed, calls, journal = [0.0], [], Journal()
    shared, stop = {"error": None}, asyncio.Event()
    original_sleep = asyncio.sleep
    base = 100 * MINUTE

    def now():
        return base + int(elapsed[0] * 1000)

    async def sleep(seconds):
        elapsed[0] += seconds
        await original_sleep(0)

    async def connection(symbols, pending, tap, wire_stop, status):
        identity = status["connection_generation"]
        number = int(identity.rpartition(":")[2])
        calls.append(number)
        if mode == "persistent_failure":
            status["connection_error"] = "ConnectionClosedError"
            return
        status["subscription_confirmed"] = True
        emitted = None
        while not wire_stop.is_set() and not stop.is_set():
            closing = (now() - 1001) // MINUTE * MINUTE
            if (
                closing >= (status["connection_started_ms"] // MINUTE + 1) * MINUTE
                and closing != emitted
            ):
                emitted = closing
                for symbol in symbols:
                    record = candle(symbol, closing, identity, now())
                    tap.add(record)
                    pending.append(record)
                    status["closed_messages"] += 1
                if mode == "healthy_reconnects":
                    if number == 6:
                        stop.set()
                    else:
                        status["connection_error"] = "ConnectionClosedError"
                    return
            if mode == "planned_overlap" and shared.get("planned_rotations"):
                stop.set()
            await original_sleep(0)

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    monkeypatch.setattr(module, "now_ms", now)
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    monkeypatch.setattr(module, "connection", connection)
    asyncio.run(
        module.continuous_stream(
            ["A", "B"], [], journal, stop, shared, rotation_seconds=180, generation_prefix="fixture"
        )
    )
    if mode == "persistent_failure":
        assert calls == [1, 2, 3, 4] and "连续四次" in shared["error"]
    elif mode == "healthy_reconnects":
        assert calls == [1, 2, 3, 4, 5, 6] and shared["error"] is None
    else:
        assert calls == [1, 2] and shared["error"] is None
        assert len(shared["planned_rotations"]) == verify_handoffs(journal.records, ["A", "B"]) == 1
        broken = copy.deepcopy(journal.records)
        event = next(r for r in broken if r["kind"] == "public_stream_overlap_handoff")
        event["verified_members"] = 1
        with pytest.raises(ValueError):
            verify_handoffs(broken, ["A", "B"])
