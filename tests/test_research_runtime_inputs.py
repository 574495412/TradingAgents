"""反证有限时钟重试的原文、成功条件和原决策期限。"""

import asyncio

import pytest

from tests.test_research_paper_account import public
from tradingagents.dataflows.research.live_capture import clock_valid
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting import runtime_inputs as module


@pytest.mark.parametrize(
    "mode",
    [
        "late_762ms",
        "already_known",
        "never",
        "original_deadline",
        "clock_backwards",
        "late_after_limit",
    ],
)
def test_owned_mark_wait_is_bounded_and_never_uses_future_or_previous_minute(monkeypatch, mode):
    closing = 6_000_000
    now, elapsed, records = [closing + 2258], [0.0], []
    arrival = closing + (
        0
        if mode == "already_known"
        else 3020
        if mode == "late_762ms"
        else 7259
        if mode == "late_after_limit"
        else 100_000
    )
    deadline = now[0] + (200 if mode == "original_deadline" else 30_000)
    original_sleep = asyncio.sleep

    class Bars:
        def at(self, symbol, boundary, decision):
            assert symbol == "A" and boundary == closing
            return {"row": ["当前分钟"], "received_ms": arrival} if decision >= arrival else None

    class Journal:
        def add(self, record):
            records.append(record)

    async def sleep(seconds):
        step = int(round(seconds * 1000))
        elapsed[0] += step
        now[0] += -100 if mode == "clock_backwards" else step
        if mode == "late_after_limit":
            elapsed[0] = 5001
            now[0] = arrival
        await original_sleep(0)

    monkeypatch.setattr(module, "milliseconds", lambda: now[0])
    monkeypatch.setattr(module, "monotonic_ms", lambda: elapsed[0])
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    result = asyncio.run(module.wait_for_owned_marks(Bars(), ["A"], closing, Journal(), deadline))
    assert result["ready"] == (mode in {"late_762ms", "already_known"})
    assert result["original_deadline_ms"] == deadline
    assert records[0]["maximum_wait_ms"] == 5000 and records[0]["wait_id"] == result["wait_id"]
    if mode == "late_762ms":
        assert 762 <= result["elapsed_monotonic_ms"] <= 812
        assert result["received_ms"] >= arrival
    elif mode == "already_known":
        assert result["elapsed_monotonic_ms"] == 0
    elif mode == "original_deadline":
        assert (
            result["reason"] == "ORIGINAL_DEADLINE_REACHED"
            and result["elapsed_monotonic_ms"] <= 200
        )
    elif mode == "clock_backwards":
        assert result["reason"] == "LOCAL_CLOCK_MOVED_BACKWARDS"
    else:
        assert result["ready"] is False
        assert result["reason"] == "OWNED_MARK_WAIT_LIMIT_REACHED"


@pytest.mark.parametrize("arrives", [False, True])
def test_exit_quote_is_requested_after_current_owned_mark_wait_and_timeout_keeps_unknown(
    monkeypatch, arrives
):
    from types import SimpleNamespace

    from tests.test_research_paper_account import frame
    from tradingagents.research.quant.backtesting import paper_live as live

    closing, now, elapsed, records = 6_000_000, [6_002_258], [0.0], []
    arrival = closing + (3020 if arrives else 100_000)
    original_sleep = asyncio.sleep
    point = frame(100)["marks"]["A"] | {"received_ms": arrival}

    class Bars:
        values = {"A": {}}

        def at(self, symbol, boundary, decision):
            assert boundary == closing
            return point if decision >= arrival else None

    class Journal:
        def add(self, record):
            records.append(record)

    class Client:
        def batch(self, jobs):
            return []

        def request(self, kind, symbol, params):
            started = now[0]
            now[0] += 50
            elapsed[0] += 50
            value = (
                {"serverTime": started + 25}
                if kind == "clock"
                else [
                    {
                        "symbol": "A",
                        "bidPrice": "100",
                        "askPrice": "100",
                        "bidQty": "10",
                        "askQty": "10",
                        "time": started,
                    }
                ]
            )
            return public(kind, value, started, now[0])

    async def sleep(seconds):
        step = int(round(seconds * 1000))
        now[0] += step
        elapsed[0] += step
        await original_sleep(0)

    store = SimpleNamespace(
        protocol={
            "symbols": ["A"],
            "simulation_end_ms": closing + 180_000,
            "account_preparation_version": module.OWNED_MARK_WAIT_VERSION,
        },
        accounts={
            "3": SimpleNamespace(
                positions={"A": {"symbol": "A", "entry_ms": closing - 1000}}, trades=[]
            )
        },
        source=Journal(),
    )
    monkeypatch.setattr(module, "milliseconds", lambda: now[0])
    monkeypatch.setattr(module, "monotonic_ms", lambda: elapsed[0])
    monkeypatch.setattr(live, "milliseconds", lambda: now[0])
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    result, _ = asyncio.run(
        live.capture_frame(
            Client(), store, Bars(), frame(100)["contracts"], closing, "exit_only", "人工来源"
        )
    )
    preparation = result["owned_mark_preparation"]
    assert preparation["ready"] == arrives
    assert result["quotes"]["requested_ms"] >= preparation["received_ms"]
    assert ("A" in result["marks"]) == arrives
    if arrives:
        assert result["quotes"]["requested_ms"] >= arrival
    else:
        assert preparation["reason"] == "OWNED_MARK_WAIT_LIMIT_REACHED" and result["marks"] == {}


@pytest.mark.parametrize(
    "failure", [0, 503, "persistent", "invalid_success", "late", 429, "expired"]
)
def test_retry_only_recovers_temporary_errors_and_never_moves_the_deadline(monkeypatch, failure):
    now, calls, records = [1_000_000], [], []
    deadline = now[0] + 60_000
    original_sleep = asyncio.sleep

    class Client:
        def request(self, kind, symbol, params):
            calls.append(kind)
            requested = now[0]
            now[0] += 100
            if failure == "late":
                now[0] = deadline + 1
            status = failure if isinstance(failure, int) else 0
            if failure in (0, 503) and len(calls) > 1:
                status = 200
            if failure in {"invalid_success", "late"}:
                status = 200
            if status == 200:
                return public(
                    "clock",
                    {"serverTime": 0 if failure == "invalid_success" else now[0] - 20},
                    requested,
                    now[0],
                )
            return received_record(b"", received_ms=now[0]) | {
                "kind": "clock",
                "status": status,
                "requested_ms": requested,
                "local_clock_consistent": True,
            }

    class Journal:
        def add(self, record):
            records.append(record)

    async def sleep(seconds):
        now[0] += int(seconds * 1000)
        await original_sleep(0)

    monkeypatch.setattr(module, "milliseconds", lambda: now[0])
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    if failure == "expired":
        now[0] = deadline
    if failure in (0, 503):
        result = asyncio.run(module.received_clock(Client(), Journal(), deadline))
        assert clock_valid(result) and len(calls) == 2
        assert records[0]["body_base64"] == "" and records[0]["status"] == failure
        assert records[1]["kind"] == "public_clock_retry" and records[1]["deadline_ms"] == deadline
    else:
        with pytest.raises(ValueError):
            asyncio.run(module.received_clock(Client(), Journal(), deadline))
        assert len(calls) == (3 if failure == "persistent" else 0 if failure == "expired" else 1)
