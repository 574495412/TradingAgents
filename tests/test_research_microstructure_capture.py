"""独立观察器的原文、连接边界与有限重试反证。"""

import asyncio
import json

import pytest

from tradingagents.dataflows.research.live_capture import body
from tradingagents.research.quant.backtesting import microstructure_capture as module


class Journal:
    def __init__(self):
        self.records = []

    def add(self, record):
        self.records.append(record)

    def seal(self):
        return {}


def test_unknown_clock_is_fixed_at_receipt_and_cannot_be_repaired_later():
    clock = {"verified": False, "received_ms": 900, "record_sha256": "旧时钟"}
    record = module.market_record(b"not-json", "quote", "来源一", {}, clock, 2000, 1000)
    clock.update(verified=True, received_ms=1100, record_sha256="后来时钟")
    assert record["clock_verified"] is False and record["clock_record_sha256"] == "旧时钟"
    assert record["body_base64"] == "bm90LWpzb24="
    expired = module.market_record(b"{}", "quote", "来源一", {}, clock, 200_000, 122_000)
    assert expired["clock_verified"] is False


@pytest.mark.parametrize("failure", ["none", "duplicate_ack", "invalid_body", "no_ack"])
def test_full_subscription_acknowledgement_and_end_boundary_preserve_originals(
    monkeypatch, failure
):
    from websockets.asyncio import client

    now, stop, journal, sent = [100_000], asyncio.Event(), Journal(), []
    ending = 300_000
    symbols = [f"A{k:03}USDT" for k in range(220)]
    state = {
        "confirmed_ms": None,
        "pong_acks": 0,
        "after_end_messages": 0,
        "market_messages": 0,
        "clock_verified_messages": 0,
        "ready_messages": 0,
        "observed_symbols": set(),
    }

    class Socket:
        def __init__(self):
            self.incoming = []

        async def send(self, raw):
            request = json.loads(raw)
            sent.append(request)
            identity = "observer-quote-0" if failure == "duplicate_ack" else request["id"]
            self.incoming.append(json.dumps({"result": None, "id": identity}))

        async def recv(self):
            now[0] += 1
            if failure == "no_ack":
                now[0] += 16_000
                raise TimeoutError
            if self.incoming:
                return self.incoming.pop(0)
            if failure == "invalid_body":
                return b"<html>invalid</html>"
            now[0] = ending + 1
            return b"late-original"

    socket = Socket()

    class Connection:
        async def __aenter__(self):
            return socket

        async def __aexit__(self, *args):
            return False

    async def beat(*args):
        await args[-2].wait()

    async def sleep(seconds):
        now[0] += int(seconds * 1000)

    monkeypatch.setattr(client, "connect", lambda *a, **kw: Connection())
    monkeypatch.setattr(module, "heartbeat", beat)
    monkeypatch.setattr(module, "milliseconds", lambda: now[0])
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    work = module.connection(
        "quote",
        symbols,
        journal,
        stop,
        {"verified": True, "received_ms": 100_000},
        ending,
        "来源一",
        state,
    )
    if failure == "none":
        asyncio.run(work)
        assert state["after_end_messages"] == 1 and state["market_messages"] == 0
        assert journal.records[-1]["after_capture_end"] is True
    else:
        with pytest.raises(TimeoutError if failure == "no_ack" else ValueError):
            asyncio.run(work)
    assert [len(v["params"]) for v in sent] == [100, 100, 20]
    assert len({v["id"] for v in sent}) == 3
    assert all(v["method"] == "SUBSCRIBE" for v in sent)
    assert (state["confirmed_ms"] is not None) == (failure in {"none", "invalid_body"})
    originals = [v for v in journal.records if v["kind"] == "observer_message"]
    if failure == "invalid_body":
        assert originals[-1]["body_base64"] == "PGh0bWw+aW52YWxpZDwvaHRtbD4="
        with pytest.raises(ValueError):
            body(originals[-1])


@pytest.mark.parametrize("healthy_middle", [False, True])
def test_four_consecutive_unhealthy_connections_stop_but_a_healthy_connection_resets_budget(
    monkeypatch, healthy_middle
):
    now, journal, stop, shared, calls = [100_000], Journal(), asyncio.Event(), {}, []

    async def connection(role, symbols, journal, stop, clock, ending, generation, state):
        calls.append(generation)
        if healthy_middle and len(calls) == 4:
            state.update(confirmed_ms=now[0], pong_acks=3)
            now[0] += 61_000
        else:
            now[0] += 100
        raise OSError("人工断连")

    async def sleep(seconds):
        now[0] += int(seconds * 1000)

    monkeypatch.setattr(module, "milliseconds", lambda: now[0])
    monkeypatch.setattr(module, "connection", connection)
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    asyncio.run(
        module.source_loop("quote", ["AUSDT"], journal, stop, {}, 1_000_000, "会话", shared)
    )
    assert stop.is_set() and shared["failure"]["consecutive_unhealthy"] == 4
    assert len(calls) == (7 if healthy_middle else 4)
    assert len(set(calls)) == len(calls)
    assert shared["quote"]["connections"] == len(calls)
    assert len([v for v in journal.records if v["kind"] == "observer_connection_ended"]) == len(
        calls
    )


@pytest.mark.parametrize("failure", ["source_task", "signal"])
def test_unexpected_source_task_or_early_signal_cannot_silently_run_to_a_successful_end(
    monkeypatch, tmp_path, failure
):
    handlers, journal = {}, Journal()
    journal.path = tmp_path
    original_sleep = asyncio.sleep

    async def clock(client, journal, deadline):
        if failure == "signal":
            callback, args = handlers[module.signal.SIGINT]
            callback(*args)
        return {"received_ms": 100_000}

    async def source(role, symbols, journal, stop, clock, ending, session, status):
        if role == "quote" and failure == "source_task":
            raise OSError("人工写入故障")
        await stop.wait()

    async def sleep(seconds):
        await original_sleep(0)

    async def work():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(
            loop,
            "add_signal_handler",
            lambda name, callback, *args: handlers.update({name: (callback, args)}),
        )
        monkeypatch.setattr(loop, "remove_signal_handler", lambda name: True)
        return await module.capture(
            {"capture_end_ms": 200_000, "symbols": ["AUSDT"]}, journal, object(), "人工会话"
        )

    monkeypatch.setattr(module, "milliseconds", lambda: 100_000)
    monkeypatch.setattr(module, "received_clock", clock)
    monkeypatch.setattr(module, "source_loop", source)
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    result = asyncio.run(work())
    assert result["failure"] is not None and result["capture_end_reached"] is False
    if failure == "signal":
        assert result["failure"]["error"] == "STOP_REQUESTED_BEFORE_REGISTERED_END"
        assert any(v["kind"] == "observer_stop_requested" for v in journal.records)
    else:
        assert result["failure"]["error"] == "RuntimeError"
        assert any(v["kind"] == "observer_failure" for v in journal.records)
