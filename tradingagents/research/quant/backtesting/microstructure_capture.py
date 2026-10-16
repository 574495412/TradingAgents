"""独立采集固定全集的同期盘口和聚合成交，不写入或推进交易账户。"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import shutil
import signal
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path

from tradingagents.dataflows.research.live_capture import body, digest
from tradingagents.dataflows.research.public_http import PublicHTTP
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.microstructure import VERSION, parse_message
from tradingagents.research.quant.backtesting.paper_store import atomic
from tradingagents.research.quant.backtesting.raw_chain import VERSION as RAW_VERSION
from tradingagents.research.quant.backtesting.runtime_inputs import received_clock
from tradingagents.research.quant.backtesting.segmented_chain import FORMAT, SegmentedChain
from tradingagents.research.quant.paths import PROJECT_ROOT, source_path
from tradingagents.research.quant.strategy_audit import read, save, sha

ROUTES = {
    "quote": ("wss://fstream.binance.com/public/stream", "@bookTicker"),
    "aggregate_trade": ("wss://fstream.binance.com/market/stream", "@aggTrade"),
}


def milliseconds():
    return time.time_ns() // 1_000_000


def market_record(raw, role, generation, state, clock, ending, received):
    """接收包装在原文入链之前固定；后来恢复时钟不能改写早先的已知状态。"""
    record = received_record(raw, received_ms=received)
    record.update(
        kind="observer_message",
        source_role=role,
        generation=generation,
        subscription_confirmed_ms=state.get("confirmed_ms"),
        clock_verified=clock["verified"] and 0 <= received - clock.get("received_ms", 0) <= 120_000,
        clock_record_sha256=clock.get("record_sha256"),
        after_capture_end=received >= ending,
    )
    return record


async def heartbeat(socket, journal, role, generation, local_stop, state):
    """独立等待PONG确认，静默市场不能靠虚构行情证明连接健康。"""
    while not local_stop.is_set():
        try:
            await asyncio.wait_for(local_stop.wait(), timeout=20)
            return
        except TimeoutError:
            pass
        sent = milliseconds()
        journal.add(
            {
                "kind": "observer_ping_sent",
                "received_ms": sent,
                "source_role": role,
                "generation": generation,
            }
        )
        try:
            pong = await socket.ping(b"observer")
            await asyncio.wait_for(pong, timeout=10)
        except Exception:
            state["heartbeat_failed"] = True
            with suppress(Exception):
                await socket.close()
            return
        received = milliseconds()
        state["pong_acks"] += 1
        journal.add(
            {
                "kind": "observer_pong_ack",
                "received_ms": received,
                "sent_ms": sent,
                "source_role": role,
                "generation": generation,
            }
        )


async def connection(role, symbols, journal, stop, clock, ending, generation, state):
    """只订阅预定公开流，保留回应、市场消息和失败原文。"""
    from websockets.asyncio.client import connect

    url, suffix = ROUTES[role]
    local_stop, beat = asyncio.Event(), None
    try:
        async with connect(
            url,
            open_timeout=15,
            close_timeout=3,
            ping_interval=None,
            max_size=4_194_304,
            max_queue=128,
        ) as socket:
            awaiting = set()
            for offset in range(0, len(symbols), 100):
                identity = f"observer-{role}-{offset // 100}"
                awaiting.add(identity)
                request = {
                    "method": "SUBSCRIBE",
                    "params": [s.lower() + suffix for s in symbols[offset : offset + 100]],
                    "id": identity,
                }
                journal.add(
                    {
                        "kind": "observer_subscription_request",
                        "received_ms": milliseconds(),
                        "source_role": role,
                        "generation": generation,
                        "request": request,
                    }
                )
                await socket.send(json.dumps(request))
                await asyncio.sleep(0.3)
            confirmation_deadline = milliseconds() + 15_000
            beat = asyncio.create_task(
                heartbeat(socket, journal, role, generation, local_stop, state)
            )
            while not stop.is_set() and milliseconds() < ending:
                if awaiting and milliseconds() >= confirmation_deadline:
                    raise TimeoutError("诊断订阅未在十五秒内全部确认")
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=1)
                except TimeoutError:
                    continue
                received = milliseconds()
                record = market_record(raw, role, generation, state, clock, ending, received)
                journal.add(record)
                if received >= ending:
                    state["after_end_messages"] += 1
                    break
                value = body(record)
                if isinstance(value, dict) and "id" in value:
                    if value.get("result", "missing") is not None or value["id"] not in awaiting:
                        raise ValueError("诊断订阅没有唯一成功回应")
                    awaiting.remove(value["id"])
                    if not awaiting:
                        state["confirmed_ms"] = received
                        journal.add(
                            {
                                "kind": "observer_subscription_confirmed",
                                "received_ms": received,
                                "source_role": role,
                                "generation": generation,
                                "members": len(symbols),
                            }
                        )
                    continue
                try:
                    parsed = parse_message(record, symbols)
                except (ValueError, KeyError, TypeError) as exc:
                    journal.add(
                        {
                            "kind": "observer_payload_rejected",
                            "received_ms": received,
                            "source_role": role,
                            "generation": generation,
                            "record_sha256": digest(record),
                            "error": str(exc),
                        }
                    )
                    raise ValueError("诊断市场消息格式异常，原文已保留") from exc
                state["market_messages"] += 1
                state["clock_verified_messages"] += parsed["clock_verified"]
                state["ready_messages"] += parsed["ready"]
                state["observed_symbols"].add(parsed["symbol"])
    finally:
        local_stop.set()
        if beat is not None:
            await beat


async def source_loop(role, symbols, journal, stop, clock, ending, session, shared):
    """连续四次不健康连接后停止；每次重连形成新诊断区间，不能填旧缺口。"""
    attempt, failures = 0, 0
    totals = {
        "connections": 0,
        "market_messages": 0,
        "clock_verified_messages": 0,
        "ready_messages": 0,
        "pong_acks": 0,
        "after_end_messages": 0,
    }
    observed = set()
    while not stop.is_set() and milliseconds() < ending:
        attempt += 1
        generation, started = f"{session}:{role}:{attempt}", milliseconds()
        state = {
            "confirmed_ms": None,
            "heartbeat_failed": False,
            "observed_symbols": set(),
            **dict.fromkeys(totals, 0),
        }
        journal.add(
            {
                "kind": "observer_connection_started",
                "received_ms": started,
                "source_role": role,
                "generation": generation,
                "url": ROUTES[role][0],
            }
        )
        error, fatal, close = None, False, None
        try:
            await connection(role, symbols, journal, stop, clock, ending, generation, state)
            if state["heartbeat_failed"]:
                error = "PONG_TIMEOUT"
        except Exception as exc:
            error = type(exc).__name__
            fatal = isinstance(exc, (ValueError, KeyError, TypeError))
            close = {
                side: getattr(getattr(exc, side, None), "code", None) for side in ("rcvd", "sent")
            }
            fatal = fatal or close["rcvd"] in {1002, 1007, 1008}
        ended = milliseconds()
        journal.add(
            {
                "kind": "observer_connection_ended",
                "received_ms": ended,
                "source_role": role,
                "generation": generation,
                "error": error,
                "close": close,
                "registered_end_reached": ended >= ending,
            }
        )
        totals["connections"] += 1
        for key in totals.keys() - {"connections"}:
            totals[key] += state[key]
        observed.update(state["observed_symbols"])
        shared[role] = {**totals, "observed_symbols": sorted(observed)}
        if stop.is_set() or ended >= ending:
            return
        healthy = (
            state["confirmed_ms"] is not None
            and ended - state["confirmed_ms"] >= 60_000
            and state["pong_acks"] >= 2
        )
        failures = 1 if healthy else failures + 1
        if fatal or failures >= 4:
            shared["failure"] = {
                "source_role": role,
                "error": error or "CONNECTION_ENDED_WITHOUT_END",
                "consecutive_unhealthy": failures,
            }
            stop.set()
            return
        await asyncio.sleep(min(2 ** (failures - 1), 8))


async def capture(protocol, journal, client, session):
    """公开时钟与来源分开运行，接收时未知的时钟状态从不事后回填。"""
    stop, clock, status = asyncio.Event(), {"verified": False}, {"failure": None}
    loop, tasks, ending = asyncio.get_running_loop(), [], protocol["capture_end_ms"]

    def requested_stop(name):
        received = milliseconds()
        status["stop_request"] = {"signal": name, "received_ms": received}
        if received < ending:
            status["failure"] = {
                "error": "STOP_REQUESTED_BEFORE_REGISTERED_END",
                **status["stop_request"],
            }
        journal.add({"kind": "observer_stop_requested", **status["stop_request"]})
        stop.set()

    for name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(name, requested_stop, name.name)
    try:
        first = await received_clock(client, journal, min(ending, milliseconds() + 30_000))
        clock.update(verified=True, received_ms=first["received_ms"], record_sha256=digest(first))
        tasks = [
            asyncio.create_task(
                source_loop(
                    role, protocol["symbols"], journal, stop, clock, ending, session, status
                )
            )
            for role in ROUTES
        ]
        next_clock, next_disk = milliseconds() + 60_000, milliseconds()
        while not stop.is_set() and milliseconds() < ending:
            now = milliseconds()
            for task in tasks:
                if task.done():
                    error = task.exception()
                    if error is not None:
                        raise RuntimeError("诊断来源任务异常退出") from error
                    if not stop.is_set() and milliseconds() < ending:
                        raise ValueError("诊断来源任务提前退出但未声明失败")
            if now >= next_disk:
                if shutil.disk_usage(journal.path).free < 2 * 1024**3:
                    raise ValueError("诊断原文磁盘余量低于二GiB")
                next_disk = now + 60_000
            if now >= next_clock:
                try:
                    value = await received_clock(client, journal, min(ending, now + 30_000))
                    clock.update(
                        verified=True, received_ms=value["received_ms"], record_sha256=digest(value)
                    )
                except Exception as exc:
                    clock["verified"] = False
                    journal.add(
                        {
                            "kind": "observer_clock_unknown",
                            "received_ms": milliseconds(),
                            "error": str(exc),
                        }
                    )
                journal.seal()
                next_clock = milliseconds() + 60_000
            await asyncio.sleep(0.1)
    except Exception as exc:
        status["failure"] = {"error": type(exc).__name__, "message": str(exc)}
        journal.add(
            {"kind": "observer_failure", "received_ms": milliseconds(), **status["failure"]}
        )
    finally:
        stop.set()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException) and status["failure"] is None:
                status["failure"] = {
                    "error": "SOURCE_TASK_FAILURE",
                    "message": type(outcome).__name__,
                }
                journal.add(
                    {"kind": "observer_failure", "received_ms": milliseconds(), **status["failure"]}
                )
        for name in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(name)
    status.update(capture_end_reached=milliseconds() >= ending, stopped_ms=milliseconds())
    return status


def run(study, output, duration=None):
    """只读关联研究协议，另建不可覆盖的原文目录，账户完全不受观察器控制。"""
    parent = read(study / "protocol.json")
    symbols = parent["symbols"]
    if (
        parent.get("orders_enabled") is not False
        or len(symbols) != 220
        or symbols != sorted(set(symbols))
    ):
        raise ValueError("观察器仅关联固定220合约的禁用订单研究")
    now = milliseconds()
    ending = (
        min(parent["simulation_end_ms"] + 180_000, now + duration * 1000)
        if duration is not None
        else parent["simulation_end_ms"] + 180_000
    )
    if not 180_000 <= ending - now <= 86_400_000 or (
        duration is not None and not 180 <= duration <= 3600
    ):
        raise ValueError("有限观察限三至六十分钟，关联研究观察最长二十四小时")
    if shutil.disk_usage(study).free < 10 * 1024**3:
        raise ValueError("观察器启动前磁盘余量须至少十GiB")
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if name == "tradingagents" or name.startswith("tradingagents.")
        if getattr(module, "__file__", None) and Path(module.__file__).suffix == ".py"
    }
    paths.add(Path(__file__).resolve())
    plan = source_path("profit_factor_microstructure_plan.md")
    paths.add(plan)
    package = PROJECT_ROOT
    protocol = {
        "schema": "cz-public-microstructure-observer.v1",
        "diagnostic_version": VERSION,
        "symbols": symbols,
        "linked_study_protocol_sha256": sha(study / "protocol.json"),
        "capture_start_ms": now,
        "capture_end_ms": ending,
        "source_routes": ROUTES,
        "storage": FORMAT,
        "raw_validation": RAW_VERSION,
        "maximum_pair_age_ms": 5000,
        "trade_age_groups_ms": [5000, 60_000],
        "all_type_trade_age_proven": False,
        "source_sha256": {str(p): sha(p) for p in paths},
        "predeclared_plan_sha256": sha(plan),
        "runtime": {
            "python": sys.version,
            **{name: importlib.metadata.version(name) for name in ("websockets", "requests")},
        },
        "account_state_read_or_written": False,
        "can_emit_trade_signals": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "protocol.json", protocol)
    source = output / "source"
    source.mkdir()
    for path in paths:
        target = source / path.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    session = {"id": uuid.uuid4().hex, "pid": os.getpid(), "started_ms": milliseconds()}
    atomic(output / "current_process.json", session)
    journal, client = SegmentedChain(output / "public_events"), PublicHTTP()
    try:
        result = asyncio.run(capture(protocol, journal, client, session["id"]))
        if any(sha(path) != protocol["source_sha256"][str(path)] for path in paths):
            raise ValueError("观察器运行源码发生变化")
        result.update(
            source_prefix=journal.seal(),
            strategy_profitability_evaluated=False,
            account_state_read_or_written=False,
            orders_enabled=False,
            production_eligible=False,
        )
        save(output / "result.json", result)
    finally:
        client.close()
        journal.close()
    save(
        output / "receipts.json",
        {
            str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    print({k: v for k, v in result.items() if k not in ROUTES}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int)
    args = parser.parse_args()
    result = run(args.study.resolve(), args.output.resolve(), args.duration_seconds)
    if result["failure"] or not result["capture_end_reached"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
