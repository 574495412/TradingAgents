"""连续公开收盘采样：有限故障重试与双连接完整分钟交接。"""

from __future__ import annotations

import asyncio
import time
import uuid

from tradingagents.dataflows.research.live_capture import MINUTE, body
from tradingagents.dataflows.research.stream import closed_candle, unwrap
from tradingagents.research.quant.backtesting.forward_inputs import normalized_row
from tradingagents.research.quant.backtesting.forward_pilot import connection

VERSION = "continuous_completed_minutes_v1"
ROTATION_SECONDS = 23 * 60 * 60


def now_ms():
    return time.time_ns() // 1_000_000


def verify_handoff(event, originals, symbols, starts):
    """两份完整原文逐成员相同且交接时已收到，不能只相信成功标志。"""
    if (
        event["verified_members"] != len(symbols)
        or [v["symbol"] for v in event["members"]] != sorted(symbols)
        or event["from_generation"] == event["to_generation"]
        or event["received_ms"] < event["closing_ms"] + 1000
        or starts.get(event["to_generation"]) != event["new_connection_started_ms"]
        or event["closing_ms"] < (event["new_connection_started_ms"] // MINUTE + 1) * MINUTE
    ):
        raise ValueError("连接交接缺少完整固定集合或真实时钟")
    for member in event["members"]:
        rows = []
        for side, generation in (
            ("old", event["from_generation"]),
            ("new", event["to_generation"]),
        ):
            key = member[side + "_body_sha256"], generation, member[side + "_received_ms"]
            record = originals.get(key)
            parsed = (
                closed_candle(record, decision_ms=event["received_ms"], clock_verified=True)
                if record
                else None
            )
            if (
                not parsed
                or parsed[0] != member["symbol"]
                or parsed[1][6] + 1 != event["closing_ms"]
            ):
                raise ValueError("连接交接引用的收盘原文不存在或尚未可用")
            rows.append(normalized_row(parsed[1]))
        if rows[0] != rows[1]:
            raise ValueError("连接交接的两份分钟终值不同")


class Tap:
    """转发全部原文，交接缓存仅保存已经成熟的最近三个完整分钟。"""

    def __init__(self, journal, symbols):
        self.journal, self.symbols = journal, set(symbols)
        self.waiting, self.rows = [], {}
        self.latest_complete_ms = None

    def add(self, record):
        self.journal.add(record)
        if record["kind"] == "websocket":
            value = unwrap(body(record))
            if isinstance(value, dict) and value.get("k", {}).get("x") is True:
                self.waiting.append(record)
                if len(self.waiting) > 4 * len(self.symbols):
                    raise ValueError("公开完成分钟等待队列超出固定容量")

    def refresh(self, actual_ms):
        remaining = []
        for record in self.waiting:
            parsed = closed_candle(record, decision_ms=actual_ms, clock_verified=True)
            if parsed is None:
                remaining.append(record)
                continue
            symbol, row = parsed
            if symbol not in self.symbols:
                raise ValueError("推送包含固定全集之外的成员")
            closing = row[6] + 1
            group = self.rows.setdefault(closing, {})
            previous = group.get(symbol)
            if previous and normalized_row(previous["row"]) != normalized_row(row):
                raise ValueError("同一连接的完成分钟发生终值冲突")
            if not previous or record["received_ms"] < previous["record"]["received_ms"]:
                group[symbol] = {"row": row, "record": record}
            if set(group) == self.symbols:
                self.latest_complete_ms = max(self.latest_complete_ms or 0, closing)
        self.waiting = remaining
        cutoff = (actual_ms - 1000) // MINUTE * MINUTE - 2 * MINUTE
        self.rows = {clock: rows for clock, rows in self.rows.items() if clock >= cutoff}

    def overlap(self, other, old_generation, new_generation, actual_ms, new_started_ms):
        for closing in sorted(set(self.rows) & set(other.rows), reverse=True):
            if closing < (new_started_ms // MINUTE + 1) * MINUTE:
                continue
            old, new = self.rows[closing], other.rows[closing]
            if set(old) != self.symbols or set(new) != self.symbols:
                continue
            members = []
            for symbol in sorted(self.symbols):
                if normalized_row(old[symbol]["row"]) != normalized_row(new[symbol]["row"]):
                    raise ValueError("重叠连接的完成分钟发生终值冲突")
                first, second = old[symbol]["record"], new[symbol]["record"]
                members.append(
                    {
                        "symbol": symbol,
                        "old_body_sha256": first["body_sha256"],
                        "old_received_ms": first["received_ms"],
                        "new_body_sha256": second["body_sha256"],
                        "new_received_ms": second["received_ms"],
                    }
                )
            return {
                "kind": "public_stream_overlap_handoff",
                "received_ms": actual_ms,
                "closing_ms": closing,
                "from_generation": old_generation,
                "to_generation": new_generation,
                "verified_members": len(members),
                "members": members,
                "new_connection_started_ms": new_started_ms,
            }
        return None


def invalid_protocol(status):
    return status.get("connection_error") in {
        "ValueError",
        "KeyError",
        "JSONDecodeError",
    } or status.get("last_close_details", {}).get("rcvd", {}).get("code") in {1002, 1007, 1008}


async def continuous_stream(
    symbols,
    pending,
    journal,
    stop,
    status,
    *,
    rotation_seconds=ROTATION_SECONDS,
    generation_prefix=None,
):
    """健康连接后的故障重新计有限重试；计划交接先证明双路完整分钟一致。"""
    if rotation_seconds < 180:
        raise ValueError("公开连接轮换间隔不得小于三分钟")
    wires, generation, failures = {}, 0, 0
    generation_prefix = generation_prefix or uuid.uuid4().hex
    primary = replacement = None
    retry_at = 0.0
    totals = dict.fromkeys(
        ("messages", "closed_messages", "unfinished_discarded", "keepalive_pongs"), 0
    )
    status.update(transient_disconnects=[], planned_rotations=[], runtime_version=VERSION)

    def start_wire():
        nonlocal generation
        generation += 1
        identity = f"{generation_prefix}:{generation}"
        wire_status = {
            "connection_generation": identity,
            "connection_started_ms": now_ms(),
            "connection_error": None,
            "subscription_confirmed": False,
            **dict.fromkeys(totals, 0),
        }
        wire_stop, tap = asyncio.Event(), Tap(journal, symbols)
        wire = {
            "status": wire_status,
            "stop": wire_stop,
            "tap": tap,
            "started_tick": time.monotonic(),
        }
        wire["task"] = asyncio.create_task(
            connection(symbols, pending, tap, wire_stop, wire_status)
        )
        wires[identity] = wire
        journal.add(
            {
                "kind": "public_stream_connection_started",
                "received_ms": wire_status["connection_started_ms"],
                "generation": identity,
            }
        )
        return identity

    def forget(identity):
        wire = wires.pop(identity)
        for key in totals:
            totals[key] += wire["status"].get(key, 0)

    try:
        primary = start_wire()
        while not stop.is_set():
            actual, tick = now_ms(), time.monotonic()
            for wire in wires.values():
                wire["tap"].refresh(actual)
            for identity in list(wires):
                wire = wires[identity]
                if not wire["task"].done():
                    if actual - wire["status"]["connection_started_ms"] > 180_000 and (
                        wire["tap"].latest_complete_ms is None
                        or actual - wire["tap"].latest_complete_ms > 180_000
                    ):
                        wire["status"]["connection_error"] = "COMPLETE_MINUTE_WATCHDOG_EXPIRED"
                        wire["stop"].set()
                    continue
                await wire["task"]
                if wire["stop"].is_set() and wire["status"]["connection_error"] is None:
                    forget(identity)
                    continue
                event = {
                    "kind": "websocket_disconnect",
                    "received_ms": actual,
                    "generation": identity,
                    "error": wire["status"]["connection_error"],
                    "close": wire["status"].get("last_close_details"),
                }
                status["transient_disconnects"].append(event)
                journal.add(event)
                if invalid_protocol(wire["status"]):
                    raise ValueError("公开连接协议或终值异常，不能用重试掩盖")
                was_replacement = identity == replacement
                healthy = wire["tap"].latest_complete_ms is not None
                failures = 1 if healthy else failures + 1
                forget(identity)
                if identity == primary:
                    primary, replacement = replacement, None
                elif identity == replacement:
                    replacement = None
                if failures >= 4:
                    if (
                        was_replacement
                        and primary is not None
                        and wires[primary]["tap"].latest_complete_ms is not None
                    ):
                        failures, retry_at = 0, tick + 60
                        journal.add(
                            {
                                "kind": "public_stream_rotation_retry_pause",
                                "received_ms": actual,
                                "cooldown_seconds": 60,
                            }
                        )
                        continue
                    raise ValueError("连续四次连接没有恢复完整分钟")
                retry_at = tick + 2 ** (failures - 1)
            if primary is None and tick >= retry_at:
                primary = start_wire()
            if primary is not None:
                wire = wires[primary]
                if (
                    replacement is None
                    and tick - wire["started_tick"] >= rotation_seconds
                    and tick >= retry_at
                ):
                    replacement = start_wire()
                if replacement is not None:
                    next_wire = wires[replacement]
                    proof = wire["tap"].overlap(
                        next_wire["tap"],
                        primary,
                        replacement,
                        actual,
                        next_wire["status"]["connection_started_ms"],
                    )
                    if proof is not None:
                        journal.add(proof)
                        status["planned_rotations"].append(
                            {k: v for k, v in proof.items() if k != "members"}
                        )
                        wire["stop"].set()
                        primary, replacement = replacement, None
                        failures = 0
                wire_status = wires[primary]["status"]
                status.update(
                    {
                        k: wire_status[k]
                        for k in (
                            "connection_generation",
                            "connection_started_ms",
                            "subscription_confirmed",
                        )
                    }
                )
            else:
                status["subscription_confirmed"] = False
            for key in totals:
                status[key] = totals[key] + sum(w["status"].get(key, 0) for w in wires.values())
            await asyncio.sleep(0.1)
    except Exception as exc:
        status["error"] = type(exc).__name__ + ": " + str(exc)
        journal.add(
            {"kind": "public_stream_failure", "received_ms": now_ms(), "error": status["error"]}
        )
        stop.set()
    finally:
        for wire in wires.values():
            wire["stop"].set()
        await asyncio.gather(*(wire["task"] for wire in wires.values()), return_exceptions=True)
        for key in totals:
            status[key] = totals[key] + sum(w["status"].get(key, 0) for w in wires.values())
