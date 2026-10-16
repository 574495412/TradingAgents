"""真实决策期限内的有限公开时钟重试，保留全部失败原文。"""

from __future__ import annotations

import asyncio
import time
import uuid

from tradingagents.dataflows.research.live_capture import clock_valid

OWNED_MARK_WAIT_VERSION = "owned_completed_minute_wait_v1"


def milliseconds():
    return time.time_ns() // 1_000_000


def monotonic_ms():
    return time.monotonic_ns() / 1_000_000


async def wait_for_owned_marks(bars, symbols, closing, journal, deadline_ms):
    """报价前最多等五秒，绝不延长原期限、倒退时钟或沿用上一分钟价格。"""
    started, tick = milliseconds(), monotonic_ms()
    identity = uuid.uuid4().hex
    symbols = sorted(set(symbols))
    journal.add(
        {
            "kind": "owned_mark_wait_started",
            "received_ms": started,
            "wait_id": identity,
            "version": OWNED_MARK_WAIT_VERSION,
            "canonical_close_ms": closing,
            "symbols": symbols,
            "original_deadline_ms": deadline_ms,
            "maximum_wait_ms": 5000,
        }
    )
    while True:
        now, elapsed = milliseconds(), monotonic_ms() - tick
        missing = [s for s in symbols if bars.at(s, closing, now) is None]
        if now < started or elapsed < 0:
            reason = "LOCAL_CLOCK_MOVED_BACKWARDS"
        elif now >= deadline_ms:
            reason = "ORIGINAL_DEADLINE_REACHED"
        elif not missing and elapsed <= 5000:
            reason = None
        elif elapsed >= 5000:
            reason = "OWNED_MARK_WAIT_LIMIT_REACHED"
        else:
            await asyncio.sleep(min(0.05, (deadline_ms - now) / 1000, (5000 - elapsed) / 1000))
            continue
        record = {
            "kind": "owned_mark_wait_finished",
            "received_ms": now,
            "wait_id": identity,
            "version": OWNED_MARK_WAIT_VERSION,
            "canonical_close_ms": closing,
            "symbols": symbols,
            "original_deadline_ms": deadline_ms,
            "elapsed_monotonic_ms": elapsed,
            "ready": reason is None,
            "reason": reason,
            "missing_symbols": missing,
        }
        journal.add(record)
        return record


async def received_clock(client, journal, deadline_ms):
    """只重试网络和服务端暂时失败；有效响应也必须在原期限内收到。"""
    for attempt in range(3):
        if milliseconds() >= deadline_ms:
            raise ValueError("公开时钟重试已到原决策期限")
        record = await asyncio.to_thread(client.request, "clock", None, {})
        journal.add(record)
        if milliseconds() >= deadline_ms or record.get("received_ms", deadline_ms) >= deadline_ms:
            raise ValueError("公开时钟响应超过原决策期限")
        if clock_valid(record):
            return record
        if record.get("status") not in {0, 500, 502, 503, 504}:
            raise ValueError("公开时钟成功响应异常或不可重试，停止本帧")
        journal.add(
            {
                "kind": "public_clock_retry",
                "received_ms": milliseconds(),
                "completed_attempts": attempt + 1,
                "maximum_attempts": 3,
                "deadline_ms": deadline_ms,
            }
        )
        if attempt < 2:
            await asyncio.sleep(0.25 * (attempt + 1))
    raise ValueError("公开时钟在三次有限请求后仍无法核验")
