"""限时验证公开收盘推送与指数推送，记录真实接收时刻，不执行策略。"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter
from pathlib import Path

from tradingagents.dataflows.research.live_capture import (
    MINUTE,
    Pacer,
    body,
    canonical,
    clock_valid,
    request,
)
from tradingagents.research.quant.strategy_audit import read, save, sha

URL = "wss://fstream.binance.com/market/stream"


def unwrap(value):
    """兼容官方组合流封装，保留原文身份后再读取负载。"""
    return value.get("data", value) if isinstance(value, dict) else value


def received_record(raw, *, received_ms):
    """原始消息逐字节封存，交易所事件时钟与本地接收时钟分开保存。"""
    raw = raw.encode() if isinstance(raw, str) else raw
    return {
        "kind": "websocket",
        "transport": "public_websocket",
        "received_ms": received_ms,
        "body_base64": base64.b64encode(raw).decode(),
        "body_sha256": hashlib.sha256(raw).hexdigest(),
    }


def closed_candle(record, *, decision_ms, clock_verified):
    """只有真实完成推送且已在决策前收到的分钟，才能作为已知输入。"""
    if not clock_verified or record["received_ms"] > decision_ms:
        return None
    value = unwrap(body(record))
    if (
        not isinstance(value, dict)
        or value.get("e") != "kline"
        or value.get("k", {}).get("x") is not True
    ):
        return None
    k = value["k"]
    if (
        k["i"] != "1m"
        or k["s"] != value["s"]
        or type(k["t"]) is not int
        or type(k["T"]) is not int
        or k["t"] % MINUTE
        or k["T"] != k["t"] + MINUTE - 1
        or type(value["E"]) is not int
        or value["E"] > record["received_ms"] + 1000
    ):
        raise ValueError("收盘推送标的、分钟或事件时钟不合法")
    if decision_ms < k["T"] + 1 + 1000:
        return None
    if any(not math.isfinite(float(k[key])) or float(k[key]) <= 0 for key in ("o", "h", "l", "c")):
        raise ValueError("收盘推送价格必须为有限正数")
    if (
        any(not math.isfinite(float(k[key])) or float(k[key]) < 0 for key in ("v", "q", "V", "Q"))
        or type(k["n"]) is not int
        or k["n"] < 0
    ):
        raise ValueError("收盘推送成交量或笔数不合法")
    row = [
        k["t"],
        k["o"],
        k["h"],
        k["l"],
        k["c"],
        k["v"],
        k["T"],
        k["q"],
        k["n"],
        k["V"],
        k["Q"],
        "0",
    ]
    return value["s"], row


async def observe(symbols, duration):
    """只保存完成分钟、指数消息与订阅回应，未完成分钟不会成为研究终值。"""
    from websockets.asyncio.client import connect

    records, counts = [], Counter()
    subscription_ok, error, close_detail = False, None, None
    started = time.monotonic()
    try:
        async with connect(
            URL,
            open_timeout=15,
            close_timeout=3,
            ping_interval=20,
            ping_timeout=20,
            max_size=4_194_304,
            max_queue=128,
        ) as socket:
            streams = [f"{s.lower()}@kline_1m" for s in symbols] + ["!markPrice@arr@1s"]
            pending = set()
            for offset in range(0, len(streams), 100):
                identity = f"factor-research-{offset // 100}"
                pending.add(identity)
                await socket.send(
                    json.dumps(
                        {
                            "method": "SUBSCRIBE",
                            "params": streams[offset : offset + 100],
                            "id": identity,
                        }
                    )
                )
                await asyncio.sleep(0.3)
            until = time.monotonic() + duration
            print(f"公开推送已连接，验证 {len(streams)} 个订阅，限定 {duration} 秒。", flush=True)
            while time.monotonic() < until:
                remaining = until - time.monotonic()
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=min(10, remaining))
                except TimeoutError:
                    continue
                received = time.time_ns() // 1_000_000
                record = received_record(raw, received_ms=received)
                value = unwrap(body(record))
                counts["all_messages"] += 1
                if isinstance(value, list):
                    if value and all(
                        isinstance(item, dict) and item.get("e") == "markPriceUpdate"
                        for item in value
                    ):
                        counts["index_arrays"] += 1
                        records.append(record)
                elif isinstance(value, dict):
                    if "id" in value:
                        records.append(record)
                        if value.get("result", "absent") is not None or value["id"] not in pending:
                            error = "PUBLIC_SUBSCRIPTION_REJECTED"
                            break
                        pending.remove(value["id"])
                        subscription_ok = not pending
                    elif value.get("e") == "kline":
                        if value.get("k", {}).get("x") is True:
                            counts["closed_candles"] += 1
                            records.append(record)
                        else:
                            counts["unfinished_candles_discarded"] += 1
    except Exception as exc:
        error = type(exc).__name__
        closing = getattr(exc, "rcvd", None)
        close_detail = {
            "code": getattr(closing, "code", None),
            "reason": str(getattr(closing, "reason", ""))[:240],
        }
    return records, {
        "counts": dict(counts),
        "subscription_ok": subscription_ok,
        "error": error,
        "received_close": close_detail,
        "elapsed_seconds": time.monotonic() - started,
    }


def summarize(records, symbols, *, decision_ms, clock_verified):
    """覆盖与延迟只评价输入通道，不能充当收益或稳定性证据。"""
    minutes, delays, marks = {}, [], set()
    invalid, conflicting, seen = 0, [], {}
    for record in records:
        value = unwrap(body(record))
        if isinstance(value, list):
            marks.update(
                item["s"]
                for item in value
                if item.get("s") in symbols and item.get("i") is not None
            )
            continue
        try:
            result = closed_candle(record, decision_ms=decision_ms, clock_verified=clock_verified)
        except (ValueError, KeyError, TypeError):
            invalid += 1
            continue
        if result:
            symbol, row = result
            if symbol not in symbols:
                invalid += 1
                continue
            key = symbol, row[0]
            if key in seen and seen[key] != row:
                conflicting.append([symbol, row[0]])
            seen[key] = row
            minutes.setdefault(row[6] + 1, set()).add(symbol)
            delays.append(record["received_ms"] - row[6] - 1)
    ordered = sorted(delays)
    return {
        "completed_minute_symbol_counts": {str(t): len(v) for t, v in sorted(minutes.items())},
        "closed_candle_receive_delay_ms_min_median_max": [
            ordered[0],
            ordered[len(ordered) // 2],
            ordered[-1],
        ]
        if ordered
        else None,
        "index_stream_symbols_seen": len(marks),
        "invalid_completed_messages": invalid,
        "conflicting_completed_messages": conflicting,
        "clock_verified": clock_verified,
        "full_strategy_input_ready": False,
        "index_stream_and_archived_close_equivalence_verified": False,
        "positioning_timeliness_verified": False,
        "profitability_evaluated": False,
        "orders_enabled": False,
    }


def run(panel, review, output, duration):
    """固定本次有限采样身份；所有结果均与真实策略前向验收分开。"""
    manifest, model = read(panel / "manifest.json"), read(review)
    if (
        manifest.get("leader_labels_read") is not False
        or manifest.get("orders_enabled") is not False
        or not model.get("model_inference_performed")
        or model.get("production_eligible") is not False
    ):
        raise ValueError("公开推送验证要求纯公开集合与真实研究审查")
    sources = (
        Path(__file__).resolve(),
        Path(body.__code__.co_filename).resolve(),
        Path(read.__code__.co_filename).resolve(),
    )
    symbols = sorted(manifest["symbols"])
    if len(symbols) + 1 > 1024:
        raise ValueError("订阅数量超过单连接限制，不能静默缩减集合")
    protocol = {
        "schema": "cz-profit-forward-public-stream.v2",
        "created_ms": time.time_ns() // 1_000_000,
        "source_files": {str(p): sha(p) for p in sources},
        "symbols": symbols,
        "panel_manifest_sha256": sha(panel / "manifest.json"),
        "review_sha256": sha(review),
        "public_websocket_url": URL,
        "duration_seconds": duration,
        "subscription_batch_size": 100,
        "subscription_message_interval_seconds": 0.3,
        "websockets_version": importlib.metadata.version("websockets"),
        "retained": "closed kline messages, all index-price arrays and subscription replies",
        "capture_only": True,
        "forward_strategy_acceptance_registered": False,
        "credentials_used": False,
        "leader_labels_read": False,
        "orders_enabled": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", protocol)
    (output / "source").mkdir()
    for p in sources:
        (output / "source" / p.name).write_bytes(p.read_bytes())
    pacer = Pacer()
    before = request("clock", None, {}, pacer)
    if not clock_valid(before):
        save(output / "clock_failure.json", before)
        raise ValueError("推送采集前时钟核验失败")
    records, status = asyncio.run(observe(symbols, duration))
    after = request("clock", None, {}, pacer)
    verified = clock_valid(before) and clock_valid(after)
    payload = output / "messages.json.gz"
    with payload.open("xb") as stream:
        stream.write(gzip.compress(canonical(records), mtime=0))
    save(output / "clock.json", {"before": before, "after": after})
    result = status | summarize(
        records, set(symbols), decision_ms=time.time_ns() // 1_000_000, clock_verified=verified
    )
    save(output / "result.json", result)
    for path, expected in protocol["source_files"].items():
        if sha(path) != expected:
            raise ValueError("推送验证过程中源码变化，结果不能采用")
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=90)
    args = parser.parse_args()
    if not 10 <= args.duration_seconds <= 900:
        raise ValueError("本阶段仅允许十至九百秒的公开推送验证")
    run(args.panel.resolve(), args.review.resolve(), args.output.resolve(), args.duration_seconds)


if __name__ == "__main__":
    main()
