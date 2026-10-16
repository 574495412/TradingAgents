"""完整固定全集的盘口时间与显示深度采样，不修改执行规则或计算收益。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import time
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode

import requests

from tradingagents.dataflows.research.live_capture import BASE, body, clock_valid
from tradingagents.dataflows.research.public_http import PublicHTTP
from tradingagents.research.quant.backtesting.paper_inputs import number
from tradingagents.research.quant.backtesting.paper_store import Chain
from tradingagents.research.quant.strategy_audit import read, save, sha


def depth_request(client, symbol):
    """仅开放同一公开主机的五档快照，不接受认证或任意路径参数。"""
    if not re.fullmatch(r"[A-Z0-9]+USDT", symbol):
        raise ValueError("深度探针合约标识不合法")
    url = BASE + "/fapi/v1/depth?" + urlencode({"symbol": symbol, "limit": 5})
    if not client.pacer.acquire(2):
        return {"kind": "depth", "symbol": symbol, "status": "CANCELLED_BEFORE_REQUEST"}
    started, tick = time.time_ns() // 1_000_000, time.monotonic()
    session = client.session()
    session.cookies.clear()
    status, raw, error, headers = 0, b"", None, {}
    try:
        with session.get(url, timeout=(4, 4), allow_redirects=False) as response:
            status, raw = response.status_code, response.content
            headers = {k.lower(): v for k, v in response.headers.items()}
    except requests.RequestException as exc:
        error = type(exc).__name__
    finally:
        session.cookies.clear()
    if status in (418, 429):
        client.pacer.stop.set()
    received, elapsed = time.time_ns() // 1_000_000, (time.monotonic() - tick) * 1000
    return {
        "kind": "depth",
        "symbol": symbol,
        "url": url,
        "status": status,
        "requested_ms": started,
        "received_ms": received,
        "elapsed_monotonic_ms": elapsed,
        "local_clock_consistent": received >= started and abs(received - started - elapsed) <= 250,
        "body_base64": base64.b64encode(raw).decode(),
        "body_sha256": hashlib.sha256(raw).hexdigest(),
        "server_http_date": headers.get("date"),
        "error": error,
        "credentials_used": False,
        "body_stage": "application_response_after_http_decoding",
    }


def describe_pair(symbol, quote_record, depth_record):
    """交易事件、消息输出与接收时间分别报告；两次查询不假定同时发生。"""
    if any(
        r.get("status") != 200 or not r.get("local_clock_consistent")
        for r in (quote_record, depth_record)
    ):
        return {"ready": False, "reason": "PUBLIC_RESPONSE_UNAVAILABLE"}
    q, d = body(quote_record), body(depth_record)
    if (
        not isinstance(q, dict)
        or q.get("symbol") != symbol
        or quote_record.get("symbol") != symbol
        or depth_record.get("symbol") != symbol
        or quote_record.get("kind") != "quotes"
        or depth_record.get("kind") != "depth"
        or any(r.get("credentials_used") is not False for r in (quote_record, depth_record))
    ):
        raise ValueError("配对盘口的合约身份不同")
    if any(
        type(v) is not int for v in (q.get("time"), d.get("E"), d.get("T"), d.get("lastUpdateId"))
    ):
        raise ValueError("盘口缺少明确事件时钟或更新序号")
    if (
        q["time"] > quote_record["received_ms"] + 1000
        or d["E"] > depth_record["received_ms"] + 1000
        or d["T"] > d["E"]
    ):
        raise ValueError("盘口事件时钟晚于实际收到时间")
    levels = {}
    for side, reverse in (("bids", True), ("asks", False)):
        values = [(number(p, positive=True), number(qty, positive=True)) for p, qty in d[side]]
        if not 1 <= len(values) <= 5 or [p for p, _ in values] != sorted(
            {p for p, _ in values}, reverse=reverse
        ):
            raise ValueError("五档盘口为空、重复或顺序错误")
        levels[side] = values
    if levels["bids"][0][0] > levels["asks"][0][0]:
        raise ValueError("深度盘口买卖价交叉")
    quote_age = quote_record["received_ms"] - q["time"]
    return {
        "ready": True,
        "book_event_age_at_receive_ms": quote_age,
        "depth_output_age_at_receive_ms": depth_record["received_ms"] - d["E"],
        "depth_transaction_age_at_receive_ms": depth_record["received_ms"] - d["T"],
        "depth_output_minus_transaction_ms": d["E"] - d["T"],
        "book_event_within_five_seconds_at_receive_only": -1000 <= quote_age <= 5000,
        "separate_queries_top_prices_equal": number(q["bidPrice"], positive=True)
        == levels["bids"][0][0]
        and number(q["askPrice"], positive=True) == levels["asks"][0][0],
        "visible_notional_usdt": {
            side: {
                "best": str(values[0][0] * values[0][1]),
                "five_levels": str(sum((p * qty for p, qty in values), Decimal(0))),
            }
            for side, values in levels.items()
        },
        "timestamps_and_capacity_do_not_prove_real_fills": True,
    }


def run(panel, output):
    """先冻结完整全集和目的，再保存每个已安排成员的两份独立原文。"""
    manifest = read(panel / "manifest.json")
    symbols = sorted(
        s for s, c in manifest["contracts"].items() if c["asset_class"] == "TRADIFI_PERPETUAL"
    )
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        Path(__file__).resolve(),
        *(
            Path(f.__code__.co_filename).resolve()
            for f in (PublicHTTP.request, body, number, Chain.add, save)
        ),
    }
    protocol = {
        "schema": "cz-public-depth-semantics-probe.v1",
        "symbols": symbols,
        "frozen_ms": time.time_ns() // 1_000_000,
        "manifest_sha256": sha(panel / "manifest.json"),
        "source_sha256": {str(p): sha(p) for p in paths},
        "depth_limit": 5,
        "purpose": "PUBLIC_TIMESTAMP_AND_VISIBLE_CAPACITY_ONLY",
        "execution_policy_unchanged": True,
        "strategy_profitability_evaluated": False,
        "orders_enabled": False,
    }
    save(output / "protocol.json", protocol)
    source = output / "source"
    source.mkdir()
    for path in paths:
        (source / path.name).write_bytes(path.read_bytes())
    chain, client = Chain(output / "public_events.jsonl"), PublicHTTP()
    rows, failure = {}, None
    try:
        clock = client.request("clock", None, {})
        chain.add(clock)
        if not clock_valid(clock):
            raise ValueError("探针启动时钟无法核验")

        def paired(symbol):
            q = client.request("quotes", symbol, {"symbol": symbol})
            return symbol, q, depth_request(client, symbol)

        for symbol, q, d in client.executor.map(paired, symbols):
            chain.add(q)
            chain.add(d)
            try:
                rows[symbol] = describe_pair(symbol, q, d)
            except (ValueError, KeyError, TypeError) as exc:
                rows[symbol] = {"ready": False, "reason": str(exc)}
        clock = client.request("clock", None, {})
        chain.add(clock)
        if not clock_valid(clock):
            raise ValueError("探针结束时钟无法核验")
        if any(sha(path) != protocol["source_sha256"][str(path)] for path in paths):
            raise ValueError("探针运行源码发生变化")
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        client.close()
        prefix = chain.seal()
        chain.close()
    result = {
        "failure": failure,
        "scheduled_members": len(symbols),
        "recorded_members": len(rows),
        "ready_members": sum(v["ready"] for v in rows.values()),
        "rows": rows,
        "source_prefix": prefix,
        "strategy_profitability_evaluated": False,
        "execution_policy_unchanged": True,
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    print({k: v for k, v in result.items() if k != "rows"}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if run(args.panel.resolve(), args.output.resolve())["failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
