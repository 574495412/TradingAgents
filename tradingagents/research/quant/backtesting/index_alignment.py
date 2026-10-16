"""只检验公开指数推送与官方分钟终值的一致性，禁止回填过去的信号。"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from tradingagents.dataflows.research.live_capture import (
    MINUTE,
    Pacer,
    body,
    canonical,
    clock_valid,
    request,
)
from tradingagents.dataflows.research.stream import unwrap
from tradingagents.research.quant.strategy_audit import read, save, sha


def last_received_index(records, symbol, closing_ms):
    """原一分钟滞后假设要求在下一分钟决策前收到，过时指数不沿用。"""
    candidates = []
    for r in records:
        if r["received_ms"] > closing_ms + MINUTE:
            continue
        values = unwrap(body(r))
        if not isinstance(values, list):
            continue
        for v in values:
            if (
                v.get("s") == symbol
                and type(v.get("E")) is int
                and closing_ms - 1000 <= v["E"] < closing_ms
            ):
                price = float(v["i"])
                if not math.isfinite(price) or price <= 0:
                    raise ValueError("公开指数推送价格不合法")
                candidates.append((v["E"], r["received_ms"], v["i"]))
    return max(candidates, key=lambda v: (v[0], -v[1])) if candidates else None


def compare(stream_price, official_price):
    """精确十进制相等才叫同一终值；差值仅作输入诊断，不用于调因子阈值。"""
    left, right = Decimal(stream_price), Decimal(official_price)
    if not left.is_finite() or not right.is_finite() or left <= 0 or right <= 0:
        raise ValueError("对齐检验必须使用有限正指数价格")
    return {
        "exact_decimal_equal": left == right,
        "relative_difference_bps": float((left / right - 1) * 10000),
    }


def run(stream_root, panel, output):
    """采样分钟完全由已封存覆盖清单给出，不据价格或收益选择对齐样本。"""
    for f, expected in read(stream_root / "receipts.json").items():
        if sha(stream_root / f) != expected:
            raise ValueError("指数对齐的推送原文身份变化")
    manifest, parent, result = (
        read(panel / "manifest.json"),
        read(stream_root / "protocol.json"),
        read(stream_root / "result.json"),
    )
    if (
        sha(panel / "manifest.json") != parent["panel_manifest_sha256"]
        or not result["clock_verified"]
        or not result["subscription_ok"]
        or result["error"]
    ):
        raise ValueError("指数对齐需要已核验的完整公开推送")
    minutes = sorted(int(t) for t in result["completed_minute_symbol_counts"])
    if not minutes:
        raise ValueError("没有完成分钟可核对")
    symbols = sorted(
        s for s, v in manifest["contracts"].items() if v["asset_class"] == "TRADIFI_PERPETUAL"
    )
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__).resolve(),
        Path(body.__code__.co_filename).resolve(),
        Path(unwrap.__code__.co_filename).resolve(),
        Path(read.__code__.co_filename).resolve(),
    ]
    protocol = {
        "schema": "cz-profit-live-index-equivalence.v1",
        "minutes": minutes,
        "symbols": symbols,
        "source_files": {str(p): sha(p) for p in sources},
        "stream_receipts_sha256": sha(stream_root / "receipts.json"),
        "panel_manifest_sha256": sha(panel / "manifest.json"),
        "exact_equality_required_for_same_input_claim": True,
        "maximum_preclose_index_age_ms": 1000,
        "maximum_receipt_time_after_close_ms": MINUTE,
        "later_rest_values_not_forward_inputs": True,
        "profitability_evaluated": False,
        "orders_enabled": False,
    }
    save(output / "protocol.json", protocol)
    (output / "source").mkdir()
    for p in sources:
        (output / "source" / p.name).write_bytes(p.read_bytes())
    with gzip.open(stream_root / "messages.json.gz", "rt") as f:
        messages = json.load(f)
    pacer = Pacer()
    before = request("clock", None, {}, pacer)
    params = {
        "interval": "1m",
        "startTime": minutes[0] - MINUTE,
        "endTime": minutes[-1] - 1,
        "limit": 30,
    }
    with ThreadPoolExecutor(max_workers=12) as pool:
        records = list(
            pool.map(lambda s: request("index", s, {"pair": s} | params, pacer), symbols)
        )
    after = request("clock", None, {}, pacer)
    verified = clock_valid(before) and clock_valid(after)
    with (output / "official_responses.json.gz").open("xb") as f:
        f.write(gzip.compress(canonical(records), mtime=0))
    save(output / "clock.json", {"before": before, "after": after})
    rows, missing = [], []
    for r in records:
        official = {row[6] + 1: row[4] for row in body(r)} if r["status"] == 200 else {}
        for closing in minutes:
            streamed = last_received_index(messages, r["symbol"], closing)
            if not verified or closing not in official or streamed is None:
                missing.append(
                    {
                        "symbol": r["symbol"],
                        "closing_ms": closing,
                        "http_status": r["status"],
                        "stream_value_present": streamed is not None,
                    }
                )
                continue
            rows.append(
                {
                    "symbol": r["symbol"],
                    "closing_ms": closing,
                    "stream_event_ms": streamed[0],
                    "stream_received_ms": streamed[1],
                    "stream_price": streamed[2],
                    "official_close": official[closing],
                    **compare(streamed[2], official[closing]),
                }
            )
    differences = sorted(abs(r["relative_difference_bps"]) for r in rows)
    conclusion = {
        "expected_comparisons": len(symbols) * len(minutes),
        "compared": len(rows),
        "exact_matches": sum(r["exact_decimal_equal"] for r in rows),
        "absolute_difference_bps_min_median_max": [
            differences[0],
            differences[len(differences) // 2],
            differences[-1],
        ]
        if differences
        else None,
        "missing": missing,
        "rows": rows,
        "clock_verified": verified,
        "same_input_claim_supported": len(rows) == len(symbols) * len(minutes)
        and all(r["exact_decimal_equal"] for r in rows),
        "later_rest_values_used_for_forward_signals": False,
        "profitability_evaluated": False,
        "orders_enabled": False,
    }
    save(output / "result.json", conclusion)
    for p, expected in protocol["source_files"].items():
        if sha(p) != expected:
            raise ValueError("指数对齐过程中源码发生变化")
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(
        json.dumps(
            {k: v for k, v in conclusion.items() if k not in ("rows", "missing")},
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stream.resolve(), args.panel.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
