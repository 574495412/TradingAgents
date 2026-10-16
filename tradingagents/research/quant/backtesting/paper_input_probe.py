"""对冻结全集核验真实财务输入，不运行策略或计算实际账户收益。"""

from __future__ import annotations

import argparse
import gzip
import time
from collections import Counter
from pathlib import Path

from tradingagents.dataflows.research.live_capture import MINUTE, canonical, clock_valid
from tradingagents.dataflows.research.public_http import PublicHTTP
from tradingagents.research.quant.backtesting.paper_inputs import (
    contract_rules,
    funding_window,
    quote,
)
from tradingagents.research.quant.strategy_audit import read, save, sha


def run(panel, output):
    """所有固定成员均查询，不因接口失败、品种名称或行情结果删减。"""
    manifest = read(panel / "manifest.json")
    symbols = sorted(
        s for s, v in manifest["contracts"].items() if v["asset_class"] == "TRADIFI_PERPETUAL"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    paths = [
        Path(__file__).resolve(),
        Path(funding_window.__code__.co_filename),
        Path(PublicHTTP.request.__code__.co_filename),
        Path(clock_valid.__code__.co_filename),
        Path(save.__code__.co_filename),
    ]
    for path in paths:
        (output / "source" / path.name).write_bytes(path.read_bytes())
    freeze = time.time_ns() // 1_000_000
    ending = freeze // MINUTE * MINUTE - MINUTE
    save(
        output / "protocol.json",
        {
            "schema": "cz-paper-financial-input-probe.v1",
            "frozen_ms": freeze,
            "symbols": symbols,
            "funding_start_ms": ending - 24 * 60 * MINUTE,
            "funding_end_ms": ending,
            "manifest_sha256": sha(panel / "manifest.json"),
            "source_sha256": {str(p): sha(p) for p in paths},
            "profitability_evaluated": False,
            "orders_enabled": False,
        },
    )
    client = PublicHTTP()
    records = []
    try:
        before = client.request("clock", None, {})
        metadata = client.request("exchange", None, {})
        records.extend([before, metadata])
        jobs = [
            (
                "funding",
                s,
                {
                    "symbol": s,
                    "startTime": ending - 24 * 60 * MINUTE,
                    "endTime": ending,
                    "limit": 1000,
                },
            )
            for s in symbols
        ]
        funding = client.batch(jobs)
        records.extend(funding)
        quotes = client.request("quotes", None, {})
        after = client.request("clock", None, {})
        records.extend([quotes, after])
    finally:
        client.close()
        with (output / "public_inputs.json.gz").open("xb") as target:
            target.write(gzip.compress(canonical(records), mtime=0))
    now = time.time_ns() // 1_000_000
    results, types = [], Counter()
    for symbol, funding_record in zip(symbols, funding, strict=True):
        row = {"symbol": symbol}
        for name, calculate in (
            ("contract_rules", lambda symbol=symbol: contract_rules(metadata, symbol, now)),
            (
                "received_quote",
                lambda symbol=symbol: quote(quotes, symbol, now, not_before_ms=freeze),
            ),
            (
                "funding_window",
                lambda funding_record=funding_record: funding_window(funding_record, now),
            ),
        ):
            try:
                value = calculate()
                row[name] = {"ready": True, "source_sha256": value["source_sha256"]}
                if name == "funding_window":
                    types.update(v["rate_type"] for v in value["events"])
                    row[name]["events"] = len(value["events"])
            except (ValueError, KeyError, TypeError) as exc:
                row[name] = {"ready": False, "error": str(exc)}
        results.append(row)
    result = {
        "schema": "cz-paper-financial-input-probe.v1",
        "expected_members": len(symbols),
        "evaluated_members": len(results),
        "clock_verified": clock_valid(before) and clock_valid(after),
        "elapsed_ms": now - freeze,
        "ready_counts": {
            name: sum(r[name]["ready"] for r in results)
            for name in ("contract_rules", "received_quote", "funding_window")
        },
        "funding_event_types": dict(types),
        "rows": results,
        "profitability_evaluated": False,
        "real_fills_proven": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "result.json", result)
    for path in paths:
        if sha(path) != read(output / "protocol.json")["source_sha256"][str(path)]:
            raise ValueError("公开财务输入核验期间源码变化")
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    print({k: v for k, v in result.items() if k != "rows"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.panel.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
