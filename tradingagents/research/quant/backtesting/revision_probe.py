"""重取整个固定集合的同一完成分钟，判断跨通道冲突是否随后收敛。"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from tradingagents.dataflows.research.live_capture import MINUTE, body, canonical, clock_valid
from tradingagents.dataflows.research.public_http import PublicHTTP
from tradingagents.dataflows.research.stream import unwrap
from tradingagents.research.quant.backtesting.forward_inputs import normalized_row
from tradingagents.research.quant.strategy_audit import read, save, sha


def run(parent, output):
    """时刻取自首次共同分钟绑定，全集取自原协议，不据价格或收益选择。"""
    for relative, expected in read(parent / "receipts.json").items():
        path = (parent / relative).resolve()
        if not path.is_relative_to(parent.resolve()) or sha(path) != expected:
            raise ValueError("冲突原始采样身份变化")
    protocol = read(parent / "protocol.json")
    records = [json.loads(line)["record"] for line in (parent / "public_events.jsonl").open()]
    closing = next(
        r["canonical_close_ms"] for r in records if r["kind"] == "startup_warmup_binding"
    )
    websocket, early = {}, {}
    for record in records:
        if record["kind"] == "websocket":
            value = unwrap(body(record))
            if (
                isinstance(value, dict)
                and value.get("e") == "kline"
                and value["k"]["T"] + 1 == closing
            ):
                k = value["k"]
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
                websocket.setdefault(value["s"], normalized_row(row))
        elif record["kind"] == "klines" and record["status"] == 200:
            for row in body(record):
                if row[6] + 1 == closing:
                    early.setdefault(record["symbol"], normalized_row(row))
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__).resolve(),
        Path(normalized_row.__code__.co_filename).resolve(),
        Path(body.__code__.co_filename).resolve(),
        Path(PublicHTTP.request.__code__.co_filename).resolve(),
    ]
    save(
        output / "protocol.json",
        {
            "schema": "cz-public-completed-minute-revision-probe.v1",
            "closing_ms": closing,
            "symbols": protocol["symbols"],
            "parent_receipts_sha256": sha(parent / "receipts.json"),
            "source_files": {str(p): sha(p) for p in sources},
            "full_frozen_universe_queried": True,
            "profitability_evaluated": False,
            "orders_enabled": False,
        },
    )
    (output / "source").mkdir()
    for p in sources:
        (output / "source" / p.name).write_bytes(p.read_bytes())
    client = PublicHTTP()
    before = client.request("clock", None, {})
    replies = client.batch(
        [
            (
                "klines",
                s,
                {
                    "symbol": s,
                    "interval": "1m",
                    "startTime": closing - MINUTE,
                    "endTime": closing - 1,
                    "limit": 1,
                },
            )
            for s in protocol["symbols"]
        ]
    )
    after = client.request("clock", None, {})
    client.close()
    with (output / "official_responses.json.gz").open("xb") as f:
        f.write(gzip.compress(canonical(replies), mtime=0))
    save(output / "clock.json", {"before": before, "after": after})
    comparisons = []
    for record in replies:
        symbol = record["symbol"]
        rows = body(record) if record["status"] == 200 else []
        late = normalized_row(rows[0]) if len(rows) == 1 and rows[0][6] + 1 == closing else None
        first, initial = websocket.get(symbol), early.get(symbol)
        comparisons.append(
            {
                "symbol": symbol,
                "http_status": record["status"],
                "early_rest_equal_stream": first is not None and first == initial,
                "later_rest_equal_stream": first is not None and first == late,
                "early_rest_equal_later_rest": initial is not None and initial == late,
                "all_values_present": first is not None
                and initial is not None
                and late is not None,
            }
        )
    result = {
        "expected": len(comparisons),
        "all_values_present": sum(v["all_values_present"] for v in comparisons),
        "early_rest_equal_stream": sum(v["early_rest_equal_stream"] for v in comparisons),
        "later_rest_equal_stream": sum(v["later_rest_equal_stream"] for v in comparisons),
        "changed_rest_values": [
            v["symbol"]
            for v in comparisons
            if not v["early_rest_equal_later_rest"] and v["all_values_present"]
        ],
        "clock_verified": clock_valid(before) and clock_valid(after),
        "comparisons": comparisons,
        "profitability_evaluated": False,
        "orders_enabled": False,
    }
    save(output / "result.json", result)
    for p in sources:
        if sha(p) != read(output / "protocol.json")["source_files"][str(p)]:
            raise ValueError("完成分钟修订核验过程中源码改变")
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print({k: v for k, v in result.items() if k != "comparisons"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
