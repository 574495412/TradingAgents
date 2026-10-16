"""在实际已收到的公开前缀上评估完整双向全集，仅验证可计算性，不计算收益。"""

from __future__ import annotations

import argparse
import gzip
import time
from collections import Counter
from pathlib import Path

from tradingagents.dataflows.research.live_capture import (
    available_rows,
    body,
    canonical,
    clock_valid,
)
from tradingagents.dataflows.research.public_http import PublicHTTP
from tradingagents.dataflows.research.stream import closed_candle
from tradingagents.research.quant.backtesting.bidirectional import HYPOTHESIS, evidence, index_jobs
from tradingagents.research.quant.backtesting.forward_audit import verify_chain, verify_consensus
from tradingagents.research.quant.backtesting.forward_inputs import Bars, venue_contracts
from tradingagents.research.quant.strategy_audit import read, save, sha


def run(parent, panel, review, output):
    """采用最新已封存分钟，先冻结全集和输入前缀，再请求官方指数。"""
    manifest, proposal = read(panel / "manifest.json"), read(review)
    if (
        not proposal.get("model_inference_performed")
        or proposal.get("production_eligible") is not False
        or not any(h["id"] == HYPOTHESIS for h in proposal["review"]["hypotheses"])
    ):
        raise ValueError("新双向假说必须有对应真实模型提案")
    decision_path = sorted((parent / "decisions").glob("*.json"))[-1]
    decision = read(decision_path)
    if not decision["within_fixed_one_minute_deadline"]:
        raise ValueError("最新父输入分钟不满足其实际决策时序")
    output.mkdir(parents=True, exist_ok=False)
    prefix_path = output / "parent_public_prefix.jsonl"
    with (parent / "public_events.jsonl").open("rb") as source, prefix_path.open("xb") as target:
        for _ in range(decision["source_prefix"]["event_count"]):
            line = source.readline()
            if not line.endswith(b"\n"):
                raise ValueError("父决策引用的原文前缀尚未完整落盘")
            target.write(line)
    records, heads = verify_chain(prefix_path)
    if heads[-1] != decision["source_prefix"]["event_chain_sha256"]:
        raise ValueError("父决策原文前缀身份变化")
    verify_consensus(records)
    closing = decision["canonical_close_ms"]
    contracts = {
        s: v for s, v in manifest["contracts"].items() if v["asset_class"] == "TRADIFI_PERPETUAL"
    }
    sources = [
        Path(__file__).resolve(),
        Path(evidence.__code__.co_filename).resolve(),
        Path(verify_chain.__code__.co_filename).resolve(),
        Path(Bars.add.__code__.co_filename).resolve(),
        Path(PublicHTTP.request.__code__.co_filename).resolve(),
        Path(body.__code__.co_filename).resolve(),
        Path(closed_candle.__code__.co_filename).resolve(),
        Path(read.__code__.co_filename).resolve(),
    ]
    save(
        output / "protocol.json",
        {
            "schema": "cz-live-bidirectional-input-probe.v1",
            "hypothesis": HYPOTHESIS,
            "canonical_close_ms": closing,
            "parent_decision_sha256": sha(decision_path),
            "parent_protocol_sha256": sha(parent / "protocol.json"),
            "parent_public_prefix_sha256": sha(prefix_path),
            "model_proposal_sha256": sha(review),
            "panel_manifest_sha256": sha(panel / "manifest.json"),
            "symbols": sorted(contracts),
            "source_files": {str(p): sha(p) for p in sources},
            "latest_committed_parent_minute_used": True,
            "profitability_evaluated": False,
            "orders_enabled": False,
        },
    )
    (output / "source").mkdir()
    for p in sources:
        (output / "source" / p.name).write_bytes(p.read_bytes())
    bars, now = Bars(contracts), time.time_ns() // 1_000_000
    interim = {
        (v["symbol"], v["closing_ms"], v["conflicting_source_sha256"])
        for r in records
        if r["kind"] == "startup_public_value_conflict"
        for v in r["conflicts"]
    }
    for record in records:
        if record["kind"] == "websocket":
            value = closed_candle(record, decision_ms=now, clock_verified=True)
            selected = [(value[0], value[1])] if value and value[0] in contracts else []
        elif record["kind"] == "klines" and record["symbol"] in contracts:
            selected = [
                (record["symbol"], row)
                for row in available_rows(
                    record, decision_ms=now, cutoff_ms=closing, clock_verified=True
                )
            ]
        else:
            continue
        for symbol, row in selected:
            if (symbol, row[6] + 1, record["body_sha256"]) in interim:
                continue
            bars.add(
                symbol,
                row,
                received_ms=record["received_ms"],
                source_sha256=record["body_sha256"],
                completed_verified=True,
            )
    client = PublicHTTP()
    before = client.request("clock", None, {})
    metadata = client.request("exchange", None, {})
    responses = client.batch(index_jobs(contracts, closing))
    after = client.request("clock", None, {})
    client.close()
    decided = time.time_ns() // 1_000_000
    verified = clock_valid(before) and clock_valid(after)
    eligible = venue_contracts(metadata, contracts, decision_ms=decided)
    rows = []
    for record in responses:
        value = evidence(
            bars,
            record["symbol"],
            record,
            closing=closing,
            decision_ms=decided,
            clock_verified=verified,
        )
        member = eligible.get(record["symbol"]) if eligible else None
        active = bool(
            member
            and member["trading"]
            and member["onboard_ms"] <= closing - 60000
            and (not member["delivery_ms"] or closing - 60000 < member["delivery_ms"])
        )
        rows.append({"symbol": record["symbol"], "known_entry_eligible": active, **value})
    with (output / "official_inputs.json.gz").open("xb") as f:
        f.write(gzip.compress(canonical([before, metadata, *responses, after]), mtime=0))
    result = {
        "expected_fixed_members": len(contracts),
        "evaluated_members": len(rows),
        "ready_members": sum(v["ready"] for v in rows),
        "readiness_reasons": dict(Counter(v["reason"] for v in rows)),
        "eligible_direction_counts": dict(
            Counter(str(v["direction"]) for v in rows if v["known_entry_eligible"] and v["ready"])
        ),
        "clock_verified": verified,
        "actual_computation_ms": decided,
        "canonical_close_ms": closing,
        "rows": rows,
        "signal_activity_and_account_not_started": True,
        "profitability_evaluated": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "result.json", result)
    for path in sources:
        if sha(path) != read(output / "protocol.json")["source_files"][str(path)]:
            raise ValueError("新因子输入检查期间源码改变")
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print({k: v for k, v in result.items() if k != "rows"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent.resolve(), args.panel.resolve(), args.review.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
