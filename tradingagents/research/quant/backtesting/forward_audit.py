"""独立核验公开原文链及真实决策时钟，不把输入审计当作盈利证明。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from tradingagents.dataflows.research.live_capture import MINUTE, body, canonical
from tradingagents.research.quant.strategy_audit import read, save, sha


def verify_chain(path):
    """逐条重算前缀身份；损坏原文或不连续序号不能被后续清单覆盖。"""
    records, heads, previous = [], [None], None
    with Path(path).open() as stream:
        for number, line in enumerate(stream):
            value = json.loads(line)
            if value["sequence"] != number or value["previous_sha256"] != previous:
                raise ValueError("公开原文链序号或前缀不连续")
            record = value["record"]
            if "body_base64" in record:
                raw = base64.b64decode(record["body_base64"], validate=True)
                if hashlib.sha256(raw).hexdigest() != record["body_sha256"]:
                    raise ValueError("公开原文与接收记录不一致")
            previous = hashlib.sha256(canonical(value)).hexdigest()
            records.append(record)
            heads.append(previous)
    return records, heads


def verify_prefix(decision, records, heads):
    """核验决策只引用已落盘且已收到的前缀，执行等待从真实决策时刻起算。"""
    count = decision["source_prefix"]["event_count"]
    if (
        type(count) is not int
        or not 0 <= count < len(heads)
        or heads[count] != decision["source_prefix"]["event_chain_sha256"]
    ):
        raise ValueError("决策引用的公开原文前缀不存在")
    decided, closing = decision["actual_decision_ms"], decision["canonical_close_ms"]
    if closing % MINUTE or decided < closing + 1000:
        raise ValueError("决策早于已完成分钟的安全边界")
    if any(r.get("received_ms", 0) > decided for r in records[:count]):
        raise ValueError("决策前缀包含尚未收到的原文")
    within = decided < closing + MINUTE
    if within != decision["within_fixed_one_minute_deadline"] or (
        not within and decision["emitted"]
    ):
        raise ValueError("过时决策仍然产生研究信号")
    if decision["orders_enabled"] is not False:
        raise ValueError("公开输入审计不允许订单权限")
    for signal in decision["emitted"]:
        if (
            signal["actual_decision_ms"] != decided
            or signal["earliest_execution_ms"] < decided + MINUTE
            or signal["symbol"] not in decision["price_ready"]
            or signal["orders_enabled"] is not False
        ):
            raise ValueError("研究信号时间、候选来源或权限不成立")
    canary = decision.get("source_only_canary")
    if canary and (
        canary["can_never_emit"] is not True or decision["price_ready"] or decision["emitted"]
    ):
        raise ValueError("数据探针被用作策略信号")
    verify_consensus(records[:count])


def original_row(record, symbol, closing):
    """独立取原文字段，不读取活动状态里的价格或被覆盖缓存。"""
    if "body_base64" not in record:
        return None
    value = body(record)
    if record["kind"] == "websocket":
        value = value.get("data", value)
        if not isinstance(value, dict) or value.get("e") != "kline" or value.get("s") != symbol:
            return None
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
        ]
        return tuple(Decimal(str(v)) for v in row) if k["T"] + 1 == closing else None
    if (
        record.get("kind") == "klines"
        and record.get("symbol") == symbol
        and record.get("status") == 200
    ):
        rows = [r for r in value if r[6] + 1 == closing]
        return tuple(Decimal(str(v)) for v in rows[0][:11]) if len(rows) == 1 else None
    return None


def verify_consensus(records):
    """每个已记录冲突必须在决策前被精确复核，不能只核对本批最后一根。"""
    resolved = 0
    for event_number, event in enumerate(records):
        if event["kind"] != "startup_public_value_conflict":
            continue
        for conflict in event["conflicts"]:
            first = [
                r
                for r in records[:event_number]
                if r.get("body_sha256") == conflict["first_source_sha256"]
                and r.get("received_ms", 0) <= event["received_ms"]
                and original_row(r, conflict["symbol"], conflict["closing_ms"]) is not None
            ]
            expected = (
                original_row(first[0], conflict["symbol"], conflict["closing_ms"])
                if first
                else None
            )
            if expected is None:
                raise ValueError("冲突的首次价格没有原文依据")
            valid = [
                r
                for r in records
                if r.get("purpose") == "STARTUP_CONSENSUS_RECHECK_ONLY"
                and r.get("status") == 200
                and r["received_ms"] >= event["received_ms"]
                and r["requested_ms"] >= conflict["closing_ms"] + MINUTE + 1000
                and original_row(r, conflict["symbol"], conflict["closing_ms"]) == expected
            ]
            if not valid:
                raise ValueError("存在未被原文精确复核的启动终值冲突")
            resolved += 1
    return resolved


def run(root, output):
    """原始文件、归档源码及每个决策均核对；没有收益账本就不输出盈利资格。"""
    root = root.resolve()
    for relative, expected in read(root / "receipts.json").items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("前向采样封存文件身份失败")
    protocol, result = read(root / "protocol.json"), read(root / "result.json")
    for origin, expected in protocol["source_files"].items():
        matches = [
            p for p in (root / "source").iterdir() if p.name.partition("_")[2] == Path(origin).name
        ]
        if len(matches) != 1 or sha(matches[0]) != expected:
            raise ValueError("前向采样归档源码身份失败")
    records, heads = verify_chain(root / "public_events.jsonl")
    final = result["source_prefix"]
    if final["event_count"] != len(records) or final["event_chain_sha256"] != heads[-1]:
        raise ValueError("终态与原文链不能相互重建")
    decisions = [read(p) for p in sorted((root / "decisions").glob("*.json"))]
    confirmed_conflicts = verify_consensus(records) if decisions else None
    used, last_close = set(), None
    for decision in decisions:
        verify_prefix(decision, records, heads)
        closing = decision["canonical_close_ms"]
        if last_close is not None and closing <= last_close:
            raise ValueError("共同分钟决策重复或逆序")
        last_close = closing
        path = root / "snapshots" / f"{closing}.parquet"
        if sha(path) != decision["snapshot_sha256"]:
            raise ValueError("决策依赖的共同截面身份变化")
        for signal in decision["emitted"]:
            key = signal["arm"], signal["episode"]
            if key in used:
                raise ValueError("同一活动被重复授予同一研究臂")
            used.add(key)
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "result.json",
        {
            "schema": "cz-profit-forward-input-audit.v1",
            "source_receipts_sha256": sha(root / "receipts.json"),
            "audit_source_sha256": sha(__file__),
            "original_records_verified": len(records),
            "decision_prefixes_verified": len(decisions),
            "unique_emitted_arm_activities": len(used),
            "startup_conflicts_confirmed_from_raw": confirmed_conflicts,
            "pipeline_decisions_exercised": bool(decisions),
            "dependency_path_exercised": any(
                d["evaluated"] or d.get("source_only_canary") for d in decisions
            ),
            "runtime_clean": result["failure"] is None
            and result["clock_verified"]
            and not result.get("stream", {}).get("error")
            and not result.get("stream", {}).get("transient_disconnects"),
            "journal_and_clock_invariants_passed": True,
            "profitability_evaluated": False,
            "production_eligible": False,
            "orders_enabled": False,
        },
    )
    (output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(read(output / "result.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.output.resolve())


if __name__ == "__main__":
    main()
