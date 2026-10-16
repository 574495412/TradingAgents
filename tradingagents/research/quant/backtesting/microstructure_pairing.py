"""把全部原规则活动与同期观测对齐；缺口、拒绝和两成本账户均独立保留。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sqlite3
import sys
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from tradingagents.dataflows.research.live_capture import MINUTE, available_rows, body
from tradingagents.research.quant.backtesting.microstructure import changes, endpoint, parse_message
from tradingagents.research.quant.backtesting.microstructure_audit import audit as observer_audit
from tradingagents.research.quant.backtesting.paper_account import PaperAccount
from tradingagents.research.quant.backtesting.paper_live_audit import (
    official_batch,
    require_loaded_sources,
)
from tradingagents.research.quant.backtesting.raw_chain import raw_entries
from tradingagents.research.quant.backtesting.segmented_chain import segment_entries
from tradingagents.research.quant.backtesting.source_cursor import SourceCursor
from tradingagents.research.quant.paths import PROJECT_ROOT
from tradingagents.research.quant.strategy_audit import read, save, sha


def verify_receipts(root):
    """摘要必须覆盖完整文件集合，既不能漏文件，也不能在核验后偷换输入。"""
    root = Path(root)
    receipts = read(root / "receipts.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"receipts.json"}
    if actual != set(receipts) or any(
        sha(root / name) != value for name, value in receipts.items()
    ):
        raise ValueError("配对输入的封存身份发生变化")


def clock_epoch(value, epoch, intervals, end):
    """物理区间、真实接收水位和已知时钟取交集，不跨未知时钟继续沿用旧值。"""
    if value is None or epoch is None or epoch["confirmed_ms"] is None:
        return None
    for window in intervals:
        if window["known_from_ms"] <= value["received_ms"] <= window["known_through_ms"]:
            return {
                "generation": epoch["generation"],
                "confirmed_ms": max(epoch["confirmed_ms"], window["known_from_ms"]),
                "end_ms": min(
                    end,
                    window["known_through_ms"],
                    epoch["end_ms"]
                    if epoch["end_ms"] is not None
                    else epoch["observed_through_ms"],
                ),
            }
    return None


class ReceivedMessages:
    """临时索引只来自已验证原文；最新坏状态不能退回较早有利报价。"""

    def __init__(self, captures, database, study_protocol_sha256):
        self.database = sqlite3.connect(database)
        self.database.execute(
            "CREATE TABLE observations (role TEXT, symbol TEXT, transaction_ms INTEGER, received_ms INTEGER, sequence INTEGER, capture INTEGER, payload TEXT)"
        )
        self.sources, seen_generations, spans = [], set(), []
        try:
            for identity, root in enumerate(map(Path, captures)):
                protocol = read(root / "protocol.json")
                if protocol["linked_study_protocol_sha256"] != study_protocol_sha256:
                    raise ValueError("同期来源关联了其他研究账户")
                expected = {
                    Path(name).name: value for name, value in protocol["source_sha256"].items()
                }
                if expected.get("microstructure.py") != sha(
                    Path(parse_message.__code__.co_filename)
                ):
                    raise ValueError("实际加载的配对解析器与来源冻结版本不同")
                result = observer_audit(root)
                if result["payload_errors"]:
                    raise ValueError("非法同期负载不能参与价格配对，原文仍保留")
                epochs = {
                    (e["source_role"], e["generation"]): e for e in result["connection_epochs"]
                }
                if seen_generations.intersection(epochs):
                    raise ValueError("不能把同一物理来源的重复前缀冒充新增观察")
                seen_generations.update(epochs)
                for (role, _), epoch in epochs.items():
                    if epoch["confirmed_ms"] is None:
                        continue
                    beginning = epoch["confirmed_ms"]
                    ending = (
                        epoch["end_ms"]
                        if epoch["end_ms"] is not None
                        else epoch["observed_through_ms"]
                    )
                    if any(
                        r == role and max(beginning, start) <= min(ending, end)
                        for r, start, end in spans
                    ):
                        raise ValueError("同期来源物理区间重叠，不能事后挑选报价渠道")
                    spans.append((role, beginning, ending))
                self.sources.append(
                    {
                        "root": str(root),
                        "protocol": protocol,
                        "audit": result,
                        "epochs": epochs,
                        "receipt_sha256": sha(root / "receipts.json"),
                    }
                )
                for sequence, record, _ in raw_entries(root / "public_events"):
                    if record["kind"] != "observer_message" or record["after_capture_end"]:
                        continue
                    envelope = body(record)
                    if "id" in envelope:
                        continue
                    value = parse_message(record, protocol["symbols"])
                    self.database.execute(
                        "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            value["role"],
                            value["symbol"],
                            value["transaction_ms"],
                            value["received_ms"],
                            sequence,
                            identity,
                            json.dumps(value, sort_keys=True),
                        ),
                    )
                verify_receipts(root)
            self.database.execute(
                "CREATE INDEX endpoint_lookup ON observations (role, symbol, transaction_ms DESC, received_ms DESC, sequence DESC)"
            )
            self.database.commit()
        except BaseException:
            self.database.close()
            raise

    def latest(self, role, symbol, data_ms, decision_ms):
        """先选择最新已发生且已收到的消息，再检查质量；不按结果挑选来源。"""
        row = self.database.execute(
            "SELECT capture, payload FROM observations WHERE role=? AND symbol=? AND transaction_ms<=? AND received_ms<=? ORDER BY transaction_ms DESC, received_ms DESC, sequence DESC LIMIT 1",
            (role, symbol, data_ms, decision_ms),
        ).fetchone()
        if row is None:
            return None, None
        source, value = self.sources[row[0]], json.loads(row[1])
        epoch = source["epochs"].get((role, value["generation"]))
        return value, clock_epoch(
            value, epoch, source["audit"]["clock_intervals"], source["protocol"]["capture_end_ms"]
        )

    def close(self):
        self.database.close()


def price_point(cursor, frame, symbol, episode, ledger):
    """按原公式的共同分钟重建P、I和此前60根基线，实际分钟末尾为边界减一毫秒。"""
    value = frame["evidence"][symbol]
    if not value.get("ready"):
        return {
            "endpoint_available": False,
            "reason": value.get("reason", "DIRECTION_EVIDENCE_UNKNOWN"),
            "actual_ms": frame["actual_ms"],
        }
    closing, decision = value["data_clock_ms"], value["actual_decision_ms"]
    records = official_batch(
        [r for _, r in cursor.indices], cursor.symbols, frame["canonical_close_ms"]
    )
    record = records[symbol]
    rows = available_rows(
        record, decision_ms=decision, cutoff_ms=frame["canonical_close_ms"], clock_verified=True
    )
    indices = {r[6] + 1: Decimal(str(r[4])) for r in rows}
    clocks = list(range(closing - 60 * MINUTE, closing + MINUTE, MINUTE))
    bars = [cursor.bars.at(symbol, stamp, decision) for stamp in clocks]
    if any(v is None for v in bars) or any(stamp not in indices for stamp in clocks):
        raise ValueError("原活动的已知共同分钟与基线不能重建")
    with localcontext() as context:
        context.prec = 28
        baseline = median(
            [
                Decimal(str(bar["row"][4])) / indices[stamp] - 1
                for stamp, bar in zip(clocks[:-1], bars[:-1], strict=True)
            ]
        )
    return {
        "endpoint_available": True,
        "arguments": {
            "price": str(bars[-1]["row"][4]),
            "index": str(indices[closing]),
            "median_basis": str(baseline),
            "symbol": symbol,
            "activity_id": episode,
            "ledger_identity": ledger,
            "data_ms": closing - 1,
            "decision_ms": decision,
            "price_received_ms": bars[-1]["received_ms"],
            "index_received_ms": record["received_ms"],
        },
        "source_minute_boundary_ms": closing,
    }


def activities(source, scratch):
    """完整重放两个原账户来恢复活动权限与拒绝，不从最终成交倒推筛选机会。"""
    protocol, checkpoint = read(source / "protocol.json"), read(source / "checkpoint.json")
    require_loaded_sources(protocol)
    models = {
        name: PaperAccount(
            protocol["symbols"], int(name), period_end_ms=protocol["simulation_end_ms"]
        )
        for name in ("3", "10")
    }
    results = {name: {} for name in models}
    cursor = SourceCursor(
        segment_entries(source / "public_events"),
        protocol["symbols"],
        scratch / "financial_originals.db",
    )
    try:
        for _, wrapper, _ in segment_entries(source / "account_frames"):
            frame = wrapper["frame"]
            cursor.advance(
                wrapper["source_prefix"],
                now=frame["actual_ms"],
                closing=frame["canonical_close_ms"],
            )
            for name, account in models.items():
                ledger = sha(source / "protocol.json") + ":slippage-bps-" + name
                before = {
                    p["episode"]: (s, p["direction"])
                    for s, p in {**account.pending, **account.positions}.items()
                }
                fills_before, declines_before = len(account.fills), len(account.declines)
                state = account.apply(frame)
                created = [
                    p | {"symbol": s}
                    for s, p in state["pending"].items()
                    if p["episode"] not in results[name]
                ]
                fresh_declines, fresh_fills = (
                    state["declines"][declines_before:],
                    state["fills"][fills_before:],
                )
                created.extend(
                    {
                        "episode": d["episode"],
                        "symbol": d["symbol"],
                        "direction": frame["evidence"][d["symbol"]]["direction"],
                        "signal_ms": frame["actual_ms"],
                        "due_ms": None,
                    }
                    for d in fresh_declines
                    if d["episode"] and d["episode"] not in results[name]
                )
                for p in created:
                    results[name][p["episode"]] = {
                        "symbol": p["symbol"],
                        "direction": p["direction"],
                        "activity_id": p["episode"],
                        "ledger_identity": ledger,
                        "signal_ms": p["signal_ms"],
                        "scheduled_boundary_ms": p["due_ms"],
                        "fills": [],
                        "declines": [],
                        "points": [],
                    }
                    before[p["episode"]] = p["symbol"], p["direction"]
                for fill in fresh_fills:
                    results[name][fill["episode"]]["fills"].append(fill)
                for decline in fresh_declines:
                    if decline["episode"]:
                        results[name][decline["episode"]]["declines"].append(decline)
                for episode, (symbol, _) in before.items():
                    stages = []
                    if any(p["episode"] == episode for p in created):
                        stages.append("signal")
                    stages.extend(
                        f["action"].lower() for f in fresh_fills if f["episode"] == episode
                    )
                    stages.extend(d["reason"] for d in fresh_declines if d["episode"] == episode)
                    point = price_point(cursor, frame, symbol, episode, ledger)
                    results[name][episode]["points"].append(
                        {
                            "frame_actual_ms": frame["actual_ms"],
                            "stages": stages or ["held_or_pending"],
                            **point,
                        }
                    )
        cursor.finish()
        for name, account in models.items():
            if account.checkpoint() != checkpoint["accounts"][name]:
                raise ValueError("活动权限重放未精确恢复原账户")
            observed = set(results[name])
            required = {
                v["episode"]
                for key in ("fills", "declines")
                for v in checkpoint["accounts"][name][key]
                if v["episode"]
            }
            required.update(p["episode"] for p in checkpoint["accounts"][name]["pending"].values())
            if observed != required:
                raise ValueError("原账户活动遗漏或混入其他账户")
        return results, {name: model.metrics() for name, model in models.items()}
    finally:
        cursor.close()


def run(source, audited, captures, output):
    """独立审计绑定后逐活动配对；代数变化和原账本损益分列，不能作为新策略回测。"""
    source, audited, output = Path(source), Path(audited), Path(output)
    verify_receipts(source)
    verify_receipts(audited)
    binding, audited_result = (
        read(audited / "run_binding.json"),
        read(audited / "audit/result.json"),
    )
    metadata = read(source / "prefix_snapshot.json")
    if (
        binding["exit_code"] != 0
        or binding["audited_snapshot_receipts_sha256"] != sha(source / "receipts.json")
        or binding["audit_receipts_sha256"] != sha(audited / "audit/receipts.json")
    ):
        raise ValueError("原账户独立审计没有绑定当前固定前缀")
    if (
        audited_result["account_frames_verified"] != metadata["frame_prefix"]["event_count"]
        or audited_result["source_records_verified"] != metadata["source_prefix"]["event_count"]
    ):
        raise ValueError("独立审计未覆盖当前完整前缀")
    runtime = {
        "python": sys.version,
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("requests", "websockets", "pandas", "numpy")
        },
    }
    if runtime != read(source / "protocol.json")["runtime"]:
        raise ValueError("活动重放运行库与原研究冻结版本不同")
    paths = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        if (name == "tradingagents" or name.startswith("tradingagents.")) and getattr(
            module, "__file__", None
        ):
            paths.add(Path(module.__file__).resolve())
    identities = {str(path): sha(path) for path in paths}
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "helper_sources"
    package = PROJECT_ROOT
    for path in paths:
        target = archive / path.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    save(
        output / "run_protocol.json",
        {
            "schema": "cz-causal-pairing-run.v1",
            "source_sha256": identities,
            "runtime": runtime,
            "sqlite_version": sqlite3.sqlite_version,
            "financial_snapshot_receipts_sha256": sha(source / "receipts.json"),
            "financial_audit_receipts_sha256": sha(audited / "receipts.json"),
            "observer_receipts_sha256": {
                str(root): sha(Path(root) / "receipts.json") for root in captures
            },
            "orders_enabled": False,
        },
    )
    with TemporaryDirectory(prefix="cz-causal-pairing-") as directory:
        scratch = Path(directory)
        result, metrics = activities(source, scratch)
        messages = ReceivedMessages(
            captures, scratch / "observations.db", sha(source / "protocol.json")
        )
        try:
            for ledger in result.values():
                for activity in ledger.values():
                    for point in activity["points"]:
                        if not point["endpoint_available"]:
                            continue
                        args = point["arguments"]
                        quote, qe = messages.latest(
                            "quote", args["symbol"], args["data_ms"], args["decision_ms"]
                        )
                        trade, te = messages.latest(
                            "aggregate_trade", args["symbol"], args["data_ms"], args["decision_ms"]
                        )
                        point["diagnostic"] = endpoint(
                            **args, quote=quote, quote_epoch=qe, trade=trade, trade_epoch=te
                        )
                    first, last = activity["points"][0], activity["points"][-1]
                    activity["first_to_last_changes"] = (
                        changes(first["diagnostic"], last["diagnostic"], activity["direction"])
                        if first.get("diagnostic")
                        and last.get("diagnostic")
                        and first["diagnostic"]["data_ms"] < last["diagnostic"]["data_ms"]
                        else None
                    )
            roots = [
                {"root": s["root"], "receipt_sha256": s["receipt_sha256"], "audit": s["audit"]}
                for s in messages.sources
            ]
        finally:
            messages.close()
    summary = {}
    for name, ledger in result.items():
        points = [p for a in ledger.values() for p in a["points"]]
        diagnostics = [p["diagnostic"] for p in points if "diagnostic" in p]
        summary[name] = {
            "activities": len(ledger),
            "activities_with_open_fills": sum(
                any(f["action"] == "OPEN" for f in a["fills"]) for a in ledger.values()
            ),
            "points": len(points),
            "unavailable_reference_points": len(points) - len(diagnostics),
            "known_quote_points": sum(p["quote_unknown_reason"] is None for p in diagnostics),
            "known_age_proxy_points": sum(
                p["trade_age_unknown_reason"] is None for p in diagnostics
            ),
            "quote_unknown_reasons": dict(
                Counter(p["quote_unknown_reason"] for p in diagnostics if p["quote_unknown_reason"])
            ),
            "trade_age_unknown_reasons": dict(
                Counter(
                    p["trade_age_unknown_reason"]
                    for p in diagnostics
                    if p["trade_age_unknown_reason"]
                )
            ),
            "original_account_metrics": metrics[name],
        }
    record = {
        "schema": "cz-causal-microstructure-pairing.v1",
        "financial_snapshot_receipts_sha256": sha(source / "receipts.json"),
        "financial_audit_receipts_sha256": sha(audited / "receipts.json"),
        "observer_sources": roots,
        "summary": summary,
        "activities": result,
        "original_strategy_unchanged": True,
        "account_returns_reselected": False,
        "causal_artifact_proven": False,
        "stable_profitability_proven": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    if any(sha(path) != expected for path, expected in identities.items()):
        raise ValueError("活动配对期间源码发生变化")
    record["run_protocol_sha256"] = sha(output / "run_protocol.json")
    save(output / "result.json", record)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    print(
        {
            name: {k: v for k, v in value.items() if k != "original_account_metrics"}
            for name, value in summary.items()
        }
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financial-prefix", type=Path, required=True)
    parser.add_argument("--financial-audit", type=Path, required=True)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.financial_prefix.resolve(),
        args.financial_audit.resolve(),
        [p.resolve() for p in args.capture],
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
