"""独立把真实公开前缀、因子证据和连续账户联结起来，不授予收益晋级。"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import sqlite3
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlparse

from tradingagents.dataflows.research.live_capture import (
    MINUTE,
    available_rows,
    canonical,
    clock_valid,
    digest,
)
from tradingagents.dataflows.research.stream import closed_candle
from tradingagents.research.quant.backtesting.bidirectional import evidence, index_jobs
from tradingagents.research.quant.backtesting.continuous_feed import VERSION, verify_handoff
from tradingagents.research.quant.backtesting.forward_audit import verify_consensus
from tradingagents.research.quant.backtesting.forward_inputs import Bars
from tradingagents.research.quant.backtesting.isolated_restore import restore_in_isolation
from tradingagents.research.quant.backtesting.paper_account import PaperAccount
from tradingagents.research.quant.backtesting.paper_audit import audit
from tradingagents.research.quant.backtesting.paper_inputs import funding_window
from tradingagents.research.quant.backtesting.paper_recover import transport_patch
from tradingagents.research.quant.backtesting.paper_store import entries
from tradingagents.research.quant.backtesting.runtime_inputs import (
    OWNED_MARK_WAIT_VERSION,
    wait_for_owned_marks,
)
from tradingagents.research.quant.backtesting.segmented_chain import (
    FORMAT,
    load_chain,
    segment_entries,
)
from tradingagents.research.quant.backtesting.source_cursor import SourceCursor
from tradingagents.research.quant.strategy_audit import read, save, sha


def require_original(record, originals):
    """同时核验原文和接收包装，不能只复用原文哈希而伪造收到时刻。"""
    if digest(record) not in originals:
        raise ValueError("财务帧引用的原文包装不在已提交前缀中")


def require_account_contract(state, protocol, name):
    """账户键、完整全集与原结束时间共同绑定，不能互换基础和压力标签。"""
    if name not in {"3", "10"}:
        raise ValueError("未知模拟成本账户")
    expected = PaperAccount(
        protocol["symbols"], int(name), period_end_ms=protocol["simulation_end_ms"]
    ).protocol
    if state["protocol"] != expected:
        raise ValueError("账户成本、固定全集或结束时间与原协议不同")


def require_loaded_sources(protocol, recovery=None):
    """实际载入的公式与模型也必须匹配，不能只核对另一个目录的冻结文件。"""
    expected = {}
    for origin, identity in protocol["source_sha256"].items():
        name = Path(origin).name
        if name in expected and expected[name] != identity:
            raise ValueError("冻结源码存在同名但不同身份")
        expected[name] = identity
    if recovery is not None:
        expected["paper_store.py"] = recovery["recovered_store_sha256"]
    functions = [
        evidence,
        Bars.add,
        clock_valid,
        closed_candle,
        PaperAccount.apply,
        funding_window,
        entries,
        read,
        save,
        sha,
    ]
    if protocol.get("runtime_version") == VERSION:
        functions.append(verify_handoff)
    if protocol.get("storage") == FORMAT:
        functions.append(segment_entries)
    if protocol.get("account_preparation_version") is not None:
        if protocol["account_preparation_version"] != OWNED_MARK_WAIT_VERSION:
            raise ValueError("未知的持仓估值等待版本")
        functions.append(wait_for_owned_marks)
    for function in functions:
        loaded = Path(function.__code__.co_filename).resolve()
        if expected.get(loaded.name) != sha(loaded):
            raise ValueError("实际载入的核验公式或账户模型与冻结源码不同")


def require_preparation(frame, protocol, originals, identities, bars):
    """独立验证声明的等待原文、原期限与实际已知价格，不把等待成功当成盈利资格。"""
    value = frame.get("owned_mark_preparation")
    if value is None:
        return
    require_original(value, identities)
    starts = [
        r
        for r in originals
        if r["kind"] == "owned_mark_wait_started" and r["wait_id"] == value["wait_id"]
    ]
    if len(starts) != 1:
        raise ValueError("估值等待缺少唯一实际开始原文")
    start = starts[0]
    if (
        protocol.get("account_preparation_version") != OWNED_MARK_WAIT_VERSION
        or value["version"] != OWNED_MARK_WAIT_VERSION
        or type(value.get("ready")) is not bool
        or type(value.get("elapsed_monotonic_ms")) not in {int, float}
        or not math.isfinite(value["elapsed_monotonic_ms"])
        or start["version"] != OWNED_MARK_WAIT_VERSION
        or frame["phase"] != "exit_only"
        or value["kind"] != "owned_mark_wait_finished"
        or start["symbols"] != value["symbols"]
        or start["symbols"] != sorted(set(start["symbols"]))
        or not set(start["symbols"]) <= set(protocol["symbols"])
        or start["canonical_close_ms"] != frame["canonical_close_ms"]
        or value["canonical_close_ms"] != frame["canonical_close_ms"]
        or start["maximum_wait_ms"] != 5000
        or start["original_deadline_ms"] != value["original_deadline_ms"]
        or not frame["canonical_close_ms"] + 1000
        <= start["received_ms"]
        <= value["received_ms"]
        <= frame["quotes"]["requested_ms"]
        or value["original_deadline_ms"]
        > min(start["received_ms"] + 30_000, protocol["simulation_end_ms"] + 3 * MINUTE)
        or value["received_ms"] >= value["original_deadline_ms"]
        or value["elapsed_monotonic_ms"] < 0
        or abs(value["received_ms"] - start["received_ms"] - value["elapsed_monotonic_ms"]) > 250
    ):
        raise ValueError("估值等待身份、时序或原期限不合法")
    if value["ready"]:
        if (
            value["reason"] is not None
            or value["missing_symbols"]
            or value["elapsed_monotonic_ms"] > 5000
        ):
            raise ValueError("估值等待成功声明超过固定边界")
        if any(
            bars.at(s, frame["canonical_close_ms"], value["received_ms"]) is None
            for s in value["symbols"]
        ):
            raise ValueError("估值等待成功引用了尚未收到的价格")
    elif value["reason"] != "OWNED_MARK_WAIT_LIMIT_REACHED" or value["elapsed_monotonic_ms"] < 5000:
        raise ValueError("不能在原期限已过或时钟倒退后提交财务帧")


def official_batch(records, symbols, closing):
    """完整冻结全集必须有相同预定区间的真实请求，不能删去失败成员。"""
    selected = {}
    expected = {
        s: {k: [str(v)] for k, v in params.items()} for _, s, params in index_jobs(symbols, closing)
    }
    for record in records:
        s = record.get("symbol")
        if record.get("kind") != "index" or s not in expected or "url" not in record:
            continue
        query = parse_qs(urlparse(record["url"]).query)
        if query != expected[s]:
            continue
        if s in selected:
            raise ValueError("同一决策批次的官方指数请求重复")
        selected[s] = record
    if set(selected) != set(symbols):
        raise ValueError("固定指数全集的请求或失败原文未完整保留")
    return selected


def received_bars(records, symbols, now):
    """按真实会话重建输入缓存；恢复后预热不能伪装成早先已经收到。"""
    bars = Bars(symbols)
    interim = {
        (v["symbol"], v["closing_ms"], v["conflicting_source_sha256"])
        for r in records
        if r["kind"] == "startup_public_value_conflict"
        for v in r["conflicts"]
    }
    for record in records:
        if record["kind"] == "paper_process_session":
            bars = Bars(symbols)
        if record["kind"] == "websocket":
            parsed = closed_candle(record, decision_ms=now, clock_verified=True)
            rows = [parsed] if parsed and now >= parsed[1][6] + 1001 else []
        elif record["kind"] == "klines" and record.get("symbol") in symbols:
            rows = [
                (record["symbol"], row)
                for row in available_rows(
                    record,
                    decision_ms=now,
                    cutoff_ms=(now - 1000) // MINUTE * MINUTE,
                    clock_verified=True,
                )
            ]
        else:
            rows = []
        for symbol, row in rows:
            if (symbol, row[6] + 1, record["body_sha256"]) in interim:
                continue
            bars.add(
                symbol,
                row,
                received_ms=record["received_ms"],
                source_sha256=record["body_sha256"],
                completed_verified=True,
            )
    return bars


def chain_iterator(root, name, storage):
    """不同存储只改变读取器，帧和公开事件仍逐项使用原有哈希身份。"""
    if storage not in {FORMAT, "flat-jsonl"}:
        raise ValueError("未知研究存储版本")
    return segment_entries(root / name) if storage == FORMAT else entries(root / f"{name}.jsonl")


def verify_handoffs(records, symbols):
    """独立从实际前缀取两路原文，不能因连接自报成功而信任交接。"""
    originals = {
        (r["body_sha256"], r.get("connection_generation"), r["received_ms"]): r
        for r in records
        if r["kind"] == "websocket"
    }
    starts = {
        r["generation"]: r["received_ms"]
        for r in records
        if r["kind"] == "public_stream_connection_started"
    }
    events = [r for r in records if r["kind"] == "public_stream_overlap_handoff"]
    for event in events:
        verify_handoff(event, originals, symbols, starts)
    return len(events)


def _run(root, output, streaming, scratch, resources):
    """核对原文前缀和全部方向公式，再分别重建两种成本的完整财务。"""
    protocol = read(root / "protocol.json")
    helpers = {
        Path(f.__code__.co_filename).resolve()
        for f in (
            audit,
            verify_consensus,
            transport_patch,
            SourceCursor.advance,
            restore_in_isolation,
            verify_handoff,
        )
    }
    helpers.add(Path(__file__).resolve())
    helper_identities = {str(p): sha(p) for p in sorted(helpers)}
    runtime = {
        "python": sys.version,
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("requests", "websockets", "pandas", "numpy")
        },
    }
    if runtime != protocol["runtime"]:
        raise ValueError("独立审计运行库与原采样冻结版本不同")
    if (
        (protocol.get("finite_probe_only") is True)
        == (protocol.get("diagnostic_soak_only") is True)
        or protocol["orders_enabled"] is not False
        or protocol["long_term_test_registered"] is not False
    ):
        raise ValueError("这里只审计有限公开模拟，不处理生产库存")
    technical_recovery = None
    for filename, expected in protocol["source_sha256"].items():
        archived = root / "source" / Path(filename).name
        if sha(archived) != expected:
            raise ValueError("重建所需源码与实际运行冻结身份不同")
        if sha(Path(filename)) != expected:
            technical_recovery = transport_patch(root)
            manifests = [read(p) for p in root.glob("transport_recovery_*/manifest.json")]
            if not any(
                all(m.get(k) == v for k, v in technical_recovery.items()) for m in manifests
            ):
                raise ValueError("技术读取修复没有追加的真实运行身份")
    require_loaded_sources(protocol, technical_recovery)
    storage = protocol.get("storage", "flat-jsonl")
    checkpoint = read(root / "checkpoint.json")
    cursor = None
    if streaming:
        cursor = SourceCursor(
            chain_iterator(root, "public_events", storage),
            protocol["symbols"],
            scratch / "originals.db",
        )
        resources.callback(cursor.close)
        wrappers = chain_iterator(root, "account_frames", storage)
    else:
        records, source_heads = load_chain(root, "public_events", storage)
        values, frame_heads = load_chain(root, "account_frames", storage)
        wrappers = ((index, value, frame_heads[index + 1]) for index, value in enumerate(values))
    frames, previous_count, previous_sessions, full, exit_only, known_omissions = [], 0, 0, 0, 0, 0
    completed, frame_head = 0, None
    handoffs, previous_missed = 0, 0
    for index, wrapper, current_head in wrappers:
        frame_head = current_head
        completed = index + 1
        frame, prefix = wrapper["frame"], wrapper["source_prefix"]
        now, closing = frame["actual_ms"], frame["canonical_close_ms"]
        count = prefix["event_count"]
        if streaming:
            cursor.advance(prefix, now=now, closing=closing)
            original = [r for _, r in cursor.indices] + cursor.clocks + cursor.preparations
            identities, bars, sessions = cursor, cursor.bars, cursor.sessions
            handoffs, missed = cursor.handoffs, cursor.missed_minutes
        else:
            if (
                not previous_count <= count < len(source_heads)
                or source_heads[count] != prefix["event_chain_sha256"]
            ):
                raise ValueError("账户帧没有对应的单调公开原文前缀")
            original = records[:count]
            if any(r.get("received_ms", 0) > now for r in original):
                raise ValueError("账户前缀包含尚未实际收到的公开输入")
            identities = {digest(r) for r in original}
            verify_consensus(original)
            bars = received_bars(original, protocol["symbols"], now)
            sessions = sum(r["kind"] == "paper_process_session" for r in original)
            handoffs = verify_handoffs(original, protocol["symbols"])
            missed = sum(r["kind"] == "missed_source_minute" for r in original)
        if not now <= wrapper["processing_completed_ms"] <= now + 5000:
            raise ValueError("账户处理完成时间不支持引用的报价时效")
        for record in [frame["contracts"], frame["quotes"], *frame["funding"]]:
            require_original(record, identities)
        for symbol, value in frame["marks"].items():
            expected = bars.at(symbol, closing, now)
            expected = (
                {k: expected[k] for k in ("row", "received_ms", "source_sha256")}
                if expected
                else None
            )
            if canonical(value) != canonical(expected):
                raise ValueError("财务估值不能从首次收到的原文重建")
        known_omissions += sum(
            bars.at(s, closing, now) is not None and s not in frame["marks"]
            for s in protocol["symbols"]
        )
        require_preparation(frame, protocol, original, identities, bars)
        if index and sessions > previous_sessions and frame.get("source_restart") is not True:
            raise ValueError("进程恢复后没有撤销未知活动权限")
        if (
            protocol.get("runtime_version") == VERSION
            and missed > previous_missed
            and frame.get("source_restart") is not True
        ):
            raise ValueError("真实缺失分钟之后没有撤销未知活动权限")
        phase = frame.get("phase", "decision")
        if phase == "decision":
            full += 1
            if protocol.get("runtime_version") == VERSION and set(frame["marks"]) != set(
                protocol["symbols"]
            ):
                raise ValueError("完整决策缺少冻结全集的实际完成分钟")
            if now >= closing + MINUTE or wrapper["processing_completed_ms"] >= closing + MINUTE:
                raise ValueError("完整决策超过原分钟期限")
            sources = official_batch(original, protocol["symbols"], closing)
            stamps = {v["actual_decision_ms"] for v in frame["evidence"].values() if v.get("ready")}
            if len(stamps) > 1:
                raise ValueError("同帧方向证据的实际决策时钟不一致")
            clocks = [r for r in original if r["kind"] == "clock" and clock_valid(r)]
            after = [
                r
                for r in clocks
                if frame["quotes"]["received_ms"] <= r["requested_ms"] and r["received_ms"] <= now
            ]
            before = [r for r in clocks if r["received_ms"] <= frame["quotes"]["requested_ms"]]
            if not before or not after:
                raise ValueError("方向证据与报价没有前后服务端时钟依据")
            evaluation_ms = next(iter(stamps)) if stamps else before[-1]["received_ms"]
            if not before[-1]["received_ms"] <= evaluation_ms <= frame["quotes"]["requested_ms"]:
                raise ValueError("报价不是在真实方向复核之后请求")
            rebuilt = {
                s: evidence(
                    bars, s, r, closing=closing, decision_ms=evaluation_ms, clock_verified=True
                )
                for s, r in sources.items()
            }
            if canonical(rebuilt) != canonical(frame["evidence"]):
                raise ValueError("完整双向方向证据不能由实际收到的原文重建")
        elif phase == "exit_only":
            exit_only += 1
        else:
            raise ValueError("未知账户处理阶段")
        if not streaming:
            frames.append(frame)
        previous_count, previous_sessions = count, sessions
        previous_missed = missed
    if checkpoint["frame_count"] != completed or checkpoint["frame_chain_sha256"] != frame_head:
        raise ValueError("期末检查点尚未覆盖完整已提交账户日志")
    if streaming:
        cursor.finish()

    def account_frames():
        return (
            wrapper["frame"] for _, wrapper, _ in chain_iterator(root, "account_frames", storage)
        )

    finance = {}
    for name in ("3", "10"):
        state = checkpoint["accounts"][name]
        require_account_contract(state, protocol, name)
        finance[name] = audit(account_frames() if streaming else frames, state)
        restored = (
            restore_in_isolation(account_frames(), state)
            if streaming
            else PaperAccount.restore(frames, state)
        )
        if restored.protocol["period_end_ms"] != protocol["simulation_end_ms"]:
            raise ValueError("恢复改变了原预定结束时间")
    result = {
        "schema": "cz-live-paper-independent-audit.v1",
        "source_records_verified": cursor.count if streaming else len(records),
        "account_frames_verified": completed,
        "full_decision_frames": full,
        "exit_only_frames": exit_only,
        "known_but_omitted_marks": known_omissions,
        "process_sessions": cursor.sessions
        if streaming
        else sum(r["kind"] == "paper_process_session" for r in records),
        "streaming_source_verification": streaming,
        "maximum_cached_bars": cursor.maximum_cached_bars if streaming else None,
        "maximum_prefix_increment_events": cursor.maximum_prefix_events if streaming else None,
        "maximum_prefix_increment_bytes": cursor.maximum_prefix_bytes if streaming else None,
        "temporary_index_sqlite_version": sqlite3.sqlite_version if streaming else None,
        "source_identities_and_direction_formulas_verified": True,
        "financial_reconstruction": finance,
        "checkpoint_replay_exact": True,
        "isolated_full_replay": streaming,
        "declared_owned_mark_preparations_verified": protocol.get("account_preparation_version")
        == OWNED_MARK_WAIT_VERSION,
        "overlap_handoffs_verified_in_account_prefix": handoffs,
        "audit_helper_sha256": helper_identities,
        "audit_runtime": runtime,
        "transport_reader_recovery": technical_recovery,
        "long_term_test_registered": False,
        "stable_profitability_proven": False,
        "real_fills_proven": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    helper_root = output / "helper_sources"
    helper_root.mkdir()
    for filename, expected in helper_identities.items():
        path = Path(filename)
        if sha(path) != expected:
            raise ValueError("审计期间辅助源码发生变化")
        target = helper_root / path.name
        target.write_bytes(path.read_bytes())
        if sha(target) != expected:
            raise ValueError("归档的辅助源码身份发生变化")
    save(output / "result.json", result)
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    print(result)
    return result


def run(root, output, streaming=False):
    """临时磁盘成员索引由原文重新生成，任何退出路径均关闭并清理。"""
    with ExitStack() as resources:
        scratch = (
            Path(resources.enter_context(TemporaryDirectory(prefix="cz-paper-audit-")))
            if streaming
            else None
        )
        return _run(root, output, streaming, scratch, resources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--streaming", action="store_true", help="逐前缀核验，避免全部原文和财务帧同时驻留内存"
    )
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve(), args.streaming)


if __name__ == "__main__":
    main()
