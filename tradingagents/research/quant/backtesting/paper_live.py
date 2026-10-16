"""限时贯通固定全集、真实公开输入和两成本模拟账户，支持原目录恢复。"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import shutil
import sys
import time
from pathlib import Path

from tradingagents.dataflows.research.live_capture import MINUTE, available_rows, body, clock_valid
from tradingagents.dataflows.research.public_http import PublicHTTP
from tradingagents.dataflows.research.stream import closed_candle, unwrap
from tradingagents.research.quant.backtesting.bidirectional import HYPOTHESIS, evidence, index_jobs
from tradingagents.research.quant.backtesting.continuous_feed import (
    ROTATION_SECONDS,
    VERSION,
    continuous_stream,
)
from tradingagents.research.quant.backtesting.forward_inputs import Bars, venue_contracts
from tradingagents.research.quant.backtesting.forward_pilot import stream
from tradingagents.research.quant.backtesting.forward_startup import (
    active_members,
    common_stream_close,
    merge_history,
    require_confirmed_conflicts,
)
from tradingagents.research.quant.backtesting.paper_inputs import covers, funding_window
from tradingagents.research.quant.backtesting.paper_store import StudyStore
from tradingagents.research.quant.backtesting.runtime_inputs import (
    OWNED_MARK_WAIT_VERSION,
    received_clock,
    wait_for_owned_marks,
)
from tradingagents.research.quant.backtesting.segmented_chain import FORMAT
from tradingagents.research.quant.paths import source_path
from tradingagents.research.quant.strategy_audit import read, save, sha


def milliseconds():
    return time.time_ns() // 1_000_000


def bootstrap_expired(status, initial_ms, now_ms, ending_ms):
    """每个真实连接代次重新计预热上限，原研究结束时间保持固定。"""
    return now_ms < ending_ms and now_ms - status.get("connection_started_ms", initial_ms) > 180_000


def fresh_marks(bars, closing, now):
    """缺价保留缺失，不用之前分钟的价格填充当前估值。"""
    return {
        s: {k: value[k] for k in ("row", "received_ms", "source_sha256")}
        for s in bars.values
        if (value := bars.at(s, closing, now)) is not None
    }


def trade_coverage(account, trade):
    """分别核验平仓后首轮查询与稍后复查，不能因另有持仓重复查旧交易。"""
    windows = account.funding_windows[trade["symbol"]]
    early = [w for w in windows if w["requested_ms"] <= trade["exit_ms"] + MINUTE]
    late = [w for w in windows if w["requested_ms"] >= trade["exit_ms"] + MINUTE + 1000]
    return covers(early, trade["entry_ms"] + 1, trade["exit_ms"]), covers(
        late, trade["entry_ms"] + 1, trade["exit_ms"]
    )


def require_recoverable_funding(accounts, now):
    """首轮查询时序已经无法补回时停止研究，不无限重查或虚构完整费用。"""
    for account in accounts.values():
        for trade in account.trades:
            if now > trade["exit_ms"] + MINUTE and not trade_coverage(account, trade)[0]:
                raise ValueError("平仓后首轮资金费确认期限已错过，账本保留为未完成")


def funding_jobs(accounts, ending):
    """只查真正有归属或尚待复查的库存，覆盖其完整暴露寿命。"""
    needs = {}
    for account in accounts.values():
        for position in account.positions.values():
            s = position["symbol"]
            needs[s] = min(needs.get(s, ending), position["entry_ms"] + 1)
        for trade in account.trades:
            s = trade["symbol"]
            if not all(trade_coverage(account, trade)):
                needs[s] = min(needs.get(s, ending), trade["entry_ms"] + 1)
    return [
        ("funding", s, {"symbol": s, "startTime": start, "endTime": ending, "limit": 1000})
        for s, start in sorted(needs.items())
        if start <= ending
    ]


def exit_wakeup(accounts, ending):
    """时间退出不等全量指数查询；费用复查在退出后另行唤醒。"""
    times = []
    for account in accounts.values():
        for position in account.positions.values():
            times.extend([position["entry_ms"] + 20 * MINUTE, ending])
            times.extend(position[k] for k in ("risk_due_ms", "profit_due_ms") if k in position)
        for trade in account.trades:
            early, late = trade_coverage(account, trade)
            if not early:
                times.append(trade["exit_ms"] + 1001)
            elif not late:
                times.append(trade["exit_ms"] + MINUTE + 1000)
        if account.clock is None or account.clock < ending:
            times.append(ending)
    return min(times) if times else None


async def history_records(client, journal, jobs, closing, *, purpose=None):
    """同一绑定分钟最多请求三次，只重试失败或缺少共同终值的成员。"""
    remaining, retained = list(jobs), []
    for attempt in range(3):
        records = await asyncio.to_thread(client.batch, remaining)
        for record in records:
            if purpose is not None:
                record["purpose"] = purpose
            journal.add(record)
        retained.extend(records)
        completed = {
            r["symbol"]
            for r in records
            if any(
                row[6] + 1 == closing
                for row in available_rows(
                    r, decision_ms=milliseconds(), cutoff_ms=closing, clock_verified=True
                )
            )
        }
        remaining = [job for job in remaining if job[1] not in completed]
        if not remaining:
            break
        journal.add(
            {
                "kind": "startup_history_availability_retry",
                "received_ms": milliseconds(),
                "canonical_close_ms": closing,
                "completed_attempts": attempt + 1,
                "remaining_symbols": [j[1] for j in remaining],
                "maximum_attempts": 3,
            }
        )
        if attempt < 2:
            await asyncio.sleep(attempt + 1)
    return retained


async def warm_history(client, journal, bars, symbols, closing, members):
    """精确交接历史与推送，每个冲突点都需稍后官方原文确认。"""
    jobs = [
        ("klines", s, {"symbol": s, "interval": "1m", "limit": 300, "endTime": closing - 1})
        for s in symbols
    ]
    records = await history_records(client, journal, jobs, closing)
    merged = merge_history(bars, records, closing=closing, now_ms=milliseconds())
    if merged["conflicts"]:
        journal.add(
            {
                "kind": "startup_public_value_conflict",
                "received_ms": milliseconds(),
                "conflicts": merged["conflicts"],
                "decisions_blocked": True,
            }
        )
        await asyncio.sleep(max(0, (closing + MINUTE + 1000 - milliseconds()) / 1000))
        confirm_jobs = [
            ("klines", s, {"symbol": s, "interval": "1m", "limit": 300, "endTime": closing - 1})
            for s in sorted({v["symbol"] for v in merged["conflicts"]})
        ]
        confirmations = await history_records(
            client, journal, confirm_jobs, closing, purpose="STARTUP_CONSENSUS_RECHECK_ONLY"
        )
        second = merge_history(bars, confirmations, closing=closing, now_ms=milliseconds())
        require_confirmed_conflicts(merged["conflicts"], second)
        merged["overlap"].update(second["overlap"])
    if not set(members) <= merged["overlap"]:
        raise ValueError("启动历史和实时推送没有完整交接")
    return {
        "binding_ms": closing,
        "stream_overlap_members": len(merged["overlap"]),
        "startup_conflicts": len(merged["conflicts"]),
    }


async def capture_frame(
    client, store, bars, metadata, closing, phase, generation, index_records=None
):
    """先构造方向证据，再查询资金费及报价，不把迟到数据写回旧决策。"""
    deadline = min(
        store.protocol["simulation_end_ms"] + 3 * MINUTE,
        closing + MINUTE if phase == "decision" else milliseconds() + 30_000,
    )
    preparation = None
    if (
        phase == "exit_only"
        and store.protocol.get("account_preparation_version") == OWNED_MARK_WAIT_VERSION
    ):
        held = {s for account in store.accounts.values() for s in account.positions}
        if held:
            preparation = await wait_for_owned_marks(bars, held, closing, store.source, deadline)
            if preparation["reason"] == "LOCAL_CLOCK_MOVED_BACKWARDS":
                raise ValueError("退出估值等待期间本地时钟倒退")
    before = await received_clock(client, store.source, deadline)
    if metadata is None or milliseconds() - metadata.get("received_ms", 0) >= 4 * MINUTE:
        metadata = await asyncio.to_thread(client.request, "exchange", None, {})
        store.source.add(metadata)
    evaluated_ms = milliseconds()
    evaluated = {
        s: {"ready": False, "reason": "EXIT_ONLY_NO_NEW_DIRECTION"}
        for s in store.protocol["symbols"]
    }
    if phase == "decision":
        evaluated = {
            r["symbol"]: evidence(
                bars,
                r["symbol"],
                r,
                closing=closing,
                decision_ms=evaluated_ms,
                clock_verified=clock_valid(before),
            )
            for r in index_records
        }
    records = await asyncio.to_thread(
        client.batch, funding_jobs(store.accounts, milliseconds() - 1001)
    )
    valid, errors = [], []
    for record in records:
        store.source.add(record)
        try:
            funding_window(record, milliseconds())
            valid.append(record)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(
                {
                    "reason": "FUNDING_INPUT_UNAVAILABLE",
                    "symbol": record.get("symbol"),
                    "error": str(exc),
                }
            )
    quotes = await asyncio.to_thread(client.request, "quotes", None, {})
    store.source.add(quotes)
    after = await received_clock(client, store.source, deadline)
    now = milliseconds()
    if not clock_valid(before) or not clock_valid(after):
        raise ValueError("账户帧前后服务端时钟核验失败")
    if phase == "decision" and now >= closing + MINUTE:
        phase = "exit_only"
        errors.append({"reason": "MISSED_REAL_DECISION_DEADLINE"})
        closing = (now - 1000) // MINUTE * MINUTE
    return {
        "actual_ms": now,
        "canonical_close_ms": closing,
        "phase": phase,
        "source_generation": generation,
        "clock_verified": True,
        "marks": fresh_marks(bars, closing, now),
        "evidence": evaluated,
        "contracts": metadata,
        "quotes": quotes,
        "funding": valid,
        "operational_errors": errors,
        "owned_mark_preparation": preparation,
    }, metadata


async def startup_identity(client, journal, fixed):
    """先核验时钟，再有限获取完整目录，所有失败响应保留原文。"""
    for attempt in range(3):
        clock = await asyncio.to_thread(client.request, "clock", None, {})
        journal.add(clock)
        if clock_valid(clock):
            break
        if attempt < 2:
            await asyncio.sleep(attempt + 1)
    else:
        raise ValueError("启动服务端时钟无法核验")
    for attempt in range(3):
        metadata = await asyncio.to_thread(client.request, "exchange", None, {})
        journal.add(metadata)
        contracts = venue_contracts(metadata, fixed, decision_ms=milliseconds())
        if contracts is not None:
            return metadata, contracts
        journal.add(
            {
                "kind": "startup_identity_retry",
                "received_ms": milliseconds(),
                "completed_attempts": attempt + 1,
                "maximum_attempts": 3,
            }
        )
        if attempt < 2:
            await asyncio.sleep(attempt + 1)
    raise ValueError("固定全集的公开身份在有限重试后仍未覆盖")


async def live(store, client):
    """有限采样保持原绝对结束时间，恢复只推进未来，不重写旧决策。"""
    symbols, ending = store.protocol["symbols"], store.protocol["simulation_end_ms"]
    fixed = store.protocol["contracts"]
    bars, pending, stop = Bars(symbols), [], asyncio.Event()
    status = {
        "subscription_confirmed": False,
        "messages": 0,
        "closed_messages": 0,
        "unfinished_discarded": 0,
        "error": None,
    }
    producer = asyncio.create_task(
        continuous_stream(
            symbols,
            pending,
            store.source,
            stop,
            status,
            rotation_seconds=store.protocol.get("feed_rotation_seconds", ROTATION_SECONDS),
            generation_prefix=store.session["id"],
        )
    )
    startup = milliseconds()
    bootstrap_anchor = startup
    metadata, contracts = None, None
    warm = indices = finance = None
    loaded, next_close = False, None
    restart_pending = store.session["resume"] and store.frames.count > 0
    first_count = store.frames.count
    warm_result, missing, failure, retry_after = None, [], None, 0
    next_disk_check = startup
    try:
        metadata, contracts = await startup_identity(client, store.source, fixed)
        while milliseconds() < ending + 3 * MINUTE:
            now = milliseconds()
            if now >= next_disk_check:
                if shutil.disk_usage(store.root).free < 2 * 1024**3:
                    raise ValueError("研究日志磁盘余量不足二GiB，保留状态并停止")
                next_disk_check = now + MINUTE
            remaining = []
            for record in pending:
                value = unwrap(body(record))
                if now < value["k"]["T"] + 1001:
                    remaining.append(record)
                    continue
                parsed = closed_candle(record, decision_ms=now, clock_verified=True)
                if parsed:
                    bars.add(
                        parsed[0],
                        parsed[1],
                        received_ms=record["received_ms"],
                        source_sha256=record["body_sha256"],
                        completed_verified=True,
                    )
            pending[:] = remaining
            if status.get("error"):
                raise ValueError("公开推送连接终止：" + str(status["error"]))
            current_generation = status.get("connection_generation")
            if finance is not None and finance.done():
                frame, metadata = finance.result()
                frame["actual_ms"] = milliseconds()
                if (
                    frame["phase"] == "decision"
                    and frame["actual_ms"] >= frame["canonical_close_ms"] + MINUTE
                ):
                    frame["phase"] = "exit_only"
                    frame["canonical_close_ms"] = (frame["actual_ms"] - 1000) // MINUTE * MINUTE
                    frame["marks"] = fresh_marks(
                        bars, frame["canonical_close_ms"], frame["actual_ms"]
                    )
                    frame["operational_errors"].append({"reason": "DECISION_DEADLINE_EXCEEDED"})
                frame["source_restart"] = restart_pending
                frame["marks"] = fresh_marks(bars, frame["canonical_close_ms"], frame["actual_ms"])
                if frame["phase"] == "decision" and set(frame["marks"]) != set(symbols):
                    raise ValueError("完整决策缺少固定全集的实际收盘原文")
                store.commit(frame)
                restart_pending = False
                print(
                    f"账户帧已落盘：{store.frames.count}，{frame['phase']}，两成本持仓 {[len(a.positions) for a in store.accounts.values()]}。",
                    flush=True,
                )
                finance = None
                retry_after = milliseconds() + 1000
                bars.prune(frame["canonical_close_ms"])
                contracts = (
                    venue_contracts(metadata, fixed, decision_ms=milliseconds()) or contracts
                )
            if warm is None and not loaded and now < ending and status["subscription_confirmed"]:
                closing, members = common_stream_close(bars, contracts, now_ms=now)
                if (
                    closing is not None
                    and closing >= (status["connection_started_ms"] // MINUTE + 1) * MINUTE
                ):
                    store.source.add(
                        {
                            "kind": "startup_warmup_binding",
                            "received_ms": now,
                            "canonical_close_ms": closing,
                            "stream_members": len(members),
                        }
                    )
                    warm = asyncio.create_task(
                        warm_history(client, store.source, bars, symbols, closing, members)
                    )
            if warm is not None and warm.done() and not loaded:
                warm_result = warm.result()
                loaded = True
                next_close = (milliseconds() // MINUTE + 1) * MINUTE
                print(f"真实预热已交接：{warm_result}。", flush=True)
            if not loaded and bootstrap_expired(
                {
                    "connection_started_ms": max(
                        status.get("connection_started_ms", startup), bootstrap_anchor
                    )
                },
                startup,
                now,
                ending,
            ):
                raise ValueError("完整共同分钟预热超过限时")
            if finance is None:
                require_recoverable_funding(store.accounts, now)
            wake = exit_wakeup(store.accounts, ending)
            if finance is None and wake is not None and now >= max(wake, retry_after):
                closing = (now - 1000) // MINUTE * MINUTE
                finance = asyncio.create_task(
                    capture_frame(
                        client, store, bars, metadata, closing, "exit_only", current_generation
                    )
                )
            if indices is not None and indices.done() and finance is None:
                closing, records, source_generation = indices.result()
                for record in records:
                    store.source.add(record)
                if milliseconds() < ending:
                    finance = asyncio.create_task(
                        capture_frame(
                            client,
                            store,
                            bars,
                            metadata,
                            closing,
                            "decision",
                            source_generation,
                            records,
                        )
                    )
                indices = None
            if (
                loaded
                and now < ending
                and indices is None
                and next_close is not None
                and now >= next_close + 1000
            ):
                if now >= next_close + MINUTE:
                    missing.append(
                        {
                            "closing_ms": next_close,
                            "recorded_ms": now,
                            "reason": "MISSED_SOURCE_MINUTE",
                        }
                    )
                    store.source.add(
                        {"kind": "missed_source_minute", "received_ms": now, **missing[-1]}
                    )
                    loaded, restart_pending, warm, next_close = False, True, None, None
                    bootstrap_anchor = now
                elif all(
                    bars.at(s, next_close, now) is not None
                    for s in active_members(contracts, next_close)
                ):

                    async def collect(closing, source_generation):
                        return (
                            closing,
                            await asyncio.to_thread(client.batch, index_jobs(symbols, closing)),
                            source_generation,
                        )

                    indices = asyncio.create_task(collect(next_close, current_generation))
                    next_close += MINUTE
            if (
                now >= ending
                and finance is None
                and all(
                    not a.positions
                    and not a.pending
                    and a.funding_verified()
                    and a.clock is not None
                    and a.clock >= ending
                    for a in store.accounts.values()
                )
            ):
                break
            await asyncio.sleep(0.1)
        if any(
            a.positions or a.pending or not a.funding_verified() for a in store.accounts.values()
        ):
            raise ValueError("有限采样结束后退出或资金费复查未完成")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "recorded_ms": milliseconds()}
        store.source.add(
            {"kind": "paper_process_failure", "received_ms": milliseconds(), **failure}
        )
    finally:
        for task in (warm, indices, finance):
            if task is not None:
                try:
                    result = await task
                    if task is indices:
                        for record in result[1]:
                            store.source.add(record)
                except Exception:
                    pass
        stop.set()
        await producer
        store.source.seal()
    return {
        "schema": "cz-live-paper-probe.v1",
        "failure": failure,
        "new_frames": store.frames.count - first_count,
        "total_frames": store.frames.count,
        "warmup": warm_result,
        "missed_minutes": missing,
        "stream": status,
        "runtime_version": store.protocol.get("runtime_version"),
        "profile": store.protocol.get("profile", "finite"),
        "finite_probe_only": store.protocol.get("finite_probe_only", True),
        "diagnostic_soak_only": store.protocol.get("diagnostic_soak_only", False),
        "accounts": {name: a.metrics() for name, a in store.accounts.items()},
        "long_term_test_registered": False,
        "stable_profitability_proven": False,
        "orders_enabled": False,
        "production_eligible": False,
    }


def run(
    panel,
    review,
    output,
    duration,
    resume,
    storage="flat-jsonl",
    profile="finite",
    rotation_seconds=ROTATION_SECONDS,
):
    """研究目录和源码冻结，恢复不能延长原结束时间或变更规则。"""
    if profile not in {"finite", "diagnostic-soak"} or not (
        180 <= duration <= 1800 if profile == "finite" else 3600 <= duration <= 86400
    ):
        raise ValueError("有限验证限三至三十分钟，诊断持续采样限一至二十四小时")
    if not 180 <= rotation_seconds <= ROTATION_SECONDS:
        raise ValueError("公开连接轮换间隔必须为三分钟至二十三小时")
    if profile == "diagnostic-soak" and storage != FORMAT:
        raise ValueError("持续诊断只允许分段压缩存储")
    manifest, proposal = read(panel / "manifest.json"), read(review)
    if (
        not proposal.get("model_inference_performed")
        or proposal.get("production_eligible") is not False
        or not any(h["id"] == HYPOTHESIS for h in proposal["review"]["hypotheses"])
    ):
        raise ValueError("双向假说必须来自对应真实模型提案")
    fixed = {
        s: c for s, c in manifest["contracts"].items() if c["asset_class"] == "TRADIFI_PERPETUAL"
    }
    paths = [
        Path(__file__).resolve(),
        *(
            Path(f.__code__.co_filename).resolve()
            for f in (
                evidence,
                stream,
                continuous_stream,
                received_clock,
                merge_history,
                closed_candle,
                funding_window,
                StudyStore.commit,
                PublicHTTP.request,
                Bars.add,
                clock_valid,
                save,
            )
        ),
    ]
    paths.extend(source_path(p) for p in ("paper_account.py", "requirements-forward.txt"))
    paths.append(source_path("profit_factor_segmented_chain.py"))
    runtime = {
        "python": sys.version,
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("requests", "websockets", "pandas", "numpy")
        },
    }
    if resume:
        protocol = read(output / "protocol.json")
        if (
            protocol["manifest_sha256"] != sha(panel / "manifest.json")
            or protocol["proposal_sha256"] != sha(review)
            or protocol["symbols"] != sorted(fixed)
            or protocol["duration_seconds"] != duration
            or protocol["source_sha256"] != {str(p): sha(p) for p in paths}
            or protocol["runtime"] != runtime
            or protocol.get("storage", "flat-jsonl") != storage
            or protocol.get("runtime_version") != VERSION
            or protocol.get("profile") != profile
            or protocol.get("feed_rotation_seconds") != rotation_seconds
            or protocol.get("account_preparation_version") != OWNED_MARK_WAIT_VERSION
        ):
            raise ValueError("恢复时输入、源码或原时长身份不同")
    else:
        now = milliseconds()
        parent = next(p for p in (output.parent, *output.parents) if p.exists())
        available = shutil.disk_usage(parent).free
        if profile == "diagnostic-soak" and available < 10 * 1024**3:
            raise ValueError("持续诊断启动前磁盘余量须至少十GiB")
        protocol = {
            "schema": "cz-live-paper-probe.v1",
            "hypothesis": HYPOTHESIS,
            "execution_version": "received_quote_paper_v1",
            "symbols": sorted(fixed),
            "contracts": fixed,
            "frozen_ms": now,
            "simulation_end_ms": now + duration * 1000,
            "duration_seconds": duration,
            "manifest_sha256": sha(panel / "manifest.json"),
            "proposal_sha256": sha(review),
            "source_sha256": {str(p): sha(p) for p in paths},
            "runtime": runtime,
            "storage": storage,
            "runtime_version": VERSION,
            "profile": profile,
            "feed_rotation_seconds": rotation_seconds,
            "account_preparation_version": OWNED_MARK_WAIT_VERSION,
            "disk_free_bytes_at_start": available,
            "finite_probe_only": profile == "finite",
            "diagnostic_soak_only": profile == "diagnostic-soak",
            "long_term_test_registered": False,
            "orders_enabled": False,
            "production_eligible": False,
        }
    store = StudyStore(output, protocol, resume=resume)
    client = PublicHTTP()
    try:
        if not resume:
            (output / "source").mkdir()
            for path in paths:
                (output / "source" / path.name).write_bytes(path.read_bytes())
        result = asyncio.run(live(store, client))
        for path in paths:
            if sha(path) != protocol["source_sha256"][str(path)]:
                raise ValueError("模拟进程运行期间源码发生变化")
        save(output / f"result_{store.session['id']}.json", result)
        store.write_checkpoint()
        print({k: v for k, v in result.items() if k not in ("accounts", "stream")}, flush=True)
        return result
    finally:
        client.close()
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--storage", choices=("flat-jsonl", FORMAT), default="flat-jsonl")
    parser.add_argument("--profile", choices=("finite", "diagnostic-soak"), default="finite")
    parser.add_argument("--feed-rotation-seconds", type=int, default=ROTATION_SECONDS)
    args = parser.parse_args()
    result = run(
        args.panel.resolve(),
        args.review.resolve(),
        args.output.resolve(),
        args.duration_seconds,
        args.resume,
        args.storage,
        args.profile,
        args.feed_rotation_seconds,
    )
    if result["failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
