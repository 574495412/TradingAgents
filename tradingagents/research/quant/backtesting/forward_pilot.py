"""限时贯通真实公开输入与研究信号，不生成成交、收益或生产资格。"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

from tradingagents.dataflows.research.live_capture import (
    MINUTE,
    body,
    canonical,
    clock_valid,
    digest,
)
from tradingagents.dataflows.research.public_http import PublicHTTP
from tradingagents.dataflows.research.stream import URL, closed_candle, received_record, unwrap
from tradingagents.research.quant.backtesting.forward_inputs import (
    Bars,
    dependency_jobs,
    evaluate_dependencies,
    price_ready,
    snapshot,
    venue_contracts,
)
from tradingagents.research.quant.backtesting.forward_startup import (
    active_members,
    common_stream_close,
    merge_history,
    require_confirmed_conflicts,
)
from tradingagents.research.quant.paths import source_path
from tradingagents.research.quant.reference import market_features as market
from tradingagents.research.quant.strategy_audit import read, save, sha

SEEDS = tuple(range(1729, 1749))
PROBABILITY = 17 / 104


class Activities:
    """初始已活跃及断流后的未知活动不得当作新活动，每个研究臂只接受一次。"""

    def __init__(self):
        self.last_clock, self.previous, self.active = None, set(), {}
        self.used = {
            arm: set() for arm in ("parent", "basis", *(f"random_{seed}" for seed in SEEDS))
        }

    def gap(self):
        self.last_clock = None

    def observe(self, ready, clock):
        current = set(ready)
        contiguous = self.last_clock is not None and clock == self.last_clock + MINUTE
        if not contiguous:
            self.active = {s: {"anchor": clock, "startup_or_gap": True} for s in current}
        else:
            self.active = {s: v for s, v in self.active.items() if s in current}
            for symbol in current - self.previous:
                self.active[symbol] = {"anchor": clock, "startup_or_gap": False}
        self.previous, self.last_clock = current, clock

    def emit(self, evaluated, frame, *, closing, decision_ms):
        emitted = []
        for symbol, evidence in sorted(evaluated.items()):
            activity = self.active.get(symbol)
            if not activity or activity["startup_or_gap"] or not evidence.get("positioning_pass"):
                continue
            episode = f"shock:{symbol}:{activity['anchor']}"
            conditions = {"parent": True, "basis": evidence.get("basis_short_pass", False)}
            for seed in SEEDS:
                uniform = (
                    int.from_bytes(hashlib.sha256(f"{seed}|{episode}".encode()).digest()[:8], "big")
                    / 2**64
                )
                conditions[f"random_{seed}"] = uniform < PROBABILITY
            for arm, condition in conditions.items():
                if not condition or episode in self.used[arm]:
                    continue
                self.used[arm].add(episode)
                emitted.append(
                    {
                        "arm": arm,
                        "episode": episode,
                        "symbol": symbol,
                        "direction": -1,
                        "priority": abs(float(frame.loc[symbol, "ret_20_pct"]))
                        / max(float(frame.loc[symbol, "atr_pct"]), 1e-9),
                        "canonical_close_ms": closing,
                        "actual_decision_ms": decision_ms,
                        "earliest_execution_ms": decision_ms + MINUTE,
                        "orders_enabled": False,
                    }
                )
        return emitted

    def state(self):
        return {
            "last_clock": self.last_clock,
            "previous": sorted(self.previous),
            "active": self.active,
            "used": {k: sorted(v) for k, v in self.used.items()},
        }

    @classmethod
    def restore(cls, state):
        value = cls()
        value.last_clock, value.previous, value.active = (
            state["last_clock"],
            set(state["previous"]),
            state["active"],
        )
        value.used = {k: set(v) for k, v in state["used"].items()}
        if set(value.used) != set(cls().used):
            raise ValueError("活动检查点不属于同一研究臂集合")
        return value


class Journal:
    """顺序原文使用哈希链；每次决策前把对应前缀落盘，不依赖可覆盖状态文件。"""

    def __init__(self, root):
        self.file = (root / "public_events.jsonl").open("xb")
        self.sequence, self.head = 0, None

    def add(self, record):
        entry = {"sequence": self.sequence, "previous_sha256": self.head, "record": record}
        self.file.write(canonical(entry) + b"\n")
        self.head, self.sequence = digest(entry), self.sequence + 1

    def seal(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        return {"event_count": self.sequence, "event_chain_sha256": self.head}

    def close(self):
        self.file.close()


async def keepalive(socket, journal, status, last_tick, now_tick):
    """空PONG控制帧每二十秒维持链路，发送记录单独留痕。"""
    if now_tick - last_tick < 20:
        return last_tick
    await socket.pong(b"")
    status["keepalive_pongs"] = status.get("keepalive_pongs", 0) + 1
    journal.add(
        {
            "kind": "public_websocket_keepalive",
            "received_ms": time.time_ns() // 1_000_000,
            "connection_generation": status["connection_generation"],
            "control_frame": "PONG",
            "market_observation": False,
        }
    )
    return now_tick


async def connection(symbols, pending, journal, stop, status):
    """公开收盘逐条收到即记录，未完成更新只计数，不授予完成权限。"""
    from websockets.asyncio.client import connect

    try:
        async with connect(
            URL,
            open_timeout=15,
            close_timeout=3,
            ping_interval=None,
            max_size=4_194_304,
            max_queue=128,
        ) as socket:
            streams = [f"{s.lower()}@kline_1m" for s in symbols]
            awaiting = set()
            for start in range(0, len(streams), 100):
                identity = f"factor-pilot-{start // 100}"
                awaiting.add(identity)
                await socket.send(
                    json.dumps(
                        {
                            "method": "SUBSCRIBE",
                            "params": streams[start : start + 100],
                            "id": identity,
                        }
                    )
                )
                await asyncio.sleep(0.3)
            last_tick = time.monotonic()
            while not stop.is_set():
                last_tick = await keepalive(socket, journal, status, last_tick, time.monotonic())
                try:
                    raw = await asyncio.wait_for(socket.recv(), timeout=1)
                except TimeoutError:
                    continue
                record = received_record(raw, received_ms=time.time_ns() // 1_000_000)
                record["connection_generation"] = status["connection_generation"]
                value = unwrap(body(record))
                status["messages"] += 1
                if isinstance(value, dict) and "id" in value:
                    journal.add(record)
                    if value.get("result", "absent") is not None or value["id"] not in awaiting:
                        raise ValueError("公开订阅未获得唯一确认")
                    awaiting.remove(value["id"])
                    status["subscription_confirmed"] = not awaiting
                elif isinstance(value, dict) and value.get("e") == "kline":
                    if value.get("k", {}).get("x") is True:
                        journal.add(record)
                        pending.append(record)
                        status["closed_messages"] += 1
                    else:
                        status["unfinished_discarded"] += 1
    except Exception as exc:
        status["connection_error"] = type(exc).__name__
        status["subscription_confirmed"] = False
        status["last_close_details"] = {
            side: {
                "code": getattr(getattr(exc, side, None), "code", None),
                "reason": str(getattr(getattr(exc, side, None), "reason", ""))[:240],
            }
            for side in ("rcvd", "sent")
        }


async def stream(symbols, pending, journal, stop, status):
    """仅在真实连接终止后有限重连，缺口与重连代次明确留痕。"""
    status["transient_disconnects"] = []
    for attempt in range(4):
        if stop.is_set():
            return
        status.update(
            connection_generation=attempt + 1,
            connection_started_ms=time.time_ns() // 1_000_000,
            connection_error=None,
            subscription_confirmed=False,
        )
        await connection(symbols, pending, journal, stop, status)
        if stop.is_set():
            return
        event = {
            "kind": "websocket_disconnect",
            "received_ms": time.time_ns() // 1_000_000,
            "generation": attempt + 1,
            "error": status["connection_error"],
            "close": status.get("last_close_details"),
        }
        status["transient_disconnects"].append(event)
        journal.add(event)
        closed_code = status.get("last_close_details", {}).get("rcvd", {}).get("code")
        invalid_protocol = status["connection_error"] in (
            "ValueError",
            "KeyError",
            "JSONDecodeError",
        ) or closed_code in (1002, 1007, 1008)
        if attempt == 3 or invalid_protocol:
            status["error"] = status["connection_error"]
            stop.set()
            return
        await asyncio.sleep(2**attempt)


async def pilot(manifest, output, duration, journal, client):
    """先完成公开预热，再只处理真实后续分钟；错过截止的分钟不会补决策。"""
    symbols = sorted(manifest["symbols"])
    bars, activities = Bars(symbols), Activities()
    pending, stop = [], asyncio.Event()
    status = {
        "subscription_confirmed": False,
        "messages": 0,
        "closed_messages": 0,
        "unfinished_discarded": 0,
        "error": None,
    }
    producer = asyncio.create_task(stream(symbols, pending, journal, stop, status))
    beginning = time.monotonic()
    warm_close, warmup, initial_members, warm_generation = None, None, [], None
    loaded, warmup_journaled, metadata, next_close = False, False, None, None
    decisions, gaps, emitted, dependency_count, canary_done = [], [], [], 0, False
    startup_conflicts, consensus_rechecks = [], 0
    overlap = set()
    try:
        metadata = await asyncio.to_thread(client.request, "exchange", None, {})
        journal.add(metadata)
        startup_contracts = venue_contracts(
            metadata, manifest["contracts"], decision_ms=time.time_ns() // 1_000_000
        )
        if startup_contracts is None:
            raise ValueError("启动时固定集合的公开合约身份未覆盖")
        while time.monotonic() - beginning < duration and not stop.is_set():
            now = time.time_ns() // 1_000_000
            waiting = []
            for record in pending:
                value = unwrap(body(record))
                if now < value["k"]["T"] + 1001:
                    waiting.append(record)
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
            pending[:] = waiting
            generation = status.get("connection_generation")
            if warmup is not None and generation != warm_generation:
                activities.gap()
                if not warmup.done():
                    await asyncio.sleep(0.1)
                    continue
                if not warmup_journaled:
                    for record in warmup.result():
                        journal.add(record)
                warmup, loaded, warmup_journaled, next_close = None, False, False, None
                overlap.clear()
                metadata = await asyncio.to_thread(client.request, "exchange", None, {})
                journal.add(metadata)
                startup_contracts = venue_contracts(
                    metadata, manifest["contracts"], decision_ms=time.time_ns() // 1_000_000
                )
                if startup_contracts is None:
                    raise ValueError("重连后的公开身份未重新核验")
                journal.add(
                    {
                        "kind": "future_input_reinitialization",
                        "received_ms": time.time_ns() // 1_000_000,
                        "generation": generation,
                        "past_decisions_not_rewritten": True,
                    }
                )
            if warmup is None:
                if status["subscription_confirmed"]:
                    warm_close, initial_members = common_stream_close(
                        bars, startup_contracts, now_ms=now
                    )
                    minimum = (status["connection_started_ms"] // MINUTE + 1) * MINUTE
                    if warm_close is not None and warm_close < minimum:
                        warm_close = None
                if warm_close is None:
                    if now - status.get("connection_started_ms", now) >= min(180, duration) * 1000:
                        raise ValueError("未在启动期限内收到完整共同收盘")
                    await asyncio.sleep(0.1)
                    continue
                journal.add(
                    {
                        "kind": "startup_warmup_binding",
                        "received_ms": now,
                        "canonical_close_ms": warm_close,
                        "stream_members": len(initial_members),
                        "history_received_not_yet": True,
                    }
                )
                jobs = [
                    (
                        "klines",
                        s,
                        {"symbol": s, "interval": "1m", "limit": 300, "endTime": warm_close - 1},
                    )
                    for s in symbols
                ]
                warm_generation = generation
                warmup = asyncio.create_task(asyncio.to_thread(client.batch, jobs))
                print(
                    f"首个共同分钟已绑定预热：{warm_close}，{len(initial_members)} 个可交易成员。",
                    flush=True,
                )
            if warmup.done() and not loaded:
                records = warmup.result()
                now = time.time_ns() // 1_000_000
                for record in records:
                    journal.add(record)
                warmup_journaled = True
                merged = merge_history(bars, records, closing=warm_close, now_ms=now)
                overlap.update(merged["overlap"])
                if merged["conflicts"]:
                    startup_conflicts.extend(merged["conflicts"])
                    journal.add(
                        {
                            "kind": "startup_public_value_conflict",
                            "received_ms": time.time_ns() // 1_000_000,
                            "conflicts": merged["conflicts"],
                            "decisions_blocked": True,
                        }
                    )
                    await asyncio.sleep(
                        max(0, (warm_close + MINUTE + 1000 - time.time_ns() // 1_000_000) / 1000)
                    )
                    repairs = [
                        (
                            "klines",
                            s,
                            {
                                "symbol": s,
                                "interval": "1m",
                                "limit": 300,
                                "endTime": warm_close - 1,
                            },
                        )
                        for s in sorted({v["symbol"] for v in merged["conflicts"]})
                    ]
                    confirmations = await asyncio.to_thread(client.batch, repairs)
                    for record in confirmations:
                        record["purpose"] = "STARTUP_CONSENSUS_RECHECK_ONLY"
                        journal.add(record)
                    consensus_rechecks += 1
                    verified = merge_history(
                        bars, confirmations, closing=warm_close, now_ms=time.time_ns() // 1_000_000
                    )
                    require_confirmed_conflicts(merged["conflicts"], verified)
                    overlap.update(verified["overlap"])
                if not set(initial_members).issubset(overlap):
                    raise ValueError("历史与实时共同分钟没有完整重叠核验")
                loaded = True
                now = time.time_ns() // 1_000_000
                next_close = now // MINUTE * MINUTE
                print(
                    f"公开预热完成，固定集合 {len(symbols)}，接下来只处理真实到达的后续分钟。",
                    flush=True,
                )
            if not loaded or not status["subscription_confirmed"]:
                await asyncio.sleep(0.1)
                continue
            now = time.time_ns() // 1_000_000
            if now < next_close + 1000:
                await asyncio.sleep(0.1)
                continue
            if now >= next_close + MINUTE:
                gaps.append(
                    {
                        "closing_ms": next_close,
                        "reason": "MISSED_REAL_DECISION_DEADLINE",
                        "recorded_ms": now,
                    }
                )
                activities.gap()
                next_close += MINUTE
                continue
            contracts = (
                venue_contracts(metadata, manifest["contracts"], decision_ms=now)
                if metadata
                else None
            )
            if contracts is None:
                metadata = await asyncio.to_thread(client.request, "exchange", None, {})
                journal.add(metadata)
                contracts = venue_contracts(
                    metadata, manifest["contracts"], decision_ms=time.time_ns() // 1_000_000
                )
            if contracts is None:
                await asyncio.sleep(1)
                continue
            active = active_members(contracts, next_close)
            if any(bars.at(s, next_close, now) is None for s in active):
                await asyncio.sleep(0.1)
                continue
            clock = await asyncio.to_thread(client.request, "clock", None, {})
            journal.add(clock)
            if not clock_valid(clock):
                raise ValueError("共同截面决策前时钟未核验")
            frame, quality = await asyncio.to_thread(
                snapshot,
                bars,
                contracts,
                closing=next_close,
                decision_ms=time.time_ns() // 1_000_000,
            )
            if frame is None:
                gaps.append(
                    {
                        "closing_ms": next_close,
                        "recorded_ms": time.time_ns() // 1_000_000,
                        **quality,
                    }
                )
                activities.gap()
                next_close += MINUTE
                continue
            ready = price_ready(frame)
            canary = (
                next(
                    (
                        s
                        for s in sorted(active)
                        if contracts[s]["asset_class"] == "TRADIFI_PERPETUAL"
                    ),
                    None,
                )
                if not canary_done and not ready
                else None
            )
            jobs = dependency_jobs(ready or ([canary] if canary else []), next_close)
            records = await asyncio.to_thread(client.batch, jobs) if jobs else []
            for r in records:
                journal.add(r)
            dependency_count += len(records)
            if (
                status["connection_generation"] != warm_generation
                or not status["subscription_confirmed"]
            ):
                activities.gap()
                gaps.append(
                    {
                        "closing_ms": next_close,
                        "recorded_ms": time.time_ns() // 1_000_000,
                        "reason": "SOURCE_CONNECTION_CHANGED_DURING_DECISION",
                    }
                )
                next_close += MINUTE
                continue
            evaluation_ms = time.time_ns() // 1_000_000
            evaluated = {
                s: evaluate_dependencies(
                    bars,
                    records,
                    s,
                    closing=next_close,
                    decision_ms=evaluation_ms,
                    clock_verified=True,
                )
                for s in ready
            }
            canary_evaluation = (
                evaluate_dependencies(
                    bars,
                    records,
                    canary,
                    closing=next_close,
                    decision_ms=evaluation_ms,
                    clock_verified=True,
                )
                if canary
                else None
            )
            canary_done = canary_done or canary is not None
            decision_ms = time.time_ns() // 1_000_000
            path = output / "snapshots" / f"{next_close}.parquet"
            if path.exists():
                raise ValueError("共同截面不得覆盖")
            frame.to_parquet(path)
            activities.observe(ready, next_close)
            signals = (
                activities.emit(evaluated, frame, closing=next_close, decision_ms=decision_ms)
                if decision_ms < next_close + MINUTE
                else []
            )
            if decision_ms >= next_close + MINUTE:
                activities.gap()
            decision_ms = time.time_ns() // 1_000_000
            if decision_ms >= next_close + MINUTE:
                signals = []
                activities.gap()
            for signal in signals:
                signal["actual_decision_ms"] = decision_ms
                signal["earliest_execution_ms"] = decision_ms + MINUTE
            decision = {
                "canonical_close_ms": next_close,
                "actual_decision_ms": decision_ms,
                "decision_age_ms": decision_ms - next_close,
                "within_fixed_one_minute_deadline": decision_ms < next_close + MINUTE,
                "price_ready": ready,
                "evaluated": evaluated,
                "source_only_canary": {
                    "symbol": canary,
                    "evaluation": canary_evaluation,
                    "can_never_emit": True,
                }
                if canary
                else None,
                "emitted": signals,
                "snapshot_sha256": sha(path),
                "source_prefix": journal.seal(),
                "orders_enabled": False,
            }
            save(output / "decisions" / f"{next_close}.json", decision)
            decisions.append(decision)
            emitted.extend(signals)
            print(
                f"真实分钟 {next_close}：决策距收盘 {decision_ms - next_close} 毫秒，价格候选 {len(ready)}，研究信号 {len(signals)}。",
                flush=True,
            )
            bars.prune(next_close)
            next_close += MINUTE
    finally:
        stop.set()
        await producer
        if warmup is not None and not warmup.done():
            client.pacer.stop.set()
            await warmup
        if warmup is not None and not warmup_journaled:
            records = warmup.result()
            for record in records:
                journal.add(record)
    checkpoint = {
        "bars": bars.values,
        "activities": activities.state(),
        "last_processed_close": None if next_close is None else next_close - MINUTE,
        "public_source_prefix": journal.seal(),
        "source_owner": "profit_factor_forward_research",
        "orders_enabled": False,
    }
    with (output / "input_checkpoint.json.gz").open("xb") as f:
        f.write(gzip.compress(canonical(checkpoint), mtime=0))
    return {
        "stream": status,
        "warmup_loaded": loaded,
        "warmup_stream_overlap_members": len(overlap),
        "warmup_stream_boundary_ms": warm_close,
        "startup_value_conflicts": startup_conflicts,
        "startup_consensus_rechecks": consensus_rechecks,
        "conflicting_prices_not_overwritten": True,
        "decisions": len(decisions),
        "decision_age_ms": [d["decision_age_ms"] for d in decisions],
        "emitted_signals": len(emitted),
        "signals_by_arm": {arm: sum(s["arm"] == arm for s in emitted) for arm in activities.used},
        "dependency_requests": dependency_count,
        "source_only_canary_exercised": canary_done,
        "gaps": gaps,
        "initial_activities_are_suppressed": True,
        "signal_only": True,
        "profitability_evaluated": False,
        "production_eligible": False,
        "orders_enabled": False,
    }


def run(panel, review, output, duration):
    """公开输入与模型提案先冻结；本阶段没有收益账户或生产执行器。"""
    manifest, model = read(panel / "manifest.json"), read(review)
    if (
        manifest.get("orders_enabled") is not False
        or manifest.get("leader_labels_read") is not False
        or not model.get("model_inference_performed")
        or model.get("production_eligible") is not False
    ):
        raise ValueError("前向研究只接受公开集合与无生产权限的模型提案")
    names = (
        "forward_pilot.py",
        "forward_inputs.py",
        "live_capture.py",
        "stream.py",
        "forward_startup.py",
        "public_http.py",
        "basis.py",
        "search.py",
        "account.py",
        "signals.py",
        "strategy_audit.py",
    )
    sources = [
        *(source_path(n) for n in names),
        Path(market.__file__),
        *Path(market.__file__).parent.joinpath("factors").glob("*.py"),
    ]
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("source", "snapshots", "decisions"):
        (output / folder).mkdir()
    protocol = {
        "schema": "cz-profit-live-signal-pilot.v4-startup-value-consensus",
        "created_ms": time.time_ns() // 1_000_000,
        "duration_seconds": duration,
        "source_files": {str(p.resolve()): sha(p) for p in sources},
        "panel_manifest_sha256": sha(panel / "manifest.json"),
        "model_review_sha256": sha(review),
        "symbols": sorted(manifest["symbols"]),
        "basis_limit_bps": 45,
        "reference_lag_minutes": 1,
        "basis_prior_median_minutes": 60,
        "confirmation_bars": 2,
        "random_seeds": SEEDS,
        "fixed_random_probability": PROBABILITY,
        "random_draw_unit": "initial price episode, one draw per seed",
        "one_entry_per_arm_per_price_episode": True,
        "initial_and_unknown_activities_suppressed": True,
        "maximum_decision_age_ms": MINUTE,
        "earliest_execution_after_actual_decision_ms": MINUTE,
        "all_arms_share_actual_pipeline_decision_clock": True,
        "warmup_bound_after_first_complete_stream_minute": True,
        "warmup_stream_overlap_required": True,
        "maximum_websocket_reconnects": 3,
        "reconnect_resets_unknown_activity_authority": True,
        "pooled_public_http_without_implicit_authentication": True,
        "conflicting_startup_prices_block_decisions_until_exact_consensus": True,
        "startup_conflict_recheck_minimum_age_ms": MINUTE + 1000,
        "startup_conflict_rechecks_per_binding": 1,
        "source_only_canary": "first lexicographic currently trading TradFi member, once when no price candidate; never emits signals",
        "signal_only": True,
        "forward_strategy_acceptance_registered": False,
        "profitability_evaluated": False,
        "credentials_used": False,
        "leader_labels_read": False,
        "orders_enabled": False,
        "runtime": {
            "python": sys.version.split()[0],
            **{
                n: importlib.metadata.version(n)
                for n in ("numpy", "pandas", "websockets", "requests")
            },
        },
    }
    save(output / "protocol.json", protocol)
    for number, path in enumerate(sources):
        (output / "source" / f"{number:02d}_{path.name}").write_bytes(path.read_bytes())
    journal, client = Journal(output), PublicHTTP()
    before = client.request("clock", None, {})
    journal.add(before)
    failure = None
    result = {
        "signal_only": True,
        "orders_enabled": False,
        "profitability_evaluated": False,
        "production_eligible": False,
    }
    try:
        if not clock_valid(before):
            raise ValueError("公开前向启动时钟核验失败")
        result |= asyncio.run(pilot(manifest, output, duration, journal, client))
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:300]}
    after = client.request("clock", None, {})
    journal.add(after)
    result |= {
        "clock_verified": clock_valid(before) and clock_valid(after),
        "failure": failure,
        "source_prefix": journal.seal(),
    }
    journal.close()
    client.close()
    save(output / "result.json", result)
    for p, expected in protocol["source_files"].items():
        if sha(p) != expected:
            raise ValueError("前向采样期间源码发生变化，结果不能采用")
    save(
        output / "receipts.json",
        {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()},
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 60 <= args.duration_seconds <= 1200:
        raise ValueError("此阶段只运行六十至一千二百秒的有限信号采样")
    result = run(
        args.panel.resolve(), args.review.resolve(), args.output.resolve(), args.duration_seconds
    )
    if (
        result["failure"]
        or not result["clock_verified"]
        or result.get("stream", {}).get("error")
        or not result.get("decisions")
    ):
        raise SystemExit("前向采样未通过，原始现场已封存")


if __name__ == "__main__":
    main()
