"""逐前缀核验公开原文，内存只保存分钟窗口和未完成交接。"""

from __future__ import annotations

import sqlite3
from urllib.parse import parse_qs, urlparse

from tradingagents.dataflows.research.live_capture import (
    MINUTE,
    available_rows,
    body,
    canonical,
    clock_valid,
    digest,
)
from tradingagents.dataflows.research.stream import closed_candle, unwrap
from tradingagents.research.quant.backtesting.continuous_feed import verify_handoff
from tradingagents.research.quant.backtesting.forward_inputs import Bars, normalized_row


class SourceCursor:
    """原文成员身份放在临时磁盘索引，索引只能由已验证的事件链生成。"""

    def __init__(self, iterator, symbols, database):
        self.iterator, self.symbols = iter(iterator), symbols
        self.database = sqlite3.connect(database)
        self.database.execute("CREATE TABLE originals (identity TEXT PRIMARY KEY)")
        self.count, self.head, self.maximum_received, self.sessions = 0, None, 0, 0
        self.bars = Bars(symbols)
        self.clocks, self.indices, self.pending = [], [], []
        self.observed_conflicts, self.unresolved = {}, {}
        self.confirmed_conflicts = self.maximum_cached_bars = self.maximum_prefix_events = 0
        self.maximum_prefix_bytes = 0
        self.recent_closed, self.connection_starts = {}, {}
        self.handoffs = self.missed_minutes = 0
        self.preparations = []

    def __contains__(self, identity):
        return (
            self.database.execute(
                "SELECT 1 FROM originals WHERE identity = ?", (identity,)
            ).fetchone()
            is not None
        )

    def _next(self):
        sequence, record, head = next(self.iterator)
        if sequence != self.count:
            raise ValueError("公开前缀游标的序号不连续")
        self.count, self.head = sequence + 1, head
        self.maximum_received = max(self.maximum_received, record.get("received_ms", 0))
        return record

    def _rows(self, record, rows):
        for symbol, row in rows:
            closing = row[6] + 1
            normalized = normalized_row(row)
            previous = self.bars.values[symbol].get(closing)
            if previous and previous["normalized"] != normalized:
                key = symbol, closing, previous["source_sha256"], record["body_sha256"]
                self.observed_conflicts[key] = previous["normalized"], previous["received_ms"]
                continue
            for key, (expected, declared_ms) in list(self.unresolved.items()):
                if (
                    key[:2] == (symbol, closing)
                    and record.get("purpose") == "STARTUP_CONSENSUS_RECHECK_ONLY"
                    and record.get("status") == 200
                    and record["requested_ms"] >= closing + MINUTE + 1000
                    and record["received_ms"] >= declared_ms
                    and normalized == expected
                ):
                    del self.unresolved[key]
                    self.confirmed_conflicts += 1
            self.bars.add(
                symbol,
                row,
                received_ms=record["received_ms"],
                source_sha256=record["body_sha256"],
                completed_verified=True,
            )

    def _consume(self, record, now):
        kind = record["kind"]
        if kind in {"owned_mark_wait_started", "owned_mark_wait_finished"}:
            self.preparations.append(record)
        if kind == "paper_process_session":
            self.bars, self.pending = Bars(self.symbols), []
            self.sessions += 1
        elif kind == "websocket":
            parsed = closed_candle(record, decision_ms=now, clock_verified=True)
            if parsed:
                key = (
                    record["body_sha256"],
                    record.get("connection_generation"),
                    record["received_ms"],
                )
                self.recent_closed[key] = parsed[1][6] + 1, record
                self._rows(record, [parsed])
            else:
                value = unwrap(body(record))
                if isinstance(value, dict) and value.get("k", {}).get("x") is True:
                    self.pending.append(record)
        elif kind == "klines" and record.get("symbol") in self.symbols:
            rows = available_rows(
                record,
                decision_ms=now,
                cutoff_ms=(now - 1000) // MINUTE * MINUTE,
                clock_verified=True,
            )
            self._rows(record, [(record["symbol"], row) for row in rows])
        elif kind == "startup_public_value_conflict":
            for conflict in record["conflicts"]:
                key = tuple(
                    conflict[k]
                    for k in (
                        "symbol",
                        "closing_ms",
                        "first_source_sha256",
                        "conflicting_source_sha256",
                    )
                )
                if key not in self.observed_conflicts:
                    raise ValueError("交接冲突缺少先前实际收到的两份原文")
                expected, received = self.observed_conflicts.pop(key)
                if received > record["received_ms"]:
                    raise ValueError("冲突的首次原文尚未收到")
                self.unresolved[key] = expected, record["received_ms"]
        elif kind == "public_stream_connection_started":
            self.connection_starts[record["generation"]] = record["received_ms"]
        elif kind == "public_stream_overlap_handoff":
            verify_handoff(
                record,
                {k: v[1] for k, v in self.recent_closed.items()},
                self.symbols,
                self.connection_starts,
            )
            self.handoffs += 1
        elif kind == "missed_source_minute":
            self.missed_minutes += 1
        if kind == "clock" and clock_valid(record):
            self.clocks.append(record)
            self.clocks = self.clocks[-10:]
        if kind == "index" and "url" in record:
            query = parse_qs(urlparse(record["url"]).query)
            end = query.get("endTime", [])
            if len(end) == 1:
                self.indices.append((int(end[0]) + 1, record))

    def advance(self, prefix, *, now, closing):
        """只推进到本帧声明的前缀；未来事件不能进入原文身份或价格缓存。"""
        self.preparations = []
        count = prefix["event_count"]
        if type(count) is not int or count < self.count:
            raise ValueError("公开前缀不能倒退或使用非整数数量")
        new, decoded_bytes = [], 0
        try:
            while self.count < count:
                record = self._next()
                decoded_bytes += len(canonical(record))
                new.append(record)
                if len(new) > 10_000 or decoded_bytes > 256 * 1024 * 1024:
                    raise ValueError("单帧公开前缀增量超过固定审计容量，原文仍保留")
        except StopIteration:
            raise ValueError("账户引用的公开原文前缀尚不存在") from None
        if self.head != prefix["event_chain_sha256"] or self.maximum_received > now:
            raise ValueError("公开前缀身份错误或包含尚未收到的输入")
        self.maximum_prefix_events = max(self.maximum_prefix_events, len(new))
        self.maximum_prefix_bytes = max(self.maximum_prefix_bytes, decoded_bytes)
        self.database.executemany(
            "INSERT OR IGNORE INTO originals VALUES (?)", ((digest(r),) for r in new)
        )
        self.database.commit()
        pending, self.pending = self.pending, []
        for record in pending:
            self._consume(record, now)
        for record in new:
            self._consume(record, now)
        if self.observed_conflicts or self.unresolved:
            raise ValueError("存在未声明或尚未逐点精确确认的公开终值冲突")
        self.indices = [(t, r) for t, r in self.indices if t >= closing - 2 * MINUTE]
        self.bars.prune(closing)
        self.recent_closed = {
            k: v for k, v in self.recent_closed.items() if v[0] >= closing - 2 * MINUTE
        }
        self.maximum_cached_bars = max(
            self.maximum_cached_bars, sum(len(v) for v in self.bars.values.values())
        )

    def finish(self):
        """最终账户之后的原文也核验完整链，但不赋予它们先前决策权限。"""
        for sequence, record, head in self.iterator:
            if sequence != self.count:
                raise ValueError("公开尾部序号不连续")
            self.count, self.head = sequence + 1, head
            self.sessions += record["kind"] == "paper_process_session"

    def close(self):
        self.database.close()
