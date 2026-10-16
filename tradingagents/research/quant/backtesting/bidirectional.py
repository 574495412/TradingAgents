"""TradingAgents 新提案的独立双向基差证据计算，尚未授予账户或生产资格。"""

from __future__ import annotations

from decimal import Decimal
from statistics import median

from tradingagents.dataflows.research.live_capture import MINUTE, available_rows

HYPOTHESIS = "basis_reversion_bidirectional_v1"
LIMIT = Decimal("0.0045")
MAXIMUM_DATA_AGE_MS = 2 * MINUTE


def index_jobs(symbols, closing):
    """完整采集冻结全集；两根候选与各自六十根基线外，保留一分钟到达余量。"""
    return [
        (
            "index",
            s,
            {
                "pair": s,
                "interval": "1m",
                "startTime": closing - 64 * MINUTE,
                "endTime": closing - 1,
                "limit": 64,
            },
        )
        for s in sorted(symbols)
    ]


def evidence(bars, symbol, record, *, closing, decision_ms, clock_verified):
    """选择最新共同已知分钟；不因较老窗口更有利而退回过去重算。"""
    rejected = {
        "hypothesis": HYPOTHESIS,
        "ready": False,
        "direction": 0,
        "production_eligible": False,
        "orders_enabled": False,
    }
    if closing % MINUTE or decision_ms < closing + 1000:
        return rejected | {"reason": "CANONICAL_MINUTE_NOT_SETTLED"}
    if not clock_verified or not record.get("local_clock_consistent"):
        return rejected | {"reason": "CLOCK_UNVERIFIED"}
    if record.get("status") != 200:
        return rejected | {"reason": "OFFICIAL_INDEX_UNAVAILABLE"}
    if record["symbol"] != symbol or record["kind"] != "index":
        raise ValueError("指数身份不属于该合约")
    if record["received_ms"] > decision_ms:
        return rejected | {"reason": "INDEX_RECEIVED_AFTER_DECISION"}
    rows = available_rows(record, decision_ms=decision_ms, cutoff_ms=closing, clock_verified=True)
    indices = {r[6] + 1: Decimal(str(r[4])) for r in rows}
    if len(indices) != len(rows):
        raise ValueError("官方指数分钟重复")
    common = [t for t in indices if t <= closing and bars.at(symbol, t, decision_ms) is not None]
    if not common:
        return rejected | {"reason": "NO_COMMON_RECEIVED_MINUTE"}
    data_clock = max(common)
    if decision_ms - data_clock > MAXIMUM_DATA_AGE_MS:
        return rejected | {"reason": "LATEST_COMMON_MINUTE_STALE", "data_clock_ms": data_clock}
    clocks = list(range(data_clock - 61 * MINUTE, data_clock + MINUTE, MINUTE))
    if any(t not in indices or bars.at(symbol, t, decision_ms) is None for t in clocks):
        return rejected | {"reason": "CONTIGUOUS_BASIS_WINDOW_MISSING", "data_clock_ms": data_clock}
    basis = [
        Decimal(str(bars.at(symbol, t, decision_ms)["row"][4])) / indices[t] - 1 for t in clocks
    ]
    older, latest = basis[60] - median(basis[:60]), basis[61] - median(basis[1:61])
    active = len({indices[t] for t in clocks[-5:]}) > 1
    direction = (
        -1 if older > LIMIT and latest > LIMIT else 1 if older < -LIMIT and latest < -LIMIT else 0
    )
    if not active:
        direction = 0
    return {
        "hypothesis": HYPOTHESIS,
        "ready": True,
        "reason": "READY",
        "symbol": symbol,
        "direction": direction,
        "older_basis_deviation": float(older),
        "latest_basis_deviation": float(latest),
        "two_completed_minutes_within_band": abs(older) <= LIMIT and abs(latest) <= LIMIT,
        "reference_active": active,
        "data_clock_ms": data_clock,
        "actual_decision_ms": decision_ms,
        "data_age_ms": decision_ms - data_clock,
        "latest_input_received_ms": max(
            record["received_ms"], *(bars.at(symbol, t, decision_ms)["received_ms"] for t in clocks)
        ),
        "official_index_source_sha256": record["body_sha256"],
        "production_eligible": False,
        "orders_enabled": False,
    }
