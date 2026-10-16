"""验证双向假说使用真实已收到输入、严格阈值以及连续完整基线。"""

import json
from decimal import Decimal

import pytest

from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.bidirectional import MINUTE, evidence, index_jobs
from tradingagents.research.quant.backtesting.forward_inputs import Bars


def candle(clock, price):
    return [
        clock - MINUTE,
        str(price),
        str(price + 1),
        str(price - 1),
        str(price),
        "1",
        clock - 1,
        str(price),
        1,
        "0.5",
        str(price / 2),
        "0",
    ]


def inputs(premium):
    closing = 100 * MINUTE
    bars, rows = Bars(["PUBLIC"]), []
    for minute in range(37, 101):
        reference = 100 + minute / 100_000
        price = reference * (1 + premium) if minute >= 99 else reference
        bars.add(
            "PUBLIC",
            candle(minute * MINUTE, price),
            received_ms=minute * MINUTE + 100,
            source_sha256="test",
            completed_verified=True,
        )
        rows.append(candle(minute * MINUTE, reference))
    r = received_record(json.dumps(rows), received_ms=closing + 4000) | {
        "symbol": "PUBLIC",
        "kind": "index",
        "status": 200,
        "requested_ms": closing + 2000,
        "local_clock_consistent": True,
    }
    return bars, r, closing


@pytest.mark.parametrize("premium,direction", [(0.006, -1), (-0.006, 1), (0, 0)])
def test_dislocation_sign_determines_both_directions_without_parent_filters(premium, direction):
    bars, record, closing = inputs(premium)
    result = evidence(
        bars, "PUBLIC", record, closing=closing, decision_ms=closing + 5000, clock_verified=True
    )
    assert result["ready"] and result["direction"] == direction
    assert result["production_eligible"] is False


def test_late_index_and_missing_baseline_do_not_create_a_fallback_signal():
    bars, record, closing = inputs(0.006)
    record["received_ms"] = closing + 6000
    assert (
        evidence(
            bars, "PUBLIC", record, closing=closing, decision_ms=closing + 5000, clock_verified=True
        )["reason"]
        == "INDEX_RECEIVED_AFTER_DECISION"
    )
    record["received_ms"] = closing + 4000
    bars.values["PUBLIC"].pop(closing - 20 * MINUTE)
    result = evidence(
        bars, "PUBLIC", record, closing=closing, decision_ms=closing + 5000, clock_verified=True
    )
    assert result["reason"] == "CONTIGUOUS_BASIS_WINDOW_MISSING" and result["direction"] == 0


def test_index_query_preserves_the_entire_frozen_universe_and_completed_cutoff():
    jobs = index_jobs(["B", "A"], 100 * MINUTE)
    assert [symbol for _, symbol, _ in jobs] == ["A", "B"]
    assert all(p["endTime"] == 100 * MINUTE - 1 and p["limit"] == 64 for _, _, p in jobs)


def test_exact_threshold_is_neutral_and_does_not_depend_on_float_rounding():
    closing, bars, rows = 100 * MINUTE, Bars(["PUBLIC"]), []
    for minute in range(37, 101):
        reference = Decimal("100") + Decimal(minute) / 100000
        price = reference * Decimal("1.0045") if minute >= 99 else reference
        bars.add(
            "PUBLIC",
            candle(minute * MINUTE, price),
            received_ms=minute * MINUTE + 100,
            source_sha256="test",
            completed_verified=True,
        )
        rows.append(candle(minute * MINUTE, reference))
    record = received_record(json.dumps(rows), received_ms=closing + 4000) | {
        "symbol": "PUBLIC",
        "kind": "index",
        "status": 200,
        "requested_ms": closing + 2000,
        "local_clock_consistent": True,
    }
    result = evidence(
        bars, "PUBLIC", record, closing=closing, decision_ms=closing + 5000, clock_verified=True
    )
    assert (
        result["ready"] and result["direction"] == 0 and result["two_completed_minutes_within_band"]
    )


def test_stale_common_minute_cannot_be_reused_for_a_new_decision():
    bars, record, closing = inputs(0.006)
    result = evidence(
        bars,
        "PUBLIC",
        record,
        closing=closing + 3 * MINUTE,
        decision_ms=closing + 3 * MINUTE + 5000,
        clock_verified=True,
    )
    assert result["reason"] == "LATEST_COMMON_MINUTE_STALE" and result["direction"] == 0
