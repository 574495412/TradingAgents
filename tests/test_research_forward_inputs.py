"""验证共同截面、首次接收、固定依赖窗口及两次滞后基差确认。"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tradingagents.dataflows.research.live_capture import canonical
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.forward_inputs import (
    MINUTE,
    RATIOS,
    Bars,
    dependency_jobs,
    evaluate_dependencies,
    snapshot,
)
from tradingagents.research.quant.reference import market_features as market


def candle(closing, price=100):
    return [
        closing - MINUTE,
        str(price),
        str(price + 1),
        str(price - 1),
        str(price),
        "1",
        closing - 1,
        str(price),
        1,
        "0.5",
        str(price * 0.5),
        "0",
    ]


def add(bars, symbol, closing, price=100, receive=None):
    bars.add(
        symbol,
        candle(closing, price),
        received_ms=receive or closing + 100,
        source_sha256="test",
        completed_verified=True,
    )


def response(kind, values, closing):
    r = received_record(canonical(values), received_ms=closing + 4000)
    return r | {
        "kind": kind,
        "symbol": "PUBLIC",
        "status": 200,
        "requested_ms": closing + 2000,
        "local_clock_consistent": True,
    }


def test_bars_require_finality_and_reject_conflicts_but_preserve_first_receipt():
    bars = Bars(["PUBLIC"])
    with pytest.raises(ValueError):
        bars.add("PUBLIC", candle(MINUTE), received_ms=MINUTE + 100, source_sha256="test")
    add(bars, "PUBLIC", MINUTE)
    row = candle(MINUTE)
    row[4] = "100.000"
    bars.add(
        "PUBLIC", row, received_ms=MINUTE + 200, source_sha256="later", completed_verified=True
    )
    assert bars.at("PUBLIC", MINUTE, MINUTE + 99) is None
    assert bars.at("PUBLIC", MINUTE, MINUTE + 200)["received_ms"] == MINUTE + 100
    with pytest.raises(ValueError, match="不同终值"):
        add(bars, "PUBLIC", MINUTE, 101)


def test_snapshot_matches_native_formulas_and_does_not_compress_missing_minutes():
    symbols, frames, contracts = ("PUBLIC", "OTHER"), {}, {}
    bars = Bars(symbols)
    closing = 300 * MINUTE
    for number, symbol in enumerate(symbols):
        raw = []
        for minute in range(1, 301):
            price = 100 + minute * (0.01 + number * 0.01)
            add(bars, symbol, minute * MINUTE, price)
            raw.append([price, price + 1, price - 1, price, 1, price, price * 0.5])
        frames[symbol] = pd.DataFrame(
            raw,
            index=pd.date_range("1970-01-01T00:01Z", periods=300, freq="min"),
            columns=market.RAW_COLUMNS,
        )
        contracts[symbol] = {"asset_class": "TRADIFI_PERPETUAL", "onboard_ms": 0, "delivery_ms": 0}
    result, status = snapshot(bars, contracts, closing=closing, decision_ms=closing + 5000)
    expected = market.build_feature_panel(frames).xs(
        pd.Timestamp(closing, unit="ms", tz="UTC"), level="bar_time"
    )
    assert status["reason"] == "READY"
    columns = [c for c in result if c in expected]
    np.testing.assert_allclose(
        result[columns], expected[columns], equal_nan=True, rtol=1e-12, atol=1e-12
    )
    bars.values["PUBLIC"].pop(closing - 20 * MINUTE)
    result, status = snapshot(bars, contracts, closing=closing, decision_ms=closing + 5000)
    assert result is None and status["reason"] == "INCOMPLETE_CAUSAL_CALENDAR"


def test_late_bar_cannot_enter_a_sealed_common_snapshot():
    bars = Bars(["PUBLIC"])
    for minute in range(1, 301):
        add(
            bars,
            "PUBLIC",
            minute * MINUTE,
            receive=300 * MINUTE + 10_000 if minute == 300 else None,
        )
    contracts = {"PUBLIC": {"asset_class": "TRADIFI_PERPETUAL", "onboard_ms": 0, "delivery_ms": 0}}
    frame, _ = snapshot(bars, contracts, closing=300 * MINUTE, decision_ms=300 * MINUTE + 5000)
    assert frame is None


def test_dependency_jobs_cover_exactly_two_lagged_sixty_minute_medians():
    jobs = dependency_jobs(["PUBLIC"], 100 * MINUTE)
    assert len(jobs) == 4
    index = jobs[-1][2]
    assert index["startTime"] == 37 * MINUTE
    assert index["endTime"] == 99 * MINUTE - 1 and index["limit"] == 62


def test_fixed_basis_rule_uses_received_official_closes_and_two_confirmations():
    closing = 100 * MINUTE
    bars, indices = Bars(["PUBLIC"]), []
    for minute in range(38, 100):
        reference = 100 + minute / 100_000
        premium = 0.006 if minute >= 98 else 0
        add(bars, "PUBLIC", minute * MINUTE, reference * (1 + premium))
        indices.append(candle(minute * MINUTE, reference))
    records = [
        response(
            kind, [{"symbol": "PUBLIC", "timestamp": 95 * MINUTE, "longShortRatio": "0.5"}], closing
        )
        for kind in RATIOS
    ]
    records.append(response("index", indices, closing))
    result = evaluate_dependencies(
        bars, records, "PUBLIC", closing=closing, decision_ms=closing + 5000, clock_verified=True
    )
    assert result["ready"] and result["basis_short_pass"] and result["positioning_pass"]
    assert result["older_basis_delta"] == pytest.approx(0.006)
    late = deepcopy(records)
    late[-1]["received_ms"] = closing + 6000
    rejected = evaluate_dependencies(
        bars, late, "PUBLIC", closing=closing, decision_ms=closing + 5000, clock_verified=True
    )
    assert rejected["reason"] == "DEPENDENCY_RECEIVED_AFTER_DECISION"
    missing = deepcopy(records)
    missing[-1] = response("index", indices[:-1], closing)
    assert not evaluate_dependencies(
        bars, missing, "PUBLIC", closing=closing, decision_ms=closing + 5000, clock_verified=True
    )["ready"]
