"""验证订阅跨分钟时不会在预热与实时输入之间留下未验证的缺口。"""

import pytest

from tradingagents.dataflows.research.live_capture import canonical
from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.forward_inputs import Bars
from tradingagents.research.quant.backtesting.forward_startup import (
    MINUTE,
    common_stream_close,
    merge_history,
    require_confirmed_conflicts,
)


def add(bars, symbol, clock, received=None):
    row = [clock - MINUTE, "100", "101", "99", "100", "1", clock - 1, "100", 1, "0.5", "50", "0"]
    bars.add(
        symbol,
        row,
        received_ms=received or clock + 100,
        source_sha256="test",
        completed_verified=True,
    )


def test_partial_first_minute_is_not_a_valid_historical_stream_boundary():
    contracts = {s: {"trading": True, "onboard_ms": 0, "delivery_ms": 0} for s in ("A", "B")}
    bars = Bars(contracts)
    add(bars, "A", MINUTE)
    assert common_stream_close(bars, contracts, now_ms=MINUTE + 2000) == (None, [])
    for symbol in contracts:
        add(bars, symbol, 2 * MINUTE)
    assert common_stream_close(bars, contracts, now_ms=2 * MINUTE + 2000) == (
        2 * MINUTE,
        ["A", "B"],
    )


def test_late_or_too_recent_final_push_cannot_complete_startup():
    contracts = {"A": {"trading": True, "onboard_ms": 0, "delivery_ms": 0}}
    bars = Bars(contracts)
    add(bars, "A", MINUTE, received=MINUTE + 3000)
    assert common_stream_close(bars, contracts, now_ms=MINUTE + 2000) == (None, [])
    assert common_stream_close(bars, contracts, now_ms=MINUTE + 3000)[0] == MINUTE
    add(bars, "A", 2 * MINUTE)
    assert common_stream_close(bars, contracts, now_ms=2 * MINUTE + 500)[0] == MINUTE


def test_cross_channel_conflict_is_not_merged_and_later_consensus_keeps_first_receipt():
    bars = Bars(["A"])
    add(bars, "A", MINUTE)
    original = bars.at("A", MINUTE, MINUTE + 100)
    wrong = list(original["row"])
    wrong[4] = "100.5"

    def response(row, received):
        return received_record(canonical([row]), received_ms=received) | {
            "kind": "klines",
            "symbol": "A",
            "status": 200,
            "requested_ms": received - 100,
            "local_clock_consistent": True,
        }

    conflict = merge_history(
        bars, [response(wrong, MINUTE + 3000)], closing=MINUTE, now_ms=MINUTE + 4000
    )
    assert conflict["conflicts"] and not conflict["overlap"]
    assert bars.at("A", MINUTE, MINUTE + 4000)["row"][4] == "100"
    confirmed = merge_history(
        bars,
        [response(original["row"], 2 * MINUTE + 3000)],
        closing=MINUTE,
        now_ms=2 * MINUTE + 4000,
    )
    assert not confirmed["conflicts"] and confirmed["overlap"] == {"A"}
    assert bars.at("A", MINUTE, 2 * MINUTE + 4000)["received_ms"] == MINUTE + 100
    require_confirmed_conflicts(conflict["conflicts"], confirmed)


def test_missing_older_conflict_is_not_hidden_by_latest_minute_overlap():
    conflicts = [{"symbol": "A", "closing_ms": MINUTE}]
    confirmation = {"overlap": {"A"}, "conflicts": [], "confirmed_keys": {("A", 2 * MINUTE)}}
    with pytest.raises(ValueError, match="逐分钟"):
        require_confirmed_conflicts(conflicts, confirmation)
