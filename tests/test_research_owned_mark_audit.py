"""等待声明不能延长期限、跳过真实原文或把未来价格当作已知。"""

import copy

import pytest

from tradingagents.dataflows.research.live_capture import digest
from tradingagents.research.quant.backtesting.paper_live_audit import require_preparation
from tradingagents.research.quant.backtesting.runtime_inputs import OWNED_MARK_WAIT_VERSION


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "future_price",
        "extended_deadline",
        "missing_original",
        "invalid_ready",
        "over_wait_limit",
    ],
)
def test_preparation_needs_original_identity_deadline_and_known_price(failure):
    closing, arrival = 6_000_000, 6_003_020
    start = {
        "kind": "owned_mark_wait_started",
        "received_ms": 6_002_258,
        "wait_id": "人工等待",
        "version": OWNED_MARK_WAIT_VERSION,
        "canonical_close_ms": closing,
        "symbols": ["A"],
        "original_deadline_ms": 6_032_258,
        "maximum_wait_ms": 5000,
    }
    end = {
        "kind": "owned_mark_wait_finished",
        "received_ms": 6_003_058,
        "wait_id": "人工等待",
        "version": OWNED_MARK_WAIT_VERSION,
        "canonical_close_ms": closing,
        "symbols": ["A"],
        "original_deadline_ms": 6_032_258,
        "elapsed_monotonic_ms": 800,
        "ready": True,
        "reason": None,
        "missing_symbols": [],
    }
    protocol = {
        "symbols": ["A"],
        "account_preparation_version": OWNED_MARK_WAIT_VERSION,
        "simulation_end_ms": 6_100_000,
    }
    if failure == "future_price":
        arrival = 6_003_100
    elif failure == "extended_deadline":
        start["original_deadline_ms"] = end["original_deadline_ms"] = 6_042_258
    elif failure == "invalid_ready":
        end["ready"] = "true"
    elif failure == "over_wait_limit":
        end.update(received_ms=6_007_259, elapsed_monotonic_ms=5001)
    originals = [start, end]
    identities = {digest(r) for r in originals if not (failure == "missing_original" and r is end)}
    frame = {
        "owned_mark_preparation": end,
        "phase": "exit_only",
        "canonical_close_ms": closing,
        "quotes": {"requested_ms": end["received_ms"] + 50},
    }

    class Bars:
        def at(self, symbol, clock, now):
            assert symbol == "A" and clock == closing
            return {"received_ms": arrival} if now >= arrival else None

    before = copy.deepcopy(frame)
    if failure:
        with pytest.raises(ValueError):
            require_preparation(frame, protocol, originals, identities, Bars())
    else:
        require_preparation(frame, protocol, originals, identities, Bars())
    assert frame == before
