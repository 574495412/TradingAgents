"""验证公开推送的完成标志、真实到达时序及重复冲突记录。"""

import json

import pytest

from tradingagents.dataflows.research.stream import closed_candle, received_record, summarize


def message(*, closed=True):
    return {
        "stream": "public@kline_1m",
        "data": {
            "e": "kline",
            "E": 120_100,
            "s": "PUBLIC",
            "k": {
                "s": "PUBLIC",
                "i": "1m",
                "t": 60_000,
                "T": 119_999,
                "o": "100",
                "h": "101",
                "l": "99",
                "c": "100",
                "v": "1",
                "q": "100",
                "n": 1,
                "V": "1",
                "Q": "100",
                "x": closed,
            },
        },
    }


def record(value=None):
    return received_record(json.dumps(value or message()), received_ms=120_200)


def test_only_final_push_received_before_decision_is_available():
    r = record()
    assert closed_candle(r, decision_ms=121_000, clock_verified=True)[1][0] == 60_000
    assert closed_candle(r, decision_ms=120_199, clock_verified=True) is None
    assert closed_candle(r, decision_ms=121_000, clock_verified=False) is None
    assert (
        closed_candle(record(message(closed=False)), decision_ms=121_000, clock_verified=True)
        is None
    )


@pytest.mark.parametrize("change", [{"s": "OTHER"}, {"T": 119_998}, {"c": "nan"}, {"Q": "-1"}])
def test_invalid_final_push_cannot_enter_minute_features(change):
    value = message()
    value["data"]["k"].update(change)
    with pytest.raises(ValueError):
        closed_candle(record(value), decision_ms=121_000, clock_verified=True)


def test_push_conflicts_are_disclosed_without_claiming_strategy_readiness():
    changed = message()
    changed["data"]["k"]["c"] = "100.5"
    summary = summarize(
        [record(), record(changed)], {"PUBLIC"}, decision_ms=121_000, clock_verified=True
    )
    assert summary["completed_minute_symbol_counts"] == {"120000": 1}
    assert summary["conflicting_completed_messages"] == [["PUBLIC", 60_000]]
    assert summary["closed_candle_receive_delay_ms_min_median_max"] == [200, 200, 200]
    assert summary["full_strategy_input_ready"] is False
    assert summary["profitability_evaluated"] is False
