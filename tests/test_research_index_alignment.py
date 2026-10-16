"""验证指数消息不能越过决策时钟或以近似价格冒充同一终值。"""

import json

import pytest

from tradingagents.dataflows.research.stream import received_record
from tradingagents.research.quant.backtesting.index_alignment import compare, last_received_index


def observation(event, receive, price="100"):
    return received_record(
        json.dumps({"data": [{"s": "PUBLIC", "E": event, "i": price}]}), received_ms=receive
    )


def test_index_alignment_does_not_use_later_or_stale_observations():
    records = [
        observation(59_100, 59_200),
        observation(59_900, 120_001, "101"),
        observation(60_000, 60_010, "102"),
    ]
    assert last_received_index(records, "PUBLIC", 60_000) == (59_100, 59_200, "100")
    assert last_received_index([observation(58_999, 59_100)], "PUBLIC", 60_000) is None


def test_close_equivalence_requires_exact_decimal_value():
    assert compare("100.00", "100")["exact_decimal_equal"]
    assert not compare("100.00001", "100")["exact_decimal_equal"]
    assert compare("101", "100")["relative_difference_bps"] == 100


def test_nonfinite_stream_price_is_not_a_usable_index():
    with pytest.raises(ValueError):
        last_received_index([observation(59_900, 60_100, "nan")], "PUBLIC", 60_000)
