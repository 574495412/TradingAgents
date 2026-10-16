"""人工盘口反证时间字段混用、隐含成交与异常深度。"""

import pytest

from tests.test_research_paper_account import public
from tradingagents.research.quant.backtesting.depth_probe import describe_pair


def pair(depth=None):
    quote = {
        "symbol": "AUSDT",
        "time": 10_000,
        "bidPrice": "100",
        "bidQty": "0.1",
        "askPrice": "101",
        "askQty": "0.1",
    }
    depth = depth or {
        "E": 70_000,
        "T": 10_000,
        "lastUpdateId": 1,
        "bids": [["100", "0.1"], ["99", "2"]],
        "asks": [["101", "0.1"], ["102", "2"]],
    }
    q = public("quotes", quote, 70_000, 70_100, symbol="AUSDT")
    d = public("quotes", depth, 70_200, 70_300, symbol="AUSDT") | {"kind": "depth"}
    return q, d


def test_fresh_output_does_not_relabel_an_old_transaction_or_prove_a_fill():
    result = describe_pair("AUSDT", *pair())
    assert result["book_event_age_at_receive_ms"] == 60_100
    assert result["depth_output_age_at_receive_ms"] == 300
    assert result["depth_transaction_age_at_receive_ms"] == 60_300
    assert result["book_event_within_five_seconds_at_receive_only"] is False
    assert result["timestamps_and_capacity_do_not_prove_real_fills"]
    assert result["visible_notional_usdt"]["asks"] == {"best": "10.1", "five_levels": "214.1"}


@pytest.mark.parametrize(
    "failure", ["wrong_symbol", "future_output", "unordered", "crossed", "zero_quantity"]
)
def test_malformed_or_unmatched_depth_is_not_treated_as_execution_capacity(failure):
    q, d = pair()
    if failure == "wrong_symbol":
        q["symbol"] = "OTHERUSDT"
    else:
        depth = {
            "E": 70_000,
            "T": 10_000,
            "lastUpdateId": 1,
            "bids": [["100", "1"]],
            "asks": [["101", "1"]],
        }
        if failure == "future_output":
            depth["E"] = 80_000
        elif failure == "unordered":
            depth["bids"] = [["99", "1"], ["100", "1"]]
        elif failure == "crossed":
            depth["asks"] = [["99", "1"]]
        else:
            depth["asks"] = [["101", "0"]]
        q, d = pair(depth)
    with pytest.raises(ValueError):
        describe_pair("AUSDT", q, d)
