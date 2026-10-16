"""原库存退出诊断不借未来报价、不放宽数量、不改变两种成本路径。"""

import copy

import pytest

from tests.test_research_microstructure import message
from tests.test_research_paper_account import frame
from tradingagents.research.quant.backtesting import exit_quote_probe as module
from tradingagents.research.quant.backtesting.microstructure import parse_message
from tradingagents.research.quant.backtesting.paper_account import PaperAccount


@pytest.mark.parametrize(
    "failure", [None, "future", "stale", "predue", "small", "gap", "missing", "unverified"]
)
def test_original_quantity_and_due_are_checked_without_replacing_bad_quotes(failure):
    value = parse_message(message("quote"), ["AUSDT"])
    epoch = {"generation": "observer:1", "confirmed_ms": 60_000, "end_ms": 500_000}
    due = 119_000
    if failure == "future":
        value["received_ms"] = 120_001
    elif failure == "stale":
        value["transaction_ms"] = 114_999
    elif failure == "predue":
        due = 120_000
    elif failure == "small":
        value["bid_qty"] = "9.99"
    elif failure == "gap":
        epoch["end_ms"] = 119_999
    elif failure == "missing":
        value = None
    elif failure == "unverified":
        value["clock_verified"] = False
    result = module.quote_conditions(value, epoch, now=120_000, due=due, direction=1, quantity="10")
    assert result["market_quote_conditions_met"] == (failure is None)
    assert result["real_fill_proven"] is False and result["orders_enabled"] is False
    if failure == "small":
        assert result["reason"] == "ORIGINAL_QUANTITY_EXCEEDS_DISPLAYED_BEST_SIZE"


@pytest.mark.parametrize("direction", [-1, 1])
def test_all_original_successful_and_rejected_attempts_are_retained_with_exact_replay(direction):
    protocol = {"symbols": ["A"], "simulation_end_ms": None}
    models = {name: PaperAccount(["A"], int(name)) for name in ("3", "10")}
    frames = [frame(t, direction=direction if t >= 103 else 0) for t in range(100, 106)]
    frames.extend([frame(125, quantity="0.1", now=125 * module.MINUTE + 8000), frame(126)])
    for value in frames:
        for model in models.values():
            model.apply(value)
    expected = {name: model.checkpoint() for name, model in models.items()}
    before = copy.deepcopy(expected)
    attempts = list(module.exit_attempts(iter(frames), protocol, expected))
    assert expected == before
    for name in models:
        selected = [a for a in attempts if a["cost_account"] == name]
        assert [a["original_outcome"] for a in selected] == [
            "EXIT_DISPLAYED_CAPACITY_INSUFFICIENT",
            "SIMULATED_CLOSE",
        ]
        assert len({a["episode"] for a in selected}) == 1
        assert all(a["direction"] == direction for a in selected)
        assert all(a["exit_due_reason"] == "time" for a in selected)


def test_short_exit_uses_ask_size_and_long_exit_uses_bid_size():
    value = parse_message(message("quote"), ["AUSDT"])
    value["bid_qty"] = "0.2"
    epoch = {"generation": "observer:1", "confirmed_ms": 60_000, "end_ms": 500_000}
    short = module.quote_conditions(
        value, epoch, now=120_000, due=119_000, direction=-1, quantity="5"
    )
    long = module.quote_conditions(
        value, epoch, now=120_000, due=119_000, direction=1, quantity="5"
    )
    assert short["market_quote_conditions_met"] is True and short["side"] == "ask"
    assert long["market_quote_conditions_met"] is False and long["side"] == "bid"
