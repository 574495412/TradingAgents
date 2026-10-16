"""成本门槛必须只用已知特征，多空批次缺腿与退出规则必须明确。"""

from copy import deepcopy

import numpy as np
import pandas as pd

from tests.test_research_search import market, step
from tradingagents.research.quant.backtesting.account import Account
from tradingagents.research.quant.backtesting.forecast import cancel_incomplete_batch, decisions
from tradingagents.research.quant.backtesting.search import Audit
from tradingagents.research.quant.backtesting.signals import MINUTE, Signal


def test_decisions_select_balanced_extremes_and_ignore_future_outcomes():
    data = pd.DataFrame(
        {
            "symbol": [f"A{i:03}" for i in range(40)],
            "factor": np.arange(40),
            "target_60": np.arange(40),
        }
    )
    first = decisions(data, {"factor": 0.01}, ["factor"])
    data.target_60 *= -100
    assert first == decisions(data, {"factor": 0.01}, ["factor"])
    assert len(first) == 4 and sum(direction for _, direction, _ in first) == 0
    assert {symbol for symbol, _, _ in first} == {"A000", "A001", "A038", "A039"}
    assert not decisions(data, {"factor": 0.0001}, ["factor"])


def test_all_known_features_define_population_even_if_a_model_uses_fewer():
    data = pd.DataFrame(
        {"symbol": [f"A{i:03}" for i in range(40)], "factor": np.arange(40), "other": 1.0}
    )
    data.loc[39, "other"] = np.nan
    chosen = decisions(data, {"factor": 0.01}, ["factor", "other"])
    assert "A039" not in {s for s, _, _ in chosen}


def test_missing_open_cancels_the_complete_pair_without_altering_existing_state():
    account = Account("paired", 60, 3, [], take_profit_fraction=None)
    for symbol, direction in (("A", 1), ("B", -1)):
        account.schedule(Signal(symbol, direction, 1.0, MINUTE, "batch"), MINUTE)
    before = deepcopy(account.pending)
    quotes = {"A": (100, 100, True, True), "B": (np.nan, np.nan, True, False)}
    cancel_incomplete_batch(account, quotes, 2 * MINUTE)
    assert len(before) == len(account.declines) == 2 and not account.pending
    assert not account.fills and account.cash == 1000


def test_timed_exit_can_disable_profit_target_without_changing_default_accounts():
    default = Account("default", 20, 3, [])
    timed = Account("timed", 20, 3, [], take_profit_fraction=None)
    for account in (default, timed):
        audit = Audit(account, [])
        for clock in range(MINUTE, 6 * MINUTE, MINUTE):
            pulse = [Signal("PUBLIC", 1, 1, MINUTE, "batch")] if clock == MINUTE else []
            step(account, audit, clock, 100 if clock < 4 * MINUTE else 103, pulse)
    assert default.closes and not timed.closes
    assert len(timed.positions) == 1 and not timed.trades
    assert market(100)["PUBLIC"][0] == 100
