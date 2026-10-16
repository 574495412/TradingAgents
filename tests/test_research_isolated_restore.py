"""非零库存和费用验证隔离重放，不把人工输入作为市场收益。"""

import copy

import pytest

from tests.test_research_paper_account import MINUTE, PaperAccount, entered, event, frame, payment
from tradingagents.research.quant.backtesting.isolated_restore import restore_in_isolation


@pytest.mark.parametrize("direction,cost", [(-1, 3), (-1, 10), (1, 3), (1, 10)])
def test_isolated_restore_matches_transactional_replay_with_closed_and_open_inventory(
    direction, cost
):
    account, frames = entered(direction=direction, slip=cost)
    open_state = account.checkpoint()
    assert restore_in_isolation(iter(frames), open_state).checkpoint() == open_state
    for minute in range(106, 128):
        funding = []
        if minute == 108:
            funding = [payment(minute, [event(108, "0.001"), event(108, "0.002", "Special")])]
        value = frame(minute, direction=direction, funding=funding, now=minute * MINUTE + 8000)
        frames.append(value)
        account.apply(value)
    expected = account.checkpoint()
    original_frames, original_state = copy.deepcopy(frames), copy.deepcopy(expected)
    isolated = restore_in_isolation(iter(frames), expected)
    regular = PaperAccount.restore(iter(frames), expected)
    assert isolated.checkpoint() == regular.checkpoint() == expected
    assert isolated.trades and len(isolated.payments) == 2
    assert frames == original_frames and expected == original_state


@pytest.mark.parametrize("failure", ["missing_frame", "changed_state", "illegal_frame"])
def test_failed_isolated_replay_cannot_publish_an_account_or_modify_expected_state(failure):
    account, frames = entered()
    expected = account.checkpoint()
    if failure == "missing_frame":
        frames.pop()
    elif failure == "changed_state":
        expected["cash"] = "1000"
    else:
        frames[-1]["actual_ms"] = frames[0]["actual_ms"]
    original = copy.deepcopy(expected)
    with pytest.raises(ValueError):
        restore_in_isolation(iter(frames), expected)
    assert expected == original
