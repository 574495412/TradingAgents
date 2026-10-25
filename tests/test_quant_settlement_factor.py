"""结算退出因子只使用过去事件，已知零不伪装覆盖缺口。"""

import numpy as np
import pytest

from tradingagents.research.quant.factors.funding import DAY, LAG
from tradingagents.research.quant.factors.settlement import H01_NAME, MINUTE, calculate


def event(clock, rate=0.001, symbol="PUBLIC"):
    return {"symbol": symbol, "time_ms": clock, "rate": rate}


def test_visibility_and_linear_support_are_exact():
    stamp = 3 * DAY
    clocks = [stamp + k * MINUTE for k in (4, 5, 35, 64, 65, 66)]
    frame = calculate(clocks, [event(stamp)], 0, 5 * DAY)
    assert frame[H01_NAME].tolist() == [0, 0.001, 0.0005, 0.001 / 60, 0, 0]
    assert np.isnan(frame.iloc[0].settlement_latest_event_ms)
    assert frame.iloc[1].settlement_latest_event_ms == stamp


def test_full_48_hour_coverage_and_exclusive_source_end_are_not_forward_filled():
    clocks = [2 * DAY + LAG - 1, 2 * DAY + LAG, 3 * DAY + LAG, 3 * DAY + LAG + 1]
    frame = calculate(clocks, [], 0, 3 * DAY)
    assert np.isnan(frame[H01_NAME].iloc[0])
    assert frame[H01_NAME].iloc[1:3].tolist() == [0, 0]
    assert np.isnan(frame[H01_NAME].iloc[3])


def test_true_zero_expired_event_and_no_event_remain_known_zero():
    stamp = 3 * DAY
    assert calculate([stamp + LAG], [event(stamp, 0)], 0, 4 * DAY)[H01_NAME].iloc[0] == 0
    assert calculate([stamp + 65 * MINUTE], [event(stamp)], 0, 4 * DAY)[H01_NAME].iloc[0] == 0
    assert calculate([stamp + LAG], [], 0, 4 * DAY)[H01_NAME].iloc[0] == 0


def test_future_event_and_rate_cannot_change_prior_value():
    stamp = 3 * DAY
    clocks = [stamp + LAG, stamp + 35 * MINUTE]
    before = calculate(clocks, [event(stamp, -0.002)], 0, 5 * DAY)
    after = calculate(clocks, [event(stamp, -0.002), event(4 * DAY, 1e308)], 0, 5 * DAY)
    assert before.equals(after)
    assert before[H01_NAME].tolist() == [-0.002, -0.001]


@pytest.mark.parametrize(
    "clocks,events",
    [
        ([2 * DAY + LAG, 2 * DAY + LAG], []),
        ([2 * DAY + LAG, 2 * DAY], []),
        ([True], []),
        ([float(2 * DAY)], []),
        ([3 * DAY], [event(2 * DAY), event(2 * DAY, 0.002)]),
        ([3 * DAY], [event(2 * DAY), event(2 * DAY + 1, symbol="OTHER")]),
        ([3 * DAY], [event(2 * DAY, np.inf)]),
        ([3 * DAY], [event(2 * DAY, True)]),
        ([3 * DAY], [event(-1)]),
    ],
)
def test_conflicts_and_invalid_inputs_are_not_silently_repaired(clocks, events):
    with pytest.raises(ValueError):
        calculate(clocks, events, 0, 5 * DAY)
