"""固定资金费因子须保留窗口边界、覆盖状态、真实零值与过去结果。"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.factors.funding import DAY, LAG, NAMES, calculate


def event(timestamp, rate, symbol="BTCUSDT", **extra):
    """生成单一合约的规范化测试事件。"""
    return {"symbol": symbol, "time_ms": timestamp, "rate": rate, **extra}


def test_fixed_names_and_hand_calculated_24h_and_48h_sums():
    clocks = pd.Index([DAY + LAG, 2 * DAY + LAG], name="decision_ms")
    events = [event(2 * DAY, 0.04), event(DAY // 2, 0.01), event(DAY, -0.02)]
    result = calculate(clocks, events, 0, 3 * DAY)

    assert NAMES == ("funding_carry_24h", "funding_carry_change_24h")
    pd.testing.assert_index_equal(result.index, clocks)
    assert result[NAMES[0]].tolist() == pytest.approx([0.01, -0.04])
    assert pd.isna(result.loc[DAY + LAG, NAMES[1]])
    assert result.loc[2 * DAY + LAG, NAMES[1]] == pytest.approx(-0.05)
    assert result["funding_event_count_24h"].tolist() == [2, 1]
    assert result["funding_previous_event_count_24h"].tolist() == [0, 2]
    assert result["funding_latest_event_ms"].tolist() == [DAY, 2 * DAY]
    assert events[0]["time_ms"] == 2 * DAY


def test_five_minute_visibility_lag_and_all_window_boundaries():
    clocks = [2 * DAY + LAG - 1, 2 * DAY + LAG, 2 * DAY + LAG + 1]
    events = [
        event(0, 100.0),
        event(1, 0.01),
        event(DAY, 0.02),
        event(DAY + 1, 0.03),
        event(2 * DAY, 0.04),
    ]
    result = calculate(clocks, events, -DAY, 3 * DAY)

    assert result[NAMES[0]].tolist() == pytest.approx([-0.05, -0.07, -0.04])
    assert result[NAMES[1]].tolist() == pytest.approx([99.96, -0.04, 0.01])
    assert result["funding_event_count_24h"].tolist() == [2, 2, 1]
    assert result["funding_previous_event_count_24h"].tolist() == [2, 2, 2]
    assert result["funding_latest_event_ms"].tolist() == [DAY + 1, 2 * DAY, 2 * DAY]


def test_a_real_zero_rate_is_known_but_an_empty_window_remains_unknown():
    clocks = [DAY + LAG, 2 * DAY + LAG, 3 * DAY + LAG]
    result = calculate(clocks, [event(DAY, 0.0), event(2 * DAY, 0.0)], 0, 4 * DAY)

    assert result.loc[clocks[0], NAMES[0]] == 0
    assert result.loc[clocks[1], NAMES[1]] == 0
    assert pd.isna(result.loc[clocks[2], NAMES[0]])
    assert pd.isna(result.loc[clocks[2], NAMES[1]])
    assert result.loc[clocks[2], "funding_event_count_24h"] == 0


def test_an_empty_previous_window_prevents_a_change_value():
    result = calculate([2 * DAY + LAG], [event(2 * DAY, 0.1)], 0, 3 * DAY)
    assert result.iloc[0][NAMES[0]] == pytest.approx(-0.1)
    assert pd.isna(result.iloc[0][NAMES[1]])
    assert result.iloc[0]["funding_previous_event_count_24h"] == 0


def test_coverage_start_and_end_apply_to_each_complete_window():
    clocks = [DAY + LAG - 1, DAY + LAG, 2 * DAY + LAG, 2 * DAY + LAG + 1]
    result = calculate(clocks, [event(DAY // 2, 0.01), event(2 * DAY, 0.02)], 0, 2 * DAY)

    assert pd.isna(result.loc[clocks[0], NAMES[0]])
    assert result.loc[clocks[1], NAMES[0]] == pytest.approx(-0.01)
    assert pd.isna(result.loc[clocks[1], NAMES[1]])
    assert result.loc[clocks[2], NAMES[0]] == pytest.approx(-0.02)
    assert result.loc[clocks[2], NAMES[1]] == pytest.approx(-0.01)
    assert result.loc[clocks[3], list(NAMES)].isna().all()


def test_future_event_values_presence_and_timing_cannot_change_the_past():
    clocks = pd.Index([2 * DAY + LAG], name="original_clock")
    past = [event(DAY // 2, 0.01), event(DAY + 1, 0.02)]
    first = calculate(clocks, past, 0, 4 * DAY)
    with_future = calculate(clocks, past + [event(2 * DAY + 1, 0.9)], 0, 4 * DAY)
    changed_future = calculate(
        clocks, past + [event(3 * DAY, -0.9, rateType="Special")], 0, 4 * DAY
    )

    pd.testing.assert_frame_equal(first, with_future)
    pd.testing.assert_frame_equal(first, changed_future)


def test_only_declared_coverage_controls_completeness_not_a_future_settlement_interval():
    clocks = [2 * DAY + LAG]
    past = [event(DAY // 2, 0.01), event(DAY + 1, 0.02)]
    result = calculate(clocks, past, 0, 4 * DAY)
    assert result.iloc[0][NAMES[0]] == pytest.approx(-0.02)
    assert result.iloc[0][NAMES[1]] == pytest.approx(-0.01)


def test_empty_events_keep_values_and_latest_event_unknown():
    result = calculate([2 * DAY + LAG], [], 0, 3 * DAY)
    assert result.loc[:, list(NAMES)].isna().all().all()
    assert result.iloc[0]["funding_event_count_24h"] == 0
    assert result.iloc[0]["funding_previous_event_count_24h"] == 0
    assert pd.isna(result.iloc[0]["funding_latest_event_ms"])


def test_empty_clocks_preserve_the_original_named_index():
    clocks = pd.Index([], dtype="int64", name="empty_clock")
    result = calculate(clocks, [], 0, 2 * DAY)
    pd.testing.assert_index_equal(result.index, clocks)
    assert result.empty


@pytest.mark.parametrize("clocks", [[1, 1], [2, 1], [1.0], [True], [None], ["1"]])
def test_clocks_require_unique_strictly_increasing_integer_milliseconds(clocks):
    with pytest.raises(ValueError, match="研究时钟"):
        calculate(clocks, [], 0, 3 * DAY)


@pytest.mark.parametrize("timestamp", [1.0, True, None, "1", np.nan])
def test_event_times_require_integer_milliseconds(timestamp):
    with pytest.raises(ValueError, match="事件时间"):
        calculate([2 * DAY + LAG], [event(timestamp, 0.01)], 0, 3 * DAY)


@pytest.mark.parametrize("rate", [np.nan, np.inf, -np.inf, True, None, "0.01"])
def test_event_rates_require_finite_numeric_values(rate):
    with pytest.raises(ValueError, match="有限数值"):
        calculate([2 * DAY + LAG], [event(DAY, rate)], 0, 3 * DAY)


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([event(DAY, 0.01), event(DAY, 0.02)], "不能重复"),
        ([event(DAY, 0.01), event(2 * DAY, 0.02, symbol="ETHUSDT")], "单一合约"),
        ([event(-1, 0.01)], "覆盖区间"),
        ([event(3 * DAY + 1, 0.01)], "覆盖区间"),
        ([event(DAY, 0.01, symbol="")], "合约"),
        ([{"time_ms": DAY, "rate": 0.01}], "合约"),
        ([None], "规范化字典"),
    ],
)
def test_invalid_event_identity_duplicates_or_coverage_are_rejected(events, message):
    with pytest.raises(ValueError, match=message):
        calculate([2 * DAY + LAG], events, 0, 3 * DAY)


@pytest.mark.parametrize("start,end", [(0.0, 3 * DAY), (0, True), (3 * DAY, 0)])
def test_coverage_requires_ordered_integer_milliseconds(start, end):
    with pytest.raises(ValueError, match="覆盖"):
        calculate([2 * DAY + LAG], [], start, end)


def test_numpy_integer_milliseconds_and_zero_integer_rates_are_valid():
    result = calculate(
        pd.Index([np.int64(2 * DAY + LAG)]),
        [event(np.int64(DAY), np.int64(0)), event(np.int64(2 * DAY), np.int64(0))],
        np.int64(0),
        np.int64(3 * DAY),
    )
    assert result.loc[:, list(NAMES)].eq(0).all().all()


@pytest.mark.parametrize(
    ("first_rate", "second_rate", "total"), [(0.1, 0.2, 0.3), (1e-8, 2e-8, 3e-8)]
)
def test_different_early_history_preserves_equal_current_values_and_cross_sectional_ties(
    first_rate, second_rate, total
):
    windows = [
        event(DAY + 1, first_rate),
        event(2 * DAY, second_rate),
        event(2 * DAY + 1, first_rate),
        event(3 * DAY, second_rate),
    ]
    histories = [[], [event(0, 0.01)], [event(0, 0.1)], [event(0, 1e308)], [event(0, -1e308)]]
    values = pd.DataFrame(
        [
            calculate([3 * DAY + LAG], history + windows, 0, 4 * DAY).iloc[0][list(NAMES)]
            for history in histories
        ]
    )

    assert values[NAMES[0]].eq(-total).all()
    assert values[NAMES[1]].eq(0.0).all()
    assert values.rank(method="average").eq(3.0).all().all()


def test_equal_decimal_window_sums_keep_ties_despite_different_event_decompositions():
    first = calculate([2 * DAY + LAG], [event(DAY, 0.3), event(2 * DAY, 0.3)], 0, 3 * DAY)
    second = calculate(
        [2 * DAY + LAG],
        [event(DAY, 0.3), event(DAY + 1, 0.1), event(2 * DAY, 0.2)],
        0,
        3 * DAY,
    )

    pd.testing.assert_frame_equal(first.loc[:, list(NAMES)], second.loc[:, list(NAMES)])
    assert first.iloc[0][NAMES[0]] == -0.3
    assert first.iloc[0][NAMES[1]] == 0.0


@pytest.mark.parametrize("rate", [1e308, -1e308, 5e-324, -5e-324])
def test_finite_extreme_rates_remain_valid_without_losing_small_window_values(rate):
    result = calculate(
        [3 * DAY + LAG],
        [event(0, 1e308), event(2 * DAY, rate), event(3 * DAY, rate)],
        0,
        4 * DAY,
    )
    assert result.iloc[0][NAMES[0]] == -rate
    assert result.iloc[0][NAMES[1]] == 0.0


def test_a_window_sum_outside_finite_float_range_remains_unknown():
    result = calculate(
        [2 * DAY + LAG],
        [event(DAY, 0.1), event(DAY + 1, 1e308), event(2 * DAY, 1e308)],
        0,
        3 * DAY,
    )
    assert result.loc[:, list(NAMES)].isna().all().all()


def test_future_extreme_rates_cannot_change_decimal_precision_of_past_values():
    clocks = [2 * DAY + LAG]
    past = [event(DAY, 0.3), event(DAY + 1, 0.1), event(2 * DAY, 0.2)]
    before = calculate(clocks, past, 0, 4 * DAY)
    after = calculate(clocks, past + [event(3 * DAY, 1e308), event(4 * DAY, 5e-324)], 0, 4 * DAY)
    pd.testing.assert_frame_equal(before, after)
