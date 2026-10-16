"""响应因子须隔离当前冲击拟合、未来数据、缺分钟和零分母。"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.factors.response import NAMES, calculate


def minutes():
    generator = np.random.default_rng(20261016)
    flow = generator.uniform(-0.7, 0.7, 600)
    returns = 0.00002 + 0.0005 * flow + generator.normal(0, 0.0001, 600)
    close = np.exp(np.cumsum(returns)) * 100
    return pd.DataFrame(
        {"open": close, "close": close, "quote_volume": 100, "taker_buy_quote": (flow + 1) * 50},
        index=np.arange(600) * 60_000,
    )


def test_future_prices_and_volume_cannot_change_current_response_factors():
    raw = minutes()
    first = calculate(raw)
    changed = raw.copy()
    changed.loc[450 * 60_000 :, "close"] *= 2
    changed.loc[450 * 60_000 :, "taker_buy_quote"] = 0
    second = calculate(changed)
    pd.testing.assert_series_equal(first.loc[400 * 60_000], second.loc[400 * 60_000])
    assert list(first) == list(NAMES)


def test_impact_uses_a_disjoint_past_ols_with_intercept_and_correct_current_window():
    raw = minutes()
    at = 400
    r = np.log(raw.close).diff().to_numpy()
    f = (2 * raw.taker_buy_quote / raw.quote_volume - 1).to_numpy()
    # 过去120分钟结束于 t-20，当前窗口是 t-19 到 t，两者没有重复行。
    x, y = f[at - 139 : at - 19], r[at - 139 : at - 19]
    matrix = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
    expected = -(r[at - 19 : at + 1].sum() - 20 * alpha - beta * f[at - 19 : at + 1].sum()) / (
        y.std(ddof=1) * np.sqrt(20)
    )
    assert calculate(raw).iloc[at, 0] == pytest.approx(expected, rel=1e-9, abs=1e-10)


def test_zero_flow_variance_and_zero_price_volatility_remain_unknown():
    raw = minutes()
    raw.taker_buy_quote = 55
    assert calculate(raw)[NAMES[0]].iloc[400:].isna().all()
    raw.close = 100
    values = calculate(raw)
    assert values.iloc[400:].isna().all().all()


def test_missing_minutes_or_zero_volume_do_not_become_zero_information():
    raw = minutes()
    raw.loc[390 * 60_000, "quote_volume"] = 0
    raw.loc[390 * 60_000, "taker_buy_quote"] = 0
    assert pd.isna(calculate(raw).loc[400 * 60_000, NAMES[0]])
    with pytest.raises(ValueError, match="分钟日历"):
        calculate(raw.drop(index=10 * 60_000))


def test_skip_momentum_ignores_the_last_hour_but_keeps_the_original_calendar():
    raw = minutes()
    before = calculate(raw)
    altered = raw.copy()
    altered.loc[350 * 60_000 : 400 * 60_000, "close"] *= 1.2
    after = calculate(altered)
    assert before.loc[400 * 60_000, NAMES[2]] == after.loc[400 * 60_000, NAMES[2]]
