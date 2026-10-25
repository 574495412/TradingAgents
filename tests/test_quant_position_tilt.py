"""H02时间与回归边界验证，合成数据不作为市场盈利证明。"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.factors.position_tilt import (
    A_COLUMN,
    FEATURES,
    LABEL_DELAY,
    MINUTE,
    NAME,
    P_COLUMN,
    TARGET,
    minute_features,
    tilt,
)
from tradingagents.research.quant.position_tilt_study import (
    ASSETS,
    CUTOFF,
    DAY,
    START,
    evaluate,
    models,
)
from tradingagents.research.quant.ridge import predict


def minutes():
    """每分钟指数价格便于直接区别简单收益与对数收益。"""
    index = pd.Index(np.arange(200, dtype=np.int64) * MINUTE, name="clock_ms")
    frame = pd.DataFrame(index=index)
    frame["contract_1m_close"] = np.exp(np.arange(200) * 0.001)
    frame["contract_1m_open"] = np.arange(200, dtype=float) + 100
    frame["quote_volume"] = np.arange(200, dtype=float) + 1
    frame["taker_buy_quote_volume"] = frame.quote_volume * 0.75
    for column in list(frame):
        frame[column + "_value_status"] = "KNOWN_RAW_VALUE"
    return frame


def test_exact_label_and_simple_returns():
    frame = minutes()
    row = minute_features(frame).loc[80 * MINUTE]
    assert row[TARGET] == pytest.approx((100 + 142) / (100 + 82) - 1)
    assert row.simple_return_3m == pytest.approx(np.exp(0.003) - 1)
    assert row.simple_return_60m == pytest.approx(np.exp(0.06) - 1)
    assert row.flow_3m == pytest.approx(0.5)
    assert row.population_volatility_60m == pytest.approx(0, abs=1e-15)
    assert row.log_quote_volume_60m == pytest.approx(np.log1p(sum(range(22, 82))))
    assert row.label_path_complete
    assert pd.isna(minute_features(frame).loc[199 * MINUTE, TARGET])


def test_population_std_and_minute_boundaries():
    frame = minutes()
    frame["contract_1m_close"] = np.exp(np.arange(200) ** 2 * 0.00001)
    row = minute_features(frame).loc[80 * MINUTE]
    expected = np.diff(np.log(frame.contract_1m_close.iloc[20:81]))
    assert row.population_volatility_60m == pytest.approx(expected.std(ddof=0))
    assert pd.isna(minute_features(frame).loc[59 * MINUTE, "population_volatility_60m"])


@pytest.mark.parametrize("damage", ["contract_1m_close", "quote_volume", "taker_buy_quote_volume"])
def test_unknown_state_is_not_repaired_by_numeric_text(damage):
    frame = minutes()
    frame.loc[79 * MINUTE, damage + "_value_status"] = "MISSING_CLOCK"
    row = minute_features(frame).loc[80 * MINUTE]
    column = "simple_return_3m" if damage == "contract_1m_close" else "flow_3m"
    assert pd.isna(row[column])


def test_future_mutation_does_not_change_current_features():
    frame = minutes()
    changed = frame.copy()
    changed.loc[changed.index > 80 * MINUTE, "contract_1m_close"] = 999.0
    changed.loc[changed.index > 80 * MINUTE, "quote_volume"] = 999.0
    pd.testing.assert_series_equal(
        minute_features(frame).loc[80 * MINUTE, list(FEATURES)],
        minute_features(changed).loc[80 * MINUTE, list(FEATURES)],
    )


def test_zero_volume_and_missing_path_remain_distinct():
    frame = minutes()
    frame.loc[70 * MINUTE : 80 * MINUTE, "quote_volume"] = 0.0
    frame.loc[70 * MINUTE : 80 * MINUTE, "taker_buy_quote_volume"] = 0.0
    row = minute_features(frame).loc[80 * MINUTE]
    assert pd.isna(row.flow_3m)
    assert np.isfinite(row.log_quote_volume_60m)
    frame.loc[100 * MINUTE, "contract_1m_close_value_status"] = "MISSING_CLOCK"
    row = minute_features(frame).loc[80 * MINUTE]
    assert not row.label_path_complete
    assert np.isfinite(row[TARGET])


def test_volume_scale_change_uses_only_the_exact_current_window():
    frame = minutes()
    frame.loc[: 50 * MINUTE, "quote_volume"] = 1e16
    frame.loc[: 50 * MINUTE, "taker_buy_quote_volume"] = 2e15
    frame.loc[51 * MINUTE :, "quote_volume"] = 0.000123456789
    frame.loc[51 * MINUTE :, "taker_buy_quote_volume"] = 0.000023456789
    row = minute_features(frame).loc[80 * MINUTE]
    assert row.flow_3m == pytest.approx(2 * 0.000023456789 / 0.000123456789 - 1, abs=1e-14)
    assert row.flow_12m == pytest.approx(row.flow_3m, abs=1e-14)


@pytest.mark.parametrize("kind", ["gap", "duplicate", "unsorted", "float"])
def test_invalid_calendar_rejected(kind):
    frame = minutes()
    if kind == "gap":
        frame = frame.drop(50 * MINUTE)
    elif kind == "duplicate":
        frame = pd.concat([frame, frame.iloc[-1:]])
    elif kind == "unsorted":
        frame = frame.iloc[::-1]
    else:
        frame.index = frame.index.astype(float)
    with pytest.raises(ValueError, match="日历"):
        minute_features(frame)


@pytest.mark.parametrize(
    "p,a,expected",
    [
        (4, 2, np.log(2)),
        (0, 2, None),
        (2, 0, None),
        (-1, 1, None),
        (np.inf, 1, None),
        (1, None, None),
    ],
)
def test_factor_zero_unknown_and_positive_direction(p, a, expected):
    frame = pd.DataFrame({P_COLUMN: [p], A_COLUMN: [a]})
    for name in (P_COLUMN, A_COLUMN):
        frame[name + "_value_status"] = "KNOWN_RAW_VALUE"
    value = tilt(frame).iloc[0]
    assert pd.isna(value) if expected is None else value == pytest.approx(expected)
    frame[P_COLUMN + "_value_status"] = "EMPTY"
    assert pd.isna(tilt(frame).iloc[0])


def training_frame():
    rng = np.random.default_rng(513)
    frame = pd.DataFrame(rng.normal(size=(80, len(FEATURES))), columns=FEATURES)
    frame[NAME] = rng.normal(size=len(frame))
    frame[TARGET] = frame[NAME] * 0.02 + rng.normal(scale=0.001, size=len(frame))
    frame["clock_ms"] = np.arange(len(frame), dtype=np.int64) * 3_600_000 + START
    frame["symbol"] = "SYNTHETIC"
    return frame


def test_lambda_one_matches_augmented_least_squares():
    frame = training_frame()
    record = models(frame)["augmented_common"]
    names = list(FEATURES) + [NAME]
    x = frame[names].to_numpy()
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
    y = frame[TARGET].to_numpy()
    # 独立增广最小二乘，不复用生产正规方程求解。
    matrix = np.vstack([z, np.sqrt(len(frame)) * np.eye(len(names))])
    beta = np.linalg.lstsq(matrix, np.r_[y - y.mean(), np.zeros(len(names))], rcond=None)[0]
    actual = record["fits"][0]
    assert actual["penalty"] == 1.0
    assert np.array(list(actual["coefficients"].values())) == pytest.approx(beta)
    assert actual["intercept"] == pytest.approx(y.mean())


def test_strict_label_visibility_and_common_population():
    frame = training_frame()
    frame.loc[0, NAME] = np.nan
    frame.loc[79, "clock_ms"] = CUTOFF - LABEL_DELAY
    before = models(frame)
    frame.loc[79, TARGET] = 1e8
    after = models(frame)
    assert before == after
    assert before["baseline_common"]["known_target_rows"] == 78
    assert before["baseline_own"]["known_target_rows"] == 79
    assert before["augmented_common"]["maximum_fitted_label_end_ms"] < CUTOFF
    prediction = predict(frame, before["baseline_common"], 1.0)
    assert pd.isna(prediction.iloc[0])


def test_constant_factor_and_future_rows_do_not_change_fit():
    frame = training_frame()
    frame[NAME] = 4.0
    before = models(frame)
    frame.loc[len(frame)] = frame.iloc[-1]
    frame.loc[len(frame) - 1, "clock_ms"] = CUTOFF
    frame.loc[len(frame) - 1, TARGET] = 1e8
    after = models(frame)
    assert before == after
    record = before["augmented_common"]
    assert NAME in record["constant_features"]
    assert record["fits"][0]["coefficients"][NAME] == 0.0


def test_full_evaluation_keeps_both_tests_unknown_days_and_contributions():
    rng = np.random.default_rng(513)
    rows = []
    for asset in ASSETS:
        for clock in [START + i * 3_600_000 for i in range(24)] + [
            CUTOFF + i * DAY for i in range(10)
        ]:
            for symbol in range(20):
                row = dict(zip(FEATURES, rng.normal(size=len(FEATURES)), strict=True))
                row.update(
                    symbol=f"S{symbol:02}",
                    asset_class=asset,
                    clock_ms=clock,
                    label_path_complete=True,
                )
                row[NAME] = rng.normal()
                row[TARGET] = row[NAME] * 0.02 + rng.normal(scale=0.001)
                if clock == CUTOFF + 4 * DAY:
                    row[TARGET] = np.nan
                if asset == ASSETS[1]:
                    row[NAME] = np.nan
                rows.append(row)
    result, _, summaries, tests, _ = evaluate(pd.DataFrame(rows))
    assert len(result) == 400
    assert len(tests) == 2
    assert all(row["bh_family_size"] == 2 for row in tests)
    assert tests[1]["unknown_p_counted_as_one"]
    assert not tests[1]["exploratory_predictive_screen_passed"]
    assert summaries[ASSETS[0]]["known_primary_days"] == 9
    assert result[
        result.asset_class.eq(ASSETS[0])
    ].daily_equal_weight_contribution.sum() == pytest.approx(tests[0]["statistic"]["mean"])
    assert tests[0]["daily"][4]["mse_improvement"] is None
