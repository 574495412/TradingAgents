"""No current-bar flow, no unmatured labels, and explicit signal lease boundaries."""

import numpy as np
import pandas as pd

from factors.common.mechanisms import PRICE_COLUMNS, features, lease_model_score, rule_scores
from research_tools.factor_discovery.run import fit_month


def market():
    clocks = pd.date_range("2024-01-01", periods=1800, freq="h", tz="UTC")
    rng = np.random.default_rng(9)
    close = 100 * np.exp(np.cumsum(rng.normal(0.00005, 0.002, len(clocks))))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1e5,
            "quote_volume": 1e8,
            "taker_buy_quote_volume": 1e8 * (0.5 + 0.03 * np.sin(np.arange(len(clocks)) / 5)),
        },
        index=clocks,
    )
    funding = pd.DataFrame({"funding_rate": 0.0001, "mark": 100}, index=clocks[::8])
    return frame, funding


def test_current_and_future_aggressor_data_cannot_change_available_features():
    frame, rates = market()
    original, _ = features(frame, rates)
    altered = frame.copy()
    altered.iloc[1700:, altered.columns.get_loc("taker_buy_quote_volume")] = 1e8
    other, _ = features(altered, rates)
    pd.testing.assert_frame_equal(original.iloc[:1701], other.iloc[:1701])
    prefix, _ = features(frame.iloc[:1701], rates)
    pd.testing.assert_frame_equal(prefix, original.iloc[:1701])


def test_settled_funding_is_delayed_and_missing_real_flow_stays_unknown():
    frame, rates = market()
    changed = rates.copy()
    changed.loc[changed.index[10], "funding_rate"] = 0.1
    a, _ = features(frame, rates)
    b, _ = features(frame, changed)
    clock = changed.index[10]
    assert a.loc[clock, "funding_last"] == b.loc[clock, "funding_last"]
    assert b.loc[clock + pd.Timedelta(hours=1), "funding_last"] == 0.1
    missing, _ = features(frame.drop(columns="taker_buy_quote_volume"), rates)
    assert missing.aggressor24.isna().all()
    assert missing.aggressor_surprise24.isna().all()


def test_equity_features_preserve_observed_session_calendar():
    frame, _ = market()
    session = frame.loc[
        (frame.index.dayofweek < 5) & (frame.index.hour >= 14) & (frame.index.hour < 20)
    ]
    output, _ = features(session, asset_class="equity")
    assert output.index.equals(session.index)
    assert output.funding_last.isna().all()


def test_both_rules_and_models_have_seven_day_signal_lease_and_rearm_gap():
    clocks = pd.date_range("2025-01-01", periods=24 * 10, freq="h", tz="UTC")
    score = pd.Series(0.02, index=clocks)
    model = lease_model_score(score)
    assert model.loc["2025-01-07"].gt(0).all()
    assert model.loc["2025-01-08"].eq(0).all()
    assert model.loc["2025-01-09"].gt(0).all()
    f = pd.DataFrame(1.0, index=clocks, columns=PRICE_COLUMNS + ["quote_day28"])
    f["quote_day28"] = 1e9
    f["trend_consensus"] = 0.5
    f["aggressor_surprise24"] = 0.0
    rule = rule_scores(f, "slow_trend")
    assert rule.loc["2025-01-07"].gt(0).all()
    assert rule.loc["2025-01-08"].eq(0).all()
    assert rule.loc["2025-01-09"].gt(0).all()


def test_72hour_label_maturity_and_training_preprocessing_are_isolated():
    clocks = pd.date_range("2024-06-01", "2025-01-05", freq="h", tz="UTC")
    ix = pd.MultiIndex.from_product([clocks, ["A", "B"]], names=["time", "symbol"])
    rng = np.random.default_rng(2)
    x = pd.DataFrame(rng.normal(size=(len(ix), 2)), index=ix, columns=["a", "b"])
    y = x.a * 0.01 + x.b * 0.002
    first, model = fit_month(x, y, "2025-01-01", "2025-01-05", list(x))
    later = y.copy()
    later.loc[ix.get_level_values("time") >= pd.Timestamp("2024-12-29", tz="UTC")] = 999
    second, other = fit_month(x, later, "2025-01-01", "2025-01-05", list(x))
    pd.testing.assert_series_equal(first, second)
    assert model == other
    assert pd.Timestamp(model["latest_matured_label"]) < pd.Timestamp(model["evaluation_start"])
