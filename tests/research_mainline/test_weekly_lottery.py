import numpy as np
import pandas as pd
import pytest

from factors.common.weekly_lottery import reversal_book, weekly_features


def test_week_uses_eight_prices_with_decision_day_excluded():
    close = pd.Series(
        [100, 100, 100, 100, 100, 100, 100, 120, 1],
        index=pd.date_range("2025-01-01", periods=9, tz="UTC"),
        dtype=float,
    )
    row = weekly_features(close).iloc[-1]
    assert row.max7 == pytest.approx(np.log(1.2))
    assert row.reversal7 == pytest.approx(np.log(1.2))
    close.iloc[-1] = 10000
    pd.testing.assert_series_equal(weekly_features(close).iloc[-1], row)


def test_gap_and_bad_price_do_not_become_seven_available_observations():
    close = pd.Series(100.0, index=pd.date_range("2025-01-01", periods=10, tz="UTC"))
    with pytest.raises(ValueError):
        weekly_features(close.drop(close.index[3]))
    close.iloc[3] = np.nan
    assert weekly_features(close).iloc[-1].isna().all()
    close.iloc[3] = 0
    assert weekly_features(close).iloc[-1].isna().all()


def test_maximum_log_rank_equals_monotone_maximum_simple_rank():
    daily = np.array([-0.1, 0.2, 0.01, -0.02, 0.03, 0.05, -0.01])
    close = pd.Series(
        np.r_[100, 100 * np.cumprod(1 + daily), 1234],
        index=pd.date_range("2025-01-01", periods=9, tz="UTC"),
    )
    assert weekly_features(close).iloc[-1].max7 == pytest.approx(np.log1p(max(daily)))


def test_opposite_weekly_mechanism_is_explicit_and_same_control_construction():
    rows = {f"s{n:02}": {"max7": n, "reversal7": n} for n in range(30)}
    for candidate in ["max7_lottery_reversal", "ordinary7_reversal_control"]:
        weights, meta = reversal_book(rows, candidate)
        assert len(weights) == 12 and meta["long"] == [f"s{n:02}" for n in range(6)]
        assert sum(weights.values()) == pytest.approx(0)
        assert sum(abs(v) for v in weights.values()) == pytest.approx(1)
        assert weights["s00"] > 0 and weights["s29"] < 0


def test_missing_score_is_not_silently_removed_or_replaced():
    rows = {str(n): {"max7": float(n)} for n in range(10)}
    rows["1"]["max7"] = np.nan
    with pytest.raises(ValueError):
        reversal_book(rows, "max7_lottery_reversal")
    with pytest.raises(ValueError):
        reversal_book(rows, "flip_old_max28")
