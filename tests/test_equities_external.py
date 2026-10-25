"""External rule causality and dividend mathematics, independent of profitability."""

import numpy as np
import pandas as pd
import pytest

from factors.common.total_return import gross_return_level
from factors.equity.external_monthly import targets
from tradingagents.research.equities.data import sessions


def sample():
    calendar = sessions("US", "2018-01-01", "2021-01-01")
    frame = pd.DataFrame(
        {"close": 100 + np.arange(len(calendar)) * 0.2, "dividend": 0.0}, index=calendar.index
    )
    return frame, calendar


def test_dividend_adjustment_is_known_ex_date_and_does_not_back_adjust_prior_history():
    close = pd.Series([100.0, 90.0, 91.0])
    dividend = pd.Series([0.0, 10.0, 0.0])
    result = gross_return_level(close, dividend)
    assert result.tolist() == pytest.approx([1.0, 1.0, 91 / 90])
    pd.testing.assert_series_equal(
        result.iloc[:2], gross_return_level(close.iloc[:2], dividend.iloc[:2])
    )
    dividend.iloc[1] = np.nan
    with pytest.raises(ValueError, match="complete"):
        gross_return_level(close, dividend)


def test_first_signal_is_actual_tenth_month_end_not_midmonth_prefix_end():
    frame, calendar = sample()
    weights, _ = targets({"SPY.US": frame}, calendar, "faber10m")
    assert not weights.loc[:"2018-10-30"].any().any()
    assert weights.at[pd.Timestamp("2018-10-31"), "SPY.US"] == 0.8
    cut = calendar.index.get_loc(pd.Timestamp("2018-10-15")) + 1
    prefix, _ = targets({"SPY.US": frame.iloc[:cut]}, calendar, "faber10m")
    pd.testing.assert_frame_equal(prefix, weights.iloc[:cut])
    assert not prefix.any().any()


def test_monthly_rule_ignores_midmonth_crash_until_next_registered_month_end():
    frame, calendar = sample()
    frame.loc["2020-05-15":, "close"] *= 0.1
    weights, _ = targets({"SPY.US": frame}, calendar, "faber10m")
    assert weights.at[pd.Timestamp("2020-05-15"), "SPY.US"] == 0.8
    assert weights.at[pd.Timestamp("2020-05-29"), "SPY.US"] == 0
    # Calendar month-end is Sunday; signal is known on the actual Friday close.
    assert pd.Timestamp("2020-05-31") not in weights.index


@pytest.mark.parametrize("candidate", ["faber10m", "faber10m_vol12"])
def test_prefixes_and_future_price_dividend_mutations_are_invariant(candidate):
    frame, calendar = sample()
    full, _ = targets({"A": frame, "B": frame.copy()}, calendar, candidate)
    for cut in (100, 350, 600):
        prefix, _ = targets({"A": frame.iloc[:cut], "B": frame.iloc[:cut]}, calendar, candidate)
        pd.testing.assert_frame_equal(prefix, full.iloc[:cut])
    changed = frame.copy()
    changed.iloc[600:, 0] *= 5
    changed.iloc[601:, 1] = 20
    perturbed, _ = targets({"A": changed, "B": changed}, calendar, candidate)
    pd.testing.assert_frame_equal(perturbed.iloc[:600], full.iloc[:600])
    assert full.sum(axis=1).le(0.800000001).all()
