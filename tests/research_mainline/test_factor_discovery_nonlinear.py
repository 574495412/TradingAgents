"""Shallow-tree training uses daily matured cohorts without a random early-stop split."""

import numpy as np
import pandas as pd

from factors.common.regime import condition_market
from research_tools.factor_discovery.nonlinear import PARAMS, fit


def test_market_conditioning_uses_same_clock_information_only():
    t = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    ix = pd.MultiIndex.from_product([t, ["BTCUSDT", "ETHUSDT"]], names=["time", "symbol"])
    x = pd.DataFrame(
        {"trend_consensus": np.linspace(-1, 1, len(ix)), "mom_z168": np.arange(len(ix))}, index=ix
    )
    a = condition_market(x)
    changed = x.copy()
    changed.loc[(t[-1], "BTCUSDT"), :] = 999
    b = condition_market(changed)
    pd.testing.assert_frame_equal(a.iloc[:-2], b.iloc[:-2])
    assert a.loc[(t[1], "BTCUSDT"), "relative_mom7"] == 0


def test_tree_cannot_see_unmatured_labels_and_disables_random_validation():
    t = pd.date_range("2024-07-01", "2025-01-04", freq="h", tz="UTC")
    ix = pd.MultiIndex.from_product([t, ["A", "B", "C", "D"]], names=["time", "symbol"])
    rng = np.random.default_rng(4)
    x = pd.DataFrame(rng.normal(size=(len(ix), 2)), columns=["a", "b"], index=ix)
    y = x.a * 0.01 + x.b * 0.002
    first, _, m = fit(x, y, "2025-01-01", "2025-01-04", list(x))
    altered = y.copy()
    altered.loc[ix.get_level_values("time") >= pd.Timestamp("2024-12-29", tz="UTC")] = 999
    second, _, m2 = fit(x, altered, "2025-01-01", "2025-01-04", list(x))
    pd.testing.assert_series_equal(first, second)
    assert m == m2
    assert m["training_rows"] < 720
    assert pd.Timestamp(m["latest_matured_label"]) < pd.Timestamp(m["evaluation_start"])
    assert PARAMS["early_stopping"] is False
