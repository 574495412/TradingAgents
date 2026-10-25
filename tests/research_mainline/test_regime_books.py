"""Public reversion definition and source intent availability invariants."""

import numpy as np
import pandas as pd

from factors.common.regime_books import FAMILIES, intents, source_features


def test_mean_reversion_is_inactive_inside_the_registered_three_sigma_wings():
    index = pd.date_range("2023-01-01", periods=500, freq="D", tz="UTC")
    rng = np.random.default_rng(31)
    price = 100 * np.exp(rng.normal(0, 0.02, len(index)).cumsum())
    d = pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1000,
            "quote_volume": 1e8,
        },
        index=index,
    )
    f = source_features(d)
    fast = (d.close.ewm(span=4).mean() - d.close.ewm(span=16).mean()) / (f.relative_vol * d.close)
    interior = fast.abs() < fast.rolling(5000, min_periods=3).std() * 3
    assert (f.loc[interior, "mr_wings4"] == 0).all()
    wings = ~interior & fast.notna()
    np.testing.assert_allclose(f.loc[wings, "mr_wings4"], -fast.loc[wings])
    frames = {s: d.copy() for s in ["BTCUSDT"] + [f"A{i}" for i in range(11)]}
    rates = {
        s: pd.DataFrame({"funding_rate": rng.normal(0, 0.001, len(index))}, index=index)
        for s in frames
    }
    eligible = pd.DataFrame(True, index=index, columns=sorted(frames))
    normalization = dict.fromkeys(["pst_ewmac32", "pst_breakout10", "mr_wings4"], 1.0)
    full, _ = intents(frames, rates, eligible, normalization)
    for frame in frames.values():
        frame.loc[index[-1], "close"] *= 100
    for frame in rates.values():
        frame.loc[index[-1], "funding_rate"] = 100
    changed, _ = intents(frames, rates, eligible, normalization)
    assert set(full) == set(FAMILIES)
    for key in full:
        pd.testing.assert_frame_equal(full[key], changed[key])
        assert (full[key].abs().sum(axis=1) <= 1 + 1e-12).all()
    assert (full["carry_momentum"].sum(axis=1).abs() < 1e-12).all()
