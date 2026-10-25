import numpy as np
import pandas as pd

from factors.common.max_effect import book, features


def test_lagged_extreme_return_and_beta_use_only_completed_daily_observations():
    dates = pd.date_range("2024-01-01", periods=220, tz="UTC")
    market = np.sin(np.arange(220)) * 0.01
    frames = {
        "BTCUSDT": pd.DataFrame(
            {"close": 100 * np.exp(np.cumsum(market)), "quote_volume": 1e9}, index=dates
        ),
        "X": pd.DataFrame(
            {"close": 100 * np.exp(np.cumsum(2 * market)), "quote_volume": 1e8}, index=dates
        ),
    }
    original = features(frames)
    cutoff = dates[200]
    frames["X"].loc[cutoff:, "close"] *= 1000
    changed = features(frames)
    pd.testing.assert_frame_equal(original["X"].loc[:cutoff], changed["X"].loc[:cutoff])
    raw = original["X"].loc[cutoff]
    returns = frames["X"].loc[frames["X"].index < cutoff, "close"].pct_change().dropna().tail(28)
    assert raw.max28 == returns.max()


def test_source_crypto_direction_and_beta_target_with_gross_scaled_down_only():
    rows = {
        f"X{i:02d}": pd.Series(
            {
                "max28": i * 0.01,
                "momentum28": -0.01 * i,
                "beta60": 0.5 + i * 0.1,
                "quote28": 1e8,
                "observations": 200,
            }
        )
        for i in range(12)
    }
    rows["BTCUSDT"] = pd.Series(
        {"max28": 0.2, "momentum28": 0.3, "beta60": 1.0, "quote28": 1e9, "observations": 200}
    )
    p = {
        "minimum_listing_observations": 180,
        "minimum_daily_quote_volume": 5e7,
        "maximum_eligible": 30,
        "minimum_eligible": 10,
        "long_names": 4,
        "short_names": 4,
    }
    raw, _ = book(rows, p, "max28")
    hedged, receipt = book(rows, p, "max28_btc_beta")
    assert raw["X11"] > 0 and raw["X00"] < 0 and raw["BTCUSDT"] == 0
    assert hedged.abs().sum() <= 1 + 1e-12 and abs(receipt["target_beta"]) < 1e-12
    assert receipt["gross_before_downscale"] >= 1
