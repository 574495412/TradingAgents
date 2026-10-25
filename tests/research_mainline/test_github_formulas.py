"""Source formula mathematics, completed-bar clocks and bounded name weights."""

import numpy as np
import pandas as pd

from factors.common.github_formulas import (
    PRICE_RULES,
    QLIB_RULES,
    books,
    capped_momentum,
    daily,
    formulas,
)
from research_tools.github_factor_study.run import snapshot_sources


def sample():
    index = pd.date_range("2023-01-01", periods=800, freq="D", tz="UTC")
    rng = np.random.default_rng(19)
    close = 100 * np.exp(rng.normal(0.0003, 0.02, len(index)).cumsum())
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1000, 2000, len(index)),
            "quote_volume": 1e8,
        },
        index=index,
    )


def test_ewmac_and_acceleration_follow_registered_definitions():
    d = sample()
    f = formulas(d)
    vol = d.close.diff().ewm(span=35, min_periods=10).std().clip(lower=1e-10)
    floor = vol.rolling(500, min_periods=100).quantile(0.05)
    floor.iloc[0] = 0
    reference_vol = pd.Series(np.maximum(vol, floor.ffill()), index=d.index)
    expected = (d.close.ewm(span=32).mean() - d.close.ewm(span=128).mean()) / reference_vol
    pd.testing.assert_series_equal(f.pst_ewmac32, expected, check_names=False)
    fast = (d.close.ewm(span=4).mean() - d.close.ewm(span=16).mean()) / reference_vol
    pd.testing.assert_series_equal(f.pst_accel4, fast - fast.shift(4), check_names=False)


def test_qlib_endpoint_residual_matches_independent_least_squares():
    d = sample()
    f = formulas(d)
    y = d.close.iloc[-20:].to_numpy()
    design = np.column_stack([np.ones(20), np.arange(20)])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    reference = (y[-1] - design[-1] @ beta) / y[-1]
    assert abs(reference - f.qlib_resi20.iloc[-1]) < 1e-12


def test_prefix_values_and_stock_calendar_are_preserved():
    d = sample()
    a = formulas(d)
    b = formulas(d.iloc[:650])
    pd.testing.assert_frame_equal(a.iloc[:650], b)
    altered = d.copy()
    altered.iloc[650:, :] = 1
    pd.testing.assert_frame_equal(formulas(altered).iloc[:650], a.iloc[:650])
    equity = d.loc[d.index.dayofweek < 5]
    assert daily(equity, "equity").index.equals(equity.index)


def test_momentum_allocation_is_capped_without_inventing_extra_gross():
    w = capped_momentum(pd.Series({"A": 10.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0}))
    assert abs(w.sum() - 1) < 1e-12
    assert w.max() <= 0.4 + 1e-12


def test_book_decisions_do_not_see_current_day_price_or_funding():
    index = pd.date_range("2024-01-01", periods=160, freq="D", tz="UTC")
    symbols = ["BTCUSDT"] + [f"C{i:02d}" for i in range(1, 12)]
    frames = {}
    rates = {}
    for i, s in enumerate(symbols):
        close = 100 * np.exp(np.arange(len(index)) * 0.0001 * i)
        frames[s] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1000 + np.arange(len(index)),
                "quote_volume": 1e8,
            },
            index=index,
        )
        rates[s] = pd.DataFrame(
            {"funding_rate": -i * 1e-5},
            index=pd.date_range(index[0], periods=len(index) * 3, freq="8h"),
        )
    norm = dict.fromkeys(PRICE_RULES, 1.0)
    direction = dict.fromkeys(QLIB_RULES, 1)
    original, _, _, _ = books(frames, rates, norm, direction)
    decision = index[-1]
    for s in symbols:
        frames[s].loc[decision, "close"] *= 10
        rates[s].loc[rates[s].index >= decision, "funding_rate"] = 0.5
    altered, _, _, _ = books(frames, rates, norm, direction)
    for k in original:
        pd.testing.assert_frame_equal(original[k], altered[k])
    last = original["sentinel_carry21"].loc[decision]
    assert set(last.loc[last > 0].index) == set(symbols[-5:])
    assert set(last.loc[last < 0].index) == set(symbols[:5])


def test_executed_source_archive_preserves_same_filename_in_different_packages(tmp_path):
    first = tmp_path / "a/run.py"
    second = tmp_path / "b/run.py"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("FIRST = 1\n")
    second.write_text("SECOND = 2\n")
    folder = tmp_path / "snapshot"
    folder.mkdir()
    manifest = snapshot_sources([first, second], folder)
    a = manifest[str(first.resolve())]["snapshot"]
    b = manifest[str(second.resolve())]["snapshot"]
    assert a != b
    assert (folder / a).read_bytes() == first.read_bytes()
    assert (folder / b).read_bytes() == second.read_bytes()
