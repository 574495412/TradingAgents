"""用独立逐窗口参照核对固定特征，不读取真实市场数据。"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.research.quant.factors.alpha158 import NAMES, PROFILE, WINDOWS, calculate

KBAR = ("KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2")
FAMILIES = (
    "ROC",
    "MA",
    "STD",
    "BETA",
    "RSQR",
    "RESI",
    "MAX",
    "MIN",
    "QTLU",
    "QTLD",
    "RANK",
    "RSV",
    "IMAX",
    "IMIN",
    "IMXD",
    "CORR",
    "CORD",
    "CNTP",
    "CNTN",
    "CNTD",
    "SUMP",
    "SUMN",
    "SUMD",
    "VMA",
    "VSTD",
    "WVMA",
    "VSUMP",
    "VSUMN",
    "VSUMD",
)


def minutes(count=160):
    x = np.arange(count, dtype=float)
    close = 20 + 0.025 * x + 0.45 * np.sin(0.39 * x) + 0.14 * np.cos(0.73 * x)
    opening = close * (1 + 0.003 * np.sin(0.2 * x))
    volume = 200 + 15 * np.cos(0.31 * x) + 0.9 * x + 25 * np.sin(0.7 * x) ** 2
    return pd.DataFrame(
        {
            "open": opening,
            "high": np.maximum(opening, close) + 0.1 + 0.03 * np.cos(x) ** 2,
            "low": np.minimum(opening, close) - 0.1 - 0.02 * np.sin(x) ** 2,
            "close": close,
            "base_volume": volume,
            "quote_volume": volume * (opening + close) / 2,
        },
        index=pd.Index(np.arange(1, count + 1) * 60_000, name="clock_ms"),
    )


def reference(frame, position):
    """逐窗重算；不使用模块辅助函数、Pandas rolling 或共用回归实现。"""
    if position < 60:
        return dict.fromkeys(NAMES, np.nan)
    history = frame.iloc[position - 60 : position + 1]
    if not np.isfinite(history.to_numpy()).all() or not (np.diff(history.index) == 60_000).all():
        return dict.fromkeys(NAMES, np.nan)
    opening, high, low, close, volume, quote = frame.iloc[position]
    span = high - low
    upper, lower = high - max(opening, close), min(opening, close) - low
    displacement = 2 * close - high - low
    values = dict(
        zip(
            KBAR,
            (
                (close - opening) / opening,
                span / opening,
                (close - opening) / (span + 1e-12),
                upper / opening,
                upper / (span + 1e-12),
                lower / opening,
                lower / (span + 1e-12),
                displacement / opening,
                displacement / (span + 1e-12),
            ),
            strict=True,
        )
    )
    values.update(OPEN0=opening / close, HIGH0=high / close, LOW0=low / close)
    values["VWAP0"] = quote / volume / close if volume > 0 else np.nan

    def corr(left, right):
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            return np.nan
        if np.std(left, ddof=1) <= 2e-5 or np.std(right, ddof=1) <= 2e-5:
            return np.nan
        return np.corrcoef(left, right)[0, 1]

    for n in (5, 10, 20, 30, 60):
        window = frame.iloc[position - n + 1 : position + 1]
        prior = frame.iloc[position - n : position]
        c, h, lo, v = (window[name].to_numpy() for name in ("close", "high", "low", "base_volume"))
        previous_c, previous_v = (prior[name].to_numpy() for name in ("close", "base_volume"))
        x = np.column_stack([np.ones(n), np.arange(1, n + 1)])
        intercept, slope = np.linalg.lstsq(x, c, rcond=None)[0]
        residuals = c - (intercept + slope * x[:, 1])
        total_square = np.sum((c - np.mean(c)) ** 2)
        square = 1 - residuals @ residuals / total_square if np.std(c, ddof=1) > 2e-5 else np.nan
        price_difference, volume_difference = c - previous_c, v - previous_v
        gain, loss = np.maximum(price_difference, 0).sum(), np.maximum(-price_difference, 0).sum()
        volume_gain, volume_loss = (
            np.maximum(volume_difference, 0).sum(),
            np.maximum(-volume_difference, 0).sum(),
        )
        weighted = np.abs(c / previous_c - 1) * v
        volume_ratio = np.divide(v, previous_v, out=np.full(n, np.nan), where=previous_v > 0)
        up, down = np.mean(c > previous_c), np.mean(c < previous_c)
        family = {
            "ROC": previous_c[0] / close,
            "MA": np.mean(c) / close,
            "STD": np.std(c, ddof=1) / close,
            "BETA": slope / close,
            "RSQR": square,
            "RESI": residuals[-1] / close,
            "MAX": np.max(h) / close,
            "MIN": np.min(lo) / close,
            "QTLU": np.quantile(c, 0.8, method="linear") / close,
            "QTLD": np.quantile(c, 0.2, method="linear") / close,
            "RANK": (np.sum(c < close) + (np.sum(c == close) + 1) / 2) / n,
            "RSV": (close - np.min(lo)) / (np.max(h) - np.min(lo) + 1e-12),
            "IMAX": (np.argmax(h) + 1) / n,
            "IMIN": (np.argmin(lo) + 1) / n,
            "IMXD": (np.argmax(h) - np.argmin(lo)) / n,
            "CORR": corr(c, np.log(v + 1)),
            "CORD": corr(c / previous_c, np.log(volume_ratio + 1)),
            "CNTP": up,
            "CNTN": down,
            "CNTD": up - down,
            "SUMP": gain / (np.abs(price_difference).sum() + 1e-12),
            "SUMN": loss / (np.abs(price_difference).sum() + 1e-12),
            "SUMD": (gain - loss) / (np.abs(price_difference).sum() + 1e-12),
            "VMA": np.mean(v) / (volume + 1e-12),
            "VSTD": np.std(v, ddof=1) / (volume + 1e-12),
            "WVMA": np.std(weighted, ddof=1) / (np.mean(weighted) + 1e-12),
            "VSUMP": volume_gain / (np.abs(volume_difference).sum() + 1e-12),
            "VSUMN": volume_loss / (np.abs(volume_difference).sum() + 1e-12),
            "VSUMD": (volume_gain - volume_loss) / (np.abs(volume_difference).sum() + 1e-12),
        }
        values.update({f"{name}{n}": value for name, value in family.items()})
    return values


def test_fixed_default_names_order_and_first_complete_boundary():
    expected = (
        KBAR
        + ("OPEN0", "HIGH0", "LOW0", "VWAP0")
        + tuple(f"{family}{n}" for family in FAMILIES for n in (5, 10, 20, 30, 60))
    )
    assert PROFILE == "alpha158_1m_strict_v1"
    assert WINDOWS == (5, 10, 20, 30, 60)
    assert expected == NAMES and len(set(NAMES)) == 158
    source = minutes()
    original = source.copy(deep=True)
    result = calculate(source)
    assert tuple(result.columns) == expected and result.index.equals(source.index)
    assert result.iloc[:60].isna().all().all()
    assert result.iloc[60].notna().all()
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.parametrize("variant", ["ordinary", "zero_volume", "ties", "low_price"])
def test_all_158_values_match_independent_window_reference(variant):
    frame = minutes()
    if variant == "zero_volume":
        frame.loc[frame.index[80], ["base_volume", "quote_volume"]] = 0
    elif variant == "ties":
        frame["high"] = 30.0
        frame["low"] = 10.0
        frame.loc[frame.index[120:125], "close"] = frame.close.iloc[120]
        frame.loc[frame.index[120:125], "open"] = frame.close.iloc[120]
        frame["quote_volume"] = frame.base_volume * (frame.open + frame.close) / 2
    elif variant == "low_price":
        frame[["open", "high", "low", "close", "quote_volume"]] *= 1e-6
    result = calculate(frame)
    for position in (60, 61, 79, 80, 81, 84, 85, 100, 120, 124, 140, 159):
        expected = reference(frame, position)
        np.testing.assert_allclose(
            result.iloc[position].to_numpy(),
            [expected[name] for name in NAMES],
            rtol=1e-7,
            atol=1e-9,
            equal_nan=True,
            err_msg=f"全部特征族独立参照：{variant}，位置{position}",
        )


def test_future_price_and_volume_cannot_change_prior_features():
    frame = minutes()
    changed = frame.copy()
    changed.loc[changed.index[110:], ["open", "high", "low", "close"]] *= 3
    changed.loc[changed.index[110:], "base_volume"] *= 7
    changed.loc[changed.index[110:], "quote_volume"] *= 21
    pd.testing.assert_frame_equal(calculate(frame).iloc[:110], calculate(changed).iloc[:110])


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "base_volume", "quote_volume"])
def test_each_raw_missing_field_requires_61_new_complete_minutes(field):
    frame = minutes()
    frame.loc[frame.index[80], field] = np.nan
    result = calculate(frame)
    assert result.iloc[79].notna().all()
    assert result.iloc[80:141].isna().all().all()
    assert result.iloc[141].notna().all()


def test_missing_clock_is_not_compressed_and_preserves_original_index():
    original = minutes()
    frame = original.drop(original.index[80])
    result = calculate(frame)
    assert result.index.equals(frame.index)
    assert result.loc[original.index[81:141]].isna().all().all()
    assert result.loc[original.index[141]].notna().all()


def test_zero_volume_keeps_vwap_unknown_and_cord_full_window_unknown():
    frame = minutes()
    frame.loc[frame.index[80], ["base_volume", "quote_volume"]] = 0
    result = calculate(frame)
    assert np.isnan(result.VWAP0.iloc[80])
    assert result.VMA5.iloc[80] > 1e12
    assert result.KMID.iloc[80] == pytest.approx(reference(frame, 80)["KMID"])
    for n in WINDOWS:
        assert result[f"CORD{n}"].iloc[81 : 81 + n].isna().all()
        assert np.isfinite(result[f"CORD{n}"].iloc[81 + n])
    assert not np.isinf(result.to_numpy()).any()


def test_constant_prices_have_valid_zero_statistics_and_unknown_correlations():
    frame = minutes()
    frame[["open", "high", "low", "close"]] = 5.0
    frame["quote_volume"] = 5 * frame.base_volume
    row = calculate(frame).iloc[-1]
    assert all(row[name] == 0 for name in KBAR)
    assert all(row[name] == 1 for name in ("OPEN0", "HIGH0", "LOW0", "VWAP0"))
    for n in WINDOWS:
        assert row[f"STD{n}"] == row[f"BETA{n}"] == row[f"RESI{n}"] == 0
        assert row[f"RANK{n}"] == (n + 1) / (2 * n)
        assert row[f"IMAX{n}"] == row[f"IMIN{n}"] == 1 / n
        assert row[f"IMXD{n}"] == row[f"WVMA{n}"] == 0
        assert np.isnan(row[f"RSQR{n}"])
        assert np.isnan(row[f"CORR{n}"]) and np.isnan(row[f"CORD{n}"])


def test_original_absolute_std_threshold_retains_price_unit_bias():
    original = minutes()
    tiny = original.copy()
    tiny[["open", "high", "low", "close", "quote_volume"]] *= 1e-6
    regular, scaled = calculate(original), calculate(tiny)
    for n in WINDOWS:
        assert regular[f"RSQR{n}"].iloc[60:].notna().all()
        assert regular[f"CORR{n}"].iloc[60:].notna().all()
        assert scaled[f"RSQR{n}"].iloc[60:].isna().all()
        assert scaled[f"CORR{n}"].iloc[60:].isna().all()
        np.testing.assert_allclose(
            regular[f"CORD{n}"], scaled[f"CORD{n}"], rtol=1e-7, atol=1e-8, equal_nan=True
        )


def test_constant_volume_is_known_zero_variance_not_missing_coverage():
    frame = minutes()
    frame["base_volume"] = 100.0
    frame["quote_volume"] = 100 * (frame.open + frame.close) / 2
    row = calculate(frame).iloc[-1]
    for n in WINDOWS:
        assert row[f"VMA{n}"] == pytest.approx(1)
        assert row[f"VSTD{n}"] == row[f"VSUMP{n}"] == row[f"VSUMN{n}"] == row[f"VSUMD{n}"] == 0
        assert np.isnan(row[f"CORR{n}"]) and np.isnan(row[f"CORD{n}"])
    assert np.isfinite(row.KMID)


def test_empty_and_short_inputs_keep_fixed_output_shape():
    assert calculate(minutes().iloc[:0]).shape == (0, 158)
    short = calculate(minutes(60))
    assert short.shape == (60, 158) and short.isna().all().all()


@pytest.mark.parametrize(
    "bad_index",
    [
        [60_000, 60_000],
        [120_000, 60_000],
        [60_001, 120_000],
        [60_000.0, 120_000.0],
        [False, 60_000],
        [-60_000, 0],
    ],
)
def test_invalid_clock_contract_is_rejected(bad_index):
    frame = minutes(2)
    frame.index = bad_index
    with pytest.raises(ValueError):
        calculate(frame)


@pytest.mark.parametrize(
    "field,value",
    [
        ("open", 0),
        ("close", -1),
        ("high", 1),
        ("low", 100),
        ("base_volume", -1),
        ("quote_volume", -1),
        ("close", np.inf),
        ("base_volume", 0),
        ("quote_volume", 0),
        ("quote_volume", 1e12),
    ],
)
def test_malformed_raw_values_are_rejected(field, value):
    frame = minutes()
    frame.loc[frame.index[80], field] = value
    with pytest.raises(ValueError):
        calculate(frame)


def test_missing_duplicate_or_non_numeric_columns_are_rejected():
    with pytest.raises(ValueError):
        calculate(minutes().drop(columns="base_volume"))
    duplicate = minutes()
    duplicate.columns = ["open", "high", "low", "close", "quote_volume", "quote_volume"]
    with pytest.raises(ValueError):
        calculate(duplicate)
    for value in ("string", True, 1 + 2j):
        frame = minutes()
        frame["base_volume"] = value
        with pytest.raises(ValueError):
            calculate(frame)
