"""从市场面板构建方向一致的反趋势候选特征。"""

# 说明：以下逻辑仅使用决策时点可获得的数据。
# ruff: noqa: RUF003

from __future__ import annotations

import numpy as np
import pandas as pd

COUNTERTREND_FEATURE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "ret3",
    "ret12",
    "ret20",
    "ret60",
    "rel_ret20",
    "rel_ret60",
    "ret20_rank_pct",
    "ret60_rank_pct",
    "market_down_breadth20",
    "market_up_breadth20",
    "donchian60_pos",
    "rsi14",
    "adx14",
    "atr14_pct",
    "atr14_ratio120",
    "range_expansion20",
    "quote_volume_ratio_med30",
    "taker_buy_ratio",
    "tf15_ret16",
    "tf1h_ret12",
)

_DERIVED_COLUMNS = COUNTERTREND_FEATURE_COLUMNS[4:]
_RAW_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def build_countertrend_feature_rows(
    raw_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """为多空两个方向构建因果反趋势特征记录。"""

    parts = [
        _build_single_symbol_features(frame, symbol=symbol)
        for symbol, frame in sorted(raw_frames.items())
        if frame is not None and not frame.empty
    ]
    if not parts:
        return pd.DataFrame(
            columns=["symbol", "bar_time", *COUNTERTREND_FEATURE_COLUMNS],
        )
    rows = pd.concat(parts, ignore_index=True, sort=False)
    rows = rows.sort_values(["symbol", "bar_time"], kind="stable").reset_index(
        drop=True,
    )
    groupers = [rows["bar_time"]]
    for column in ("ret20", "ret60"):
        values = pd.to_numeric(rows[column], errors="coerce")
        market_mean = values.groupby(groupers).transform("mean")
        rows[f"rel_{column}"] = values - market_mean
        rows[f"{column}_rank_pct"] = values.groupby(groupers).rank(
            pct=True,
            method="average",
        )
    ret20 = pd.to_numeric(rows["ret20"], errors="coerce")
    rows["market_down_breadth20"] = ret20.lt(0.0).groupby(groupers).transform("mean")
    rows["market_up_breadth20"] = ret20.gt(0.0).groupby(groupers).transform("mean")
    for column in _DERIVED_COLUMNS:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (
        rows.loc[
            :,
            ["symbol", "bar_time", *COUNTERTREND_FEATURE_COLUMNS],
        ]
        .sort_values(["bar_time", "symbol"], kind="stable")
        .reset_index(drop=True)
    )


def _build_single_symbol_features(
    raw: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    missing = sorted(set(_RAW_PRICE_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(
            "countertrend raw frame missing columns: " + ", ".join(missing),
        )
    frame = raw.copy().sort_index()
    for column in (*_RAW_PRICE_COLUMNS, "taker_buy_quote"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["bar_time"] = pd.to_datetime(frame.index, errors="coerce")
    frame["symbol"] = str(symbol)
    frame = frame.dropna(subset=["bar_time"])
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    for bars in (3, 12, 20, 60):
        frame[f"ret{bars}"] = _pct_change(close, close.shift(bars))

    # 计算 quote_volume，仅使用当前及历史数据。
    quote_volume = volume * close
    frame["quote_volume_ratio_med30"] = quote_volume / (
        quote_volume.shift(1).rolling(30, min_periods=5).median()
    ).replace(0.0, np.nan)
    taker_buy_quote = pd.to_numeric(
        frame.get("taker_buy_quote"),
        errors="coerce",
    )
    frame["taker_buy_ratio"] = (
        (taker_buy_quote / quote_volume.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0).fillna(0.5)
    )

    low60 = low.rolling(60, min_periods=20).min()
    high60 = high.rolling(60, min_periods=20).max()
    frame["donchian60_pos"] = (close - low60) / (high60 - low60).replace(
        0.0,
        np.nan,
    )
    frame["rsi14"] = _rsi(close, 14)
    _plus_di, _minus_di, frame["adx14"] = _adx(high, low, close, 14)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=7,
    ).mean()
    frame["atr14_pct"] = _pct_of_price(atr14, close)
    frame["atr14_ratio120"] = atr14 / (atr14.shift(1).rolling(120, min_periods=30).median()).replace(0.0, np.nan)
    frame["range_expansion20"] = true_range / (true_range.shift(1).rolling(20, min_periods=10).median()).replace(
        0.0,
        np.nan,
    )
    return _add_closed_timeframe_returns(frame)


def _add_closed_timeframe_returns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    timestamps = pd.to_datetime(output["bar_time"], errors="coerce")
    valid = timestamps.notna()
    base = output.loc[valid].copy()
    if base.empty:
        return output
    row_index = base.index
    base.index = pd.DatetimeIndex(timestamps.loc[valid])
    market = base[["high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    for prefix, rule, bars in (("tf15", "15min", 16), ("tf1h", "1h", 12)):
        candles = market.resample(
            rule,
            closed="right",
            label="right",
            origin="start_day",
        ).agg({"high": "max", "low": "min", "close": "last"})
        values = _pct_change(candles["close"], candles["close"].shift(bars))
        output.loc[row_index, f"{prefix}_ret{bars}"] = values.reindex(
            base.index,
            method="ffill",
        ).to_numpy()
    return output


def _rsi(close: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce")
    delta = values.diff()
    gain = (
        delta.clip(lower=0.0)
        .ewm(
            alpha=1.0 / float(window),
            adjust=False,
            min_periods=max(2, int(window) // 2),
        )
        .mean()
    )
    loss = (
        (-delta.clip(upper=0.0))
        .ewm(
            alpha=1.0 / float(window),
            adjust=False,
            min_periods=max(2, int(window) // 2),
        )
        .mean()
    )
    relative_strength = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    upward = high.diff()
    downward = -low.diff()
    plus_dm = pd.Series(
        np.where((upward > downward) & (upward > 0.0), upward, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((downward > upward) & (downward > 0.0), downward, 0.0),
        index=high.index,
    )
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    alpha = 1.0 / float(window)
    atr = true_range.ewm(
        alpha=alpha,
        adjust=False,
        min_periods=max(2, int(window) // 2),
    ).mean()
    plus_di = (
        100.0
        * plus_dm.ewm(
            alpha=alpha,
            adjust=False,
            min_periods=max(2, int(window) // 2),
        ).mean()
        / atr.replace(0.0, np.nan)
    )
    minus_di = (
        100.0
        * minus_dm.ewm(
            alpha=alpha,
            adjust=False,
            min_periods=max(2, int(window) // 2),
        ).mean()
        / atr.replace(0.0, np.nan)
    )
    directional_index = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan) * 100.0
    adx = directional_index.ewm(
        alpha=alpha,
        adjust=False,
        min_periods=max(2, int(window) // 2),
    ).mean()
    return plus_di, minus_di, adx


def _pct_change(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
        - 1.0
    ) * 100.0


def _pct_of_price(value: pd.Series, price: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce") / pd.to_numeric(price, errors="coerce").replace(0.0, np.nan) * 100.0


__all__ = ["COUNTERTREND_FEATURE_COLUMNS", "build_countertrend_feature_rows"]
