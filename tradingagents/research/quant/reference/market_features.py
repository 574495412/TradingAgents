"""自主量化研究使用的已收盘行情特征。

输入按收盘时刻索引；运行时传入带时区的 as_of 和结算延迟，剔除当时尚未完成的分钟。
输出统一使用 UTC，各时点独立计算横截面，保留缺失日历。"""

# 公共行情特征入口：先筛选当时已收盘的数据，再计算滚动收益、波动及横截面排名。
# 索引采用收盘时刻；实盘传入 as_of 与结算缓冲，避免正在形成的 K 线进入决策。

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .factors import (
    aggressive_buy_share,
    average_true_range,
    close_location,
    cross_sectional_rank,
    market_breadth,
    market_return,
    momentum,
    order_flow_imbalance,
    path_efficiency,
    quote_liquidity,
    range_distance,
    realized_volatility,
    relative_volume,
    residual_momentum,
)

RETURN_WINDOWS = (1, 3, 12, 20, 60, 240)
RANGE_WINDOWS = (20, 60)
FLOW_WINDOWS = (3, 12)
VOLATILITY_WINDOWS = (12, 60)

RAW_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_quote",
)
FEATURE_COLUMNS = (
    *(f"ret_{window}_pct" for window in RETURN_WINDOWS),
    "atr_pct",
    "quote_volume_sum_12",
    "quote_volume_ratio_30",
    "close_position",
    "taker_buy_ratio",
    "taker_imbalance_3",
    "taker_imbalance_12",
    "return_volatility_12_pct",
    "return_volatility_60_pct",
    "path_efficiency_20",
    "from_high_20_pct",
    "from_low_20_pct",
    "from_high_60_pct",
    "from_low_60_pct",
    "ret_60_rank_pct",
    "ret_20_rank_pct",
    "ret_240_rank_pct",
    "atr_rank_pct",
    "liquidity_rank_pct",
    "volume_ratio_rank_pct",
    "taker_buy_rank_pct",
    "market_ret_1_median_pct",
    "market_ret_20_median_pct",
    "market_ret_60_dispersion_pct",
    "market_breadth_ret_1_positive",
    "market_breadth_ret_20_positive",
    "market_breadth_ret_20_change_5",
    "ret_20_residual_pct",
    "ret_60_residual_pct",
)
PANEL_COLUMNS = (*RAW_COLUMNS, *FEATURE_COLUMNS)
_CROSS_SECTIONAL_INPUTS = {
    "ret_60_rank_pct": frozenset({"ret_60_pct"}),
    "ret_20_rank_pct": frozenset({"ret_20_pct"}),
    "ret_240_rank_pct": frozenset({"ret_240_pct"}),
    "atr_rank_pct": frozenset({"atr_pct"}),
    "liquidity_rank_pct": frozenset({"quote_volume_sum_12"}),
    "volume_ratio_rank_pct": frozenset({"quote_volume_ratio_30"}),
    "taker_buy_rank_pct": frozenset({"taker_buy_ratio"}),
    "market_ret_1_median_pct": frozenset({"ret_1_pct"}),
    "market_ret_20_median_pct": frozenset({"ret_20_pct"}),
    "market_ret_60_dispersion_pct": frozenset({"ret_60_pct"}),
    "market_breadth_ret_1_positive": frozenset({"ret_1_pct"}),
    "market_breadth_ret_20_positive": frozenset({"ret_20_pct"}),
    "market_breadth_ret_20_change_5": frozenset({"ret_20_pct"}),
    "ret_20_residual_pct": frozenset({"ret_20_pct"}),
    "ret_60_residual_pct": frozenset({"ret_60_pct"}),
}


# 先完成单币已收盘特征，再在同一时间截面计算排名和市场广度。
def build_feature_panel(  # noqa: C901
    frames_by_symbol: Mapping[str, pd.DataFrame],
    *,
    as_of: pd.Timestamp | str | None = None,
    settle_seconds: int = 0,
    output_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """构建按时间与合约排序的已收盘特征面板。

    每个输入使用带时区的收盘索引或 close_time 列，分钟必须唯一且严格递增。
    as_of 存在时，仅保留不晚于 as_of 减去结算延迟的收盘；省略表示全部为已收盘历史。
    输出使用 UTC 的 bar_time 与 symbol 复合索引，先构建单币特征，再逐截面计算排名。"""

    if not isinstance(frames_by_symbol, Mapping):
        message = "frames_by_symbol must be a mapping"
        raise TypeError(message)
    selected_columns = _validated_output_columns(output_columns)
    validated_settle_seconds = _validated_settle_seconds(settle_seconds)
    cutoff = _closed_bar_cutoff(
        as_of,
        settle_seconds=validated_settle_seconds,
    )
    calculation_columns = _calculation_columns(selected_columns)
    symbol_frames = _build_symbol_frames(
        frames_by_symbol,
        cutoff=cutoff,
        columns=calculation_columns,
    )

    if not symbol_frames:
        return _empty_panel(columns=selected_columns)

    panel = pd.concat(symbol_frames, ignore_index=True, sort=False)
    panel = panel.set_index(["bar_time", "symbol"]).sort_index()
    if not panel.index.is_unique:
        message = "feature panel index must be unique"
        raise ValueError(message)

    rank_sources = {
        "ret_60_rank_pct": "ret_60_pct",
        "ret_20_rank_pct": "ret_20_pct",
        "ret_240_rank_pct": "ret_240_pct",
        "atr_rank_pct": "atr_pct",
        "liquidity_rank_pct": "quote_volume_sum_12",
        "volume_ratio_rank_pct": "quote_volume_ratio_30",
        "taker_buy_rank_pct": "taker_buy_ratio",
    }
    for output, source in rank_sources.items():
        if output in selected_columns:
            panel[output] = cross_sectional_rank.calculate(panel[source])
    transform_sources = {
        "market_ret_1_median_pct": ("ret_1_pct", "median"),
        "market_ret_20_median_pct": ("ret_20_pct", "median"),
        "market_ret_60_dispersion_pct": ("ret_60_pct", "std"),
    }
    for output, (source, operation) in transform_sources.items():
        if output in selected_columns:
            panel[output] = (
                market_return.median(panel[source])
                if operation == "median"
                else market_return.dispersion(panel[source])
            )
    breadth_sources = {
        "market_breadth_ret_1_positive": "ret_1_pct",
        "market_breadth_ret_20_positive": "ret_20_pct",
    }
    for output, source in breadth_sources.items():
        needed_for_change = (
            output == "market_breadth_ret_20_positive"
            and "market_breadth_ret_20_change_5" in selected_columns
        )
        if output in selected_columns or needed_for_change:
            panel[output] = market_breadth.calculate(panel[source])
    if "market_breadth_ret_20_change_5" in selected_columns:
        panel["market_breadth_ret_20_change_5"] = market_breadth.change(
            panel["market_breadth_ret_20_positive"]
        )
    for window in (20, 60):
        if f"ret_{window}_residual_pct" in selected_columns:
            panel[f"ret_{window}_residual_pct"] = residual_momentum.calculate(
                panel[f"ret_{window}_pct"]
            )
    return panel.loc[:, list(selected_columns)]


def _validated_output_columns(columns: Iterable[str] | None) -> tuple[str, ...]:
    if columns is None:
        return PANEL_COLUMNS
    requested = frozenset(str(column) for column in columns)
    unknown = sorted(requested.difference(PANEL_COLUMNS))
    if unknown:
        message = f"unknown feature panel columns: {', '.join(unknown)}"
        raise ValueError(message)
    return tuple(column for column in PANEL_COLUMNS if column in requested)


def _calculation_columns(selected_columns: Iterable[str]) -> tuple[str, ...]:
    required = set(selected_columns)
    for column in selected_columns:
        required.update(_CROSS_SECTIONAL_INPUTS.get(column, ()))
    cross_outputs = _CROSS_SECTIONAL_INPUTS.keys()
    return tuple(
        column for column in PANEL_COLUMNS if column in required and column not in cross_outputs
    )


def _validated_settle_seconds(value: int) -> int:
    message = "settle_seconds must be a non-negative integer"
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(message)
    validated = int(value)
    if validated < 0:
        raise ValueError(message)
    return validated


# 从当前时刻扣除结算缓冲，明确哪些收盘数据已经允许被本次决策使用。
def _closed_bar_cutoff(
    as_of: pd.Timestamp | str | None,
    *,
    settle_seconds: int,
) -> pd.Timestamp | None:
    if as_of is None:
        return None
    return _aware_utc_timestamp(as_of, name="as_of") - pd.Timedelta(
        seconds=settle_seconds,
    )


def _build_symbol_frames(
    frames_by_symbol: Mapping[str, pd.DataFrame],
    *,
    cutoff: pd.Timestamp | None,
    columns: Iterable[str],
) -> list[pd.DataFrame]:
    symbol_frames: list[pd.DataFrame] = []
    seen_symbols: set[str] = set()
    for raw_symbol, raw_frame in frames_by_symbol.items():
        symbol = _normalize_symbol(raw_symbol)
        if symbol in seen_symbols:
            message = f"duplicate normalized symbol: {symbol}"
            raise ValueError(message)
        seen_symbols.add(symbol)
        normalized = _normalize_market_frame(raw_frame, symbol=symbol)
        if cutoff is not None:
            normalized = normalized.loc[normalized.index <= cutoff]
        if normalized.empty:
            continue
        featured = _build_symbol_features(
            normalized,
            required_columns=frozenset(columns),
        )
        featured["symbol"] = symbol
        featured.index.name = "bar_time"
        symbol_frames.append(
            featured.loc[:, [*columns, "symbol"]].reset_index(),
        )
    return symbol_frames


def _normalize_market_frame(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        message = f"market frame for {symbol} must be a DataFrame"
        raise TypeError(message)
    missing = {"open", "high", "low", "close", "volume"}.difference(frame.columns)
    if missing:
        joined = ",".join(sorted(missing))
        message = f"market frame for {symbol} is missing: {joined}"
        raise ValueError(message)

    out = frame.copy()
    index = _close_time_index(out)
    _validate_close_time_index(index, symbol=symbol)
    out.index = index.tz_convert("UTC")
    _coerce_numeric_columns(out, symbol=symbol)
    _validate_prices_and_volume(out, symbol=symbol)
    _ensure_quote_volume(out, symbol=symbol)
    _ensure_taker_buy_quote(out, symbol=symbol)
    return out


def _close_time_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if "close_time" in frame.columns:
        close_time = pd.to_datetime(frame.pop("close_time"), errors="raise")
        return pd.DatetimeIndex(close_time, name="bar_time")
    converted = pd.to_datetime(frame.index, errors="raise")
    return pd.DatetimeIndex(converted, name="bar_time")


# 时间索引必须有时区、无重复且有序，否则滚动窗口与跨币对齐会失去确定性。
def _validate_close_time_index(index: pd.DatetimeIndex, *, symbol: str) -> None:
    if index.tz is None:
        message = f"close timestamps for {symbol} must be timezone-aware"
        raise ValueError(message)
    if index.has_duplicates:
        message = f"close timestamps for {symbol} must be unique"
        raise ValueError(message)
    if not index.is_monotonic_increasing:
        message = f"close timestamps for {symbol} must be strictly increasing"
        raise ValueError(message)


def _coerce_numeric_columns(frame: pd.DataFrame, *, symbol: str) -> None:
    numeric_candidates = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_quote",
        "taker_buy_quote_volume",
        "taker_buy_base",
        "taker_buy_ratio",
    }
    for column in numeric_candidates.intersection(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            message = f"{column} for {symbol} must contain finite values"
            raise ValueError(message)


def _validate_prices_and_volume(frame: pd.DataFrame, *, symbol: str) -> None:
    prices = frame.loc[:, ["open", "high", "low", "close"]]
    if (prices <= 0.0).any(axis=None):
        message = f"OHLC prices for {symbol} must be positive"
        raise ValueError(message)
    invalid_high = frame["high"] < frame[["open", "close"]].max(axis=1)
    invalid_low = frame["low"] > frame[["open", "close"]].min(axis=1)
    if invalid_high.any() or invalid_low.any():
        message = f"OHLC bounds for {symbol} are inconsistent"
        raise ValueError(message)
    if (frame["volume"] < 0.0).any():
        message = f"volume for {symbol} cannot be negative"
        raise ValueError(message)


def _ensure_quote_volume(frame: pd.DataFrame, *, symbol: str) -> None:
    if "quote_volume" not in frame:
        frame["quote_volume"] = frame["volume"] * frame["close"]
    if (frame["quote_volume"] < 0.0).any():
        message = f"quote_volume for {symbol} cannot be negative"
        raise ValueError(message)


def _ensure_taker_buy_quote(frame: pd.DataFrame, *, symbol: str) -> None:
    if "taker_buy_quote" not in frame:
        if "taker_buy_quote_volume" in frame:
            frame["taker_buy_quote"] = frame["taker_buy_quote_volume"]
        else:
            frame["taker_buy_quote"] = np.nan
    if frame["taker_buy_quote"].notna().any():
        present = frame["taker_buy_quote"].dropna()
        if (present < 0.0).any():
            message = f"taker_buy_quote for {symbol} cannot be negative"
            raise ValueError(message)
        quote_present = frame.loc[present.index, "quote_volume"]
        if (present > quote_present).any():
            message = f"taker_buy_quote for {symbol} cannot exceed quote_volume"
            raise ValueError(message)


def _build_symbol_features(
    frame: pd.DataFrame,
    *,
    required_columns: frozenset[str] | None = None,
) -> pd.DataFrame:
    """在已收盘输入边界内组合独立因子，保留原分钟日历。"""
    out = frame.copy()
    required = frozenset(FEATURE_COLUMNS) if required_columns is None else required_columns
    for window in RETURN_WINDOWS:
        if (name := f"ret_{window}_pct") in required:
            out[name] = momentum.calculate(frame, window)
    single_factors = {
        "atr_pct": average_true_range.calculate,
        "quote_volume_sum_12": quote_liquidity.calculate,
        "quote_volume_ratio_30": relative_volume.calculate,
        "close_position": close_location.calculate,
        "path_efficiency_20": path_efficiency.calculate,
    }
    for name, calculate in single_factors.items():
        if name in required:
            out[name] = calculate(frame)
    if required.intersection({"taker_buy_ratio", *(f"taker_imbalance_{w}" for w in FLOW_WINDOWS)}):
        buy_share = aggressive_buy_share.calculate(frame)
        if "taker_buy_ratio" in required:
            out["taker_buy_ratio"] = buy_share
        for window in FLOW_WINDOWS:
            if (name := f"taker_imbalance_{window}") in required:
                out[name] = order_flow_imbalance.calculate(buy_share, window)
    for window in VOLATILITY_WINDOWS:
        if (name := f"return_volatility_{window}_pct") in required:
            out[name] = realized_volatility.calculate(frame, window)
    for window in RANGE_WINDOWS:
        for side in ("high", "low"):
            if (name := f"from_{side}_{window}_pct") in required:
                out[name] = range_distance.calculate(frame, window, side)
    return out.replace([np.inf, -np.inf], np.nan)


def _normalize_symbol(value: object) -> str:
    symbol = str(value).strip().upper().replace("/", "").replace("-", "")
    if not symbol:
        message = "market symbol cannot be empty"
        raise ValueError(message)
    return symbol


def _aware_utc_timestamp(value: pd.Timestamp | str, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        message = f"{name} must be timezone-aware"
        raise ValueError(message)
    return timestamp.tz_convert("UTC")


def _empty_panel(*, columns: Iterable[str] = PANEL_COLUMNS) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        (
            pd.DatetimeIndex([], tz="UTC"),
            pd.Index([], dtype="object"),
        ),
        names=("bar_time", "symbol"),
    )
    return pd.DataFrame(index=index, columns=list(columns), dtype=float)
