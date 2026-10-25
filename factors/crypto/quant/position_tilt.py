"""H02单快照与固定分钟基线；只生成研究特征和标签。"""

import numpy as np
import pandas as pd

MINUTE = 60_000
HOUR = 60 * MINUTE
NAME = "top_position_account_tilt"
P_COLUMN = "sum_toptrader_long_short_ratio"
A_COLUMN = "count_toptrader_long_short_ratio"
FEATURES = (
    "simple_return_3m",
    "simple_return_20m",
    "simple_return_60m",
    "flow_3m",
    "flow_12m",
    "population_volatility_60m",
    "log_quote_volume_60m",
)
TARGET = "target_60m"
LABEL_DELAY = 62 * MINUTE


def numeric(frame, column):
    """来源状态与有限值必须同时有效，原文未知不能被数字文本覆盖。"""
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    valid = frame[column + "_value_status"].eq("KNOWN_RAW_VALUE")
    return values.where(valid & np.isfinite(values))


def tilt(hourly):
    """不把零比率填成极小数，不用跨钟插值或滑动平滑。"""
    p, a = numeric(hourly, P_COLUMN), numeric(hourly, A_COLUMN)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(p.where(p > 0)) - np.log(a.where(a > 0))
    return result.where(np.isfinite(result))


def _window(series, size, *, standard_deviation=False):
    """逐窗口直接归约，避免累计滚动状态在量级突变后遗留消减误差。"""
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) >= size:
        windows = np.lib.stride_tricks.sliding_window_view(values, size)
        result[size - 1 :] = (
            windows.std(axis=1, ddof=0) if standard_deviation else windows.sum(axis=1)
        )
    return pd.Series(result, index=series.index)


def minute_features(minutes):
    """完成钟s的open属于原分钟[s−1,s)，标签须映射到t+2和t+62。"""
    if (
        minutes.index.has_duplicates
        or not pd.api.types.is_integer_dtype(minutes.index.dtype)
        or not minutes.index.is_monotonic_increasing
        or (len(minutes) > 1 and not np.all(np.diff(minutes.index) == MINUTE))
    ):
        raise ValueError("分钟表必须为整数、连续且唯一的完成钟日历")
    close = numeric(minutes, "contract_1m_close").where(lambda v: v > 0)
    opening = numeric(minutes, "contract_1m_open").where(lambda v: v > 0)
    volume = numeric(minutes, "quote_volume").where(lambda v: v >= 0)
    buy = numeric(minutes, "taker_buy_quote_volume").where(lambda v: v >= 0)
    buy = buy.where(buy <= volume)
    result = pd.DataFrame(index=minutes.index)
    for k in (3, 20, 60):
        # 即使两端价格完整，中途缺价仍须使回看窗口未知。
        known = close.rolling(k + 1, min_periods=k + 1).count().eq(k + 1)
        result[f"simple_return_{k}m"] = (close / close.shift(k) - 1).where(known)
    for k in (3, 12):
        denominator = _window(volume, k)
        numerator = _window(buy, k)
        result[f"flow_{k}m"] = (2 * numerator / denominator - 1).where(denominator > 0)
    log_returns = np.log(close).diff()
    result["population_volatility_60m"] = _window(log_returns, 60, standard_deviation=True)
    result["log_quote_volume_60m"] = np.log1p(_window(volume, 60))
    result[TARGET] = opening.shift(-62) / opening.shift(-2) - 1
    # 路径覆盖只作诊断，不能用未来路径完整与否改变当前特征人口。
    path_known = close.notna().astype(int).rolling(61, min_periods=61).sum().shift(-62)
    result["label_path_complete"] = path_known.eq(61)
    return result.replace([np.inf, -np.inf], np.nan)
