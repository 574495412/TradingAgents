"""固定 Alpha158 的分钟适配；只使用完成分钟，不产生标签或交易信号。

公式依据 Microsoft Qlib 的 MIT 授权版本；确切版本见 QLIB_COMMIT。
严格完整窗口与零分母未知是本适配的显式扩展，不是默认日频工作流。
"""

from itertools import pairwise
from numbers import Integral

import numpy as np
import pandas as pd

PROFILE = "alpha158_1m_strict_v1"
QLIB_COMMIT = "54355232463878d2eebb91fe0ee5fa7fa1f5976c"
MINUTE = 60_000
HISTORY = 61
EPSILON = 1e-12
ZERO_STD_ATOL = 2e-5
INPUTS = ("open", "high", "low", "close", "base_volume", "quote_volume")
WINDOWS = (5, 10, 20, 30, 60)
KBAR_NAMES = ("KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2")
PRICE_NAMES = ("OPEN0", "HIGH0", "LOW0", "VWAP0")
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
NAMES = KBAR_NAMES + PRICE_NAMES + tuple(f"{family}{n}" for family in FAMILIES for n in WINDOWS)


def _inputs(minutes):
    """拒绝非法原文；NaN 表达缺失，零成交量仍是合法观测。"""
    if not isinstance(minutes, pd.DataFrame) or minutes.columns.has_duplicates:
        raise ValueError("分钟输入必须是列名唯一的 DataFrame")
    if not set(INPUTS) <= set(minutes.columns):
        raise ValueError("分钟输入缺少 OHLC、base_volume 或 quote_volume")
    clocks = list(minutes.index)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in clocks):
        raise ValueError("分钟完成边界必须是整数毫秒")
    if any(v < 0 or v % MINUTE for v in clocks) or any(
        right <= left for left, right in pairwise(clocks)
    ):
        raise ValueError("分钟完成边界必须非负、对齐、唯一且严格递增")
    if any(
        not pd.api.types.is_numeric_dtype(minutes[name])
        or pd.api.types.is_bool_dtype(minutes[name])
        or pd.api.types.is_complex_dtype(minutes[name])
        for name in INPUTS
    ):
        raise ValueError("分钟量价必须是实数列，不能使用字符串或布尔值")
    values = minutes.loc[:, list(INPUTS)].astype(float)
    data = values.to_numpy()
    if np.isinf(data).any():
        raise ValueError("分钟量价不能含无穷数")
    if (data[:, :4] <= 0).any() or (data[:, 4:] < 0).any():
        raise ValueError("价格必须为正数，成交量和金额不能为负")
    opening, high, low, close, volume, quote = (values[name] for name in INPUTS)
    if ((high < opening) | (high < close) | (low > opening) | (low > close) | (high < low)).any():
        raise ValueError("分钟 OHLC 次序不合法")
    if (((volume == 0) & (quote > 0)) | ((volume > 0) & (quote == 0))).any():
        raise ValueError("分钟 base 和 quote 的零量状态不一致")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vwap = quote / volume.where(volume > 0)
    if ((vwap < low * (1 - 1e-8)) | (vwap > high * (1 + 1e-8))).any():
        raise ValueError("分钟 quote/base 超出 OHLC 的数值容差")

    # 缺失时钟不会压缩为连续观察序列；任一原始字段缺失也重启完整窗口。
    complete = np.isfinite(data).all(axis=1)
    ready, run = np.zeros(len(values), dtype=bool), 0
    for i, known in enumerate(complete):
        continuous = i > 0 and int(clocks[i]) - int(clocks[i - 1]) == MINUTE
        run = (run + 1 if continuous else 1) if known else 0
        ready[i] = run >= HISTORY
    return values, vwap, ready


def _finite(series):
    """派生的溢出或零分母保持未知，不能让 rolling 跳过后补成已知。"""
    return series.where(np.isfinite(series))


def _rolling(series, n):
    return _finite(series).rolling(n, min_periods=n)


def _correlation(left, right, n):
    """保留官方对两侧输入的绝对标准差阈值，包括其币价单位依赖。"""
    left, right = _finite(left), _finite(right)
    result = left.rolling(n, min_periods=n).corr(right)
    low_variance = np.isclose(_rolling(left, n).std(ddof=1), 0, atol=ZERO_STD_ATOL) | np.isclose(
        _rolling(right, n).std(ddof=1), 0, atol=ZERO_STD_ATOL
    )
    return result.mask(low_variance)


def _regression(close, n):
    """窗口时间序号的一元回归；中心化计算等价斜率、末点残差和 R²。"""
    x = np.arange(1, n + 1, dtype=float)
    centered_x = x - x.mean()
    x_square = centered_x @ centered_x
    rolling = _rolling(close, n)
    slope = rolling.apply(lambda y: centered_x @ (y - y.mean()) / x_square, raw=True)
    residual = close - (rolling.mean() + slope * centered_x[-1])
    square = slope.pow(2) * x_square / (rolling.var(ddof=0) * n)
    square = square.mask(np.isclose(rolling.std(ddof=1), 0, atol=ZERO_STD_ATOL))
    return slope, square, residual


def calculate(minutes):
    """返回相同索引、固定顺序的 158 列，未知保留 NaN。

    每次仅处理一个合约。索引必须是唯一、递增、对齐的整数毫秒完成边界；
    调用者负责仅传入已完成分钟并执行合约资格筛选。本函数不会推断当前时间。
    列为原始 open/high/low/close、base_volume、quote_volume，允许显式 NaN 缺口。
    漏掉分钟行与显式缺口均使后续窗口未知，不插值、不前填；不改变传入索引。
    所有列须有最近 61 个连续且六字段有限的完成分钟，派生算子还须完整有定义。
    """
    values, vwap, ready = _inputs(minutes)
    opening, high, low, close, volume, _quote = (values[name] for name in INPUTS)
    result = {}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        spread = high - low
        upper = high - np.maximum(opening, close)
        lower = np.minimum(opening, close) - low
        shift = 2 * close - high - low
        result.update(
            KMID=(close - opening) / opening,
            KLEN=spread / opening,
            KMID2=(close - opening) / (spread + EPSILON),
            KUP=upper / opening,
            KUP2=upper / (spread + EPSILON),
            KLOW=lower / opening,
            KLOW2=lower / (spread + EPSILON),
            KSFT=shift / opening,
            KSFT2=shift / (spread + EPSILON),
            OPEN0=opening / close,
            HIGH0=high / close,
            LOW0=low / close,
            VWAP0=vwap / close,
        )
        price_ratio = _finite(close / close.shift(1))
        # 分母为零时不加 epsilon：原 CORD 表达式没有这一修正项。
        volume_ratio = _finite(volume / volume.shift(1).where(volume.shift(1) > 0))
        log_volume_ratio = _finite(np.log(volume_ratio + 1))
        price_change, volume_change = close.diff(), volume.diff()
        positive_price, negative_price = price_change.clip(lower=0), (-price_change).clip(lower=0)
        positive_volume, negative_volume = (
            volume_change.clip(lower=0),
            (-volume_change).clip(lower=0),
        )
        weighted_change = _finite((price_ratio - 1).abs() * volume)
        for n in WINDOWS:
            price = _rolling(close, n)
            quantity = _rolling(volume, n)
            slope, square, residual = _regression(close, n)
            minimum, maximum = _rolling(low, n).min(), _rolling(high, n).max()
            high_index = _rolling(high, n).apply(lambda y: np.argmax(y) + 1, raw=True)
            low_index = _rolling(low, n).apply(lambda y: np.argmin(y) + 1, raw=True)
            gain, loss = _rolling(positive_price, n).sum(), _rolling(negative_price, n).sum()
            absolute_price = _rolling(price_change.abs(), n).sum() + EPSILON
            volume_gain = _rolling(positive_volume, n).sum()
            volume_loss = _rolling(negative_volume, n).sum()
            absolute_volume = _rolling(volume_change.abs(), n).sum() + EPSILON
            up = _rolling((close > close.shift(1)).astype(float), n).mean()
            down = _rolling((close < close.shift(1)).astype(float), n).mean()
            family = {
                "ROC": close.shift(n) / close,
                "MA": price.mean() / close,
                "STD": price.std(ddof=1) / close,
                "BETA": slope / close,
                "RSQR": square,
                "RESI": residual / close,
                "MAX": maximum / close,
                "MIN": minimum / close,
                "QTLU": price.quantile(0.8, interpolation="linear") / close,
                "QTLD": price.quantile(0.2, interpolation="linear") / close,
                "RANK": price.rank(method="average", pct=True),
                "RSV": (close - minimum) / (maximum - minimum + EPSILON),
                "IMAX": high_index / n,
                "IMIN": low_index / n,
                "IMXD": (high_index - low_index) / n,
                "CORR": _correlation(close, np.log(volume + 1), n),
                "CORD": _correlation(price_ratio, log_volume_ratio, n),
                "CNTP": up,
                "CNTN": down,
                "CNTD": up - down,
                "SUMP": gain / absolute_price,
                "SUMN": loss / absolute_price,
                "SUMD": (gain - loss) / absolute_price,
                "VMA": quantity.mean() / (volume + EPSILON),
                "VSTD": quantity.std(ddof=1) / (volume + EPSILON),
                "WVMA": _rolling(weighted_change, n).std(ddof=1)
                / (_rolling(weighted_change, n).mean() + EPSILON),
                "VSUMP": volume_gain / absolute_volume,
                "VSUMN": volume_loss / absolute_volume,
                "VSUMD": (volume_gain - volume_loss) / absolute_volume,
            }
            result.update({f"{name}{n}": series for name, series in family.items()})
    output = pd.DataFrame(result, index=minutes.index).reindex(columns=NAMES)
    output.loc[~ready, :] = np.nan
    return output.replace([np.inf, -np.inf], np.nan)
