"""构建统一反转生命周期所需的单标的和全市场上下文面板。"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from factors.crypto.code.weijie.large_cap.countertrend.features import (
    COUNTERTREND_FEATURE_COLUMNS,
)
from factors.crypto.code.weijie.market_features import (
    build_replica_feature_frame,
    build_symbol_feature_rows,
)

TARGET_SYMBOL = "ETHUSDT"
CONTEXT_SYMBOLS = (
    "ETHUSDT",
    "BTCUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LTCUSDT",
    "BCHUSDT",
)
MARKET_CONTEXT_COLUMNS = (
    "market_ret3_mean",
    "market_ret12_mean",
    "market_ret20_dispersion",
    "market_down_breadth3",
    "market_up_breadth3",
    "market_quote_expansion_breadth",
    "market_atr_expansion_breadth",
    "rel_ret3",
    "rel_ret12",
)
UNIFIED_MARKET_COLUMNS = tuple(
    dict.fromkeys((*COUNTERTREND_FEATURE_COLUMNS, *MARKET_CONTEXT_COLUMNS)),
)


def build_unified_regime_market_panel(
    raw_frames: Mapping[str, pd.DataFrame],
    *,
    target_symbol: str = TARGET_SYMBOL,
    required_context_symbols: tuple[str, ...] = CONTEXT_SYMBOLS,
) -> pd.DataFrame:
    """合并目标标的特征和同一已收盘时点的市场状态。"""

    normalized = {
        _clean_symbol(symbol): frame
        for symbol, frame in raw_frames.items()
        if frame is not None and not frame.empty
    }
    required = tuple(_clean_symbol(symbol) for symbol in required_context_symbols)
    missing_symbols = sorted(set(required).difference(normalized))
    if missing_symbols:
        raise ValueError(
            "ETH unified market panel missing context symbols: "
            + ", ".join(missing_symbols),
        )
    target = _clean_symbol(target_symbol)
    feature_frames: dict[str, pd.DataFrame] = {}
    for symbol in required:
        features = build_replica_feature_frame(normalized[symbol])
        features["symbol"] = symbol
        features["bar_time"] = features.index
        feature_frames[symbol] = features.reset_index(drop=True)
    features = build_symbol_feature_rows(feature_frames)
    features = _add_market_context(features)
    target_rows = features.loc[
        features["symbol"].astype(str).eq(target)
    ].copy()
    if target_rows.empty:
        raise ValueError(f"ETH unified market panel missing target: {target}")
    missing_columns = sorted(
        set(UNIFIED_MARKET_COLUMNS).difference(target_rows.columns),
    )
    if missing_columns:
        raise ValueError(
            "ETH unified market panel missing factor columns: "
            + ", ".join(missing_columns),
        )

    source_time = pd.DatetimeIndex(
        pd.to_datetime(target_rows["bar_time"], errors="coerce"),
    )
    if source_time.isna().any():
        raise ValueError("ETH unified market panel contains invalid bar times")
    if source_time.tz is None:
        source_time = source_time.tz_localize("Asia/Shanghai")
    else:
        source_time = source_time.tz_convert("Asia/Shanghai")

    panel = target_rows.loc[:, list(UNIFIED_MARKET_COLUMNS)].copy()
    panel["source_bar_time"] = source_time.tz_convert("UTC")
    panel["signal_time"] = panel["source_bar_time"] + pd.Timedelta(minutes=5)
    panel["symbol"] = target
    panel["unified_market_data_valid"] = panel.loc[
        :,
        list(UNIFIED_MARKET_COLUMNS),
    ].notna().all(axis=1)
    panel["eligibility_data_valid"] = panel["unified_market_data_valid"]
    panel["countertrend_data_valid"] = panel["unified_market_data_valid"]
    return panel.sort_values("signal_time", kind="stable").reset_index(drop=True)


def _add_market_context(rows: pd.DataFrame) -> pd.DataFrame:
    """补充方向无关的广度、波动率和趋势状态字段。"""

    output = rows.copy().sort_values(["bar_time", "symbol"], kind="stable")
    ret3 = pd.to_numeric(output["ret3"], errors="coerce")
    ret12 = pd.to_numeric(output["ret12"], errors="coerce")
    ret20 = pd.to_numeric(output["ret20"], errors="coerce")
    output["market_ret3_mean"] = ret3.groupby(
        output["bar_time"],
    ).transform("mean")
    output["market_ret12_mean"] = ret12.groupby(
        output["bar_time"],
    ).transform("mean")
    output["market_ret20_dispersion"] = ret20.groupby(
        output["bar_time"],
    ).transform("std")
    output["market_down_breadth3"] = ret3.lt(0.0).groupby(
        output["bar_time"],
    ).transform("mean")
    output["market_up_breadth3"] = ret3.gt(0.0).groupby(
        output["bar_time"],
    ).transform("mean")
    quote = pd.to_numeric(
        output["quote_volume_ratio_med30"],
        errors="coerce",
    )
    atr = pd.to_numeric(output["atr14_ratio120"], errors="coerce")
    output["market_quote_expansion_breadth"] = quote.gt(1.25).groupby(
        output["bar_time"],
    ).transform("mean")
    output["market_atr_expansion_breadth"] = atr.gt(1.05).groupby(
        output["bar_time"],
    ).transform("mean")
    output["rel_ret3"] = ret3 - output["market_ret3_mean"]
    output["rel_ret12"] = ret12 - output["market_ret12_mean"]
    return output


def _clean_symbol(value: object) -> str:
    return str(value or "").upper().replace("/", "").replace("-", "")


__all__ = [
    "CONTEXT_SYMBOLS",
    "MARKET_CONTEXT_COLUMNS",
    "TARGET_SYMBOL",
    "UNIFIED_MARKET_COLUMNS",
    "build_unified_regime_market_panel",
]
