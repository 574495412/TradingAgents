"""计算反转压力、吸收、恢复和趋势成熟度评分。"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCORE_COMPONENTS = (
    "short_adverse_pressure",
    "mid_adverse_pressure",
    "long_adverse_pressure",
    "relative_dislocation",
    "extreme_location",
    "breadth_pressure",
    "flow_pressure",
    "trend_maturity",
    "liquidity_expansion",
    "rejection_quality",
    "short_recovery",
    "mid_recovery",
    "relative_recovery",
    "breadth_recovery",
    "flow_recovery",
)

REQUIRED_COLUMNS = frozenset(
    {
        "signal_time",
        "open",
        "high",
        "low",
        "close",
        "ret3",
        "ret12",
        "ret20",
        "ret60",
        "rel_ret20",
        "donchian60_pos",
        "rsi14",
        "adx14",
        "atr14_pct",
        "atr14_ratio120",
        "range_expansion20",
        "quote_volume_ratio_med30",
        "taker_buy_ratio",
        "market_down_breadth20",
        "market_up_breadth20",
        "countertrend_data_valid",
    },
)


def build_reversal_lifecycle_scores(market_panel: pd.DataFrame) -> pd.DataFrame:
    """由单标的市场面板构建多空方向的反转生命周期评分。"""

    missing = sorted(REQUIRED_COLUMNS.difference(market_panel.columns))
    if missing:
        raise ValueError(
            "reversal lifecycle panel missing columns: " + ", ".join(missing),
        )
    market = market_panel.copy().sort_values("signal_time", kind="stable").reset_index(drop=True)
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    numeric = REQUIRED_COLUMNS.difference(
        {"signal_time", "countertrend_data_valid"},
    )
    for column in numeric:
        market[column] = pd.to_numeric(market[column], errors="coerce")

    atr = market["atr14_pct"].clip(lower=0.05)
    candle_range = (market["high"] - market["low"]).clip(lower=1e-12)
    upper_wick = (market["high"] - market[["open", "close"]].max(axis=1)).clip(lower=0.0)
    lower_wick = (market[["open", "close"]].min(axis=1) - market["low"]).clip(lower=0.0)
    close_position = ((market["close"] - market["low"]) / candle_range).clip(0.0, 1.0)
    market_balance = (
        market["market_up_breadth20"] - market["market_down_breadth20"]
    ).clip(-1.0, 1.0)
    liquidity_expansion = (
        0.34 * _ramp(market["atr14_ratio120"], 1.0, 2.2)
        + 0.33 * _ramp(market["range_expansion20"], 1.0, 3.5)
        + 0.33 * _log_ramp(market["quote_volume_ratio_med30"], 0.8, 8.0)
    ).clip(0.0, 1.0)
    trend_maturity = _ramp(market["adx14"], 18.0, 50.0)

    parts: list[pd.DataFrame] = []
    for action, direction in (("OPEN_LONG", 1.0), ("OPEN_SHORT", -1.0)):
        short_signed = direction * market["ret3"] / (atr * np.sqrt(3.0))
        mid_signed = direction * market["ret20"] / (atr * np.sqrt(20.0))
        long_signed = direction * market["ret60"] / (atr * np.sqrt(60.0))
        relative_signed = direction * market["rel_ret20"] / atr
        location_signed = direction * (market["donchian60_pos"] - 0.5) * 2.0
        breadth_signed = direction * market_balance
        flow_signed = direction * (market["taker_buy_ratio"] - 0.5) * 2.0
        directional_wick = lower_wick if direction > 0.0 else upper_wick
        directional_close = close_position if direction > 0.0 else 1.0 - close_position
        rejection_quality = (
            0.55 * _ramp(directional_wick / candle_range, 0.20, 0.65)
            + 0.45 * _ramp(directional_close, 0.45, 0.85)
        ).clip(0.0, 1.0)

        part = pd.DataFrame(
            {
                "signal_time": market["signal_time"],
                "symbol": market.get("symbol", ""),
                "action": action,
                "market_data_valid": market["countertrend_data_valid"].astype(bool),
                "short_adverse_pressure": _ramp(-short_signed, 0.05, 1.35),
                "mid_adverse_pressure": _ramp(-mid_signed, 0.05, 1.10),
                "long_adverse_pressure": _ramp(-long_signed, 0.05, 0.90),
                "relative_dislocation": _ramp(-relative_signed, 0.15, 1.60),
                "extreme_location": _ramp(-location_signed, 0.25, 0.90),
                "breadth_pressure": _ramp(-breadth_signed, 0.20, 0.90),
                "flow_pressure": _ramp(-flow_signed, 0.05, 0.45),
                "trend_maturity": trend_maturity,
                "liquidity_expansion": liquidity_expansion,
                "rejection_quality": rejection_quality,
                "short_recovery": _ramp(short_signed, 0.05, 1.10),
                "mid_recovery": _ramp(mid_signed, 0.00, 0.85),
                "relative_recovery": _ramp(relative_signed, 0.00, 1.20),
                "breadth_recovery": _ramp(breadth_signed, 0.05, 0.80),
                "flow_recovery": _ramp(flow_signed, 0.00, 0.40),
            },
        )
        pressure = (
            0.18 * part["short_adverse_pressure"]
            + 0.22 * part["mid_adverse_pressure"]
            + 0.16 * part["long_adverse_pressure"]
            + 0.13 * part["relative_dislocation"]
            + 0.13 * part["extreme_location"]
            + 0.10 * part["breadth_pressure"]
            + 0.08 * part["flow_pressure"]
        ).clip(0.0, 1.0)
        part["exhaustion_score"] = (
            0.62 * pressure
            + 0.16 * part["trend_maturity"]
            + 0.12 * part["liquidity_expansion"]
            + 0.10 * part["rejection_quality"]
        ).clip(0.0, 1.0)
        # 计算 part["pullback_score"]，仅使用当前及历史数据。
        # 计算 part["pullback_score"]，仅使用当前及历史数据。
        part["pullback_score"] = (
            0.28 * _ramp(long_signed, 0.05, 0.90)
            + 0.25 * part["short_adverse_pressure"]
            + 0.14 * part["relative_dislocation"]
            + 0.13 * part["trend_maturity"]
            + 0.10 * part["liquidity_expansion"]
            + 0.10 * part["rejection_quality"]
        ).clip(0.0, 1.0)
        part["liquidation_score"] = (
            0.40 * pressure
            + 0.32 * part["liquidity_expansion"]
            + 0.18 * part["rejection_quality"]
            + 0.10 * part["flow_pressure"]
        ).clip(0.0, 1.0)
        part["entry_score"] = part[
            ["exhaustion_score", "pullback_score", "liquidation_score"]
        ].max(axis=1)
        part["entry_mechanism"] = np.select(
            [
                part["liquidation_score"].ge(part["exhaustion_score"])
                & part["liquidation_score"].ge(part["pullback_score"]),
                part["pullback_score"].ge(part["exhaustion_score"]),
            ],
            ["liquidation_absorption", "pullback_recovery"],
            default="exhaustion_reversal",
        )
        part["recovery_score"] = (
            0.30 * part["short_recovery"]
            + 0.22 * part["mid_recovery"]
            + 0.14 * part["relative_recovery"]
            + 0.12 * part["breadth_recovery"]
            + 0.10 * part["flow_recovery"]
            + 0.07 * part["liquidity_expansion"]
            + 0.05 * part["rejection_quality"]
        ).clip(0.0, 1.0)
        score_columns = [
            *SCORE_COMPONENTS,
            "exhaustion_score",
            "pullback_score",
            "liquidation_score",
            "entry_score",
            "recovery_score",
        ]
        part.loc[~part["market_data_valid"], score_columns] = 0.0
        parts.append(part)
    return pd.concat(parts, ignore_index=True, sort=False).sort_values(
        ["signal_time", "action"],
        kind="stable",
    ).reset_index(drop=True)


def _ramp(values: pd.Series, start: float, full: float) -> pd.Series:
    """对序列计算限制于 [0, 1] 的线性斜坡。"""

    if full <= start:
        message = "ramp full must exceed start"
        raise ValueError(message)
    return ((pd.to_numeric(values, errors="coerce") - start) / (full - start)).clip(
        0.0,
        1.0,
    ).fillna(0.0)


def _log_ramp(values: pd.Series, start: float, full: float) -> pd.Series:
    """对正值序列计算对数尺度的 [0, 1] 斜坡。"""

    safe = pd.to_numeric(values, errors="coerce").clip(lower=0.0)
    return _ramp(np.log1p(safe), np.log1p(start), np.log1p(full))


__all__ = ["REQUIRED_COLUMNS", "SCORE_COMPONENTS", "build_reversal_lifecycle_scores"]
