"""计算大盘标的反趋势压力、衰竭和恢复质量评分。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .policy import CountertrendFactorPolicy

COUNTERTREND_MECHANISMS = (
    "liquidity_shock_exhaustion",
    "relative_capitulation",
    "multi_timeframe_overshoot",
)

MECHANISM_COMPONENTS = {
    "liquidity_shock_exhaustion": (
        "shock_opposing_short_trend",
        "shock_opposing_mid_trend",
        "shock_extreme_location",
        "shock_mature_trend",
        "shock_atr_expansion",
        "shock_range_expansion",
        "shock_quote_expansion",
    ),
    "relative_capitulation": (
        "relative_short_dislocation",
        "relative_mid_dislocation",
        "relative_short_rank_extreme",
        "relative_mid_rank_extreme",
        "relative_market_breadth",
        "relative_range_expansion",
        "relative_quote_expansion",
    ),
    "multi_timeframe_overshoot": (
        "overshoot_5m_trend",
        "overshoot_15m_trend",
        "overshoot_1h_trend",
        "overshoot_rsi_extreme",
        "overshoot_range_extreme",
        "overshoot_mature_trend",
        "overshoot_quote_expansion",
    ),
}

REQUIRED_COLUMNS = frozenset(
    {
        "signal_time",
        "source_bar_time",
        "open",
        "high",
        "low",
        "close",
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
        "tf15_ret16",
        "tf1h_ret12",
        "countertrend_data_valid",
    },
)


def build_countertrend_scores(
    market_panel: pd.DataFrame,
    *,
    policy: CountertrendFactorPolicy | None = None,
) -> pd.DataFrame:
    cfg = policy or CountertrendFactorPolicy()
    unknown = sorted(set(cfg.enabled_mechanisms).difference(COUNTERTREND_MECHANISMS))
    if unknown:
        raise ValueError(f"unknown enabled countertrend mechanisms: {', '.join(unknown)}")
    known_components = {
        component for values in MECHANISM_COMPONENTS.values() for component in values
    }
    unknown_components = sorted(set(cfg.disabled_components).difference(known_components))
    if unknown_components:
        raise ValueError(
            "unknown disabled countertrend components: " + ", ".join(unknown_components),
        )
    missing = sorted(REQUIRED_COLUMNS.difference(market_panel.columns))
    if missing:
        raise ValueError(f"countertrend panel missing columns: {', '.join(missing)}")
    frame = market_panel.copy().sort_values("signal_time", kind="stable").reset_index(drop=True)
    numeric = REQUIRED_COLUMNS.difference(
        {"signal_time", "source_bar_time", "countertrend_data_valid"},
    )
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    atr = frame["atr14_pct"].clip(lower=1e-6)
    short_scale = atr * np.sqrt(20.0)
    mid_scale = atr * np.sqrt(60.0)
    tf15_scale = atr * np.sqrt(48.0)
    tf1h_scale = atr * np.sqrt(144.0)
    common = {
        "mature_trend": _ramp(frame["adx14"], cfg.adx_start, cfg.adx_full),
        "atr_expansion": _ramp(
            frame["atr14_ratio120"],
            cfg.atr_expansion_start,
            cfg.atr_expansion_full,
        ),
        "range_expansion": _ramp(
            frame["range_expansion20"],
            cfg.range_expansion_start,
            cfg.range_expansion_full,
        ),
        "quote_expansion": _ramp(
            frame["quote_volume_ratio_med30"],
            cfg.quote_expansion_start,
            cfg.quote_expansion_full,
        ),
    }
    records: list[pd.DataFrame] = []
    for action, direction in (("OPEN_LONG", 1.0), ("OPEN_SHORT", -1.0)):
        range_extreme = 0.5 - direction * (frame["donchian60_pos"] - 0.5)
        short_rank_extreme = 0.5 - direction * (frame["ret20_rank_pct"] - 0.5)
        mid_rank_extreme = 0.5 - direction * (frame["ret60_rank_pct"] - 0.5)
        breadth = (
            frame["market_down_breadth20"]
            if direction > 0.0
            else frame["market_up_breadth20"]
        )
        part = pd.DataFrame(
            {
                "signal_time": frame["signal_time"],
                "source_bar_time": frame["source_bar_time"],
                "symbol": frame.get("symbol", ""),
                "action": action,
                "market_data_valid": frame["countertrend_data_valid"].astype(bool),
                "shock_opposing_short_trend": _ramp(
                    -direction * frame["ret20"] / short_scale,
                    cfg.short_impulse_start,
                    cfg.short_impulse_full,
                ),
                "shock_opposing_mid_trend": _ramp(
                    -direction * frame["ret60"] / mid_scale,
                    cfg.mid_impulse_start,
                    cfg.mid_impulse_full,
                ),
                "shock_extreme_location": _ramp(
                    range_extreme,
                    cfg.extreme_position_start,
                    cfg.extreme_position_full,
                ),
                "shock_mature_trend": common["mature_trend"],
                "shock_atr_expansion": common["atr_expansion"],
                "shock_range_expansion": common["range_expansion"],
                "shock_quote_expansion": common["quote_expansion"],
                "relative_short_dislocation": _ramp(
                    -direction * frame["rel_ret20"] / atr,
                    cfg.relative_impulse_start,
                    cfg.relative_impulse_full,
                ),
                "relative_mid_dislocation": _ramp(
                    -direction * frame["rel_ret60"] / atr,
                    cfg.relative_impulse_start,
                    cfg.relative_impulse_full,
                ),
                "relative_short_rank_extreme": _ramp(
                    short_rank_extreme,
                    cfg.extreme_position_start,
                    cfg.extreme_position_full,
                ),
                "relative_mid_rank_extreme": _ramp(
                    mid_rank_extreme,
                    cfg.extreme_position_start,
                    cfg.extreme_position_full,
                ),
                "relative_market_breadth": _ramp(
                    breadth,
                    cfg.breadth_start,
                    cfg.breadth_full,
                ),
                "relative_range_expansion": common["range_expansion"],
                "relative_quote_expansion": common["quote_expansion"],
                "overshoot_5m_trend": _ramp(
                    -direction * frame["ret20"] / short_scale,
                    cfg.short_impulse_start,
                    cfg.short_impulse_full,
                ),
                "overshoot_15m_trend": _ramp(
                    -direction * frame["tf15_ret16"] / tf15_scale,
                    cfg.short_impulse_start,
                    cfg.short_impulse_full,
                ),
                "overshoot_1h_trend": _ramp(
                    -direction * frame["tf1h_ret12"] / tf1h_scale,
                    cfg.mid_impulse_start,
                    cfg.mid_impulse_full,
                ),
                "overshoot_rsi_extreme": _ramp(
                    -direction * (frame["rsi14"] - 50.0),
                    cfg.rsi_extreme_start,
                    cfg.rsi_extreme_full,
                ),
                "overshoot_range_extreme": _ramp(
                    range_extreme,
                    cfg.extreme_position_start,
                    cfg.extreme_position_full,
                ),
                "overshoot_mature_trend": common["mature_trend"],
                "overshoot_quote_expansion": common["quote_expansion"],
            },
        )
        for mechanism, components in MECHANISM_COMPONENTS.items():
            enabled = tuple(
                component for component in components if component not in cfg.disabled_components
            )
            part[f"{mechanism}_score"] = (
                part.loc[:, list(enabled)].mean(axis=1) if enabled else 0.0
            )
            if mechanism not in cfg.enabled_mechanisms:
                part[f"{mechanism}_score"] = 0.0
        score_columns = [f"{name}_score" for name in COUNTERTREND_MECHANISMS]
        part.loc[~part["market_data_valid"], score_columns] = 0.0
        records.append(part)
    return pd.concat(records, ignore_index=True, sort=False).sort_values(
        ["signal_time", "action"],
        kind="stable",
    ).reset_index(drop=True)


def _ramp(values: pd.Series, start: float, full: float) -> pd.Series:
    if not full > start:
        raise ValueError("countertrend ramp full value must exceed start")
    return ((values - start) / (full - start)).clip(0.0, 1.0).fillna(0.0)


__all__ = [
    "COUNTERTREND_MECHANISMS",
    "MECHANISM_COMPONENTS",
    "build_countertrend_scores",
]
