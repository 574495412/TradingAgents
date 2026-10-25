"""为反转候选增加相对市场、横截面和历史状态上下文。"""

from __future__ import annotations

import pandas as pd

CONTEXT_COLUMNS = (
    "oi_value_log_change_60m",
    "top_position_crowding_gap_change_60m",
    "taker_long_short_log_ratio_change_15m",
    "premium_change_15m",
    "perpetual_minus_spot_flow",
    "perpetual_spot_basis_change_bps",
)

CONTEXT_COMPONENTS = (
    "oi_deleveraging",
    "oi_expansion",
    "crowding_dislocation",
    "directional_taker_follow",
    "directional_premium_unwind",
    "spot_lead_confirmation",
    "basis_dislocation",
    "trend_continuation_score",
    "contextual_reversal_score",
    "spot_absorption_score",
    "contextual_recovery_score",
)

MINIMUM_VALID_BASELINE_BARS = 2


def add_contextual_lifecycle_scores(  # noqa: PLR0915
    market_panel: pd.DataFrame,
    base_scores: pd.DataFrame,
    *,
    baseline_bars: int = 576,
    minimum_baseline_bars: int = 144,
) -> pd.DataFrame:
    """用当时可见的市场上下文修正生命周期候选评分。"""

    missing = sorted(
        {
            "signal_time",
            "contextual_data_valid",
            *CONTEXT_COLUMNS,
        }.difference(market_panel.columns),
    )
    if missing:
        raise ValueError("contextual lifecycle panel missing columns: " + ", ".join(missing))
    if (
        minimum_baseline_bars < MINIMUM_VALID_BASELINE_BARS
        or baseline_bars < minimum_baseline_bars
    ):
        message = "contextual lifecycle baseline window is invalid"
        raise ValueError(message)

    market = market_panel.copy().sort_values("signal_time", kind="stable").reset_index(drop=True)
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    standardized: dict[str, pd.Series] = {}
    for column in CONTEXT_COLUMNS:
        values = pd.to_numeric(market[column], errors="coerce")
        standardized[column] = _prior_zscore(
            values,
            baseline_bars=baseline_bars,
            minimum_baseline_bars=minimum_baseline_bars,
        )

    scores = base_scores.copy().sort_values(
        ["signal_time", "action"],
        kind="stable",
    ).reset_index(drop=True)
    scores["signal_time"] = pd.to_datetime(scores["signal_time"], utc=True)
    if len(scores) != 2 * len(market):
        message = "contextual lifecycle scores must contain two directions per bar"
        raise ValueError(message)

    context = pd.DataFrame({"signal_time": market["signal_time"]})
    for column, values in standardized.items():
        context[f"_{column}_z"] = values
    context["_standardization_valid"] = pd.concat(
        standardized.values(),
        axis=1,
    ).notna().all(axis=1)
    scores = scores.merge(context, on="signal_time", how="left", validate="many_to_one")
    valid_by_time = market.set_index("signal_time")["contextual_data_valid"].astype(bool)
    scores["market_data_valid"] &= scores["signal_time"].map(valid_by_time).fillna(
        value=False,
    )
    scores["market_data_valid"] &= scores["_standardization_valid"].fillna(
        value=False,
    )

    direction = scores["action"].map({"OPEN_LONG": 1.0, "OPEN_SHORT": -1.0})
    if direction.isna().any():
        message = "contextual lifecycle scores contain unsupported actions"
        raise ValueError(message)
    oi_z = scores["_oi_value_log_change_60m_z"]
    crowding_z = scores["_top_position_crowding_gap_change_60m_z"]
    taker_z = scores["_taker_long_short_log_ratio_change_15m_z"]
    premium_z = scores["_premium_change_15m_z"]
    perp_minus_spot_z = scores["_perpetual_minus_spot_flow_z"]
    basis_z = scores["_perpetual_spot_basis_change_bps_z"]

    scores["oi_deleveraging"] = _ramp(-oi_z, 0.20, 2.50)
    scores["oi_expansion"] = _ramp(oi_z, 0.20, 2.50)
    scores["crowding_dislocation"] = _ramp(crowding_z.abs(), 0.50, 3.00)
    scores["directional_taker_follow"] = _ramp(direction * taker_z, 0.00, 2.50)
    scores["directional_premium_unwind"] = _ramp(-direction * premium_z, 0.00, 2.50)
    # perpetual_minus_spot_flow < 0 表示现货相对领先，
    # > 0 表示永续合约资金流相对领先。
    scores["spot_lead_confirmation"] = _ramp(
        -direction * perp_minus_spot_z,
        0.00,
        2.50,
    )
    scores["basis_dislocation"] = _ramp(basis_z.abs(), 0.50, 3.00)

    trend_core = (
        0.18 * scores["short_recovery"]
        + 0.23 * scores["mid_recovery"]
        + 0.13 * scores["relative_recovery"]
        + 0.10 * scores["breadth_recovery"]
        + 0.08 * scores["flow_recovery"]
        + 0.18 * scores["trend_maturity"]
        + 0.10 * scores["liquidity_expansion"]
    ).clip(0.0, 1.0)
    base_reversal = scores[["exhaustion_score", "liquidation_score"]].max(axis=1)
    scores["trend_continuation_score"] = (
        0.72 * trend_core
        + 0.10 * scores["oi_expansion"]
        + 0.10 * scores["directional_taker_follow"]
        + 0.08 * scores["spot_lead_confirmation"]
    ).clip(0.0, 1.0)
    scores["contextual_reversal_score"] = (
        0.58 * base_reversal
        + 0.18 * scores["oi_deleveraging"]
        + 0.14 * scores["directional_premium_unwind"]
        + 0.10 * scores["crowding_dislocation"]
    ).clip(0.0, 1.0)
    scores["spot_absorption_score"] = (
        0.60 * base_reversal
        + 0.22 * scores["spot_lead_confirmation"]
        + 0.18 * scores["basis_dislocation"]
    ).clip(0.0, 1.0)
    scores["contextual_recovery_score"] = (
        0.45 * scores["directional_taker_follow"]
        + 0.35 * _ramp(direction * premium_z, 0.00, 2.50)
        + 0.20 * scores["spot_lead_confirmation"]
    ).clip(0.0, 1.0)
    scores["entry_score"] = scores[
        [
            "pullback_score",
            "trend_continuation_score",
            "contextual_reversal_score",
            "spot_absorption_score",
        ]
    ].max(axis=1)
    scores["entry_mechanism"] = scores[
        [
            "pullback_score",
            "trend_continuation_score",
            "contextual_reversal_score",
            "spot_absorption_score",
        ]
    ].idxmax(axis=1).str.removesuffix("_score")
    scores["recovery_score"] = (
        0.72 * scores["recovery_score"]
        + 0.28 * scores["contextual_recovery_score"]
    ).clip(0.0, 1.0)

    score_columns = [
        *CONTEXT_COMPONENTS,
        "entry_score",
        "recovery_score",
    ]
    scores.loc[~scores["market_data_valid"], score_columns] = 0.0
    return scores.drop(
        columns=[column for column in scores.columns if column.startswith("_")],
    )


def _prior_zscore(
    values: pd.Series,
    *,
    baseline_bars: int,
    minimum_baseline_bars: int,
) -> pd.Series:
    """仅使用当前值之前的历史窗口计算因果标准分。"""

    prior = values.shift(1)
    mean = prior.rolling(baseline_bars, min_periods=minimum_baseline_bars).mean()
    std = prior.rolling(baseline_bars, min_periods=minimum_baseline_bars).std()
    return (values - mean) / std.mask(std.eq(0.0))


def _ramp(values: pd.Series, start: float, full: float) -> pd.Series:
    """在给定起点和满分点之间计算 [0, 1] 斜坡。"""

    if full <= start:
        message = "contextual lifecycle ramp full must exceed start"
        raise ValueError(message)
    return ((values - start) / (full - start)).clip(0.0, 1.0).fillna(0.0)


__all__ = [
    "CONTEXT_COLUMNS",
    "CONTEXT_COMPONENTS",
    "add_contextual_lifecycle_scores",
]
