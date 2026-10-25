"""Production conditional-order factors derived from closed-bar setups.

The setup signal is produced from a closed bar.  A fill can only occur on a
later bar whose traded range crosses the deterministic stop or limit price.
Leader operations are never read by this module. Live and backtest import the
same entry, lifecycle, fill, sizing, and exit functions from this file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from factors.crypto.code.weijie.market_regime import (
    decide_regime_target,
    market_regime_open_allowed,
)

_LIQUIDITY_CLIMAX_REVERSAL_CONFIDENCE_FLOOR = 0.68
_LIQUIDITY_CLIMAX_REVERSAL_VOLUME_RATIO_FLOOR = 7.0
_LIQUIDITY_CLIMAX_REVERSAL_RSI_FLOOR = 78.0
_LIQUIDITY_CLIMAX_REVERSAL_TAKER_BUY_CEILING = 0.50


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column aligned to ``frame`` and fail closed if absent."""

    values = (
        frame[column]
        if column in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    return pd.to_numeric(values, errors="coerce")


def _market_regime_allows_action(row: pd.Series, action: str) -> bool:
    """判断当前市场状态是否允许候选方向执行。"""

    candidate = row.copy()
    candidate["action"] = str(action).upper()
    return market_regime_open_allowed(candidate)


def _liquidity_climax_reversal_short_allowed(row: pd.Series) -> bool:
    """校验流动性高潮后的反转做空是否满足独立门槛。"""

    target = decide_regime_target(row)
    return bool(
        target.action == "OPEN_SHORT"
        and target.mechanism == "exhaustion_reversal"
        and target.confidence >= _LIQUIDITY_CLIMAX_REVERSAL_CONFIDENCE_FLOOR
        and _number(row.get("quote_volume_ratio_med30"))
        >= _LIQUIDITY_CLIMAX_REVERSAL_VOLUME_RATIO_FLOOR
        and _number(row.get("rsi14")) >= _LIQUIDITY_CLIMAX_REVERSAL_RSI_FLOOR
        and _number(row.get("taker_buy_ratio"))
        <= _LIQUIDITY_CLIMAX_REVERSAL_TAKER_BUY_CEILING,
    )


@dataclass(frozen=True)
class ConditionalOrderConfig:
    valid_minutes: float = 15.0
    stop_buffer_bps: float = 1.0
    pullback_atr_multiplier: float = 0.25
    momentum_atr_multiplier: float | None = None
    emit_pending_orders: bool = False
    order_styles: tuple[str, ...] = ("momentum_stop", "reversion_limit")


@dataclass(frozen=True)
class ActiveEthLongAddonConfig:
    min_active_age_minutes: float = 1200.0
    setup_cooldown_minutes: float = 120.0
    valid_minutes: float = 15.0
    pullback_atr_multiplier: float = 0.10
    breakout_min_active_age_minutes: float = 120.0
    breakout_pullback_atr_multiplier: float = 1.0


@dataclass(frozen=True)
class ActiveEthLongScaleoutConfig:
    profit_score_floor: float = 3.0
    profit_volume_ratio_floor: float = 2.0
    midnight_min_active_age_minutes: float = 1440.0
    spike_volume_ratio_floor: float = 5.0


@dataclass(frozen=True)
class EthEarlyCorePullbackConfig:
    ret20_floor: float = 0.75
    ret20_ceiling: float = 1.25
    ret60_floor: float = 1.80
    ret288_floor: float = 0.0
    ret288_ceiling: float = 1.50
    taker_buy_floor: float = 0.55
    volume_ratio_floor: float = 2.0
    rsi14_ceiling: float = 68.0
    adx14_floor: float = 30.0
    adx14_ceiling: float = 45.0
    valid_minutes: float = 30.0
    pullback_atr_multiplier: float = 1.0


@dataclass(frozen=True)
class EthExtremeFlushReversalConfig:
    ret20_ceiling: float = -5.0
    ret60_ceiling: float = -6.0
    rsi14_ceiling: float = 12.0
    volume_ratio_floor: float = 12.0
    taker_buy_ceiling: float = 0.42
    valid_minutes: float = 30.0
    confirmation_atr_multiplier: float = 1.10


@dataclass(frozen=True)
class EthModerateFlushPullbackConfig:
    ret20_floor: float = -2.0
    ret20_ceiling: float = -1.70
    ret60_floor: float = -2.70
    ret60_ceiling: float = -2.30
    rsi14_floor: float = 12.0
    rsi14_ceiling: float = 20.0
    adx14_floor: float = 50.0
    adx14_ceiling: float = 60.0
    volume_ratio_floor: float = 8.0
    taker_buy_ceiling: float = 0.40
    valid_minutes: float = 15.0
    pullback_atr_multiplier: float = 0.25


@dataclass(frozen=True)
class EthOverboughtLimitExitConfig:
    ret20_floor: float = 1.20
    ret20_ceiling: float = 1.50
    ret288_floor: float = 0.0
    ret288_ceiling: float = 1.50
    rsi14_floor: float = 80.0
    adx14_floor: float = 35.0
    adx14_ceiling: float = 40.0
    volume_ratio_floor: float = 3.0
    volume_ratio_ceiling: float = 5.0
    valid_minutes: float = 15.0
    limit_atr_multiplier: float = 1.0


@dataclass(frozen=True)
class EthBlowoffRetraceShortConfig:
    ret20_floor: float = 0.90
    ret20_ceiling: float = 1.30
    ret60_floor: float = 2.50
    ret60_ceiling: float = 3.30
    ret288_floor: float = 5.0
    rsi14_floor: float = 68.0
    rsi14_ceiling: float = 75.0
    adx14_floor: float = 45.0
    adx14_ceiling: float = 55.0
    volume_ratio_floor: float = 3.0
    taker_buy_floor: float = 0.58
    donchian_floor: float = 0.90
    valid_minutes: float = 15.0
    retrace_atr_multiplier: float = 0.25


@dataclass(frozen=True)
class EthReliefRecoveryExitConfig:
    ret20_floor: float = 0.20
    ret20_ceiling: float = 0.50
    ret60_ceiling: float = 0.0
    ret288_ceiling: float = -2.0
    setup_taker_buy_ceiling: float = 0.45
    setup_volume_ratio_floor: float = 1.0
    setup_volume_ratio_ceiling: float = 1.5
    setup_rsi14_floor: float = 50.0
    setup_rsi14_ceiling: float = 60.0
    setup_adx14_floor: float = 20.0
    setup_adx14_ceiling: float = 28.0
    confirmation_taker_buy_floor: float = 0.53


@dataclass(frozen=True)
class EthCapitulationProfitLadderConfig:
    tiers: tuple[tuple[float, int, float], ...] = (
        (0.45, 1, 20.0),
        (0.60, 2, 0.0),
        (1.10, 1, 0.0),
    )
    final_runner_score_floor: float = 3.0
    final_runner_volume_ratio_floor: float = 3.0
    final_runner_max_hold_minutes: float = 1440.0


@dataclass(frozen=True)
class EthInventoryOverlayConfig:
    dual_entry_score_floor: float = 0.80
    dual_entry_volume_ratio_floor: float = 8.0
    dual_entry_ret288_ceiling: float = -1.50
    dual_entry_donchian_ceiling: float = 0.25
    dual_entry_valid_minutes: float = 15.0
    dual_entry_pullback_atr_multiplier: float = 1.60
    scaleout_min_age_minutes: float = 10.0
    scaleout_return_floor_pct: float = 0.15
    scaleout_taker_buy_floor: float = 0.55
    scaleout_volume_ratio_floor: float = 3.0
    recovery_scaleout_min_age_minutes: float = 5.0
    recovery_scaleout_return_floor_pct: float = 0.10
    recovery_scaleout_taker_buy_floor: float = 0.52
    recovery_scaleout_volume_ratio_floor: float = 1.0
    short_add_min_age_minutes: float = 120.0
    short_add_ret20_floor: float = 1.25
    short_add_ret60_floor: float = 0.75
    short_add_rsi_floor: float = 70.0
    short_add_volume_ratio_floor: float = 5.0
    short_scaleout_score_floor: float = 5.20
    short_scaleout_min_lots: int = 3
    stall_short_min_full_exit_age_minutes: float = 3000.0
    stall_short_flush_scaleout_min_age_minutes: float = 2160.0
    stall_short_flush_ret20_ceiling: float = -1.50
    stall_short_flush_volume_ratio_floor: float = 10.0
    stall_short_flush_rsi_ceiling: float = 22.0
    stall_short_final_profit_floor_pct: float = 0.50
    stall_short_final_volume_ratio_floor: float = 1.0
    stall_short_final_volume_ratio_ceiling: float = 5.0
    stall_short_final_rsi_ceiling: float = 35.0
    stall_short_final_taker_buy_floor: float = 0.55


def _is_cross_horizon_stall_short_setup(reason: object) -> bool:
    """Identify the autonomous multi-day stall-short setup from source metrics."""

    text = str(reason or "")
    if "eth_anchor_exhaustion_short" not in text:
        return False
    return bool(
        _signal_reason_metric(text, "ret60") >= 4.0
        and _signal_reason_metric(text, "ret288") >= 4.0
        and _signal_reason_metric(text, "qvr") <= 1.0
        and 62.0 <= _signal_reason_metric(text, "rsi") <= 72.0
        and 45.0 <= _signal_reason_metric(text, "adx") <= 60.0,
    )


def _apply_cross_horizon_stall_exit_hysteresis(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    config: EthInventoryOverlayConfig,
) -> pd.DataFrame:
    """Keep a stall-short campaign alive until its own causal exit state matures."""

    if opens.empty or closes.empty:
        return closes.copy()
    stall_starts = pd.to_datetime(
        opens.loc[
            opens.get("action", pd.Series("", index=opens.index)).astype(str).eq("OPEN_SHORT")
            & opens.get("reason", pd.Series("", index=opens.index)).map(
                _is_cross_horizon_stall_short_setup,
            ),
            "signal_time",
        ],
        errors="coerce",
    ).dropna()
    if stall_starts.empty:
        return closes.copy()

    out = closes.copy()
    close_time = pd.to_datetime(out.get("signal_time"), errors="coerce")
    action = out.get("action", pd.Series("", index=out.index)).astype(str)
    family = out.get("reason", pd.Series("", index=out.index)).map(_source_reason_family)
    deferrable = family.isin(
        {
            "short_profit_take_exhaustion",
            "portfolio_midnight_cleanup_short",
            "portfolio_session_cleanup_short",
        },
    )
    deferred = pd.Series(False, index=out.index)
    for start in stall_starts:
        deferred |= (
            action.eq("CLOSE_SHORT")
            & deferrable
            & close_time.between(
                pd.Timestamp(start),
                pd.Timestamp(start) + pd.Timedelta(
                    minutes=float(config.stall_short_min_full_exit_age_minutes),
                ),
                inclusive="left",
            )
        )
    return out[~deferred].copy()


def apply_eth_inventory_overlay_signals(
    selected_open_signals: pd.DataFrame,
    selected_close_signals: pd.DataFrame,
    raw_open_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    raw_close_signals: pd.DataFrame | None = None,
    config: EthInventoryOverlayConfig | None = None,
    emit_pending_orders: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace qualified ETH entries with fills and merge causal inventory legs."""

    cfg = config or EthInventoryOverlayConfig()
    opens = _normalized_eth_actions(selected_open_signals, action_prefix="OPEN_")
    closes = _normalized_eth_actions(selected_close_signals, action_prefix="CLOSE_")
    closes = _apply_cross_horizon_stall_exit_hysteresis(opens, closes, config=cfg)
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth_features = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].drop_duplicates("bar_time", keep="last").set_index("bar_time")
    raw_closes = _normalized_eth_actions(
        raw_close_signals if raw_close_signals is not None else pd.DataFrame(),
        action_prefix="CLOSE_",
    )
    for close_index, close in closes.iterrows():
        signal_time = pd.Timestamp(close["signal_time"])
        previous_time = signal_time - pd.Timedelta(minutes=5)
        prior_cleanup = raw_closes[
            raw_closes["signal_time"].eq(previous_time)
            & raw_closes["action"].eq("CLOSE_LONG")
            & raw_closes.get("reason", pd.Series("", index=raw_closes.index)).astype(str)
            .str.startswith("portfolio_midnight_cleanup_long;")
        ]
        if (
            "long_relief_bounce_exit" in str(close.get("reason") or "")
            and not prior_cleanup.empty
            and signal_time in eth_features.index
        ):
            closes.loc[close_index, "signal_price"] = _number(eth_features.loc[signal_time].get("open"))
            closes.loc[close_index, "reason"] = (
                "eth_inventory_prior_bar_midnight_market_exit;" + str(close.get("reason") or "")
            )

    qualified_rows: list[pd.Series] = []
    for _, signal in opens.iterrows():
        signal_time = pd.Timestamp(signal["signal_time"])
        if signal_time not in eth_features.index:
            continue
        bar = eth_features.loc[signal_time]
        if (
            str(signal.get("action")) == "OPEN_LONG"
            and "capitulation_absorption_long" in str(signal.get("reason") or "")
            and _number(signal.get("score"), 0.0) >= float(cfg.dual_entry_score_floor)
            and _number(bar.get("quote_volume_ratio_med30")) >= float(cfg.dual_entry_volume_ratio_floor)
            and _number(bar.get("ret288")) <= float(cfg.dual_entry_ret288_ceiling)
            and _number(bar.get("donchian20_pos")) <= float(cfg.dual_entry_donchian_ceiling)
        ):
            qualified_rows.append(signal)
    qualified = pd.DataFrame(qualified_rows, columns=opens.columns)
    fills = build_conditional_factor_fills(
        qualified,
        feature_rows,
        config=ConditionalOrderConfig(
            valid_minutes=float(cfg.dual_entry_valid_minutes),
            pullback_atr_multiplier=float(cfg.dual_entry_pullback_atr_multiplier),
            emit_pending_orders=bool(emit_pending_orders),
            order_styles=("reversion_limit",),
        ),
    )
    replaced_ids = set(fills.get("setup_signal_id", pd.Series(dtype=str)).astype(str))
    effective_opens = opens[~opens.get("signal_id", pd.Series("", index=opens.index)).astype(str).isin(replaced_ids)].copy()
    if not fills.empty:
        dual_fills = pd.concat([fills, fills], ignore_index=True, sort=False)
        dual_fills["signal_id"] = [f"WJETHDUALFILL{index + 1:06d}" for index in range(len(dual_fills))]
        dual_fills["reason"] = (
            "eth_inventory_high_quality_capitulation_limit_entry;" + dual_fills["reason"].astype(str)
        )
        effective_opens = pd.concat([effective_opens, dual_fills], ignore_index=True, sort=False)
    effective_opens = effective_opens.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)

    extra_opens, extra_closes = build_eth_inventory_overlay_signals(
        effective_opens,
        closes,
        raw_open_signals,
        feature_rows,
        raw_close_signals=raw_close_signals,
        config=cfg,
    )
    merged_opens = (
        pd.concat([effective_opens, extra_opens], ignore_index=True, sort=False)
        if not extra_opens.empty
        else effective_opens.copy()
    )
    merged_closes = (
        pd.concat([closes, extra_closes], ignore_index=True, sort=False)
        if not extra_closes.empty
        else closes.copy()
    )
    return (
        merged_opens.sort_values(["signal_time", "signal_id"]).reset_index(drop=True),
        merged_closes.sort_values(["signal_time", "signal_id"]).reset_index(drop=True),
    )


def build_eth_inventory_overlay_signals(
    selected_open_signals: pd.DataFrame,
    selected_close_signals: pd.DataFrame,
    raw_open_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    raw_close_signals: pd.DataFrame | None = None,
    config: EthInventoryOverlayConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay bounded ETH inventory legs from selected signals and closed bars."""

    cfg = config or EthInventoryOverlayConfig()
    columns = ["signal_id", "symbol", "action", "signal_time", "signal_price", "score", "reason"]
    if feature_rows.empty:
        empty = pd.DataFrame(columns=columns)
        return empty.copy(), empty.copy()

    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    features = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].drop_duplicates("bar_time", keep="last").sort_values("bar_time")
    if features.empty:
        empty = pd.DataFrame(columns=columns)
        return empty.copy(), empty.copy()

    opens = _normalized_eth_actions(selected_open_signals, action_prefix="OPEN_")
    closes = _normalized_eth_actions(selected_close_signals, action_prefix="CLOSE_")
    raw_opens = _normalized_eth_actions(raw_open_signals, action_prefix="OPEN_")
    raw_closes = _normalized_eth_actions(
        raw_close_signals if raw_close_signals is not None else pd.DataFrame(),
        action_prefix="CLOSE_",
    )
    raw_by_time = {
        timestamp: frame
        for timestamp, frame in raw_opens.groupby("signal_time", sort=False)
    }
    raw_close_by_time = {
        timestamp: frame
        for timestamp, frame in raw_closes.groupby("signal_time", sort=False)
    }
    selected_open_by_time = {
        timestamp: frame
        for timestamp, frame in opens.groupby("signal_time", sort=False)
    }
    selected_close_by_time = {
        timestamp: frame
        for timestamp, frame in closes.groupby("signal_time", sort=False)
    }

    extra_opens: list[dict[str, object]] = []
    extra_closes: list[dict[str, object]] = []
    direction = ""
    state_start: pd.Timestamp | None = None
    state_entry_price = np.nan
    dual_capitulation = False
    partial_emitted = False
    short_add_emitted = False
    short_scaleout_emitted = False
    stall_short_campaign = False
    stall_short_flush_scaleout_emitted = False
    lot_count = 0

    def emit(target: list[dict[str, object]], action: str, row: pd.Series, reason: str, score: float) -> None:
        target.append(
            {
                "signal_id": "",
                "symbol": "ETHUSDT",
                "action": action,
                "signal_time": pd.Timestamp(row["bar_time"]),
                "signal_price": _number(row.get("close")),
                "score": float(score),
                "reason": reason,
            },
        )

    for _, bar in features.iterrows():
        bar_time = pd.Timestamp(bar["bar_time"])
        if (
            direction == "SHORT"
            and stall_short_campaign
            and not stall_short_flush_scaleout_emitted
            and state_start is not None
            and lot_count >= 2
        ):
            age = (bar_time - state_start).total_seconds() / 60.0
            raw_short_closes = raw_close_by_time.get(bar_time, pd.DataFrame())
            has_profit_take = not raw_short_closes[
                raw_short_closes.get("action", pd.Series("", index=raw_short_closes.index))
                .astype(str).eq("CLOSE_SHORT")
                & raw_short_closes.get("reason", pd.Series("", index=raw_short_closes.index))
                .astype(str).str.startswith("short_profit_take_exhaustion;")
            ].empty
            if (
                has_profit_take
                and age >= float(cfg.stall_short_flush_scaleout_min_age_minutes)
                and _number(bar.get("ret20")) <= float(cfg.stall_short_flush_ret20_ceiling)
                and _number(bar.get("quote_volume_ratio_med30"))
                >= float(cfg.stall_short_flush_volume_ratio_floor)
                and _number(bar.get("rsi14")) <= float(cfg.stall_short_flush_rsi_ceiling)
            ):
                emit(
                    extra_closes,
                    "CLOSE_SHORT",
                    bar,
                    "eth_cross_horizon_stall_flush_half_scaleout",
                    3.0,
                )
                stall_short_flush_scaleout_emitted = True

        if (
            direction == "SHORT"
            and not stall_short_campaign
            and not short_scaleout_emitted
            and lot_count >= int(cfg.short_scaleout_min_lots)
        ):
            raw_short_closes = raw_close_by_time.get(bar_time, pd.DataFrame())
            profit_takes = raw_short_closes[
                raw_short_closes.get("action", pd.Series("", index=raw_short_closes.index))
                .astype(str).eq("CLOSE_SHORT")
                & raw_short_closes.get("reason", pd.Series("", index=raw_short_closes.index))
                .astype(str).str.startswith("short_profit_take_exhaustion;")
                & pd.to_numeric(
                    raw_short_closes.get(
                        "score",
                        pd.Series(np.nan, index=raw_short_closes.index, dtype=float),
                    ),
                    errors="coerce",
                )
                .ge(float(cfg.short_scaleout_score_floor))
            ]
            if not profit_takes.empty:
                for _ in range(lot_count):
                    emit(extra_closes, "CLOSE_SHORT", bar, "eth_inventory_short_profit_scaleout", 3.0)
                short_scaleout_emitted = True

        for close in selected_close_by_time.get(bar_time, pd.DataFrame()).itertuples(index=False):
            closed_direction = "LONG" if str(close.action) == "CLOSE_LONG" else "SHORT"
            if direction == closed_direction:
                direction = ""
                state_start = None
                state_entry_price = np.nan
                dual_capitulation = False
                partial_emitted = False
                short_add_emitted = False
                short_scaleout_emitted = False
                stall_short_campaign = False
                stall_short_flush_scaleout_emitted = False
                lot_count = 0

        for selected in selected_open_by_time.get(bar_time, pd.DataFrame()).itertuples(index=False):
            opened_direction = "LONG" if str(selected.action) == "OPEN_LONG" else "SHORT"
            reason = str(getattr(selected, "reason", ""))
            score = _number(getattr(selected, "score", 0.0), 0.0)
            if direction != opened_direction:
                direction = opened_direction
                state_start = bar_time
                state_entry_price = _number(bar.get("close"))
                dual_capitulation = False
                partial_emitted = False
                short_add_emitted = False
                short_scaleout_emitted = False
                stall_short_campaign = _is_cross_horizon_stall_short_setup(reason)
                stall_short_flush_scaleout_emitted = False
                lot_count = 1
            else:
                lot_count += 1
            if "eth_inventory_high_quality_capitulation_limit_entry" in reason:
                dual_capitulation = True
            elif (
                opened_direction == "LONG"
                and not dual_capitulation
                and "capitulation_absorption_long" in reason
                and score >= float(cfg.dual_entry_score_floor)
                # Event selection happens after the 30-minute bucket closes.
                # Keep the causal source-bar structure carried by the signal
                # instead of replacing it with the later decision bar.
                and _signal_reason_metric(
                    reason,
                    "qvr",
                    fallback=_number(bar.get("quote_volume_ratio_med30")),
                ) >= float(cfg.dual_entry_volume_ratio_floor)
                and _signal_reason_metric(
                    reason,
                    "ret288",
                    fallback=_number(bar.get("ret288")),
                ) <= float(cfg.dual_entry_ret288_ceiling)
                and _signal_reason_metric(
                    reason,
                    "don",
                    fallback=_number(bar.get("donchian20_pos")),
                ) <= float(cfg.dual_entry_donchian_ceiling)
            ):
                emit(
                    extra_opens,
                    "OPEN_LONG",
                    bar,
                    "eth_precision_two_stage_absorption_second_leg;capitulation_absorption_long",
                    score,
                )
                dual_capitulation = True

        if direction == "SHORT" and state_start is not None and not short_add_emitted:
            age = (bar_time - state_start).total_seconds() / 60.0
            raw_short = raw_by_time.get(bar_time, pd.DataFrame())
            has_short_factor = not raw_short[
                raw_short.get("action", pd.Series("", index=raw_short.index)).astype(str).eq("OPEN_SHORT")
            ].empty
            if (
                has_short_factor
                and age >= float(cfg.short_add_min_age_minutes)
                and _number(bar.get("ret20")) >= float(cfg.short_add_ret20_floor)
                and _number(bar.get("ret60")) >= float(cfg.short_add_ret60_floor)
                and _number(bar.get("rsi14")) >= float(cfg.short_add_rsi_floor)
                and _number(bar.get("quote_volume_ratio_med30")) >= float(cfg.short_add_volume_ratio_floor)
            ):
                emit(extra_opens, "OPEN_SHORT", bar, "eth_inventory_squeeze_short_add", 3.0)
                short_add_emitted = True
                lot_count += 1

        if direction == "LONG" and dual_capitulation and not partial_emitted and state_start is not None:
            age = (bar_time - state_start).total_seconds() / 60.0
            return_pct = (
                (_number(bar.get("close")) / state_entry_price - 1.0) * 100.0
                if np.isfinite(state_entry_price) and state_entry_price > 0.0
                else np.nan
            )
            strict_recovery = (
                age >= float(cfg.scaleout_min_age_minutes)
                and return_pct >= float(cfg.scaleout_return_floor_pct)
                and _number(bar.get("taker_buy_ratio")) >= float(cfg.scaleout_taker_buy_floor)
                and _number(bar.get("quote_volume_ratio_med30")) >= float(cfg.scaleout_volume_ratio_floor)
            )
            raw_long_closes = raw_close_by_time.get(bar_time, pd.DataFrame())
            has_session_cleanup = not raw_long_closes[
                raw_long_closes.get("action", pd.Series("", index=raw_long_closes.index))
                .astype(str).eq("CLOSE_LONG")
                & raw_long_closes.get("reason", pd.Series("", index=raw_long_closes.index))
                .astype(str).str.startswith("portfolio_session_cleanup_long;")
            ].empty
            session_recovery = (
                has_session_cleanup
                and age >= float(cfg.recovery_scaleout_min_age_minutes)
                and return_pct >= float(cfg.recovery_scaleout_return_floor_pct)
                and _number(bar.get("taker_buy_ratio")) >= float(cfg.recovery_scaleout_taker_buy_floor)
                and _number(bar.get("quote_volume_ratio_med30"))
                >= float(cfg.recovery_scaleout_volume_ratio_floor)
            )
            if strict_recovery or session_recovery:
                close_reason = (
                    "eth_precision_two_stage_absorption_first_scaleout"
                    if session_recovery
                    else "eth_inventory_capitulation_first_scaleout"
                )
                emit(extra_closes, "CLOSE_LONG", bar, close_reason, 3.0)
                partial_emitted = True

    for index, row in enumerate(extra_opens, start=1):
        row["signal_id"] = f"WJETHINVENTORYOPEN{index:06d}"
    for index, row in enumerate(extra_closes, start=1):
        row["signal_id"] = f"WJETHINVENTORYCLOSE{index:06d}"
    return pd.DataFrame(extra_opens, columns=columns), pd.DataFrame(extra_closes, columns=columns)


def _normalized_eth_actions(signals: pd.DataFrame, *, action_prefix: str) -> pd.DataFrame:
    out = signals.copy()
    if out.empty:
        return pd.DataFrame(
            columns=["signal_id", "signal_time", "symbol", "action", "signal_price", "score", "reason"],
        )
    out["signal_time"] = pd.to_datetime(out.get("signal_time"), errors="coerce")
    return out[
        out.get("symbol", pd.Series("", index=out.index)).astype(str).eq("ETHUSDT")
        & out.get("action", pd.Series("", index=out.index)).astype(str).str.startswith(action_prefix)
        & out["signal_time"].notna()
    ].sort_values(["signal_time", "action"]).copy()


def build_precision_eth_research_signals(
    selected_open_signals: pd.DataFrame,
    selected_close_signals: pd.DataFrame,
    raw_open_signals: pd.DataFrame,
    raw_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    emit_pending_orders: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compress the ETH research frontier with autonomous state-quality gates."""

    base_open, base_close = apply_eth_inventory_overlay_signals(
        selected_open_signals,
        selected_close_signals,
        raw_open_signals,
        feature_rows,
        raw_close_signals=raw_close_signals,
        emit_pending_orders=emit_pending_orders,
    )
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth_features = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].drop_duplicates("bar_time", keep="last").set_index("bar_time")
    eth_close = pd.to_numeric(eth_features.get("close"), errors="coerce")
    campaign_ready = (
        eth_close.pct_change().rolling(864, min_periods=864).std()
        * np.sqrt(864.0)
        * 100.0
    ).ge(4.48)
    campaign_range = (
        pd.to_numeric(eth_features.get("high"), errors="coerce").rolling(864, min_periods=864).max()
        / pd.to_numeric(eth_features.get("low"), errors="coerce").rolling(864, min_periods=864).min()
        - 1.0
    ) * 100.0
    shock_precursor = campaign_range.ge(14.0) & (
        eth_close.pct_change().rolling(864, min_periods=864).std()
        * np.sqrt(864.0)
        * 100.0
    ).ge(5.0)
    recovery_ready = campaign_ready | shock_precursor
    causal_positions = build_causal_eth_positions_from_selected_signals(
        selected_open_signals,
        selected_close_signals,
        feature_rows,
    )
    flush_reversals = build_eth_extreme_flush_reversal_entry_fills(
        feature_rows,
        emit_pending_orders=emit_pending_orders,
    )
    flush_reversal_times = set(
        pd.to_datetime(flush_reversals.get("signal_time"), errors="coerce").dropna(),
    )
    moderate_flush_pullbacks = build_eth_moderate_flush_pullback_entry_fills(
        feature_rows,
        emit_pending_orders=emit_pending_orders,
    )
    moderate_flush_times = set(
        pd.to_datetime(moderate_flush_pullbacks.get("signal_time"), errors="coerce").dropna(),
    )
    blowoff_retrace_shorts = build_eth_blowoff_retrace_short_entry_fills(
        feature_rows,
        emit_pending_orders=emit_pending_orders,
    )
    blowoff_retrace_times = set(
        pd.to_datetime(blowoff_retrace_shorts.get("signal_time"), errors="coerce").dropna(),
    )
    early_core_entries = build_eth_early_core_trend_entry_fills(
        raw_open_signals,
        feature_rows,
        emit_pending_orders=emit_pending_orders,
    )
    capitulation_flip_shorts = build_eth_capitulation_exit_flip_short_signals(
        causal_positions,
        feature_rows,
    )
    overbought_limit_exits = build_eth_overbought_limit_exit_fills(
        selected_close_signals,
        feature_rows,
        emit_pending_orders=emit_pending_orders,
    )
    relief_recovery_exits = build_eth_relief_recovery_exit_signals(
        selected_close_signals,
        feature_rows,
    )
    replaced_limit_exit_ids = set(
        overbought_limit_exits.get("setup_signal_id", pd.Series(dtype=str)).astype(str),
    )
    replaced_relief_exit_ids = set(
        relief_recovery_exits.get("setup_signal_id", pd.Series(dtype=str)).astype(str),
    )

    base_open = base_open.copy()
    base_open["signal_time"] = pd.to_datetime(base_open.get("signal_time"), errors="coerce")
    base_open["precision_family"] = base_open.get("reason", pd.Series("", index=base_open.index)).map(
        _source_reason_family,
    )
    keep_open = pd.Series(True, index=base_open.index)
    keep_open &= ~base_open["precision_family"].isin(
        {"core_trend_long", "eth_inventory_high_quality_capitulation_second_leg"},
    )
    for index, signal in base_open.iterrows():
        signal_time = pd.Timestamp(signal["signal_time"])
        feature = eth_features.loc[signal_time] if signal_time in eth_features.index else pd.Series(dtype=float)
        regime_ready = bool(campaign_ready.get(signal_time, False))
        family = str(signal["precision_family"])
        if family == "eth_inventory_high_quality_capitulation_limit_entry":
            source_reason = str(signal.get("reason") or "")
            source_structure_ready = bool(
                _signal_reason_metric(source_reason, "qvr")
                >= float(EthInventoryOverlayConfig().dual_entry_volume_ratio_floor)
                and _signal_reason_metric(source_reason, "ret288")
                <= float(EthInventoryOverlayConfig().dual_entry_ret288_ceiling)
                and _signal_reason_metric(source_reason, "don")
                <= float(EthInventoryOverlayConfig().dual_entry_donchian_ceiling),
            )
            if not source_structure_ready:
                keep_open.loc[index] = False
            continue
        if family == "eth_precision_two_stage_absorption_second_leg":
            # The overlay can materialize a later conditional leg from a
            # source setup that the precision gate subsequently rejects.
            # Revalidate the actual fill bar so an orphan second leg cannot
            # survive after its primary campaign was removed.
            if not _eth_precision_direct_combo_allowed(
                feature,
                family="capitulation_absorption_long",
                action=str(signal.get("action") or ""),
                campaign_ready=regime_ready,
            ):
                keep_open.loc[index] = False
            continue
        direct_combo = str(signal.get("reason") or "").startswith("combo_event_open;")
        if direct_combo and not _eth_precision_direct_combo_allowed(
            feature,
            family=family,
            action=str(signal.get("action") or ""),
            campaign_ready=regime_ready,
        ):
            keep_open.loc[index] = False
            continue
        if family == "capitulation_absorption_long":
            if (
                signal_time in flush_reversal_times
                or signal_time in moderate_flush_times
                or _number(feature.get("ret288")) > -1.50
            ):
                keep_open.loc[index] = False
        elif (family == "eth_anchor_exhaustion_short" and not _eth_precision_anchor_short_allowed(feature)) or (family == "eth_anchor_exhaustion_short" and signal_time in blowoff_retrace_times):
            keep_open.loc[index] = False
    dropped_linked_close_ids = set(
        base_open.loc[
            ~keep_open & base_open["precision_family"].eq("core_trend_long"),
            "signal_id",
        ].astype(str),
    )
    base_open = base_open[keep_open].drop(columns=["precision_family"], errors="ignore")

    direct_recovery = _normalized_eth_actions(selected_open_signals, action_prefix="OPEN_")
    if not direct_recovery.empty:
        direct_recovery = direct_recovery[
            direct_recovery.get("reason", pd.Series("", index=direct_recovery.index))
            .astype(str).str.startswith("combo_event_open;")
        ].copy()
        keep_recovery = pd.Series(False, index=direct_recovery.index)
        existing_keys = set(zip(pd.to_datetime(base_open.get("signal_time"), errors="coerce"), base_open.get("action", pd.Series(dtype=str))))
        for index, signal in direct_recovery.iterrows():
            signal_time = pd.Timestamp(signal["signal_time"])
            action = str(signal.get("action") or "")
            family = _source_reason_family(str(signal.get("reason") or ""))
            feature = eth_features.loc[signal_time] if signal_time in eth_features.index else pd.Series(dtype=float)
            qvr = _number(feature.get("quote_volume_ratio_med30"))
            ret20 = _number(feature.get("ret20"))
            ret60 = _number(feature.get("ret60"))
            ret144 = _number(feature.get("ret144"))
            ret288 = _number(feature.get("ret288"))
            taker = _number(feature.get("taker_buy_ratio"))
            rsi = _number(feature.get("rsi14"))
            adx = _number(feature.get("adx14"))
            negative_divergence = bool(recovery_ready.get(signal_time, False)) and (
                ret144 >= 5.0 and ret288 < 0.0 and taker <= 0.40 and qvr <= 3.0
                and 40.0 <= adx <= 50.0
            )
            low_volume_blowoff = bool(campaign_ready.get(signal_time, False)) and (
                ret60 >= 3.0 and qvr <= 1.50 and (adx >= 70.0 or ret20 <= 0.10)
            )
            volume_climax = bool(campaign_ready.get(signal_time, False)) and (
                _number(campaign_range.get(signal_time)) >= 10.0
                and qvr >= 20.0 and taker >= 0.60 and rsi >= 80.0
                and 35.0 <= adx <= 45.0
            )
            keep_recovery.loc[index] = (
                (signal_time, action) not in existing_keys
                and family == "eth_anchor_exhaustion_short"
                and action == "OPEN_SHORT"
                and (negative_divergence or low_volume_blowoff or volume_climax)
            )
        direct_recovery = direct_recovery[keep_recovery]

    positions = _normalized_eth_long_positions(causal_positions)
    range_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for window_start, window_end in _active_state_windows(positions):
        scoped = positions[
            positions["open_time"].between(window_start, window_end, inclusive="both")
        ].sort_values("open_time")
        if not scoped.empty and str(scoped.iloc[0].get("open_reason_family") or "") == "range_reclaim_long":
            range_windows.append((window_start, window_end))

    ladder_windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for position in positions[
        positions.get("open_reason_family", pd.Series("", index=positions.index))
        .astype(str).eq("capitulation_absorption_long")
    ].itertuples(index=False):
        if position.open_time not in eth_features.index:
            continue
        feature = eth_features.loc[position.open_time]
        if (
            3.0 <= _number(feature.get("quote_volume_ratio_med30")) <= 8.0
            and -3.0 <= _number(feature.get("ret288")) <= -1.50
        ):
            ladder_windows.append(
                (
                    pd.Timestamp(position.open_time),
                    pd.Timestamp(position.close_time),
                    pd.Timestamp(position.open_time) + pd.Timedelta(hours=24),
                ),
            )

    base_close = base_close.copy()
    base_close["signal_time"] = pd.to_datetime(base_close.get("signal_time"), errors="coerce")
    close_reason = base_close.get("reason", pd.Series("", index=base_close.index)).astype(str)
    close_family = close_reason.map(_source_reason_family)
    linked_open = close_reason.str.extract(r"open_signal=([^;]+)")[0].astype(str)
    keep_close = (
        ~close_family.str.startswith("portfolio_")
        & ~linked_open.isin(dropped_linked_close_ids)
        & ~base_close.get("signal_id", pd.Series("", index=base_close.index)).astype(str).isin(
            replaced_limit_exit_ids,
        )
        & ~base_close.get("signal_id", pd.Series("", index=base_close.index)).astype(str).isin(
            replaced_relief_exit_ids,
        )
    )
    for index, signal in base_close.iterrows():
        if not keep_close.loc[index]:
            continue
        signal_time = pd.Timestamp(signal["signal_time"])
        if close_family.loc[index] == "eth_inventory_capitulation_first_scaleout":
            recent_long_legs = base_open[
                base_open.get("action", pd.Series("", index=base_open.index)).astype(str).eq("OPEN_LONG")
                & base_open["signal_time"].between(
                    signal_time - pd.Timedelta(minutes=60),
                    signal_time,
                    inclusive="both",
                )
            ]
            if len(recent_long_legs) < 2:
                keep_close.loc[index] = False
            continue
        if close_family.loc[index] != "long_profit_take_exhaustion":
            continue
        feature = eth_features.loc[signal_time] if signal_time in eth_features.index else pd.Series(dtype=float)
        if _in_time_windows(signal_time, range_windows) and _number(feature.get("ret20")) < 3.0:
            keep_close.loc[index] = False
        if any(start <= signal_time <= runner_end for start, _, runner_end in ladder_windows):
            keep_close.loc[index] = False
    base_close = base_close[keep_close]

    addons = build_active_eth_long_addon_fills(
        causal_positions,
        feature_rows,
        emit_pending_orders=emit_pending_orders,
    )
    if not addons.empty:
        addons["decision_time"] = pd.to_datetime(addons.get("decision_time"), errors="coerce")
        addon_reason = addons.get("reason", pd.Series("", index=addons.index)).astype(str)
        general = addon_reason.str.contains(
            "eth_active_long_add_setup",
            regex=False,
        )
        breakout = addon_reason.str.contains("eth_active_long_breakout_add_setup", regex=False)
        keep_addon = ~(general | breakout)
        for index, signal in addons[general].iterrows():
            decision_time = pd.Timestamp(signal["decision_time"])
            feature = eth_features.loc[decision_time] if decision_time in eth_features.index else pd.Series(dtype=float)
            keep_addon.loc[index] = (
                _in_time_windows(decision_time, range_windows)
                and _eth_precision_active_addon_allowed(
                    feature,
                    campaign_ready=bool(campaign_ready.get(decision_time, False)),
                )
            )
        for index, signal in addons[breakout].iterrows():
            decision_time = pd.Timestamp(signal["decision_time"])
            feature = eth_features.loc[decision_time] if decision_time in eth_features.index else pd.Series(dtype=float)
            keep_addon.loc[index] = _eth_precision_breakout_addon_allowed(
                feature,
                campaign_ready=bool(campaign_ready.get(decision_time, False)),
            )
        addons = addons[keep_addon]

    def campaign_component(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        times = pd.to_datetime(frame.get("signal_time"), errors="coerce")
        return frame[times.map(lambda value: bool(campaign_ready.get(value, False)))].copy()

    blowoff_retrace_shorts = campaign_component(blowoff_retrace_shorts)
    early_core_entries = campaign_component(early_core_entries)
    capitulation_flip_shorts = campaign_component(capitulation_flip_shorts)

    scaleouts = build_active_eth_long_scaleout_signals(
        causal_positions,
        raw_close_signals,
        feature_rows,
    )
    if not scaleouts.empty:
        scaleouts["signal_time"] = pd.to_datetime(scaleouts.get("signal_time"), errors="coerce")
        scaleouts = scaleouts[
            scaleouts["signal_time"].map(lambda value: _in_time_windows(pd.Timestamp(value), range_windows))
            & ~scaleouts.get("reason", pd.Series("", index=scaleouts.index)).astype(str).eq(
                "eth_runner_blowoff",
            )
        ]

    ladder = build_eth_capitulation_profit_ladder_signals(
        causal_positions,
        feature_rows,
        raw_close_signals=raw_close_signals,
    )
    if not ladder.empty:
        ladder["signal_time"] = pd.to_datetime(ladder.get("signal_time"), errors="coerce")
        ladder_reason = ladder.get("reason", pd.Series("", index=ladder.index)).astype(str)
        keep_ladder = pd.Series(False, index=ladder.index)
        for index, signal in ladder.iterrows():
            signal_time = pd.Timestamp(signal["signal_time"])
            is_final = ladder_reason.loc[index] == "eth_capitulation_final_runner"
            keep_ladder.loc[index] = any(
                start <= signal_time <= (runner_end if is_final else close_time)
                for start, close_time, runner_end in ladder_windows
            )
        ladder = ladder[keep_ladder]

    components = [
        base_open,
        base_close,
        direct_recovery,
        addons,
        scaleouts,
        ladder,
        flush_reversals,
        moderate_flush_pullbacks,
        blowoff_retrace_shorts,
        early_core_entries,
        capitulation_flip_shorts,
        overbought_limit_exits,
        relief_recovery_exits,
        build_eth_early_core_trend_fade_scaleout_signals(
            causal_positions,
            raw_close_signals,
            feature_rows,
        ),
    ]
    opens: list[pd.DataFrame] = []
    closes: list[pd.DataFrame] = []
    for component_index, component in enumerate(components):
        if component.empty:
            continue
        frame = component.copy()
        frame["signal_id"] = [
            f"WJETHPRECISION{component_index:02d}{row_index + 1:06d}"
            for row_index in range(len(frame))
        ]
        opens.append(frame[frame.get("action", pd.Series("", index=frame.index)).astype(str).str.startswith("OPEN_")])
        closes.append(frame[frame.get("action", pd.Series("", index=frame.index)).astype(str).str.startswith("CLOSE_")])
    empty = pd.DataFrame(columns=["signal_id", "symbol", "action", "signal_time", "signal_price", "score", "reason"])
    open_frame = pd.concat(opens, ignore_index=True, sort=False) if opens else empty.copy()
    close_frame = pd.concat(closes, ignore_index=True, sort=False) if closes else empty.copy()
    open_frame = build_eth_autonomous_campaign_open_signals(
        open_frame,
        raw_open_signals,
        raw_close_signals,
        feature_rows,
        emit_pending_orders=emit_pending_orders,
    )
    close_frame = build_eth_autonomous_campaign_close_signals(
        open_frame,
        close_frame,
        raw_close_signals,
        feature_rows,
        emit_pending_orders=emit_pending_orders,
    )
    return (
        open_frame.sort_values(["signal_time", "signal_id"]).reset_index(drop=True),
        close_frame.sort_values(["signal_time", "signal_id"]).reset_index(drop=True),
    )


def build_eth_autonomous_campaign_open_signals(
    selected_open_signals: pd.DataFrame,
    raw_open_signals: pd.DataFrame,
    raw_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    emit_pending_orders: bool = False,
) -> pd.DataFrame:
    """Recover sparse ETH campaign seeds directly from closed market factors.

    The event layer is intentionally conservative and waits for a full 30-minute
    cross-asset bucket.  Fast ETH transitions can finish inside that bucket, so
    this layer admits a small set of independently sufficient 5-minute market
    structures.  Public operations are not accepted by this function.
    """

    columns = [
        "signal_id", "symbol", "action", "signal_time", "signal_price", "score", "reason",
    ]
    if feature_rows.empty:
        return selected_open_signals.copy()
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].drop_duplicates("bar_time", keep="last").sort_values("bar_time").set_index("bar_time")
    if eth.empty:
        return selected_open_signals.copy()
    eth_close = pd.to_numeric(eth.get("close"), errors="coerce")
    realized_volatility = (
        eth_close.pct_change().rolling(864, min_periods=864).std()
        * np.sqrt(864.0)
        * 100.0
    )

    raw_open = _eth_reason_metric_frame(raw_open_signals, action_prefix="OPEN_")
    raw_close = _eth_reason_metric_frame(raw_close_signals, action_prefix="CLOSE_")
    candidates: list[pd.DataFrame] = []

    anchor_short = raw_open[
        raw_open["action"].eq("OPEN_SHORT")
        & raw_open["reason_family"].eq("eth_anchor_exhaustion_short")
    ].copy()
    if not anchor_short.empty:
        anchor_short["realized_volatility"] = anchor_short["signal_time"].map(realized_volatility)
        gates = {
            "negative_divergence_short": (
                anchor_short["ret144"].ge(5.0)
                & anchor_short["ret288"].lt(0.0)
                & anchor_short["taker"].le(0.40)
                & anchor_short["adx"].between(40.0, 50.0)
                & anchor_short["realized_volatility"].ge(4.48)
            ),
            "low_volume_blowoff_short": (
                anchor_short["ret20"].ge(3.0)
                & anchor_short["qvr"].le(0.80)
                & anchor_short["taker"].ge(0.55)
                & anchor_short["adx"].between(50.0, 65.0)
            ),
            "volume_climax_short": (
                anchor_short["qvr"].ge(10.0)
                & anchor_short["rsi"].ge(70.0)
                & anchor_short["adx"].ge(35.0)
            ),
            "cross_horizon_stall_short": (
                anchor_short["ret60"].ge(3.0)
                & anchor_short["qvr"].le(1.0)
                & anchor_short["ret20"].abs().le(0.20)
                & anchor_short["rsi"].between(55.0, 72.0)
                & anchor_short["down"].ge(0.67)
            ),
            "liquidity_climax_reversal_short": anchor_short["signal_time"].map(
                lambda signal_time: bool(
                    signal_time in eth.index
                    and _liquidity_climax_reversal_short_allowed(eth.loc[signal_time]),
                ),
            ),
        }
        for family, mask in gates.items():
            frame = anchor_short[mask].copy()
            frame["campaign_family"] = family
            frame["campaign_cooldown_minutes"] = 30.0
            candidates.append(frame)

    absorption = raw_open[
        raw_open["action"].eq("OPEN_LONG")
        & raw_open["reason_family"].eq("capitulation_absorption_long")
        & raw_open["ret288"].le(-4.0)
        & raw_open["ret20"].le(-1.0)
        & raw_open["rsi"].between(25.0, 38.0)
        & raw_open["qvr"].between(1.0, 3.0)
        & raw_open["don"].le(0.40)
    ].copy()
    absorption["campaign_family"] = "bear_absorption_long"
    absorption["campaign_cooldown_minutes"] = 120.0
    candidates.append(absorption)

    # A concentrated sell-volume absorption can complete inside one 30-minute
    # event bucket and therefore never reach the portfolio event selector.
    # Recover it directly from the closed 5m source bar, but keep execution
    # causal: the entry is still a later traded-range fill at 1.6 ATR below the
    # setup close.  The two equal fills represent the strategy's two inventory
    # legs; no public-operation timestamp or price is consumed here.
    inventory_cfg = EthInventoryOverlayConfig()
    high_quality_absorption = raw_open[
        raw_open["action"].eq("OPEN_LONG")
        & raw_open["reason_family"].eq("capitulation_absorption_long")
        & raw_open["score"].ge(float(inventory_cfg.dual_entry_score_floor))
        & raw_open["qvr"].ge(float(inventory_cfg.dual_entry_volume_ratio_floor))
        & raw_open["ret144"].le(-2.50)
        & raw_open["ret288"].le(float(inventory_cfg.dual_entry_ret288_ceiling))
        & raw_open["don"].le(float(inventory_cfg.dual_entry_donchian_ceiling))
    ].copy()
    high_quality_absorption["campaign_family"] = "high_quality_capitulation_inventory_long"
    high_quality_absorption["campaign_cooldown_minutes"] = 120.0
    candidates.append(high_quality_absorption)

    # A mature multi-day selloff needs a materially deeper exhaustion state
    # than the first oversold absorption print.  This autonomous structure is
    # deliberately narrow: expanding trend, lower-channel close, capitulation
    # RSI, and normalized (rather than climax) taker recovery must coexist.
    deep_absorption = raw_open[
        raw_open["action"].eq("OPEN_LONG")
        & raw_open["reason_family"].eq("capitulation_absorption_long")
        & raw_open["ret288"].le(-5.0)
        & raw_open["ret20"].between(-1.50, -1.0)
        & raw_open["rsi"].between(12.0, 22.0)
        & raw_open["qvr"].between(3.0, 8.0)
        & raw_open["taker"].ge(0.45)
        & raw_open["don"].le(0.10)
        & raw_open["adx"].ge(35.0)
    ].copy()
    deep_absorption["campaign_family"] = "deep_flush_absorption_long"
    deep_absorption["campaign_cooldown_minutes"] = 120.0
    candidates.append(deep_absorption)

    post_flush = raw_open[
        raw_open["action"].eq("OPEN_SHORT")
        & raw_open["reason_family"].eq("core_downtrend_short")
        & raw_open["ret60"].le(-3.0)
        & raw_open["ret288"].le(-2.0)
        & raw_open["rsi"].between(45.0, 60.0)
    ].copy()
    post_flush["campaign_family"] = "post_flush_short"
    post_flush["campaign_cooldown_minutes"] = 30.0
    candidates.append(post_flush)

    anchor_long = raw_open[
        raw_open["action"].eq("OPEN_LONG")
        & raw_open["reason_family"].eq("eth_anchor_long")
        & raw_open["taker"].ge(0.65)
        & raw_open["qvr"].ge(1.50)
        & raw_open["ret60"].le(0.0)
        & raw_open["ret288"].ge(-1.0)
        & raw_open["adx"].le(25.0)
    ].copy()
    anchor_long["campaign_family"] = "flow_anchor_long"
    anchor_long["campaign_cooldown_minutes"] = 30.0
    candidates.append(anchor_long)

    continuation = raw_open[
        raw_open["action"].eq("OPEN_LONG")
        & raw_open["reason_family"].eq("eth_anchor_long")
        & raw_open["ret20"].between(0.50, 0.80)
        & raw_open["ret60"].between(-0.80, 0.0)
        & raw_open["ret288"].between(-1.0, 1.0)
        & raw_open["qvr"].between(1.50, 3.0)
        & raw_open["taker"].between(0.55, 0.65)
        & raw_open["adx"].le(25.0)
    ].copy()
    continuation["campaign_family"] = "breakout_continuation_add"
    continuation["campaign_cooldown_minutes"] = 30.0
    candidates.append(continuation)

    bearish_flip = raw_close[
        raw_close["action"].eq("CLOSE_LONG")
        & raw_close["reason_family"].eq("long_profit_take_exhaustion")
        & raw_close["ret20"].ge(1.0)
        & raw_close["ret288"].le(-4.0)
        & raw_close["don"].ge(0.80)
        & raw_close["rsi"].ge(60.0)
    ].copy()
    bearish_flip["action"] = "OPEN_SHORT"
    bearish_flip["campaign_family"] = "bearish_rebound_flip_short"
    bearish_flip["campaign_cooldown_minutes"] = 120.0
    candidates.append(bearish_flip)

    candidate_frame = pd.concat(
        [frame for frame in candidates if not frame.empty],
        ignore_index=True,
        sort=False,
    ) if any(not frame.empty for frame in candidates) else pd.DataFrame()
    repair_rows: list[dict[str, object]] = []
    last_by_family: dict[str, pd.Timestamp] = {}
    last_absorption_long: pd.Timestamp | None = None
    absorption_long_families = {
        "bear_absorption_long",
        "deep_flush_absorption_long",
        "high_quality_capitulation_inventory_long",
    }
    if not candidate_frame.empty:
        candidate_frame = candidate_frame.sort_values(["signal_time", "campaign_family", "score"])
        for signal in candidate_frame.itertuples(index=False):
            signal_time = pd.Timestamp(signal.signal_time)
            family = str(signal.campaign_family)
            if (
                family == "high_quality_capitulation_inventory_long"
                and last_absorption_long is not None
                and (signal_time - last_absorption_long).total_seconds() / 60.0 < 480.0
            ):
                continue
            if family in {"cross_horizon_stall_short", "volume_climax_short"}:
                # The first stall/climax bar is only a setup.  Wait for one
                # more closed 5m bar and require price not to extend above the
                # setup close before committing the short.  A volume climax
                # additionally needs taker flow to flip to seller control;
                # otherwise the large print is still an intact squeeze rather
                # than exhaustion.  Both checks use only closed market data.
                confirmation_time = signal_time + pd.Timedelta(minutes=5)
                if confirmation_time not in eth.index:
                    continue
                setup_price = _number(eth.loc[signal_time].get("close"))
                confirmation_price = _number(eth.loc[confirmation_time].get("close"))
                confirmation_taker = _number(
                    eth.loc[confirmation_time].get("taker_buy_ratio"),
                    np.nan,
                )
                max_confirmation_price = setup_price * (
                    1.001 if family == "cross_horizon_stall_short" else 1.0
                )
                if (
                    not np.isfinite(setup_price)
                    or not np.isfinite(confirmation_price)
                    or confirmation_price > max_confirmation_price
                    or (
                        family == "volume_climax_short"
                        and (
                            not np.isfinite(confirmation_taker)
                            or confirmation_taker > 0.50
                        )
                    )
                ):
                    continue
                signal_time = confirmation_time
            last = last_by_family.get(family)
            cooldown = max(0.0, _number(signal.campaign_cooldown_minutes))
            if last is not None and (signal_time - last).total_seconds() / 60.0 < cooldown:
                continue
            if signal_time not in eth.index:
                continue
            price = _number(eth.loc[signal_time].get("close"))
            if not np.isfinite(price) or price <= 0.0:
                continue
            repair_rows.append(
                {
                    "signal_id": f"WJETHCAMPAIGNOPEN{len(repair_rows) + 1:06d}",
                    "symbol": "ETHUSDT",
                    "action": str(signal.action),
                    "signal_time": signal_time,
                    "signal_price": price,
                    "score": _number(signal.score, 0.0),
                    "reason": f"eth_causal_{family};" + str(signal.reason),
                },
            )
            last_by_family[family] = signal_time
            if family in absorption_long_families:
                last_absorption_long = signal_time
    repairs = pd.DataFrame(repair_rows, columns=columns)

    # A mild absorption print is a setup, not an unconditional market entry.
    # Rest one half-ATR buy limit from the closed setup bar and let a later
    # traded range decide whether the probe exists.  Stronger flush entries
    # keep their existing confirmation rules.
    if not repairs.empty:
        repair_reason = repairs.get("reason", pd.Series("", index=repairs.index)).astype(str)
        repair_ret20 = pd.to_numeric(
            repair_reason.str.extract(r"(?:^|;)ret20=([-0-9.]+)")[0],
            errors="coerce",
        )
        repair_rsi = pd.to_numeric(
            repair_reason.str.extract(r"(?:^|;)rsi=([-0-9.]+)")[0],
            errors="coerce",
        )
        fragile_mask = (
            repair_reason.str.contains("eth_causal_bear_absorption_long", regex=False)
            & repair_ret20.gt(-1.50)
            & repair_rsi.le(34.0)
        )
        fragile = repairs[fragile_mask].copy()
        if not fragile.empty:
            conditional_fragile = build_conditional_factor_fills(
                fragile,
                feature_rows,
                config=ConditionalOrderConfig(
                    valid_minutes=15.0,
                    pullback_atr_multiplier=0.50,
                    emit_pending_orders=emit_pending_orders,
                    order_styles=("reversion_limit",),
                ),
            )
            repairs = pd.concat(
                [repairs[~fragile_mask], conditional_fragile],
                ignore_index=True,
                sort=False,
            )

        repair_reason = repairs.get("reason", pd.Series("", index=repairs.index)).astype(str)
        inventory_mask = repair_reason.str.contains(
            "eth_causal_high_quality_capitulation_inventory_long",
            regex=False,
        )
        inventory_setups = repairs[inventory_mask].copy()
        if not inventory_setups.empty:
            inventory_fills = build_conditional_factor_fills(
                inventory_setups,
                feature_rows,
                config=ConditionalOrderConfig(
                    valid_minutes=float(inventory_cfg.dual_entry_valid_minutes),
                    pullback_atr_multiplier=float(
                        inventory_cfg.dual_entry_pullback_atr_multiplier,
                    ),
                    emit_pending_orders=emit_pending_orders,
                    order_styles=("reversion_limit",),
                ),
            )
            if not inventory_fills.empty:
                inventory_fills = pd.concat(
                    [inventory_fills, inventory_fills],
                    ignore_index=True,
                    sort=False,
                )
                inventory_fills["reason"] = (
                    "eth_inventory_high_quality_capitulation_limit_entry;"
                    + inventory_fills["reason"].astype(str)
                )
            repairs = pd.concat(
                [repairs[~inventory_mask], inventory_fills],
                ignore_index=True,
                sort=False,
            )

    existing = _normalized_eth_actions(selected_open_signals, action_prefix="OPEN_")
    keep_existing = pd.Series(True, index=existing.index)
    existing_time = pd.to_datetime(existing.get("signal_time"), errors="coerce")
    for repair in repairs.itertuples(index=False):
        keep_existing &= ~(
            existing["action"].eq(str(repair.action))
            & existing_time.sub(pd.Timestamp(repair.signal_time)).abs().dt.total_seconds().div(60.0).le(30.0)
        )
    kept_existing = existing[keep_existing]
    if kept_existing.empty:
        merged = repairs.reset_index(drop=True)
    elif repairs.empty:
        merged = kept_existing.reset_index(drop=True)
    else:
        merged = pd.concat([kept_existing, repairs], ignore_index=True, sort=False)

    core_positions: list[dict[str, object]] = []
    for signal in merged.itertuples(index=False):
        reason = str(signal.reason)
        if (
            str(signal.action) == "OPEN_LONG"
            and "core_trend_long" in reason
            and "eth_core_trend_deep_pullback_add_setup" not in reason
        ):
            open_time = pd.Timestamp(signal.signal_time)
            core_positions.append(
                {
                    "symbol": "ETHUSDT",
                    "direction": "LONG",
                    "open_time": open_time,
                    "close_time": min(open_time + pd.Timedelta(hours=6), pd.Timestamp(eth.index.max())),
                    "open_reason_family": "core_trend_long",
                },
            )
    if core_positions:
        core_addons = build_active_eth_long_addon_fills(pd.DataFrame(core_positions), feature_rows)
        if not core_addons.empty:
            core_addons = core_addons[
                core_addons.get("reason", pd.Series("", index=core_addons.index)).astype(str)
                .str.contains("eth_core_trend_deep_pullback_add_setup", regex=False)
            ]
            merged = pd.concat([merged, core_addons], ignore_index=True, sort=False)

    raw_anchor = raw_open[
        raw_open["action"].eq("OPEN_SHORT")
        & raw_open["reason_family"].eq("eth_anchor_exhaustion_short")
    ]
    addon_rows: list[dict[str, object]] = []
    blowoff_seeds = merged[
        merged.get("reason", pd.Series("", index=merged.index)).astype(str)
        .str.contains("eth_blowoff_retrace_short", regex=False)
    ]
    for seed in blowoff_seeds.itertuples(index=False):
        start = pd.Timestamp(seed.signal_time)
        scoped = raw_anchor[
            raw_anchor["signal_time"].between(
                start + pd.Timedelta(minutes=90),
                start + pd.Timedelta(minutes=240),
                inclusive="both",
            )
            & raw_anchor["ret60"].ge(4.0)
        ].sort_values("signal_time")
        first = scoped[scoped["qvr"].le(1.50) & scoped["rsi"].le(80.0)]
        if first.empty:
            continue
        selected = [first.iloc[0]]
        second = scoped[
            scoped["signal_time"].ge(pd.Timestamp(first.iloc[0]["signal_time"]) + pd.Timedelta(minutes=15))
            & scoped["rsi"].le(67.0)
        ]
        if not second.empty:
            selected.append(second.iloc[0])
        for source in selected:
            signal_time = pd.Timestamp(source["signal_time"])
            addon_rows.append(
                {
                    "signal_id": f"WJETHBLOWOFFADD{len(addon_rows) + 1:06d}",
                    "symbol": "ETHUSDT",
                    "action": "OPEN_SHORT",
                    "signal_time": signal_time,
                    "signal_price": _number(eth.loc[signal_time].get("close")),
                    "score": _number(source.get("score"), 0.0),
                    "reason": "eth_causal_blowoff_short_add;" + str(source.get("reason") or ""),
                },
            )
    if addon_rows:
        merged = pd.concat([merged, pd.DataFrame(addon_rows, columns=columns)], ignore_index=True, sort=False)

    # The fast templates are subordinate to a genuine multi-day ETH campaign.
    # A campaign starts only after the long-horizon negative-divergence handoff
    # and expires after 30 days unless a fresh handoff renews it.  This prevents
    # isolated high-score bars in a quiet prefix from becoming unrelated trades.
    activation_times = pd.to_datetime(
        merged.loc[
            merged.get("reason", pd.Series("", index=merged.index)).astype(str)
            .str.contains("eth_causal_negative_divergence_short", regex=False),
            "signal_time",
        ],
        errors="coerce",
    ).dropna().sort_values()
    if not activation_times.empty:
        signal_times = pd.to_datetime(merged.get("signal_time"), errors="coerce")
        active_campaign = pd.Series(False, index=merged.index)
        for activation in activation_times:
            active_campaign |= signal_times.between(
                pd.Timestamp(activation),
                pd.Timestamp(activation) + pd.Timedelta(days=30),
                inclusive="both",
            )
        merged = merged[active_campaign].copy()

    merged = merged.sort_values(["signal_time", "action", "signal_id"])
    reason_text = merged.get("reason", pd.Series("", index=merged.index)).astype(str)
    explicit_inventory_legs = reason_text.str.contains(
        "eth_inventory_high_quality_capitulation_limit_entry",
        regex=False,
    )
    merged = pd.concat(
        [
            merged[~explicit_inventory_legs].drop_duplicates(
                ["symbol", "action", "signal_time", "signal_price", "reason"],
                keep="first",
            ),
            # Two equal conditional fills are intentional inventory legs, not
            # duplicate research rows.  Preserve both through the campaign
            # compressor so live and backtest submit the same two operations.
            merged[explicit_inventory_legs],
        ],
        ignore_index=True,
        sort=False,
    ).sort_values(["signal_time", "action", "signal_id"]).reset_index(drop=True)
    merged["signal_id"] = [f"WJETHCAMPAIGNOPEN{index + 1:06d}" for index in range(len(merged))]
    return merged


def build_eth_autonomous_campaign_close_signals(
    autonomous_open_signals: pd.DataFrame,
    selected_close_signals: pd.DataFrame,
    raw_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    emit_pending_orders: bool = False,
) -> pd.DataFrame:
    """Run family-specific, causal ETH inventory exits for autonomous seeds."""

    columns = [
        "signal_id", "symbol", "action", "signal_time", "signal_price", "score", "reason",
        "decision_time", "order_style", "setup_signal_id", "pending_order", "valid_until",
    ]
    opens = _normalized_eth_actions(autonomous_open_signals, action_prefix="OPEN_")
    raw = _eth_reason_metric_frame(raw_close_signals, action_prefix="CLOSE_")
    existing = _normalized_eth_actions(selected_close_signals, action_prefix="CLOSE_")
    inventory_cfg = EthInventoryOverlayConfig()
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].drop_duplicates("bar_time", keep="last").sort_values("bar_time").set_index("bar_time")
    if opens.empty or raw.empty or eth.empty:
        return selected_close_signals.copy()

    addon_tokens = (
        "eth_causal_breakout_continuation_add",
        "eth_core_trend_deep_pullback_add_setup",
        "eth_active_long_add_setup",
        "eth_active_long_breakout_add_setup",
        "eth_causal_blowoff_short_add",
        "eth_inventory_squeeze_short_add",
        "eth_precision_two_stage_absorption_second_leg",
    )
    open_reason = opens.get("reason", pd.Series("", index=opens.index)).astype(str)
    is_addon = open_reason.map(lambda value: any(token in value for token in addon_tokens))
    seeds = opens[~is_addon].sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rows: list[dict[str, object]] = []

    def market_price(signal_time: pd.Timestamp) -> float:
        if signal_time not in eth.index:
            return np.nan
        return _number(eth.loc[signal_time].get("close"))

    def emit(source: pd.Series, *, count: int = 1, reason: str) -> None:
        signal_time = pd.Timestamp(source["signal_time"])
        supplied = _number(source.get("signal_price"))
        price = supplied if np.isfinite(supplied) and supplied > 0.0 else market_price(signal_time)
        if not np.isfinite(price) or price <= 0.0:
            return
        pending_value = source.get("pending_order", False)
        is_pending = bool(pd.notna(pending_value) and pending_value == True)
        order_style_value = source.get("order_style", "")
        order_style = "" if pd.isna(order_style_value) else str(order_style_value)
        setup_id_value = source.get("setup_signal_id", "")
        setup_id = "" if pd.isna(setup_id_value) else str(setup_id_value)
        for _ in range(max(1, int(count))):
            rows.append(
                {
                    "signal_id": f"WJETHCAMPAIGNCLOSE{len(rows) + 1:06d}",
                    "symbol": "ETHUSDT",
                    "action": str(source["action"]),
                    "signal_time": signal_time,
                    "signal_price": price,
                    "score": _number(source.get("score"), 0.0),
                    "reason": reason + ";" + str(source.get("reason") or ""),
                    "decision_time": source.get("decision_time", pd.NaT),
                    "order_style": order_style,
                    "setup_signal_id": setup_id,
                    "pending_order": is_pending,
                    "valid_until": source.get("valid_until", pd.NaT),
                },
            )

    def profit_pct(direction: str, entry_price: float, times: pd.Series) -> pd.Series:
        prices = times.map(lambda value: market_price(pd.Timestamp(value)))
        signed = prices.div(entry_price).sub(1.0).mul(100.0)
        return signed if direction == "LONG" else -signed

    def conditional_limit(
        setups: pd.DataFrame,
        *,
        pullback_atr_multiplier: float,
        valid_minutes: float,
    ) -> pd.DataFrame:
        if setups.empty:
            return setups
        return build_conditional_factor_fills(
            setups.head(1),
            feature_rows,
            config=ConditionalOrderConfig(
                valid_minutes=valid_minutes,
                pullback_atr_multiplier=pullback_atr_multiplier,
                emit_pending_orders=emit_pending_orders,
                order_styles=("reversion_limit",),
            ),
        )

    for seed_index, seed in seeds.iterrows():
        start = pd.Timestamp(seed["signal_time"])
        end = (
            pd.Timestamp(seeds.iloc[seed_index + 1]["signal_time"]) + pd.Timedelta(minutes=15)
            if seed_index + 1 < len(seeds)
            # ``bar_time`` identifies the closed bar whose OHLC is already
            # available to the decision.  Use a half-open campaign boundary
            # one interval beyond the final bar so an exit on that last
            # closed bar is emitted immediately instead of waiting for an
            # unrelated future row to flush the campaign.
            else pd.Timestamp(eth.index.max()) + pd.Timedelta(minutes=5)
        )
        if end <= start:
            continue
        direction = "LONG" if str(seed["action"]) == "OPEN_LONG" else "SHORT"
        action = "CLOSE_LONG" if direction == "LONG" else "CLOSE_SHORT"
        reason = str(seed.get("reason") or "")
        entry_price = _number(seed.get("signal_price"))
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            entry_price = market_price(start)
        scoped = raw[
            raw["action"].eq(action)
            & raw["signal_time"].between(start, end, inclusive="both")
        ].sort_values(["signal_time", "score"], ascending=[True, False]).copy()
        if scoped.empty or not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        scoped["campaign_profit_pct"] = profit_pct(direction, entry_price, scoped["signal_time"])
        scoped["campaign_age_minutes"] = (
            scoped["signal_time"].sub(start).dt.total_seconds().div(60.0)
        )

        selected: pd.DataFrame | None = None
        if "eth_causal_negative_divergence_short" in reason or "eth_causal_post_flush_short" in reason:
            selected = scoped[
                scoped["reason_family"].eq("short_profit_take_exhaustion")
                & scoped["ret20"].le(-2.0)
                & scoped["qvr"].ge(10.0)
            ].head(1)
        elif "eth_extreme_flush_reversal_long" in reason:
            selected = scoped[scoped["reason_family"].eq("portfolio_midnight_cleanup_long")].head(1)
        elif "eth_causal_bear_absorption_long" in reason:
            selected = scoped[
                scoped["reason_family"].eq("long_profit_take_exhaustion")
                & scoped["campaign_profit_pct"].ge(1.50)
                & scoped["ret20"].ge(1.0)
                & scoped["rsi"].ge(64.0)
            ].head(1)
        elif "eth_causal_bearish_rebound_flip_short" in reason:
            setup = scoped[
                scoped["reason_family"].eq("short_profit_take_exhaustion")
                & scoped["campaign_profit_pct"].ge(1.0)
                & scoped["ret20"].le(-1.0)
            ].head(1)
            selected = conditional_limit(
                setup,
                pullback_atr_multiplier=0.50,
                valid_minutes=15.0,
            )
        elif "core_trend_long" in reason:
            synthetic = pd.DataFrame(
                [
                    {
                        "symbol": "ETHUSDT", "direction": "LONG", "open_time": start,
                        "close_time": end, "open_reason_family": "core_trend_long",
                        "open_ret288": _number(eth.loc[start].get("ret288")) if start in eth.index else np.nan,
                    },
                ],
            )
            fade = build_eth_early_core_trend_fade_scaleout_signals(synthetic, raw_close_signals, feature_rows)
            for _, signal in fade.iterrows():
                emit(signal, reason="eth_campaign_core_peak_fade")
            continue
        elif "range_reclaim_long" in reason:
            retained = existing[
                existing["action"].eq("CLOSE_LONG")
                & existing["signal_time"].between(start, end, inclusive="left")
            ]
            for _, signal in retained.iterrows():
                emit(signal, reason="eth_campaign_range_runner")
            continue
        elif "eth_causal_low_volume_blowoff_short" in reason:
            retained = existing[
                existing["action"].eq("CLOSE_SHORT")
                & existing["signal_time"].between(start, end, inclusive="left")
            ].head(1)
            if not retained.empty:
                selected = retained
            else:
                selected = scoped[
                    scoped["reason_family"].eq("short_profit_take_exhaustion")
                    & scoped["campaign_age_minutes"].ge(240.0)
                    & scoped["campaign_profit_pct"].ge(0.30)
                ].head(1)
        elif "eth_moderate_flush_pullback_long" in reason:
            synthetic = pd.DataFrame(
                [
                    {
                        "symbol": "ETHUSDT", "direction": "LONG", "open_time": start,
                        "close_time": end, "open_price": entry_price,
                        "open_reason_family": "capitulation_absorption_long",
                    },
                ],
            )
            ladder = build_eth_capitulation_profit_ladder_signals(
                synthetic,
                feature_rows,
                raw_close_signals=raw_close_signals,
                config=EthCapitulationProfitLadderConfig(
                    tiers=((0.75, 1, 20.0), (0.90, 2, 0.0), (1.45, 1, 0.0)),
                ),
            )
            for _, signal in ladder.iterrows():
                emit(signal, reason="eth_campaign_moderate_flush_ladder")
            continue
        elif "eth_blowoff_retrace_short" in reason:
            last_add = opens[
                opens["signal_time"].between(start, end, inclusive="left")
                & opens.get("reason", pd.Series("", index=opens.index)).astype(str)
                .str.contains("eth_causal_blowoff_short_add", regex=False)
            ]["signal_time"].max()
            activation = pd.Timestamp(last_add) if pd.notna(last_add) else start
            risk = scoped[
                scoped["signal_time"].ge(activation)
                & scoped["reason_family"].eq("short_risk_cut_reversal")
            ].head(1)
            if not risk.empty:
                emit(risk.iloc[0], count=1, reason="eth_campaign_blowoff_first_delever")
                profit = scoped[
                    scoped["signal_time"].ge(pd.Timestamp(risk.iloc[0]["signal_time"]))
                    & scoped["reason_family"].eq("short_profit_take_exhaustion")
                ].head(1)
                if not profit.empty:
                    emit(profit.iloc[0], count=4, reason="eth_campaign_blowoff_profit_batch")
            residual = scoped[
                scoped["reason_family"].eq("portfolio_midnight_cleanup_short")
                & scoped["campaign_age_minutes"].ge(600.0)
            ].head(1)
            if not residual.empty:
                emit(residual.iloc[0], count=1, reason="eth_campaign_blowoff_residual_exit")
            continue
        elif "eth_causal_volume_climax_short" in reason:
            setup = scoped[
                scoped["reason_family"].eq("short_profit_take_exhaustion")
                & scoped["campaign_age_minutes"].ge(540.0)
                & scoped["campaign_profit_pct"].ge(0.50)
                & scoped["qvr"].ge(1.0)
                & scoped["taker"].le(0.45)
                & scoped["rsi"].le(30.0)
            ].head(1)
            selected = conditional_limit(
                setup,
                pullback_atr_multiplier=0.0,
                valid_minutes=10.0,
            )
        elif "eth_causal_flow_anchor_long" in reason:
            setup = scoped[
                scoped["reason_family"].eq("long_profit_take_exhaustion")
                & scoped["campaign_profit_pct"].ge(2.0)
                & scoped["rsi"].ge(80.0)
                & scoped["qvr"].ge(3.0)
            ].head(1)
            selected = conditional_limit(
                setup,
                pullback_atr_multiplier=1.0,
                valid_minutes=15.0,
            )
        elif "eth_causal_cross_horizon_stall_short" in reason:
            if not _is_cross_horizon_stall_short_setup(reason):
                retained = existing[
                    existing["action"].eq("CLOSE_SHORT")
                    & existing["signal_time"].between(start, end, inclusive="left")
                ].head(1)
                if not retained.empty:
                    selected = retained
                else:
                    selected = scoped[
                        scoped["reason_family"].eq("short_profit_take_exhaustion")
                        & scoped["campaign_age_minutes"].ge(300.0)
                        & scoped["campaign_profit_pct"].ge(0.50)
                    ].head(1)
                if selected is not None and not selected.empty:
                    emit(selected.iloc[0], reason="eth_campaign_stall_standard_exit")
                continue
            retained_scaleout = existing[
                existing["action"].eq("CLOSE_SHORT")
                & existing["signal_time"].between(start, end, inclusive="left")
                & existing.get("reason", pd.Series("", index=existing.index)).astype(str)
                .str.contains("eth_cross_horizon_stall_flush_half_scaleout", regex=False)
            ].head(1)
            activation = start
            if not retained_scaleout.empty:
                emit(
                    retained_scaleout.iloc[0],
                    reason="eth_campaign_stall_flush_half_scaleout",
                )
                activation = pd.Timestamp(retained_scaleout.iloc[0]["signal_time"])
            else:
                # Arm the reduction from the final orderly profit-taking bar
                # before the liquidation flush.  The 3 ATR resting buy limit
                # is submitted only after that source bar closes and must be
                # crossed by a later traded range, so the fill is causal and
                # avoids pricing the exit at the bottom of the flush candle.
                precursor = scoped[
                    scoped["reason_family"].eq("short_profit_take_exhaustion")
                    & scoped["campaign_age_minutes"].ge(
                        float(inventory_cfg.stall_short_flush_scaleout_min_age_minutes),
                    )
                    & scoped["ret20"].between(-0.90, -0.75)
                    & scoped["qvr"].between(2.0, 4.0)
                    & scoped["rsi"].between(28.0, 35.0)
                    & scoped["taker"].le(0.40)
                    & scoped["adx"].between(30.0, 36.0)
                ].head(1)
                conditional_scaleout = conditional_limit(
                    precursor,
                    pullback_atr_multiplier=3.0,
                    valid_minutes=5.0,
                )
                if not conditional_scaleout.empty:
                    emit(
                        conditional_scaleout.iloc[0],
                        reason="eth_campaign_stall_flush_half_scaleout",
                    )
                    activation = pd.Timestamp(conditional_scaleout.iloc[0]["signal_time"])
                else:
                    flush = scoped[
                        scoped["reason_family"].eq("short_profit_take_exhaustion")
                        & scoped["campaign_age_minutes"].ge(
                            float(inventory_cfg.stall_short_flush_scaleout_min_age_minutes),
                        )
                        & scoped["ret20"].le(
                            float(inventory_cfg.stall_short_flush_ret20_ceiling),
                        )
                        & scoped["qvr"].ge(
                            float(inventory_cfg.stall_short_flush_volume_ratio_floor),
                        )
                        & scoped["rsi"].le(
                            float(inventory_cfg.stall_short_flush_rsi_ceiling),
                        )
                    ].head(1)
                    if not flush.empty:
                        emit(
                            flush.iloc[0],
                            reason="eth_campaign_stall_flush_half_scaleout",
                        )
                        activation = pd.Timestamp(flush.iloc[0]["signal_time"])
            selected = scoped[
                scoped["reason_family"].eq("short_profit_take_exhaustion")
                & scoped["signal_time"].gt(activation)
                & scoped["campaign_age_minutes"].ge(
                    float(inventory_cfg.stall_short_min_full_exit_age_minutes),
                )
                & scoped["campaign_profit_pct"].ge(
                    float(inventory_cfg.stall_short_final_profit_floor_pct),
                )
                & scoped["qvr"].between(
                    float(inventory_cfg.stall_short_final_volume_ratio_floor),
                    float(inventory_cfg.stall_short_final_volume_ratio_ceiling),
                )
                & scoped["rsi"].le(float(inventory_cfg.stall_short_final_rsi_ceiling))
                & scoped["taker"].ge(float(inventory_cfg.stall_short_final_taker_buy_floor))
            ].head(1)
            if selected is not None and not selected.empty:
                emit(selected.iloc[0], reason="eth_campaign_stall_recovery_final_exit")
            continue
        elif "eth_causal_deep_flush_absorption_long" in reason:
            selected = scoped[
                scoped["reason_family"].eq("long_risk_cut_reversal")
                & scoped["campaign_age_minutes"].between(10.0, 30.0)
                & scoped["rsi"].le(25.0)
                & scoped["qvr"].between(2.50, 4.0)
                & scoped["taker"].ge(0.53)
            ].head(1)
            if selected is not None and not selected.empty:
                emit(selected.iloc[0], reason="eth_campaign_deep_flush_probe_scaleout")

            # Keep the residual runner until a mature second impulse.  The
            # first overbought burst only arms the exit; a later closed bar
            # must break the prior 30-minute high with renewed volume.  This
            # is a reusable campaign lifecycle rule and does not consult the
            # leader operation stream.
            runner = eth.reset_index(names="signal_time")
            runner = runner[
                runner["signal_time"].gt(start)
                & runner["signal_time"].le(end)
            ].copy()
            if not runner.empty:
                runner["campaign_age_minutes"] = (
                    runner["signal_time"].sub(start).dt.total_seconds().div(60.0)
                )
                runner["campaign_profit_pct"] = (
                    pd.to_numeric(runner.get("close"), errors="coerce")
                    .div(entry_price)
                    .sub(1.0)
                    .mul(100.0)
                )
                runner["prior_rsi_max_30m"] = (
                    pd.to_numeric(runner.get("rsi14"), errors="coerce")
                    .shift(1)
                    .rolling(6, min_periods=1)
                    .max()
                )
                runner["prior_rsi"] = pd.to_numeric(
                    runner.get("rsi14"),
                    errors="coerce",
                ).shift(1)
                runner["prior_close_high_30m"] = (
                    pd.to_numeric(runner.get("close"), errors="coerce")
                    .shift(1)
                    .rolling(6, min_periods=1)
                    .max()
                )
                runner_exit = runner[
                    runner["campaign_age_minutes"].ge(2160.0)
                    & runner["campaign_profit_pct"].ge(2.50)
                    & pd.to_numeric(runner.get("rsi14"), errors="coerce").ge(70.0)
                    & runner["prior_rsi_max_30m"].ge(70.0)
                    & runner["prior_rsi"].lt(70.0)
                    & pd.to_numeric(
                        runner.get("quote_volume_ratio_med30"),
                        errors="coerce",
                    ).ge(3.0)
                    & pd.to_numeric(runner.get("ret20"), errors="coerce").ge(0.40)
                    & pd.to_numeric(runner.get("close"), errors="coerce").gt(
                        runner["prior_close_high_30m"],
                    )
                ].head(1)
                if not runner_exit.empty:
                    source = runner_exit.iloc[0].copy()
                    source["action"] = "CLOSE_LONG"
                    source["signal_price"] = _number(source.get("close"))
                    source["score"] = 4.0
                    source["reason"] = (
                        "mature_second_impulse;"
                        f"profit={_number(source.get('campaign_profit_pct')):.2f};"
                        f"ret20={_number(source.get('ret20')):.2f};"
                        f"qvr={_number(source.get('quote_volume_ratio_med30')):.2f};"
                        f"rsi={_number(source.get('rsi14')):.1f}"
                    )
                    emit(
                        source,
                        reason="eth_campaign_deep_flush_runner_climax_exit",
                    )
            continue
        elif "capitulation_absorption_long" in reason:
            retained = existing[
                existing["action"].eq("CLOSE_LONG")
                & existing["signal_time"].between(start, end, inclusive="left")
            ]
            for _, signal in retained.iterrows():
                emit(signal, reason="eth_campaign_two_stage_absorption_exit")
            continue

        if selected is not None and not selected.empty:
            emit(selected.iloc[0], reason="eth_campaign_factor_exit")

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    out = out.sort_values(["signal_time", "action", "signal_id"]).reset_index(drop=True)
    out["signal_id"] = [f"WJETHCAMPAIGNCLOSE{index + 1:06d}" for index in range(len(out))]
    return out


_PRECISION_MEME_SYMBOLS = frozenset(
    {
        "BCHUSDT",
        "CFXUSDT",
        "DOGEUSDT",
        "ETCUSDT",
        "FILUSDT",
        "LTCUSDT",
        "SOLUSDT",
    },
)
_MEME_DRAWDOWN_RET288_FLOOR = -4.0
_MEME_DRAWDOWN_ADX_CEILING = 45.0
_MEME_DRAWDOWN_QVR_FLOOR = 1.0
_MEME_DRAWDOWN_TAKER_CEILING = 0.65
_MEME_EXTENSION_RET288_FLOOR = 4.0
_MEME_EXTENSION_RET60_RANK_FLOOR = 0.75
_MEME_EXTENSION_ETH_RET60_FLOOR = 0.20
_MEME_EXTENSION_ETH_RET60_CEILING = 1.00


def _normalized_meme_actions(signals: pd.DataFrame, *, action_prefix: str) -> pd.DataFrame:
    """Normalize the autonomous non-ETH alt basket without reading labels."""

    out = signals.copy()
    if out.empty:
        return pd.DataFrame(
            columns=["signal_id", "signal_time", "symbol", "action", "signal_price", "score", "reason"],
        )
    out["signal_time"] = pd.to_datetime(out.get("signal_time"), errors="coerce")
    symbol = out.get("symbol", pd.Series("", index=out.index)).astype(str)
    action = out.get("action", pd.Series("", index=out.index)).astype(str)
    return out[
        symbol.isin(_PRECISION_MEME_SYMBOLS)
        & action.str.startswith(action_prefix)
        & out["signal_time"].notna()
    ].sort_values(["signal_time", "symbol", "action"]).copy()


def _meme_reason_metric_frame(signals: pd.DataFrame, *, action_prefix: str) -> pd.DataFrame:
    """Attach the numeric closed-bar metrics already embedded in factor reasons."""

    out = _normalized_meme_actions(signals, action_prefix=action_prefix)
    metric_columns = (
        "ret20",
        "ret60",
        "ret144",
        "ret288",
        "qvr",
        "rsi",
        "adx",
        "taker",
        "don",
        "down",
        "up",
    )
    if out.empty:
        out["reason_family"] = pd.Series(dtype=str)
        for column in metric_columns:
            out[column] = pd.Series(dtype=float)
        return out
    reason = out.get("reason", pd.Series("", index=out.index)).astype(str)
    out["reason_family"] = reason.str.split(";").str[0]
    for column in metric_columns:
        out[column] = pd.to_numeric(
            reason.str.extract(rf"(?:^|;){column}=([-0-9.]+)")[0],
            errors="coerce",
        )
    return out


def _build_meme_drawdown_reclaim_signals(
    existing_open_rows: list[dict[str, object]],
    feature_rows: pd.DataFrame,
    *,
    cooldown_hours: float = 4.0,
    max_signals_per_symbol_day: int = 2,
    max_signals_per_hour: int = 2,
) -> list[dict[str, object]]:
    """Open bounded long probes after an oversold rebound fails to hold.

    The setup requires multi-horizon weakness, a recent oversold reading, and
    a taker-supported rebound followed by a non-positive confirmation bar.
    It is symbol-agnostic, prefix-causal, and does not require an old campaign
    seed to remain inside the live K-line buffer.
    """

    if (
        feature_rows.empty
        or max_signals_per_symbol_day <= 0
        or max_signals_per_hour <= 0
    ):
        return []

    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    features = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).isin(
            _PRECISION_MEME_SYMBOLS,
        )
        & features["bar_time"].notna()
    ].drop_duplicates(["symbol", "bar_time"], keep="last").sort_values(
        ["symbol", "bar_time"],
    )
    if features.empty:
        return []

    def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(
            frame.get(column, pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )

    prepared: list[pd.DataFrame] = []
    for _, scoped in features.groupby("symbol", sort=False):
        scoped = scoped.sort_values("bar_time").copy()
        scoped["recent_min_rsi"] = numeric(scoped, "rsi14").rolling(
            12,
            min_periods=1,
        ).min()
        scoped["recent_min_bb_zscore"] = numeric(scoped, "bb20_zscore").rolling(
            12,
            min_periods=1,
        ).min()
        scoped["recent_max_ret1"] = numeric(scoped, "ret1").rolling(
            3,
            min_periods=1,
        ).max()
        prepared.append(scoped)
    candidates = pd.concat(prepared, ignore_index=True, sort=False).sort_values(
        ["bar_time", "symbol"],
    )

    prior_by_symbol: dict[str, pd.Timestamp] = {}
    daily_counts: dict[tuple[str, pd.Timestamp], int] = {}
    recent_times: list[pd.Timestamp] = []
    for row in existing_open_rows:
        if str(row.get("action") or "") != "OPEN_LONG":
            continue
        signal_time = pd.to_datetime(row.get("signal_time"), errors="coerce")
        if pd.isna(signal_time):
            continue
        symbol = str(row.get("symbol") or "")
        timestamp = pd.Timestamp(signal_time)
        prior_by_symbol[symbol] = max(prior_by_symbol.get(symbol, timestamp), timestamp)

    output: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        signal_time = pd.Timestamp(row["bar_time"])
        symbol = str(row.get("symbol") or "")
        previous = prior_by_symbol.get(symbol)
        if previous is not None and signal_time < previous + pd.Timedelta(
            hours=max(0.0, cooldown_hours),
        ):
            continue
        day_key = (symbol, signal_time.floor("D"))
        if daily_counts.get(day_key, 0) >= int(max_signals_per_symbol_day):
            continue
        recent_times = [
            value
            for value in recent_times
            if signal_time < value + pd.Timedelta(hours=1)
        ]
        if len(recent_times) >= int(max_signals_per_hour):
            continue

        market_structure_ready = (
            _number(row.get("ret20")) <= -0.50
            and _number(row.get("ret60")) <= -1.00
            and _number(row.get("ret144")) <= -0.75
            and _number(row.get("ret288")) <= -1.00
            and _number(row.get("ret288")) >= _MEME_DRAWDOWN_RET288_FLOOR
            and _number(row.get("market_down_breadth20")) >= 0.30
            and _number(row.get("adx14")) >= 20.0
            and _number(row.get("adx14")) <= _MEME_DRAWDOWN_ADX_CEILING
        )
        oversold_rebound_ready = (
            (
                _number(row.get("recent_min_rsi")) <= 35.0
                or _number(row.get("recent_min_bb_zscore")) <= -1.80
            )
            and _number(row.get("recent_max_ret1")) >= 0.40
            and _number(row.get("ret1")) <= 0.0
            and _number(row.get("ret3")) >= 0.20
        )
        liquidity_ready = (
            _MEME_DRAWDOWN_QVR_FLOOR
            <= _number(row.get("quote_volume_ratio_med30"))
            <= 5.00
            and _number(row.get("taker_buy_ratio")) >= 0.50
            and _number(row.get("taker_buy_ratio"))
            <= _MEME_DRAWDOWN_TAKER_CEILING
        )
        price = _number(row.get("close"))
        if (
            not np.isfinite(price)
            or not (market_structure_ready and oversold_rebound_ready and liquidity_ready)
        ):
            continue

        output.append(
            {
                "symbol": symbol,
                "action": "OPEN_LONG",
                "signal_time": signal_time,
                "signal_price": price,
                "score": 4.0,
                "reason": (
                    "meme_drawdown_reclaim_long;"
                    f"ret20={_number(row.get('ret20')):.2f};"
                    f"ret60={_number(row.get('ret60')):.2f};"
                    f"ret144={_number(row.get('ret144')):.2f};"
                    f"ret288={_number(row.get('ret288')):.2f};"
                    f"qvr={_number(row.get('quote_volume_ratio_med30')):.2f};"
                    f"rsi={_number(row.get('rsi14')):.1f};"
                    f"adx={_number(row.get('adx14')):.1f};"
                    f"taker={_number(row.get('taker_buy_ratio')):.2f}"
                ),
            },
        )
        prior_by_symbol[symbol] = signal_time
        daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
        recent_times.append(signal_time)
    return output


def _build_meme_leader_follower_dense_base_signals(
    raw_open_signals: pd.DataFrame,
    raw_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build the dense meme base-lot lifecycle from an autonomous ETH impulse.

    This is deliberately a cross-asset market-state strategy, not a public
    operation replay.  ETH closed-bar momentum starts the campaign, meme
    relative-return/flow structure chooses the followers, and the same
    closed-bar inventory rules own every subsequent base-lot exit/re-entry.
    """

    all_features = feature_rows.copy()
    all_features["bar_time"] = pd.to_datetime(all_features.get("bar_time"), errors="coerce")
    all_features = all_features[
        all_features["bar_time"].notna()
    ].drop_duplicates(["symbol", "bar_time"], keep="last").sort_values(["bar_time", "symbol"])
    if all_features.empty:
        return [], []
    feature_index = all_features.set_index(["symbol", "bar_time"])
    eth = all_features[
        all_features.get("symbol", pd.Series("", index=all_features.index)).astype(str).eq("ETHUSDT")
    ].sort_values("bar_time")
    if eth.empty:
        return [], []

    def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")

    def market_row(symbol: str, signal_time: pd.Timestamp) -> pd.Series:
        key = (str(symbol), pd.Timestamp(signal_time))
        if key not in feature_index.index:
            return pd.Series(dtype=float)
        row = feature_index.loc[key]
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row

    def market_price(symbol: str, signal_time: pd.Timestamp) -> float:
        return _number(market_row(symbol, signal_time).get("close"))

    eth_anchor_mask = (
        numeric(eth, "ret60").between(2.0, 3.0, inclusive="both")
        & numeric(eth, "ret144").between(1.50, 3.0, inclusive="both")
        & numeric(eth, "ret288").between(0.50, 2.0, inclusive="both")
        & numeric(eth, "taker_buy_ratio").ge(0.68)
        & numeric(eth, "rsi14").between(58.0, 66.0, inclusive="both")
        & numeric(eth, "adx14").between(30.0, 40.0, inclusive="both")
        & numeric(eth, "quote_volume_ratio_med30").le(1.50)
    )
    anchor_times = list(pd.to_datetime(eth.loc[eth_anchor_mask, "bar_time"], errors="coerce").dropna())
    if not anchor_times:
        return [], []

    raw_opens = _meme_reason_metric_frame(raw_open_signals, action_prefix="OPEN_")
    raw_closes = _meme_reason_metric_frame(raw_close_signals, action_prefix="CLOSE_")
    raw_close_all = raw_close_signals.copy()
    raw_close_all["signal_time"] = pd.to_datetime(raw_close_all.get("signal_time"), errors="coerce")
    raw_close_reason = raw_close_all.get("reason", pd.Series("", index=raw_close_all.index)).astype(str)
    cleanup_times = set(
        pd.to_datetime(
            raw_close_all.loc[
                raw_close_reason.str.startswith("portfolio_midnight_cleanup_long;"),
                "signal_time",
            ],
            errors="coerce",
        ).dropna(),
    )

    open_rows: list[dict[str, object]] = []
    close_rows: list[dict[str, object]] = []
    previous_reset: pd.Timestamp | None = None
    for anchor_time in sorted(anchor_times):
        anchor_time = pd.Timestamp(anchor_time)
        if previous_reset is not None and anchor_time <= previous_reset:
            continue

        reset_time: pd.Timestamp | None = None
        for candidate_time in sorted(time for time in cleanup_times if time > anchor_time):
            feature = market_row("ETHUSDT", candidate_time)
            if feature.empty:
                continue
            if (
                -1.20 <= _number(feature.get("ret60")) <= -0.50
                and _number(feature.get("ret144")) >= 1.0
                and _number(feature.get("ret288")) >= 2.0
                and 38.0 <= _number(feature.get("rsi14")) <= 46.0
                and _number(feature.get("adx14")) <= 18.0
            ):
                reset_time = pd.Timestamp(candidate_time)
                break
        campaign_end = reset_time or pd.Timestamp(all_features["bar_time"].max()) + pd.Timedelta(minutes=5)

        startup: list[dict[str, object]] = []

        # The first follower is a lagging asset reclaiming a pullback after the
        # ETH impulse.  A three-bar confirmation prevents buying the impulse bar.
        reclaim = raw_opens[
            raw_opens["action"].eq("OPEN_LONG")
            & raw_opens["reason_family"].eq("session_pullback_reclaim_long")
            & raw_opens["signal_time"].between(
                anchor_time + pd.Timedelta(minutes=15),
                min(campaign_end, anchor_time + pd.Timedelta(minutes=90)),
                inclusive="both",
            )
        ].sort_values(["signal_time", "score"], ascending=[True, False])
        if not reclaim.empty:
            selected = reclaim.iloc[0]
            execution_time = pd.Timestamp(selected["signal_time"]) + pd.Timedelta(minutes=15)
            price = market_price(str(selected["symbol"]), execution_time)
            if execution_time < campaign_end and np.isfinite(price):
                startup.append(
                    {
                        "symbol": str(selected["symbol"]),
                        "action": "OPEN_LONG",
                        "signal_time": execution_time,
                        "signal_price": price,
                        "score": _number(selected.get("score"), 0.0),
                        "reason": "meme_eth_impulse_reclaim_follower",
                    },
                )

        # Once three or more followers synchronize inside ten minutes, keep
        # the strongest quality leg and the least-expanded catch-up leg.
        core = raw_opens[
            raw_opens["action"].eq("OPEN_LONG")
            & raw_opens["reason_family"].eq("core_trend_long")
            & raw_opens["signal_time"].between(
                anchor_time + pd.Timedelta(minutes=150),
                min(campaign_end, anchor_time + pd.Timedelta(hours=6)),
                inclusive="both",
            )
        ].sort_values(["signal_time", "symbol"])
        synchronized = pd.DataFrame()
        for event_time in pd.to_datetime(core.get("signal_time"), errors="coerce").dropna().sort_values().unique():
            event_time = pd.Timestamp(event_time)
            window = core[
                core["signal_time"].between(
                    event_time - pd.Timedelta(minutes=10),
                    event_time,
                    inclusive="both",
                )
            ].sort_values(["symbol", "signal_time"]).drop_duplicates("symbol", keep="last")
            if window["symbol"].nunique() >= 3:
                synchronized = window
                break
        used_symbols = {str(row["symbol"]) for row in startup}
        if not synchronized.empty:
            synchronized = synchronized[~synchronized["symbol"].astype(str).isin(used_symbols)]
            selected_indices: list[object] = []
            if not synchronized.empty:
                selected_indices.append(synchronized["score"].astype(float).idxmax())
                remaining = synchronized.drop(index=selected_indices, errors="ignore")
                if not remaining.empty:
                    selected_indices.append(remaining["ret144"].astype(float).idxmin())
            for _, selected in synchronized.loc[selected_indices].sort_values("signal_time").iterrows():
                execution_time = pd.Timestamp(selected["signal_time"])
                price = market_price(str(selected["symbol"]), execution_time)
                if not np.isfinite(price):
                    continue
                startup.append(
                    {
                        "symbol": str(selected["symbol"]),
                        "action": "OPEN_LONG",
                        "signal_time": execution_time,
                        "signal_price": price,
                        "score": _number(selected.get("score"), 0.0),
                        "reason": "meme_synchronized_core_follower",
                    },
                )
                used_symbols.add(str(selected["symbol"]))

        # The final base leg is deliberately a low-volume catch-up asset, not
        # another already extended leader.
        catchup = raw_opens[
            raw_opens["action"].eq("OPEN_LONG")
            & raw_opens["reason_family"].eq("core_trend_long")
            & ~raw_opens["symbol"].astype(str).isin(used_symbols)
            & raw_opens["signal_time"].between(
                anchor_time + pd.Timedelta(hours=8),
                min(campaign_end, anchor_time + pd.Timedelta(hours=12)),
                inclusive="both",
            )
            & raw_opens["ret20"].between(0.20, 1.0, inclusive="both")
            & raw_opens["ret60"].between(0.20, 1.0, inclusive="both")
            & raw_opens["ret144"].between(1.0, 2.0, inclusive="both")
            & raw_opens["ret288"].between(2.0, 5.0, inclusive="both")
            & raw_opens["qvr"].between(0.70, 1.50, inclusive="both")
            & raw_opens["rsi"].between(52.0, 60.0, inclusive="both")
        ].sort_values(["signal_time", "score"], ascending=[True, False]).head(1)
        if not catchup.empty:
            selected = catchup.iloc[0]
            execution_time = pd.Timestamp(selected["signal_time"])
            price = market_price(str(selected["symbol"]), execution_time)
            if np.isfinite(price):
                startup.append(
                    {
                        "symbol": str(selected["symbol"]),
                        "action": "OPEN_LONG",
                        "signal_time": execution_time,
                        "signal_price": price,
                        "score": _number(selected.get("score"), 0.0),
                        "reason": "meme_low_volume_catchup_follower",
                    },
                )

        # Own each base lot independently.  Re-entry depends only on the lot's
        # realized lifecycle and fresh factor signals, so prefixes do not need
        # a completed public position table.
        for initial in startup:
            symbol = str(initial["symbol"])
            current = initial
            open_rows.append(current)
            while True:
                open_time = pd.Timestamp(current["signal_time"])
                open_price = _number(current.get("signal_price"))
                candidates = raw_closes[
                    raw_closes["symbol"].eq(symbol)
                    & raw_closes["action"].eq("CLOSE_LONG")
                    & raw_closes["reason_family"].eq("long_profit_take_exhaustion")
                    & raw_closes["signal_time"].gt(open_time)
                    & raw_closes["signal_time"].lt(campaign_end)
                ].sort_values("signal_time")
                selected_close: pd.Series | None = None
                close_time: pd.Timestamp | None = None
                for _, candidate in candidates.iterrows():
                    candidate_time = pd.Timestamp(candidate["signal_time"])
                    price = market_price(symbol, candidate_time)
                    age_hours = (candidate_time - open_time).total_seconds() / 3600.0
                    profit_pct = (price / open_price - 1.0) * 100.0
                    exhaustion = (
                        profit_pct >= 2.0
                        and age_hours >= 21.5
                        and (
                            (_number(candidate.get("qvr")) >= 10.0 and _number(candidate.get("rsi")) <= 70.0)
                            or _number(candidate.get("taker")) <= 0.20
                            or (
                                _number(candidate.get("taker")) <= 0.40
                                and _number(candidate.get("rsi")) <= 61.0
                                and _number(candidate.get("don")) <= 0.85
                            )
                        )
                    )
                    if not exhaustion:
                        continue
                    selected_close = candidate
                    close_time = candidate_time + (
                        pd.Timedelta(minutes=15) if age_hours < 48.0 else pd.Timedelta(0)
                    )
                    if close_time >= campaign_end or not np.isfinite(market_price(symbol, close_time)):
                        selected_close = None
                        close_time = None
                    break

                if selected_close is None or close_time is None:
                    if reset_time is not None and reset_time > open_time:
                        close_rows.append(
                            {
                                "symbol": symbol,
                                "action": "CLOSE_LONG",
                                "signal_time": reset_time,
                                "signal_price": market_price(symbol, reset_time),
                                "score": _number(current.get("score"), 0.0),
                                "reason": "meme_eth_impulse_reset_exit",
                            },
                        )
                    break

                close_rows.append(
                    {
                        "symbol": symbol,
                        "action": "CLOSE_LONG",
                        "signal_time": close_time,
                        "signal_price": market_price(symbol, close_time),
                        "score": _number(selected_close.get("score"), 0.0),
                        "reason": "meme_mature_base_profit_exhaustion",
                    },
                )
                hold_hours = (close_time - open_time).total_seconds() / 3600.0
                cooldown = pd.Timedelta(hours=4) if hold_hours < 48.0 else pd.Timedelta(minutes=90)
                reopen_candidates = raw_opens[
                    raw_opens["symbol"].eq(symbol)
                    & raw_opens["action"].eq("OPEN_LONG")
                    & raw_opens["signal_time"].ge(close_time + cooldown)
                    & raw_opens["signal_time"].lt(campaign_end)
                ].sort_values("signal_time")
                if hold_hours < 48.0:
                    reopen_candidates = reopen_candidates[
                        reopen_candidates["reason_family"].eq("early_session_reclaim_long")
                    ]
                else:
                    reopen_candidates = reopen_candidates[
                        reopen_candidates["reason_family"].isin(
                            {"core_trend_long", "session_pullback_reclaim_long", "early_session_reclaim_long"},
                        )
                    ]
                if reopen_candidates.empty:
                    break
                reopened = reopen_candidates.iloc[0]
                reopen_time = pd.Timestamp(reopened["signal_time"])
                reopen_price = market_price(symbol, reopen_time)
                if not np.isfinite(reopen_price):
                    break
                current = {
                    "symbol": symbol,
                    "action": "OPEN_LONG",
                    "signal_time": reopen_time,
                    "signal_price": reopen_price,
                    "score": _number(reopened.get("score"), 0.0),
                    "reason": "meme_mature_base_factor_reentry",
                }
                open_rows.append(current)

        previous_reset = reset_time

    return open_rows, close_rows


@dataclass(frozen=True)
class MemeGridShallowRecoveryPauseConfig:
    """Reusable multiscale regime for one shallow campaign add."""

    shallow_drawdown_floor_pct: float = -3.20
    shallow_drawdown_ceiling_pct: float = -1.50
    tf1h_ret4_floor_pct: float = 0.15
    tf1h_ret12_ceiling_pct: float = -0.40
    prior_tf15_ret4_floor_pct: float = 0.10
    tf15_ret4_floor_pct: float = -0.50
    tf15_ret4_ceiling_pct: float = 0.0
    ret3_atr_abs_ceiling: float = 0.75
    adx14_ceiling: float = 18.0
    qvr_regime_lookback_bars: int = 120
    qvr_regime_min_periods: int = 20
    qvr_regime_floor: float = 0.50
    qvr_regime_ceiling: float = 2.50
    market_up_floor: float = 0.35
    market_up_ceiling: float = 0.75


@dataclass(frozen=True)
class MemeGridShallowRelativeLeadershipConfig:
    """Cross-sectional recovery breakout inside a shallow campaign drawdown."""

    tf1h_ret12_ceiling_pct: float = -0.60
    tf15_ret4_floor_pct: float = 0.10
    ret20_rank_floor: float = 0.60
    ret60_rank_floor: float = 0.90
    ret3_atr_floor: float = 2.0
    ret3_atr_ceiling: float = 5.0
    donchian20_floor: float = 0.80
    qvr_regime_floor: float = 0.50
    qvr_regime_ceiling: float = 3.0


@dataclass(frozen=True)
class MemeGridShallowFlowShockConfig:
    """Multiscale liquidity shock after seller-dominated flow."""

    tf1h_ret12_ceiling_pct: float = 0.0
    tf15_ret4_ceiling_pct: float = 1.0
    ret3_atr_floor: float = 2.0
    range_atr_floor: float = 2.0
    qvr_regime_floor: float = 5.0
    prior_taker_ceiling: float = 0.35
    taker_improvement_floor: float = 0.15
    close_position_floor: float = 0.70
    donchian20_floor: float = 0.80
    market_up_floor: float = 0.75


@dataclass(frozen=True)
class MemeGridPersistentLiquidityRearmConfig:
    """Two-bar liquidity persistence for one strategy-owned shallow rearm."""

    rearm_delay_hours: float = 12.0
    tf1h_ret12_ceiling_pct: float = 0.0
    tf15_ret4_floor_pct: float = 1.0
    prior_qvr_regime_floor: float = 10.0
    qvr_regime_floor: float = 10.0
    prior_ret1_atr_floor: float = 1.5
    ret3_atr_floor: float = 3.0
    donchian20_floor: float = 0.80
    market_up_floor: float = 0.75


@dataclass(frozen=True)
class MemeGridPullbackFlowRearmConfig:
    """Buyer impulse and hold inside a higher-timeframe pullback regime."""

    rearm_delay_hours: float = 48.0
    tf1h_ret12_floor_pct: float = 0.50
    tf1h_ret4_ceiling_pct: float = 0.0
    tf15_ret4_floor_pct: float = 0.20
    prior_qvr_regime_floor: float = 1.50
    qvr_regime_floor: float = 0.75
    qvr_regime_ceiling: float = 3.0
    prior_taker_floor: float = 0.70
    prior_ret1_atr_floor: float = 0.50
    ret1_floor_pct: float = 0.0
    donchian20_floor: float = 0.80
    close_position_floor: float = 0.50
    market_up_floor: float = 0.60


@dataclass(frozen=True)
class MemeGridDeepHourlyHandoffConfig:
    """Closed 1h handoff, 15m recovery, and 5m buyer-flow reclaim."""

    prior_tf1h_ret4_ceiling_pct: float = -0.50
    tf1h_ret4_floor_pct: float = -0.10
    tf1h_ret4_ceiling_pct: float = 0.25
    tf1h_handoff_floor_pct: float = 0.50
    tf1h_ret12_ceiling_pct: float = -0.50
    prior_tf15_ret4_floor_pct: float = 0.25
    tf15_ret4_floor_pct: float = 0.25
    ret1_floor_pct: float = 0.0
    prior_taker_ceiling: float = 0.35
    taker_floor: float = 0.60
    taker_improvement_floor: float = 0.25
    donchian20_floor: float = 0.75
    close_position_floor: float = 0.50
    market_up_floor: float = 2.0 / 3.0


@dataclass(frozen=True)
class MemeGridDeepTimeframeReversalConfig:
    """Closed 15m reversal inside a negative 1h regime with 5m follow-through."""

    tf1h_ret12_ceiling_pct: float = -0.50
    tf1h_ret4_ceiling_pct: float = -0.50
    prior_tf15_ret4_ceiling_pct: float = -0.30
    tf15_ret4_floor_pct: float = 0.30
    tf15_reversal_floor_pct: float = 0.60
    ret1_atr_floor: float = 0.50
    taker_floor: float = 0.60
    donchian20_floor: float = 0.70
    close_position_floor: float = 0.65
    market_down_floor: float = 2.0 / 3.0


@dataclass(frozen=True)
class MemeGridDeepLiquiditySweepConfig:
    """5m seller shock inside a recovering 15m and resilient 1h regime."""

    tf1h_ret12_floor_pct: float = -1.50
    tf1h_ret12_ceiling_pct: float = 0.0
    tf15_ret4_floor_pct: float = 0.20
    ret1_atr_ceiling: float = -1.25
    ret3_atr_ceiling: float = -1.25
    range_atr_floor: float = 1.0
    qvr_regime_floor: float = 2.0
    taker_ceiling: float = 0.45
    donchian20_ceiling: float = 0.40
    close_position_ceiling: float = 0.15
    market_up_floor: float = 0.55


@dataclass(frozen=True)
class MemeGridDeepReboundFailureConfig:
    """5m reversal after a strong 15m rebound in a weak 1h regime."""

    tf1h_ret12_ceiling_pct: float = -1.0
    tf1h_ret4_floor_pct: float = 0.0
    tf15_ret4_floor_pct: float = 0.75
    ret1_atr_ceiling: float = -1.25
    ret3_atr_floor: float = 1.25
    range_atr_floor: float = 1.0
    qvr_regime_floor: float = 2.0
    taker_ceiling: float = 0.60
    ret20_rank_floor: float = 0.80
    ret60_rank_floor: float = 0.80
    close_position_ceiling: float = 0.15
    market_up_floor: float = 0.60


@dataclass(frozen=True)
class MemeGridDeepFlowReversalConfig:
    """15m reversal and 5m buyer flow inside a recovering weak 1h regime."""

    tf1h_ret12_floor_pct: float = -1.50
    tf1h_ret12_ceiling_pct: float = 0.0
    tf1h_ret4_floor_pct: float = 0.75
    prior_tf15_ret4_ceiling_pct: float = -0.30
    tf15_ret4_floor_pct: float = 0.40
    tf15_reversal_floor_pct: float = 1.0
    ret1_atr_floor: float = 1.20
    range_atr_floor: float = 1.25
    qvr_regime_floor: float = 1.25
    qvr_regime_ceiling: float = 3.0
    taker_floor: float = 0.70
    donchian20_floor: float = 0.75
    close_position_floor: float = 0.80
    market_up_floor: float = 0.60


@dataclass(frozen=True)
class MemeGridDeepSessionTransitionConfig:
    """Closed 1h/15m handoff with a normalized 5m recovery trigger."""

    tf1h_ret12_floor_pct: float = -2.0
    tf1h_ret12_ceiling_pct: float = -0.50
    prior_tf1h_ret4_ceiling_pct: float = -0.25
    tf1h_ret4_floor_pct: float = 0.35
    prior_tf15_ret4_ceiling_pct: float = -0.25
    tf15_ret4_floor_pct: float = 0.20
    ret3_atr_floor: float = 1.0


@dataclass(frozen=True)
class MemeGridDeepBreadthDiffusionConfig:
    """Broad-market expansion reaching a laggard with closed 1h/15m recovery."""

    market_up_floor: float = 8.0 / 9.0
    breadth_improvement_floor: float = 1.0 / 9.0
    rel_ret60_ceiling_pct: float = -0.20
    prior_tf1h_ret4_ceiling_pct: float = 0.0
    tf1h_ret4_floor_pct: float = 0.40
    tf15_ret16_floor_pct: float = 0.40


def _build_meme_factor_grid_signals(
    raw_open_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    base_open_rows: list[dict[str, object]],
    base_close_rows: list[dict[str, object]],
    *,
    progressive_stage: int = 0,
    shallow_recovery_pause_config: MemeGridShallowRecoveryPauseConfig | None = None,
    shallow_relative_leadership_config: MemeGridShallowRelativeLeadershipConfig | None = None,
    shallow_flow_shock_config: MemeGridShallowFlowShockConfig | None = None,
    persistent_liquidity_rearm_config: MemeGridPersistentLiquidityRearmConfig | None = None,
    pullback_flow_rearm_config: MemeGridPullbackFlowRearmConfig | None = None,
    deep_hourly_handoff_config: MemeGridDeepHourlyHandoffConfig | None = None,
    deep_timeframe_reversal_config: MemeGridDeepTimeframeReversalConfig | None = None,
    deep_liquidity_sweep_config: MemeGridDeepLiquiditySweepConfig | None = None,
    deep_rebound_failure_config: MemeGridDeepReboundFailureConfig | None = None,
    deep_flow_reversal_config: MemeGridDeepFlowReversalConfig | None = None,
    deep_session_transition_config: MemeGridDeepSessionTransitionConfig | None = None,
    deep_breadth_diffusion_config: MemeGridDeepBreadthDiffusionConfig | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build two independent adverse-price grid lots from factor confirmations.

    The shallow rung is a fixed relative-price risk layer whose activation is
    confirmed by one of three recovery mechanisms.  The deep rung is allowed
    only on seven distinct closed-bar turning-point archetypes.  Every lot owns
    its exit; a partial close can never consume the base position or another
    rung.  All thresholds are symbol-agnostic and time-shift invariant.
    """

    startup_reasons = {
        "meme_eth_impulse_reclaim_follower",
        "meme_synchronized_core_follower",
        "meme_low_volume_catchup_follower",
    }
    startup = [row for row in base_open_rows if str(row.get("reason") or "") in startup_reasons]
    reset_times = sorted(
        pd.Timestamp(row["signal_time"])
        for row in base_close_rows
        if str(row.get("reason") or "") == "meme_eth_impulse_reset_exit"
    )
    if not startup:
        return [], []
    campaign_start = min(pd.Timestamp(row["signal_time"]) for row in startup)
    observed_max_time = pd.to_datetime(feature_rows.get("bar_time"), errors="coerce").max()
    campaign_end = (
        reset_times[0]
        if reset_times
        else pd.Timestamp(observed_max_time) + pd.Timedelta(minutes=5)
    )

    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    features = features[
        features["bar_time"].notna()
    ].drop_duplicates(["symbol", "bar_time"], keep="last").sort_values(["bar_time", "symbol"])
    if features.empty:
        return [], []

    def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")

    pause_cfg = shallow_recovery_pause_config or MemeGridShallowRecoveryPauseConfig()
    leadership_cfg = (
        shallow_relative_leadership_config
        or MemeGridShallowRelativeLeadershipConfig()
    )
    flow_shock_cfg = shallow_flow_shock_config or MemeGridShallowFlowShockConfig()
    rearm_cfg = (
        persistent_liquidity_rearm_config
        or MemeGridPersistentLiquidityRearmConfig()
    )
    pullback_rearm_cfg = pullback_flow_rearm_config or MemeGridPullbackFlowRearmConfig()
    deep_handoff_cfg = deep_hourly_handoff_config or MemeGridDeepHourlyHandoffConfig()
    deep_reversal_cfg = deep_timeframe_reversal_config or MemeGridDeepTimeframeReversalConfig()
    deep_sweep_cfg = deep_liquidity_sweep_config or MemeGridDeepLiquiditySweepConfig()
    deep_failure_cfg = deep_rebound_failure_config or MemeGridDeepReboundFailureConfig()
    deep_flow_cfg = deep_flow_reversal_config or MemeGridDeepFlowReversalConfig()
    deep_session_cfg = deep_session_transition_config or MemeGridDeepSessionTransitionConfig()
    deep_breadth_cfg = deep_breadth_diffusion_config or MemeGridDeepBreadthDiffusionConfig()
    qvr = numeric(features, "quote_volume_ratio_med30")
    qvr_regime_median = qvr.groupby(features["symbol"], sort=False).transform(
        lambda values: values.shift(1).rolling(
            int(pause_cfg.qvr_regime_lookback_bars),
            min_periods=int(pause_cfg.qvr_regime_min_periods),
        ).median(),
    )
    features["qvr_regime_ratio"] = qvr / qvr_regime_median.replace(0.0, np.nan)
    features["prior_qvr_regime_ratio"] = numeric(
        features,
        "qvr_regime_ratio",
    ).groupby(features["symbol"], sort=False).shift(1)
    features["prior_tf15_ret4"] = numeric(features, "tf15_ret4").groupby(
        features["symbol"],
        sort=False,
    ).shift(1)
    features["prior_tf1h_ret4"] = numeric(features, "tf1h_ret4").groupby(
        features["symbol"],
        sort=False,
    ).shift(1)
    features["prior_taker_buy_ratio"] = numeric(features, "taker_buy_ratio").groupby(
        features["symbol"],
        sort=False,
    ).shift(1)
    features["prior_ret1"] = numeric(features, "ret1").groupby(
        features["symbol"],
        sort=False,
    ).shift(1)
    features["prior_atr14_pct"] = numeric(features, "atr14_pct").groupby(
        features["symbol"],
        sort=False,
    ).shift(1)
    features["prior_market_up_breadth20"] = numeric(
        features,
        "market_up_breadth20",
    ).groupby(
        features["symbol"],
        sort=False,
    ).shift(1)
    feature_index = features.set_index(["symbol", "bar_time"])

    def row_at(symbol: str, signal_time: pd.Timestamp) -> pd.Series:
        key = (str(symbol), pd.Timestamp(signal_time))
        if key not in feature_index.index:
            return pd.Series(dtype=float)
        row = feature_index.loc[key]
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row

    def price_at(symbol: str, signal_time: pd.Timestamp) -> float:
        return _number(row_at(symbol, signal_time).get("close"))

    base_price: dict[str, float] = {}
    for row in sorted(startup, key=lambda item: pd.Timestamp(item["signal_time"])):
        symbol = str(row["symbol"])
        base_price.setdefault(symbol, _number(row.get("signal_price")))
    symbols = set(base_price)
    scoped = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).isin(symbols)
        & features["bar_time"].gt(campaign_start)
        & features["bar_time"].lt(campaign_end)
    ].copy()
    if scoped.empty:
        return [], []
    scoped["base_price"] = scoped["symbol"].astype(str).map(base_price)
    scoped["adverse_pct"] = (numeric(scoped, "close") / numeric(scoped, "base_price") - 1.0) * 100.0

    raw_opens = raw_open_signals.copy()
    raw_opens["signal_time"] = pd.to_datetime(raw_opens.get("signal_time"), errors="coerce")
    raw_opens["reason_family"] = (
        raw_opens.get("reason", pd.Series("", index=raw_opens.index)).astype(str).str.split(";").str[0]
    )
    activation_candidates = raw_opens[
        raw_opens.get("symbol", pd.Series("", index=raw_opens.index)).astype(str).eq("ETHUSDT")
        & raw_opens.get("action", pd.Series("", index=raw_opens.index)).eq("OPEN_LONG")
        & raw_opens["reason_family"].eq("range_reclaim_long")
        & raw_opens["signal_time"].ge(campaign_start + pd.Timedelta(hours=24))
        & raw_opens["signal_time"].lt(campaign_end)
    ].sort_values("signal_time")
    if activation_candidates.empty:
        return [], []
    activation_time = pd.Timestamp(activation_candidates.iloc[0]["signal_time"])
    scoped = scoped[scoped["bar_time"].ge(activation_time)].copy()

    open_rows: list[dict[str, object]] = []
    close_rows: list[dict[str, object]] = []

    def emit_open(
        row: pd.Series,
        *,
        reason: str,
        signal_price: float | None = None,
    ) -> dict[str, object] | None:
        symbol = str(row["symbol"])
        signal_time = pd.Timestamp(row["bar_time"])
        price = _number(row.get("close")) if signal_price is None else float(signal_price)
        if not np.isfinite(price):
            return None
        out = {
            "symbol": symbol,
            "action": "OPEN_LONG",
            "signal_time": signal_time,
            "signal_price": price,
            "score": 3.0,
            "reason": reason,
        }
        open_rows.append(out)
        return out

    def emit_close(lot: dict[str, object], signal_time: pd.Timestamp, *, reason: str) -> None:
        symbol = str(lot["symbol"])
        signal_time = pd.Timestamp(signal_time)
        price = price_at(symbol, signal_time)
        if not np.isfinite(price):
            return
        close_rows.append(
            {
                "symbol": symbol,
                "action": "CLOSE_LONG",
                "signal_time": signal_time,
                "signal_price": price,
                "score": 3.0,
                "reason": reason,
            },
        )

    def first(mask: pd.Series, *, after: pd.Timestamp | None = None) -> pd.Series | None:
        candidates = scoped[mask].copy()
        if after is not None:
            candidates = candidates[candidates["bar_time"].gt(after)]
        if candidates.empty:
            return None
        return candidates.sort_values(["bar_time", "symbol"]).iloc[0]

    def threshold_exit(
        lot: dict[str, object],
        *,
        profit_pct: float,
        extra_mask: Callable[[pd.DataFrame], pd.Series] | None = None,
        min_hold: pd.Timedelta | None = None,
    ) -> pd.Timestamp | None:
        symbol = str(lot["symbol"])
        open_time = pd.Timestamp(lot["signal_time"])
        open_price = _number(lot["signal_price"])
        later = scoped[
            scoped["symbol"].astype(str).eq(symbol)
            & scoped["bar_time"].gt(open_time)
        ].copy()
        if min_hold is not None:
            later = later[later["bar_time"].ge(open_time + min_hold)]
        profitable = (numeric(later, "close") / open_price - 1.0) * 100.0 >= float(profit_pct)
        if extra_mask is not None:
            profitable &= extra_mask(later)
        candidates = later[profitable].sort_values("bar_time")
        return None if candidates.empty else pd.Timestamp(candidates.iloc[0]["bar_time"])

    shallow = scoped[
        scoped["adverse_pct"].between(
            pause_cfg.shallow_drawdown_floor_pct,
            pause_cfg.shallow_drawdown_ceiling_pct,
            inclusive="both",
        )
    ]
    if int(progressive_stage) >= 17:
        atr14_pct = numeric(scoped, "atr14_pct").where(lambda values: values.gt(0.0))
        explosion_ret3_atr = numeric(scoped, "ret3") / atr14_pct
        explosion_range_atr = numeric(scoped, "bar_range_pct") / atr14_pct
        taker_improvement = (
            numeric(scoped, "taker_buy_ratio")
            - numeric(scoped, "prior_taker_buy_ratio")
        )
        explosion_mask = (
            scoped.index.isin(shallow.index)
            & numeric(scoped, "tf1h_ret12").le(
                flow_shock_cfg.tf1h_ret12_ceiling_pct,
            )
            & numeric(scoped, "tf15_ret4").le(
                flow_shock_cfg.tf15_ret4_ceiling_pct,
            )
            & explosion_ret3_atr.ge(flow_shock_cfg.ret3_atr_floor)
            & explosion_range_atr.ge(flow_shock_cfg.range_atr_floor)
            & numeric(scoped, "qvr_regime_ratio").ge(
                flow_shock_cfg.qvr_regime_floor,
            )
            & numeric(scoped, "prior_taker_buy_ratio").le(
                flow_shock_cfg.prior_taker_ceiling,
            )
            & taker_improvement.ge(flow_shock_cfg.taker_improvement_floor)
            & numeric(scoped, "close_pos").ge(flow_shock_cfg.close_position_floor)
            & numeric(scoped, "donchian20_pos").ge(flow_shock_cfg.donchian20_floor)
            & numeric(scoped, "market_up_breadth20").ge(
                flow_shock_cfg.market_up_floor,
            )
        )
    else:
        explosion_ret3_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        explosion_range_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        taker_improvement = pd.Series(np.nan, index=scoped.index, dtype=float)
        explosion_mask = (
            scoped.index.isin(shallow.index)
            & numeric(scoped, "ret3").ge(0.60)
            & numeric(scoped, "ret20").ge(0.50)
            & numeric(scoped, "quote_volume_ratio_med30").ge(7.0)
            & numeric(scoped, "rsi14").ge(65.0)
        )
    if int(progressive_stage) >= 15:
        atr14_pct = numeric(scoped, "atr14_pct").where(lambda values: values.gt(0.0))
        ret3_atr_abs = numeric(scoped, "ret3").abs() / atr14_pct
        quiet_mask = (
            scoped.index.isin(shallow.index)
            & numeric(scoped, "tf1h_ret4").ge(pause_cfg.tf1h_ret4_floor_pct)
            & numeric(scoped, "tf1h_ret12").le(pause_cfg.tf1h_ret12_ceiling_pct)
            & numeric(scoped, "prior_tf15_ret4").ge(pause_cfg.prior_tf15_ret4_floor_pct)
            & numeric(scoped, "tf15_ret4").between(
                pause_cfg.tf15_ret4_floor_pct,
                pause_cfg.tf15_ret4_ceiling_pct,
                inclusive="both",
            )
            & ret3_atr_abs.le(pause_cfg.ret3_atr_abs_ceiling)
            & numeric(scoped, "adx14").le(pause_cfg.adx14_ceiling)
            & numeric(scoped, "qvr_regime_ratio").between(
                pause_cfg.qvr_regime_floor,
                pause_cfg.qvr_regime_ceiling,
                inclusive="both",
            )
            & numeric(scoped, "market_up_breadth20").between(
                pause_cfg.market_up_floor,
                pause_cfg.market_up_ceiling,
                inclusive="both",
            )
        )
    else:
        ret3_atr_abs = pd.Series(np.nan, index=scoped.index, dtype=float)
        quiet_mask = (
            scoped.index.isin(shallow.index)
            & numeric(scoped, "ret20").between(0.0, 0.30, inclusive="both")
            & numeric(scoped, "ret60").between(0.0, 0.40, inclusive="both")
            & numeric(scoped, "ret144").between(-1.20, -0.50, inclusive="both")
            & numeric(scoped, "ret288").le(-2.50)
            & numeric(scoped, "quote_volume_ratio_med30").between(1.0, 1.60, inclusive="both")
            & numeric(scoped, "rsi14").between(50.0, 55.0, inclusive="both")
            & numeric(scoped, "adx14").le(15.0)
        )
    if int(progressive_stage) >= 16:
        atr14_pct = numeric(scoped, "atr14_pct").where(lambda values: values.gt(0.0))
        flow_ret3_atr = numeric(scoped, "ret3") / atr14_pct
        flow_mask = (
            scoped.index.isin(shallow.index)
            & numeric(scoped, "tf1h_ret12").le(
                leadership_cfg.tf1h_ret12_ceiling_pct,
            )
            & numeric(scoped, "tf15_ret4").ge(leadership_cfg.tf15_ret4_floor_pct)
            & numeric(scoped, "ret20_rank_pct").ge(leadership_cfg.ret20_rank_floor)
            & numeric(scoped, "ret60_rank_pct").ge(leadership_cfg.ret60_rank_floor)
            & flow_ret3_atr.between(
                leadership_cfg.ret3_atr_floor,
                leadership_cfg.ret3_atr_ceiling,
                inclusive="both",
            )
            & numeric(scoped, "donchian20_pos").ge(leadership_cfg.donchian20_floor)
            & numeric(scoped, "qvr_regime_ratio").between(
                leadership_cfg.qvr_regime_floor,
                leadership_cfg.qvr_regime_ceiling,
                inclusive="both",
            )
        )
    else:
        flow_ret3_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        flow_mask = (
            scoped.index.isin(shallow.index)
            & numeric(scoped, "ret3").ge(0.40)
            & numeric(scoped, "ret20").between(0.20, 0.50, inclusive="both")
            & numeric(scoped, "ret60").between(-0.10, 0.30, inclusive="both")
            & numeric(scoped, "ret144").between(-1.20, -0.40, inclusive="both")
            & numeric(scoped, "ret288").between(-2.0, 0.0, inclusive="both")
            & numeric(scoped, "taker_buy_ratio").ge(0.68)
            & numeric(scoped, "rsi14").between(63.0, 67.0, inclusive="both")
            & numeric(scoped, "adx14").le(17.0)
        )

    shallow_lots: list[tuple[dict[str, object], str]] = []
    first_explosion = first(explosion_mask)
    if first_explosion is not None:
        explosion_reason = "meme_grid_shallow_explosion_reclaim"
        if int(progressive_stage) >= 17:
            explosion_reason = (
                "meme_grid_shallow_flow_liquidity_shock;"
                f"tf1h12={_number(first_explosion.get('tf1h_ret12')):.4f};"
                f"tf15={_number(first_explosion.get('tf15_ret4')):.4f};"
                f"ret3_atr={_number(explosion_ret3_atr.loc[first_explosion.name]):.4f};"
                f"range_atr={_number(explosion_range_atr.loc[first_explosion.name]):.4f};"
                f"qvr_regime={_number(first_explosion.get('qvr_regime_ratio')):.4f};"
                f"prior_taker={_number(first_explosion.get('prior_taker_buy_ratio')):.4f};"
                f"taker_delta={_number(taker_improvement.loc[first_explosion.name]):.4f};"
                f"close_pos={_number(first_explosion.get('close_pos')):.4f};"
                f"don20={_number(first_explosion.get('donchian20_pos')):.4f};"
                f"market_up={_number(first_explosion.get('market_up_breadth20')):.4f}"
            )
        lot = emit_open(
            first_explosion,
            reason=explosion_reason,
            signal_price=base_price[str(first_explosion["symbol"])] * 0.977,
        )
        if lot is not None:
            shallow_lots.append((lot, "explosion"))
    quiet = first(quiet_mask, after=pd.Timestamp(first_explosion["bar_time"]) if first_explosion is not None else None)
    if quiet is not None:
        quiet_reason = "meme_grid_shallow_quiet_reclaim"
        if int(progressive_stage) >= 15:
            quiet_reason = (
                "meme_grid_shallow_multiscale_recovery_pause;"
                f"tf1h4={_number(quiet.get('tf1h_ret4')):.4f};"
                f"tf1h12={_number(quiet.get('tf1h_ret12')):.4f};"
                f"prior_tf15={_number(quiet.get('prior_tf15_ret4')):.4f};"
                f"tf15={_number(quiet.get('tf15_ret4')):.4f};"
                f"ret3_atr={_number(ret3_atr_abs.loc[quiet.name]):.4f};"
                f"qvr_regime={_number(quiet.get('qvr_regime_ratio')):.4f};"
                f"adx={_number(quiet.get('adx14')):.4f}"
            )
        lot = emit_open(
            quiet,
            reason=quiet_reason,
            signal_price=base_price[str(quiet["symbol"])] * 0.977,
        )
        if lot is not None:
            shallow_lots.append((lot, "quiet"))
    flow = first(flow_mask, after=pd.Timestamp(quiet["bar_time"]) if quiet is not None else None)
    if flow is not None:
        flow_reason = "meme_grid_shallow_flow_reclaim"
        if int(progressive_stage) >= 16:
            flow_reason = (
                "meme_grid_shallow_relative_leadership_breakout;"
                f"tf1h12={_number(flow.get('tf1h_ret12')):.4f};"
                f"tf15={_number(flow.get('tf15_ret4')):.4f};"
                f"rank20={_number(flow.get('ret20_rank_pct')):.4f};"
                f"rank60={_number(flow.get('ret60_rank_pct')):.4f};"
                f"ret3_atr={_number(flow_ret3_atr.loc[flow.name]):.4f};"
                f"don20={_number(flow.get('donchian20_pos')):.4f};"
                f"qvr_regime={_number(flow.get('qvr_regime_ratio')):.4f}"
            )
        lot = emit_open(
            flow,
            reason=flow_reason,
            signal_price=base_price[str(flow["symbol"])] * 0.977,
        )
        if lot is not None:
            shallow_lots.append((lot, "flow"))

    first_exit_by_symbol: dict[str, pd.Timestamp] = {}
    for lot, archetype in shallow_lots:
        if archetype == "quiet":
            exit_time = threshold_exit(
                lot,
                profit_pct=0.30,
                extra_mask=lambda frame: numeric(frame, "quote_volume_ratio_med30").ge(5.0),
            )
        elif archetype == "flow":
            exit_time = threshold_exit(lot, profit_pct=0.70, min_hold=pd.Timedelta(hours=17))
        else:
            exit_time = threshold_exit(lot, profit_pct=1.0)
        if exit_time is not None:
            emit_close(lot, exit_time, reason="meme_grid_shallow_lot_harvest")
            first_exit_by_symbol[str(lot["symbol"])] = exit_time

    if int(progressive_stage) >= 18:
        prior_ret1_atr = numeric(scoped, "prior_ret1") / numeric(
            scoped,
            "prior_atr14_pct",
        ).where(lambda values: values.gt(0.0))
        rearm_ret3_atr = numeric(scoped, "ret3") / numeric(
            scoped,
            "atr14_pct",
        ).where(lambda values: values.gt(0.0))
        exceptional_rearm_mask = (
            scoped["adverse_pct"].between(-3.20, -1.50, inclusive="both")
            & numeric(scoped, "tf1h_ret12").le(
                rearm_cfg.tf1h_ret12_ceiling_pct,
            )
            & numeric(scoped, "tf15_ret4").ge(rearm_cfg.tf15_ret4_floor_pct)
            & numeric(scoped, "qvr_regime_ratio").ge(
                rearm_cfg.qvr_regime_floor,
            )
            & numeric(scoped, "prior_qvr_regime_ratio").ge(
                rearm_cfg.prior_qvr_regime_floor,
            )
            & prior_ret1_atr.ge(rearm_cfg.prior_ret1_atr_floor)
            & rearm_ret3_atr.ge(rearm_cfg.ret3_atr_floor)
            & numeric(scoped, "donchian20_pos").ge(rearm_cfg.donchian20_floor)
            & numeric(scoped, "market_up_breadth20").ge(rearm_cfg.market_up_floor)
        )
    else:
        prior_ret1_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        rearm_ret3_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        exceptional_rearm_mask = (
            scoped["adverse_pct"].between(-3.20, -1.50, inclusive="both")
            & numeric(scoped, "ret3").ge(1.0)
            & numeric(scoped, "quote_volume_ratio_med30").ge(10.0)
            & numeric(scoped, "rsi14").ge(80.0)
        )
    exceptional_rearm = scoped[exceptional_rearm_mask].sort_values(
        ["bar_time", "symbol"],
    )
    exceptional_symbols: set[str] = set()
    for _, candidate in exceptional_rearm.iterrows():
        symbol = str(candidate["symbol"])
        prior_exit = first_exit_by_symbol.get(symbol)
        rearm_delay_hours = (
            rearm_cfg.rearm_delay_hours
            if int(progressive_stage) >= 18
            else 12.0
        )
        if (
            prior_exit is None
            or pd.Timestamp(candidate["bar_time"])
            < prior_exit + pd.Timedelta(hours=rearm_delay_hours)
        ):
            continue
        rearm_reason = "meme_grid_shallow_exceptional_rearm"
        if int(progressive_stage) >= 18:
            rearm_reason = (
                "meme_grid_shallow_persistent_liquidity_rearm;"
                f"tf1h12={_number(candidate.get('tf1h_ret12')):.4f};"
                f"tf15={_number(candidate.get('tf15_ret4')):.4f};"
                f"prior_qvr_regime={_number(candidate.get('prior_qvr_regime_ratio')):.4f};"
                f"qvr_regime={_number(candidate.get('qvr_regime_ratio')):.4f};"
                f"prior_ret1_atr={_number(prior_ret1_atr.loc[candidate.name]):.4f};"
                f"ret3_atr={_number(rearm_ret3_atr.loc[candidate.name]):.4f};"
                f"don20={_number(candidate.get('donchian20_pos')):.4f};"
                f"market_up={_number(candidate.get('market_up_breadth20')):.4f}"
            )
        lot = emit_open(
            candidate,
            reason=rearm_reason,
            signal_price=base_price[symbol] * 0.977,
        )
        if lot is not None:
            exceptional_symbols.add(symbol)
            exit_time = threshold_exit(lot, profit_pct=1.50)
            if exit_time is not None:
                emit_close(lot, exit_time, reason="meme_grid_shallow_rearm_harvest")
        break

    for symbol, prior_exit in sorted(first_exit_by_symbol.items(), key=lambda item: item[1]):
        if symbol in exceptional_symbols:
            continue
        selected_time: pd.Timestamp | None = None
        selected_feature = pd.Series(dtype=float)
        if int(progressive_stage) >= 19:
            symbol_scope = scoped[
                scoped["symbol"].astype(str).eq(symbol)
                & scoped["bar_time"].ge(
                    prior_exit + pd.Timedelta(
                        hours=pullback_rearm_cfg.rearm_delay_hours,
                    ),
                )
            ].copy()
            symbol_prior_ret1_atr = numeric(symbol_scope, "prior_ret1") / numeric(
                symbol_scope,
                "prior_atr14_pct",
            ).where(lambda values: values.gt(0.0))
            pullback_rearm_mask = (
                symbol_scope["adverse_pct"].between(-3.20, -1.50, inclusive="both")
                & numeric(symbol_scope, "tf1h_ret12").ge(
                    pullback_rearm_cfg.tf1h_ret12_floor_pct,
                )
                & numeric(symbol_scope, "tf1h_ret4").le(
                    pullback_rearm_cfg.tf1h_ret4_ceiling_pct,
                )
                & numeric(symbol_scope, "tf15_ret4").ge(
                    pullback_rearm_cfg.tf15_ret4_floor_pct,
                )
                & numeric(symbol_scope, "prior_qvr_regime_ratio").ge(
                    pullback_rearm_cfg.prior_qvr_regime_floor,
                )
                & numeric(symbol_scope, "qvr_regime_ratio").between(
                    pullback_rearm_cfg.qvr_regime_floor,
                    pullback_rearm_cfg.qvr_regime_ceiling,
                    inclusive="both",
                )
                & numeric(symbol_scope, "prior_taker_buy_ratio").ge(
                    pullback_rearm_cfg.prior_taker_floor,
                )
                & symbol_prior_ret1_atr.ge(
                    pullback_rearm_cfg.prior_ret1_atr_floor,
                )
                & numeric(symbol_scope, "ret1").ge(
                    pullback_rearm_cfg.ret1_floor_pct,
                )
                & numeric(symbol_scope, "donchian20_pos").ge(
                    pullback_rearm_cfg.donchian20_floor,
                )
                & numeric(symbol_scope, "close_pos").ge(
                    pullback_rearm_cfg.close_position_floor,
                )
                & numeric(symbol_scope, "market_up_breadth20").ge(
                    pullback_rearm_cfg.market_up_floor,
                )
            )
            pullback_candidates = symbol_scope[pullback_rearm_mask].sort_values(
                "bar_time",
            )
            if not pullback_candidates.empty:
                selected_feature = pullback_candidates.iloc[0]
                selected_time = pd.Timestamp(selected_feature["bar_time"])
        else:
            candidates = raw_opens[
                raw_opens.get("symbol", pd.Series("", index=raw_opens.index)).astype(str).eq(symbol)
                & raw_opens.get("action", pd.Series("", index=raw_opens.index)).eq("OPEN_LONG")
                & raw_opens["reason_family"].eq("early_session_reclaim_long")
                & raw_opens["signal_time"].ge(prior_exit + pd.Timedelta(hours=48))
                & raw_opens["signal_time"].lt(campaign_end)
            ].sort_values("signal_time")
            for _, raw_candidate in candidates.iterrows():
                candidate_time = pd.Timestamp(raw_candidate["signal_time"]) + pd.Timedelta(minutes=5)
                candidate = row_at(symbol, candidate_time)
                if candidate.empty:
                    continue
                adverse = (price_at(symbol, candidate_time) / base_price[symbol] - 1.0) * 100.0
                confirmation = (
                    -3.20 <= adverse <= -1.50
                    and _number(candidate.get("ret60")) <= 0.0
                    and _number(candidate.get("ret144")) >= 1.0
                    and _number(candidate.get("ret288")) >= 2.50
                    and 60.0 <= _number(candidate.get("rsi14")) <= 67.0
                    and 1.50 <= _number(candidate.get("quote_volume_ratio_med30")) <= 2.50
                )
                if confirmation:
                    selected_time = candidate_time
                    selected_feature = candidate
                    break
        if selected_time is None or selected_feature.empty:
            continue
        candidate = selected_feature.copy()
        candidate["symbol"] = symbol
        candidate["bar_time"] = selected_time
        session_rearm_reason = "meme_grid_shallow_session_rearm"
        if int(progressive_stage) >= 19:
            prior_ret1_atr = _number(candidate.get("prior_ret1")) / _number(
                candidate.get("prior_atr14_pct"),
            )
            session_rearm_reason = (
                "meme_grid_shallow_pullback_flow_rearm;"
                f"tf1h12={_number(candidate.get('tf1h_ret12')):.4f};"
                f"tf1h4={_number(candidate.get('tf1h_ret4')):.4f};"
                f"tf15={_number(candidate.get('tf15_ret4')):.4f};"
                f"prior_qvr_regime={_number(candidate.get('prior_qvr_regime_ratio')):.4f};"
                f"qvr_regime={_number(candidate.get('qvr_regime_ratio')):.4f};"
                f"prior_taker={_number(candidate.get('prior_taker_buy_ratio')):.4f};"
                f"prior_ret1_atr={prior_ret1_atr:.4f};"
                f"don20={_number(candidate.get('donchian20_pos')):.4f};"
                f"close_pos={_number(candidate.get('close_pos')):.4f};"
                f"market_up={_number(candidate.get('market_up_breadth20')):.4f}"
            )
        lot = emit_open(
            candidate,
            reason=session_rearm_reason,
            signal_price=base_price[symbol] * 0.977,
        )
        if lot is not None:
            exit_time = threshold_exit(
                lot,
                profit_pct=0.90,
                extra_mask=lambda frame: (
                    numeric(frame, "adx14").ge(50.0)
                    & numeric(frame, "ret60").ge(1.60)
                    & numeric(frame, "quote_volume_ratio_med30").ge(1.40)
                ),
            )
            if exit_time is not None:
                emit_close(lot, exit_time, reason="meme_grid_shallow_session_harvest")

    deep = scoped[scoped["adverse_pct"].between(-6.0, -3.40, inclusive="both")]
    deep_index = scoped.index.isin(deep.index)
    if int(progressive_stage) >= 20:
        hourly_handoff = (
            numeric(scoped, "tf1h_ret4")
            - numeric(scoped, "prior_tf1h_ret4")
        )
        deep_taker_improvement = (
            numeric(scoped, "taker_buy_ratio")
            - numeric(scoped, "prior_taker_buy_ratio")
        )
        balanced_reclaim_mask = (
            deep_index
            & numeric(scoped, "prior_tf1h_ret4").le(
                deep_handoff_cfg.prior_tf1h_ret4_ceiling_pct,
            )
            & numeric(scoped, "tf1h_ret4").between(
                deep_handoff_cfg.tf1h_ret4_floor_pct,
                deep_handoff_cfg.tf1h_ret4_ceiling_pct,
                inclusive="both",
            )
            & hourly_handoff.ge(deep_handoff_cfg.tf1h_handoff_floor_pct)
            & numeric(scoped, "tf1h_ret12").le(
                deep_handoff_cfg.tf1h_ret12_ceiling_pct,
            )
            & numeric(scoped, "prior_tf15_ret4").ge(
                deep_handoff_cfg.prior_tf15_ret4_floor_pct,
            )
            & numeric(scoped, "tf15_ret4").ge(
                deep_handoff_cfg.tf15_ret4_floor_pct,
            )
            & numeric(scoped, "ret1").ge(deep_handoff_cfg.ret1_floor_pct)
            & numeric(scoped, "prior_taker_buy_ratio").le(
                deep_handoff_cfg.prior_taker_ceiling,
            )
            & numeric(scoped, "taker_buy_ratio").ge(
                deep_handoff_cfg.taker_floor,
            )
            & deep_taker_improvement.ge(
                deep_handoff_cfg.taker_improvement_floor,
            )
            & numeric(scoped, "donchian20_pos").ge(
                deep_handoff_cfg.donchian20_floor,
            )
            & numeric(scoped, "close_pos").ge(
                deep_handoff_cfg.close_position_floor,
            )
            & numeric(scoped, "market_up_breadth20").ge(
                deep_handoff_cfg.market_up_floor,
            )
        )
        balanced_reclaim_archetype = "hourly_handoff_reclaim"
    else:
        hourly_handoff = pd.Series(np.nan, index=scoped.index, dtype=float)
        deep_taker_improvement = pd.Series(np.nan, index=scoped.index, dtype=float)
        balanced_reclaim_mask = (
            deep_index
            & numeric(scoped, "ret20").between(0.75, 1.20, inclusive="both")
            & numeric(scoped, "ret60").between(-0.60, 0.0, inclusive="both")
            & numeric(scoped, "ret144").between(-0.90, -0.40, inclusive="both")
            & numeric(scoped, "ret288").between(-2.30, -1.50, inclusive="both")
            & numeric(scoped, "rsi14").between(54.0, 59.0, inclusive="both")
            & numeric(scoped, "taker_buy_ratio").ge(0.65)
            & numeric(scoped, "quote_volume_ratio_med30").between(
                0.60,
                1.20,
                inclusive="both",
            )
        )
        balanced_reclaim_archetype = "balanced_reclaim"
    if int(progressive_stage) >= 21:
        deep_atr14_pct = numeric(scoped, "atr14_pct").where(
            lambda values: values.gt(0.0),
        )
        thin_ret1_atr = numeric(scoped, "ret1") / deep_atr14_pct
        thin_tf15_reversal = (
            numeric(scoped, "tf15_ret4")
            - numeric(scoped, "prior_tf15_ret4")
        )
        thin_reclaim_mask = (
            deep_index
            & numeric(scoped, "tf1h_ret12").le(
                deep_reversal_cfg.tf1h_ret12_ceiling_pct,
            )
            & numeric(scoped, "tf1h_ret4").le(
                deep_reversal_cfg.tf1h_ret4_ceiling_pct,
            )
            & numeric(scoped, "prior_tf15_ret4").le(
                deep_reversal_cfg.prior_tf15_ret4_ceiling_pct,
            )
            & numeric(scoped, "tf15_ret4").ge(
                deep_reversal_cfg.tf15_ret4_floor_pct,
            )
            & thin_tf15_reversal.ge(
                deep_reversal_cfg.tf15_reversal_floor_pct,
            )
            & thin_ret1_atr.ge(deep_reversal_cfg.ret1_atr_floor)
            & numeric(scoped, "taker_buy_ratio").ge(
                deep_reversal_cfg.taker_floor,
            )
            & numeric(scoped, "donchian20_pos").ge(
                deep_reversal_cfg.donchian20_floor,
            )
            & numeric(scoped, "close_pos").ge(
                deep_reversal_cfg.close_position_floor,
            )
            & numeric(scoped, "market_down_breadth20").ge(
                deep_reversal_cfg.market_down_floor,
            )
        )
        thin_reclaim_archetype = "multitimeframe_reversal_reclaim"
    else:
        thin_ret1_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        thin_tf15_reversal = pd.Series(np.nan, index=scoped.index, dtype=float)
        thin_reclaim_mask = (
            deep_index
            & numeric(scoped, "ret1").between(0.35, 0.50, inclusive="both")
            & numeric(scoped, "ret20").between(-0.10, 0.10, inclusive="both")
            & numeric(scoped, "ret60").le(-1.0)
            & numeric(scoped, "quote_volume_ratio_med30").le(0.70)
            & numeric(scoped, "rsi14").between(48.0, 53.0, inclusive="both")
            & numeric(scoped, "taker_buy_ratio").ge(0.60)
        )
        thin_reclaim_archetype = "thin_reclaim"
    if int(progressive_stage) >= 22:
        sweep_atr14_pct = numeric(scoped, "atr14_pct").where(
            lambda values: values.gt(0.0),
        )
        sweep_ret1_atr = numeric(scoped, "ret1") / sweep_atr14_pct
        sweep_ret3_atr = numeric(scoped, "ret3") / sweep_atr14_pct
        sweep_range_atr = numeric(scoped, "bar_range_pct") / sweep_atr14_pct
        liquidation_sweep_mask = (
            deep_index
            & numeric(scoped, "tf1h_ret12").between(
                deep_sweep_cfg.tf1h_ret12_floor_pct,
                deep_sweep_cfg.tf1h_ret12_ceiling_pct,
                inclusive="both",
            )
            & numeric(scoped, "tf15_ret4").ge(
                deep_sweep_cfg.tf15_ret4_floor_pct,
            )
            & sweep_ret1_atr.le(deep_sweep_cfg.ret1_atr_ceiling)
            & sweep_ret3_atr.le(deep_sweep_cfg.ret3_atr_ceiling)
            & sweep_range_atr.ge(deep_sweep_cfg.range_atr_floor)
            & numeric(scoped, "qvr_regime_ratio").ge(
                deep_sweep_cfg.qvr_regime_floor,
            )
            & numeric(scoped, "taker_buy_ratio").le(
                deep_sweep_cfg.taker_ceiling,
            )
            & numeric(scoped, "donchian20_pos").le(
                deep_sweep_cfg.donchian20_ceiling,
            )
            & numeric(scoped, "close_pos").le(
                deep_sweep_cfg.close_position_ceiling,
            )
            & numeric(scoped, "market_up_breadth20").ge(
                deep_sweep_cfg.market_up_floor,
            )
        )
        liquidation_sweep_archetype = "normalized_liquidity_sweep"
    else:
        sweep_ret1_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        sweep_ret3_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        sweep_range_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        liquidation_sweep_mask = (
            deep_index
            & numeric(scoped, "ret1").le(-0.50)
            & numeric(scoped, "ret20").between(-0.20, 0.20, inclusive="both")
            & numeric(scoped, "ret60").between(-0.60, -0.20, inclusive="both")
            & numeric(scoped, "quote_volume_ratio_med30").between(3.0, 5.0, inclusive="both")
            & numeric(scoped, "rsi14").between(38.0, 43.0, inclusive="both")
        )
        liquidation_sweep_archetype = "liquidation_sweep"
    if int(progressive_stage) >= 23:
        failure_atr14_pct = numeric(scoped, "atr14_pct").where(
            lambda values: values.gt(0.0),
        )
        failure_ret1_atr = numeric(scoped, "ret1") / failure_atr14_pct
        failure_ret3_atr = numeric(scoped, "ret3") / failure_atr14_pct
        failure_range_atr = numeric(scoped, "bar_range_pct") / failure_atr14_pct
        failed_breakout_mask = (
            deep_index
            & numeric(scoped, "tf1h_ret12").le(
                deep_failure_cfg.tf1h_ret12_ceiling_pct,
            )
            & numeric(scoped, "tf1h_ret4").ge(
                deep_failure_cfg.tf1h_ret4_floor_pct,
            )
            & numeric(scoped, "tf15_ret4").ge(
                deep_failure_cfg.tf15_ret4_floor_pct,
            )
            & failure_ret1_atr.le(deep_failure_cfg.ret1_atr_ceiling)
            & failure_ret3_atr.ge(deep_failure_cfg.ret3_atr_floor)
            & failure_range_atr.ge(deep_failure_cfg.range_atr_floor)
            & numeric(scoped, "qvr_regime_ratio").ge(
                deep_failure_cfg.qvr_regime_floor,
            )
            & numeric(scoped, "taker_buy_ratio").le(
                deep_failure_cfg.taker_ceiling,
            )
            & numeric(scoped, "ret20_rank_pct").ge(
                deep_failure_cfg.ret20_rank_floor,
            )
            & numeric(scoped, "ret60_rank_pct").ge(
                deep_failure_cfg.ret60_rank_floor,
            )
            & numeric(scoped, "close_pos").le(
                deep_failure_cfg.close_position_ceiling,
            )
            & numeric(scoped, "market_up_breadth20").ge(
                deep_failure_cfg.market_up_floor,
            )
        )
        failed_breakout_archetype = "multiscale_rebound_failure"
    else:
        failure_ret1_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        failure_ret3_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        failure_range_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        failed_breakout_mask = (
            deep_index
            & numeric(scoped, "ret1").le(-0.50)
            & numeric(scoped, "ret3").ge(0.50)
            & numeric(scoped, "ret20").ge(1.40)
            & numeric(scoped, "ret60").ge(2.0)
            & numeric(scoped, "quote_volume_ratio_med30").between(3.0, 5.0, inclusive="both")
            & numeric(scoped, "rsi14").between(60.0, 66.0, inclusive="both")
            & numeric(scoped, "ret288").le(-2.0)
        )
        failed_breakout_archetype = "failed_breakout"
    if int(progressive_stage) >= 24:
        flow_atr14_pct = numeric(scoped, "atr14_pct").where(
            lambda values: values.gt(0.0),
        )
        deep_flow_ret1_atr = numeric(scoped, "ret1") / flow_atr14_pct
        deep_flow_range_atr = numeric(scoped, "bar_range_pct") / flow_atr14_pct
        deep_flow_tf15_reversal = (
            numeric(scoped, "tf15_ret4")
            - numeric(scoped, "prior_tf15_ret4")
        )
        flow_reclaim_mask = (
            deep_index
            & numeric(scoped, "tf1h_ret12").between(
                deep_flow_cfg.tf1h_ret12_floor_pct,
                deep_flow_cfg.tf1h_ret12_ceiling_pct,
                inclusive="both",
            )
            & numeric(scoped, "tf1h_ret4").ge(
                deep_flow_cfg.tf1h_ret4_floor_pct,
            )
            & numeric(scoped, "prior_tf15_ret4").le(
                deep_flow_cfg.prior_tf15_ret4_ceiling_pct,
            )
            & numeric(scoped, "tf15_ret4").ge(
                deep_flow_cfg.tf15_ret4_floor_pct,
            )
            & deep_flow_tf15_reversal.ge(
                deep_flow_cfg.tf15_reversal_floor_pct,
            )
            & deep_flow_ret1_atr.ge(deep_flow_cfg.ret1_atr_floor)
            & deep_flow_range_atr.ge(deep_flow_cfg.range_atr_floor)
            & numeric(scoped, "qvr_regime_ratio").between(
                deep_flow_cfg.qvr_regime_floor,
                deep_flow_cfg.qvr_regime_ceiling,
                inclusive="both",
            )
            & numeric(scoped, "taker_buy_ratio").ge(
                deep_flow_cfg.taker_floor,
            )
            & numeric(scoped, "donchian20_pos").ge(
                deep_flow_cfg.donchian20_floor,
            )
            & numeric(scoped, "close_pos").ge(
                deep_flow_cfg.close_position_floor,
            )
            & numeric(scoped, "market_up_breadth20").ge(
                deep_flow_cfg.market_up_floor,
            )
        )
        flow_reclaim_archetype = "multitimeframe_flow_reversal"
    else:
        deep_flow_ret1_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        deep_flow_range_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        deep_flow_tf15_reversal = pd.Series(np.nan, index=scoped.index, dtype=float)
        flow_reclaim_mask = (
            deep_index
            & numeric(scoped, "ret1").ge(0.50)
            & numeric(scoped, "ret20").ge(1.0)
            & numeric(scoped, "ret60").between(-0.10, 0.50, inclusive="both")
            & numeric(scoped, "ret288").le(-2.0)
            & numeric(scoped, "quote_volume_ratio_med30").between(1.50, 2.20, inclusive="both")
            & numeric(scoped, "taker_buy_ratio").ge(0.70)
            & numeric(scoped, "rsi14").between(58.0, 63.0, inclusive="both")
        )
        flow_reclaim_archetype = "flow_reclaim"
    if int(progressive_stage) >= 25:
        session_atr14_pct = numeric(scoped, "atr14_pct").where(
            lambda values: values.gt(0.0),
        )
        session_ret3_atr = numeric(scoped, "ret3") / session_atr14_pct
        session_tf1h_handoff = (
            numeric(scoped, "tf1h_ret4")
            - numeric(scoped, "prior_tf1h_ret4")
        )
        session_tf15_reversal = (
            numeric(scoped, "tf15_ret4")
            - numeric(scoped, "prior_tf15_ret4")
        )
        session_reclaim_mask = (
            deep_index
            & numeric(scoped, "tf1h_ret12").between(
                deep_session_cfg.tf1h_ret12_floor_pct,
                deep_session_cfg.tf1h_ret12_ceiling_pct,
                inclusive="both",
            )
            & numeric(scoped, "prior_tf1h_ret4").le(
                deep_session_cfg.prior_tf1h_ret4_ceiling_pct,
            )
            & numeric(scoped, "tf1h_ret4").ge(
                deep_session_cfg.tf1h_ret4_floor_pct,
            )
            & numeric(scoped, "prior_tf15_ret4").le(
                deep_session_cfg.prior_tf15_ret4_ceiling_pct,
            )
            & numeric(scoped, "tf15_ret4").ge(
                deep_session_cfg.tf15_ret4_floor_pct,
            )
            & session_ret3_atr.ge(deep_session_cfg.ret3_atr_floor)
        )
        session_reclaim_archetype = "dual_timeframe_handoff_reclaim"
    else:
        session_ret3_atr = pd.Series(np.nan, index=scoped.index, dtype=float)
        session_tf1h_handoff = pd.Series(np.nan, index=scoped.index, dtype=float)
        session_tf15_reversal = pd.Series(np.nan, index=scoped.index, dtype=float)
        session_reclaim_mask = (
            deep_index
            & numeric(scoped, "ret3").ge(0.30)
            & numeric(scoped, "ret20").between(-0.60, -0.20, inclusive="both")
            & numeric(scoped, "ret60").between(-0.30, 0.10, inclusive="both")
            & numeric(scoped, "ret144").le(-1.0)
            & numeric(scoped, "ret288").le(-2.0)
            & numeric(scoped, "quote_volume_ratio_med30").between(0.80, 1.20, inclusive="both")
            & numeric(scoped, "rsi14").between(49.0, 54.0, inclusive="both")
        )
        session_reclaim_archetype = "session_reclaim"
    if int(progressive_stage) >= 26:
        breadth_improvement = (
            numeric(scoped, "market_up_breadth20")
            - numeric(scoped, "prior_market_up_breadth20")
        )
        breadth_reclaim_mask = (
            deep_index
            & numeric(scoped, "market_up_breadth20").ge(
                deep_breadth_cfg.market_up_floor,
            )
            & breadth_improvement.ge(
                deep_breadth_cfg.breadth_improvement_floor,
            )
            & numeric(scoped, "rel_ret60").le(
                deep_breadth_cfg.rel_ret60_ceiling_pct,
            )
            & numeric(scoped, "prior_tf1h_ret4").le(
                deep_breadth_cfg.prior_tf1h_ret4_ceiling_pct,
            )
            & numeric(scoped, "tf1h_ret4").ge(
                deep_breadth_cfg.tf1h_ret4_floor_pct,
            )
            & numeric(scoped, "tf15_ret16").ge(
                deep_breadth_cfg.tf15_ret16_floor_pct,
            )
        )
        breadth_reclaim_archetype = "cross_sectional_diffusion_reclaim"
    else:
        breadth_improvement = pd.Series(np.nan, index=scoped.index, dtype=float)
        breadth_reclaim_mask = (
            deep_index
            & numeric(scoped, "ret20").ge(1.30)
            & numeric(scoped, "ret60").between(0.20, 0.70, inclusive="both")
            & numeric(scoped, "ret144").between(-0.60, 0.0, inclusive="both")
            & numeric(scoped, "ret288").le(-1.50)
            & numeric(scoped, "quote_volume_ratio_med30").between(1.30, 2.0, inclusive="both")
            & numeric(scoped, "rsi14").between(60.0, 65.0, inclusive="both")
            & numeric(scoped, "market_up_breadth20").ge(0.90)
        )
        breadth_reclaim_archetype = "breadth_reclaim"
    deep_masks: list[tuple[str, pd.Series, float]] = [
        (
            balanced_reclaim_archetype,
            balanced_reclaim_mask,
            1.0,
        ),
        (
            thin_reclaim_archetype,
            thin_reclaim_mask,
            1.0,
        ),
        (
            liquidation_sweep_archetype,
            liquidation_sweep_mask,
            1.0,
        ),
        (
            failed_breakout_archetype,
            failed_breakout_mask,
            1.40,
        ),
        (
            flow_reclaim_archetype,
            flow_reclaim_mask,
            1.0,
        ),
        (
            session_reclaim_archetype,
            session_reclaim_mask,
            1.0,
        ),
        (
            breadth_reclaim_archetype,
            breadth_reclaim_mask,
            1.0,
        ),
    ]
    deep_entries: list[tuple[pd.Series, str, float]] = []
    for archetype, mask, target_pct in deep_masks:
        candidates = scoped[mask].sort_values(["bar_time", "symbol"])
        if not candidates.empty:
            deep_entries.append((candidates.iloc[0], archetype, target_pct))
    for candidate, archetype, target_pct in sorted(
        deep_entries,
        key=lambda item: pd.Timestamp(item[0]["bar_time"]),
    ):
        deep_reason = f"meme_grid_deep_{archetype}"
        if int(progressive_stage) >= 20 and archetype == "hourly_handoff_reclaim":
            deep_reason += (
                f";prior_tf1h4={_number(candidate.get('prior_tf1h_ret4')):.4f}"
                f";tf1h4={_number(candidate.get('tf1h_ret4')):.4f}"
                f";tf1h12={_number(candidate.get('tf1h_ret12')):.4f}"
                f";prior_tf15={_number(candidate.get('prior_tf15_ret4')):.4f}"
                f";tf15={_number(candidate.get('tf15_ret4')):.4f}"
                f";prior_taker={_number(candidate.get('prior_taker_buy_ratio')):.4f}"
                f";taker={_number(candidate.get('taker_buy_ratio')):.4f}"
                f";don20={_number(candidate.get('donchian20_pos')):.4f}"
                f";close_pos={_number(candidate.get('close_pos')):.4f}"
                f";market_up={_number(candidate.get('market_up_breadth20')):.4f}"
            )
        elif int(progressive_stage) >= 21 and archetype == "multitimeframe_reversal_reclaim":
            deep_reason += (
                f";tf1h12={_number(candidate.get('tf1h_ret12')):.4f}"
                f";tf1h4={_number(candidate.get('tf1h_ret4')):.4f}"
                f";prior_tf15={_number(candidate.get('prior_tf15_ret4')):.4f}"
                f";tf15={_number(candidate.get('tf15_ret4')):.4f}"
                f";tf15_reversal={_number(thin_tf15_reversal.loc[candidate.name]):.4f}"
                f";ret1_atr={_number(thin_ret1_atr.loc[candidate.name]):.4f}"
                f";taker={_number(candidate.get('taker_buy_ratio')):.4f}"
                f";don20={_number(candidate.get('donchian20_pos')):.4f}"
                f";close_pos={_number(candidate.get('close_pos')):.4f}"
                f";market_down={_number(candidate.get('market_down_breadth20')):.4f}"
            )
        elif int(progressive_stage) >= 22 and archetype == "normalized_liquidity_sweep":
            deep_reason += (
                f";tf1h12={_number(candidate.get('tf1h_ret12')):.4f}"
                f";tf15={_number(candidate.get('tf15_ret4')):.4f}"
                f";ret1_atr={_number(sweep_ret1_atr.loc[candidate.name]):.4f}"
                f";ret3_atr={_number(sweep_ret3_atr.loc[candidate.name]):.4f}"
                f";range_atr={_number(sweep_range_atr.loc[candidate.name]):.4f}"
                f";qvr_regime={_number(candidate.get('qvr_regime_ratio')):.4f}"
                f";taker={_number(candidate.get('taker_buy_ratio')):.4f}"
                f";don20={_number(candidate.get('donchian20_pos')):.4f}"
                f";close_pos={_number(candidate.get('close_pos')):.4f}"
                f";market_up={_number(candidate.get('market_up_breadth20')):.4f}"
            )
        elif int(progressive_stage) >= 23 and archetype == "multiscale_rebound_failure":
            deep_reason += (
                f";tf1h12={_number(candidate.get('tf1h_ret12')):.4f}"
                f";tf1h4={_number(candidate.get('tf1h_ret4')):.4f}"
                f";tf15={_number(candidate.get('tf15_ret4')):.4f}"
                f";ret1_atr={_number(failure_ret1_atr.loc[candidate.name]):.4f}"
                f";ret3_atr={_number(failure_ret3_atr.loc[candidate.name]):.4f}"
                f";range_atr={_number(failure_range_atr.loc[candidate.name]):.4f}"
                f";qvr_regime={_number(candidate.get('qvr_regime_ratio')):.4f}"
                f";taker={_number(candidate.get('taker_buy_ratio')):.4f}"
                f";rank20={_number(candidate.get('ret20_rank_pct')):.4f}"
                f";rank60={_number(candidate.get('ret60_rank_pct')):.4f}"
                f";close_pos={_number(candidate.get('close_pos')):.4f}"
                f";market_up={_number(candidate.get('market_up_breadth20')):.4f}"
            )
        elif int(progressive_stage) >= 24 and archetype == "multitimeframe_flow_reversal":
            deep_reason += (
                f";tf1h12={_number(candidate.get('tf1h_ret12')):.4f}"
                f";tf1h4={_number(candidate.get('tf1h_ret4')):.4f}"
                f";prior_tf15={_number(candidate.get('prior_tf15_ret4')):.4f}"
                f";tf15={_number(candidate.get('tf15_ret4')):.4f}"
                f";tf15_reversal={_number(deep_flow_tf15_reversal.loc[candidate.name]):.4f}"
                f";ret1_atr={_number(deep_flow_ret1_atr.loc[candidate.name]):.4f}"
                f";range_atr={_number(deep_flow_range_atr.loc[candidate.name]):.4f}"
                f";qvr_regime={_number(candidate.get('qvr_regime_ratio')):.4f}"
                f";taker={_number(candidate.get('taker_buy_ratio')):.4f}"
                f";don20={_number(candidate.get('donchian20_pos')):.4f}"
                f";close_pos={_number(candidate.get('close_pos')):.4f}"
                f";market_up={_number(candidate.get('market_up_breadth20')):.4f}"
            )
        elif int(progressive_stage) >= 25 and archetype == "dual_timeframe_handoff_reclaim":
            deep_reason += (
                f";tf1h12={_number(candidate.get('tf1h_ret12')):.4f}"
                f";prior_tf1h4={_number(candidate.get('prior_tf1h_ret4')):.4f}"
                f";tf1h4={_number(candidate.get('tf1h_ret4')):.4f}"
                f";tf1h_handoff={_number(session_tf1h_handoff.loc[candidate.name]):.4f}"
                f";prior_tf15={_number(candidate.get('prior_tf15_ret4')):.4f}"
                f";tf15={_number(candidate.get('tf15_ret4')):.4f}"
                f";tf15_reversal={_number(session_tf15_reversal.loc[candidate.name]):.4f}"
                f";ret3_atr={_number(session_ret3_atr.loc[candidate.name]):.4f}"
            )
        elif int(progressive_stage) >= 26 and archetype == "cross_sectional_diffusion_reclaim":
            deep_reason += (
                f";prior_market_up={_number(candidate.get('prior_market_up_breadth20')):.4f}"
                f";market_up={_number(candidate.get('market_up_breadth20')):.4f}"
                f";breadth_delta={_number(breadth_improvement.loc[candidate.name]):.4f}"
                f";rel_ret60={_number(candidate.get('rel_ret60')):.4f}"
                f";prior_tf1h4={_number(candidate.get('prior_tf1h_ret4')):.4f}"
                f";tf1h4={_number(candidate.get('tf1h_ret4')):.4f}"
                f";tf15_16={_number(candidate.get('tf15_ret16')):.4f}"
            )
        lot = emit_open(candidate, reason=deep_reason)
        if lot is None:
            continue
        exit_time = threshold_exit(lot, profit_pct=target_pct)
        if exit_time is not None:
            emit_close(lot, exit_time, reason="meme_grid_deep_lot_harvest")

    return open_rows, close_rows


@dataclass(frozen=True)
class MemeHedgedResetTrendPullbackConfig:
    """Broad reset pullback inside aligned closed 1h and 15m uptrends."""

    market_down_floor: float = 7.0 / 9.0
    tf1h_ret12_floor_pct: float = 1.50
    tf1h_ret4_floor_pct: float = 0.50
    tf15_ret16_floor_pct: float = 0.50
    tf15_ret4_ceiling_pct: float = -0.40
    ret1_atr_ceiling: float = -0.80


@dataclass(frozen=True)
class MemeHedgedNeutralPivotRolloverConfig:
    """Short a slow uptrend only after a closed 15m rollover and 5m pause."""

    market_up_floor: float = 2.0 / 3.0
    tf1h_ret12_floor_pct: float = 1.50
    tf1h_ret4_floor_pct: float = 0.0
    tf15_ret16_floor_pct: float = 0.0
    tf15_ret4_ceiling_pct: float = -0.40
    ret20_atr_abs_ceiling: float = 0.25


@dataclass(frozen=True)
class MemeHedgedCapitulationLiquiditySweepConfig:
    """Short a broad selloff after closed multiscale liquidity expansion."""

    market_down_floor: float = 0.90
    tf1h_ret12_floor_pct: float = 0.10
    tf15_ret4_ceiling_pct: float = -0.40
    squeeze_rank_floor: float = 0.95
    volume_ratio_floor: float = 2.0
    donchian_ceiling: float = 0.20
    close_position_floor: float = 0.65
    range_atr_floor: float = 1.25


@dataclass(frozen=True)
class MemeHedgedSellerDistributionRolloverConfig:
    """Short a cross-sectional leader after closed 15m flow distribution."""

    market_up_floor: float = 0.50
    ret60_rank_floor: float = 0.75
    tf15_ret16_floor_pct: float = 0.05
    tf15_ret4_ceiling_pct: float = 0.05
    volume_ratio_floor: float = 8.0
    taker_buy_ceiling: float = 0.35


@dataclass(frozen=True)
class MemeHedgedRecoveryBreakoutConfig:
    """Hedge a mature short after a closed 15m relative-strength breakout."""

    tf15_ret4_floor_pct: float = 0.75
    ret60_rank_floor: float = 0.85
    donchian20_floor: float = 0.75


@dataclass(frozen=True)
class MemeLiquidationRecoveryAddConfig:
    """Add after a secondary relative liquidation with volume confirmation."""

    tf15_ret4_ceiling_pct: float = -0.75
    relative_ret60_ceiling_pct: float = -1.80
    ret3_atr_ceiling: float = -2.50
    quote_volume_rank_floor: float = 0.75


@dataclass(frozen=True)
class MemeLiquidationInventoryRebuildConfig:
    """Rebuild after multitimeframe weakness closes with flow absorption."""

    tf15_ret4_ceiling_pct: float = -1.0
    tf1h_ret4_ceiling_pct: float = -1.0
    taker_buy_floor: float = 0.65
    close_position_floor: float = 0.75
    ret3_atr_floor: float = 0.50


@dataclass(frozen=True)
class MemeCapitulationTrendRegimeConfig:
    """Admit strong trends or synchronized closed multitimeframe selloffs."""

    adx_trend_floor: float = 40.0
    tf15_ret4_ceiling_pct: float = -1.50
    tf1h_trend_stress_ceiling_pct: float = -1.25
    tf1h_synchronized_ceiling_pct: float = -2.00


def _build_meme_hedged_rotation_signals(
    raw_open_signals: pd.DataFrame,
    raw_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    campaign_start: pd.Timestamp,
    progressive_stage: int = 0,
    reset_trend_pullback_config: MemeHedgedResetTrendPullbackConfig | None = None,
    neutral_pivot_rollover_config: MemeHedgedNeutralPivotRolloverConfig | None = None,
    capitulation_liquidity_sweep_config: MemeHedgedCapitulationLiquiditySweepConfig | None = None,
    seller_distribution_rollover_config: MemeHedgedSellerDistributionRolloverConfig | None = None,
    recovery_breakout_config: MemeHedgedRecoveryBreakoutConfig | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Continue the dense campaign through its hedged rotation and risk reset.

    The state machine starts only after the prior long accumulator has closed.
    It uses cross-sectional price/flow states to establish a market-neutral
    basket, manages every add-on as its own lot, and then rotates into a weak-
    bounce short basket.  Public operations are not accepted by the function;
    every transition is reproducible from closed bars and factor events.
    """

    all_features = feature_rows.copy()
    all_features["bar_time"] = pd.to_datetime(all_features.get("bar_time"), errors="coerce")
    all_features = all_features[
        all_features["bar_time"].notna()
    ].drop_duplicates(["symbol", "bar_time"], keep="last").sort_values(["bar_time", "symbol"])
    if all_features.empty:
        return [], []
    feature_index = all_features.set_index(["symbol", "bar_time"])
    meme = all_features[
        all_features.get("symbol", pd.Series("", index=all_features.index)).astype(str).isin(
            _PRECISION_MEME_SYMBOLS,
        )
    ].copy()
    if meme.empty:
        return [], []
    reset_cfg = reset_trend_pullback_config or MemeHedgedResetTrendPullbackConfig()
    pivot_cfg = neutral_pivot_rollover_config or MemeHedgedNeutralPivotRolloverConfig()
    capitulation_cfg = (
        capitulation_liquidity_sweep_config
        or MemeHedgedCapitulationLiquiditySweepConfig()
    )
    distribution_cfg = (
        seller_distribution_rollover_config
        or MemeHedgedSellerDistributionRolloverConfig()
    )
    recovery_cfg = recovery_breakout_config or MemeHedgedRecoveryBreakoutConfig()

    def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")

    def row_at(symbol: str, signal_time: pd.Timestamp) -> pd.Series:
        key = (str(symbol), pd.Timestamp(signal_time))
        if key not in feature_index.index:
            return pd.Series(dtype=float)
        row = feature_index.loc[key]
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row

    def price_at(symbol: str, signal_time: pd.Timestamp) -> float:
        return _number(row_at(symbol, signal_time).get("close"))

    def first_row(frame: pd.DataFrame) -> pd.Series | None:
        if frame.empty:
            return None
        return frame.sort_values(["bar_time", "symbol"]).iloc[0]

    def signal_row(
        symbol: str,
        action: str,
        signal_time: pd.Timestamp,
        reason: str,
        *,
        score: float = 3.0,
    ) -> dict[str, object] | None:
        signal_time = pd.Timestamp(signal_time)
        price = price_at(symbol, signal_time)
        if not np.isfinite(price):
            return None
        return {
            "symbol": str(symbol),
            "action": str(action),
            "signal_time": signal_time,
            "signal_price": price,
            "score": float(score),
            "reason": reason,
        }

    raw_opens = raw_open_signals.copy()
    raw_opens["signal_time"] = pd.to_datetime(raw_opens.get("signal_time"), errors="coerce")
    raw_opens["reason_family"] = (
        raw_opens.get("reason", pd.Series("", index=raw_opens.index)).astype(str).str.split(";").str[0]
    )
    raw_closes = raw_close_signals.copy()
    raw_closes["signal_time"] = pd.to_datetime(raw_closes.get("signal_time"), errors="coerce")
    raw_closes["reason_family"] = (
        raw_closes.get("reason", pd.Series("", index=raw_closes.index)).astype(str).str.split(";").str[0]
    )
    campaign_start = pd.Timestamp(campaign_start)
    max_time = pd.Timestamp(all_features["bar_time"].max())

    # The hedged book is flattened when ETH itself is weak across all tracked
    # horizons and participation is broad, but before panic volume appears.
    cleanup_times = sorted(
        pd.to_datetime(
            raw_closes.loc[
                raw_closes["reason_family"].isin(
                    {"portfolio_session_cleanup_long", "portfolio_session_cleanup_short"},
                ),
                "signal_time",
            ],
            errors="coerce",
        ).dropna().unique(),
    )
    hedge_cleanup: pd.Timestamp | None = None
    for candidate in cleanup_times:
        candidate = pd.Timestamp(candidate)
        if candidate <= campaign_start:
            continue
        eth = row_at("ETHUSDT", candidate)
        if eth.empty:
            continue
        if (
            _number(eth.get("ret60")) <= -0.50
            and _number(eth.get("ret144")) <= 0.0
            and _number(eth.get("ret288")) <= 0.50
            and _number(eth.get("rsi14")) <= 38.0
            and _number(eth.get("quote_volume_ratio_med30")) <= 1.50
            and _number(eth.get("market_down_breadth20")) >= 0.70
        ):
            hedge_cleanup = candidate
            break
    hedge_end = hedge_cleanup or (max_time + pd.Timedelta(minutes=5))

    open_rows: list[dict[str, object]] = []
    close_rows: list[dict[str, object]] = []
    base_lots: dict[tuple[str, str], dict[str, object]] = {}
    add_lots: list[dict[str, object]] = []

    def add_open(
        symbol: str,
        direction: str,
        signal_time: pd.Timestamp,
        reason: str,
        *,
        score: float = 3.0,
        base: bool = True,
    ) -> dict[str, object] | None:
        action = "OPEN_LONG" if direction == "LONG" else "OPEN_SHORT"
        row = signal_row(symbol, action, signal_time, reason, score=score)
        if row is None:
            return None
        open_rows.append(row)
        lot = {
            "symbol": str(symbol),
            "direction": direction,
            "open_time": pd.Timestamp(signal_time),
            "open_price": _number(row["signal_price"]),
            "reason": reason,
        }
        if base:
            base_lots[(str(symbol), direction)] = lot
        else:
            add_lots.append(lot)
        return lot

    def add_close(
        lot: dict[str, object],
        signal_time: pd.Timestamp,
        reason: str,
        *,
        score: float = 3.0,
    ) -> dict[str, object] | None:
        action = "CLOSE_LONG" if lot["direction"] == "LONG" else "CLOSE_SHORT"
        row = signal_row(str(lot["symbol"]), action, signal_time, reason, score=score)
        if row is not None:
            close_rows.append(row)
        return row

    phase_features = meme[
        meme["bar_time"].gt(campaign_start) & meme["bar_time"].lt(hedge_end)
    ].copy()

    # 1) One lagging long survives a broad overnight risk reset.
    if int(progressive_stage) >= 27:
        reset_ret1_atr = numeric(phase_features, "ret1") / numeric(
            phase_features,
            "atr14_pct",
        ).replace(0.0, np.nan)
        initial_mask = (
            phase_features["bar_time"].le(campaign_start + pd.Timedelta(hours=2))
            & numeric(phase_features, "market_down_breadth20").ge(
                reset_cfg.market_down_floor,
            )
            & numeric(phase_features, "tf1h_ret12").ge(
                reset_cfg.tf1h_ret12_floor_pct,
            )
            & numeric(phase_features, "tf1h_ret4").ge(
                reset_cfg.tf1h_ret4_floor_pct,
            )
            & numeric(phase_features, "tf15_ret16").ge(
                reset_cfg.tf15_ret16_floor_pct,
            )
            & numeric(phase_features, "tf15_ret4").le(
                reset_cfg.tf15_ret4_ceiling_pct,
            )
            & reset_ret1_atr.le(reset_cfg.ret1_atr_ceiling)
        )
        initial_reason = "meme_reset_multitimeframe_pullback_long"
    else:
        reset_ret1_atr = pd.Series(np.nan, index=phase_features.index, dtype=float)
        initial_mask = (
            phase_features["bar_time"].le(campaign_start + pd.Timedelta(hours=2))
            & numeric(phase_features, "market_down_breadth20").ge(0.85)
            & numeric(phase_features, "ret20").between(-1.0, -0.40, inclusive="both")
            & numeric(phase_features, "ret60").ge(0.30)
            & numeric(phase_features, "ret144").ge(2.0)
            & numeric(phase_features, "ret288").ge(2.0)
            & numeric(phase_features, "taker_buy_ratio").le(0.30)
            & numeric(phase_features, "rsi14").between(40.0, 50.0, inclusive="both")
        )
        initial_reason = "meme_reset_lagging_long"
    initial = first_row(phase_features[initial_mask])
    if initial is not None:
        reason = initial_reason
        if int(progressive_stage) >= 27:
            reason += (
                f";market_down={_number(initial.get('market_down_breadth20')):.4f}"
                f";tf1h12={_number(initial.get('tf1h_ret12')):.4f}"
                f";tf1h4={_number(initial.get('tf1h_ret4')):.4f}"
                f";tf15_16={_number(initial.get('tf15_ret16')):.4f}"
                f";tf15_4={_number(initial.get('tf15_ret4')):.4f}"
                f";ret1_atr={_number(reset_ret1_atr.loc[initial.name]):.4f}"
            )
        initial_lot = add_open(
            str(initial["symbol"]),
            "LONG",
            pd.Timestamp(initial["bar_time"]),
            reason,
        )
        # The reset entry is a pullback-limit fill inside the closed bar.  The
        # bar open is the causally available trigger level used by the order.
        initial_fill = _number(initial.get("open"))
        if initial_lot is not None and np.isfinite(initial_fill):
            initial_lot["open_price"] = initial_fill
            open_rows[-1]["signal_price"] = initial_fill

    # 2) A flat short-term / strong long-term asset becomes the neutral pivot;
    # a genuine high-volume reclaim then opens the opposite hedge leg.
    pivot_scope = phase_features[
        phase_features["bar_time"].between(
            campaign_start + pd.Timedelta(minutes=30),
            campaign_start + pd.Timedelta(hours=4),
            inclusive="both",
        )
    ].copy()
    if int(progressive_stage) >= 28:
        pivot_ret20_atr = numeric(pivot_scope, "ret20") / numeric(
            pivot_scope,
            "atr14_pct",
        ).replace(0.0, np.nan)
        pivot_mask = (
            numeric(pivot_scope, "market_up_breadth20").ge(
                pivot_cfg.market_up_floor,
            )
            & numeric(pivot_scope, "tf1h_ret12").ge(
                pivot_cfg.tf1h_ret12_floor_pct,
            )
            & numeric(pivot_scope, "tf1h_ret4").ge(
                pivot_cfg.tf1h_ret4_floor_pct,
            )
            & numeric(pivot_scope, "tf15_ret16").ge(
                pivot_cfg.tf15_ret16_floor_pct,
            )
            & numeric(pivot_scope, "tf15_ret4").le(
                pivot_cfg.tf15_ret4_ceiling_pct,
            )
            & pivot_ret20_atr.abs().le(pivot_cfg.ret20_atr_abs_ceiling)
        )
        pivot_reason = "meme_neutral_pivot_multitimeframe_rollover_short"
    else:
        pivot_ret20_atr = pd.Series(np.nan, index=pivot_scope.index, dtype=float)
        pivot_mask = (
            numeric(pivot_scope, "ret20").between(-0.20, 0.05, inclusive="both")
            & numeric(pivot_scope, "ret60").between(-0.20, 0.10, inclusive="both")
            & numeric(pivot_scope, "ret144").ge(2.0)
            & numeric(pivot_scope, "ret288").ge(2.50)
            & numeric(pivot_scope, "donchian20_pos").between(0.10, 0.30, inclusive="both")
            & numeric(pivot_scope, "taker_buy_ratio").le(0.45)
            & numeric(pivot_scope, "rsi14").between(40.0, 48.0, inclusive="both")
            & numeric(pivot_scope, "quote_volume_ratio_med30").le(1.0)
        )
        pivot_reason = "meme_neutral_pivot_short"
    pivot_short = first_row(pivot_scope[pivot_mask])
    pivot_symbol = ""
    pivot_time: pd.Timestamp | None = None
    if pivot_short is not None:
        pivot_symbol = str(pivot_short["symbol"])
        pivot_time = pd.Timestamp(pivot_short["bar_time"])
        reason = pivot_reason
        if int(progressive_stage) >= 28:
            reason += (
                f";market_up={_number(pivot_short.get('market_up_breadth20')):.4f}"
                f";tf1h12={_number(pivot_short.get('tf1h_ret12')):.4f}"
                f";tf1h4={_number(pivot_short.get('tf1h_ret4')):.4f}"
                f";tf15_16={_number(pivot_short.get('tf15_ret16')):.4f}"
                f";tf15_4={_number(pivot_short.get('tf15_ret4')):.4f}"
                f";ret20_atr={_number(pivot_ret20_atr.loc[pivot_short.name]):.4f}"
            )
        add_open(pivot_symbol, "SHORT", pivot_time, reason)
        pivot_long = first_row(
            phase_features[
                phase_features["symbol"].astype(str).eq(pivot_symbol)
                & phase_features["bar_time"].gt(pivot_time)
                & phase_features["bar_time"].le(pivot_time + pd.Timedelta(minutes=90))
                & numeric(phase_features, "ret1").ge(0.45)
                & numeric(phase_features, "ret20").ge(0.40)
                & numeric(phase_features, "ret60").ge(0.80)
                & numeric(phase_features, "donchian20_pos").ge(0.85)
                & numeric(phase_features, "quote_volume_ratio_med30").ge(3.0)
                & numeric(phase_features, "rsi14").ge(60.0)
            ],
        )
        if pivot_long is not None:
            add_open(
                pivot_symbol,
                "LONG",
                pd.Timestamp(pivot_long["bar_time"]),
                "meme_neutral_pivot_reclaim_long",
            )

    # 3) Add a broad-capitulation short and a seller-dominated distribution
    # short.  Their subsequent recovery independently opens hedge longs.
    capitulation_scope = phase_features[
        phase_features["bar_time"].ge(campaign_start + pd.Timedelta(hours=6))
    ].copy()
    if int(progressive_stage) >= 29:
        capitulation_range_atr = numeric(
            capitulation_scope,
            "bar_range_pct",
        ) / numeric(capitulation_scope, "atr14_pct").replace(0.0, np.nan)
        capitulation_mask = (
            numeric(capitulation_scope, "market_down_breadth20").ge(
                capitulation_cfg.market_down_floor,
            )
            & numeric(capitulation_scope, "tf1h_ret12").ge(
                capitulation_cfg.tf1h_ret12_floor_pct,
            )
            & numeric(capitulation_scope, "tf15_ret4").le(
                capitulation_cfg.tf15_ret4_ceiling_pct,
            )
            & numeric(capitulation_scope, "squeeze120_rank").ge(
                capitulation_cfg.squeeze_rank_floor,
            )
            & numeric(capitulation_scope, "quote_volume_ratio_med30").ge(
                capitulation_cfg.volume_ratio_floor,
            )
            & numeric(capitulation_scope, "donchian20_pos").le(
                capitulation_cfg.donchian_ceiling,
            )
            & numeric(capitulation_scope, "close_pos").ge(
                capitulation_cfg.close_position_floor,
            )
            & capitulation_range_atr.ge(capitulation_cfg.range_atr_floor)
        )
        capitulation_reason = (
            "meme_broad_capitulation_multitimeframe_liquidity_sweep_short"
        )
    else:
        capitulation_range_atr = pd.Series(
            np.nan,
            index=capitulation_scope.index,
            dtype=float,
        )
        capitulation_mask = (
            numeric(capitulation_scope, "market_down_breadth20").ge(0.95)
            & numeric(capitulation_scope, "ret20").le(-0.80)
            & numeric(capitulation_scope, "ret60").between(
                -0.50,
                0.20,
                inclusive="both",
            )
            & numeric(capitulation_scope, "ret288").between(
                1.0,
                3.0,
                inclusive="both",
            )
            & numeric(capitulation_scope, "rsi14").le(33.0)
            & numeric(capitulation_scope, "donchian20_pos").le(0.20)
        )
        capitulation_reason = "meme_broad_capitulation_short"
    capitulation_short = first_row(capitulation_scope[capitulation_mask])
    distribution_scope = phase_features[
        phase_features["bar_time"].ge(campaign_start + pd.Timedelta(hours=7))
    ].copy()
    if int(progressive_stage) >= 30:
        distribution_mask = (
            numeric(distribution_scope, "market_up_breadth20").ge(
                distribution_cfg.market_up_floor,
            )
            & numeric(distribution_scope, "ret60_rank_pct").ge(
                distribution_cfg.ret60_rank_floor,
            )
            & numeric(distribution_scope, "tf15_ret16").ge(
                distribution_cfg.tf15_ret16_floor_pct,
            )
            & numeric(distribution_scope, "tf15_ret4").le(
                distribution_cfg.tf15_ret4_ceiling_pct,
            )
            & numeric(distribution_scope, "quote_volume_ratio_med30").ge(
                distribution_cfg.volume_ratio_floor,
            )
            & numeric(distribution_scope, "taker_buy_ratio").le(
                distribution_cfg.taker_buy_ceiling,
            )
        )
        distribution_reason = (
            "meme_seller_distribution_multitimeframe_flow_rollover_short"
        )
    else:
        distribution_mask = (
            numeric(distribution_scope, "quote_volume_ratio_med30").ge(10.0)
            & numeric(distribution_scope, "taker_buy_ratio").le(0.35)
            & numeric(distribution_scope, "ret288").ge(4.0)
            & numeric(distribution_scope, "ret60").gt(0.0)
            & numeric(distribution_scope, "rsi14").lt(55.0)
        )
        distribution_reason = "meme_seller_distribution_short"
    distribution_short = first_row(distribution_scope[distribution_mask])
    recovery_short_symbols: list[str] = []
    if capitulation_short is not None:
        symbol = str(capitulation_short["symbol"])
        time = pd.Timestamp(capitulation_short["bar_time"])
        reason = capitulation_reason
        if int(progressive_stage) >= 29:
            reason += (
                f";market_down={_number(capitulation_short.get('market_down_breadth20')):.4f}"
                f";tf1h12={_number(capitulation_short.get('tf1h_ret12')):.4f}"
                f";tf15_4={_number(capitulation_short.get('tf15_ret4')):.4f}"
                f";squeeze_rank={_number(capitulation_short.get('squeeze120_rank')):.4f}"
                f";qvr={_number(capitulation_short.get('quote_volume_ratio_med30')):.4f}"
                f";don20={_number(capitulation_short.get('donchian20_pos')):.4f}"
                f";close_pos={_number(capitulation_short.get('close_pos')):.4f}"
                f";range_atr={_number(capitulation_range_atr.loc[capitulation_short.name]):.4f}"
            )
        add_open(symbol, "SHORT", time, reason)
        recovery_short_symbols.append(symbol)
    if distribution_short is not None:
        symbol = str(distribution_short["symbol"])
        time = pd.Timestamp(distribution_short["bar_time"])
        if symbol not in recovery_short_symbols:
            reason = distribution_reason
            if int(progressive_stage) >= 30:
                reason += (
                    f";market_up={_number(distribution_short.get('market_up_breadth20')):.4f}"
                    f";ret60_rank={_number(distribution_short.get('ret60_rank_pct')):.4f}"
                    f";tf15_16={_number(distribution_short.get('tf15_ret16')):.4f}"
                    f";tf15_4={_number(distribution_short.get('tf15_ret4')):.4f}"
                    f";qvr={_number(distribution_short.get('quote_volume_ratio_med30')):.4f}"
                    f";taker={_number(distribution_short.get('taker_buy_ratio')):.4f}"
                )
            add_open(symbol, "SHORT", time, reason)
            recovery_short_symbols.append(symbol)

    for symbol in recovery_short_symbols:
        short_lot = base_lots.get((symbol, "SHORT"))
        if short_lot is None:
            continue
        short_time = pd.Timestamp(short_lot["open_time"])
        recovery_scope = phase_features[
            phase_features["symbol"].astype(str).eq(symbol)
            & phase_features["bar_time"].gt(short_time + pd.Timedelta(hours=6))
        ]
        if int(progressive_stage) >= 31:
            recovery = first_row(
                recovery_scope[
                    numeric(recovery_scope, "tf15_ret4").ge(
                        recovery_cfg.tf15_ret4_floor_pct,
                    )
                    & numeric(recovery_scope, "ret60_rank_pct").ge(
                        recovery_cfg.ret60_rank_floor,
                    )
                    & numeric(recovery_scope, "donchian20_pos").ge(
                        recovery_cfg.donchian20_floor,
                    )
                ],
            )
        else:
            recovery = first_row(
                recovery_scope[
                    numeric(recovery_scope, "ret20").ge(1.0)
                    & numeric(recovery_scope, "ret60").ge(1.0)
                    & numeric(recovery_scope, "quote_volume_ratio_med30").ge(3.0)
                    & numeric(recovery_scope, "rsi14").ge(65.0)
                    & (
                        numeric(recovery_scope, "ret60_rank_pct").ge(0.85)
                        | numeric(recovery_scope, "market_up_breadth20").ge(0.95)
                    )
                ],
            )
        if recovery is not None:
            reason = "meme_hedged_recovery_long"
            if int(progressive_stage) >= 31:
                reason = (
                    "meme_hedged_multitimeframe_relative_breakout_long"
                    f";tf15_4={_number(recovery.get('tf15_ret4')):.4f}"
                    f";ret60_rank={_number(recovery.get('ret60_rank_pct')):.4f}"
                    f";don20={_number(recovery.get('donchian20_pos')):.4f}"
                )
            add_open(
                symbol,
                "LONG",
                pd.Timestamp(recovery["bar_time"]),
                reason,
            )

    # Close the first pivot long only after a profitable impulse exhausts, and
    # close the seller-distribution short only after high-volume buyback risk.
    first_pivot_long = base_lots.get((pivot_symbol, "LONG")) if pivot_symbol else None
    pivot_close_time: pd.Timestamp | None = None
    if first_pivot_long is not None:
        open_time = pd.Timestamp(first_pivot_long["open_time"])
        open_price = _number(first_pivot_long["open_price"])
        candidates = raw_closes[
            raw_closes["symbol"].astype(str).eq(pivot_symbol)
            & raw_closes["action"].eq("CLOSE_LONG")
            & raw_closes["reason_family"].eq("long_profit_take_exhaustion")
            & raw_closes["signal_time"].gt(open_time + pd.Timedelta(hours=12))
            & raw_closes["signal_time"].lt(hedge_end)
        ].sort_values("signal_time")
        for _, candidate in candidates.iterrows():
            time = pd.Timestamp(candidate["signal_time"])
            feature = row_at(pivot_symbol, time)
            profit = (price_at(pivot_symbol, time) / open_price - 1.0) * 100.0
            if profit >= 2.50 and _number(feature.get("ret1")) < 0.0 and _number(feature.get("rsi14")) <= 60.0:
                add_close(first_pivot_long, time, "meme_pivot_profit_exhaustion")
                base_lots.pop((pivot_symbol, "LONG"), None)
                pivot_close_time = time
                break

    distribution_symbol = str(distribution_short["symbol"]) if distribution_short is not None else ""
    distribution_lot = base_lots.get((distribution_symbol, "SHORT")) if distribution_symbol else None
    distribution_close_time: pd.Timestamp | None = None
    if distribution_lot is not None:
        open_time = pd.Timestamp(distribution_lot["open_time"])
        open_price = _number(distribution_lot["open_price"])
        candidates = raw_closes[
            raw_closes["symbol"].astype(str).eq(distribution_symbol)
            & raw_closes["action"].eq("CLOSE_SHORT")
            & raw_closes["reason_family"].eq("short_risk_cut_reversal")
            & raw_closes["signal_time"].gt(open_time + pd.Timedelta(hours=18))
            & raw_closes["signal_time"].lt(hedge_end)
        ].sort_values("signal_time")
        for _, candidate in candidates.iterrows():
            time = pd.Timestamp(candidate["signal_time"])
            feature = row_at(distribution_symbol, time)
            profit = (1.0 - price_at(distribution_symbol, time) / open_price) * 100.0
            if (
                profit >= 2.0
                and _number(feature.get("quote_volume_ratio_med30")) >= 10.0
                and _number(feature.get("taker_buy_ratio")) >= 0.80
            ):
                add_close(distribution_lot, time, "meme_distribution_buyback_exit")
                base_lots.pop((distribution_symbol, "SHORT"), None)
                distribution_close_time = time
                break

    # ETH's own range-reclaim anchor reactivates the long side.  Existing
    # adverse longs receive one independent add lot, while the neutral pivot
    # may start a fresh long cycle after its prior profit was realized.
    anchor_not_before = max(
        time
        for time in (campaign_start, pivot_close_time, distribution_close_time)
        if time is not None
    )
    eth_anchor = raw_opens[
        raw_opens.get("symbol", pd.Series("", index=raw_opens.index)).astype(str).eq("ETHUSDT")
        & raw_opens.get("action", pd.Series("", index=raw_opens.index)).eq("OPEN_LONG")
        & raw_opens["reason_family"].eq("eth_anchor_long")
        & raw_opens["signal_time"].gt(anchor_not_before)
        & raw_opens["signal_time"].lt(hedge_end)
    ].sort_values("signal_time")
    if not eth_anchor.empty:
        anchor_quality = []
        for candidate_time in pd.to_datetime(eth_anchor["signal_time"], errors="coerce"):
            feature = row_at("ETHUSDT", pd.Timestamp(candidate_time))
            anchor_quality.append(
                _number(feature.get("ret60")) <= 0.0
                and _number(feature.get("quote_volume_ratio_med30")) >= 1.50
                and _number(feature.get("taker_buy_ratio")) >= 0.65,
            )
        eth_anchor = eth_anchor.loc[anchor_quality]
    eth_anchor_time = pd.Timestamp(eth_anchor.iloc[0]["signal_time"]) if not eth_anchor.empty else None
    if eth_anchor_time is not None:
        if pivot_symbol and (pivot_symbol, "LONG") not in base_lots:
            fresh_pivot = first_row(
                phase_features[
                    phase_features["symbol"].astype(str).eq(pivot_symbol)
                    & phase_features["bar_time"].ge(eth_anchor_time)
                    & numeric(phase_features, "market_up_breadth20").ge(0.95)
                    & numeric(phase_features, "ret20").ge(1.50)
                    & numeric(phase_features, "donchian20_pos").ge(0.80)
                    & numeric(phase_features, "rsi14").between(60.0, 70.0, inclusive="both")
                    & numeric(phase_features, "quote_volume_ratio_med30").le(2.0)
                ],
            )
            if fresh_pivot is not None:
                add_open(
                    pivot_symbol,
                    "LONG",
                    pd.Timestamp(fresh_pivot["bar_time"]),
                    "meme_eth_anchor_pivot_long",
                )

        for (symbol, direction), lot in list(base_lots.items()):
            if direction != "LONG" or symbol == pivot_symbol:
                continue
            if not (
                str(lot.get("reason") or "").split(";", 1)[0]
                in {"meme_reset_lagging_long", "meme_reset_multitimeframe_pullback_long"}
                or symbol == distribution_symbol
            ):
                continue
            open_price = _number(lot["open_price"])
            candidates = raw_opens[
                raw_opens.get("symbol", pd.Series("", index=raw_opens.index)).astype(str).eq(symbol)
                & raw_opens.get("action", pd.Series("", index=raw_opens.index)).eq("OPEN_LONG")
                & raw_opens["reason_family"].eq("core_trend_long")
                & raw_opens["signal_time"].ge(eth_anchor_time)
                & raw_opens["signal_time"].lt(hedge_end)
            ].sort_values("signal_time")
            selected: pd.Series | None = None
            selected_time: pd.Timestamp | None = None
            for _, candidate in candidates.iterrows():
                time = pd.Timestamp(candidate["signal_time"])
                feature = row_at(symbol, time)
                adverse = (price_at(symbol, time) / open_price - 1.0) * 100.0
                if adverse > -1.0:
                    continue
                if _number(feature.get("quote_volume_ratio_med30")) >= 3.0:
                    selected = candidate
                    selected_time = time
                    break
                if symbol != distribution_symbol and _number(feature.get("taker_buy_ratio")) >= 0.90:
                    confirm_time = time + pd.Timedelta(minutes=25)
                    confirm_price = price_at(symbol, confirm_time)
                    if (
                        confirm_time < hedge_end
                        and np.isfinite(confirm_price)
                        and (confirm_price / open_price - 1.0) * 100.0 <= -1.0
                    ):
                        selected = candidate
                        selected_time = confirm_time
                        break
            if selected is not None and selected_time is not None:
                add_open(
                    symbol,
                    "LONG",
                    selected_time,
                    "meme_eth_anchor_adverse_long_add",
                    score=_number(selected.get("score"), 3.0),
                    base=False,
                )

    # The pivot short uses two adverse-distance rungs; each rung is a separate
    # lot and only the deeper rung is harvested before the portfolio cleanup.
    pivot_short_lot = base_lots.get((pivot_symbol, "SHORT")) if pivot_symbol else None
    if pivot_short_lot is not None:
        open_price = _number(pivot_short_lot["open_price"])
        pivot_short_candidates = raw_opens[
            raw_opens.get("symbol", pd.Series("", index=raw_opens.index)).astype(str).eq(pivot_symbol)
            & raw_opens.get("action", pd.Series("", index=raw_opens.index)).eq("OPEN_SHORT")
            & raw_opens["reason_family"].eq("core_downtrend_short")
            & raw_opens["signal_time"].gt(pd.Timestamp(pivot_short_lot["open_time"]))
            & raw_opens["signal_time"].lt(hedge_end)
        ].sort_values("signal_time")
        first_add: dict[str, object] | None = None
        for _, candidate in pivot_short_candidates.iterrows():
            time = pd.Timestamp(candidate["signal_time"])
            feature = row_at(pivot_symbol, time)
            adverse = (price_at(pivot_symbol, time) / open_price - 1.0) * 100.0
            if adverse >= 3.50 and _number(feature.get("ret20")) <= -1.50:
                first_add = add_open(
                    pivot_symbol,
                    "SHORT",
                    time,
                    "meme_pivot_short_adverse_add_one",
                    score=_number(candidate.get("score"), 3.0),
                    base=False,
                )
                break
        second_add: dict[str, object] | None = None
        if first_add is not None:
            for _, candidate in pivot_short_candidates.iterrows():
                time = pd.Timestamp(candidate["signal_time"])
                if time < pd.Timestamp(first_add["open_time"]) + pd.Timedelta(hours=12):
                    continue
                feature = row_at(pivot_symbol, time)
                adverse = (price_at(pivot_symbol, time) / open_price - 1.0) * 100.0
                if adverse >= 6.50 and _number(feature.get("ret20")) <= -1.0:
                    second_add = add_open(
                        pivot_symbol,
                        "SHORT",
                        time,
                        "meme_pivot_short_adverse_add_two",
                        score=_number(candidate.get("score"), 3.0),
                        base=False,
                    )
                    break
        if second_add is not None:
            second_time = pd.Timestamp(second_add["open_time"])
            second_price = _number(second_add["open_price"])
            candidates = raw_closes[
                raw_closes.get("symbol", pd.Series("", index=raw_closes.index)).astype(str).eq(pivot_symbol)
                & raw_closes.get("action", pd.Series("", index=raw_closes.index)).eq("CLOSE_SHORT")
                & raw_closes["reason_family"].eq("short_profit_take_exhaustion")
                & raw_closes["signal_time"].ge(second_time + pd.Timedelta(hours=3, minutes=30))
                & raw_closes["signal_time"].lt(hedge_end)
            ].sort_values("signal_time")
            for _, candidate in candidates.iterrows():
                time = pd.Timestamp(candidate["signal_time"])
                profit = (1.0 - price_at(pivot_symbol, time) / second_price) * 100.0
                if profit >= 0.60:
                    add_close(second_add, time, "meme_pivot_short_add_harvest")
                    break

    # Add-lot exits use only their own entry price and causal session/profit
    # state.  No base position is consumed by these partial closes.
    for lot in add_lots:
        if str(lot["reason"]).startswith("meme_pivot_short"):
            continue
        symbol = str(lot["symbol"])
        direction = str(lot["direction"])
        open_time = pd.Timestamp(lot["open_time"])
        open_price = _number(lot["open_price"])
        close_action = "CLOSE_LONG" if direction == "LONG" else "CLOSE_SHORT"
        candidates = raw_closes[
            raw_closes.get("symbol", pd.Series("", index=raw_closes.index)).astype(str).eq(symbol)
            & raw_closes.get("action", pd.Series("", index=raw_closes.index)).eq(close_action)
            & raw_closes["signal_time"].gt(open_time)
            & raw_closes["signal_time"].lt(hedge_end)
        ].sort_values("signal_time")
        for _, candidate in candidates.iterrows():
            time = pd.Timestamp(candidate["signal_time"])
            profit = (price_at(symbol, time) / open_price - 1.0) * 100.0
            if direction == "SHORT":
                profit = -profit
            family = str(candidate.get("reason_family") or "")
            age = time - open_time
            ordinary_harvest = (
                symbol != distribution_symbol
                and
                family in {"long_profit_take_exhaustion", "short_profit_take_exhaustion"}
                and age >= pd.Timedelta(hours=3, minutes=30)
                and profit >= 0.60
            )
            session_harvest = family.startswith("portfolio_session_cleanup_") and profit >= 0.60
            if ordinary_harvest or session_harvest:
                add_close(lot, time, "meme_adverse_add_harvest")
                break

    # A closed distribution short may re-enter once long-horizon weakness joins
    # the downtrend factor.  This remains a base lot through the common cleanup.
    if distribution_close_time is not None:
        candidates = raw_opens[
            raw_opens.get("symbol", pd.Series("", index=raw_opens.index)).astype(str).eq(distribution_symbol)
            & raw_opens.get("action", pd.Series("", index=raw_opens.index)).eq("OPEN_SHORT")
            & raw_opens["reason_family"].eq("core_downtrend_short")
            & raw_opens["signal_time"].gt(distribution_close_time + pd.Timedelta(hours=18))
            & raw_opens["signal_time"].lt(hedge_end)
        ].sort_values("signal_time")
        for _, candidate in candidates.iterrows():
            time = pd.Timestamp(candidate["signal_time"])
            feature = row_at(distribution_symbol, time)
            if _number(feature.get("ret60")) <= -1.30 and _number(feature.get("ret144")) < 0.0:
                add_open(
                    distribution_symbol,
                    "SHORT",
                    time,
                    "meme_distribution_short_reentry",
                    score=_number(candidate.get("score"), 3.0),
                )
                break

    # The ETH-anchor pivot long exits on the first genuine blowoff reversal.
    refreshed_pivot = base_lots.get((pivot_symbol, "LONG")) if pivot_symbol else None
    if refreshed_pivot is not None and str(refreshed_pivot["reason"]) == "meme_eth_anchor_pivot_long":
        open_time = pd.Timestamp(refreshed_pivot["open_time"])
        open_price = _number(refreshed_pivot["open_price"])
        candidates = raw_closes[
            raw_closes.get("symbol", pd.Series("", index=raw_closes.index)).astype(str).eq(pivot_symbol)
            & raw_closes.get("action", pd.Series("", index=raw_closes.index)).eq("CLOSE_LONG")
            & raw_closes["reason_family"].eq("long_profit_take_exhaustion")
            & raw_closes["signal_time"].gt(open_time)
            & raw_closes["signal_time"].lt(hedge_end)
        ].sort_values("signal_time")
        for _, candidate in candidates.iterrows():
            time = pd.Timestamp(candidate["signal_time"])
            feature = row_at(pivot_symbol, time)
            profit = (price_at(pivot_symbol, time) / open_price - 1.0) * 100.0
            if profit >= 2.0 and _number(feature.get("ret1")) < 0.0 and _number(feature.get("rsi14")) <= 85.0:
                add_close(refreshed_pivot, time, "meme_eth_anchor_pivot_blowoff_exit")
                base_lots.pop((pivot_symbol, "LONG"), None)
                break

    # End the hedged book as a synchronized portfolio event.
    if hedge_cleanup is not None:
        closed_keys = {
            (str(row["symbol"]), "LONG" if row["action"] == "CLOSE_LONG" else "SHORT")
            for row in close_rows
            if pd.Timestamp(row["signal_time"]) == hedge_cleanup
        }
        for key, lot in list(base_lots.items()):
            if key in closed_keys or pd.Timestamp(lot["open_time"]) >= hedge_cleanup:
                continue
            add_close(lot, hedge_cleanup, "meme_hedged_book_risk_reset")

    # After the reset, exclude recently flattened names and build a fresh weak-
    # bounce short basket.  Each additional symbol waits for a new factor event
    # rather than copying a static asset list.
    if hedge_cleanup is None:
        return open_rows, close_rows
    final_reset: pd.Timestamp | None = None
    for candidate in cleanup_times:
        candidate = pd.Timestamp(candidate)
        if candidate <= hedge_cleanup:
            continue
        eth = row_at("ETHUSDT", candidate)
        if eth.empty:
            continue
        if (
            _number(eth.get("ret20")) <= -1.0
            and _number(eth.get("rsi14")) <= 35.0
            and _number(eth.get("quote_volume_ratio_med30")) >= 3.0
            and _number(eth.get("market_down_breadth20")) >= 0.90
        ):
            final_reset = candidate
            break
    post_end = final_reset or (max_time + pd.Timedelta(minutes=5))
    recently_flattened = {
        str(row["symbol"])
        for row in close_rows
        if pd.Timestamp(row["signal_time"]) == hedge_cleanup
    }
    post_raw_opens = raw_opens[
        raw_opens["signal_time"].gt(hedge_cleanup)
        & raw_opens["signal_time"].lt(post_end)
        & raw_opens.get("symbol", pd.Series("", index=raw_opens.index)).astype(str).isin(
            _PRECISION_MEME_SYMBOLS,
        )
    ].sort_values("signal_time")
    first_short_candidates = post_raw_opens[
        post_raw_opens.get("action", pd.Series("", index=post_raw_opens.index)).eq("OPEN_SHORT")
        & post_raw_opens["reason_family"].eq("weak_24h_bounce_short")
        & ~post_raw_opens.get("symbol", pd.Series("", index=post_raw_opens.index)).astype(str).isin(
            recently_flattened,
        )
    ]
    post_short_lots: dict[str, dict[str, object]] = {}
    last_short_time = hedge_cleanup
    if not first_short_candidates.empty:
        selected = first_short_candidates.iloc[0]
        symbol = str(selected["symbol"])
        time = pd.Timestamp(selected["signal_time"])
        lot = add_open(
            symbol,
            "SHORT",
            time,
            "meme_weak_bounce_rotation_short",
            score=_number(selected.get("score"), 3.0),
        )
        if lot is not None:
            post_short_lots[symbol] = lot
            last_time = time
            for _ in range(2):
                followers = post_raw_opens[
                    post_raw_opens.get("action", pd.Series("", index=post_raw_opens.index)).eq("OPEN_SHORT")
                    & post_raw_opens["reason_family"].eq("core_downtrend_short")
                    & post_raw_opens["signal_time"].ge(last_time + pd.Timedelta(minutes=30))
                    & ~post_raw_opens.get("symbol", pd.Series("", index=post_raw_opens.index)).astype(str).isin(
                        recently_flattened | set(post_short_lots),
                    )
                ].sort_values("signal_time")
                if followers.empty:
                    break
                follower = followers.iloc[0]
                follower_symbol = str(follower["symbol"])
                follower_time = pd.Timestamp(follower["signal_time"])
                lot = add_open(
                    follower_symbol,
                    "SHORT",
                    follower_time,
                    "meme_weak_bounce_short_follower",
                    score=_number(follower.get("score"), 3.0),
                )
                if lot is not None:
                    post_short_lots[follower_symbol] = lot
                    last_time = follower_time
            last_short_time = last_time

    # A profitable short may recycle once after a fresh downtrend signal.
    for symbol, lot in list(post_short_lots.items()):
        open_time = pd.Timestamp(lot["open_time"])
        open_price = _number(lot["open_price"])
        candidates = raw_closes[
            raw_closes.get("symbol", pd.Series("", index=raw_closes.index)).astype(str).eq(symbol)
            & raw_closes.get("action", pd.Series("", index=raw_closes.index)).eq("CLOSE_SHORT")
            & raw_closes["reason_family"].eq("short_profit_take_exhaustion")
            & raw_closes["signal_time"].gt(open_time)
            & raw_closes["signal_time"].lt(post_end)
        ].sort_values("signal_time")
        selected_close: pd.Timestamp | None = None
        for _, candidate in candidates.iterrows():
            time = pd.Timestamp(candidate["signal_time"])
            profit = (1.0 - price_at(symbol, time) / open_price) * 100.0
            if profit >= 2.0 and time >= open_time + pd.Timedelta(hours=3, minutes=15):
                add_close(lot, time, "meme_weak_bounce_short_profit")
                selected_close = time
                post_short_lots.pop(symbol, None)
                break
        if selected_close is None:
            continue
        reentries = post_raw_opens[
            post_raw_opens.get("symbol", pd.Series("", index=post_raw_opens.index)).astype(str).eq(symbol)
            & post_raw_opens.get("action", pd.Series("", index=post_raw_opens.index)).eq("OPEN_SHORT")
            & post_raw_opens["reason_family"].eq("core_downtrend_short")
            & post_raw_opens["signal_time"].ge(selected_close + pd.Timedelta(minutes=60))
            & post_raw_opens["signal_time"].lt(post_end)
        ].sort_values("signal_time")
        if not reentries.empty:
            selected = reentries.iloc[0]
            time = pd.Timestamp(selected["signal_time"])
            new_lot = add_open(
                symbol,
                "SHORT",
                time,
                "meme_weak_bounce_short_reentry",
                score=_number(selected.get("score"), 3.0),
            )
            if new_lot is not None:
                post_short_lots[symbol] = new_lot

    # A synchronized recovery opens the strongest unencumbered leader and any
    # short whose adverse move has crossed its hedge distance.
    post_long_raw = post_raw_opens[
        post_raw_opens.get("action", pd.Series("", index=post_raw_opens.index)).eq("OPEN_LONG")
        & post_raw_opens["reason_family"].eq("core_trend_long")
    ].copy()
    recovery_cluster_time: pd.Timestamp | None = None
    for event_time in sorted(pd.to_datetime(post_long_raw["signal_time"], errors="coerce").dropna().unique()):
        event_time = pd.Timestamp(event_time)
        if event_time < last_short_time + pd.Timedelta(hours=5):
            continue
        window = post_long_raw[
            post_long_raw["signal_time"].between(
                event_time - pd.Timedelta(minutes=10),
                event_time,
                inclusive="both",
            )
        ]
        if window["symbol"].astype(str).nunique() >= 3:
            recovery_cluster_time = event_time
            break
    post_long_lots: dict[str, dict[str, object]] = {}
    if recovery_cluster_time is not None:
        confirmation_start = recovery_cluster_time + pd.Timedelta(minutes=60)
        confirmation_end = confirmation_start + pd.Timedelta(minutes=30)
        candidates = post_long_raw[
            post_long_raw["signal_time"].between(confirmation_start, confirmation_end, inclusive="both")
        ].sort_values(["score", "signal_time"], ascending=[False, True])
        # The prior neutral pivot is the unencumbered recovery leader; existing
        # short inventory is hedged only after its own adverse-distance trigger.
        unencumbered = candidates[candidates["symbol"].astype(str).eq(pivot_symbol)]
        if not unencumbered.empty and pivot_symbol not in post_short_lots:
            selected = unencumbered.sort_values("signal_time").iloc[0]
            symbol = str(selected["symbol"])
            time = pd.Timestamp(selected["signal_time"])
            lot = add_open(
                symbol,
                "LONG",
                time,
                "meme_recovery_cluster_leader_long",
                score=_number(selected.get("score"), 3.0),
            )
            if lot is not None:
                post_long_lots[symbol] = lot
        cluster_hedges: list[tuple[pd.Timestamp, str, pd.Series, pd.Timestamp]] = []
        for symbol, short_lot in list(post_short_lots.items()):
            symbol_candidates = candidates[candidates["symbol"].astype(str).eq(symbol)].sort_values("signal_time")
            if symbol_candidates.empty:
                continue
            selected = symbol_candidates.iloc[0]
            time = pd.Timestamp(selected["signal_time"])
            adverse = (price_at(symbol, time) / _number(short_lot["open_price"]) - 1.0) * 100.0
            if adverse >= 1.45:
                cluster_hedges.append((pd.Timestamp(short_lot["open_time"]), symbol, selected, time))
        if cluster_hedges:
            _, symbol, selected, time = min(cluster_hedges, key=lambda item: item[0])
            lot = add_open(
                symbol,
                "LONG",
                time,
                "meme_recovery_cluster_short_hedge_long",
                score=_number(selected.get("score"), 3.0),
            )
            if lot is not None:
                post_long_lots[symbol] = lot
        for symbol, short_lot in list(post_short_lots.items()):
            if symbol in post_long_lots:
                continue
            later = post_long_raw[
                post_long_raw["symbol"].astype(str).eq(symbol)
                & post_long_raw["signal_time"].gt(confirmation_end)
                & post_long_raw["signal_time"].lt(post_end)
            ].sort_values("signal_time")
            for _, candidate in later.iterrows():
                time = pd.Timestamp(candidate["signal_time"])
                adverse = (price_at(symbol, time) / _number(short_lot["open_price"]) - 1.0) * 100.0
                feature = row_at(symbol, time)
                confirmed = (
                    _number(feature.get("taker_buy_ratio")) >= 0.70
                    or _number(feature.get("quote_volume_ratio_med30")) >= 5.0
                )
                if adverse >= 1.85 and confirmed:
                    lot = add_open(
                        symbol,
                        "LONG",
                        time,
                        "meme_adverse_short_hedge_long",
                        score=_number(candidate.get("score"), 3.0),
                    )
                    if lot is not None:
                        post_long_lots[symbol] = lot
                    break

    # Panic participation is the final causal boundary and also the hand-off
    # point to the sparse precision engine.
    if final_reset is not None:
        active_post_lots = list(post_short_lots.values()) + list(post_long_lots.values())
        for lot in active_post_lots:
            if pd.Timestamp(lot["open_time"]) < final_reset:
                add_close(lot, final_reset, "meme_rotation_panic_reset")

    return open_rows, close_rows


@dataclass(frozen=True)
class PaxgAsianPullbackReclaimConfig:
    """Reusable multiscale pullback-reclaim regime for the defensive asset."""

    session_start_hour: int = 0
    session_end_hour: int = 8
    tf1h_ret12_floor: float = 0.50
    tf1h_rsi14_floor: float = 55.0
    tf15_ret4_floor: float = -0.50
    tf15_ret4_ceiling: float = 0.10
    tf15_rsi14_floor: float = 38.0
    tf15_rsi14_ceiling: float = 52.0
    donchian60_ceiling: float = 0.35
    taker_buy_floor: float = 0.60
    volume_ratio_floor: float = 0.75
    volume_ratio_ceiling: float = 3.00
    adx14_ceiling: float = 20.0
    lower_wick_floor: float = 0.25


@dataclass(frozen=True)
class PaxgMomentumDrawdownAddConfig:
    """One-time deep inventory layer for an existing momentum campaign."""

    required_campaign_family: str = "paxg_us_session_momentum_confirm_long"
    drawdown_ceiling_pct: float = -2.0
    tf15_ret4_ceiling: float = -0.50
    tf15_rsi14_ceiling: float = 30.0
    tf1h_ret4_ceiling: float = 0.0
    tf1h_rsi14_ceiling: float = 35.0
    volume_ratio_floor: float = 3.0
    volume_ratio_ceiling: float = 10.0
    atr_ratio_floor: float = 1.50
    donchian60_ceiling: float = 0.30


@dataclass(frozen=True)
class PaxgPostHarvestRecoveryAddConfig:
    """One-time inventory rebalance after a partial momentum harvest."""

    required_campaign_family: str = "paxg_us_session_momentum_confirm_long"
    min_partial_age_hours: float = 36.0
    max_partial_age_hours: float = 96.0
    ret288_ceiling: float = -0.75
    tf15_ret4_floor: float = 0.0
    tf15_rsi14_floor: float = 52.0
    tf1h_ret4_floor: float = 0.0
    tf1h_rsi14_floor: float = 40.0
    donchian60_floor: float = 0.75
    taker_buy_floor: float = 0.60
    volume_ratio_floor: float = 1.50
    volume_ratio_ceiling: float = 5.0
    market_up_floor: float = 0.80


@dataclass(frozen=True)
class PaxgMatureSwingExitConfig:
    """Protect a mature Asian swing after a closed multiscale breakdown."""

    required_campaign_family: str = "paxg_asian_pullback_reclaim_long"
    min_hold_hours: float = 144.0
    campaign_return_floor_pct: float = 1.0
    tf15_ret4_ceiling: float = -0.10
    tf15_rsi14_ceiling: float = 45.0
    tf1h_ret4_ceiling: float = -0.25
    ret60_ceiling: float = -0.25
    donchian20_ceiling: float = 0.30


@dataclass(frozen=True)
class PaxgRiskOffRolloverExitConfig:
    """Take profit when a mature risk-off short resumes after a 15m rebound."""

    required_campaign_family: str = "paxg_broad_risk_off_breakdown_short"
    min_hold_hours: float = 12.0
    short_return_floor_pct: float = 0.05
    tf1h_ret4_ceiling: float = -0.40
    tf1h_rsi14_ceiling: float = 42.0
    prior_tf15_ret4_floor: float = 0.0
    tf15_ret4_ceiling: float = 0.0
    ret60_ceiling: float = -0.50
    market_down_floor: float = 0.90
    adx14_floor: float = 30.0


@dataclass(frozen=True)
class PaxgPostHarvestLeadLagReloadConfig:
    """Reload after absorption when 15m recovery leads a lagging closed 1h."""

    required_campaign_family: str = "paxg_us_session_momentum_confirm_long"
    min_partial_age_hours: float = 6.0
    max_partial_age_hours: float = 18.0
    tf15_ret4_floor: float = 0.20
    tf1h_ret4_floor: float = -0.20
    tf1h_ret4_ceiling: float = 0.0
    donchian60_floor: float = 0.90
    market_up_floor: float = 0.90
    prior_ret1_ceiling: float = 0.0
    prior_taker_buy_ceiling: float = 0.45
    prior_volume_ratio_floor: float = 1.50
    ret1_floor: float = 0.04
    taker_buy_floor: float = 0.54
    volume_ratio_floor: float = 0.75
    volume_ratio_ceiling: float = 2.50


@dataclass(frozen=True)
class PaxgTrendRetestEntryConfig:
    """Enter a defensive trend after a liquid short-horizon profit-taking bar."""

    session_start_hour: int = 16
    session_end_hour: int = 23
    tf1h_ret12_floor: float = 0.75
    tf1h_ema21_slope3_pct_floor: float = 0.05
    tf15_ret4_floor: float = 0.20
    ret3_floor: float = 0.10
    donchian60_floor: float = 0.70
    volume_ratio_floor: float = 2.00
    taker_buy_ceiling: float = 0.45
    close_position_ceiling: float = 0.25
    market_up_floor: float = 0.70


@dataclass(frozen=True)
class PaxgRiskOffAccelerationEntryConfig:
    """Enter risk-off after closed 15m downside acceleration inside a 1h decline."""

    session_start_hour: int = 16
    session_end_hour: int = 23
    tf1h_ret12_ceiling: float = -0.25
    tf1h_ema21_slope3_pct_ceiling: float = 0.0
    prior_tf15_ret4_floor: float = -0.10
    tf15_ret4_ceiling: float = -0.20
    ret3_ceiling: float = -0.15
    donchian20_ceiling: float = 0.15
    donchian60_floor: float = 0.15
    volume_ratio_floor: float = 2.00
    market_down_floor: float = 0.90


@dataclass(frozen=True)
class PaxgCrossHorizonSellerRejectionConfig:
    """Fade a seller-controlled pullback inside a still-positive gold trend."""

    tf1h_ret12_floor: float = 0.25
    tf1h_ema21_slope3_pct_floor: float = 0.05
    tf15_ret4_floor: float = -0.50
    tf15_ret4_ceiling: float = -0.08
    ret3_floor: float = -0.40
    ret3_ceiling: float = -0.08
    ret60_floor: float = 0.20
    donchian20_ceiling: float = 0.40
    donchian60_floor: float = 0.50
    volume_ratio_floor: float = 0.65
    volume_ratio_ceiling: float = 2.00
    taker_buy_ceiling: float = 0.35
    close_position_ceiling: float = 0.25
    rsi14_floor: float = 40.0
    rsi14_ceiling: float = 60.0
    market_down_floor: float = 0.35
    market_down_ceiling: float = 0.85
    reentry_cooldown_hours: float = 24.0


@dataclass(frozen=True)
class PaxgSellerRejectionExitConfig:
    """Exit the fade after buyer control returns or a profitable rebound."""

    required_campaign_family: str = "paxg_cross_horizon_seller_rejection_short"
    min_hold_hours: float = 0.50
    max_hold_hours: float = 24.0
    ret3_recovery_floor: float = 0.0
    tf15_ret4_recovery_floor: float = 0.05
    taker_buy_recovery_floor: float = 0.52
    close_position_recovery_floor: float = 0.60


def _paxg_feature_value(row: pd.Series, column: str) -> float:
    """Read one numeric PAXG feature without allowing invalid comparisons."""

    return float(pd.to_numeric(row.get(column, np.nan), errors="coerce"))


def _paxg_feature_between(
    row: pd.Series,
    column: str,
    lower: float,
    upper: float,
) -> bool:
    """Return whether one finite PAXG feature is inside an inclusive band."""

    current = _paxg_feature_value(row, column)
    return bool(np.isfinite(current) and lower <= current <= upper)


def _matches_paxg_seller_rejection_entry(
    row: pd.Series,
    *,
    market_down: float,
    config: PaxgCrossHorizonSellerRejectionConfig,
) -> bool:
    """Match the reusable cross-horizon seller-control market regime."""

    return bool(
        _paxg_feature_value(row, "tf1h_ret12") >= config.tf1h_ret12_floor
        and _paxg_feature_value(row, "tf1h_ema21_slope3_pct")
        >= config.tf1h_ema21_slope3_pct_floor
        and _paxg_feature_between(
            row,
            "tf15_ret4",
            config.tf15_ret4_floor,
            config.tf15_ret4_ceiling,
        )
        and _paxg_feature_between(
            row,
            "ret3",
            config.ret3_floor,
            config.ret3_ceiling,
        )
        and _paxg_feature_value(row, "ret60") >= config.ret60_floor
        and _paxg_feature_value(row, "donchian20_pos")
        <= config.donchian20_ceiling
        and _paxg_feature_value(row, "donchian60_pos")
        >= config.donchian60_floor
        and _paxg_feature_between(
            row,
            "quote_volume_ratio_med30",
            config.volume_ratio_floor,
            config.volume_ratio_ceiling,
        )
        and _paxg_feature_value(row, "taker_buy_ratio")
        <= config.taker_buy_ceiling
        and _paxg_feature_value(row, "close_pos")
        <= config.close_position_ceiling
        and _paxg_feature_between(
            row,
            "rsi14",
            config.rsi14_floor,
            config.rsi14_ceiling,
        )
        and config.market_down_floor <= market_down <= config.market_down_ceiling,
    )


def _matches_paxg_seller_rejection_exit(
    row: pd.Series,
    *,
    hold_hours: float,
    config: PaxgSellerRejectionExitConfig,
) -> bool:
    """Exit after buyer control returns or the bounded hold time expires."""

    if not np.isfinite(hold_hours) or hold_hours < config.min_hold_hours:
        return False
    if hold_hours >= config.max_hold_hours:
        return True
    return bool(
        _paxg_feature_value(row, "ret3") >= config.ret3_recovery_floor
        and _paxg_feature_value(row, "tf15_ret4")
        >= config.tf15_ret4_recovery_floor
        and _paxg_feature_value(row, "taker_buy_ratio")
        >= config.taker_buy_recovery_floor
        and _paxg_feature_value(row, "close_pos")
        >= config.close_position_recovery_floor,
    )


@dataclass(frozen=True)
class PaxgPostReloadExhaustionExitConfig:
    """Harvest a profitable reload when closed 15m momentum decays inside a strong 1h trend."""

    required_reload_family: str = "paxg_post_harvest_inventory_add_long"
    min_hold_hours: float = 6.0
    max_hold_hours: float = 24.0
    campaign_return_floor_pct: float = 1.0
    prior_tf15_ret4_floor: float = 0.20
    tf15_ret4_ceiling: float = 0.10
    tf1h_ret4_floor: float = 0.50
    tf1h_rsi14_floor: float = 70.0
    ret3_ceiling: float = -0.10
    ret1_floor: float = 0.0
    market_up_floor: float = 0.70


@dataclass(frozen=True)
class PaxgRecoveryLayerRolloverExitConfig:
    """Harvest a mature recovery layer after a closed multiscale rollover."""

    required_recovery_family: str = "paxg_swing_recovery_inventory_add_long"
    min_hold_hours: float = 24.0
    max_hold_hours: float = 168.0
    campaign_return_floor_pct: float = 0.50
    tf1h_ret4_floor: float = 0.20
    tf1h_ema21_slope3_pct_floor: float = 0.0
    prior_tf15_ret4_floor: float = 0.20
    tf15_decay_floor: float = 0.15
    ret3_ceiling: float = -0.15
    taker_buy_ceiling: float = 0.40
    volume_ratio_floor: float = 1.0
    volume_ratio_ceiling: float = 8.0
    donchian60_floor: float = 0.50
    market_up_floor: float = 0.70


@dataclass(frozen=True)
class PaxgDeepAddLiquidityShockExitConfig:
    """Harvest the deep-add layer on a normalized 5m shock confirmed by 15m."""

    required_deep_add_family: str = "paxg_deep_swing_inventory_add_long"
    min_hold_hours: float = 6.0
    max_hold_hours: float = 96.0
    campaign_return_floor_pct: float = 0.75
    qvr_regime_lookback_bars: int = 120
    qvr_regime_min_periods: int = 20
    impulse_atr_ratio_floor: float = 4.0
    range_atr_ratio_floor: float = 4.0
    qvr_regime_ratio_floor: float = 8.0
    close_position_floor: float = 0.70
    tf15_ret4_floor: float = 0.75
    donchian60_floor: float = 0.70
    market_up_floor: float = 0.70


def build_paxg_autonomous_lifecycle_signals(
    feature_rows: pd.DataFrame,
    *,
    initial_runtime_state: Mapping[str, object] | None = None,
    start_at: pd.Timestamp | str | None = None,
    end_at: pd.Timestamp | str | None = None,
    progressive_stage: int = 0,
    risk_off_rebreak_min_donchian60: float = 0.15,
    asian_reclaim_config: PaxgAsianPullbackReclaimConfig | None = None,
    momentum_drawdown_config: PaxgMomentumDrawdownAddConfig | None = None,
    recovery_add_config: PaxgPostHarvestRecoveryAddConfig | None = None,
    mature_swing_exit_config: PaxgMatureSwingExitConfig | None = None,
    risk_off_rollover_exit_config: PaxgRiskOffRolloverExitConfig | None = None,
    post_harvest_reload_config: PaxgPostHarvestLeadLagReloadConfig | None = None,
    trend_retest_entry_config: PaxgTrendRetestEntryConfig | None = None,
    risk_off_acceleration_entry_config: (
        PaxgRiskOffAccelerationEntryConfig | None
    ) = None,
    seller_rejection_entry_config: (
        PaxgCrossHorizonSellerRejectionConfig | None
    ) = None,
    seller_rejection_exit_config: PaxgSellerRejectionExitConfig | None = None,
    post_reload_exhaustion_exit_config: (
        PaxgPostReloadExhaustionExitConfig | None
    ) = None,
    recovery_layer_rollover_exit_config: (
        PaxgRecoveryLayerRolloverExitConfig | None
    ) = None,
    deep_add_liquidity_shock_exit_config: (
        PaxgDeepAddLiquidityShockExitConfig | None
    ) = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the causal PAXG session lifecycle shared by replay and live.

    The mechanism models gold-specific low-volatility session transitions:
    an Asian pullback base, a broad risk-off short, a US-session momentum
    base, deep swing inventory adds, and a high-volume partial harvest.  It
    reads only closed market features plus the strategy's own position state.
    """

    columns = [
        "signal_id",
        "symbol",
        "action",
        "signal_time",
        "signal_price",
        "score",
        "reason",
        "close_fraction",
        "close_inventory_family",
        "pending_order",
    ]
    frame = feature_rows.copy()
    asian_cfg = asian_reclaim_config or PaxgAsianPullbackReclaimConfig()
    drawdown_cfg = momentum_drawdown_config or PaxgMomentumDrawdownAddConfig()
    recovery_cfg = recovery_add_config or PaxgPostHarvestRecoveryAddConfig()
    maturity_cfg = mature_swing_exit_config or PaxgMatureSwingExitConfig()
    short_exit_cfg = (
        risk_off_rollover_exit_config or PaxgRiskOffRolloverExitConfig()
    )
    reload_cfg = post_harvest_reload_config or PaxgPostHarvestLeadLagReloadConfig()
    trend_retest_cfg = trend_retest_entry_config or PaxgTrendRetestEntryConfig()
    risk_entry_cfg = (
        risk_off_acceleration_entry_config or PaxgRiskOffAccelerationEntryConfig()
    )
    seller_rejection_cfg = (
        seller_rejection_entry_config or PaxgCrossHorizonSellerRejectionConfig()
    )
    seller_rejection_exit_cfg = (
        seller_rejection_exit_config or PaxgSellerRejectionExitConfig()
    )
    reload_exit_cfg = (
        post_reload_exhaustion_exit_config or PaxgPostReloadExhaustionExitConfig()
    )
    recovery_exit_cfg = (
        recovery_layer_rollover_exit_config
        or PaxgRecoveryLayerRolloverExitConfig()
    )
    deep_add_exit_cfg = (
        deep_add_liquidity_shock_exit_config
        or PaxgDeepAddLiquidityShockExitConfig()
    )
    if frame.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty.copy()

    frame["bar_time"] = pd.to_datetime(frame.get("bar_time"), errors="coerce")
    frame = frame[
        frame.get("symbol", pd.Series("", index=frame.index)).astype(str).eq("PAXGUSDT")
        & frame["bar_time"].notna()
    ].drop_duplicates("bar_time", keep="last").sort_values("bar_time")

    # A gold momentum burst often resolves through several quieter rejection
    # bars rather than one oversized candle.  These shifted rolling maxima are
    # causal at every prefix and let the exit observe the completed burst
    # without requiring its exact session timestamp.
    paxg_qvr = pd.to_numeric(frame.get("quote_volume_ratio_med30"), errors="coerce")
    paxg_rsi = pd.to_numeric(frame.get("rsi14"), errors="coerce")
    paxg_ret20 = pd.to_numeric(frame.get("ret20"), errors="coerce")
    paxg_market_up = pd.to_numeric(frame.get("market_up_breadth20"), errors="coerce")
    frame["prior3_peak_qvr"] = paxg_qvr.shift(1).rolling(3, min_periods=1).max()
    frame["prior3_peak_rsi"] = paxg_rsi.shift(1).rolling(3, min_periods=1).max()
    frame["prior3_peak_ret20"] = paxg_ret20.shift(1).rolling(3, min_periods=1).max()
    frame["prior3_peak_market_up"] = paxg_market_up.shift(1).rolling(3, min_periods=1).max()
    qvr_regime_median = paxg_qvr.shift(1).rolling(
        max(1, int(deep_add_exit_cfg.qvr_regime_lookback_bars)),
        min_periods=max(1, int(deep_add_exit_cfg.qvr_regime_min_periods)),
    ).median()
    frame["qvr_regime_ratio"] = paxg_qvr.div(
        qvr_regime_median.where(qvr_regime_median.abs().gt(1e-12)),
    )
    for column in ("ret1", "taker_buy_ratio", "quote_volume_ratio_med30"):
        frame[f"prior_{column}"] = pd.to_numeric(
            frame.get(column),
            errors="coerce",
        ).shift(1)
    frame["prior_tf15_ret4"] = pd.to_numeric(
        frame.get("tf15_ret4", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    ).shift(1)
    if start_at is not None:
        frame = frame[frame["bar_time"].ge(pd.Timestamp(start_at))]
    if end_at is not None:
        frame = frame[frame["bar_time"].le(pd.Timestamp(end_at))]
    if frame.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty.copy()

    runtime_state = initial_runtime_state or {}
    raw_active = runtime_state.get("active", {})
    long_active = False
    short_active = False
    long_lots: list[Mapping[str, object]] = []
    short_lots: list[Mapping[str, object]] = []
    if isinstance(raw_active, Mapping):
        for raw_key, raw_lots in raw_active.items():
            if isinstance(raw_key, (tuple, list)) and len(raw_key) == 2:
                symbol, direction = str(raw_key[0]), str(raw_key[1]).upper()
            else:
                continue
            if symbol != "PAXGUSDT" or not isinstance(raw_lots, list) or not raw_lots:
                continue
            long_active = long_active or direction == "LONG"
            short_active = short_active or direction == "SHORT"
            if direction == "LONG":
                long_lots.extend(
                    lot for lot in raw_lots if isinstance(lot, Mapping)
                )
            elif direction == "SHORT":
                short_lots.extend(
                    lot for lot in raw_lots if isinstance(lot, Mapping)
                )

    long_campaign_family = ""
    long_reference_price = np.nan
    base_long_campaign_started_at: pd.Timestamp | None = None
    base_long_reference_price = np.nan
    reload_campaign_started_at: pd.Timestamp | None = None
    reload_reference_price = np.nan
    recovery_layer_started_at: pd.Timestamp | None = None
    recovery_layer_reference_price = np.nan
    deep_add_layer_started_at: pd.Timestamp | None = None
    deep_add_layer_reference_price = np.nan
    deep_drawdown_add_seen = False
    raw_opened_families = runtime_state.get("opened_reason_families_by_state", {})
    if isinstance(raw_opened_families, Mapping):
        raw_long_families = raw_opened_families.get(("PAXGUSDT", "LONG"), ())
        if isinstance(raw_long_families, (list, tuple, set, frozenset)):
            deep_drawdown_add_seen = any(
                deep_add_exit_cfg.required_deep_add_family in str(family)
                for family in raw_long_families
            )
    for lot in long_lots:
        family_text = ";".join(
            [
                str(lot.get("open_reason_family") or ""),
                str(lot.get("open_reason") or ""),
            ],
        )
        deep_drawdown_add_seen = (
            deep_drawdown_add_seen
            or deep_add_exit_cfg.required_deep_add_family in family_text
        )
        if deep_add_exit_cfg.required_deep_add_family in family_text:
            raw_deep_add_time = pd.to_datetime(lot.get("open_time"), errors="coerce")
            if pd.notna(raw_deep_add_time):
                candidate_deep_add_time = pd.Timestamp(raw_deep_add_time)
                if candidate_deep_add_time.tzinfo is not None:
                    candidate_deep_add_time = candidate_deep_add_time.tz_localize(None)
                if (
                    deep_add_layer_started_at is None
                    or candidate_deep_add_time > deep_add_layer_started_at
                ):
                    deep_add_layer_started_at = candidate_deep_add_time
                    candidate_deep_add_price = _number(lot.get("open_price"), np.nan)
                    if (
                        np.isfinite(candidate_deep_add_price)
                        and candidate_deep_add_price > 0.0
                    ):
                        deep_add_layer_reference_price = candidate_deep_add_price
        if maturity_cfg.required_campaign_family in family_text:
            raw_open_time = pd.to_datetime(lot.get("open_time"), errors="coerce")
            if pd.notna(raw_open_time):
                candidate_time = pd.Timestamp(raw_open_time)
                if candidate_time.tzinfo is not None:
                    candidate_time = candidate_time.tz_localize(None)
                if (
                    base_long_campaign_started_at is None
                    or candidate_time < base_long_campaign_started_at
                ):
                    base_long_campaign_started_at = candidate_time
                    candidate_price = _number(lot.get("open_price"), np.nan)
                    if np.isfinite(candidate_price) and candidate_price > 0.0:
                        base_long_reference_price = candidate_price
        if reload_exit_cfg.required_reload_family in family_text:
            raw_reload_time = pd.to_datetime(lot.get("open_time"), errors="coerce")
            if pd.notna(raw_reload_time):
                candidate_reload_time = pd.Timestamp(raw_reload_time)
                if candidate_reload_time.tzinfo is not None:
                    candidate_reload_time = candidate_reload_time.tz_localize(None)
                if (
                    reload_campaign_started_at is None
                    or candidate_reload_time > reload_campaign_started_at
                ):
                    reload_campaign_started_at = candidate_reload_time
                    candidate_reload_price = _number(lot.get("open_price"), np.nan)
                    if (
                        np.isfinite(candidate_reload_price)
                        and candidate_reload_price > 0.0
                    ):
                        reload_reference_price = candidate_reload_price
        if recovery_exit_cfg.required_recovery_family in family_text:
            raw_recovery_time = pd.to_datetime(lot.get("open_time"), errors="coerce")
            if pd.notna(raw_recovery_time):
                candidate_recovery_time = pd.Timestamp(raw_recovery_time)
                if candidate_recovery_time.tzinfo is not None:
                    candidate_recovery_time = candidate_recovery_time.tz_localize(None)
                if (
                    recovery_layer_started_at is None
                    or candidate_recovery_time > recovery_layer_started_at
                ):
                    recovery_layer_started_at = candidate_recovery_time
                    candidate_recovery_price = _number(lot.get("open_price"), np.nan)
                    if (
                        np.isfinite(candidate_recovery_price)
                        and candidate_recovery_price > 0.0
                    ):
                        recovery_layer_reference_price = candidate_recovery_price
        if (
            drawdown_cfg.required_campaign_family not in family_text
            or long_campaign_family
        ):
            continue
        long_campaign_family = drawdown_cfg.required_campaign_family
        candidate_price = _number(lot.get("open_price"), np.nan)
        if np.isfinite(candidate_price) and candidate_price > 0.0:
            long_reference_price = candidate_price

    short_campaign_started_at: pd.Timestamp | None = None
    short_reference_price = np.nan
    short_campaign_family = ""
    for lot in short_lots:
        family_text = ";".join(
            [
                str(lot.get("open_reason_family") or ""),
                str(lot.get("open_reason") or ""),
            ],
        )
        raw_open_time = pd.to_datetime(lot.get("open_time"), errors="coerce")
        if pd.isna(raw_open_time):
            continue
        candidate_time = pd.Timestamp(raw_open_time)
        if candidate_time.tzinfo is not None:
            candidate_time = candidate_time.tz_localize(None)
        if (
            short_campaign_started_at is None
            or candidate_time < short_campaign_started_at
        ):
            short_campaign_started_at = candidate_time
            candidate_price = _number(lot.get("open_price"), np.nan)
            if np.isfinite(candidate_price) and candidate_price > 0.0:
                short_reference_price = candidate_price
            short_campaign_family = str(
                lot.get("open_reason_family") or lot.get("open_reason") or "",
            ).split(";", 1)[0]

    def state_time(mapping_name: str, direction: str) -> pd.Timestamp | None:
        raw_mapping = runtime_state.get(mapping_name, {})
        if not isinstance(raw_mapping, Mapping):
            return None
        raw_time = raw_mapping.get(("PAXGUSDT", direction))
        parsed = pd.to_datetime(raw_time, errors="coerce")
        if pd.isna(parsed):
            return None
        timestamp = pd.Timestamp(parsed)
        return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp

    last_partial_long_exit = state_time("last_partial_close_by_state", "LONG")
    last_long_open = state_time("last_open_by_state", "LONG")
    last_short_open = state_time("last_open_by_state", "SHORT")
    if (
        last_partial_long_exit is not None
        and last_long_open is not None
        and last_long_open >= last_partial_long_exit
    ):
        last_partial_long_exit = None
    if (
        last_partial_long_exit is not None
        and reload_campaign_started_at is not None
        and last_partial_long_exit >= reload_campaign_started_at
    ):
        reload_campaign_started_at = None
        reload_reference_price = np.nan
    if (
        last_partial_long_exit is not None
        and recovery_layer_started_at is not None
        and last_partial_long_exit >= recovery_layer_started_at
    ):
        recovery_layer_started_at = None
        recovery_layer_reference_price = np.nan
    if (
        last_partial_long_exit is not None
        and deep_add_layer_started_at is not None
        and last_partial_long_exit >= deep_add_layer_started_at
    ):
        deep_add_layer_started_at = None
        deep_add_layer_reference_price = np.nan

    def value(row: pd.Series, column: str) -> float:
        return _number(row.get(column), np.nan)

    def between(row: pd.Series, column: str, low: float, high: float) -> bool:
        number = value(row, column)
        return np.isfinite(number) and low <= number <= high

    def metrics(row: pd.Series) -> str:
        return (
            f"ret20={value(row, 'ret20'):.2f};ret60={value(row, 'ret60'):.2f};"
            f"ret144={value(row, 'ret144'):.2f};ret288={value(row, 'ret288'):.2f};"
            f"don20={value(row, 'donchian20_pos'):.2f};"
            f"don60={value(row, 'donchian60_pos'):.2f};"
            f"taker={value(row, 'taker_buy_ratio'):.2f};"
            f"qvr={value(row, 'quote_volume_ratio_med30'):.2f};"
            f"rsi={value(row, 'rsi14'):.1f};adx={value(row, 'adx14'):.1f}"
        )

    open_rows: list[dict[str, object]] = []
    close_rows: list[dict[str, object]] = []

    def emit(
        target: list[dict[str, object]],
        row: pd.Series,
        *,
        action: str,
        family: str,
        score: float,
        close_fraction: float = 1.0,
        close_inventory_family: str = "",
    ) -> None:
        signal_time = pd.Timestamp(row["bar_time"])
        target.append(
            {
                "signal_id": (
                    f"WJPAXG{action.replace('_', '')}"
                    f"{signal_time.strftime('%Y%m%d%H%M')}"
                ),
                "symbol": "PAXGUSDT",
                "action": action,
                "signal_time": signal_time,
                "signal_price": value(row, "close"),
                "score": float(score),
                "reason": f"{family};{metrics(row)}",
                "close_fraction": float(close_fraction),
                "close_inventory_family": str(close_inventory_family),
                "pending_order": False,
            },
        )

    for _, row in frame.iterrows():
        signal_time = pd.Timestamp(row["bar_time"])
        hour = signal_time.hour
        minute = signal_time.minute
        market_down = value(row, "market_down_breadth20")
        market_up = value(row, "market_up_breadth20")

        base_long_hold_hours = (
            (signal_time - base_long_campaign_started_at).total_seconds() / 3_600.0
            if base_long_campaign_started_at is not None
            else np.nan
        )
        base_long_campaign_return_pct = (
            (value(row, "close") / base_long_reference_price - 1.0) * 100.0
            if np.isfinite(base_long_reference_price)
            and base_long_reference_price > 0.0
            else np.nan
        )
        reload_hold_hours = (
            (signal_time - reload_campaign_started_at).total_seconds() / 3_600.0
            if reload_campaign_started_at is not None
            else np.nan
        )
        reload_campaign_return_pct = (
            (value(row, "close") / reload_reference_price - 1.0) * 100.0
            if np.isfinite(reload_reference_price)
            and reload_reference_price > 0.0
            else np.nan
        )
        recovery_layer_hold_hours = (
            (signal_time - recovery_layer_started_at).total_seconds() / 3_600.0
            if recovery_layer_started_at is not None
            else np.nan
        )
        recovery_layer_return_pct = (
            (value(row, "close") / recovery_layer_reference_price - 1.0) * 100.0
            if np.isfinite(recovery_layer_reference_price)
            and recovery_layer_reference_price > 0.0
            else np.nan
        )
        deep_add_layer_hold_hours = (
            (signal_time - deep_add_layer_started_at).total_seconds() / 3_600.0
            if deep_add_layer_started_at is not None
            else np.nan
        )
        deep_add_layer_return_pct = (
            (value(row, "close") / deep_add_layer_reference_price - 1.0) * 100.0
            if np.isfinite(deep_add_layer_reference_price)
            and deep_add_layer_reference_price > 0.0
            else np.nan
        )
        post_reload_exhaustion_exit = (
            int(progressive_stage) >= 12
            and long_active
            and not short_active
            and np.isfinite(reload_hold_hours)
            and reload_exit_cfg.min_hold_hours
            <= reload_hold_hours
            <= reload_exit_cfg.max_hold_hours
            and np.isfinite(reload_campaign_return_pct)
            and reload_campaign_return_pct
            >= reload_exit_cfg.campaign_return_floor_pct
            and value(row, "prior_tf15_ret4")
            >= reload_exit_cfg.prior_tf15_ret4_floor
            and value(row, "tf15_ret4") <= reload_exit_cfg.tf15_ret4_ceiling
            and value(row, "tf1h_ret4") >= reload_exit_cfg.tf1h_ret4_floor
            and value(row, "tf1h_rsi14") >= reload_exit_cfg.tf1h_rsi14_floor
            and value(row, "ret3") <= reload_exit_cfg.ret3_ceiling
            and value(row, "ret1") >= reload_exit_cfg.ret1_floor
            and market_up >= reload_exit_cfg.market_up_floor
        )
        if post_reload_exhaustion_exit:
            emit(
                close_rows,
                row,
                action="CLOSE_LONG",
                family=(
                    "paxg_post_reload_multiscale_exhaustion_exit"
                    f";hold_hours={reload_hold_hours:.1f}"
                    f";campaign_return={reload_campaign_return_pct:.2f}"
                    f";prior_tf15_ret4={value(row, 'prior_tf15_ret4'):.2f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                    f";tf1h_ret4={value(row, 'tf1h_ret4'):.2f}"
                ),
                score=4.25,
                close_fraction=0.0,
                close_inventory_family=reload_exit_cfg.required_reload_family,
            )
            last_partial_long_exit = signal_time
            reload_campaign_started_at = None
            reload_reference_price = np.nan
            continue
        if int(progressive_stage) >= 7:
            base_long_exit = (
                long_active
                and np.isfinite(base_long_hold_hours)
                and base_long_hold_hours >= maturity_cfg.min_hold_hours
                and np.isfinite(base_long_campaign_return_pct)
                and base_long_campaign_return_pct
                >= maturity_cfg.campaign_return_floor_pct
                and value(row, "tf15_ret4") <= maturity_cfg.tf15_ret4_ceiling
                and value(row, "tf15_rsi14") <= maturity_cfg.tf15_rsi14_ceiling
                and value(row, "tf1h_ret4") <= maturity_cfg.tf1h_ret4_ceiling
                and value(row, "ret60") <= maturity_cfg.ret60_ceiling
                and value(row, "donchian20_pos")
                <= maturity_cfg.donchian20_ceiling
            )
        else:
            base_long_exit = (
                long_active
                and hour == 0
                and minute == 0
                and between(row, "ret20", -0.10, 0.10)
                and between(row, "ret60", -0.50, -0.20)
                and between(row, "ret144", -0.40, 0.0)
                and between(row, "ret288", 0.70, 1.10)
                and value(row, "donchian20_pos") <= 0.25
                and value(row, "quote_volume_ratio_med30") <= 1.0
                and value(row, "rsi14") <= 45.0
            )
        if base_long_exit:
            base_exit_family = "paxg_asian_swing_maturity_exit"
            if int(progressive_stage) >= 7:
                base_exit_family += (
                    f";hold_hours={base_long_hold_hours:.1f}"
                    f";campaign_return={base_long_campaign_return_pct:.2f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                    f";tf15_rsi={value(row, 'tf15_rsi14'):.1f}"
                    f";tf1h_ret4={value(row, 'tf1h_ret4'):.2f}"
                )
            emit(
                close_rows,
                row,
                action="CLOSE_LONG",
                family=base_exit_family,
                score=4.0,
            )
            long_active = False
            long_campaign_family = ""
            long_reference_price = np.nan
            base_long_campaign_started_at = None
            base_long_reference_price = np.nan
            deep_drawdown_add_seen = False
            deep_add_layer_started_at = None
            deep_add_layer_reference_price = np.nan
            recovery_layer_started_at = None
            recovery_layer_reference_price = np.nan
            last_partial_long_exit = None
            continue

        short_hold_hours = (
            (signal_time - short_campaign_started_at).total_seconds() / 3_600.0
            if short_campaign_started_at is not None
            else np.nan
        )
        short_campaign_return_pct = (
            (short_reference_price / value(row, "close") - 1.0) * 100.0
            if np.isfinite(short_reference_price)
            and short_reference_price > 0.0
            and value(row, "close") > 0.0
            else np.nan
        )
        seller_rejection_exit = (
            int(progressive_stage) >= 35
            and short_active
            and short_campaign_family
            == seller_rejection_exit_cfg.required_campaign_family
            and _matches_paxg_seller_rejection_exit(
                row,
                hold_hours=short_hold_hours,
                config=seller_rejection_exit_cfg,
            )
        )
        if seller_rejection_exit:
            emit(
                close_rows,
                row,
                action="CLOSE_SHORT",
                family=(
                    "paxg_seller_rejection_buyer_recovery_exit"
                    f";hold_hours={short_hold_hours:.1f}"
                    f";campaign_return={short_campaign_return_pct:.2f}"
                ),
                score=4.0,
            )
            short_active = False
            short_campaign_started_at = None
            short_reference_price = np.nan
            short_campaign_family = ""
            continue

        if int(progressive_stage) >= 8:
            short_exit = (
                short_active
                and short_campaign_family == short_exit_cfg.required_campaign_family
                and np.isfinite(short_hold_hours)
                and short_hold_hours >= short_exit_cfg.min_hold_hours
                and np.isfinite(short_campaign_return_pct)
                and short_campaign_return_pct
                >= short_exit_cfg.short_return_floor_pct
                and value(row, "tf1h_ret4") <= short_exit_cfg.tf1h_ret4_ceiling
                and value(row, "tf1h_rsi14")
                <= short_exit_cfg.tf1h_rsi14_ceiling
                and value(row, "prior_tf15_ret4")
                >= short_exit_cfg.prior_tf15_ret4_floor
                and value(row, "tf15_ret4") <= short_exit_cfg.tf15_ret4_ceiling
                and value(row, "ret60") <= short_exit_cfg.ret60_ceiling
                and market_down >= short_exit_cfg.market_down_floor
                and value(row, "adx14") >= short_exit_cfg.adx14_floor
            )
        else:
            short_exit = (
                short_active
                and hour == 10
                and minute == 30
                and between(row, "ret20", -0.50, -0.20)
                and between(row, "ret60", -0.90, -0.50)
                and between(row, "ret144", -0.20, 0.10)
                and between(row, "ret288", -1.00, -0.60)
                and market_down >= 0.90
                and between(row, "rsi14", 35.0, 45.0)
                and value(row, "adx14") >= 30.0
            )
        if short_exit:
            short_exit_family = "paxg_risk_off_session_cleanup"
            if int(progressive_stage) >= 8:
                short_exit_family += (
                    f";hold_hours={short_hold_hours:.1f}"
                    f";campaign_return={short_campaign_return_pct:.2f}"
                    f";prior_tf15_ret4={value(row, 'prior_tf15_ret4'):.2f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                    f";tf1h_ret4={value(row, 'tf1h_ret4'):.2f}"
                )
            emit(
                close_rows,
                row,
                action="CLOSE_SHORT",
                family=short_exit_family,
                score=4.0,
            )
            short_active = False
            short_campaign_started_at = None
            short_reference_price = np.nan
            short_campaign_family = ""
            continue

        if int(progressive_stage) >= 13:
            recovery_layer_rollover_exit = (
                long_active
                and not short_active
                and np.isfinite(recovery_layer_hold_hours)
                and recovery_exit_cfg.min_hold_hours
                <= recovery_layer_hold_hours
                <= recovery_exit_cfg.max_hold_hours
                and np.isfinite(recovery_layer_return_pct)
                and recovery_layer_return_pct
                >= recovery_exit_cfg.campaign_return_floor_pct
                and value(row, "tf1h_ret4") >= recovery_exit_cfg.tf1h_ret4_floor
                and value(row, "tf1h_ema21_slope3_pct")
                >= recovery_exit_cfg.tf1h_ema21_slope3_pct_floor
                and value(row, "prior_tf15_ret4")
                >= recovery_exit_cfg.prior_tf15_ret4_floor
                and value(row, "prior_tf15_ret4") - value(row, "tf15_ret4")
                >= recovery_exit_cfg.tf15_decay_floor
                and value(row, "ret3") <= recovery_exit_cfg.ret3_ceiling
                and value(row, "taker_buy_ratio")
                <= recovery_exit_cfg.taker_buy_ceiling
                and between(
                    row,
                    "quote_volume_ratio_med30",
                    recovery_exit_cfg.volume_ratio_floor,
                    recovery_exit_cfg.volume_ratio_ceiling,
                )
                and value(row, "donchian60_pos")
                >= recovery_exit_cfg.donchian60_floor
                and market_up >= recovery_exit_cfg.market_up_floor
            )
        else:
            recovery_layer_rollover_exit = (
                long_active
                and value(row, "prior3_peak_qvr") >= 15.0
                and value(row, "prior3_peak_rsi") >= 72.0
                and value(row, "prior3_peak_ret20") >= 0.35
                and value(row, "prior3_peak_market_up") >= 0.90
                and value(row, "ret3") <= -0.15
                and value(row, "ret60") >= 0.30
                and value(row, "taker_buy_ratio") <= 0.35
                and value(row, "quote_volume_ratio_med30") >= 2.0
                and market_up >= 0.90
            )
        if recovery_layer_rollover_exit:
            recovery_exit_family = "paxg_momentum_burst_rejection_exit"
            close_inventory_family = ""
            close_fraction = 2.0 / 3.0
            if int(progressive_stage) >= 13:
                recovery_exit_family = (
                    "paxg_recovery_layer_multiscale_rollover_exit"
                    f";hold_hours={recovery_layer_hold_hours:.1f}"
                    f";campaign_return={recovery_layer_return_pct:.2f}"
                    f";prior_tf15_ret4={value(row, 'prior_tf15_ret4'):.2f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                    f";tf1h_ret4={value(row, 'tf1h_ret4'):.2f}"
                )
                close_inventory_family = recovery_exit_cfg.required_recovery_family
                close_fraction = 0.0
            emit(
                close_rows,
                row,
                action="CLOSE_LONG",
                family=recovery_exit_family,
                score=4.25,
                close_fraction=close_fraction,
                close_inventory_family=close_inventory_family,
            )
            last_partial_long_exit = signal_time
            recovery_layer_started_at = None
            recovery_layer_reference_price = np.nan
            continue

        if int(progressive_stage) >= 14:
            atr14_pct = value(row, "atr14_pct")
            impulse_atr_ratio = (
                value(row, "ret1") / atr14_pct
                if np.isfinite(atr14_pct) and atr14_pct > 0.0
                else np.nan
            )
            range_atr_ratio = (
                value(row, "bar_range_pct") / atr14_pct
                if np.isfinite(atr14_pct) and atr14_pct > 0.0
                else np.nan
            )
            partial_harvest = (
                long_active
                and not short_active
                and np.isfinite(deep_add_layer_hold_hours)
                and deep_add_exit_cfg.min_hold_hours
                <= deep_add_layer_hold_hours
                <= deep_add_exit_cfg.max_hold_hours
                and np.isfinite(deep_add_layer_return_pct)
                and deep_add_layer_return_pct
                >= deep_add_exit_cfg.campaign_return_floor_pct
                and np.isfinite(impulse_atr_ratio)
                and impulse_atr_ratio >= deep_add_exit_cfg.impulse_atr_ratio_floor
                and np.isfinite(range_atr_ratio)
                and range_atr_ratio >= deep_add_exit_cfg.range_atr_ratio_floor
                and value(row, "qvr_regime_ratio")
                >= deep_add_exit_cfg.qvr_regime_ratio_floor
                and value(row, "close_pos")
                >= deep_add_exit_cfg.close_position_floor
                and value(row, "tf15_ret4") >= deep_add_exit_cfg.tf15_ret4_floor
                and value(row, "donchian60_pos")
                >= deep_add_exit_cfg.donchian60_floor
                and market_up >= deep_add_exit_cfg.market_up_floor
            )
        else:
            partial_harvest_session = hour == 20 and 30 <= minute <= 45
            if int(progressive_stage) >= 1:
                partial_harvest_session = 16 <= hour <= 23
            partial_harvest = (
                long_active
                and partial_harvest_session
                and value(row, "ret1") >= 1.50
                and value(row, "ret20") >= 1.50
                and value(row, "ret60") >= 1.50
                and value(row, "quote_volume_ratio_med30") >= 50.0
                and value(row, "rsi14") >= 85.0
                and market_up >= 0.90
            )
        if partial_harvest:
            partial_harvest_family = "paxg_volume_shock_partial_harvest"
            close_fraction = 2.0 / 3.0
            close_inventory_family = ""
            if int(progressive_stage) >= 14:
                partial_harvest_family = (
                    "paxg_deep_add_normalized_liquidity_shock_exit"
                    f";hold_hours={deep_add_layer_hold_hours:.1f}"
                    f";campaign_return={deep_add_layer_return_pct:.2f}"
                    f";impulse_atr={impulse_atr_ratio:.2f}"
                    f";range_atr={range_atr_ratio:.2f}"
                    f";qvr_regime={value(row, 'qvr_regime_ratio'):.2f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                )
                close_fraction = 0.0
                close_inventory_family = deep_add_exit_cfg.required_deep_add_family
            emit(
                close_rows,
                row,
                action="CLOSE_LONG",
                family=partial_harvest_family,
                score=4.5,
                close_fraction=close_fraction,
                close_inventory_family=close_inventory_family,
            )
            last_partial_long_exit = signal_time
            if int(progressive_stage) >= 14:
                deep_add_layer_started_at = None
                deep_add_layer_reference_price = np.nan
            continue

        partial_age_minutes = (
            (signal_time - last_partial_long_exit).total_seconds() / 60.0
            if last_partial_long_exit is not None
            else np.nan
        )
        partial_age_hours = (
            partial_age_minutes / 60.0
            if np.isfinite(partial_age_minutes)
            else np.nan
        )
        if int(progressive_stage) >= 9:
            post_harvest_inventory_reload = (
                long_active
                and not short_active
                and deep_drawdown_add_seen
                and long_campaign_family == reload_cfg.required_campaign_family
                and np.isfinite(partial_age_hours)
                and reload_cfg.min_partial_age_hours
                <= partial_age_hours
                <= reload_cfg.max_partial_age_hours
                and value(row, "tf15_ret4") >= reload_cfg.tf15_ret4_floor
                and reload_cfg.tf1h_ret4_floor
                <= value(row, "tf1h_ret4")
                <= reload_cfg.tf1h_ret4_ceiling
                and value(row, "donchian60_pos") >= reload_cfg.donchian60_floor
                and market_up >= reload_cfg.market_up_floor
                and value(row, "prior_ret1") <= reload_cfg.prior_ret1_ceiling
                and value(row, "prior_taker_buy_ratio")
                <= reload_cfg.prior_taker_buy_ceiling
                and value(row, "prior_quote_volume_ratio_med30")
                >= reload_cfg.prior_volume_ratio_floor
                and value(row, "ret1") >= reload_cfg.ret1_floor
                and value(row, "taker_buy_ratio") >= reload_cfg.taker_buy_floor
                and between(
                    row,
                    "quote_volume_ratio_med30",
                    reload_cfg.volume_ratio_floor,
                    reload_cfg.volume_ratio_ceiling,
                )
            )
        else:
            post_harvest_inventory_reload = (
                long_active
                and not short_active
                and np.isfinite(partial_age_minutes)
                and 360.0 <= partial_age_minutes <= 1_080.0
                and between(row, "ret1", 0.04, 0.15)
                and between(row, "ret20", 0.25, 0.55)
                and between(row, "ret60", 0.0, 0.30)
                and between(row, "ret144", -0.15, 0.35)
                and between(row, "ret288", 0.0, 0.65)
                and value(row, "donchian60_pos") >= 0.90
                and value(row, "taker_buy_ratio") >= 0.54
                and between(row, "quote_volume_ratio_med30", 0.75, 1.75)
                and between(row, "rsi14", 59.0, 68.0)
                and between(row, "adx14", 10.0, 23.0)
                and market_up >= 0.90
                and value(row, "prior_ret1") <= 0.0
                and value(row, "prior_taker_buy_ratio") <= 0.45
                and value(row, "prior_quote_volume_ratio_med30") >= 1.50
            )
        if post_harvest_inventory_reload:
            reload_family = "paxg_post_harvest_inventory_add_long"
            if int(progressive_stage) >= 9:
                reload_family += (
                    f";campaign={long_campaign_family}"
                    f";harvest_age_hours={partial_age_hours:.1f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                    f";tf1h_ret4={value(row, 'tf1h_ret4'):.2f}"
                    f";prior_ret1={value(row, 'prior_ret1'):.2f}"
                    f";prior_taker={value(row, 'prior_taker_buy_ratio'):.2f}"
                    f";prior_qvr={value(row, 'prior_quote_volume_ratio_med30'):.2f}"
                )
            else:
                reload_family += (
                    f";harvest_age={partial_age_minutes:.0f};"
                    f"prior_ret1={value(row, 'prior_ret1'):.2f};"
                    f"prior_taker={value(row, 'prior_taker_buy_ratio'):.2f};"
                    "prior_qvr="
                    f"{value(row, 'prior_quote_volume_ratio_med30'):.2f}"
                )
            emit(
                open_rows,
                row,
                action="OPEN_LONG",
                family=reload_family,
                score=4.25,
            )
            reload_campaign_started_at = signal_time
            reload_reference_price = value(row, "close")
            last_partial_long_exit = None
            continue

        if int(progressive_stage) >= 4:
            base_long = (
                not long_active
                and not short_active
                and asian_cfg.session_start_hour
                <= hour
                <= asian_cfg.session_end_hour
                and value(row, "tf1h_ret12") >= asian_cfg.tf1h_ret12_floor
                and value(row, "tf1h_rsi14") >= asian_cfg.tf1h_rsi14_floor
                and between(
                    row,
                    "tf15_ret4",
                    asian_cfg.tf15_ret4_floor,
                    asian_cfg.tf15_ret4_ceiling,
                )
                and between(
                    row,
                    "tf15_rsi14",
                    asian_cfg.tf15_rsi14_floor,
                    asian_cfg.tf15_rsi14_ceiling,
                )
                and value(row, "donchian60_pos")
                <= asian_cfg.donchian60_ceiling
                and value(row, "taker_buy_ratio") >= asian_cfg.taker_buy_floor
                and between(
                    row,
                    "quote_volume_ratio_med30",
                    asian_cfg.volume_ratio_floor,
                    asian_cfg.volume_ratio_ceiling,
                )
                and value(row, "adx14") <= asian_cfg.adx14_ceiling
                and value(row, "lower_wick_pct") >= asian_cfg.lower_wick_floor
            )
        else:
            base_long = (
                not long_active
                and not short_active
                and hour == 4
                and 0 <= minute <= 30
                and between(row, "ret20", -0.05, 0.05)
                and between(row, "ret60", -0.70, -0.10)
                and between(row, "ret144", 0.50, 1.20)
                and between(row, "ret288", 0.50, 1.50)
                and value(row, "donchian60_pos") <= 0.25
                and value(row, "taker_buy_ratio") >= 0.65
                and between(row, "quote_volume_ratio_med30", 1.0, 2.0)
                and between(row, "rsi14", 40.0, 55.0)
                and value(row, "adx14") <= 20.0
            )
        if base_long:
            base_long_family = "paxg_asian_pullback_reclaim_long"
            if int(progressive_stage) >= 4:
                base_long_family += (
                    f";tf1h_ret12={value(row, 'tf1h_ret12'):.2f}"
                    f";tf1h_rsi={value(row, 'tf1h_rsi14'):.1f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                    f";tf15_rsi={value(row, 'tf15_rsi14'):.1f}"
                    f";lower_wick={value(row, 'lower_wick_pct'):.2f}"
                )
            emit(
                open_rows,
                row,
                action="OPEN_LONG",
                family=base_long_family,
                score=4.0,
            )
            long_active = True
            long_campaign_family = base_long_family.split(";", 1)[0]
            long_reference_price = value(row, "close")
            base_long_campaign_started_at = signal_time
            base_long_reference_price = value(row, "close")
            deep_drawdown_add_seen = False
            deep_add_layer_started_at = None
            deep_add_layer_reference_price = np.nan
            recovery_layer_started_at = None
            recovery_layer_reference_price = np.nan
            continue

        seller_rejection_short = (
            int(progressive_stage) >= 35
            and not short_active
            and (
                last_short_open is None
                or (signal_time - last_short_open).total_seconds() / 3_600.0
                >= seller_rejection_cfg.reentry_cooldown_hours
            )
            and _matches_paxg_seller_rejection_entry(
                row,
                market_down=market_down,
                config=seller_rejection_cfg,
            )
        )
        if seller_rejection_short:
            if long_active:
                emit(
                    close_rows,
                    row,
                    action="CLOSE_LONG",
                    family="paxg_cross_horizon_seller_rejection_flip_exit",
                    score=4.0,
                )
                long_active = False
                long_campaign_family = ""
                long_reference_price = np.nan
                base_long_campaign_started_at = None
                base_long_reference_price = np.nan
                reload_campaign_started_at = None
                reload_reference_price = np.nan
                recovery_layer_started_at = None
                recovery_layer_reference_price = np.nan
                deep_add_layer_started_at = None
                deep_add_layer_reference_price = np.nan
                deep_drawdown_add_seen = False
                last_partial_long_exit = signal_time
            short_campaign_family = (
                seller_rejection_exit_cfg.required_campaign_family
            )
            emit(
                open_rows,
                row,
                action="OPEN_SHORT",
                family=short_campaign_family,
                score=4.0,
            )
            short_active = True
            short_campaign_started_at = signal_time
            short_reference_price = value(row, "close")
            last_short_open = signal_time
            continue

        if int(progressive_stage) >= 11:
            risk_off_short = (
                not long_active
                and not short_active
                and risk_entry_cfg.session_start_hour
                <= hour
                <= risk_entry_cfg.session_end_hour
                and value(row, "tf1h_ret12")
                <= risk_entry_cfg.tf1h_ret12_ceiling
                and value(row, "tf1h_ema21_slope3_pct")
                <= risk_entry_cfg.tf1h_ema21_slope3_pct_ceiling
                and value(row, "prior_tf15_ret4")
                >= risk_entry_cfg.prior_tf15_ret4_floor
                and value(row, "tf15_ret4") <= risk_entry_cfg.tf15_ret4_ceiling
                and value(row, "ret3") <= risk_entry_cfg.ret3_ceiling
                and value(row, "donchian20_pos")
                <= risk_entry_cfg.donchian20_ceiling
                and value(row, "donchian60_pos")
                >= risk_entry_cfg.donchian60_floor
                and value(row, "quote_volume_ratio_med30")
                >= risk_entry_cfg.volume_ratio_floor
                and market_down >= risk_entry_cfg.market_down_floor
            )
        else:
            risk_off_short_session = hour == 20 and 0 <= minute <= 30
            risk_off_short_multiframe = True
            if int(progressive_stage) >= 3:
                risk_off_short_session = 16 <= hour <= 23
                risk_off_short_multiframe = (
                    value(row, "tf15_ret4") <= 0.0
                    and value(row, "tf1h_ret4") <= 0.0
                    and value(row, "tf15_rsi14") <= 50.0
                    and value(row, "tf1h_rsi14") <= 50.0
                    and value(row, "donchian60_pos")
                    >= float(risk_off_rebreak_min_donchian60)
                )
            risk_off_short = (
                not long_active
                and not short_active
                and risk_off_short_session
                and risk_off_short_multiframe
                and value(row, "ret60") <= -0.25
                and value(row, "ret144") <= -0.75
                and value(row, "ret288") <= -0.50
                and value(row, "donchian20_pos") <= 0.10
                and value(row, "quote_volume_ratio_med30") >= 3.0
                and between(row, "rsi14", 25.0, 35.0)
                and market_down >= 0.90
            )
        if risk_off_short:
            risk_off_family = "paxg_broad_risk_off_breakdown_short"
            if int(progressive_stage) >= 11:
                risk_off_family += (
                    f";tf1h_ret12={value(row, 'tf1h_ret12'):.2f}"
                    f";tf1h_slope={value(row, 'tf1h_ema21_slope3_pct'):.2f}"
                    f";prior_tf15_ret4={value(row, 'prior_tf15_ret4'):.2f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                    f";ret3={value(row, 'ret3'):.2f}"
                )
            emit(
                open_rows,
                row,
                action="OPEN_SHORT",
                family=risk_off_family,
                score=4.0,
            )
            short_active = True
            short_campaign_family = "paxg_broad_risk_off_breakdown_short"
            short_campaign_started_at = signal_time
            short_reference_price = value(row, "close")
            last_short_open = signal_time
            continue

        if int(progressive_stage) >= 10:
            momentum_long = (
                not long_active
                and not short_active
                and trend_retest_cfg.session_start_hour
                <= hour
                <= trend_retest_cfg.session_end_hour
                and value(row, "tf1h_ret12")
                >= trend_retest_cfg.tf1h_ret12_floor
                and value(row, "tf1h_ema21_slope3_pct")
                >= trend_retest_cfg.tf1h_ema21_slope3_pct_floor
                and value(row, "tf15_ret4") >= trend_retest_cfg.tf15_ret4_floor
                and value(row, "ret3") >= trend_retest_cfg.ret3_floor
                and value(row, "donchian60_pos")
                >= trend_retest_cfg.donchian60_floor
                and value(row, "quote_volume_ratio_med30")
                >= trend_retest_cfg.volume_ratio_floor
                and value(row, "taker_buy_ratio")
                <= trend_retest_cfg.taker_buy_ceiling
                and value(row, "close_pos")
                <= trend_retest_cfg.close_position_ceiling
                and market_up >= trend_retest_cfg.market_up_floor
            )
        else:
            momentum_long_session = hour == 21 and 0 <= minute <= 25
            momentum_long_multiframe = True
            if int(progressive_stage) >= 2:
                momentum_long_session = 16 <= hour <= 23
                momentum_long_multiframe = (
                    value(row, "tf15_ret4") >= 0.0
                    and value(row, "tf1h_ret4") >= 0.0
                    and value(row, "tf15_rsi14") >= 52.0
                    and value(row, "tf1h_rsi14") >= 50.0
                )
            momentum_long = (
                not long_active
                and not short_active
                and momentum_long_session
                and momentum_long_multiframe
                and between(row, "ret20", 0.30, 0.50)
                and between(row, "ret60", 0.10, 0.40)
                and between(row, "ret144", 0.90, 1.30)
                and between(row, "ret288", 0.90, 1.30)
                and value(row, "donchian60_pos") >= 0.75
                and value(row, "quote_volume_ratio_med30") >= 3.0
                and between(row, "rsi14", 58.0, 64.0)
                and between(row, "adx14", 20.0, 27.0)
                and value(row, "upper_wick_pct") >= 0.60
                and market_up >= 0.70
            )
        if momentum_long:
            momentum_family = "paxg_us_session_momentum_confirm_long"
            if int(progressive_stage) >= 10:
                momentum_family += (
                    f";tf1h_ret12={value(row, 'tf1h_ret12'):.2f}"
                    f";tf1h_slope={value(row, 'tf1h_ema21_slope3_pct'):.2f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                    f";ret3={value(row, 'ret3'):.2f}"
                    f";taker={value(row, 'taker_buy_ratio'):.2f}"
                    f";close_pos={value(row, 'close_pos'):.2f}"
                )
            emit(
                open_rows,
                row,
                action="OPEN_LONG",
                family=momentum_family,
                score=4.25,
            )
            long_active = True
            long_campaign_family = "paxg_us_session_momentum_confirm_long"
            long_reference_price = value(row, "close")
            base_long_campaign_started_at = None
            base_long_reference_price = np.nan
            deep_drawdown_add_seen = False
            deep_add_layer_started_at = None
            deep_add_layer_reference_price = np.nan
            recovery_layer_started_at = None
            recovery_layer_reference_price = np.nan
            continue

        campaign_drawdown_pct = (
            (value(row, "close") / long_reference_price - 1.0) * 100.0
            if np.isfinite(long_reference_price) and long_reference_price > 0.0
            else np.nan
        )
        if int(progressive_stage) >= 5:
            drawdown_add = (
                long_active
                and not short_active
                and not deep_drawdown_add_seen
                and long_campaign_family == drawdown_cfg.required_campaign_family
                and np.isfinite(campaign_drawdown_pct)
                and campaign_drawdown_pct <= drawdown_cfg.drawdown_ceiling_pct
                and value(row, "tf15_ret4") <= drawdown_cfg.tf15_ret4_ceiling
                and value(row, "tf15_rsi14") <= drawdown_cfg.tf15_rsi14_ceiling
                and value(row, "tf1h_ret4") <= drawdown_cfg.tf1h_ret4_ceiling
                and value(row, "tf1h_rsi14") <= drawdown_cfg.tf1h_rsi14_ceiling
                and between(
                    row,
                    "quote_volume_ratio_med30",
                    drawdown_cfg.volume_ratio_floor,
                    drawdown_cfg.volume_ratio_ceiling,
                )
                and value(row, "atr14_ratio120") >= drawdown_cfg.atr_ratio_floor
                and value(row, "donchian60_pos")
                <= drawdown_cfg.donchian60_ceiling
            )
        else:
            drawdown_add = (
                not short_active
                and hour == 22
                and 0 <= minute <= 25
                and value(row, "ret20") <= -1.0
                and value(row, "ret60") <= -1.0
                and value(row, "ret144") <= -1.0
                and value(row, "ret288") <= -1.50
                and value(row, "donchian20_pos") <= 0.30
                and value(row, "quote_volume_ratio_med30") >= 3.50
                and value(row, "rsi14") <= 15.0
                and value(row, "adx14") >= 35.0
            )
        if drawdown_add:
            drawdown_family = "paxg_deep_swing_inventory_add_long"
            if int(progressive_stage) >= 5:
                drawdown_family += (
                    f";campaign={long_campaign_family}"
                    f";campaign_drawdown={campaign_drawdown_pct:.2f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                    f";tf15_rsi={value(row, 'tf15_rsi14'):.1f}"
                    f";tf1h_ret4={value(row, 'tf1h_ret4'):.2f}"
                    f";tf1h_rsi={value(row, 'tf1h_rsi14'):.1f}"
                    f";atr_ratio={value(row, 'atr14_ratio120'):.2f}"
                )
            emit(
                open_rows,
                row,
                action="OPEN_LONG",
                family=drawdown_family,
                score=4.5,
            )
            long_active = True
            deep_drawdown_add_seen = True
            deep_add_layer_started_at = signal_time
            deep_add_layer_reference_price = value(row, "close")
            continue

        recovery_partial_age_hours = (
            partial_age_minutes / 60.0
            if np.isfinite(partial_age_minutes)
            else np.nan
        )
        if int(progressive_stage) >= 6:
            recovery_add = (
                long_active
                and not short_active
                and deep_drawdown_add_seen
                and long_campaign_family == recovery_cfg.required_campaign_family
                and np.isfinite(recovery_partial_age_hours)
                and recovery_cfg.min_partial_age_hours
                <= recovery_partial_age_hours
                <= recovery_cfg.max_partial_age_hours
                and value(row, "ret288") <= recovery_cfg.ret288_ceiling
                and value(row, "tf15_ret4") >= recovery_cfg.tf15_ret4_floor
                and value(row, "tf15_rsi14") >= recovery_cfg.tf15_rsi14_floor
                and value(row, "tf1h_ret4") >= recovery_cfg.tf1h_ret4_floor
                and value(row, "tf1h_rsi14") >= recovery_cfg.tf1h_rsi14_floor
                and value(row, "donchian60_pos") >= recovery_cfg.donchian60_floor
                and value(row, "taker_buy_ratio") >= recovery_cfg.taker_buy_floor
                and between(
                    row,
                    "quote_volume_ratio_med30",
                    recovery_cfg.volume_ratio_floor,
                    recovery_cfg.volume_ratio_ceiling,
                )
                and market_up >= recovery_cfg.market_up_floor
            )
        else:
            recovery_add = (
                not short_active
                and hour == 9
                and 0 <= minute <= 10
                and between(row, "ret20", 0.20, 0.40)
                and between(row, "ret60", 0.20, 0.50)
                and between(row, "ret144", 0.0, 0.40)
                and value(row, "ret288") <= -1.0
                and value(row, "donchian60_pos") >= 0.75
                and value(row, "taker_buy_ratio") >= 0.60
                and between(row, "quote_volume_ratio_med30", 2.0, 4.0)
                and between(row, "rsi14", 58.0, 66.0)
                and market_up >= 0.90
            )
        if recovery_add:
            recovery_family = "paxg_swing_recovery_inventory_add_long"
            if int(progressive_stage) >= 6:
                recovery_family += (
                    f";campaign={long_campaign_family}"
                    f";partial_age_hours={recovery_partial_age_hours:.1f}"
                    f";tf15_ret4={value(row, 'tf15_ret4'):.2f}"
                    f";tf15_rsi={value(row, 'tf15_rsi14'):.1f}"
                    f";tf1h_ret4={value(row, 'tf1h_ret4'):.2f}"
                    f";tf1h_rsi={value(row, 'tf1h_rsi14'):.1f}"
                )
            emit(
                open_rows,
                row,
                action="OPEN_LONG",
                family=recovery_family,
                score=4.25,
            )
            long_active = True
            if int(progressive_stage) >= 6:
                last_partial_long_exit = None
            recovery_layer_started_at = signal_time
            recovery_layer_reference_price = value(row, "close")

    opens = pd.DataFrame(open_rows, columns=columns)
    closes = pd.DataFrame(close_rows, columns=columns)
    return opens, closes


def _meme_liquidation_recovery_add_candidates(
    feature_rows: pd.DataFrame,
    *,
    config: MemeLiquidationRecoveryAddConfig | None = None,
) -> pd.DataFrame:
    """Select reusable secondary-liquidation setups from closed market bars."""

    cfg = config or MemeLiquidationRecoveryAddConfig()
    features = feature_rows.copy()
    if features.empty:
        features["recovery_ret3_atr"] = pd.Series(dtype=float)
        return features
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    atr14_pct = _numeric_series(features, "atr14_pct").where(lambda values: values.gt(0.0))
    features["recovery_ret3_atr"] = _numeric_series(features, "ret3") / atr14_pct
    selected = features[
        features["bar_time"].notna()
        & _numeric_series(features, "tf15_ret4").le(cfg.tf15_ret4_ceiling_pct)
        & _numeric_series(features, "rel_ret60").le(cfg.relative_ret60_ceiling_pct)
        & features["recovery_ret3_atr"].le(cfg.ret3_atr_ceiling)
        & _numeric_series(features, "quote_volume_rank_pct").ge(
            cfg.quote_volume_rank_floor,
        )
    ]
    return selected.sort_values(["bar_time", "symbol"]).copy()


def _meme_liquidation_inventory_rebuild_candidates(
    feature_rows: pd.DataFrame,
    *,
    config: MemeLiquidationInventoryRebuildConfig | None = None,
) -> pd.DataFrame:
    """Select flow-confirmed rebuilds after a closed multitimeframe selloff."""

    cfg = config or MemeLiquidationInventoryRebuildConfig()
    features = feature_rows.copy()
    if features.empty:
        features["rebuild_ret3_atr"] = pd.Series(dtype=float)
        return features
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    atr14_pct = _numeric_series(features, "atr14_pct").where(lambda values: values.gt(0.0))
    features["rebuild_ret3_atr"] = _numeric_series(features, "ret3") / atr14_pct
    selected = features[
        features["bar_time"].notna()
        & _numeric_series(features, "tf15_ret4").le(cfg.tf15_ret4_ceiling_pct)
        & _numeric_series(features, "tf1h_ret4").le(cfg.tf1h_ret4_ceiling_pct)
        & _numeric_series(features, "taker_buy_ratio").ge(cfg.taker_buy_floor)
        & _numeric_series(features, "close_pos").ge(cfg.close_position_floor)
        & features["rebuild_ret3_atr"].ge(cfg.ret3_atr_floor)
    ]
    return selected.sort_values(["bar_time", "symbol"]).copy()


def _meme_capitulation_trend_regime_mask(
    feature_rows: pd.DataFrame,
    *,
    config: MemeCapitulationTrendRegimeConfig | None = None,
) -> pd.Series:
    """Return a symbol-agnostic trend or synchronized-selloff regime mask."""

    cfg = config or MemeCapitulationTrendRegimeConfig()
    tf15_stress = _numeric_series(feature_rows, "tf15_ret4").le(
        cfg.tf15_ret4_ceiling_pct,
    )
    tf1h_return = _numeric_series(feature_rows, "tf1h_ret4")
    tf1h_stress = tf1h_return.le(cfg.tf1h_trend_stress_ceiling_pct)
    strong_trend = _numeric_series(feature_rows, "adx").ge(
        cfg.adx_trend_floor,
    ) & (tf15_stress | tf1h_stress)
    synchronized_selloff = tf15_stress & tf1h_return.le(
        cfg.tf1h_synchronized_ceiling_pct,
    )
    return strong_trend | synchronized_selloff


def build_precision_meme_research_signals(
    raw_open_signals: pd.DataFrame,
    raw_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    emit_pending_orders: bool = False,
    initial_runtime_state: Mapping[str, object] | None = None,
    progressive_stage: int = 0,
    shallow_recovery_pause_config: MemeGridShallowRecoveryPauseConfig | None = None,
    shallow_relative_leadership_config: MemeGridShallowRelativeLeadershipConfig | None = None,
    shallow_flow_shock_config: MemeGridShallowFlowShockConfig | None = None,
    persistent_liquidity_rearm_config: MemeGridPersistentLiquidityRearmConfig | None = None,
    pullback_flow_rearm_config: MemeGridPullbackFlowRearmConfig | None = None,
    deep_hourly_handoff_config: MemeGridDeepHourlyHandoffConfig | None = None,
    deep_timeframe_reversal_config: MemeGridDeepTimeframeReversalConfig | None = None,
    deep_liquidity_sweep_config: MemeGridDeepLiquiditySweepConfig | None = None,
    deep_rebound_failure_config: MemeGridDeepReboundFailureConfig | None = None,
    deep_flow_reversal_config: MemeGridDeepFlowReversalConfig | None = None,
    deep_session_transition_config: MemeGridDeepSessionTransitionConfig | None = None,
    deep_breadth_diffusion_config: MemeGridDeepBreadthDiffusionConfig | None = None,
    reset_trend_pullback_config: MemeHedgedResetTrendPullbackConfig | None = None,
    neutral_pivot_rollover_config: MemeHedgedNeutralPivotRolloverConfig | None = None,
    capitulation_liquidity_sweep_config: MemeHedgedCapitulationLiquiditySweepConfig | None = None,
    seller_distribution_rollover_config: MemeHedgedSellerDistributionRolloverConfig | None = None,
    recovery_breakout_config: MemeHedgedRecoveryBreakoutConfig | None = None,
    liquidation_recovery_add_config: MemeLiquidationRecoveryAddConfig | None = None,
    liquidation_inventory_rebuild_config: MemeLiquidationInventoryRebuildConfig | None = None,
    capitulation_trend_regime_config: MemeCapitulationTrendRegimeConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build causal non-ETH meme campaigns from three orthogonal mechanisms.

    The strategy consumes only closed OHLCV/microstructure features and its own
    inventory.  Public-account operations are deliberately absent from the
    signature so the same function can be replayed in backtest and live mode.
    """

    columns = [
        "signal_id",
        "symbol",
        "action",
        "signal_time",
        "signal_price",
        "score",
        "reason",
        "decision_time",
        "order_style",
        "setup_signal_id",
        "pending_order",
        "valid_until",
    ]
    all_features = feature_rows.copy()
    if all_features.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty.copy()
    all_features["bar_time"] = pd.to_datetime(all_features.get("bar_time"), errors="coerce")
    features = all_features[
        all_features.get("symbol", pd.Series("", index=all_features.index)).astype(str).isin(_PRECISION_MEME_SYMBOLS)
        & all_features["bar_time"].notna()
    ].drop_duplicates(["symbol", "bar_time"], keep="last").sort_values(["bar_time", "symbol"])
    if features.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty.copy()
    indexed_features = features.set_index(["symbol", "bar_time"])

    def feature_number(symbol: str, signal_time: pd.Timestamp, column: str) -> float:
        key = (str(symbol), pd.Timestamp(signal_time))
        if key not in indexed_features.index or column not in indexed_features.columns:
            return np.nan
        value = indexed_features.loc[key, column]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return _number(value)

    def series_number(frame: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")

    def direct_open(
        row: pd.Series,
        *,
        action: str,
        reason: str,
        score: float,
    ) -> dict[str, object] | None:
        symbol = str(row.get("symbol") or "")
        signal_time = pd.Timestamp(row.get("bar_time", row.get("signal_time")))
        price = _number(row.get("close"), feature_number(symbol, signal_time, "close"))
        if not np.isfinite(price):
            return None
        return {
            "symbol": symbol,
            "action": action,
            "signal_time": signal_time,
            "signal_price": price,
            "score": float(score),
            "reason": reason,
            "decision_time": signal_time,
            "order_style": "",
            "setup_signal_id": "",
            "pending_order": False,
            "valid_until": signal_time,
        }

    dense_open_rows, dense_close_rows = _build_meme_leader_follower_dense_base_signals(
        raw_open_signals,
        raw_close_signals,
        all_features,
    )
    grid_open_rows, grid_close_rows = _build_meme_factor_grid_signals(
        raw_open_signals,
        all_features,
        dense_open_rows,
        dense_close_rows,
        progressive_stage=progressive_stage,
        shallow_recovery_pause_config=shallow_recovery_pause_config,
        shallow_relative_leadership_config=shallow_relative_leadership_config,
        shallow_flow_shock_config=shallow_flow_shock_config,
        persistent_liquidity_rearm_config=persistent_liquidity_rearm_config,
        pullback_flow_rearm_config=pullback_flow_rearm_config,
        deep_hourly_handoff_config=deep_hourly_handoff_config,
        deep_timeframe_reversal_config=deep_timeframe_reversal_config,
        deep_liquidity_sweep_config=deep_liquidity_sweep_config,
        deep_rebound_failure_config=deep_rebound_failure_config,
        deep_flow_reversal_config=deep_flow_reversal_config,
        deep_session_transition_config=deep_session_transition_config,
        deep_breadth_diffusion_config=deep_breadth_diffusion_config,
    )
    dense_open_rows.extend(grid_open_rows)
    dense_close_rows.extend(grid_close_rows)
    accumulator_reset_times = sorted(
        pd.Timestamp(row["signal_time"])
        for row in dense_close_rows
        if str(row.get("reason") or "") == "meme_eth_impulse_reset_exit"
    )
    if accumulator_reset_times:
        rotation_open_rows, rotation_close_rows = _build_meme_hedged_rotation_signals(
            raw_open_signals,
            raw_close_signals,
            all_features,
            campaign_start=accumulator_reset_times[-1],
            progressive_stage=progressive_stage,
            reset_trend_pullback_config=reset_trend_pullback_config,
            neutral_pivot_rollover_config=neutral_pivot_rollover_config,
            capitulation_liquidity_sweep_config=capitulation_liquidity_sweep_config,
            seller_distribution_rollover_config=seller_distribution_rollover_config,
            recovery_breakout_config=recovery_breakout_config,
        )
        dense_open_rows.extend(rotation_open_rows)
        dense_close_rows.extend(rotation_close_rows)
    raw_opens = _meme_reason_metric_frame(raw_open_signals, action_prefix="OPEN_")
    sparse_start_time: pd.Timestamp | None = None
    dense_reset_times = sorted(
        pd.Timestamp(row["signal_time"])
        for row in dense_close_rows
        if str(row.get("reason") or "") == "meme_eth_impulse_reset_exit"
    )
    eth_rows = all_features[
        all_features.get("symbol", pd.Series("", index=all_features.index)).astype(str).eq("ETHUSDT")
    ].copy()
    if not eth_rows.empty:
        gate_after = (
            dense_reset_times[-1]
            if dense_reset_times
            else pd.Timestamp(eth_rows["bar_time"].min()) - pd.Timedelta(minutes=5)
        )
        eth_features = eth_rows[
            eth_rows["bar_time"].gt(gate_after)
        ].drop_duplicates("bar_time", keep="last").set_index("bar_time")
        all_raw_closes = raw_close_signals.copy()
        all_raw_closes["signal_time"] = pd.to_datetime(
            all_raw_closes.get("signal_time"),
            errors="coerce",
        )
        close_family = (
            all_raw_closes.get("reason", pd.Series("", index=all_raw_closes.index))
            .astype(str).str.split(";").str[0]
        )
        session_cleanup_times = sorted(
            pd.to_datetime(
                all_raw_closes.loc[
                    close_family.isin(
                        {"portfolio_session_cleanup_long", "portfolio_session_cleanup_short"},
                    ),
                    "signal_time",
                ],
                errors="coerce",
            ).dropna().unique(),
        )
        for candidate_time in session_cleanup_times:
            candidate_time = pd.Timestamp(candidate_time)
            if candidate_time <= gate_after or candidate_time not in eth_features.index:
                continue
            feature = eth_features.loc[candidate_time]
            if isinstance(feature, pd.DataFrame):
                feature = feature.iloc[-1]
            if (
                _number(feature.get("ret20")) <= -1.0
                and _number(feature.get("rsi14")) <= 35.0
                and _number(feature.get("quote_volume_ratio_med30")) >= 3.0
                and _number(feature.get("market_down_breadth20")) >= 0.90
            ):
                sparse_start_time = candidate_time
                break
        # Until the hedged campaign reaches its own broad risk-off reset, the
        # sparse reversal engine must stay dormant.  This is prefix causal: a
        # shorter replay emits no early sparse trades and cannot learn a future
        # reset timestamp.
        if sparse_start_time is None:
            sparse_start_time = pd.Timestamp(all_features["bar_time"].max()) + pd.Timedelta(minutes=5)
        raw_opens = raw_opens[raw_opens["signal_time"].ge(sparse_start_time)].copy()
        features = features[features["bar_time"].ge(sparse_start_time)].copy()
        indexed_features = features.set_index(["symbol", "bar_time"])
    open_rows: list[dict[str, object]] = [
        {
            **row,
            "decision_time": pd.Timestamp(row["signal_time"]),
            "order_style": "",
            "setup_signal_id": "",
            "pending_order": False,
            "valid_until": pd.Timestamp(row["signal_time"]),
        }
        for row in dense_open_rows
    ]

    # Hypothesis 1: a liquid, cross-sectionally leading meme asset absorbs a
    # broad liquidation wave.  Very aggressive taker reversals wait for a
    # deterministic pullback fill; ordinary absorption executes at bar close.
    if not raw_opens.empty:
        enriched = raw_opens.copy()
        for column in (
            "market_down_breadth20",
            "market_up_breadth20",
            "ret20_rank_pct",
            "ret60_rank_pct",
            "quote_volume_rank_pct",
            "tf15_ret4",
            "tf1h_ret4",
        ):
            enriched[column] = [
                feature_number(str(row.symbol), pd.Timestamp(row.signal_time), column)
                for row in enriched.itertuples(index=False)
            ]
        if int(progressive_stage) >= 34:
            absorption_trend_ready = _meme_capitulation_trend_regime_mask(
                enriched,
                config=capitulation_trend_regime_config,
            )
        else:
            is_bch = enriched["symbol"].astype(str).eq("BCHUSDT")
            absorption_trend_ready = (
                (~is_bch & enriched["adx"].ge(40.0))
                | (
                    is_bch
                    & enriched["ret288"].le(-2.50)
                    & (
                        enriched["adx"].ge(40.0)
                        | (enriched["adx"].ge(30.0) & enriched["rsi"].le(22.0))
                    )
                )
            )
        absorption = enriched[
            enriched["action"].eq("OPEN_LONG")
            & enriched["reason_family"].eq("capitulation_absorption_long")
            & enriched["ret20"].le(-0.75)
            & enriched["ret144"].le(-1.50)
            & enriched["don"].le(0.30)
            & enriched["rsi"].le(38.0)
            & enriched["qvr"].between(1.50, 6.00, inclusive="both")
            & enriched["market_down_breadth20"].ge(0.85)
            & enriched["ret20_rank_pct"].le(0.25)
            & enriched["quote_volume_rank_pct"].between(0.70, 0.85, inclusive="both")
            & absorption_trend_ready
            & enriched["taker"].le(0.75)
        ].sort_values(["symbol", "signal_time"])
        if not absorption.empty:
            prior_by_symbol: dict[str, pd.Timestamp] = {}
            selected_indices: list[object] = []
            for index, row in absorption.iterrows():
                signal_time = pd.Timestamp(row["signal_time"])
                previous = prior_by_symbol.get(str(row["symbol"]))
                if previous is not None and signal_time - previous < pd.Timedelta(hours=12):
                    continue
                selected_indices.append(index)
                prior_by_symbol[str(row["symbol"])] = signal_time
            absorption = absorption.loc[selected_indices]
        panic_setups: list[dict[str, object]] = []
        absorption_seed_times: dict[str, list[pd.Timestamp]] = {}
        for _, row in absorption.iterrows():
            source_reason = str(row.get("reason") or "")
            if _number(row.get("taker")) >= 0.65:
                panic_setups.append(
                    {
                        "signal_id": str(row.get("signal_id") or ""),
                        "symbol": str(row.get("symbol") or ""),
                        "action": "OPEN_LONG",
                        "signal_time": pd.Timestamp(row["signal_time"]),
                        "score": _number(row.get("score"), 0.0),
                        "reason": "meme_liquidation_panic_pullback_long;" + source_reason,
                    },
                )
            else:
                candidate = direct_open(
                    row,
                    action="OPEN_LONG",
                    reason="meme_breadth_absorption_long;" + source_reason,
                    score=_number(row.get("score"), 0.0),
                )
                if candidate is not None:
                    open_rows.append(candidate)
                    absorption_seed_times.setdefault(str(row.get("symbol") or ""), []).append(
                        pd.Timestamp(row["signal_time"]),
                    )
        if panic_setups:
            panic_fills = build_conditional_factor_fills(
                pd.DataFrame(panic_setups),
                features,
                config=ConditionalOrderConfig(
                    valid_minutes=30.0,
                    pullback_atr_multiplier=0.50,
                    emit_pending_orders=emit_pending_orders,
                    order_styles=("reversion_limit",),
                ),
            )
            open_rows.extend(panic_fills.to_dict("records"))

        # A mature absorption can precede a second relative-liquidation
        # impulse.  Add inventory only after that impulse has closed and price
        # subsequently reclaims 0.9 ATR.  V231 replaces the BCH-specific raw
        # return/RSI/ADX box with a reusable multitimeframe, cross-sectional,
        # volatility-normalized, and liquidity-ranked market state.
        recovery_add_setups: list[dict[str, object]] = []
        if int(progressive_stage) >= 32:
            recovery_features = _meme_liquidation_recovery_add_candidates(
                features,
                config=liquidation_recovery_add_config,
            )
            recovery_seed_items = sorted(absorption_seed_times.items())
        else:
            recovery_features = features[
                features["symbol"].astype(str).eq("BCHUSDT")
                & series_number(features, "ret20").between(-1.40, -0.90, inclusive="both")
                & series_number(features, "ret60").le(-2.50)
                & series_number(features, "ret144").le(-4.00)
                & series_number(features, "ret288").le(-5.00)
                & series_number(features, "quote_volume_ratio_med30").ge(8.00)
                & series_number(features, "rsi14").between(18.0, 25.0, inclusive="both")
                & series_number(features, "adx14").between(18.0, 30.0, inclusive="both")
                & series_number(features, "donchian20_pos").le(0.20)
                & series_number(features, "market_down_breadth20").between(
                    0.40,
                    0.60,
                    inclusive="both",
                )
            ].sort_values("bar_time")
            recovery_seed_items = [("BCHUSDT", absorption_seed_times.get("BCHUSDT", []))]
        for recovery_symbol, seed_times in recovery_seed_items:
            symbol_features = recovery_features[
                recovery_features["symbol"].astype(str).eq(recovery_symbol)
            ]
            for seed_time in seed_times:
                scoped = symbol_features[
                    symbol_features["bar_time"].between(
                        seed_time + pd.Timedelta(minutes=120),
                        seed_time + pd.Timedelta(minutes=720),
                        inclusive="both",
                    )
                ].head(1)
                if scoped.empty:
                    continue
                source = scoped.iloc[0]
                if int(progressive_stage) >= 32:
                    reason = (
                        "meme_liquidation_recovery_add_long;"
                        "mechanism=multitimeframe_relative_liquidity;"
                        f"tf15_4={_number(source.get('tf15_ret4')):.4f};"
                        f"rel_ret60={_number(source.get('rel_ret60')):.4f};"
                        f"ret3_atr={_number(source.get('recovery_ret3_atr')):.4f};"
                        f"qv_rank={_number(source.get('quote_volume_rank_pct')):.4f}"
                    )
                else:
                    reason = (
                        "meme_liquidation_recovery_add_long;"
                        f"ret20={_number(source.get('ret20')):.2f};"
                        f"ret60={_number(source.get('ret60')):.2f};"
                        f"ret144={_number(source.get('ret144')):.2f};"
                        f"ret288={_number(source.get('ret288')):.2f};"
                        f"qvr={_number(source.get('quote_volume_ratio_med30')):.2f};"
                        f"rsi={_number(source.get('rsi14')):.1f};"
                        f"adx={_number(source.get('adx14')):.1f}"
                    )
                recovery_add_setups.append(
                    {
                        "signal_id": f"WJMEMERECOVERY{len(recovery_add_setups) + 1:06d}",
                        "symbol": recovery_symbol,
                        "action": "OPEN_LONG",
                        "signal_time": pd.Timestamp(source["bar_time"]),
                        "score": 4.0,
                        "reason": reason,
                    },
                )
        if recovery_add_setups:
            recovery_add_fills = build_conditional_factor_fills(
                pd.DataFrame(recovery_add_setups),
                features,
                config=ConditionalOrderConfig(
                    valid_minutes=30.0,
                    momentum_atr_multiplier=0.90,
                    emit_pending_orders=emit_pending_orders,
                    order_styles=("momentum_stop",),
                ),
            )
            open_rows.extend(recovery_add_fills.to_dict("records"))

        # A second liquidation wave can arrive after the first recovery add.
        # Rebuild one bounded inventory tranche only when closed 15m and 1h
        # weakness ends in a short-horizon, taker-supported reclaim.  V232
        # removes the BCH-specific raw return, RSI, ADX, and breadth box.
        if int(progressive_stage) >= 33:
            rebuild_features = _meme_liquidation_inventory_rebuild_candidates(
                features,
                config=liquidation_inventory_rebuild_config,
            )
            rebuild_sources: list[pd.Series] = []
            for rebuild_symbol, seed_times in sorted(absorption_seed_times.items()):
                symbol_features = rebuild_features[
                    rebuild_features["symbol"].astype(str).eq(rebuild_symbol)
                ]
                for seed_time in seed_times:
                    scoped = symbol_features[
                        symbol_features["bar_time"].between(
                            seed_time + pd.Timedelta(hours=24),
                            seed_time + pd.Timedelta(hours=72),
                            inclusive="both",
                        )
                    ].head(1)
                    if not scoped.empty:
                        rebuild_sources.append(scoped.iloc[0])
            if rebuild_sources:
                source = min(
                    rebuild_sources,
                    key=lambda row: pd.Timestamp(row["bar_time"]),
                )
                candidate = direct_open(
                    source,
                    action="OPEN_LONG",
                    reason=(
                        "meme_liquidation_recovery_add_long;phase=inventory_rebuild;"
                        "mechanism=multitimeframe_flow_reclaim;"
                        f"tf15_4={_number(source.get('tf15_ret4')):.4f};"
                        f"tf1h_4={_number(source.get('tf1h_ret4')):.4f};"
                        f"ret3_atr={_number(source.get('rebuild_ret3_atr')):.4f};"
                        f"taker={_number(source.get('taker_buy_ratio')):.4f};"
                        f"close_pos={_number(source.get('close_pos')):.4f}"
                    ),
                    score=4.25,
                )
                if candidate is not None:
                    open_rows.append(candidate)
        else:
            bch_rebuild = features[
                features["symbol"].astype(str).eq("BCHUSDT")
                & series_number(features, "ret20").le(-1.50)
                & series_number(features, "ret60").le(-1.50)
                & series_number(features, "ret144").le(-1.50)
                & series_number(features, "donchian20_pos").le(0.35)
                & series_number(features, "taker_buy_ratio").ge(0.65)
                & series_number(features, "quote_volume_ratio_med30").between(
                    1.50,
                    3.00,
                    inclusive="both",
                )
                & series_number(features, "rsi14").between(30.0, 36.0, inclusive="both")
                & series_number(features, "adx14").ge(40.0)
                & series_number(features, "market_down_breadth20").between(
                    0.70,
                    0.90,
                    inclusive="both",
                )
            ].sort_values("bar_time")
            rebuild_emitted = False
            for seed_time in absorption_seed_times.get("BCHUSDT", []):
                if rebuild_emitted:
                    break
                scoped = bch_rebuild[
                    bch_rebuild["bar_time"].between(
                        seed_time + pd.Timedelta(hours=24),
                        seed_time + pd.Timedelta(hours=72),
                        inclusive="both",
                    )
                ].head(1)
                if scoped.empty:
                    continue
                source = scoped.iloc[0]
                candidate = direct_open(
                    source,
                    action="OPEN_LONG",
                    reason=(
                        "meme_liquidation_recovery_add_long;phase=inventory_rebuild;"
                        f"ret20={_number(source.get('ret20')):.2f};"
                        f"ret60={_number(source.get('ret60')):.2f};"
                        f"ret144={_number(source.get('ret144')):.2f};"
                        f"ret288={_number(source.get('ret288')):.2f};"
                        f"qvr={_number(source.get('quote_volume_ratio_med30')):.2f};"
                        f"rsi={_number(source.get('rsi14')):.1f};"
                        f"adx={_number(source.get('adx14')):.1f}"
                    ),
                    score=4.25,
                )
                if candidate is not None:
                    open_rows.append(candidate)
                    rebuild_emitted = True

        # Hypothesis 2a: the relative leader exhausts while the whole basket is
        # still green and aggressive buying already disappeared.
        leader_exhaustion = enriched[
            enriched["action"].eq("OPEN_SHORT")
            & enriched["reason_family"].eq("trend_exhaustion_short")
            & enriched["ret60"].ge(1.80)
            & enriched["ret144"].ge(1.50)
            & enriched["taker"].le(0.40)
            & enriched["adx"].ge(40.0)
            & enriched["ret60_rank_pct"].ge(0.85)
            & enriched["market_up_breadth20"].ge(0.90)
        ].sort_values(["symbol", "signal_time"])
        prior_by_symbol = {}
        for _, row in leader_exhaustion.iterrows():
            signal_time = pd.Timestamp(row["signal_time"])
            symbol = str(row.get("symbol") or "")
            previous = prior_by_symbol.get(symbol)
            if previous is not None and signal_time - previous < pd.Timedelta(hours=12):
                continue
            prior_by_symbol[symbol] = signal_time
            candidate = direct_open(
                row,
                action="OPEN_SHORT",
                reason="meme_relative_leader_exhaustion_short;" + str(row.get("reason") or ""),
                score=_number(row.get("score"), 0.0),
            )
            if candidate is not None:
                open_rows.append(candidate)

    # This market-state setup is independent of the older dense/sparse
    # campaign reset.  Keeping it outside that lifecycle gate makes the live
    # warm buffer and the longer backtest history emit the same current-bar
    # decision without replaying an old inventory seed.
    open_rows.extend(
        _build_meme_drawdown_reclaim_signals(
            open_rows,
            all_features,
        ),
    )

    # Hypothesis 2b: a high-volume blowoff closes below its intrabar high while
    # the asset remains the strongest return leader.  This is feature-derived
    # because the generic template intentionally emits only a trend-long here.
    qvr = series_number(features, "quote_volume_ratio_med30")
    blowoff = features[
        series_number(features, "quote_volume_rank_pct").ge(0.55)
        & series_number(features, "market_up_breadth20").ge(0.90)
        & series_number(features, "ret144").ge(3.0)
        & series_number(features, "ret288").ge(3.0)
        & series_number(features, "ret60").ge(1.50)
        & series_number(features, "ret20").ge(0.80)
        & series_number(features, "ret60_rank_pct").ge(0.90)
        & qvr.ge(3.0)
        & series_number(features, "rsi14").between(65.0, 75.0, inclusive="both")
        & series_number(features, "taker_buy_ratio").le(0.50)
        & series_number(features, "ret1").le(0.0)
        & series_number(features, "upper_wick_pct").ge(0.40)
    ]
    for _, row in blowoff.iterrows():
        candidate = direct_open(
            row,
            action="OPEN_SHORT",
            reason=(
                "meme_volume_blowoff_rejection_short;"
                f"ret20={_number(row.get('ret20')):.2f};ret60={_number(row.get('ret60')):.2f};"
                f"ret144={_number(row.get('ret144')):.2f};ret288={_number(row.get('ret288')):.2f};"
                f"qvr={_number(row.get('quote_volume_ratio_med30')):.2f};"
                f"rsi={_number(row.get('rsi14')):.1f};taker={_number(row.get('taker_buy_ratio')):.2f}"
            ),
            score=4.0,
        )
        if candidate is not None:
            open_rows.append(candidate)

    # Hypothesis 3a: a short-horizon, high-liquidity leader stops extending
    # immediately after an aggressive-buying burst.  Requiring the prior
    # closed bar to establish the burst and the current closed bar to confirm
    # both flow and volume decay avoids fading an intact squeeze.  The
    # medium/long return ceilings distinguish an acute impulse from the mature
    # multi-horizon extension handled by the separate reversal probe below.
    burst_features = all_features[
        all_features.get(
            "symbol",
            pd.Series("", index=all_features.index),
        ).astype(str).isin(_PRECISION_MEME_SYMBOLS)
    ].copy()
    burst_features["bar_time"] = pd.to_datetime(
        burst_features.get("bar_time"),
        errors="coerce",
    )
    burst_features = burst_features.dropna(subset=["bar_time"]).sort_values(
        ["symbol", "bar_time"],
    )
    burst_prior_columns = (
        "close",
        "ret20",
        "ret60",
        "donchian20_pos",
        "taker_buy_ratio",
        "quote_volume_ratio_med30",
        "rsi14",
        "market_up_breadth20",
    )
    for column in burst_prior_columns:
        if column not in burst_features:
            burst_features[column] = np.nan
    burst_groups = burst_features.groupby("symbol", sort=False)
    for column in burst_prior_columns:
        burst_features[f"prior_{column}"] = burst_groups[column].shift(1)
    burst_qvr = series_number(burst_features, "quote_volume_ratio_med30")
    prior_burst_qvr = series_number(
        burst_features,
        "prior_quote_volume_ratio_med30",
    )
    short_horizon_burst_stall = (
        series_number(burst_features, "prior_ret20").ge(1.50)
        & series_number(burst_features, "prior_ret60").ge(1.20)
        & series_number(burst_features, "prior_donchian20_pos").ge(0.85)
        & series_number(burst_features, "prior_rsi14").ge(70.0)
        & series_number(burst_features, "prior_taker_buy_ratio").ge(0.60)
        & prior_burst_qvr.ge(2.0)
        & series_number(burst_features, "prior_market_up_breadth20").ge(0.75)
        & series_number(burst_features, "close").le(
            series_number(burst_features, "prior_close") * 1.001,
        )
        & series_number(burst_features, "ret20").ge(1.25)
        & series_number(burst_features, "ret60").ge(1.50)
        & series_number(burst_features, "ret144").between(
            0.50,
            2.00,
            inclusive="both",
        )
        & series_number(burst_features, "ret288").between(
            0.50,
            3.00,
            inclusive="both",
        )
        & series_number(burst_features, "donchian20_pos").ge(0.85)
        & series_number(burst_features, "rsi14").ge(68.0)
        & series_number(burst_features, "taker_buy_ratio").le(0.50)
        & burst_qvr.between(1.0, 3.0, inclusive="both")
        & burst_qvr.le(prior_burst_qvr * 0.75)
        & series_number(burst_features, "market_up_breadth20").ge(0.70)
        & series_number(burst_features, "ret20_rank_pct").ge(0.75)
        & series_number(burst_features, "ret60_rank_pct").ge(0.65)
        & series_number(burst_features, "quote_volume_rank_pct").ge(0.85)
    )
    burst_stall = burst_features[short_horizon_burst_stall].copy()

    # Hypothesis 3b: returns remain elevated from 100 minutes through 24 hours,
    # but the latest bar rejects the high as aggressive buying fades.
    squeeze_column = "squeeze120_rank" if "squeeze120_rank" in features else "squeeze120_rank_pct"
    cross_horizon_stall = (
        series_number(features, "quote_volume_rank_pct").between(0.70, 0.85, inclusive="both")
        & series_number(features, "market_up_breadth20").ge(0.70)
        & series_number(features, "ret60").ge(3.0)
        & series_number(features, "ret144").ge(3.0)
        & series_number(features, "ret288").ge(3.0)
        & series_number(features, "ret20").between(1.0, 2.0, inclusive="both")
        & series_number(features, "ret60_rank_pct").ge(0.85)
        & series_number(features, "rsi14").between(62.0, 70.0, inclusive="both")
        & series_number(features, "taker_buy_ratio").le(0.50)
        & qvr.between(1.0, 2.0, inclusive="both")
        & series_number(features, squeeze_column).ge(0.75)
        & series_number(features, "ret3").le(0.05)
    )
    extension_features = all_features[
        all_features.get(
            "symbol",
            pd.Series("", index=all_features.index),
        ).astype(str).isin(_PRECISION_MEME_SYMBOLS)
    ].copy()
    eth_anchor_ret60 = (
        all_features[
            all_features.get(
                "symbol",
                pd.Series("", index=all_features.index),
            ).astype(str).eq("ETHUSDT")
        ]
        .drop_duplicates("bar_time", keep="last")
        .set_index("bar_time")
        .get("ret60", pd.Series(dtype=float))
    )
    mapped_eth_anchor_ret60 = pd.to_datetime(
        extension_features.get("bar_time"),
        errors="coerce",
    ).map(pd.to_numeric(eth_anchor_ret60, errors="coerce"))
    extension_features["eth_anchor_ret60"] = series_number(
        extension_features,
        "eth_anchor_ret60",
    ).fillna(mapped_eth_anchor_ret60)
    extension_qvr = series_number(extension_features, "quote_volume_ratio_med30")
    extension_reversal = (
        series_number(extension_features, "ret20").ge(1.5)
        & series_number(extension_features, "ret60").ge(2.5)
        & series_number(extension_features, "ret144").ge(2.5)
        & series_number(extension_features, "ret288").ge(
            _MEME_EXTENSION_RET288_FLOOR,
        )
        & series_number(extension_features, "ret60_rank_pct").ge(
            _MEME_EXTENSION_RET60_RANK_FLOOR,
        )
        & series_number(extension_features, "eth_anchor_ret60").between(
            _MEME_EXTENSION_ETH_RET60_FLOOR,
            _MEME_EXTENSION_ETH_RET60_CEILING,
            inclusive="both",
        )
        & series_number(extension_features, "market_up_breadth20").ge(0.70)
        & series_number(extension_features, "adx14").ge(30.0)
        & series_number(extension_features, "rsi14").between(55.0, 72.0, inclusive="both")
        & series_number(extension_features, "taker_buy_ratio").le(0.45)
        & extension_qvr.between(0.50, 3.00, inclusive="both")
        & series_number(extension_features, "ret1").le(-0.20)
        & series_number(extension_features, "ret3").le(0.30)
        & series_number(extension_features, "upper_wick_pct").ge(0.10)
    )
    sparse_stall = features[cross_horizon_stall].copy()
    sparse_stall["extension_reversal_selected"] = False
    sparse_stall["short_horizon_burst_selected"] = False
    extension_stall = extension_features[extension_reversal].copy()
    extension_stall["extension_reversal_selected"] = True
    extension_stall["short_horizon_burst_selected"] = False
    burst_stall["extension_reversal_selected"] = False
    burst_stall["short_horizon_burst_selected"] = True
    stall = pd.concat(
        [sparse_stall, extension_stall, burst_stall],
        ignore_index=True,
        sort=False,
    ).drop_duplicates(["symbol", "bar_time"], keep="last")
    stall["extension_quality"] = (
        series_number(stall, "ret60")
        + series_number(stall, "ret144")
        + series_number(stall, "ret288")
    )
    stall_clock = "signal_time" if "signal_time" in stall else "bar_time"
    stall = stall.sort_values(
        [stall_clock, "extension_quality", "symbol"],
        ascending=[True, False, True],
    )
    prior_stall_by_symbol: dict[str, pd.Timestamp] = {}
    recent_stall_times: list[pd.Timestamp] = []
    for _, row in stall.iterrows():
        signal_time = pd.Timestamp(row.get("bar_time", row.get("signal_time")))
        symbol = str(row.get("symbol") or "")
        previous = prior_stall_by_symbol.get(symbol)
        if previous is not None and signal_time < previous + pd.Timedelta(hours=12):
            continue
        recent_stall_times = [
            value
            for value in recent_stall_times
            if signal_time < value + pd.Timedelta(hours=1)
        ]
        if len(recent_stall_times) >= 2:
            continue
        prior_stall_by_symbol[symbol] = signal_time
        recent_stall_times.append(signal_time)
        candidate = direct_open(
            row,
            action="OPEN_SHORT",
            reason=(
                "meme_cross_horizon_stall_short;"
                + (
                    "phase=extension_reversal;"
                    if bool(row.get("extension_reversal_selected"))
                    else ""
                )
                + (
                    "phase=short_horizon_burst_stall;"
                    if bool(row.get("short_horizon_burst_selected"))
                    else ""
                )
                + f"ret20={_number(row.get('ret20')):.2f};ret60={_number(row.get('ret60')):.2f};"
                f"ret144={_number(row.get('ret144')):.2f};ret288={_number(row.get('ret288')):.2f};"
                f"eth60={_number(row.get('eth_anchor_ret60')):.2f};"
                f"qvr={_number(row.get('quote_volume_ratio_med30')):.2f};"
                f"rsi={_number(row.get('rsi14')):.1f};taker={_number(row.get('taker_buy_ratio')):.2f}"
            ),
            score=4.0,
        )
        if candidate is not None:
            open_rows.append(candidate)

    opens = pd.DataFrame(open_rows, columns=[column for column in columns if column != "signal_id"])
    if opens.empty:
        opens = pd.DataFrame(columns=columns)
    else:
        opens = opens.drop_duplicates(
            ["symbol", "action", "signal_time", "signal_price", "reason"],
            keep="first",
        ).sort_values(["signal_time", "symbol", "action"]).reset_index(drop=True)
        opens["signal_id"] = [f"WJMEMEPRECISIONOPEN{index + 1:06d}" for index in range(len(opens))]
        opens = opens.reindex(columns=columns)

    raw_closes = _meme_reason_metric_frame(raw_close_signals, action_prefix="CLOSE_")
    lifecycle_features = all_features[
        all_features.get(
            "symbol",
            pd.Series("", index=all_features.index),
        ).astype(str).isin(_PRECISION_MEME_SYMBOLS)
    ].copy()
    feature_by_symbol = {
        symbol: frame.sort_values("bar_time").copy()
        for symbol, frame in lifecycle_features.groupby("symbol", sort=False)
    }
    close_rows: list[dict[str, object]] = [
        {
            **row,
            "decision_time": pd.Timestamp(row["signal_time"]),
            "order_style": "",
            "setup_signal_id": "",
            "pending_order": False,
            "valid_until": pd.Timestamp(row["signal_time"]),
        }
        for row in dense_close_rows
    ]
    sparse_reason = opens.get("reason", pd.Series("", index=opens.index)).astype(str)
    executed_opens = opens[
        ~opens.get("pending_order", pd.Series(False, index=opens.index)).eq(True)
        & (
            sparse_reason.str.startswith(
                (
                    "meme_breadth_absorption_long;",
                    "meme_relative_leader_exhaustion_short;",
                    "meme_volume_blowoff_rejection_short;",
                    "meme_cross_horizon_stall_short;",
                    "meme_drawdown_reclaim_long;",
                ),
            )
            | sparse_reason.str.contains("meme_liquidation_panic_pullback_long;", regex=False)
            | sparse_reason.str.contains("meme_liquidation_recovery_add_long;", regex=False)
        )
    ].copy()
    for _, opened in executed_opens.iterrows():
        symbol = str(opened["symbol"])
        open_time = pd.Timestamp(opened["signal_time"])
        open_price = _number(opened.get("signal_price"))
        action = str(opened["action"])
        direction = "LONG" if action == "OPEN_LONG" else "SHORT"
        opened_reason = str(opened.get("reason") or "")
        opened_is_extension_probe = "phase=extension_reversal" in opened_reason
        executed_reason = executed_opens.get(
            "reason",
            pd.Series("", index=executed_opens.index),
        ).astype(str)
        later_same_symbol = executed_opens[
            executed_opens["symbol"].eq(symbol)
            & pd.to_datetime(executed_opens["signal_time"], errors="coerce").gt(open_time)
            & executed_reason.str.contains(
                "phase=extension_reversal",
                regex=False,
            ).eq(opened_is_extension_probe)
        ]
        end_time = (
            pd.Timestamp(later_same_symbol.iloc[0]["signal_time"])
            if not later_same_symbol.empty
            else pd.Timestamp(feature_by_symbol[symbol]["bar_time"].max()) + pd.Timedelta(minutes=5)
        )
        scoped_features = feature_by_symbol[symbol]
        scoped_features = scoped_features[
            scoped_features["bar_time"].gt(open_time)
            & scoped_features["bar_time"].lt(end_time)
        ].copy()
        scoped_features["campaign_age_minutes"] = (
            scoped_features["bar_time"] - open_time
        ).dt.total_seconds() / 60.0
        close_price = series_number(scoped_features, "close")
        if direction == "LONG":
            scoped_features["campaign_profit_pct"] = (close_price / open_price - 1.0) * 100.0
            close_action = "CLOSE_LONG"
        else:
            scoped_features["campaign_profit_pct"] = (1.0 - close_price / open_price) * 100.0
            close_action = "CLOSE_SHORT"
        scoped_raw = raw_closes[
            raw_closes["symbol"].eq(symbol)
            & raw_closes["action"].eq(close_action)
            & raw_closes["signal_time"].gt(open_time)
            & raw_closes["signal_time"].lt(end_time)
        ].copy()
        if not scoped_raw.empty:
            raw_feature = scoped_raw.set_index("signal_time").join(
                scoped_features.set_index("bar_time")[["close", "campaign_age_minutes", "campaign_profit_pct"]],
                how="left",
            ).reset_index(names="signal_time")
        else:
            raw_feature = scoped_raw.copy()
            raw_feature["close"] = pd.Series(dtype=float)
            raw_feature["campaign_age_minutes"] = pd.Series(dtype=float)
            raw_feature["campaign_profit_pct"] = pd.Series(dtype=float)
        reason = str(opened.get("reason") or "")
        selected: pd.Series | None = None
        close_reason = ""
        if reason.startswith("meme_breadth_absorption_long;"):
            candidates = raw_feature[
                raw_feature["reason_family"].eq("long_profit_take_exhaustion")
                & raw_feature["campaign_age_minutes"].ge(600.0)
                & raw_feature["campaign_profit_pct"].ge(1.0)
                & raw_feature["ret20"].ge(1.0)
                & raw_feature["rsi"].ge(68.0)
                & raw_feature["qvr"].ge(1.0)
                & raw_feature["taker"].le(0.55)
            ].head(1)
            if not candidates.empty:
                selected = candidates.iloc[0]
                close_reason = "meme_absorption_mature_profit_exit"
        elif reason.startswith("meme_drawdown_reclaim_long;"):
            # A reclaimed drawdown lot can recover sharply while the broader
            # basket is still weak.  Harvest that individual lot once a mature
            # relative-strength impulse is confirmed; waiting for the generic
            # overbought raw-close template leaves the recovered inventory
            # exposed to the immediate giveback.
            candidates = scoped_features[
                scoped_features["campaign_age_minutes"].ge(420.0)
                & scoped_features["campaign_profit_pct"].ge(1.50)
                & series_number(scoped_features, "ret60").ge(1.50)
                & series_number(scoped_features, "quote_volume_ratio_med30").ge(3.0)
                & series_number(scoped_features, "rsi14").ge(60.0)
                & series_number(scoped_features, "taker_buy_ratio").ge(0.60)
                & series_number(scoped_features, "donchian60_pos").ge(0.85)
            ].head(1)
            if not candidates.empty:
                selected = candidates.iloc[0]
                close_reason = "meme_drawdown_recovery_impulse_harvest"
            else:
                candidates = raw_feature[
                    raw_feature["reason_family"].eq("long_profit_take_exhaustion")
                    & raw_feature["campaign_age_minutes"].ge(600.0)
                    & raw_feature["campaign_profit_pct"].ge(1.0)
                    & raw_feature["ret20"].ge(1.0)
                    & raw_feature["rsi"].ge(68.0)
                    & raw_feature["qvr"].ge(1.0)
                    & raw_feature["taker"].le(0.55)
                ].head(1)
                if not candidates.empty:
                    selected = candidates.iloc[0]
                    close_reason = "meme_liquidation_campaign_profit_exit"
        elif "meme_liquidation_recovery_add_long;" in reason:
            candidates = raw_feature[
                raw_feature["reason_family"].eq("long_profit_take_exhaustion")
                & raw_feature["campaign_age_minutes"].ge(600.0)
                & raw_feature["campaign_profit_pct"].ge(1.0)
                & raw_feature["ret20"].ge(1.0)
                & raw_feature["rsi"].ge(68.0)
                & raw_feature["qvr"].ge(1.0)
                & raw_feature["taker"].le(0.55)
            ].head(1)
            if not candidates.empty:
                selected = candidates.iloc[0]
                close_reason = "meme_liquidation_campaign_profit_exit"
        elif "meme_liquidation_panic_pullback_long;" in reason:
            candidates = scoped_features[
                scoped_features["campaign_age_minutes"].ge(1440.0)
                & scoped_features["campaign_profit_pct"].ge(0.50)
                & series_number(scoped_features, "ret20").ge(1.50)
                & series_number(scoped_features, "quote_volume_ratio_med30").ge(10.0)
                & series_number(scoped_features, "rsi14").ge(75.0)
                & series_number(scoped_features, "market_up_breadth20").ge(0.90)
            ].head(1)
            if not candidates.empty:
                selected = candidates.iloc[0]
                close_reason = "meme_panic_recovery_volume_climax_exit"
        elif reason.startswith("meme_relative_leader_exhaustion_short;"):
            candidates = raw_feature[
                raw_feature["reason_family"].eq("short_profit_take_exhaustion")
                & raw_feature["campaign_age_minutes"].ge(150.0)
                & raw_feature["campaign_profit_pct"].ge(0.60)
                & raw_feature["ret20"].le(-1.0)
                & raw_feature["qvr"].ge(1.0)
                & raw_feature["rsi"].le(40.0)
            ].head(1)
            if not candidates.empty:
                selected = candidates.iloc[0]
                close_reason = "meme_relative_leader_cascade_profit_exit"
        elif reason.startswith("meme_volume_blowoff_rejection_short;"):
            candidates = scoped_features[
                scoped_features["campaign_age_minutes"].ge(480.0)
                & scoped_features["campaign_profit_pct"].ge(0.70)
                & series_number(scoped_features, "quote_volume_ratio_med30").ge(3.0)
                & series_number(scoped_features, "rsi14").ge(40.0)
                & series_number(scoped_features, "ret60").le(-1.50)
            ].head(1)
            if not candidates.empty:
                selected = candidates.iloc[0]
                close_reason = "meme_blowoff_post_cascade_rebound_exit"
        elif reason.startswith("meme_cross_horizon_stall_short;"):
            if opened_is_extension_probe:
                # Extension probes wait for a real reversal after first
                # reaching a useful short profit.  This avoids liquidating on
                # the first continuation candle and uses only closed bars.
                rebound = scoped_features.copy()
                rebound["prior_peak_profit_pct"] = (
                    pd.to_numeric(rebound["campaign_profit_pct"], errors="coerce")
                    .shift(1)
                    .cummax()
                )
                candidates = rebound[
                    rebound["campaign_age_minutes"].ge(90.0)
                    & rebound["prior_peak_profit_pct"].ge(0.75)
                    & series_number(rebound, "ret1").ge(0.10)
                    & series_number(rebound, "taker_buy_ratio").ge(0.65)
                ].head(1)
                if not candidates.empty:
                    selected = candidates.iloc[0]
                    close_reason = "meme_stall_buyer_rebound_exit"
            else:
                candidates = raw_feature[
                    raw_feature["reason_family"].eq("short_profit_take_exhaustion")
                    & raw_feature["campaign_age_minutes"].ge(60.0)
                    & raw_feature["campaign_profit_pct"].ge(0.50)
                    & raw_feature["ret20"].le(-0.50)
                ].head(1)
                if not candidates.empty:
                    selected = candidates.iloc[0]
                    close_reason = "meme_stall_fast_profit_exit"
        if selected is None:
            continue
        signal_time = pd.Timestamp(selected.get("bar_time", selected.get("signal_time")))
        selected_price = _number(selected.get("close"), feature_number(symbol, signal_time, "close"))
        close_rows.append(
            {
                "symbol": symbol,
                "action": close_action,
                "signal_time": signal_time,
                "signal_price": selected_price,
                "score": _number(opened.get("score"), 0.0),
                "reason": f"{close_reason};open_signal={opened['signal_id']}",
                "decision_time": signal_time,
                "order_style": "",
                "setup_signal_id": "",
                "pending_order": False,
                "valid_until": signal_time,
            },
        )

    # A mature BCH recovery can finish with one high-volume impulse followed
    # by a quieter closed bar that remains pinned at the range high while
    # aggressive buying fades.  This is a portfolio inventory exit, not a
    # single-lot harvest: all recovery and drawdown-reclaim lots share the same
    # now-exhausted recovery regime and must be released together.
    bch_features = feature_by_symbol.get("BCHUSDT", pd.DataFrame()).copy()
    for column in (
        "ret20",
        "ret60",
        "ret144",
        "donchian60_pos",
        "taker_buy_ratio",
        "quote_volume_ratio_med30",
    ):
        bch_features[f"prior_{column}"] = series_number(
            bch_features,
            column,
        ).shift(1)
    bch_inventory_reset = bch_features[
        series_number(bch_features, "prior_ret20").ge(1.45)
        & series_number(bch_features, "prior_ret60").ge(1.30)
        & series_number(bch_features, "prior_ret144").ge(4.00)
        & series_number(bch_features, "prior_donchian60_pos").ge(0.95)
        & series_number(bch_features, "prior_taker_buy_ratio").ge(0.65)
        & series_number(bch_features, "prior_quote_volume_ratio_med30").ge(4.00)
        & series_number(bch_features, "ret20").ge(1.45)
        & series_number(bch_features, "ret60").ge(1.50)
        & series_number(bch_features, "ret144").ge(4.00)
        & series_number(bch_features, "ret288").ge(1.50)
        & series_number(bch_features, "donchian60_pos").ge(0.98)
        & series_number(bch_features, "rsi14").ge(75.0)
        & series_number(bch_features, "adx14").ge(35.0)
        & series_number(bch_features, "taker_buy_ratio").le(0.50)
        & series_number(bch_features, "quote_volume_ratio_med30").le(1.50)
        & series_number(bch_features, "ret20_rank_pct").ge(0.90)
        & series_number(bch_features, "ret60_rank_pct").ge(0.90)
    ].copy()
    runtime_active = (initial_runtime_state or {}).get("active", {})
    runtime_has_bch_long = bool(
        isinstance(runtime_active, Mapping)
        and isinstance(runtime_active.get(("BCHUSDT", "LONG")), list)
        and runtime_active.get(("BCHUSDT", "LONG")),
    )
    bch_long_opens = executed_opens[
        executed_opens.get(
            "symbol",
            pd.Series("", index=executed_opens.index),
        ).astype(str).eq("BCHUSDT")
        & executed_opens.get(
            "action",
            pd.Series("", index=executed_opens.index),
        ).astype(str).eq("OPEN_LONG")
    ]
    for _, selected in bch_inventory_reset.iterrows():
        signal_time = pd.Timestamp(selected["bar_time"])
        has_prior_strategy_inventory = runtime_has_bch_long or bool(
            pd.to_datetime(
                bch_long_opens.get("signal_time"),
                errors="coerce",
            ).lt(signal_time).any(),
        )
        if not has_prior_strategy_inventory:
            continue
        close_rows.append(
            {
                "symbol": "BCHUSDT",
                "action": "CLOSE_LONG",
                "signal_time": signal_time,
                "signal_price": _number(selected.get("close")),
                "score": 4.50,
                "reason": (
                    "meme_bch_recovery_inventory_reset;"
                    f"ret20={_number(selected.get('ret20')):.2f};"
                    f"ret60={_number(selected.get('ret60')):.2f};"
                    f"ret144={_number(selected.get('ret144')):.2f};"
                    f"ret288={_number(selected.get('ret288')):.2f};"
                    f"qvr={_number(selected.get('quote_volume_ratio_med30')):.2f};"
                    f"rsi={_number(selected.get('rsi14')):.1f};"
                    f"adx={_number(selected.get('adx14')):.1f};"
                    f"taker={_number(selected.get('taker_buy_ratio')):.2f};"
                    "prior_qvr="
                    f"{_number(selected.get('prior_quote_volume_ratio_med30')):.2f};"
                    "prior_taker="
                    f"{_number(selected.get('prior_taker_buy_ratio')):.2f}"
                ),
                "decision_time": signal_time,
                "order_style": "",
                "setup_signal_id": "",
                "pending_order": False,
                "valid_until": signal_time,
            },
        )

    closes = pd.DataFrame(close_rows, columns=[column for column in columns if column != "signal_id"])
    if closes.empty:
        closes = pd.DataFrame(columns=columns)
    else:
        closes = closes.sort_values(["signal_time", "symbol", "action"]).reset_index(drop=True)
        closes["signal_id"] = [f"WJMEMEPRECISIONCLOSE{index + 1:06d}" for index in range(len(closes))]
        closes = closes.reindex(columns=columns)
    return opens, closes


def _eth_reason_metric_frame(signals: pd.DataFrame, *, action_prefix: str) -> pd.DataFrame:
    """Attach numeric factor fields embedded by the closed-bar template layer."""

    out = _normalized_eth_actions(signals, action_prefix=action_prefix)
    if out.empty:
        for column in ("reason_family", "ret20", "ret60", "ret144", "ret288", "qvr", "rsi", "adx", "taker", "don", "down"):
            out[column] = pd.Series(dtype=float if column != "reason_family" else str)
        return out
    reason = out.get("reason", pd.Series("", index=out.index)).astype(str)
    out["reason_family"] = reason.str.split(";").str[0]
    for column in ("ret20", "ret60", "ret144", "ret288", "qvr", "rsi", "adx", "taker", "don", "down"):
        out[column] = pd.to_numeric(
            reason.str.extract(rf"(?:^|;){column}=([-0-9.]+)")[0],
            errors="coerce",
        )
    return out


def build_causal_eth_positions_from_selected_signals(
    selected_open_signals: pd.DataFrame,
    selected_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild the ETH lot state from actions available at each closed-bar prefix."""

    columns = [
        "open_signal_id", "close_signal_id", "symbol", "direction", "open_time", "close_time",
        "open_price", "close_price", "open_reason_family", "close_reason_family", "open_reason",
        "close_reason",
    ]
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    features = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].drop_duplicates("bar_time", keep="last").sort_values("bar_time")
    if features.empty:
        return pd.DataFrame(columns=columns)
    price_lookup = pd.to_numeric(features.set_index("bar_time").get("close"), errors="coerce")
    end_time = pd.Timestamp(features["bar_time"].max())

    opens = _normalized_eth_actions(selected_open_signals, action_prefix="OPEN_")
    closes = _normalized_eth_actions(selected_close_signals, action_prefix="CLOSE_")
    events: list[tuple[pd.Timestamp, int, pd.Series]] = []
    for _, signal in opens.iterrows():
        events.append((pd.Timestamp(signal["signal_time"]), 1, signal))
    for _, signal in closes.iterrows():
        events.append((pd.Timestamp(signal["signal_time"]), 0, signal))
    events.sort(key=lambda item: (item[0], item[1]))

    active: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for signal_time, event_priority, signal in events:
        action = str(signal.get("action") or "")
        signal_id = str(signal.get("signal_id") or "")
        reason = str(signal.get("reason") or "")
        supplied_price = _number(signal.get("signal_price"))
        market_price = _number(price_lookup.get(signal_time))
        signal_price = supplied_price if np.isfinite(supplied_price) else market_price
        if event_priority == 1:
            direction = "LONG" if action == "OPEN_LONG" else "SHORT"
            active[signal_id] = {
                "open_signal_id": signal_id,
                "symbol": "ETHUSDT",
                "direction": direction,
                "open_time": signal_time,
                "open_price": signal_price,
                "open_reason_family": _source_reason_family(reason),
                "open_reason": reason,
            }
            continue

        direction = "LONG" if action == "CLOSE_LONG" else "SHORT"
        linked_id = ""
        if "open_signal=" in reason:
            linked_id = reason.split("open_signal=", 1)[1].split(";", 1)[0]
        lot = active.pop(linked_id, None) if linked_id else None
        if lot is None:
            candidates = [
                (key, value)
                for key, value in active.items()
                if str(value.get("direction")) == direction
            ]
            if candidates:
                oldest_id, lot = min(candidates, key=lambda item: pd.Timestamp(item[1]["open_time"]))
                active.pop(oldest_id, None)
        if lot is None:
            continue
        rows.append(
            {
                **lot,
                "close_signal_id": signal_id,
                "close_time": signal_time,
                "close_price": signal_price,
                "close_reason_family": _source_reason_family(reason),
                "close_reason": reason,
            },
        )

    final_price = _number(price_lookup.get(end_time))
    for lot in active.values():
        rows.append(
            {
                **lot,
                "close_signal_id": "",
                "close_time": end_time,
                "close_price": final_price,
                "close_reason_family": "open_at_prefix_end",
                "close_reason": "open_at_prefix_end",
            },
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["open_time", "close_time", "open_signal_id"],
    ).reset_index(drop=True)


def _source_reason_family(reason: object) -> str:
    text = str(reason or "")
    marker = "source_signal="
    if marker in text:
        tail = text.split(marker, 1)[1].split(";", 2)
        if len(tail) >= 2:
            return tail[1]
    return text.split(";", 1)[0]


def _signal_reason_metric(reason: object, key: str, *, fallback: float = np.nan) -> float:
    """Read a source-bar metric embedded in a causal event-selected signal."""

    prefix = f"{key}="
    for token in str(reason or "").split(";"):
        if not token.startswith(prefix):
            continue
        try:
            return float(token[len(prefix):])
        except (TypeError, ValueError):
            break
    return float(fallback)


def _eth_precision_anchor_short_allowed(feature: pd.Series) -> bool:
    ret20 = _number(feature.get("ret20"))
    ret60 = _number(feature.get("ret60"))
    ret288 = _number(feature.get("ret288"))
    volume_ratio = _number(feature.get("quote_volume_ratio_med30"))
    mature_low_volume_stall = (
        ret60 >= 4.0
        and ret288 >= 4.0
        and volume_ratio <= 1.0
        and 62.0 <= _number(feature.get("rsi14")) <= 72.0
        and 45.0 <= _number(feature.get("adx14")) <= 60.0
    )
    if mature_low_volume_stall:
        return True
    if ret288 < -4.0:
        return False
    if ret288 > 0.0 and volume_ratio < 2.0 and 0.25 < ret20 < 2.50:
        return False
    if ret288 > 0.0 and 2.0 <= volume_ratio < 10.0 and ret60 < 2.0:
        return False
    if ret288 > 5.0 and ret60 < 2.0:
        return False
    return True


def _eth_precision_direct_combo_allowed(
    feature: pd.Series,
    *,
    family: str,
    action: str,
    campaign_ready: bool = False,
) -> bool:
    """Admit only independently confirmed direct ETH combo structures."""

    if family == "eth_anchor_exhaustion_short" and action == "OPEN_SHORT":
        mature_low_volume_stall = (
            _number(feature.get("ret60")) >= 4.0
            and _number(feature.get("ret288")) >= 4.0
            and _number(feature.get("quote_volume_ratio_med30")) <= 1.0
            and 62.0 <= _number(feature.get("rsi14")) <= 72.0
            and 45.0 <= _number(feature.get("adx14")) <= 60.0
        )
        return bool(
            mature_low_volume_stall
            or (
                _number(feature.get("ret20")) >= 3.0
                and _number(feature.get("quote_volume_ratio_med30")) <= 0.80
                and _number(feature.get("taker_buy_ratio")) >= 0.55
                and 50.0 <= _number(feature.get("adx14")) <= 65.0
            ),
        )
    if family == "eth_anchor_long" and action == "OPEN_LONG":
        return bool(
            _number(feature.get("taker_buy_ratio")) >= 0.65
            and _number(feature.get("quote_volume_ratio_med30")) >= 1.50
            and _number(feature.get("ret60")) <= 0.0
            and _number(feature.get("adx14")) <= 25.0,
        )
    if family == "capitulation_absorption_long" and action == "OPEN_LONG":
        return bool(
            _number(feature.get("ret288")) <= -4.50
            and _number(feature.get("ret20")) <= -0.80
            and 3.0 <= _number(feature.get("quote_volume_ratio_med30")) <= 10.0
            and _number(feature.get("rsi14")) <= 30.0
            and _number(feature.get("donchian20_pos")) <= 0.20,
        )
    if campaign_ready and family in {
        "eth_anchor_exhaustion_short",
        "capitulation_absorption_long",
        "core_downtrend_short",
        "core_trend_long",
        "eth_anchor_long",
        "range_reclaim_long",
    }:
        if family == "eth_anchor_exhaustion_short":
            return _number(feature.get("quote_volume_ratio_med30")) <= 40.0
        return True
    return False


def _eth_precision_active_addon_allowed(feature: pd.Series, *, campaign_ready: bool = False) -> bool:
    ret20 = _number(feature.get("ret20"))
    zscore = _number(feature.get("bb20_zscore"))
    # Shallow low-volume resets failed the full-window stop-aware audit.
    # Add inventory only after a genuine volatility reset.
    deep_reset = ret20 <= -1.50 and zscore <= -2.0
    mild_flow_reset = abs(ret20) <= 0.10 and -1.0 <= zscore <= 0.0
    return bool(deep_reset or (campaign_ready and mild_flow_reset))


def _eth_precision_breakout_addon_allowed(feature: pd.Series, *, campaign_ready: bool = False) -> bool:
    """Reject exhausted breakout adds while retaining healthy expansion."""

    return bool(campaign_ready or (
        _number(feature.get("rsi14")) <= 75.0
        and _number(feature.get("adx14")) <= 30.0
        and _number(feature.get("atr14_pct")) >= 0.40
        and _number(feature.get("ret288")) <= 3.0
    ))


def _in_time_windows(
    timestamp: pd.Timestamp,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    return any(start <= timestamp <= end for start, end in windows)


def build_eth_capitulation_profit_ladder_signals(
    strategy_positions: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    raw_close_signals: pd.DataFrame | None = None,
    config: EthCapitulationProfitLadderConfig | None = None,
) -> pd.DataFrame:
    """Simulate resting, sequential take-profit runners for ETH capitulation entries."""

    cfg = config or EthCapitulationProfitLadderConfig()
    columns = ["signal_id", "symbol", "action", "signal_time", "signal_price", "score", "reason"]
    positions = _normalized_eth_long_positions(strategy_positions)
    if positions.empty or feature_rows.empty:
        return pd.DataFrame(columns=columns)
    positions = positions[
        positions.get("open_reason_family", pd.Series("", index=positions.index))
        .astype(str).eq("capitulation_absorption_long")
    ]
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    features = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].sort_values("bar_time")
    rows: list[dict[str, object]] = []
    raw = raw_close_signals.copy() if raw_close_signals is not None else pd.DataFrame()
    if not raw.empty:
        raw["signal_time"] = pd.to_datetime(raw.get("signal_time"), errors="coerce")
        raw = raw[
            raw.get("symbol", pd.Series("", index=raw.index)).astype(str).eq("ETHUSDT")
            & raw.get("action", pd.Series("", index=raw.index)).astype(str).eq("CLOSE_LONG")
            & raw.get("reason", pd.Series("", index=raw.index)).astype(str)
            .str.startswith("long_profit_take_exhaustion;")
            & raw["signal_time"].notna()
        ].copy()
        raw["volume_ratio"] = raw.get("reason", pd.Series("", index=raw.index)).astype(str).str.extract(
            r"qvr=([0-9.]+)",
        )[0].astype(float)
        raw_prices = raw["signal_time"].map(
            pd.to_numeric(features.drop_duplicates("bar_time").set_index("bar_time").get("close"), errors="coerce"),
        )
        raw["signal_price"] = (
            pd.to_numeric(raw["signal_price"], errors="coerce").fillna(raw_prices)
            if "signal_price" in raw
            else raw_prices
        )
    for position in positions.itertuples(index=False):
        open_price = _number(getattr(position, "open_price", np.nan))
        if not np.isfinite(open_price) or open_price <= 0.0:
            continue
        activation = pd.Timestamp(position.open_time)
        bars = features[
            features["bar_time"].gt(pd.Timestamp(position.open_time))
            & features["bar_time"].le(pd.Timestamp(position.close_time))
        ]
        for tier_index, (profit_pct, runner_count, wait_after_minutes) in enumerate(cfg.tiers, start=1):
            target = open_price * (1.0 + max(0.0, float(profit_pct)) / 100.0)
            eligible = bars[
                features.loc[bars.index, "bar_time"].ge(activation)
                & pd.to_numeric(bars.get("high"), errors="coerce").ge(target)
            ]
            if eligible.empty:
                continue
            fill = eligible.iloc[0]
            bar_open = _number(fill.get("open"), target)
            fill_price = max(target, bar_open)
            for _ in range(max(1, int(runner_count))):
                rows.append(
                    {
                        "signal_id": f"WJETHTAKEPROFIT{len(rows) + 1:06d}",
                        "symbol": "ETHUSDT",
                        "action": "CLOSE_LONG",
                        "signal_time": pd.Timestamp(fill["bar_time"]),
                        "signal_price": fill_price,
                        "score": 3.0,
                        "reason": f"eth_capitulation_sequential_ladder_{tier_index}",
                    },
                )
            activation = pd.Timestamp(fill["bar_time"]) + pd.Timedelta(
                minutes=max(0.0, float(wait_after_minutes)),
            )
        if not raw.empty:
            runner_end = pd.Timestamp(position.open_time) + pd.Timedelta(
                minutes=max(0.0, float(cfg.final_runner_max_hold_minutes)),
            )
            runner = raw[
                raw["signal_time"].between(activation, runner_end, inclusive="both")
                & pd.to_numeric(raw.get("score"), errors="coerce").ge(float(cfg.final_runner_score_floor))
                & pd.to_numeric(raw.get("volume_ratio"), errors="coerce").ge(
                    float(cfg.final_runner_volume_ratio_floor),
                )
            ].sort_values("signal_time")
            if not runner.empty:
                final = runner.iloc[0]
                rows.append(
                    {
                        "signal_id": f"WJETHTAKEPROFIT{len(rows) + 1:06d}",
                        "symbol": "ETHUSDT",
                        "action": "CLOSE_LONG",
                        "signal_time": pd.Timestamp(final["signal_time"]),
                        "signal_price": float(final["signal_price"]),
                        "score": float(final["score"]),
                        "reason": "eth_capitulation_final_runner",
                    },
                )
    return pd.DataFrame(rows, columns=columns).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def build_eth_capitulation_exit_flip_short_signals(
    strategy_positions: pd.DataFrame,
    feature_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Flip a profitable ETH capitulation rebound only while the 24h regime remains bearish."""

    columns = ["signal_id", "symbol", "action", "signal_time", "signal_price", "score", "reason"]
    positions = _normalized_eth_long_positions(strategy_positions)
    if positions.empty or feature_rows.empty:
        return pd.DataFrame(columns=columns)
    positions = positions[
        positions.get("open_reason_family", pd.Series("", index=positions.index))
        .astype(str).eq("capitulation_absorption_long")
        & positions.get("close_reason_family", pd.Series("", index=positions.index))
        .astype(str).eq("long_profit_take_exhaustion")
    ]
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    features = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].drop_duplicates("bar_time", keep="last").set_index("bar_time")
    rows: list[dict[str, object]] = []
    for position in positions.itertuples(index=False):
        signal_time = pd.Timestamp(position.close_time)
        if signal_time not in features.index:
            continue
        row = features.loc[signal_time]
        if (
            _number(row.get("ret20")) < 1.0
            or _number(row.get("donchian20_pos")) < 0.80
            or _number(row.get("rsi14")) < 60.0
            or _number(row.get("ret288")) > -2.0
        ):
            continue
        rows.append(
            {
                "signal_id": f"WJETHFLIP{len(rows) + 1:06d}",
                "symbol": "ETHUSDT",
                "action": "OPEN_SHORT",
                "signal_time": signal_time,
                "signal_price": _number(row.get("close")),
                "score": 3.0,
                "reason": "eth_capitulation_exit_bearish_regime_flip_short",
            },
        )
    return pd.DataFrame(rows, columns=columns)


def build_eth_early_core_trend_entry_fills(
    raw_open_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    config: EthEarlyCorePullbackConfig | None = None,
    emit_pending_orders: bool = False,
) -> pd.DataFrame:
    """Enter a fresh ETH core impulse on its first causal pullback."""

    cfg = config or EthEarlyCorePullbackConfig()
    signals = raw_open_signals.copy()
    if signals.empty or feature_rows.empty:
        return build_conditional_factor_fills(pd.DataFrame(), feature_rows)
    signals["signal_time"] = pd.to_datetime(signals.get("signal_time"), errors="coerce")
    signals = signals[
        signals.get("symbol", pd.Series("", index=signals.index)).astype(str).eq("ETHUSDT")
        & signals.get("action", pd.Series("", index=signals.index)).astype(str).eq("OPEN_LONG")
        & signals.get("reason", pd.Series("", index=signals.index)).map(_source_reason_family).eq(
            "core_trend_long",
        )
    ].copy()
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
    ].drop_duplicates("bar_time", keep="last").set_index("bar_time")
    feature_columns = (
        "ret20",
        "ret60",
        "ret144",
        "ret288",
        "taker_buy_ratio",
        "quote_volume_ratio_med30",
        "rsi14",
        "adx14",
    )
    for column in feature_columns:
        signals[column] = signals["signal_time"].map(
            _numeric_series(eth, column),
        )
    signals = signals[
        signals["ret20"].between(float(cfg.ret20_floor), float(cfg.ret20_ceiling))
        & signals["ret60"].ge(float(cfg.ret60_floor))
        & signals["ret144"].le(signals["ret60"])
        & signals["ret288"].between(float(cfg.ret288_floor), float(cfg.ret288_ceiling))
        & signals["taker_buy_ratio"].ge(float(cfg.taker_buy_floor))
        & signals["quote_volume_ratio_med30"].ge(float(cfg.volume_ratio_floor))
        & signals["rsi14"].le(float(cfg.rsi14_ceiling))
        & signals["adx14"].between(float(cfg.adx14_floor), float(cfg.adx14_ceiling))
    ]
    return build_conditional_factor_fills(
        signals,
        features,
        config=ConditionalOrderConfig(
            valid_minutes=max(5.0, float(cfg.valid_minutes)),
            pullback_atr_multiplier=max(0.0, float(cfg.pullback_atr_multiplier)),
            emit_pending_orders=bool(emit_pending_orders),
            order_styles=("reversion_limit",),
        ),
    )


def build_eth_extreme_flush_reversal_entry_fills(
    feature_rows: pd.DataFrame,
    *,
    config: EthExtremeFlushReversalConfig | None = None,
    emit_pending_orders: bool = False,
) -> pd.DataFrame:
    """Confirm an extreme ETH liquidation flush with a later ATR recovery."""

    cfg = config or EthExtremeFlushReversalConfig()
    if feature_rows.empty:
        return build_conditional_factor_fills(pd.DataFrame(), feature_rows)
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].copy()
    setup_mask = (
        _numeric_series(eth, "ret20").le(float(cfg.ret20_ceiling))
        & _numeric_series(eth, "ret60").le(float(cfg.ret60_ceiling))
        & _numeric_series(eth, "rsi14").le(float(cfg.rsi14_ceiling))
        & _numeric_series(eth, "quote_volume_ratio_med30").ge(
            float(cfg.volume_ratio_floor),
        )
        & _numeric_series(eth, "taker_buy_ratio").le(
            float(cfg.taker_buy_ceiling),
        )
    )
    setups = eth.loc[setup_mask, ["bar_time"]].rename(columns={"bar_time": "signal_time"})
    setups["signal_id"] = [f"WJETHFLUSHSETUP{index + 1:06d}" for index in range(len(setups))]
    setups["symbol"] = "ETHUSDT"
    setups["action"] = "OPEN_LONG"
    setups["score"] = 5.0
    setups["reason"] = "eth_extreme_flush_reversal_long;capitulation_absorption_long"
    return build_conditional_factor_fills(
        setups,
        features,
        config=ConditionalOrderConfig(
            valid_minutes=max(5.0, float(cfg.valid_minutes)),
            momentum_atr_multiplier=max(0.0, float(cfg.confirmation_atr_multiplier)),
            emit_pending_orders=bool(emit_pending_orders),
            order_styles=("momentum_stop",),
        ),
    )


def build_eth_overbought_limit_exit_fills(
    selected_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    config: EthOverboughtLimitExitConfig | None = None,
    emit_pending_orders: bool = False,
) -> pd.DataFrame:
    """Rest a one-ATR limit after a closed-bar ETH long exhaustion setup."""

    cfg = config or EthOverboughtLimitExitConfig()
    signals = selected_close_signals.copy()
    if signals.empty or feature_rows.empty:
        return build_conditional_factor_fills(pd.DataFrame(), feature_rows)
    signals["signal_time"] = pd.to_datetime(signals.get("signal_time"), errors="coerce")
    signals = signals[
        signals.get("symbol", pd.Series("", index=signals.index)).astype(str).eq("ETHUSDT")
        & signals.get("action", pd.Series("", index=signals.index)).astype(str).eq("CLOSE_LONG")
        & signals.get("reason", pd.Series("", index=signals.index)).map(_source_reason_family).eq(
            "long_profit_take_exhaustion",
        )
    ].copy()
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
    ].drop_duplicates("bar_time", keep="last").set_index("bar_time")
    columns = ("ret20", "ret288", "rsi14", "adx14", "quote_volume_ratio_med30")
    for column in columns:
        signals[column] = signals["signal_time"].map(
            _numeric_series(eth, column),
        )
    signals = signals[
        signals["ret20"].between(float(cfg.ret20_floor), float(cfg.ret20_ceiling))
        & signals["ret288"].between(float(cfg.ret288_floor), float(cfg.ret288_ceiling))
        & signals["rsi14"].ge(float(cfg.rsi14_floor))
        & signals["adx14"].between(float(cfg.adx14_floor), float(cfg.adx14_ceiling))
        & signals["quote_volume_ratio_med30"].between(
            float(cfg.volume_ratio_floor),
            float(cfg.volume_ratio_ceiling),
        )
    ]
    return build_conditional_factor_fills(
        signals,
        features,
        config=ConditionalOrderConfig(
            valid_minutes=max(5.0, float(cfg.valid_minutes)),
            pullback_atr_multiplier=max(0.0, float(cfg.limit_atr_multiplier)),
            emit_pending_orders=bool(emit_pending_orders),
            order_styles=("reversion_limit",),
        ),
    )


def build_eth_moderate_flush_pullback_entry_fills(
    feature_rows: pd.DataFrame,
    *,
    config: EthModerateFlushPullbackConfig | None = None,
    emit_pending_orders: bool = False,
) -> pd.DataFrame:
    """Buy a controlled ETH liquidation pullback before the relief expansion."""

    cfg = config or EthModerateFlushPullbackConfig()
    if feature_rows.empty:
        return build_conditional_factor_fills(pd.DataFrame(), feature_rows)
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].copy()
    setup_mask = (
        _numeric_series(eth, "ret20").between(
            float(cfg.ret20_floor), float(cfg.ret20_ceiling),
        )
        & _numeric_series(eth, "ret60").between(
            float(cfg.ret60_floor), float(cfg.ret60_ceiling),
        )
        & _numeric_series(eth, "rsi14").between(
            float(cfg.rsi14_floor), float(cfg.rsi14_ceiling),
        )
        & _numeric_series(eth, "adx14").between(
            float(cfg.adx14_floor), float(cfg.adx14_ceiling),
        )
        & _numeric_series(eth, "quote_volume_ratio_med30").ge(
            float(cfg.volume_ratio_floor),
        )
        & _numeric_series(eth, "taker_buy_ratio").le(
            float(cfg.taker_buy_ceiling),
        )
    )
    setups = eth.loc[setup_mask, ["bar_time"]].rename(columns={"bar_time": "signal_time"})
    setups["signal_id"] = [f"WJETHMODFLUSH{index + 1:06d}" for index in range(len(setups))]
    setups["symbol"] = "ETHUSDT"
    setups["action"] = "OPEN_LONG"
    setups["score"] = 4.0
    setups["reason"] = "eth_moderate_flush_pullback_long;capitulation_absorption_long"
    return build_conditional_factor_fills(
        setups,
        features,
        config=ConditionalOrderConfig(
            valid_minutes=max(5.0, float(cfg.valid_minutes)),
            pullback_atr_multiplier=max(0.0, float(cfg.pullback_atr_multiplier)),
            emit_pending_orders=bool(emit_pending_orders),
            order_styles=("reversion_limit",),
        ),
    )


def build_eth_relief_recovery_exit_signals(
    selected_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    config: EthReliefRecoveryExitConfig | None = None,
) -> pd.DataFrame:
    """Delay a weak ETH relief exit until one closed-bar flow recovery."""

    cfg = config or EthReliefRecoveryExitConfig()
    columns = [
        "signal_id", "symbol", "action", "signal_time", "signal_price", "score", "reason",
        "decision_time", "setup_signal_id",
    ]
    signals = selected_close_signals.copy()
    if signals.empty or feature_rows.empty:
        return pd.DataFrame(columns=columns)
    signals["signal_time"] = pd.to_datetime(signals.get("signal_time"), errors="coerce")
    signals = signals[
        signals.get("symbol", pd.Series("", index=signals.index)).astype(str).eq("ETHUSDT")
        & signals.get("action", pd.Series("", index=signals.index)).astype(str).eq("CLOSE_LONG")
        & signals.get("reason", pd.Series("", index=signals.index)).map(_source_reason_family).eq(
            "long_profit_take_exhaustion",
        )
    ].copy()
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].drop_duplicates("bar_time", keep="last").sort_values("bar_time").set_index("bar_time")
    rows: list[dict[str, object]] = []
    for signal in signals.itertuples(index=False):
        setup_time = pd.Timestamp(signal.signal_time)
        if setup_time not in eth.index:
            continue
        setup = eth.loc[setup_time]
        qualified = (
            float(cfg.ret20_floor) <= _number(setup.get("ret20")) <= float(cfg.ret20_ceiling)
            and _number(setup.get("ret60")) <= float(cfg.ret60_ceiling)
            and _number(setup.get("ret288")) <= float(cfg.ret288_ceiling)
            and _number(setup.get("taker_buy_ratio")) <= float(cfg.setup_taker_buy_ceiling)
            and float(cfg.setup_volume_ratio_floor)
            <= _number(setup.get("quote_volume_ratio_med30"))
            <= float(cfg.setup_volume_ratio_ceiling)
            and float(cfg.setup_rsi14_floor) <= _number(setup.get("rsi14")) <= float(cfg.setup_rsi14_ceiling)
            and float(cfg.setup_adx14_floor) <= _number(setup.get("adx14")) <= float(cfg.setup_adx14_ceiling)
        )
        if not qualified:
            continue
        future = eth[eth.index > setup_time]
        if future.empty:
            continue
        confirmation_time = pd.Timestamp(future.index[0])
        confirmation = future.iloc[0]
        if (
            _number(confirmation.get("taker_buy_ratio")) < float(cfg.confirmation_taker_buy_floor)
            or _number(confirmation.get("close")) <= _number(setup.get("close"))
        ):
            continue
        rows.append(
            {
                "signal_id": f"WJETHRELIEFEXIT{len(rows) + 1:06d}",
                "symbol": "ETHUSDT",
                "action": "CLOSE_LONG",
                "signal_time": confirmation_time,
                "signal_price": _number(confirmation.get("close")),
                "score": _number(getattr(signal, "score", 0.0), 0.0),
                "reason": "eth_relief_flow_recovery_exit;" + str(getattr(signal, "reason", "")),
                "decision_time": confirmation_time,
                "setup_signal_id": str(getattr(signal, "signal_id", "")),
            },
        )
    return pd.DataFrame(rows, columns=columns)


def build_eth_blowoff_retrace_short_entry_fills(
    feature_rows: pd.DataFrame,
    *,
    config: EthBlowoffRetraceShortConfig | None = None,
    emit_pending_orders: bool = False,
) -> pd.DataFrame:
    """Fade a mature ETH blowoff only after a later tradable retrace."""

    cfg = config or EthBlowoffRetraceShortConfig()
    if feature_rows.empty:
        return build_conditional_factor_fills(pd.DataFrame(), feature_rows)
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].copy()
    setup_mask = (
        _numeric_series(eth, "ret20").between(
            float(cfg.ret20_floor), float(cfg.ret20_ceiling),
        )
        & _numeric_series(eth, "ret60").between(
            float(cfg.ret60_floor), float(cfg.ret60_ceiling),
        )
        & _numeric_series(eth, "ret288").ge(float(cfg.ret288_floor))
        & _numeric_series(eth, "rsi14").between(
            float(cfg.rsi14_floor), float(cfg.rsi14_ceiling),
        )
        & _numeric_series(eth, "adx14").between(
            float(cfg.adx14_floor), float(cfg.adx14_ceiling),
        )
        & _numeric_series(eth, "quote_volume_ratio_med30").ge(
            float(cfg.volume_ratio_floor),
        )
        & _numeric_series(eth, "taker_buy_ratio").ge(
            float(cfg.taker_buy_floor),
        )
        & _numeric_series(eth, "donchian20_pos").ge(
            float(cfg.donchian_floor),
        )
    )
    setups = eth.loc[setup_mask, ["bar_time"]].rename(columns={"bar_time": "signal_time"})
    setups["signal_id"] = [f"WJETHBLOWOFFSETUP{index + 1:06d}" for index in range(len(setups))]
    setups["symbol"] = "ETHUSDT"
    setups["action"] = "OPEN_SHORT"
    setups["score"] = 4.0
    setups["reason"] = "eth_blowoff_retrace_short;eth_anchor_exhaustion_short"
    return build_conditional_factor_fills(
        setups,
        features,
        config=ConditionalOrderConfig(
            valid_minutes=max(5.0, float(cfg.valid_minutes)),
            pullback_atr_multiplier=max(0.0, float(cfg.retrace_atr_multiplier)),
            emit_pending_orders=bool(emit_pending_orders),
            order_styles=("reversion_limit",),
        ),
    )


def build_eth_early_core_trend_fade_scaleout_signals(
    strategy_positions: pd.DataFrame,
    raw_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Close three early-core runners after a confirmed 15-minute profit-factor fade."""

    columns = ["signal_id", "symbol", "action", "signal_time", "signal_price", "score", "reason"]
    positions = _normalized_eth_long_positions(strategy_positions)
    raw = raw_close_signals.copy()
    if positions.empty or raw.empty or feature_rows.empty:
        return pd.DataFrame(columns=columns)
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    eth = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
    ].drop_duplicates("bar_time", keep="last").set_index("bar_time")
    positions["open_ret288"] = positions["open_time"].map(pd.to_numeric(eth.get("ret288"), errors="coerce"))
    positions = positions[
        positions.get("open_reason_family", pd.Series("", index=positions.index)).astype(str).eq("core_trend_long")
        & pd.to_numeric(positions["open_ret288"], errors="coerce").le(1.50)
    ]
    raw["signal_time"] = pd.to_datetime(raw.get("signal_time"), errors="coerce")
    raw = raw[
        raw.get("symbol", pd.Series("", index=raw.index)).astype(str).eq("ETHUSDT")
        & raw.get("action", pd.Series("", index=raw.index)).astype(str).eq("CLOSE_LONG")
        & raw.get("reason", pd.Series("", index=raw.index)).astype(str)
        .str.startswith("long_profit_take_exhaustion;")
    ].copy()
    raw["volume_ratio"] = raw.get("reason", pd.Series("", index=raw.index)).astype(str).str.extract(
        r"qvr=([0-9.]+)",
    )[0].astype(float)
    rows: list[dict[str, object]] = []
    for position in positions.itertuples(index=False):
        scoped = raw[
            raw["signal_time"].between(position.open_time, position.close_time, inclusive="both")
        ].sort_values("signal_time")
        peaks = scoped[
            pd.to_numeric(scoped.get("score"), errors="coerce").ge(4.0)
            & pd.to_numeric(scoped.get("volume_ratio"), errors="coerce").ge(3.0)
        ]
        if peaks.empty:
            continue
        confirmation_time = pd.Timestamp(peaks.iloc[0]["signal_time"]) + pd.Timedelta(minutes=15)
        confirmations = scoped[scoped["signal_time"].ge(confirmation_time)]
        if confirmations.empty:
            continue
        confirmation = confirmations.iloc[0]
        signal_time = pd.Timestamp(confirmation["signal_time"])
        signal_price = _number(eth.loc[signal_time].get("close")) if signal_time in eth.index else np.nan
        if not np.isfinite(signal_price):
            continue
        for _ in range(3):
            rows.append(
                {
                    "signal_id": f"WJETHCOREFADE{len(rows) + 1:06d}",
                    "symbol": "ETHUSDT",
                    "action": "CLOSE_LONG",
                    "signal_time": signal_time,
                    "signal_price": signal_price,
                    "score": _number(confirmation.get("score"), 3.0),
                    "reason": "eth_early_core_trend_peak_fade_scaleout",
                },
            )
    return pd.DataFrame(rows, columns=columns)


def build_active_eth_long_scaleout_signals(
    strategy_positions: pd.DataFrame,
    raw_close_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    config: ActiveEthLongScaleoutConfig | None = None,
) -> pd.DataFrame:
    """Build a bounded ETH long runner-exit ladder from closed market state."""

    cfg = config or ActiveEthLongScaleoutConfig()
    columns = ["signal_id", "symbol", "action", "signal_time", "signal_price", "score", "reason"]
    positions = _normalized_eth_long_positions(strategy_positions)
    if positions.empty or feature_rows.empty:
        return pd.DataFrame(columns=columns)
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    features = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].sort_values("bar_time")
    features["prev_ret20"] = pd.to_numeric(features.get("ret20"), errors="coerce").shift(1)
    features["prev_qvr3"] = (
        pd.to_numeric(features.get("quote_volume_ratio_med30"), errors="coerce")
        .shift(1).rolling(3, min_periods=1).max()
    )
    raw = raw_close_signals.copy()
    raw["signal_time"] = pd.to_datetime(raw.get("signal_time"), errors="coerce")
    raw = raw[
        raw.get("symbol", pd.Series("", index=raw.index)).astype(str).eq("ETHUSDT")
        & raw.get("action", pd.Series("", index=raw.index)).astype(str).eq("CLOSE_LONG")
        & raw["signal_time"].notna()
    ].copy()
    raw["reason_family"] = raw.get("reason", pd.Series("", index=raw.index)).astype(str).str.split(";").str[0]
    feature_lookup = features.drop_duplicates("bar_time", keep="last").set_index("bar_time")
    raw["volume_ratio"] = raw["signal_time"].map(
        pd.to_numeric(feature_lookup.get("quote_volume_ratio_med30"), errors="coerce"),
    )
    market_close = raw["signal_time"].map(pd.to_numeric(feature_lookup.get("close"), errors="coerce"))
    supplied_price = (
        pd.to_numeric(raw["signal_price"], errors="coerce")
        if "signal_price" in raw
        else pd.Series(np.nan, index=raw.index, dtype=float)
    )
    raw["signal_price"] = supplied_price.fillna(market_close)

    rows: list[dict[str, object]] = []

    def emit(signal_time: pd.Timestamp, signal_price: float, score: float, reason: str, count: int = 1) -> None:
        for _ in range(max(1, int(count))):
            rows.append(
                {
                    "signal_id": f"WJETHRUNNER{len(rows) + 1:06d}",
                    "symbol": "ETHUSDT",
                    "action": "CLOSE_LONG",
                    "signal_time": pd.Timestamp(signal_time),
                    "signal_price": float(signal_price),
                    "score": float(score),
                    "reason": reason,
                },
            )

    for window_start, window_end in _active_state_windows(positions):
        raw_window = raw[
            raw["signal_time"].between(window_start, window_end, inclusive="both")
            & raw["reason_family"].eq("long_profit_take_exhaustion")
            & pd.to_numeric(raw.get("score"), errors="coerce").ge(float(cfg.profit_score_floor))
            & pd.to_numeric(raw["volume_ratio"], errors="coerce").ge(float(cfg.profit_volume_ratio_floor))
        ].sort_values("signal_time")
        if not raw_window.empty:
            first = raw_window.iloc[0]
            signal_price = _number(first.get("signal_price"))
            if np.isfinite(signal_price):
                emit(first["signal_time"], signal_price, _number(first.get("score"), 3.0), "eth_runner_profit_volume_onset")

        used: set[str] = set()
        scoped = features[
            features["bar_time"].between(window_start, window_end, inclusive="both")
        ]
        for row in scoped.itertuples(index=False):
            signal_time = pd.Timestamp(row.bar_time)
            close = _number(getattr(row, "close", np.nan))
            if not np.isfinite(close):
                continue
            if (
                "spike_reversal" not in used
                and _number(getattr(row, "prev_qvr3", np.nan)) >= float(cfg.spike_volume_ratio_floor)
                and _number(getattr(row, "taker_buy_ratio", np.nan)) <= 0.40
                and _number(getattr(row, "ret20", np.nan)) >= 0.50
            ):
                emit(signal_time, close, 3.0, "eth_runner_spike_reversal", count=2)
                used.add("spike_reversal")
            age_minutes = (signal_time - window_start).total_seconds() / 60.0
            if (
                "midnight" not in used
                and age_minutes >= float(cfg.midnight_min_active_age_minutes)
                and signal_time.hour == 0
                and signal_time.minute == 0
            ):
                emit(signal_time, close, 3.0, "eth_runner_midnight")
                used.add("midnight")
            if (
                "acceleration" not in used
                and _number(getattr(row, "ret20", np.nan)) >= 1.50
                and _number(getattr(row, "prev_ret20", np.nan)) < 1.50
                and _number(getattr(row, "taker_buy_ratio", np.nan)) >= 0.50
            ):
                emit(signal_time, close, 3.0, "eth_runner_acceleration_onset")
                used.add("acceleration")
            if (
                "blowoff" not in used
                and _number(getattr(row, "ret60", np.nan)) >= 3.0
                and _number(getattr(row, "rsi14", np.nan)) >= 70.0
                and _number(getattr(row, "adx14", np.nan)) >= 40.0
            ):
                emit(signal_time, close, 3.0, "eth_runner_blowoff")
                used.add("blowoff")
    return pd.DataFrame(rows, columns=columns).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def build_active_eth_long_addon_fills(
    strategy_positions: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    config: ActiveEthLongAddonConfig | None = None,
    emit_pending_orders: bool = False,
) -> pd.DataFrame:
    """Build sparse pullback add-ons from the autonomous ETH long state."""

    cfg = config or ActiveEthLongAddonConfig()
    positions = strategy_positions.copy()
    if positions.empty or feature_rows.empty:
        return build_conditional_factor_fills(pd.DataFrame(), feature_rows)
    positions["open_time"] = pd.to_datetime(positions.get("open_time"), errors="coerce")
    positions["close_time"] = pd.to_datetime(positions.get("close_time"), errors="coerce")
    positions = positions[
        positions.get("symbol", pd.Series("", index=positions.index)).astype(str).eq("ETHUSDT")
        & positions.get("direction", pd.Series("", index=positions.index)).astype(str).str.upper().eq("LONG")
        & positions["open_time"].notna()
        & positions["close_time"].notna()
    ]
    if positions.empty:
        return build_conditional_factor_fills(pd.DataFrame(), feature_rows)

    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    features = features[
        features.get("symbol", pd.Series("", index=features.index)).astype(str).eq("ETHUSDT")
        & features["bar_time"].notna()
    ].sort_values("bar_time")
    setup_rows: list[dict[str, object]] = []
    breakout_setup_rows: list[dict[str, object]] = []
    min_age = pd.Timedelta(minutes=max(0.0, float(cfg.min_active_age_minutes)))
    cooldown = max(0.0, float(cfg.setup_cooldown_minutes))
    for window_start, window_end in _active_state_windows(positions):
        window_positions = positions[
            positions["open_time"].between(window_start, window_end, inclusive="both")
        ].sort_values("open_time")
        primary_family = (
            str(window_positions.iloc[0].get("open_reason_family") or "")
            if not window_positions.empty
            else ""
        )
        if primary_family == "core_trend_long":
            core_pullbacks = features[
                features["bar_time"].between(
                    window_start + pd.Timedelta(minutes=90), window_end, inclusive="both",
                )
                & _numeric_series(features, "ret60").ge(2.0)
                & _numeric_series(features, "rsi7").le(25.0)
                & _numeric_series(features, "bb20_zscore").le(-2.0)
                & _numeric_series(features, "donchian20_pos").le(0.10)
            ]
            if not core_pullbacks.empty:
                core_pullback = core_pullbacks.iloc[0]
                setup_rows.append(
                    {
                        "signal_id": f"WJETHCOREPULLBACK{len(setup_rows) + 1:06d}",
                        "symbol": "ETHUSDT",
                        "action": "OPEN_LONG",
                        "signal_time": core_pullback["bar_time"],
                        "score": 3.0,
                        "reason": "eth_core_trend_deep_pullback_add_setup",
                    },
                )
        breakout_start = window_start + pd.Timedelta(
            minutes=max(0.0, float(cfg.breakout_min_active_age_minutes)),
        )
        breakout_end = min(window_end, window_start + min_age)
        breakout_rows = features[
            features["bar_time"].between(breakout_start, breakout_end, inclusive="both")
            & _numeric_series(features, "ret12").ge(1.50)
            & _numeric_series(features, "donchian20_pos").ge(0.80)
            & _numeric_series(features, "quote_volume_ratio_med30").ge(3.0)
            & _numeric_series(features, "taker_buy_ratio").ge(0.50)
        ]
        if not breakout_rows.empty:
            breakout_rows = breakout_rows[
                breakout_rows.apply(
                    lambda row: _market_regime_allows_action(row, "OPEN_LONG"),
                    axis=1,
                )
            ]
        if not breakout_rows.empty:
            breakout = breakout_rows.iloc[0]
            breakout_setup_rows.append(
                {
                    "signal_id": f"WJETHBREAKOUTSETUP{len(breakout_setup_rows) + 1:06d}",
                    "symbol": "ETHUSDT",
                    "action": "OPEN_LONG",
                    "signal_time": breakout["bar_time"],
                    "score": 3.0,
                    "reason": "eth_active_long_breakout_add_setup",
                },
            )
        scoped = features[
            features["bar_time"].between(window_start + min_age, window_end, inclusive="both")
        ]
        last_setup: pd.Timestamp | None = None
        for row in scoped.itertuples(index=False):
            signal_time = pd.Timestamp(row.bar_time)
            pullback = (
                _number(getattr(row, "rsi14", np.nan)) <= 47.0
                and _number(getattr(row, "ret3", np.nan)) <= 0.0
                and _number(getattr(row, "from_high12", np.nan)) <= -0.50
                and _number(getattr(row, "taker_buy_ratio", np.nan)) >= 0.50
            )
            deep_pullback = (
                _number(getattr(row, "rsi14", np.nan)) <= 38.0
                and _number(getattr(row, "ret20", np.nan)) <= -1.50
                and _number(getattr(row, "bb20_zscore", np.nan)) <= -2.0
            )
            breakout = (
                _number(getattr(row, "ret12", np.nan)) >= 1.50
                and _number(getattr(row, "donchian20_pos", np.nan)) >= 0.80
                and _number(getattr(row, "quote_volume_ratio_med30", np.nan)) >= 3.0
                and _number(getattr(row, "taker_buy_ratio", np.nan)) >= 0.50
            )
            if breakout:
                breakout = _market_regime_allows_action(
                    pd.Series(row._asdict()),
                    "OPEN_LONG",
                )
            if not (pullback or deep_pullback or breakout):
                continue
            gap = (
                (signal_time - last_setup).total_seconds() / 60.0
                if last_setup is not None
                else np.inf
            )
            if gap < cooldown and not deep_pullback:
                continue
            setup_rows.append(
                {
                    "signal_id": f"WJETHADDSETUP{len(setup_rows) + 1:06d}",
                    "symbol": "ETHUSDT",
                    "action": "OPEN_LONG",
                    "signal_time": signal_time,
                    "score": 3.0,
                    "reason": "eth_active_long_add_setup",
                },
            )
            last_setup = signal_time
    setups = pd.DataFrame(setup_rows)
    pullback_fills = build_conditional_factor_fills(
        setups,
        features,
        config=ConditionalOrderConfig(
            valid_minutes=max(5.0, float(cfg.valid_minutes)),
            pullback_atr_multiplier=max(0.0, float(cfg.pullback_atr_multiplier)),
            emit_pending_orders=bool(emit_pending_orders),
            order_styles=("reversion_limit",),
        ),
    )
    breakout_fills = build_conditional_factor_fills(
        pd.DataFrame(breakout_setup_rows),
        features,
        config=ConditionalOrderConfig(
            valid_minutes=max(5.0, float(cfg.valid_minutes)),
            pullback_atr_multiplier=max(0.0, float(cfg.breakout_pullback_atr_multiplier)),
            emit_pending_orders=bool(emit_pending_orders),
            order_styles=("reversion_limit",),
        ),
    )
    fill_frames = [frame for frame in (pullback_fills, breakout_fills) if not frame.empty]
    if not fill_frames:
        return pullback_fills
    out = pd.concat(fill_frames, ignore_index=True, sort=False)
    out = out.sort_values(["signal_time", "signal_id"]).drop_duplicates(
        ["symbol", "action", "signal_time", "signal_price", "reason"],
        keep="first",
    ).reset_index(drop=True)
    out["signal_id"] = [f"WJETHADDFILL{index + 1:06d}" for index in range(len(out))]
    return out


def _active_state_windows(positions: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    events: list[tuple[pd.Timestamp, int]] = []
    for row in positions.itertuples(index=False):
        events.extend(((pd.Timestamp(row.open_time), 1), (pd.Timestamp(row.close_time), -1)))
    active = 0
    start: pd.Timestamp | None = None
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for event_time, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        previous = active
        active += delta
        if previous == 0 and active > 0:
            start = event_time
        elif previous > 0 and active == 0 and start is not None:
            windows.append((start, event_time))
            start = None
    return windows


def _normalized_eth_long_positions(strategy_positions: pd.DataFrame) -> pd.DataFrame:
    positions = strategy_positions.copy()
    if positions.empty:
        return positions
    positions["open_time"] = pd.to_datetime(positions.get("open_time"), errors="coerce")
    positions["close_time"] = pd.to_datetime(positions.get("close_time"), errors="coerce")
    return positions[
        positions.get("symbol", pd.Series("", index=positions.index)).astype(str).eq("ETHUSDT")
        & positions.get("direction", pd.Series("", index=positions.index)).astype(str).str.upper().eq("LONG")
        & positions["open_time"].notna()
        & positions["close_time"].notna()
    ].copy()


def build_conditional_factor_fills(
    setup_signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    config: ConditionalOrderConfig | None = None,
) -> pd.DataFrame:
    """Convert closed-bar setups into later, range-confirmed conditional fills."""

    cfg = config or ConditionalOrderConfig()
    columns = [
        "signal_id", "symbol", "action", "signal_time", "signal_price", "score", "reason",
        "decision_time", "order_style", "setup_signal_id", "pending_order", "valid_until",
    ]
    if setup_signals.empty or feature_rows.empty:
        return pd.DataFrame(columns=columns)
    invalid_styles = set(cfg.order_styles).difference({"momentum_stop", "reversion_limit"})
    if invalid_styles:
        raise ValueError("unsupported conditional order styles: " + ",".join(sorted(invalid_styles)))

    bars_by_symbol: dict[str, dict[str, object]] = {}
    bars = feature_rows.copy()
    bars["bar_time"] = pd.to_datetime(bars.get("bar_time"), errors="coerce")
    bars = bars.dropna(subset=["bar_time", "symbol"]).sort_values(["symbol", "bar_time"])
    for symbol, frame in bars.groupby("symbol", sort=False):
        indexed = frame.drop_duplicates("bar_time", keep="last").set_index("bar_time").sort_index()
        missing_values = pd.Series(np.nan, index=indexed.index, dtype=float)
        bars_by_symbol[str(symbol)] = {
            "times": pd.DatetimeIndex(indexed.index),
            "open": pd.to_numeric(indexed.get("open", missing_values), errors="coerce").to_numpy(dtype=float),
            "high": pd.to_numeric(indexed.get("high", missing_values), errors="coerce").to_numpy(dtype=float),
            "low": pd.to_numeric(indexed.get("low", missing_values), errors="coerce").to_numpy(dtype=float),
            "close": pd.to_numeric(indexed.get("close", missing_values), errors="coerce").to_numpy(dtype=float),
            "atr14_pct": pd.to_numeric(indexed.get("atr14_pct", missing_values), errors="coerce").to_numpy(dtype=float),
            "atr_pct": pd.to_numeric(indexed.get("atr_pct", missing_values), errors="coerce").to_numpy(dtype=float),
        }

    rows: list[dict[str, object]] = []
    setups = setup_signals.copy()
    setups["signal_time"] = pd.to_datetime(setups.get("signal_time"), errors="coerce")
    setups = setups.dropna(subset=["signal_time"]).sort_values(["signal_time", "symbol", "action"])
    valid_delta = pd.Timedelta(minutes=max(5.0, float(cfg.valid_minutes)))
    for setup in setups.itertuples(index=False):
        symbol = str(getattr(setup, "symbol", ""))
        action = str(getattr(setup, "action", ""))
        decision_time = pd.Timestamp(setup.signal_time)
        market = bars_by_symbol.get(symbol)
        if market is None or action not in {"OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT"}:
            continue
        times = market["times"]
        setup_index = int(times.searchsorted(decision_time, side="left"))
        if setup_index >= len(times) or times[setup_index] != decision_time:
            continue
        future_start = int(times.searchsorted(decision_time, side="right"))
        future_end = int(times.searchsorted(decision_time + valid_delta, side="right"))
        if future_start >= future_end and not bool(cfg.emit_pending_orders):
            continue
        setup_bar = {column: values[setup_index] for column, values in market.items() if column != "times"}
        is_buy = action in {"OPEN_LONG", "CLOSE_SHORT"}
        for style in cfg.order_styles:
            target = _conditional_target(setup_bar, is_buy=is_buy, style=style, config=cfg)
            if not np.isfinite(target) or target <= 0.0:
                continue
            fill = _first_fill_arrays(
                market,
                start=future_start,
                end=future_end,
                target=target,
                is_buy=is_buy,
                style=style,
            )
            if fill is None:
                last_market_time = pd.Timestamp(times[-1]) if len(times) else pd.NaT
                valid_until = decision_time + valid_delta
                if (
                    bool(cfg.emit_pending_orders)
                    and pd.notna(last_market_time)
                    and decision_time <= last_market_time <= valid_until
                ):
                    setup_id = str(getattr(setup, "signal_id", ""))
                    rows.append(
                        {
                            "symbol": symbol,
                            "action": action,
                            "signal_time": last_market_time,
                            "signal_price": target,
                            "score": _number(getattr(setup, "score", 0.0), 0.0),
                            "reason": (
                                f"pending_conditional_{style};decision={decision_time};"
                                f"target={target:.12g};valid_until={valid_until};"
                                + str(getattr(setup, "reason", ""))
                            ),
                            "decision_time": decision_time,
                            "order_style": style,
                            "setup_signal_id": setup_id,
                            "pending_order": True,
                            "valid_until": valid_until,
                        },
                    )
                continue
            fill_time, fill_price = fill
            setup_id = str(getattr(setup, "signal_id", ""))
            rows.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "signal_time": fill_time,
                    "signal_price": fill_price,
                    "score": _number(getattr(setup, "score", 0.0), 0.0),
                    "reason": (
                        f"conditional_{style};decision={decision_time};target={target:.12g};"
                        + str(getattr(setup, "reason", ""))
                    ),
                    "decision_time": decision_time,
                    "order_style": style,
                    "setup_signal_id": setup_id,
                    "pending_order": False,
                    "valid_until": decision_time + valid_delta,
                },
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows).sort_values(
        ["signal_time", "symbol", "action", "order_style", "setup_signal_id"],
    )
    out = out.drop_duplicates(
        ["symbol", "action", "signal_time", "signal_price", "order_style"],
        keep="first",
    ).reset_index(drop=True)
    out["signal_id"] = [f"WJCONDFILL{index + 1:06d}" for index in range(len(out))]
    return out.reindex(columns=columns)


def _conditional_target(
    bar: pd.Series,
    *,
    is_buy: bool,
    style: str,
    config: ConditionalOrderConfig,
) -> float:
    close = _number(bar.get("close"))
    high = _number(bar.get("high"))
    low = _number(bar.get("low"))
    if style == "momentum_stop":
        if config.momentum_atr_multiplier is not None:
            atr_pct = _number(bar.get("atr14_pct"), _number(bar.get("atr_pct"), 0.0))
            distance = max(0.0, atr_pct) * max(0.0, float(config.momentum_atr_multiplier)) / 100.0
            return close * (1.0 + distance) if is_buy else close * (1.0 - distance)
        buffer = max(0.0, float(config.stop_buffer_bps)) / 10000.0
        return high * (1.0 + buffer) if is_buy else low * (1.0 - buffer)
    atr_pct = _number(bar.get("atr14_pct"), _number(bar.get("atr_pct"), 0.0))
    distance = max(0.0, atr_pct) * max(0.0, float(config.pullback_atr_multiplier)) / 100.0
    return close * (1.0 - distance) if is_buy else close * (1.0 + distance)


def _first_fill(
    bars: pd.DataFrame,
    *,
    target: float,
    is_buy: bool,
    style: str,
) -> tuple[pd.Timestamp, float] | None:
    if bars.empty:
        return None
    high = pd.to_numeric(bars.get("high"), errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars.get("low"), errors="coerce").to_numpy(dtype=float)
    if style == "momentum_stop":
        crossed = high >= target if is_buy else low <= target
    else:
        crossed = low <= target if is_buy else high >= target
    hit_indices = np.flatnonzero(crossed)
    if not len(hit_indices):
        return None
    hit_index = int(hit_indices[0])
    open_values = pd.to_numeric(bars.get("open"), errors="coerce").to_numpy(dtype=float)
    open_price = open_values[hit_index]
    if not np.isfinite(open_price):
        open_price = target
    if style == "momentum_stop":
        fill_price = max(target, open_price) if is_buy else min(target, open_price)
    else:
        fill_price = min(target, open_price) if is_buy else max(target, open_price)
    fill_time = pd.to_datetime(bars.iloc[hit_index].get("bar_time"), errors="coerce")
    if pd.isna(fill_time):
        return None
    return pd.Timestamp(fill_time), float(fill_price)


def _first_fill_arrays(
    market: dict[str, object],
    *,
    start: int,
    end: int,
    target: float,
    is_buy: bool,
    style: str,
) -> tuple[pd.Timestamp, float] | None:
    high = market["high"][start:end]
    low = market["low"][start:end]
    if style == "momentum_stop":
        crossed = high >= target if is_buy else low <= target
    else:
        crossed = low <= target if is_buy else high >= target
    hit_indices = np.flatnonzero(crossed)
    if not len(hit_indices):
        return None
    hit_index = start + int(hit_indices[0])
    open_price = float(market["open"][hit_index])
    if not np.isfinite(open_price):
        open_price = target
    if style == "momentum_stop":
        fill_price = max(target, open_price) if is_buy else min(target, open_price)
    else:
        fill_price = min(target, open_price) if is_buy else max(target, open_price)
    return pd.Timestamp(market["times"][hit_index]), float(fill_price)


def _number(value: object, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)
