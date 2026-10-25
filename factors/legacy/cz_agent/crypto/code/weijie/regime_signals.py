"""Autonomous Weijie signals expressed as market-regime transitions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from factors.crypto.code.weijie.common import finite_float, weijie_market_prior_group
from factors.crypto.code.weijie.market_regime import (
    MarketRegimeDecision,
    MarketRegimeState,
    decide_market_regime,
    observable_session_state,
    score_multi_timeframe_state,
)
from factors.crypto.code.weijie.signal_templates import select_signals

_ENTRY_CONFIDENCE_BY_ASSET_GROUP = {
    "large_cap_momentum": 0.36,
    "meme_momentum": 0.42,
    "liquid_alt_momentum": 0.39,
    "tradfi_equity": 0.34,
    "precious_metal_defensive": 0.34,
}

_REGIME_SIGNAL_SNAPSHOT_COLUMNS = (
    "reason_family",
    "ret20",
    "ret60",
    "ret144",
    "ret288",
    "tf15_ret4",
    "don",
    "taker",
    "seller",
    "qvr",
    "rsi",
    "adx",
    "atr_pct",
    "down",
    "up",
)


def build_regime_open_factor_signals(feature_rows: pd.DataFrame, *, config: Any) -> pd.DataFrame:
    """Generate at most one causal regime action for each symbol and bar."""

    rows: list[dict[str, Any]] = []
    prepared = feature_rows.copy()
    prepared["bar_time"] = pd.to_datetime(prepared.get("bar_time"), errors="coerce")
    decisions = [decide_market_regime(row) for _, row in prepared.iterrows()]
    prepared["_regime_decision"] = decisions
    prepared["_directional_momentum"] = [
        decision.momentum.short_momentum
        + 0.75 * decision.momentum.mid_momentum
        + 0.50 * decision.momentum.long_momentum
        for decision in decisions
    ]
    grouped = prepared.groupby("bar_time", sort=False, dropna=False)["_directional_momentum"]
    prepared["_cross_section_count"] = grouped.transform("count")
    prepared["_relative_strength_rank"] = grouped.rank(method="average", pct=True)
    for _, row in prepared.iterrows():
        symbol = str(row.get("symbol") or "")
        signal_time = pd.to_datetime(row.get("bar_time"), errors="coerce")
        if not symbol or pd.isna(signal_time):
            continue
        decision = row["_regime_decision"]
        if decision.action not in {"OPEN_LONG", "OPEN_SHORT"}:
            continue
        asset_group = weijie_market_prior_group(symbol)
        confidence_floor = _ENTRY_CONFIDENCE_BY_ASSET_GROUP.get(asset_group, 0.39)
        score = decision.score
        action_direction = 1.0 if decision.action == "OPEN_LONG" else -1.0
        confirmed_short_momentum = decision.momentum.short_momentum * action_direction
        relative_volume = finite_float(row.get("quote_volume_ratio_med30"), 0.0)
        adx = finite_float(row.get("adx14"), 0.0)
        recovery = bool(
            decision.regime is MarketRegimeState.RECOVERY
            and confirmed_short_momentum >= 0.0
            and score.trend_strength >= 0.30
            and score.liquidity_condition >= 0.40
            and score.reversal_probability <= 0.35
            and 0.90 <= relative_volume <= 6.0
            and adx >= 20.0,
        )
        trend_rollover = bool(
            decision.regime is MarketRegimeState.EXHAUSTION
            and decision.mechanism == "trend_rollover",
        )
        exhaustion_reversal = bool(
            decision.regime is MarketRegimeState.EXHAUSTION
            and (
                trend_rollover
                or (
                    score.crowding_score >= 0.60
                    and score.reversal_probability >= 0.62
                    and score.liquidity_condition >= 0.28
                )
            ),
        )
        if not (recovery or exhaustion_reversal):
            continue
        entry_confidence = decision.confidence
        if entry_confidence < confidence_floor:
            continue
        rows.append(
            _signal_row(
                symbol=symbol,
                action=decision.action,
                signal_time=pd.Timestamp(signal_time),
                score=2.0 + 3.0 * entry_confidence + score.liquidity_condition,
                reason_family=(
                    f"regime_{decision.mechanism}"
                    if decision.mechanism
                    else f"regime_{decision.regime.value}"
                ),
                row=row,
                decision=decision,
                asset_group=asset_group,
            ),
        )
    signals = select_signals(
        pd.DataFrame(rows),
        cooldown_minutes=float(getattr(config, "open_cooldown_minutes", 0.0)),
        preserve_columns=_REGIME_SIGNAL_SNAPSHOT_COLUMNS,
    )
    signals["signal_id"] = [f"WJREGOPEN{i + 1:05d}" for i in range(len(signals))]
    return signals


def build_regime_close_factor_signals(feature_rows: pd.DataFrame, *, config: Any) -> pd.DataFrame:
    """Generate close actions from exhaustion, liquidation, and recovery."""

    rows: list[dict[str, Any]] = []
    for _, row in feature_rows.iterrows():
        symbol = str(row.get("symbol") or "")
        signal_time = pd.to_datetime(row.get("bar_time"), errors="coerce")
        if not symbol or pd.isna(signal_time):
            continue
        decision = decide_market_regime(row)
        momentum = decision.momentum
        directional_momentum = (
            momentum.short_momentum
            + 0.75 * momentum.mid_momentum
            + 0.50 * momentum.long_momentum
        )
        close_action = ""
        reason_family = ""
        if decision.regime is MarketRegimeState.EXHAUSTION:
            close_action = "CLOSE_LONG" if directional_momentum >= 0.0 else "CLOSE_SHORT"
            reason_family = "regime_exhaustion_exit"
        elif decision.regime is MarketRegimeState.LIQUIDATION:
            close_action = "CLOSE_LONG" if directional_momentum < 0.0 else "CLOSE_SHORT"
            reason_family = "regime_liquidation_exit"
        elif decision.regime is MarketRegimeState.RECOVERY and decision.confidence >= 0.34:
            close_action = "CLOSE_SHORT" if decision.action == "OPEN_LONG" else "CLOSE_LONG"
            reason_family = "regime_recovery_exit"
        elif (
            decision.regime is MarketRegimeState.TRENDING
            and decision.score.reversal_probability >= 0.44
        ):
            close_action = "CLOSE_LONG" if directional_momentum >= 0.0 else "CLOSE_SHORT"
            reason_family = "regime_trend_rejection_exit"
        if not close_action:
            continue
        asset_group = weijie_market_prior_group(symbol)
        rows.append(
            _signal_row(
                symbol=symbol,
                action=close_action,
                signal_time=pd.Timestamp(signal_time),
                score=(
                    2.2
                    + 2.5 * decision.score.reversal_probability
                    + 1.2 * abs(momentum.short_momentum)
                ),
                reason_family=reason_family,
                row=row,
                decision=decision,
                asset_group=asset_group,
            ),
        )
    signals = select_signals(
        pd.DataFrame(rows),
        cooldown_minutes=float(getattr(config, "close_cooldown_minutes", 0.0)),
        preserve_columns=_REGIME_SIGNAL_SNAPSHOT_COLUMNS,
    )
    signals["signal_id"] = [f"WJREGCLOSE{i + 1:05d}" for i in range(len(signals))]
    return signals


def build_regime_rollover_flip_close_signals(
    rollover_opens: pd.DataFrame,
) -> pd.DataFrame:
    """Close opposite inventory before a qualified regime rollover entry."""

    if rollover_opens.empty:
        return pd.DataFrame(columns=rollover_opens.columns)
    reason = rollover_opens.get(
        "reason",
        pd.Series("", index=rollover_opens.index),
    ).astype(str)
    out = rollover_opens[reason.str.startswith("regime_trend_rollover;")].copy()
    if out.empty:
        return out
    out["action"] = out["action"].map(
        {"OPEN_SHORT": "CLOSE_LONG", "OPEN_LONG": "CLOSE_SHORT"},
    )
    out = out[out["action"].notna()].copy()
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce").fillna(0.0) + 0.10
    out["reason"] = out["reason"].str.replace(
        "regime_trend_rollover;",
        "regime_trend_rollover_flip_exit;",
        n=1,
        regex=False,
    )
    out["signal_id"] = [f"WJREGFLIP{i + 1:05d}" for i in range(len(out))]
    return out.reset_index(drop=True)


def select_synchronized_regime_rollover_signals(
    signals: pd.DataFrame,
    *,
    bucket_minutes: int,
    min_unique_symbols: int = 2,
) -> pd.DataFrame:
    """Keep class-level rollover events confirmed across multiple symbols.

    This is a cross-sectional breadth requirement.  It does not inspect an
    individual symbol, date, or wall-clock session, and it is symmetric for
    long and short rollover directions.
    """

    if signals.empty:
        return signals.copy()
    out = signals.copy()
    reason = out.get("reason", pd.Series("", index=out.index)).astype(str)
    out = out[reason.str.startswith("regime_trend_rollover;")].copy()
    if out.empty:
        return out
    out["_event_bucket"] = pd.to_datetime(
        out.get("signal_time"),
        errors="coerce",
    ).dt.floor(f"{max(1, int(bucket_minutes))}min")
    out["_market_prior_group"] = out.get(
        "symbol",
        pd.Series("", index=out.index),
    ).map(weijie_market_prior_group)
    unique_symbols = out.groupby(
        ["_market_prior_group", "_event_bucket", "action"],
        dropna=False,
    )["symbol"].transform("nunique")
    return out.loc[
        unique_symbols.ge(max(2, int(min_unique_symbols))),
    ].drop(columns=["_event_bucket", "_market_prior_group"], errors="ignore").reset_index(drop=True)


def _signal_row(
    *,
    symbol: str,
    action: str,
    signal_time: pd.Timestamp,
    score: float,
    reason_family: str,
    row: pd.Series,
    decision: MarketRegimeDecision,
    asset_group: str,
) -> dict[str, Any]:
    session = observable_session_state(row)
    multi_timeframe = score_multi_timeframe_state(row, action=action)
    metrics = (
        f"regime={decision.regime.value};asset_group={asset_group};"
        f"mechanism={decision.mechanism or decision.regime.value};"
        f"short_momentum={decision.momentum.short_momentum:.4f};"
        f"mid_momentum={decision.momentum.mid_momentum:.4f};"
        f"long_momentum={decision.momentum.long_momentum:.4f};"
        f"trend_strength={decision.score.trend_strength:.4f};"
        f"liquidity_condition={decision.score.liquidity_condition:.4f};"
        f"crowding_score={decision.score.crowding_score:.4f};"
        f"reversal_probability={decision.score.reversal_probability:.4f};"
        f"market_regime_score={decision.score.market_regime_score:.4f};"
        f"session_volatility_state={session.session_volatility_state:.4f};"
        f"session_liquidity_state={session.session_liquidity_state:.4f};"
        f"relative_strength_rank={finite_float(row.get('_relative_strength_rank'), 0.5):.4f};"
        f"tf15_trend={multi_timeframe.tf15_trend:.4f};"
        f"tf15_ret4={finite_float(row.get('tf15_ret4')):.4f};"
        f"tf1h_trend={multi_timeframe.tf1h_trend:.4f};"
        f"multi_timeframe_alignment={multi_timeframe.alignment_score:.4f};"
        f"ret20={finite_float(row.get('ret20')):.4f};"
        f"ret60={finite_float(row.get('ret60')):.4f};"
        f"ret144={finite_float(row.get('ret144')):.4f};"
        f"ret288={finite_float(row.get('ret288')):.4f};"
        f"don={finite_float(row.get('donchian60_pos'), 0.5):.4f};"
        f"taker={finite_float(row.get('taker_buy_ratio'), 0.5):.4f};"
        f"seller={finite_float(row.get('seller_pressure'), 0.5):.4f};"
        f"qvr={finite_float(row.get('quote_volume_ratio_med30'), 1.0):.4f};"
        f"rsi={finite_float(row.get('rsi14'), 50.0):.4f};"
        f"adx={finite_float(row.get('adx14'), 20.0):.4f};"
        f"atr_pct={finite_float(row.get('atr14_pct')):.4f};"
        f"down={finite_float(row.get('market_down_breadth20')):.4f};"
        f"up={finite_float(row.get('market_up_breadth20')):.4f}"
    )
    return {
        "signal_id": "",
        "symbol": symbol,
        "action": action,
        "signal_time": signal_time,
        "score": float(score),
        "reason": f"{reason_family};{metrics}",
        "reason_family": reason_family,
        # 说明：以下逻辑仅使用决策时点可获得的数据。
        # 说明：以下逻辑仅使用决策时点可获得的数据。
        "ret20": finite_float(row.get("ret20")),
        "ret60": finite_float(row.get("ret60")),
        "ret144": finite_float(row.get("ret144")),
        "ret288": finite_float(row.get("ret288")),
        "tf15_ret4": finite_float(row.get("tf15_ret4")),
        "don": finite_float(row.get("donchian60_pos"), 0.5),
        "taker": finite_float(row.get("taker_buy_ratio"), 0.5),
        "seller": finite_float(row.get("seller_pressure"), 0.5),
        "qvr": finite_float(row.get("quote_volume_ratio_med30"), 1.0),
        "rsi": finite_float(row.get("rsi14"), 50.0),
        "adx": finite_float(row.get("adx14"), 20.0),
        "atr_pct": finite_float(row.get("atr14_pct")),
        "down": finite_float(row.get("market_down_breadth20"), 0.5),
        "up": finite_float(row.get("market_up_breadth20"), 0.5),
    }


__all__ = [
    "build_regime_close_factor_signals",
    "build_regime_open_factor_signals",
    "build_regime_rollover_flip_close_signals",
    "select_synchronized_regime_rollover_signals",
]
