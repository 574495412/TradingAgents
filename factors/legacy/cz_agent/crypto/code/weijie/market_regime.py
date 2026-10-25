"""Symbol- and clock-agnostic market regime scoring for Weijie entries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketRegimeScore:
    """Interpretable causal regime components normalized to ``[0, 1]``."""

    trend_strength: float
    liquidity_condition: float
    crowding_score: float
    reversal_probability: float
    market_regime_score: float


class MarketRegimeState(str, Enum):
    """Reusable market states emitted before any strategy action is chosen."""

    TRENDING = "trending"
    EXHAUSTION = "exhaustion"
    LIQUIDATION = "liquidation"
    ACCUMULATION = "accumulation"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class MomentumState:
    """Compressed directional momentum replacing four raw return decisions."""

    short_momentum: float
    mid_momentum: float
    long_momentum: float


@dataclass(frozen=True)
class MarketRegimeDecision:
    """One symbol-agnostic regime-to-action decision."""

    regime: MarketRegimeState
    action: str
    confidence: float
    momentum: MomentumState
    score: MarketRegimeScore
    mechanism: str = ""


@dataclass(frozen=True)
class ObservableSessionState:
    """Market-derived session state without wall-clock identity."""

    session_volatility_state: float
    session_liquidity_state: float


@dataclass(frozen=True)
class MultiTimeframeState:
    """Directional 15m/1h confirmation derived from closed resampled bars."""

    tf15_trend: float
    tf1h_trend: float
    alignment_score: float


@dataclass(frozen=True)
class RegimeEntryScore:
    """Reusable entry mechanisms for one proposed direction."""

    exhaustion_reversal: float
    liquidation_absorption: float
    trend_continuation: float


@dataclass(frozen=True)
class RegimeTargetDecision:
    """Direction selected after reversal states take priority over continuation."""

    action: str
    mechanism: str
    confidence: float
    long_score: float
    short_score: float


# 计算 TREND_ROLLOVER_THRESHOLD_SCALE，仅使用当前及历史数据。
# 计算 TREND_ROLLOVER_THRESHOLD_SCALE，仅使用当前及历史数据。
TREND_ROLLOVER_THRESHOLD_SCALE = 1.0


def compress_momentum(row: pd.Series) -> MomentumState:
    """Compress returns into non-overlapping short, mid, and long states.

    Short and mid states retain their own causal horizon.  Keeping ``ret144``
    out of the mid state is important for rollover detection: a sharp 20-bar
    dip is not a confirmed trend reversal while the 60-bar return is flat.
    """

    ret20 = _number(row.get("ret20"), 0.0)
    ret60 = _number(row.get("ret60"), 0.0)
    ret144 = _number(row.get("ret144"), 0.0)
    ret288 = _number(row.get("ret288"), 0.0)
    return MomentumState(
        short_momentum=_signed_scale(ret20, 3.0),
        mid_momentum=_signed_scale(ret60, 3.0),
        long_momentum=_signed_scale(0.40 * ret144 + 0.60 * ret288, 8.0),
    )


def score_market_regime(row: pd.Series, *, action: str | None = None) -> MarketRegimeScore:
    """Score a directional setup without symbol identity or wall-clock inputs."""

    direction = -1.0 if str(action or row.get("action") or "").upper().endswith("SHORT") else 1.0
    momentum = compress_momentum(row)
    short_momentum = momentum.short_momentum * direction
    mid_momentum = momentum.mid_momentum * direction
    long_momentum = momentum.long_momentum * direction
    adx = _row_number(row, "adx", "adx14", default=0.0)
    qvr = max(0.0, _row_number(row, "qvr", "quote_volume_ratio_med30", default=0.0))
    rsi = _row_number(row, "rsi", "rsi14", default=50.0)
    don = _row_number(row, "don", "donchian60_pos", default=0.5)
    taker = _row_number(row, "taker", "taker_buy_ratio", default=0.5)
    down = _row_number(row, "down", "market_down_breadth20", default=0.5)
    directional_rsi = rsi if direction > 0 else 100.0 - rsi
    directional_don = don if direction > 0 else 1.0 - don
    directional_flow = taker if direction > 0 else 1.0 - taker
    directional_breadth = 1.0 - down if direction > 0 else down

    trend_strength = _clip01(
        0.24 * _scale(short_momentum, 0.0, 0.65)
        + 0.18 * _scale(mid_momentum, 0.0, 0.65)
        + 0.12 * _scale(long_momentum, 0.0, 0.65)
        + 0.24 * _scale(adx, 12.0, 45.0)
        + 0.22 * _scale(directional_don, 0.50, 0.95),
    )
    volume_support = _scale(qvr, 0.35, 3.0)
    volume_dislocation = _scale(qvr, 7.0, 16.0)
    liquidity_condition = _clip01(
        0.55 * volume_support
        + 0.30 * _scale(directional_flow, 0.45, 0.65)
        + 0.15 * _scale(directional_breadth, 0.40, 0.75)
        - 0.35 * volume_dislocation,
    )
    crowding_score = _clip01(
        0.34 * _scale(directional_rsi, 60.0, 82.0)
        + 0.28 * _scale(directional_don, 0.72, 0.98)
        + 0.22 * _scale(short_momentum, 0.10, 0.75)
        + 0.16 * volume_dislocation,
    )
    momentum_deceleration = _scale(short_momentum - mid_momentum, -0.05, -0.45)
    flow_rejection = _scale(0.50 - directional_flow, 0.0, 0.18)
    weak_liquidity = 1.0 - liquidity_condition
    reversal_probability = _clip01(
        0.45 * crowding_score
        + 0.25 * momentum_deceleration
        + 0.20 * flow_rejection
        + 0.10 * weak_liquidity,
    )
    regime_score = _clip01(
        0.50 * trend_strength
        + 0.28 * liquidity_condition
        + 0.12 * _scale(directional_flow, 0.45, 0.65)
        + 0.10 * _scale(directional_breadth, 0.40, 0.75)
        - 0.28 * crowding_score
        - 0.32 * reversal_probability,
    )
    return MarketRegimeScore(
        trend_strength=trend_strength,
        liquidity_condition=liquidity_condition,
        crowding_score=crowding_score,
        reversal_probability=reversal_probability,
        market_regime_score=regime_score,
    )


def decide_market_regime(row: pd.Series) -> MarketRegimeDecision:
    """Classify one closed-bar snapshot and map the regime to an action."""

    momentum = compress_momentum(row)
    long_score = score_market_regime(row, action="OPEN_LONG")
    short_score = score_market_regime(row, action="OPEN_SHORT")
    directional = momentum.short_momentum + 0.75 * momentum.mid_momentum + 0.50 * momentum.long_momentum
    preferred = long_score if directional >= 0.0 else short_score
    qvr = max(0.0, _row_number(row, "qvr", "quote_volume_ratio_med30", default=0.0))
    taker = _row_number(row, "taker", "taker_buy_ratio", default=0.5)
    direction = 1.0 if directional >= 0.0 else -1.0
    short_aligned = momentum.short_momentum * direction
    mid_aligned = momentum.mid_momentum * direction
    long_aligned = momentum.long_momentum * direction
    coherent_trend = bool(
        (
            short_aligned >= 0.08
            and mid_aligned >= 0.10
            and long_aligned >= 0.05
        )
        or (
            short_aligned >= -0.10
            and mid_aligned >= 0.30
            and long_aligned >= 0.20
        ),
    )
    primary_direction = 1.0 if momentum.mid_momentum + 0.75 * momentum.long_momentum >= 0.0 else -1.0
    primary_mid = momentum.mid_momentum * primary_direction
    primary_long = momentum.long_momentum * primary_direction
    primary_short = momentum.short_momentum * primary_direction
    recovery_flow_support = taker >= 0.55 if primary_direction > 0.0 else taker <= 0.45
    rollover_action, rollover_confidence = _trend_rollover_state(row, momentum)
    mechanism = ""

    if rollover_action:
        regime = MarketRegimeState.EXHAUSTION
        action = rollover_action
        preferred = long_score if action == "OPEN_LONG" else short_score
        mechanism = "trend_rollover"
    elif (
        abs(momentum.short_momentum) >= 0.75
        and abs(momentum.mid_momentum) >= 0.65
        and momentum.short_momentum * momentum.mid_momentum > 0.0
    ) or (qvr >= 7.0 and abs(momentum.short_momentum) >= 0.45):
        regime = MarketRegimeState.LIQUIDATION
        action = "HOLD"
    elif preferred.crowding_score >= 0.60 or preferred.reversal_probability >= 0.55:
        regime = MarketRegimeState.EXHAUSTION
        action = "OPEN_SHORT" if directional > 0.0 else "OPEN_LONG"
    elif (
        primary_mid >= 0.05
        and primary_long >= 0.05
        and -0.35 <= primary_short <= 0.08
        and recovery_flow_support
    ):
        regime = MarketRegimeState.RECOVERY
        action = "OPEN_LONG" if primary_direction > 0.0 else "OPEN_SHORT"
        preferred = long_score if action == "OPEN_LONG" else short_score
    elif (
        abs(momentum.short_momentum) <= 0.18
        and abs(momentum.mid_momentum) <= 0.22
        and preferred.liquidity_condition >= 0.35
    ) or not coherent_trend:
        regime = MarketRegimeState.ACCUMULATION
        action = "HOLD"
    else:
        regime = MarketRegimeState.TRENDING
        action = "OPEN_LONG" if directional >= 0.0 else "OPEN_SHORT"
    confidence = _clip01(
        preferred.market_regime_score
        + 0.20 * abs(momentum.short_momentum)
        - 0.15 * preferred.reversal_probability,
    )
    if regime is MarketRegimeState.RECOVERY:
        confidence = max(
            confidence,
            _clip01(
                0.35 * preferred.liquidity_condition
                + 0.35 * primary_mid
                + 0.30 * primary_long,
            ),
        )
    elif mechanism == "trend_rollover":
        confidence = max(confidence, rollover_confidence)
    return MarketRegimeDecision(
        regime=regime,
        action=action,
        confidence=confidence,
        momentum=momentum,
        score=preferred,
        mechanism=mechanism,
    )


def _trend_rollover_state(
    row: pd.Series,
    momentum: MomentumState,
) -> tuple[str, float]:
    """Detect a reusable long-horizon trend rolling over on shorter horizons.

    The state is intentionally symmetric and does not inspect symbol identity
    or wall-clock time.  It requires long-horizon crowding, an opposing short
    impulse, a 15m structure break, weak cross-sectional breadth, and usable
    liquidity.  This distinguishes a genuine rollover from an ordinary dip.
    """

    threshold_scale = max(0.50, min(1.50, float(TREND_ROLLOVER_THRESHOLD_SCALE)))
    if abs(momentum.long_momentum) < 0.35 * threshold_scale:
        return "", 0.0
    primary_direction = 1.0 if momentum.long_momentum >= 0.0 else -1.0
    opposing_short = momentum.short_momentum * primary_direction
    opposing_mid = momentum.mid_momentum * primary_direction
    tf15_impulse = _row_number(row, "tf15_ret4", default=0.0) * primary_direction
    don = _row_number(row, "don", "donchian60_pos", default=0.5)
    down = _row_number(row, "down", "market_down_breadth20", default=0.5)
    qvr = max(0.0, _row_number(row, "qvr", "quote_volume_ratio_med30", default=0.0))
    opposing_structure = (1.0 - don) if primary_direction > 0.0 else don
    opposing_breadth = down if primary_direction > 0.0 else 1.0 - down
    active = bool(
        opposing_short <= -0.08 * threshold_scale
        and opposing_mid <= -0.15 * threshold_scale
        and tf15_impulse <= -0.25 * threshold_scale
        and opposing_structure >= 0.65 * threshold_scale
        and opposing_breadth >= 0.65 * threshold_scale
        and qvr >= 0.50 * threshold_scale,
    )
    if not active:
        return "", 0.0
    confidence = _clip01(
        0.30 * _scale(abs(momentum.long_momentum), 0.35, 1.0)
        + 0.25 * _scale(-opposing_short, 0.08, 0.65)
        + 0.20 * _scale(-tf15_impulse, 0.25, 1.50)
        + 0.15 * _scale(opposing_structure, 0.65, 0.95)
        + 0.10 * _scale(opposing_breadth, 0.65, 0.90),
    )
    action = "OPEN_SHORT" if primary_direction > 0.0 else "OPEN_LONG"
    return action, confidence


def observable_session_state(row: pd.Series) -> ObservableSessionState:
    """Describe the current trading session from volatility and liquidity."""

    atr_pct = max(0.0, _row_number(row, "atr_pct", "atr14_pct", default=0.0))
    qvr = max(0.0, _row_number(row, "qvr", "quote_volume_ratio_med30", default=0.0))
    return ObservableSessionState(
        session_volatility_state=_clip01(atr_pct / 2.0),
        session_liquidity_state=_clip01(np.log1p(qvr) / np.log(6.0)),
    )


def score_multi_timeframe_state(row: pd.Series, *, action: str) -> MultiTimeframeState:
    """Score multi-timeframe direction without symbol or clock features."""

    direction = -1.0 if str(action).upper().endswith("SHORT") else 1.0
    tf15_trend = float(
        np.clip(
            0.40 * _signed_scale(_row_number(row, "tf15_ret4", default=0.0), 1.5)
            + 0.35 * _signed_scale(_row_number(row, "tf15_ret16", default=0.0), 3.0)
            + 0.25 * _signed_scale(
                _row_number(row, "tf15_ema21_slope4_pct", default=0.0),
                1.0,
            ),
            -1.0,
            1.0,
        ),
    )
    tf1h_trend = float(
        np.clip(
            0.40 * _signed_scale(_row_number(row, "tf1h_ret4", default=0.0), 3.0)
            + 0.35 * _signed_scale(_row_number(row, "tf1h_ret12", default=0.0), 6.0)
            + 0.25 * _signed_scale(
                _row_number(row, "tf1h_ema21_slope3_pct", default=0.0),
                1.5,
            ),
            -1.0,
            1.0,
        ),
    )
    directional_15m = tf15_trend * direction
    directional_1h = tf1h_trend * direction
    alignment_score = _clip01(
        0.50
        + 0.25 * directional_15m
        + 0.25 * directional_1h,
    )
    return MultiTimeframeState(
        tf15_trend=tf15_trend,
        tf1h_trend=tf1h_trend,
        alignment_score=alignment_score,
    )


def score_regime_entry(
    row: pd.Series,
    *,
    action: str,
    recent_opposing_crowding: float | None = None,
) -> RegimeEntryScore:
    """Score three causal entry mechanisms without symbol or clock identity.

    ``recent_opposing_crowding`` is a trailing, closed-bar maximum supplied by
    the caller.  It lets the strategy react to a rejection after an extreme,
    instead of requiring the extreme and reversal candle to be the same bar.
    """

    normalized_action = str(action).upper()
    if normalized_action not in {"OPEN_LONG", "OPEN_SHORT"}:
        raise ValueError(f"unsupported regime entry action: {action}")
    direction = 1.0 if normalized_action == "OPEN_LONG" else -1.0
    own = score_market_regime(row, action=normalized_action)
    opposite_action = "OPEN_SHORT" if direction > 0.0 else "OPEN_LONG"
    opposing = score_market_regime(row, action=opposite_action)
    opposing_crowding = (
        opposing.crowding_score
        if recent_opposing_crowding is None
        else _clip01(recent_opposing_crowding)
    )

    upper_wick = max(0.0, _row_number(row, "upper_wick_pct", default=0.0))
    lower_wick = max(0.0, _row_number(row, "lower_wick_pct", default=0.0))
    wick_total = upper_wick + lower_wick
    directional_wick_share = (
        (lower_wick if direction > 0.0 else upper_wick) / wick_total
        if wick_total > 1e-12
        else 0.50
    )
    taker = _row_number(row, "taker", "taker_buy_ratio", default=0.50)
    close_position = _row_number(row, "close_pos", default=0.50)
    directional_flow = taker if direction > 0.0 else 1.0 - taker
    directional_close = close_position if direction > 0.0 else 1.0 - close_position
    rejection_confirmation = _clip01(
        0.35 * _scale(directional_wick_share, 0.45, 0.90)
        + 0.35 * _scale(directional_flow, 0.48, 0.70)
        + 0.30 * _scale(directional_close, 0.40, 1.00),
    )

    qvr = max(0.0, _row_number(row, "qvr", "quote_volume_ratio_med30", default=0.0))
    qvr_support = _clip01(np.log1p(qvr) / np.log(8.0))
    qvr_dislocation = _scale(qvr, 3.0, 12.0)
    adx = _row_number(row, "adx", "adx14", default=0.0)
    trend_continuation = _clip01(
        0.35 * own.trend_strength
        + 0.25 * own.liquidity_condition
        + 0.15 * (1.0 - own.reversal_probability)
        + 0.15 * _scale(adx, 15.0, 45.0)
        + 0.10 * qvr_support,
    )
    return RegimeEntryScore(
        exhaustion_reversal=_clip01(
            0.60 * opposing_crowding
            + 0.25 * rejection_confirmation
            + 0.15 * qvr_support,
        ),
        liquidation_absorption=_clip01(
            0.45 * opposing_crowding
            + 0.30 * rejection_confirmation
            + 0.25 * qvr_dislocation,
        ),
        trend_continuation=trend_continuation,
    )


def decide_regime_target(
    row: pd.Series,
    *,
    reversal_priority_floor: float = 0.40,
) -> RegimeTargetDecision:
    """Choose a direction with explicit reversal-before-continuation priority.

    A strong exhaustion or liquidation state represents a change in market
    mechanism, so a same-direction continuation score cannot override it.
    This fixes the common failure where an overbought bar is classified as a
    stronger long trend exactly when the strategy should evaluate a short.
    """

    long = score_regime_entry(row, action="OPEN_LONG")
    short = score_regime_entry(row, action="OPEN_SHORT")
    long_reversal = max(long.exhaustion_reversal, long.liquidation_absorption)
    short_reversal = max(short.exhaustion_reversal, short.liquidation_absorption)
    if max(long_reversal, short_reversal) >= float(reversal_priority_floor):
        if long_reversal >= short_reversal:
            action = "OPEN_LONG"
            confidence = long_reversal
            mechanism = (
                "liquidation_absorption"
                if long.liquidation_absorption >= long.exhaustion_reversal
                else "exhaustion_reversal"
            )
        else:
            action = "OPEN_SHORT"
            confidence = short_reversal
            mechanism = (
                "liquidation_absorption"
                if short.liquidation_absorption >= short.exhaustion_reversal
                else "exhaustion_reversal"
            )
    elif long.trend_continuation >= short.trend_continuation:
        action = "OPEN_LONG"
        confidence = long.trend_continuation
        mechanism = "trend_continuation"
    else:
        action = "OPEN_SHORT"
        confidence = short.trend_continuation
        mechanism = "trend_continuation"
    return RegimeTargetDecision(
        action=action,
        mechanism=mechanism,
        confidence=float(confidence),
        long_score=max(
            long_reversal,
            long.trend_continuation,
        ),
        short_score=max(
            short_reversal,
            short.trend_continuation,
        ),
    )


def market_regime_open_allowed(
    row: pd.Series,
    *,
    minimum_score: float = 0.32,
    maximum_crowding: float = 0.78,
    maximum_reversal_probability: float = 0.62,
) -> bool:
    """Apply one reusable continuation gate across every tradable symbol."""

    regime = score_market_regime(row)
    decision = decide_market_regime(row)
    proposed_action = str(row.get("action") or "").upper()
    return bool(
        decision.regime in {MarketRegimeState.TRENDING, MarketRegimeState.RECOVERY}
        and decision.action == proposed_action
        and regime.market_regime_score >= float(minimum_score)
        and regime.crowding_score <= float(maximum_crowding)
        and regime.reversal_probability <= float(maximum_reversal_probability),
    )


def _scale(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    return _clip01((float(value) - float(lower)) / (float(upper) - float(lower)))


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _signed_scale(value: float, scale: float) -> float:
    return float(np.clip(float(value) / max(float(scale), 1e-9), -1.0, 1.0))


def _number(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _row_number(row: pd.Series, *names: str, default: float) -> float:
    for name in names:
        value = row.get(name)
        if value is not None and not pd.isna(value):
            return _number(value, default)
    return float(default)
