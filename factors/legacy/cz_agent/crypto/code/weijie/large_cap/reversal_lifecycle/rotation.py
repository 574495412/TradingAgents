"""根据市场状态切换构建趋势重入和反向轮动活动。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from .components import (
    directional_market_components,
)
from .components import (
    linear_ramp as _ramp,
)

TREND_REJOIN = "trend_rejoin_after_breadth_repair"
LOW_VOLATILITY_SHALLOW_REPAIR = "low_volatility_shallow_repair"
REBOUND_EXHAUSTION = "countertrend_rebound_exhaustion"
TRANSITION_BLOWOFF = "transition_phase_blowoff_reversal"
ROTATION_MECHANISMS = (
    TREND_REJOIN,
    REBOUND_EXHAUSTION,
    TRANSITION_BLOWOFF,
)
MINIMUM_THRESHOLD_MULTIPLIER = 0.8
MAXIMUM_THRESHOLD_MULTIPLIER = 1.2
_TREND_REJOIN_LONG_MOMENTUM_RANGE = (2.0, 5.0)
_TREND_REJOIN_MID_MOMENTUM_RANGE = (2.0, 5.0)
_TREND_REJOIN_PULLBACK_RANGE = (0.30, 1.20)
_TREND_REJOIN_BREADTH_RANGE = (0.66, 1.0)


@dataclass(frozen=True, slots=True)
class RegimeRotationPolicy:
    """状态轮动的入口、退出、持仓期限和阈值扰动参数。"""

    enabled_mechanisms: tuple[str, ...] = ROTATION_MECHANISMS
    cooldown_bars: int = 12
    maximum_holding_bars: int = 288
    minimum_breadth: float = 0.75
    trend_rejoin_minimum_strength: float = 0.58
    trend_rejoin_location_minimum: float = 0.30
    trend_rejoin_location_maximum: float = 0.75
    trend_rejoin_shallow_minimum_tiny_return_pct: float = 0.10
    trend_rejoin_shallow_minimum_prior_pullback_pct: float = 0.08
    trend_rejoin_shallow_minimum_mid_momentum_pct: float = 0.10
    trend_rejoin_shallow_maximum_mid_momentum_pct: float = 1.00
    trend_rejoin_shallow_minimum_long_momentum_pct: float = 0.10
    trend_rejoin_shallow_maximum_long_momentum_pct: float = 1.00
    trend_rejoin_shallow_recovery_return_minimum_pct: float = -0.20
    trend_rejoin_shallow_recovery_return_maximum_pct: float = 0.30
    trend_rejoin_shallow_location_minimum: float = 0.72
    trend_rejoin_shallow_location_maximum: float = 0.90
    trend_rejoin_shallow_minimum_breadth: float = 0.75
    trend_rejoin_shallow_maximum_atr_ratio: float = 1.00
    trend_rejoin_shallow_maximum_quote_volume_ratio: float = 1.50
    rebound_long_momentum_pct: float = 1.0
    rebound_counter_move_pct: float = 0.50
    rebound_extreme_location: float = 0.10
    rebound_extreme_rsi: float = 40.0
    rebound_minimum_relative_momentum_pct: float = 0.50
    rebound_minimum_quote_volume_ratio: float = 2.0
    blowoff_adverse_mid_momentum_pct: float = 2.50
    blowoff_adverse_long_minimum_pct: float = 1.0
    blowoff_adverse_long_maximum_pct: float = 3.0
    blowoff_rejection_return_pct: float = 0.10
    blowoff_extreme_location: float = 0.15
    blowoff_extreme_rsi: float = 35.0
    blowoff_minimum_adx: float = 40.0
    blowoff_maximum_quote_volume_ratio: float = 2.0
    blowoff_exit_return_pct: float = 0.50
    blowoff_exit_location: float = 0.90
    blowoff_exit_rsi: float = 65.0
    blowoff_exit_breadth: float = 0.90
    exhaustion_mid_momentum_pct: float = 2.0
    exhaustion_location: float = 0.90
    exhaustion_rsi: float = 80.0
    exhaustion_quote_volume_ratio: float = 5.0
    pullback_exit_return_pct: float = 0.50
    pullback_exit_location: float = 0.70
    pullback_exit_rsi: float = 60.0

    def with_threshold_multiplier(
        self,
        multiplier: float,
    ) -> RegimeRotationPolicy:
        """统一扰动连续门槛, 用于轮动机制的稳定性检查。"""

        if not (
            MINIMUM_THRESHOLD_MULTIPLIER
            <= multiplier
            <= MAXIMUM_THRESHOLD_MULTIPLIER
        ):
            message = "rotation threshold multiplier must be in [0.8, 1.2]"
            raise ValueError(message)
        fields = (
            "trend_rejoin_minimum_strength",
            "trend_rejoin_shallow_minimum_tiny_return_pct",
            "trend_rejoin_shallow_minimum_prior_pullback_pct",
            "trend_rejoin_shallow_minimum_mid_momentum_pct",
            "trend_rejoin_shallow_maximum_mid_momentum_pct",
            "trend_rejoin_shallow_minimum_long_momentum_pct",
            "trend_rejoin_shallow_maximum_long_momentum_pct",
            "rebound_long_momentum_pct",
            "rebound_counter_move_pct",
            "rebound_minimum_relative_momentum_pct",
            "rebound_minimum_quote_volume_ratio",
            "exhaustion_mid_momentum_pct",
            "exhaustion_quote_volume_ratio",
            "pullback_exit_return_pct",
            "blowoff_adverse_mid_momentum_pct",
            "blowoff_adverse_long_minimum_pct",
            "blowoff_adverse_long_maximum_pct",
            "blowoff_rejection_return_pct",
            "blowoff_minimum_adx",
            "blowoff_maximum_quote_volume_ratio",
            "blowoff_exit_return_pct",
        )
        return replace(
            self,
            **{
                name: float(getattr(self, name)) * multiplier
                for name in fields
            },
        )


def build_regime_rotation_actions(
    market_panel: pd.DataFrame,
    *,
    policy: RegimeRotationPolicy | None = None,
) -> pd.DataFrame:
    """扫描因果市场面板并生成非重叠的状态轮动动作链。"""

    cfg = policy or RegimeRotationPolicy()
    unknown = sorted(set(cfg.enabled_mechanisms).difference(ROTATION_MECHANISMS))
    if unknown:
        raise ValueError("unknown rotation mechanisms: " + ", ".join(unknown))
    market = _prepare_market(market_panel)
    columns = (
        "signal_id",
        "chain_id",
        "symbol",
        "action",
        "operation_type",
        "signal_time",
        "signal_price",
        "position_fraction",
        "reason_family",
        "reason",
        "score",
        "campaign_risk_profile",
    )
    if market.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    for mechanism in cfg.enabled_mechanisms:
        records.extend(
            _mechanism_actions(
                market,
                mechanism=mechanism,
                policy=cfg,
            ),
        )
    return (
        pd.DataFrame.from_records(records, columns=columns)
        .sort_values(["signal_time", "signal_id"], kind="stable")
        .reset_index(drop=True)
    )


def _mechanism_actions(
    market: pd.DataFrame,
    *,
    mechanism: str,
    policy: RegimeRotationPolicy,
) -> list[dict[str, object]]:
    """为单一轮动机制构建完整的开仓、持有和退出动作。"""

    armed = {1.0: True, -1.0: True}
    last_exit_index = -policy.cooldown_bars - 1
    position: dict[str, Any] | None = None
    records: list[dict[str, object]] = []
    sequence = 0
    for bar_index, bar in market.iterrows():
        if position is not None:
            direction = float(position["direction"])
            held_bars = bar_index - int(position["entry_index"])
            if _exit_state(
                bar,
                direction=direction,
                mechanism=mechanism,
                policy=policy,
            ) or held_bars >= policy.maximum_holding_bars:
                records.append(
                    _action_row(
                        bar,
                        chain_id=str(position["chain_id"]),
                        direction=direction,
                        operation_type="CLOSE",
                        mechanism=mechanism,
                        score=float(position["score"]),
                        campaign_risk_profile=str(
                            position.get("campaign_risk_profile", ""),
                        ),
                    ),
                )
                position = None
                last_exit_index = bar_index
            continue
        candidates: list[tuple[float, float, str]] = []
        previous_bar = market.iloc[bar_index - 1] if bar_index > 0 else None
        for direction in (1.0, -1.0):
            active, score = _entry_state(
                bar,
                previous_bar=previous_bar,
                direction=direction,
                mechanism=mechanism,
                policy=policy,
            )
            if not active:
                armed[direction] = True
            elif (
                armed[direction]
                and bar_index - last_exit_index >= policy.cooldown_bars
            ):
                risk_profile = ""
                if mechanism == TREND_REJOIN:
                    shallow_active, _ = _shallow_breadth_repair_state(
                        bar,
                        previous_bar=previous_bar,
                        direction=direction,
                        values=_directional(bar, direction),
                        policy=policy,
                    )
                    if shallow_active:
                        risk_profile = LOW_VOLATILITY_SHALLOW_REPAIR
                candidates.append((score, direction, risk_profile))
        if not candidates:
            continue
        score, direction, risk_profile = max(
            candidates,
            key=lambda value: (value[0], value[1]),
        )
        sequence += 1
        chain_id = f"rotation:{mechanism}:{sequence:06d}"
        records.append(
            _action_row(
                bar,
                chain_id=chain_id,
                direction=direction,
                operation_type="OPEN",
                mechanism=mechanism,
                score=score,
                campaign_risk_profile=risk_profile,
            ),
        )
        armed[direction] = False
        position = {
            "chain_id": chain_id,
            "direction": direction,
            "entry_index": bar_index,
            "score": score,
            "campaign_risk_profile": risk_profile,
        }
    return records


def _entry_state(
    bar: pd.Series,
    *,
    previous_bar: pd.Series | None = None,
    direction: float,
    mechanism: str,
    policy: RegimeRotationPolicy,
) -> tuple[bool, float]:
    """判断目标方向的状态轮动入口并计算入口强度。"""

    values = _directional(bar, direction)
    if mechanism == TREND_REJOIN:
        classic_score = _trend_rejoin_strength(values, policy=policy)
        classic_active = bool(
            values["long_momentum"] > 0.0
            and values["mid_momentum"] > 0.0
            and values["tiny_return"] < 0.0
            and policy.trend_rejoin_location_minimum
            <= values["location"]
            <= policy.trend_rejoin_location_maximum
            and values["adverse_breadth20"] >= policy.minimum_breadth
            and classic_score >= policy.trend_rejoin_minimum_strength,
        )
        shallow_active, shallow_score = _shallow_breadth_repair_state(
            bar,
            previous_bar=previous_bar,
            direction=direction,
            values=values,
            policy=policy,
        )
        return (
            classic_active or shallow_active,
            max(classic_score, shallow_score),
        )
    if mechanism == TRANSITION_BLOWOFF:
        adverse_mid = -values["mid_momentum"]
        adverse_long = -values["long_momentum"]
        active = bool(
            adverse_mid >= policy.blowoff_adverse_mid_momentum_pct
            and policy.blowoff_adverse_long_minimum_pct
            <= adverse_long
            <= policy.blowoff_adverse_long_maximum_pct
            and values["tiny_return"] >= policy.blowoff_rejection_return_pct
            and values["location"] <= policy.blowoff_extreme_location
            and values["rsi"] <= policy.blowoff_extreme_rsi
            and values["adverse_breadth20"] >= policy.blowoff_exit_breadth
            and float(bar["adx14"]) >= policy.blowoff_minimum_adx
            and float(bar["quote_volume_ratio_med30"])
            <= policy.blowoff_maximum_quote_volume_ratio,
        )
        score = min(
            1.0,
            0.70
            + 0.03
            * (adverse_mid / policy.blowoff_adverse_mid_momentum_pct),
        )
        return active, score
    active = bool(
        values["long_momentum"] >= policy.rebound_long_momentum_pct
        and values["mid_momentum"] <= 0.0
        and values["recovery_return"] <= -policy.rebound_counter_move_pct
        and values["location"] <= policy.rebound_extreme_location
        and values["rsi"] <= policy.rebound_extreme_rsi
        and values["relative_momentum"]
        >= policy.rebound_minimum_relative_momentum_pct
        and values["adverse_breadth20"] >= policy.minimum_breadth
        and float(bar["quote_volume_ratio_med30"])
        >= policy.rebound_minimum_quote_volume_ratio,
    )
    score = min(
        1.0,
        0.70
        + 0.03 * (values["long_momentum"] / policy.rebound_long_momentum_pct),
    )
    return active, score


def _trend_rejoin_strength(
    values: dict[str, float],
    *,
    policy: RegimeRotationPolicy,
) -> float:
    """综合趋势、回踩、广度和流动性计算趋势重入强度。"""

    long_momentum = _ramp(
        values["long_momentum"],
        *_TREND_REJOIN_LONG_MOMENTUM_RANGE,
    )
    mid_momentum = _ramp(
        values["mid_momentum"],
        *_TREND_REJOIN_MID_MOMENTUM_RANGE,
    )
    momentum_continuity = (
        mid_momentum**0.80
        * long_momentum**0.20
    )
    pullback = _ramp(
        -values["tiny_return"],
        *_TREND_REJOIN_PULLBACK_RANGE,
    )
    breadth = _ramp(
        values["adverse_breadth20"],
        *_TREND_REJOIN_BREADTH_RANGE,
    )
    band_width = (
        policy.trend_rejoin_location_maximum
        - policy.trend_rejoin_location_minimum
    )
    band_midpoint = (
        policy.trend_rejoin_location_maximum
        + policy.trend_rejoin_location_minimum
    ) / 2.0
    location_quality = max(
        0.0,
        1.0
        - abs(values["location"] - band_midpoint)
        / max(band_width / 2.0, 1e-9),
    )
    return float(
        0.75 * momentum_continuity
        + 0.10 * pullback
        + 0.10 * breadth
        + 0.05 * location_quality,
    )


def _shallow_breadth_repair_state(
    bar: pd.Series,
    *,
    previous_bar: pd.Series | None,
    direction: float,
    values: dict[str, float],
    policy: RegimeRotationPolicy,
) -> tuple[bool, float]:
    """识别低波动趋势中的浅回撤及下一根闭合柱宽基恢复。"""

    if previous_bar is None:
        return False, 0.0
    previous = _directional(previous_bar, direction)
    active = bool(
        values["tiny_return"]
        >= policy.trend_rejoin_shallow_minimum_tiny_return_pct
        and previous["tiny_return"]
        <= -policy.trend_rejoin_shallow_minimum_prior_pullback_pct
        and policy.trend_rejoin_shallow_minimum_mid_momentum_pct
        <= values["mid_momentum"]
        <= policy.trend_rejoin_shallow_maximum_mid_momentum_pct
        and policy.trend_rejoin_shallow_minimum_long_momentum_pct
        <= values["long_momentum"]
        <= policy.trend_rejoin_shallow_maximum_long_momentum_pct
        and previous["long_momentum"]
        >= policy.trend_rejoin_shallow_minimum_long_momentum_pct
        and policy.trend_rejoin_shallow_recovery_return_minimum_pct
        <= values["recovery_return"]
        <= policy.trend_rejoin_shallow_recovery_return_maximum_pct
        and policy.trend_rejoin_shallow_location_minimum
        <= values["location"]
        <= policy.trend_rejoin_shallow_location_maximum
        and values["favorable_breadth3"]
        >= policy.trend_rejoin_shallow_minimum_breadth
        and values["favorable_breadth20"]
        >= policy.trend_rejoin_shallow_minimum_breadth
        and previous["adverse_breadth3"]
        >= policy.trend_rejoin_shallow_minimum_breadth
        and float(bar["atr14_ratio120"])
        <= policy.trend_rejoin_shallow_maximum_atr_ratio
        and float(bar["quote_volume_ratio_med30"])
        <= policy.trend_rejoin_shallow_maximum_quote_volume_ratio,
    )
    if not active:
        return False, 0.0

    mid_quality = _ramp(
        values["mid_momentum"],
        policy.trend_rejoin_shallow_minimum_mid_momentum_pct,
        policy.trend_rejoin_shallow_maximum_mid_momentum_pct,
    )
    long_quality = _ramp(
        values["long_momentum"],
        policy.trend_rejoin_shallow_minimum_long_momentum_pct,
        policy.trend_rejoin_shallow_maximum_long_momentum_pct,
    )
    current_repair = _ramp(
        values["tiny_return"],
        policy.trend_rejoin_shallow_minimum_tiny_return_pct,
        0.25,
    )
    prior_pullback = _ramp(
        -previous["tiny_return"],
        policy.trend_rejoin_shallow_minimum_prior_pullback_pct,
        0.25,
    )
    breadth_quality = (
        values["favorable_breadth3"]
        + values["favorable_breadth20"]
        + previous["adverse_breadth3"]
    ) / 3.0
    location_width = (
        policy.trend_rejoin_shallow_location_maximum
        - policy.trend_rejoin_shallow_location_minimum
    )
    location_midpoint = (
        policy.trend_rejoin_shallow_location_maximum
        + policy.trend_rejoin_shallow_location_minimum
    ) / 2.0
    location_quality = max(
        0.0,
        1.0
        - abs(values["location"] - location_midpoint)
        / max(location_width / 2.0, 1e-9),
    )
    volatility_quality = 1.0 - _ramp(
        float(bar["atr14_ratio120"]),
        0.50,
        policy.trend_rejoin_shallow_maximum_atr_ratio,
    )
    score = (
        0.50
        + 0.05 * (mid_quality + long_quality)
        + 0.05 * (current_repair + prior_pullback)
        + 0.15 * breadth_quality
        + 0.10 * location_quality
        + 0.05 * volatility_quality
    )
    return score >= policy.trend_rejoin_minimum_strength, float(score)


def _exit_state(
    bar: pd.Series,
    *,
    direction: float,
    mechanism: str,
    policy: RegimeRotationPolicy,
) -> bool:
    """判断轮动持仓是否因成熟、失效或超时而退出。"""

    values = _directional(bar, direction)
    if mechanism == TREND_REJOIN:
        return bool(
            values["mid_momentum"] >= policy.exhaustion_mid_momentum_pct
            and values["location"] >= policy.exhaustion_location
            and values["rsi"] >= policy.exhaustion_rsi
            and values["favorable_breadth20"] >= policy.minimum_breadth
            and float(bar["quote_volume_ratio_med30"])
            >= policy.exhaustion_quote_volume_ratio,
        )
    if mechanism == TRANSITION_BLOWOFF:
        return bool(
            values["recovery_return"] >= policy.blowoff_exit_return_pct
            and values["location"] >= policy.blowoff_exit_location
            and values["rsi"] >= policy.blowoff_exit_rsi
            and values["favorable_breadth20"] >= policy.blowoff_exit_breadth,
        )
    return bool(
        values["recovery_return"] >= policy.pullback_exit_return_pct
        and values["location"] >= policy.pullback_exit_location
        and values["rsi"] >= policy.pullback_exit_rsi
        and values["favorable_breadth20"] >= policy.minimum_breadth,
    )


def _directional(bar: pd.Series, direction: float) -> dict[str, float]:
    """提取相对于目标方向一致的市场分量。"""

    values = directional_market_components(bar, direction)
    return {
        "tiny_return": values["tiny_return"],
        "recovery_return": values["recovery_return"],
        "mid_momentum": values["mid_momentum"],
        "long_momentum": values["long_momentum"],
        "relative_momentum": values["relative_momentum"],
        "location": values["location"],
        "rsi": values["rsi"],
        "favorable_breadth20": values["favorable_breadth20"],
        "adverse_breadth20": values["adverse_breadth20"],
        "favorable_breadth3": values["favorable_breadth3"],
        "adverse_breadth3": values["adverse_breadth3"],
    }


def _action_row(
    bar: pd.Series,
    *,
    chain_id: str,
    direction: float,
    operation_type: str,
    mechanism: str,
    score: float,
    campaign_risk_profile: str = "",
) -> dict[str, object]:
    """生成带轮动机制、动作类型和稳定标识的动作记录。"""

    direction_name = "LONG" if direction > 0.0 else "SHORT"
    timestamp = pd.Timestamp(bar["signal_time"])
    action_kind = "OPEN" if operation_type == "OPEN" else "CLOSE"
    return {
        "signal_id": f"{chain_id}:{operation_type}:{timestamp.isoformat()}",
        "chain_id": chain_id,
        "symbol": str(bar["symbol"]),
        "action": f"{action_kind}_{direction_name}",
        "operation_type": operation_type,
        "signal_time": timestamp,
        "signal_price": float(bar["close"]),
        "position_fraction": 1.0,
        "reason_family": mechanism,
        "reason": f"market_regime_rotation;mechanism={mechanism}",
        "score": float(score),
        "campaign_risk_profile": campaign_risk_profile,
    }


def _prepare_market(market_panel: pd.DataFrame) -> pd.DataFrame:
    """校验轮动市场面板并按 UTC 信号时间稳定排序。"""

    required = {
        "signal_time",
        "symbol",
        "close",
        "ret3",
        "ret12",
        "ret60",
        "tf1h_ret12",
        "rel_ret60",
        "donchian60_pos",
        "rsi14",
        "quote_volume_ratio_med30",
        "adx14",
        "market_down_breadth20",
        "market_up_breadth20",
        "market_down_breadth3",
        "market_up_breadth3",
        "atr14_ratio120",
    }
    missing = sorted(required.difference(market_panel.columns))
    if missing:
        raise ValueError("rotation panel missing columns: " + ", ".join(missing))
    market = (
        market_panel.copy()
        .sort_values("signal_time", kind="stable")
        .reset_index(drop=True)
    )
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    symbols = market["symbol"].dropna().astype(str).unique()
    if len(symbols) != 1:
        message = "rotation lifecycle requires exactly one symbol"
        raise ValueError(message)
    return market


__all__ = [
    "REBOUND_EXHAUSTION",
    "ROTATION_MECHANISMS",
    "TRANSITION_BLOWOFF",
    "TREND_REJOIN",
    "RegimeRotationPolicy",
    "build_regime_rotation_actions",
]
