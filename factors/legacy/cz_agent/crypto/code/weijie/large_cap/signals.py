"""大盘标的市场结构模块：提供 signals 因子逻辑。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from factors.crypto.code.weijie.common import weijie_market_prior_group
from factors.crypto.code.weijie.large_cap.eligibility import (
    eligible_market_support,
)
from factors.crypto.code.weijie.large_cap.lifecycle import (
    build_large_cap_lifecycle,
)
from factors.crypto.code.weijie.large_cap.mechanisms import score_large_cap_mechanisms
from factors.crypto.code.weijie.large_cap.policy import LargeCapRegimePolicy
from factors.crypto.code.weijie.large_cap.types import (
    LargeCapFactorSnapshot,
    LargeCapRegimeDecision,
)

_ACTIONS = ("OPEN_LONG", "OPEN_SHORT")
_MECHANISMS = (
    "trend_rollover",
    "liquidation_absorption",
    "pullback_recovery",
    "deleveraging_reversal",
    "crowded_extension",
)


def build_large_cap_regime_open_signals(
    feature_rows: pd.DataFrame,
    *,
    policy: LargeCapRegimePolicy | None = None,
) -> pd.DataFrame:
    """根据大盘状态和机制确认构建开仓信号。"""

    cfg = policy or LargeCapRegimePolicy()
    opens, _closes = build_large_cap_lifecycle(
        feature_rows,
        _build_signals(feature_rows, policy=cfg),
        policy=cfg,
    )
    return opens


def build_large_cap_regime_close_signals(
    feature_rows: pd.DataFrame,
    *,
    policy: LargeCapRegimePolicy | None = None,
) -> pd.DataFrame:
    """根据持仓生命周期与状态失效条件构建平仓信号。"""

    cfg = policy or LargeCapRegimePolicy()
    _opens, closes = build_large_cap_lifecycle(
        feature_rows,
        _build_signals(feature_rows, policy=cfg),
        policy=cfg,
    )
    return closes


def _build_signals(
    feature_rows: pd.DataFrame,
    *,
    policy: LargeCapRegimePolicy,
) -> pd.DataFrame:
    if feature_rows is None or feature_rows.empty:
        return _empty_signals()
    prepared = feature_rows.copy()
    prepared["bar_time"] = pd.to_datetime(prepared.get("bar_time"), errors="coerce")
    symbols = prepared.get("symbol", pd.Series("", index=prepared.index)).astype(str)
    prepared = prepared[
        prepared["bar_time"].notna()
        & symbols.map(weijie_market_prior_group).eq("large_cap_momentum")
    ].copy()
    if prepared.empty:
        return _empty_signals()

    decisions: list[tuple[pd.Timestamp, str, LargeCapRegimeDecision]] = []
    for symbol, symbol_rows in prepared.groupby("symbol", sort=False):
        ordered = symbol_rows.sort_values("bar_time", kind="stable")
        history: dict[str, list[float]] = {action: [] for action in _ACTIONS}
        armed: dict[tuple[str, str], bool] = {
            (action, mechanism): True
            for action in _ACTIONS
            for mechanism in policy.enabled_mechanisms
        }
        confirmation_bars: dict[tuple[str, str], int] = dict.fromkeys(armed, 0)
        last_emitted_at: pd.Timestamp | None = None
        for _, row in ordered.iterrows():
            signal_time = pd.Timestamp(row["bar_time"])
            bar_candidates: list[LargeCapRegimeDecision] = []
            for action in _ACTIONS:
                pullback_history = history[action]
                recent_pullback = max(
                    pullback_history[-max(1, int(policy.recent_pullback_bars)) :],
                    default=0.0,
                )
                snapshot = score_large_cap_mechanisms(
                    row,
                    action=action,
                    recent_pullback=recent_pullback,
                    enabled_trend_rollover_components=(
                        policy.enabled_trend_rollover_components
                    ),
                )
                scores = eligible_market_support(snapshot)
                for mechanism in policy.enabled_mechanisms:
                    if mechanism not in _MECHANISMS:
                        raise ValueError(f"unknown enabled large-cap mechanism: {mechanism}")
                    confidence = float(scores[mechanism])
                    threshold = policy.threshold(mechanism)
                    key = (action, mechanism)
                    release = max(0.0, threshold - float(policy.release_hysteresis))
                    if confidence <= release:
                        armed[key] = True
                        confirmation_bars[key] = 0
                        continue
                    if confidence < threshold:
                        confirmation_bars[key] = 0
                        continue
                    if not _mechanism_confirmed(
                        snapshot,
                        mechanism,
                        enabled_trend_rollover_components=(
                            policy.enabled_trend_rollover_components
                        ),
                    ):
                        confirmation_bars[key] = 0
                        continue
                    confirmation_bars[key] += 1
                    if not armed[key]:
                        continue
                    if confirmation_bars[key] < max(
                        1,
                        int(policy.minimum_confirmation_bars),
                    ):
                        continue
                    bar_candidates.append(
                        LargeCapRegimeDecision(
                            action=action,
                            mechanism=mechanism,
                            confidence=confidence,
                            snapshot=snapshot,
                        ),
                    )
                history[action].append(max(0.0, -snapshot.momentum.short))
            if not bar_candidates:
                continue
            chosen = max(
                bar_candidates,
                key=lambda decision: (
                    decision.confidence - policy.threshold(decision.mechanism),
                    decision.confidence,
                    decision.action,
                ),
            )
            # 同一根 K 线只采用置信度超过阈值最多的候选信号。
            # 其余候选同步解除 armed 状态，等待下一次独立触发。
            for candidate in bar_candidates:
                armed[(candidate.action, candidate.mechanism)] = False
            cooldown = pd.Timedelta(minutes=max(0.0, policy.open_cooldown_minutes))
            if (
                last_emitted_at is not None
                and signal_time - last_emitted_at < cooldown
            ):
                continue
            last_emitted_at = signal_time
            decisions.append((signal_time, str(symbol), chosen))

    rows = [
        _signal_row(
            signal_time=signal_time,
            symbol=symbol,
            decision=decision,
        )
        for signal_time, symbol, decision in decisions
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_signals()
    out = out.sort_values(["signal_time", "symbol", "action"], kind="stable").reset_index(drop=True)
    out["signal_id"] = [f"WJLCOPEN{index + 1:06d}" for index in range(len(out))]
    return out


def _mechanism_confirmed(
    snapshot: LargeCapFactorSnapshot,
    mechanism: str,
    *,
    enabled_trend_rollover_components: tuple[str, ...],
) -> bool:
    """检查候选动作对应的市场机制是否获得足够确认。"""

    momentum = snapshot.momentum
    liquidity = snapshot.liquidity
    rejection = snapshot.rejection
    structure = snapshot.structure
    scores = snapshot.mechanisms
    if mechanism == "trend_rollover":
        components = snapshot.exhaustion
        values = {
            "trend_divergence": components.trend_divergence,
            "low_liquidity_blowoff": components.low_liquidity_blowoff,
            "volume_climax": components.volume_climax,
            "cross_horizon_stall": components.cross_horizon_stall,
        }
        enabled_values = {
            name: values[name]
            for name in enabled_trend_rollover_components
            if name in values
        }
        if not enabled_values:
            return False
        dominant = max(enabled_values, key=enabled_values.get)
        if dominant == "volume_climax":
            return bool(
                structure.adverse_crowding >= 0.25
                and liquidity.dislocation_risk >= 0.20,
            )
        if dominant == "cross_horizon_stall":
            return bool(
                structure.adverse_crowding >= 0.25
                and abs(momentum.short) <= 0.22,
            )
        if dominant == "low_liquidity_blowoff":
            return bool(
                structure.adverse_crowding >= 0.25
                and liquidity.volume_support <= 0.30,
            )
        return bool(
            structure.adverse_crowding >= 0.30
            and (momentum.short >= 0.0 or rejection.quality >= 0.30),
        )
    if mechanism == "liquidation_absorption":
        return bool(
            structure.adverse_crowding >= 0.30
            and liquidity.volume_support >= 0.15
            and (
                rejection.quality >= 0.24
                or liquidity.dislocation_risk >= 0.25
                or scores.shock_pressure >= 0.50
            ),
        )
    if mechanism == "pullback_recovery":
        return bool(
            liquidity.quality >= 0.35
            and structure.multi_timeframe_alignment >= 0.45
            and snapshot.recovery.pullback_memory >= 0.20
            and momentum.short >= 0.08
            and momentum.mid >= 0.06
            and momentum.long >= 0.06
            and structure.trend_alignment >= 0.24,
        )
    if mechanism == "deleveraging_reversal":
        positioning = snapshot.positioning
        return bool(
            positioning.available
            and max(-momentum.short, -momentum.mid) >= 0.12
            and structure.adverse_crowding >= 0.50
            and rejection.quality >= 0.20
            and max(
                positioning.oi_deleveraging,
                positioning.oi_value_deleveraging,
            ) >= 0.12
            and liquidity.quality >= 0.10,
        )
    if mechanism == "crowded_extension":
        positioning = snapshot.positioning
        return bool(
            positioning.available
            and max(-momentum.short, -momentum.mid) >= 0.12
            and scores.shock_pressure >= 0.50
            and positioning.oi_build >= 0.10
            and max(
                positioning.oi_value_build,
                positioning.value_ratio_high,
            ) >= 0.10
            and liquidity.quality >= 0.10,
        )
    raise ValueError(f"unknown large-cap mechanism: {mechanism}")


def _signal_row(
    *,
    signal_time: pd.Timestamp,
    symbol: str,
    decision: LargeCapRegimeDecision,
) -> dict[str, Any]:
    snapshot = decision.snapshot
    action = decision.action
    reason_family = f"large_cap_{decision.mechanism}_entry"
    reason = (
        f"{reason_family};mechanism={decision.mechanism};"
        f"confidence={decision.confidence:.4f};"
        f"short_momentum={snapshot.momentum.short:.4f};"
        f"mid_momentum={snapshot.momentum.mid:.4f};"
        f"long_momentum={snapshot.momentum.long:.4f};"
        f"trend_alignment={snapshot.structure.trend_alignment:.4f};"
        f"adverse_crowding={snapshot.structure.adverse_crowding:.4f};"
        f"breadth_confirmation={snapshot.structure.breadth_confirmation:.4f};"
        f"liquidity_quality={snapshot.liquidity.quality:.4f};"
        f"rejection_quality={snapshot.rejection.quality:.4f};"
        f"shock_pressure={snapshot.mechanisms.shock_pressure:.4f}"
        f";trend_divergence={snapshot.exhaustion.trend_divergence:.4f}"
        f";low_liquidity_blowoff={snapshot.exhaustion.low_liquidity_blowoff:.4f}"
        f";volume_climax={snapshot.exhaustion.volume_climax:.4f}"
        f";cross_horizon_stall={snapshot.exhaustion.cross_horizon_stall:.4f}"
        f";liquidation_flush={snapshot.absorption.liquidation_flush:.4f}"
        f";deep_value={snapshot.absorption.deep_value:.4f}"
        f";flow_reversal={snapshot.absorption.flow_reversal:.4f}"
        f";flow_reclaim={snapshot.recovery.flow_reclaim:.4f}"
        f";trend_pullback={snapshot.recovery.trend_pullback:.4f}"
        f";continuation={snapshot.recovery.continuation:.4f}"
        f";positioning_available={int(snapshot.positioning.available)}"
        f";oi_deleveraging={snapshot.positioning.oi_deleveraging:.4f}"
        f";oi_value_deleveraging={snapshot.positioning.oi_value_deleveraging:.4f}"
        f";oi_build={snapshot.positioning.oi_build:.4f}"
        f";oi_value_build={snapshot.positioning.oi_value_build:.4f}"
        f";value_ratio_low={snapshot.positioning.value_ratio_low:.4f}"
        f";value_ratio_high={snapshot.positioning.value_ratio_high:.4f}"
        f";opposing_funding_crowding={snapshot.positioning.opposing_funding_crowding:.4f}"
    )
    return {
        "signal_id": "",
        "symbol": symbol,
        "action": action,
        "signal_time": signal_time,
        "score": 2.0 + 4.0 * decision.confidence,
        "reason": reason,
        "reason_family": reason_family,
        "ret20": snapshot.momentum.short * 3.0,
        "ret60": snapshot.momentum.mid * 5.0,
        "ret144": 0.0,
        "ret288": snapshot.momentum.long * 8.0,
        "don": 0.50,
        "taker": snapshot.liquidity.flow_support,
        "qvr": snapshot.liquidity.volume_support,
        "rsi": 50.0,
        "adx": 20.0,
        # 说明：以下逻辑仅使用决策时点可获得的数据。
        # 说明：以下逻辑仅使用决策时点可获得的数据。
        "atr_pct": snapshot.structure.atr_pct,
        "down": snapshot.structure.breadth_confirmation,
    }


def _empty_signals() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_id",
            "symbol",
            "action",
            "signal_time",
            "score",
            "reason",
            "reason_family",
        ],
    )
