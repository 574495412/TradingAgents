"""把反转评分推进为开仓、加仓、退出和冷却动作的状态机。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .policy import ReversalLifecyclePolicy
from .scoring import build_reversal_lifecycle_scores


def evaluate_reversal_lifecycle(
    market_panel: pd.DataFrame,
    *,
    policy: ReversalLifecyclePolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """evaluate_reversal_lifecycle：计算可解释的市场结构因子。"""

    scores = build_reversal_lifecycle_scores(market_panel)
    return evaluate_scored_reversal_lifecycle(
        market_panel,
        scores,
        policy=policy,
    )


def evaluate_scored_reversal_lifecycle(  # noqa: PLR0915
    market_panel: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    policy: ReversalLifecyclePolicy | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """evaluate_scored_reversal_lifecycle：计算可解释的市场结构因子。"""

    cfg = policy or ReversalLifecyclePolicy()
    market = market_panel.copy().sort_values("signal_time", kind="stable").reset_index(drop=True)
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    scores = scores.copy().sort_values(
        ["signal_time", "action"],
        kind="stable",
    ).reset_index(drop=True)
    scores["signal_time"] = pd.to_datetime(scores["signal_time"], utc=True)
    expected_times = set(market["signal_time"])
    score_times = set(scores["signal_time"])
    if score_times != expected_times:
        message = "scored lifecycle times must exactly match market panel times"
        raise ValueError(message)
    counts = scores.groupby("signal_time")["action"].nunique()
    if not counts.eq(2).all():
        message = "scored lifecycle requires two directions per bar"
        raise ValueError(message)
    by_time = {
        pd.Timestamp(timestamp): group.set_index("action")
        for timestamp, group in scores.groupby("signal_time", sort=True)
    }
    setup = {
        action: {"bars": 0, "active": False}
        for action in ("OPEN_LONG", "OPEN_SHORT")
    }
    recovery_active = False
    position: dict[str, Any] | None = None
    cycles: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    last_exit_index = -cfg.post_exit_cooldown_bars - 1
    chain_number = 0
    for bar_index, bar in market.iterrows():
        timestamp = pd.Timestamp(bar["signal_time"])
        current = by_time[timestamp]
        entries = {
            action: float(current.loc[action, "entry_score"])
            for action in setup
        }
        candidates = _advance_setups(setup, current=current, policy=cfg)
        if position is not None:
            closed, flip_action, recovery_active = _advance_position(
                bar,
                bar_index=bar_index,
                position=position,
                current=current,
                entries=entries,
                recovery_active=recovery_active,
                policy=cfg,
                actions=actions,
            )
            if closed is not None:
                cycles.append(closed)
                position = None
                last_exit_index = bar_index
                if flip_action:
                    chain_number += 1
                    position = _new_position(
                        chain_id=f"reversal_chain_{chain_number:06d}",
                        symbol=str(bar["symbol"]),
                        action=flip_action,
                        timestamp=timestamp,
                        bar_index=bar_index,
                        price=float(bar["close"]),
                        score=float(current.loc[flip_action, "entry_score"]),
                        mechanism=str(current.loc[flip_action, "entry_mechanism"]),
                    )
                    actions.append(_action_row(position, timestamp, "OPEN", float(bar["close"]), 1.0))
                continue
            continue
        if bar_index - last_exit_index < cfg.post_exit_cooldown_bars or not candidates:
            continue
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < cfg.minimum_directional_edge:
            continue
        score, action = candidates[0]
        chain_number += 1
        position = _new_position(
            chain_id=f"reversal_chain_{chain_number:06d}",
            symbol=str(bar["symbol"]),
            action=action,
            timestamp=timestamp,
            bar_index=bar_index,
            price=float(bar["close"]),
            score=score,
            mechanism=str(current.loc[action, "entry_mechanism"]),
        )
        recovery_active = False
        actions.append(_action_row(position, timestamp, "OPEN", float(bar["close"]), 1.0))

    if position is not None and not market.empty:
        last = market.iloc[-1]
        closed = _close_position(
            position,
            timestamp=pd.Timestamp(last["signal_time"]),
            price=float(last["close"]),
            reason="window_end",
            holding_bars=len(market) - 1 - int(position["entry_index"]),
            policy=cfg,
            actions=actions,
        )
        cycles.append(closed)
    return pd.DataFrame.from_records(cycles), pd.DataFrame.from_records(actions)


def _advance_setups(
    setup: dict[str, dict[str, object]],
    *,
    current: pd.DataFrame,
    policy: ReversalLifecyclePolicy,
) -> list[tuple[float, str]]:
    """推进各方向待确认设置，并丢弃超时或失效候选。"""

    candidates: list[tuple[float, str]] = []
    for action, state in setup.items():
        score = float(current.loc[action, "entry_score"])
        if score < policy.entry_threshold - policy.entry_release_hysteresis:
            state["bars"] = 0
            state["active"] = False
            continue
        if score >= policy.entry_threshold:
            state["bars"] = int(state["bars"]) + 1
            if (
                not bool(state["active"])
                and policy.minimum_setup_bars <= int(state["bars"]) <= policy.maximum_setup_bars
            ):
                candidates.append((score, action))
                state["active"] = True
    return candidates


def _advance_position(  # noqa: C901, PLR0912
    bar: pd.Series,
    *,
    bar_index: int,
    position: dict[str, Any],
    current: pd.DataFrame,
    entries: dict[str, float],
    recovery_active: bool,
    policy: ReversalLifecyclePolicy,
    actions: list[dict[str, object]],
) -> tuple[dict[str, object] | None, str, bool]:
    """依据当前评分和持仓年龄推进已有仓位。"""

    timestamp = pd.Timestamp(bar["signal_time"])
    action = str(position["action"])
    direction = float(position["direction"])
    bars_held = bar_index - int(position["entry_index"])
    average = float(position["average_entry_price"])
    stop = average * (1.0 - direction * policy.hard_stop_pct / 100.0)
    stop_price: float | None = None
    if direction > 0.0:
        if float(bar["open"]) <= stop:
            stop_price = float(bar["open"])
        elif float(bar["low"]) <= stop:
            stop_price = stop
    elif float(bar["open"]) >= stop:
        stop_price = float(bar["open"])
    elif float(bar["high"]) >= stop:
        stop_price = stop
    if stop_price is not None:
        return (
            _close_position(
                position,
                timestamp=timestamp,
                price=stop_price,
                reason="hard_stop",
                holding_bars=bars_held,
                policy=policy,
                actions=actions,
            ),
            "",
            False,
        )

    own = current.loc[action]
    recovery = float(own["recovery_score"])
    if recovery < policy.reduce_score_floor - policy.recovery_release_hysteresis:
        recovery_active = False
    rising_recovery = recovery >= policy.reduce_score_floor and not recovery_active
    if recovery >= policy.reduce_score_floor:
        recovery_active = True

    opposite = "OPEN_SHORT" if action == "OPEN_LONG" else "OPEN_LONG"
    opposite_score = entries[opposite]
    own_score = entries[action]
    gross = direction * (float(bar["close"]) / average - 1.0) * 100.0
    if bars_held >= policy.minimum_holding_bars:
        # 满足以下市场结构条件时进入该分支。
        # 满足以下市场结构条件时进入该分支。
        if (
            rising_recovery
            and int(position["reductions"]) < policy.maximum_reductions
            and bar_index - int(position["last_reduction_index"])
            >= policy.minimum_bars_between_reductions
        ):
            fraction = min(0.25, float(position["remaining_fraction"]) - 0.25)
            if fraction > 0.0:
                position["remaining_fraction"] -= fraction
                position["reductions"] += 1
                position["last_reduction_index"] = bar_index
                position["realized_weighted_gross"] += fraction * gross
                actions.append(
                    _action_row(position, timestamp, "REDUCE", float(bar["close"]), fraction),
                )
                return None, "", recovery_active
        confirmed_close = bool(
            recovery >= policy.close_score_floor
            and (
                int(position["reductions"]) > 0
                or gross >= policy.recovery_profit_floor_pct
            ),
        )
        opposite_transition = bool(
            opposite_score >= policy.entry_threshold
            and opposite_score - own_score >= policy.minimum_directional_edge,
        )
        final_target = gross >= policy.final_profit_target_pct and recovery >= policy.reduce_score_floor
        if confirmed_close or opposite_transition or final_target:
            reason = (
                "opposite_regime_transition"
                if opposite_transition
                else "confirmed_market_recovery"
                if confirmed_close
                else "recovery_profit_target"
            )
            closed = _close_position(
                position,
                timestamp=timestamp,
                price=float(bar["close"]),
                reason=reason,
                holding_bars=bars_held,
                policy=policy,
                actions=actions,
            )
            return closed, opposite if opposite_transition else "", False
    if bars_held >= policy.maximum_holding_bars:
        return (
            _close_position(
                position,
                timestamp=timestamp,
                price=float(bar["close"]),
                reason="maximum_holding_time",
                holding_bars=bars_held,
                policy=policy,
                actions=actions,
            ),
            "",
            False,
        )
    _maybe_add(
        bar,
        bar_index=bar_index,
        position=position,
        entry_score=own_score,
        policy=policy,
        actions=actions,
    )
    return None, "", recovery_active


def _maybe_add(
    bar: pd.Series,
    *,
    bar_index: int,
    position: dict[str, Any],
    entry_score: float,
    policy: ReversalLifecyclePolicy,
    actions: list[dict[str, object]],
) -> None:
    """在恢复确认、库存和时间间隔满足时生成一次加仓。"""

    if len(position["fill_prices"]) - 1 >= policy.maximum_adds:
        return
    if bar_index - int(position["last_fill_index"]) < policy.minimum_bars_between_adds:
        return
    if entry_score < policy.add_score_floor:
        return
    adverse = -float(position["direction"]) * (
        float(bar["close"]) / float(position["last_fill_price"]) - 1.0
    ) * 100.0
    if adverse < float(bar["atr14_pct"]) * policy.add_distance_atr_multiple:
        return
    price = float(bar["close"])
    position["fill_prices"].append(price)
    position["last_fill_price"] = price
    position["last_fill_index"] = bar_index
    position["average_entry_price"] = sum(position["fill_prices"]) / len(position["fill_prices"])
    actions.append(
        _action_row(
            position,
            pd.Timestamp(bar["signal_time"]),
            "ADD",
            price,
            1.0,
        ),
    )


def _new_position(
    *,
    chain_id: str,
    symbol: str,
    action: str,
    timestamp: pd.Timestamp,
    bar_index: int,
    price: float,
    score: float,
    mechanism: str,
) -> dict[str, Any]:
    """由已确认设置创建新的反转持仓状态。"""

    return {
        "chain_id": chain_id,
        "symbol": symbol,
        "action": action,
        "direction": 1.0 if action == "OPEN_LONG" else -1.0,
        "mechanism": mechanism,
        "entry_time": timestamp,
        "entry_index": bar_index,
        "entry_score": score,
        "fill_prices": [price],
        "last_fill_price": price,
        "last_fill_index": bar_index,
        "average_entry_price": price,
        "remaining_fraction": 1.0,
        "reductions": 0,
        "last_reduction_index": -10_000,
        "realized_weighted_gross": 0.0,
    }


def _close_position(
    position: dict[str, Any],
    *,
    timestamp: pd.Timestamp,
    price: float,
    reason: str,
    holding_bars: int,
    policy: ReversalLifecyclePolicy,
    actions: list[dict[str, object]],
) -> dict[str, object]:
    """生成平仓动作并清空对应的持仓状态。"""

    gross = float(position["direction"]) * (
        price / float(position["average_entry_price"]) - 1.0
    ) * 100.0
    remaining = float(position["remaining_fraction"])
    position["realized_weighted_gross"] += remaining * gross
    actions.append(_action_row(position, timestamp, "CLOSE", price, remaining))
    weighted = float(position["realized_weighted_gross"])
    fills = len(position["fill_prices"])
    cost_pct = policy.round_trip_cost_bps / 100.0 * fills
    return {
        "chain_id": position["chain_id"],
        "entry_time": position["entry_time"],
        "exit_time": timestamp,
        "action": position["action"],
        "mechanism": position["mechanism"],
        "fill_count": fills,
        "reduction_count": int(position["reductions"]),
        "holding_bars": int(holding_bars),
        "average_entry_price": float(position["average_entry_price"]),
        "exit_price": float(price),
        "exit_reason": reason,
        "gross_return_pct": weighted,
        "net_return_pct": weighted - cost_pct,
    }


def _action_row(
    position: dict[str, Any],
    timestamp: pd.Timestamp,
    operation_type: str,
    price: float,
    fraction: float,
) -> dict[str, object]:
    """将状态转换编码为标准、可审计的动作记录。"""

    direction = "LONG" if float(position["direction"]) > 0.0 else "SHORT"
    prefix = "OPEN" if operation_type in {"OPEN", "ADD"} else "CLOSE"
    return {
        "signal_id": f"{position['chain_id']}:{operation_type}:{timestamp.isoformat()}",
        "chain_id": position["chain_id"],
        "symbol": str(position["symbol"]),
        "action": f"{prefix}_{direction}",
        "operation_type": operation_type,
        "signal_time": timestamp,
        "signal_price": float(price),
        "position_fraction": float(fraction),
        "reason": (
            f"market_reversal_lifecycle;mechanism={position['mechanism']};"
            f"entry_score={float(position['entry_score']):.4f}"
        ),
    }


__all__ = [
    "evaluate_reversal_lifecycle",
    "evaluate_scored_reversal_lifecycle",
]
