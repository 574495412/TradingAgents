"""用无量纲市场评分和入场时波动率推进跨标的反趋势生命周期。"""

# 生命周期状态机集中保留顺序敏感的开仓、加仓和退出分支。
# ruff: noqa: C901, EM101, EM102, RUF002, TRY003

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from .policy import CountertrendFactorPolicy
from .scoring import build_countertrend_scores

MINIMUM_PERTURBATION_MULTIPLIER = 0.8
MAXIMUM_PERTURBATION_MULTIPLIER = 1.2


@dataclass(frozen=True, slots=True)
class VolatilityNormalizedCountertrendPolicy:
    """跨标的生命周期参数；价格距离只由入场时 ATR 和宽泛风险边界决定。"""

    maximum_adds: int = 3
    minimum_bars_between_adds: int = 6
    add_distance_atr_multiple: float = 1.0

    recovery_target_atr_multiple: float = 1.5
    minimum_recovery_target_pct: float = 0.40
    maximum_recovery_target_pct: float = 3.00

    hard_stop_atr_multiple: float = 3.0
    minimum_hard_stop_pct: float = 1.20
    maximum_hard_stop_pct: float = 6.00

    minimum_holding_bars: int = 6
    maximum_holding_bars: int = 144
    post_exit_cooldown_bars: int = 12
    round_trip_cost_bps: float = 14.0

    def with_distance_multiplier(
        self,
        multiplier: float,
    ) -> VolatilityNormalizedCountertrendPolicy:
        """为预先声明的稳健性检查等比例扰动全部 ATR 距离。"""

        if not (
            MINIMUM_PERTURBATION_MULTIPLIER
            <= multiplier
            <= MAXIMUM_PERTURBATION_MULTIPLIER
        ):
            message = "volatility-distance multiplier must be in [0.8, 1.2]"
            raise ValueError(message)
        return replace(
            self,
            add_distance_atr_multiple=(
                self.add_distance_atr_multiple * multiplier
            ),
            recovery_target_atr_multiple=(
                self.recovery_target_atr_multiple * multiplier
            ),
            hard_stop_atr_multiple=self.hard_stop_atr_multiple * multiplier,
        )


@dataclass(frozen=True)
class VolatilityNormalizedCountertrendResult:
    """一个标的的已平仓周期和逐笔因果动作。"""

    positions: pd.DataFrame
    actions: pd.DataFrame


def evaluate_volatility_normalized_countertrend(  # noqa: PLR0912, PLR0915
    market_panel: pd.DataFrame,
    *,
    factor_policy: CountertrendFactorPolicy | None = None,
    lifecycle_policy: VolatilityNormalizedCountertrendPolicy | None = None,
    execution_start: object | None = None,
    close_open_at_window_end: bool = False,
) -> VolatilityNormalizedCountertrendResult:
    """运行不读取标签、不含标的特例的跨资产反趋势状态机。

    入场使用既有的 ATR/排名/广度市场评分。止盈与止损距离在首次入场时
    冻结，后续只围绕实际成交均价平移；因此同一组无量纲参数可复用于
    不同价格和波动率尺度的标的。
    """

    factor = factor_policy or CountertrendFactorPolicy()
    lifecycle = lifecycle_policy or VolatilityNormalizedCountertrendPolicy()
    _validate_lifecycle_policy(lifecycle)
    market = market_panel.copy().sort_values(
        "signal_time",
        kind="stable",
    ).reset_index(drop=True)
    required_market = {
        "signal_time",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "atr14_pct",
    }
    missing_market = sorted(required_market.difference(market.columns))
    if missing_market:
        message = "normalized countertrend panel missing columns: "
        raise ValueError(message + ", ".join(missing_market))
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    if market["signal_time"].isna().any():
        raise ValueError("normalized countertrend panel contains invalid times")
    if market["signal_time"].duplicated().any():
        raise ValueError("normalized countertrend panel contains duplicate times")
    start = (
        _utc_timestamp(execution_start)
        if execution_start is not None
        else pd.Timestamp(market["signal_time"].min())
    )

    scores = build_countertrend_scores(market, policy=factor)
    by_time = _score_lookup(scores)
    active = {
        (action, mechanism): False
        for action in ("OPEN_LONG", "OPEN_SHORT")
        for mechanism in factor.enabled_mechanisms
    }
    position: dict[str, Any] | None = None
    positions: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    last_exit_index = -lifecycle.post_exit_cooldown_bars - 1
    chain_number = 0

    for bar_index, bar in market.iterrows():
        timestamp = pd.Timestamp(bar["signal_time"])
        current = by_time[timestamp]
        rising = _advance_score_states(
            active,
            current=current,
            factor_policy=factor,
        )
        if position is not None:
            exit_price, exit_reason = _risk_or_recovery_exit(
                bar,
                position=position,
                bar_index=bar_index,
                policy=lifecycle,
            )
            bars_held = bar_index - int(position["entry_index"])
            opposite = (
                "OPEN_SHORT"
                if position["action"] == "OPEN_LONG"
                else "OPEN_LONG"
            )
            opposite_transition = _best_rising_for_action(
                rising,
                action=opposite,
            )
            if (
                exit_price is None
                and bars_held >= lifecycle.minimum_holding_bars
                and opposite_transition is not None
            ):
                exit_price = float(bar["close"])
                exit_reason = "opposite_market_transition"
            if (
                exit_price is None
                and bars_held >= lifecycle.maximum_holding_bars
            ):
                exit_price = float(bar["close"])
                exit_reason = "maximum_holding_time"
            if exit_price is not None:
                positions.append(
                    _closed_position(
                        position,
                        exit_time=timestamp,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        holding_bars=bars_held,
                        round_trip_cost_bps=lifecycle.round_trip_cost_bps,
                    ),
                )
                actions.append(
                    _action_row(
                        position,
                        timestamp=timestamp,
                        operation_type="CLOSE",
                        price=exit_price,
                        reason=exit_reason,
                    ),
                )
                position = None
                last_exit_index = bar_index
                continue
            _maybe_add(
                bar,
                bar_index=bar_index,
                position=position,
                current=current,
                factor_policy=factor,
                lifecycle_policy=lifecycle,
                actions=actions,
            )
            continue

        if timestamp < start:
            continue
        if bar_index - last_exit_index < lifecycle.post_exit_cooldown_bars:
            continue
        candidate = _choose_rising_candidate(
            rising,
            minimum_directional_edge=factor.minimum_directional_edge,
        )
        if candidate is None:
            continue
        score, action, mechanism = candidate
        entry_atr_pct = float(bar["atr14_pct"])
        if not entry_atr_pct > 0.0:
            continue
        chain_number += 1
        price = float(bar["close"])
        position = _new_position(
            chain_id=f"normalized_countertrend_{chain_number:06d}",
            symbol=str(bar["symbol"]),
            action=action,
            mechanism=mechanism,
            timestamp=timestamp,
            bar_index=bar_index,
            price=price,
            score=score,
            entry_atr_pct=entry_atr_pct,
            policy=lifecycle,
        )
        actions.append(
            _action_row(
                position,
                timestamp=timestamp,
                operation_type="OPEN",
                price=price,
                reason=f"score_crossing:{mechanism}",
            ),
        )

    if position is not None and close_open_at_window_end and not market.empty:
        last = market.iloc[-1]
        exit_time = pd.Timestamp(last["signal_time"])
        exit_price = float(last["close"])
        holding_bars = len(market) - 1 - int(position["entry_index"])
        positions.append(
            _closed_position(
                position,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason="development_window_end",
                holding_bars=holding_bars,
                round_trip_cost_bps=lifecycle.round_trip_cost_bps,
            ),
        )
        actions.append(
            _action_row(
                position,
                timestamp=exit_time,
                operation_type="CLOSE",
                price=exit_price,
                reason="development_window_end",
            ),
        )
    return VolatilityNormalizedCountertrendResult(
        positions=pd.DataFrame.from_records(positions),
        actions=pd.DataFrame.from_records(actions),
    )


def _score_lookup(scores: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    frame = scores.copy().sort_values(
        ["signal_time", "action"],
        kind="stable",
    )
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
    counts = frame.groupby("signal_time")["action"].nunique()
    if not counts.eq(2).all():
        raise ValueError("normalized countertrend requires two directions per bar")
    return {
        pd.Timestamp(timestamp): group.set_index("action")
        for timestamp, group in frame.groupby("signal_time", sort=True)
    }


def _advance_score_states(
    active: dict[tuple[str, str], bool],
    *,
    current: pd.DataFrame,
    factor_policy: CountertrendFactorPolicy,
) -> list[tuple[float, str, str]]:
    rising: list[tuple[float, str, str]] = []
    for action, mechanism in active:
        score = float(current.loc[action, f"{mechanism}_score"])
        threshold = factor_policy.threshold(mechanism)
        key = (action, mechanism)
        if score < threshold - factor_policy.release_hysteresis:
            active[key] = False
        elif score >= threshold:
            was_active = active[key]
            active[key] = True
            if not was_active:
                rising.append((score, action, mechanism))
    return rising


def _best_rising_for_action(
    rising: list[tuple[float, str, str]],
    *,
    action: str,
) -> tuple[float, str, str] | None:
    candidates = [candidate for candidate in rising if candidate[1] == action]
    return max(candidates, default=None, key=lambda item: (item[0], item[2]))


def _choose_rising_candidate(
    rising: list[tuple[float, str, str]],
    *,
    minimum_directional_edge: float,
) -> tuple[float, str, str] | None:
    if not rising:
        return None
    ordered = sorted(rising, key=lambda item: (-item[0], item[1], item[2]))
    winner = ordered[0]
    opposite_scores = [
        score
        for score, action, _mechanism in ordered
        if action != winner[1]
    ]
    if (
        opposite_scores
        and winner[0] - max(opposite_scores) < minimum_directional_edge
    ):
        return None
    return winner


def _new_position(
    *,
    chain_id: str,
    symbol: str,
    action: str,
    mechanism: str,
    timestamp: pd.Timestamp,
    bar_index: int,
    price: float,
    score: float,
    entry_atr_pct: float,
    policy: VolatilityNormalizedCountertrendPolicy,
) -> dict[str, Any]:
    return {
        "chain_id": chain_id,
        "symbol": symbol,
        "action": action,
        "direction": 1.0 if action == "OPEN_LONG" else -1.0,
        "mechanism": mechanism,
        "entry_time": timestamp,
        "entry_index": bar_index,
        "entry_score": score,
        "entry_atr_pct": entry_atr_pct,
        "target_distance_pct": _bounded_atr_distance(
            entry_atr_pct,
            multiple=policy.recovery_target_atr_multiple,
            minimum=policy.minimum_recovery_target_pct,
            maximum=policy.maximum_recovery_target_pct,
        ),
        "stop_distance_pct": _bounded_atr_distance(
            entry_atr_pct,
            multiple=policy.hard_stop_atr_multiple,
            minimum=policy.minimum_hard_stop_pct,
            maximum=policy.maximum_hard_stop_pct,
        ),
        "fill_prices": [price],
        "fill_times": [timestamp],
        "last_fill_price": price,
        "last_fill_index": bar_index,
        "average_entry_price": price,
    }


def _risk_or_recovery_exit(  # noqa: PLR0911
    bar: pd.Series,
    *,
    position: dict[str, Any],
    bar_index: int,
    policy: VolatilityNormalizedCountertrendPolicy,
) -> tuple[float | None, str]:
    direction = float(position["direction"])
    average = float(position["average_entry_price"])
    stop = average * (
        1.0 - direction * float(position["stop_distance_pct"]) / 100.0
    )
    if direction > 0.0:
        if float(bar["open"]) <= stop:
            return float(bar["open"]), "gap_atr_hard_stop"
        if float(bar["low"]) <= stop:
            return stop, "intrabar_atr_hard_stop"
    else:
        if float(bar["open"]) >= stop:
            return float(bar["open"]), "gap_atr_hard_stop"
        if float(bar["high"]) >= stop:
            return stop, "intrabar_atr_hard_stop"
    if bar_index - int(position["entry_index"]) < policy.minimum_holding_bars:
        return None, ""
    target = average * (
        1.0 + direction * float(position["target_distance_pct"]) / 100.0
    )
    if direction > 0.0 and float(bar["high"]) >= target:
        return target, "atr_recovery_target"
    if direction < 0.0 and float(bar["low"]) <= target:
        return target, "atr_recovery_target"
    return None, ""


def _maybe_add(
    bar: pd.Series,
    *,
    bar_index: int,
    position: dict[str, Any],
    current: pd.DataFrame,
    factor_policy: CountertrendFactorPolicy,
    lifecycle_policy: VolatilityNormalizedCountertrendPolicy,
    actions: list[dict[str, object]],
) -> None:
    if len(position["fill_prices"]) - 1 >= lifecycle_policy.maximum_adds:
        return
    if (
        bar_index - int(position["last_fill_index"])
        < lifecycle_policy.minimum_bars_between_adds
    ):
        return
    action = str(position["action"])
    mechanism = str(position["mechanism"])
    own_score = float(current.loc[action, f"{mechanism}_score"])
    if own_score < factor_policy.threshold(mechanism):
        return
    atr_pct = float(bar["atr14_pct"])
    if not atr_pct > 0.0:
        return
    direction = float(position["direction"])
    adverse_move = -direction * (
        float(bar["close"]) / float(position["last_fill_price"]) - 1.0
    ) * 100.0
    if (
        adverse_move
        < atr_pct * lifecycle_policy.add_distance_atr_multiple
    ):
        return
    price = float(bar["close"])
    position["fill_prices"].append(price)
    position["fill_times"].append(pd.Timestamp(bar["signal_time"]))
    position["last_fill_price"] = price
    position["last_fill_index"] = bar_index
    position["average_entry_price"] = sum(position["fill_prices"]) / len(
        position["fill_prices"],
    )
    actions.append(
        _action_row(
            position,
            timestamp=pd.Timestamp(bar["signal_time"]),
            operation_type="ADD",
            price=price,
            reason=f"atr_adverse_add:{mechanism}",
        ),
    )


def _closed_position(
    position: dict[str, Any],
    *,
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    holding_bars: int,
    round_trip_cost_bps: float,
) -> dict[str, object]:
    average = float(position["average_entry_price"])
    direction = float(position["direction"])
    gross = direction * (exit_price / average - 1.0) * 100.0
    point_gross = direction * (
        exit_price / float(position["fill_prices"][0]) - 1.0
    ) * 100.0
    fill_count = len(position["fill_prices"])
    cost_pct = round_trip_cost_bps / 100.0 * fill_count
    return {
        "chain_id": position["chain_id"],
        "symbol": position["symbol"],
        "direction": "LONG" if direction > 0.0 else "SHORT",
        "action": position["action"],
        "mechanism": position["mechanism"],
        "open_time": position["entry_time"],
        "close_time": exit_time,
        "open_price": float(position["fill_prices"][0]),
        "close_price": float(exit_price),
        "average_entry_price": average,
        "exit_price": float(exit_price),
        "fill_count": fill_count,
        "add_count": fill_count - 1,
        "fill_times": "|".join(
            pd.Timestamp(value).isoformat()
            for value in position["fill_times"]
        ),
        "entry_score": float(position["entry_score"]),
        "entry_atr_pct": float(position["entry_atr_pct"]),
        "target_distance_pct": float(position["target_distance_pct"]),
        "stop_distance_pct": float(position["stop_distance_pct"]),
        "holding_bars": int(holding_bars),
        "exit_reason": exit_reason,
        "point_roi_pct": point_gross,
        "gross_return_pct": gross,
        "net_return_pct": gross - cost_pct,
        "round_trip_cost_bps": float(round_trip_cost_bps),
    }


def _action_row(
    position: dict[str, Any],
    *,
    timestamp: pd.Timestamp,
    operation_type: str,
    price: float,
    reason: str,
) -> dict[str, object]:
    direction = "LONG" if float(position["direction"]) > 0.0 else "SHORT"
    action_prefix = "OPEN" if operation_type in {"OPEN", "ADD"} else "CLOSE"
    return {
        "signal_id": (
            f"{position['chain_id']}:{operation_type}:"
            f"{pd.Timestamp(timestamp).isoformat()}"
        ),
        "chain_id": position["chain_id"],
        "symbol": position["symbol"],
        "action": f"{action_prefix}_{direction}",
        "operation_type": operation_type,
        "signal_time": pd.Timestamp(timestamp),
        "signal_price": float(price),
        "mechanism": position["mechanism"],
        "entry_score": float(position["entry_score"]),
        "entry_atr_pct": float(position["entry_atr_pct"]),
        "target_distance_pct": float(position["target_distance_pct"]),
        "stop_distance_pct": float(position["stop_distance_pct"]),
        "reason": reason,
    }


def _bounded_atr_distance(
    atr_pct: float,
    *,
    multiple: float,
    minimum: float,
    maximum: float,
) -> float:
    return min(float(maximum), max(float(minimum), float(atr_pct) * multiple))


def _validate_lifecycle_policy(
    policy: VolatilityNormalizedCountertrendPolicy,
) -> None:
    if policy.maximum_adds < 0:
        raise ValueError("normalized countertrend maximum adds cannot be negative")
    if policy.minimum_bars_between_adds < 1:
        raise ValueError("normalized countertrend add gap must be positive")
    if policy.minimum_holding_bars < 0:
        raise ValueError("normalized countertrend minimum holding cannot be negative")
    if policy.maximum_holding_bars <= policy.minimum_holding_bars:
        message = "normalized countertrend maximum holding must exceed minimum"
        raise ValueError(message)
    if policy.post_exit_cooldown_bars < 0:
        raise ValueError("normalized countertrend cooldown cannot be negative")
    for label, value in (
        ("add ATR multiple", policy.add_distance_atr_multiple),
        ("target ATR multiple", policy.recovery_target_atr_multiple),
        ("stop ATR multiple", policy.hard_stop_atr_multiple),
    ):
        if value <= 0.0:
            raise ValueError(f"normalized countertrend {label} must be positive")
    for label, minimum, maximum in (
        (
            "target distance",
            policy.minimum_recovery_target_pct,
            policy.maximum_recovery_target_pct,
        ),
        (
            "stop distance",
            policy.minimum_hard_stop_pct,
            policy.maximum_hard_stop_pct,
        ),
    ):
        if minimum <= 0.0 or maximum <= minimum:
            message = f"normalized countertrend {label} bounds are invalid"
            raise ValueError(message)
    if policy.round_trip_cost_bps < 0.0:
        raise ValueError("normalized countertrend cost cannot be negative")


def _utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("Asia/Shanghai").tz_convert("UTC")
    return timestamp.tz_convert("UTC")


__all__ = [
    "VolatilityNormalizedCountertrendPolicy",
    "VolatilityNormalizedCountertrendResult",
    "evaluate_volatility_normalized_countertrend",
]
