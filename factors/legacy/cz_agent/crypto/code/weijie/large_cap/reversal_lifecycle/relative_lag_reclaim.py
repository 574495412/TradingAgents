"""用相对落后与市场广度回收识别 ETH 中等回撤后的再入场。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .components import directional_market_components

RELATIVE_LAG_RECLAIM = "relative_lag_reclaim"
_DIRECTIONS = ((1.0, "LONG"), (-1.0, "SHORT"))
_MINIMUM_THRESHOLD_MULTIPLIER = 0.8
_MAXIMUM_THRESHOLD_MULTIPLIER = 1.2


@dataclass(frozen=True, slots=True)
class RelativeLagReclaimPolicy:
    """相对落后设置、同步回收确认和退出边界。"""

    setup_minimum_adverse_mid_return_pct: float = 0.75
    setup_minimum_adverse_long_return_pct: float = 1.50
    setup_maximum_relative_momentum_pct: float = -0.25
    setup_minimum_adx: float = 35.0
    setup_maximum_location: float = 0.45
    setup_maximum_quote_volume_ratio: float = 2.0
    confirmation_timeout_bars: int = 12
    reclaim_location: float = 0.45
    confirmation_minimum_tiny_return_pct: float = 0.08
    confirmation_minimum_adverse_mid_return_pct: float = 0.80
    confirmation_minimum_adverse_long_return_pct: float = 1.20
    confirmation_minimum_breadth3: float = 2.0 / 3.0
    confirmation_minimum_breadth20: float = 2.0 / 3.0
    confirmation_maximum_breadth20: float = 0.75
    recovery_target_pct: float = 1.50
    armed_exit_minimum_adverse_mid_return_pct: float = 0.50
    armed_exit_minimum_adverse_breadth20: float = 1.0
    minimum_holding_bars: int = 6
    maximum_holding_bars: int = 576

    def with_threshold_multiplier(
        self,
        multiplier: float,
    ) -> RelativeLagReclaimPolicy:
        """对连续强度门槛做统一扰动; 离散广度状态保持不变。"""

        if not (_MINIMUM_THRESHOLD_MULTIPLIER <= multiplier <= _MAXIMUM_THRESHOLD_MULTIPLIER):
            message = "relative lag reclaim multiplier must be in [0.8, 1.2]"
            raise ValueError(message)
        return replace(
            self,
            setup_minimum_adverse_mid_return_pct=(self.setup_minimum_adverse_mid_return_pct * multiplier),
            setup_minimum_adverse_long_return_pct=(self.setup_minimum_adverse_long_return_pct * multiplier),
            setup_maximum_relative_momentum_pct=(self.setup_maximum_relative_momentum_pct * multiplier),
            setup_minimum_adx=self.setup_minimum_adx * multiplier,
            confirmation_minimum_tiny_return_pct=(self.confirmation_minimum_tiny_return_pct * multiplier),
            confirmation_minimum_adverse_mid_return_pct=(self.confirmation_minimum_adverse_mid_return_pct * multiplier),
            confirmation_minimum_adverse_long_return_pct=(
                self.confirmation_minimum_adverse_long_return_pct * multiplier
            ),
            recovery_target_pct=self.recovery_target_pct * multiplier,
            armed_exit_minimum_adverse_mid_return_pct=(
                self.armed_exit_minimum_adverse_mid_return_pct * multiplier
            ),
        )


def build_relative_lag_reclaim_actions(
    market_panel: pd.DataFrame,
    *,
    policy: RelativeLagReclaimPolicy | None = None,
) -> pd.DataFrame:
    """从闭合行情构建方向对称的相对落后回收活动。"""

    cfg = policy or RelativeLagReclaimPolicy()
    _validate_policy(cfg)
    market = _prepare_market(market_panel)
    candidates = _entry_candidates(market, policy=cfg)
    records: list[dict[str, object]] = []
    blocked_until = -1
    for candidate in candidates:
        entry_index = int(candidate["bar_index"])
        if entry_index <= blocked_until:
            continue
        direction = float(candidate["direction"])
        direction_name = str(candidate["direction_name"])
        entry_bar = market.iloc[entry_index]
        entry_time = pd.Timestamp(entry_bar["signal_time"])
        entry_price = float(entry_bar["close"])
        chain_id = f"relative_lag_reclaim:{direction_name.lower()}:{entry_time.isoformat()}"
        score = float(candidate["score"])
        records.append(
            _action_row(
                chain_id=chain_id,
                symbol=str(entry_bar["symbol"]),
                direction_name=direction_name,
                operation_type="OPEN",
                signal_time=entry_time,
                signal_price=entry_price,
                score=score,
                reason_detail="phase=price_and_breadth_reclaim",
            ),
        )
        exit_bar, exit_price, exit_reason = _first_exit(
            market,
            entry_index=entry_index,
            entry_price=entry_price,
            direction=direction,
            policy=cfg,
        )
        if exit_bar is None:
            blocked_until = len(market) - 1
            continue
        exit_index = int(exit_bar.name)
        blocked_until = exit_index
        records.append(
            _action_row(
                chain_id=chain_id,
                symbol=str(entry_bar["symbol"]),
                direction_name=direction_name,
                operation_type="CLOSE",
                signal_time=pd.Timestamp(exit_bar["signal_time"]),
                signal_price=exit_price,
                score=score,
                reason_detail=f"exit={exit_reason}",
            ),
        )
    return pd.DataFrame.from_records(records, columns=_ACTION_COLUMNS)


def _entry_candidates(
    market: pd.DataFrame,
    *,
    policy: RelativeLagReclaimPolicy,
) -> list[dict[str, object]]:
    states = {direction: {"armed_index": None} for direction, _ in _DIRECTIONS}
    candidates: list[dict[str, object]] = []
    for bar_index, bar in market.iterrows():
        if not bool(bar.get("unified_market_data_valid", True)):
            continue
        for direction, direction_name in _DIRECTIONS:
            directional = directional_market_components(bar, direction)
            state = states[direction]
            if _is_setup(bar, directional=directional, policy=policy):
                state["armed_index"] = bar_index
            armed_index = state["armed_index"]
            if armed_index is None:
                continue
            if bar_index - int(armed_index) > policy.confirmation_timeout_bars:
                state["armed_index"] = None
                continue
            if not _is_confirmation(
                market,
                bar_index=bar_index,
                direction=direction,
                directional=directional,
                policy=policy,
            ):
                continue
            candidates.append(
                {
                    "bar_index": bar_index,
                    "direction": direction,
                    "direction_name": direction_name,
                    "score": _entry_score(
                        bar,
                        directional=directional,
                        policy=policy,
                    ),
                },
            )
            state["armed_index"] = None
    return sorted(
        candidates,
        key=lambda candidate: (
            int(candidate["bar_index"]),
            -float(candidate["score"]),
            -float(candidate["direction"]),
        ),
    )


def _is_setup(
    bar: pd.Series,
    *,
    directional: dict[str, float],
    policy: RelativeLagReclaimPolicy,
) -> bool:
    return bool(
        directional["adverse_mid_momentum"] >= policy.setup_minimum_adverse_mid_return_pct
        and directional["adverse_long_momentum"] >= policy.setup_minimum_adverse_long_return_pct
        and directional["relative_momentum"] <= policy.setup_maximum_relative_momentum_pct
        and float(bar["adx14"]) >= policy.setup_minimum_adx
        and directional["location"] < policy.setup_maximum_location
        and float(bar["quote_volume_ratio_med30"]) <= policy.setup_maximum_quote_volume_ratio,
    )


def _is_confirmation(
    market: pd.DataFrame,
    *,
    bar_index: int,
    direction: float,
    directional: dict[str, float],
    policy: RelativeLagReclaimPolicy,
) -> bool:
    if bar_index <= 0:
        return False
    previous = directional_market_components(
        market.iloc[bar_index - 1],
        direction,
    )
    return bool(
        previous["location"] < policy.reclaim_location <= directional["location"]
        and previous["favorable_breadth20"]
        < policy.confirmation_minimum_breadth20
        <= directional["favorable_breadth20"]
        <= policy.confirmation_maximum_breadth20
        and directional["tiny_return"] >= policy.confirmation_minimum_tiny_return_pct
        and directional["adverse_mid_momentum"] >= policy.confirmation_minimum_adverse_mid_return_pct
        and directional["adverse_long_momentum"] >= policy.confirmation_minimum_adverse_long_return_pct
        and directional["favorable_breadth3"] >= policy.confirmation_minimum_breadth3,
    )


def _entry_score(
    bar: pd.Series,
    *,
    directional: dict[str, float],
    policy: RelativeLagReclaimPolicy,
) -> float:
    mid = min(
        1.0,
        directional["adverse_mid_momentum"] / max(policy.setup_minimum_adverse_mid_return_pct * 2.0, 1e-9),
    )
    long = min(
        1.0,
        directional["adverse_long_momentum"] / max(policy.setup_minimum_adverse_long_return_pct * 2.0, 1e-9),
    )
    relative = min(
        1.0,
        abs(directional["relative_momentum"]) / max(abs(policy.setup_maximum_relative_momentum_pct) * 4.0, 1e-9),
    )
    trend = min(1.0, float(bar["adx14"]) / 50.0)
    return float(0.50 + 0.50 * (mid + long + relative + trend) / 4.0)


def _first_exit(
    market: pd.DataFrame,
    *,
    entry_index: int,
    entry_price: float,
    direction: float,
    policy: RelativeLagReclaimPolicy,
) -> tuple[pd.Series | None, float, str]:
    future = market.iloc[entry_index + 1 : entry_index + 1 + policy.maximum_holding_bars]
    recovery_armed = False
    for future_index, bar in future.iterrows():
        bars_held = int(future_index) - entry_index
        favorable_move_pct = direction * (float(bar["close"]) / entry_price - 1.0) * 100.0
        if (
            bars_held >= policy.minimum_holding_bars
            and favorable_move_pct >= policy.recovery_target_pct
        ):
            recovery_armed = True
        if not recovery_armed:
            continue
        directional = directional_market_components(bar, direction)
        if (
            directional["adverse_mid_momentum"]
            >= policy.armed_exit_minimum_adverse_mid_return_pct
            and directional["adverse_breadth20"]
            >= policy.armed_exit_minimum_adverse_breadth20
        ):
            return bar, float(bar["close"]), "armed_breadth_reversal"
    if len(future) >= policy.maximum_holding_bars:
        return future.iloc[-1], float(future.iloc[-1]["close"]), "maximum_holding"
    return None, float("nan"), ""


def _action_row(
    *,
    chain_id: str,
    symbol: str,
    direction_name: str,
    operation_type: str,
    signal_time: pd.Timestamp,
    signal_price: float,
    score: float,
    reason_detail: str,
) -> dict[str, object]:
    action = "OPEN" if operation_type == "OPEN" else "CLOSE"
    return {
        "signal_id": (f"{chain_id}:{operation_type}:{signal_time.isoformat()}"),
        "chain_id": chain_id,
        "symbol": symbol,
        "action": f"{action}_{direction_name}",
        "operation_type": operation_type,
        "signal_time": signal_time,
        "signal_price": float(signal_price),
        "position_fraction": 1.0,
        "decision_time": signal_time,
        "reason_family": RELATIVE_LAG_RECLAIM,
        "reason": (f"unified_market_regime;mechanism={RELATIVE_LAG_RECLAIM};{reason_detail}"),
        "score": float(score),
    }


def _prepare_market(market_panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        "signal_time",
        "symbol",
        "close",
        "ret3",
        "ret60",
        "tf1h_ret12",
        "rel_ret60",
        "donchian60_pos",
        "adx14",
        "quote_volume_ratio_med30",
        "market_up_breadth3",
        "market_down_breadth3",
        "market_up_breadth20",
        "market_down_breadth20",
    }
    missing = sorted(required.difference(market_panel.columns))
    if missing:
        raise ValueError(
            "relative lag reclaim market missing columns: " + ", ".join(missing),
        )
    market = market_panel.copy()
    market["signal_time"] = pd.to_datetime(market["signal_time"], utc=True)
    return market.sort_values("signal_time", kind="stable").reset_index(drop=True)


def _validate_policy(policy: RelativeLagReclaimPolicy) -> None:
    if policy.confirmation_timeout_bars <= 0:
        message = "relative lag reclaim timeout must be positive"
        raise ValueError(message)
    if not (
        0.0 < policy.reclaim_location <= 1.0
        and 0.0 <= policy.confirmation_minimum_breadth20 <= policy.confirmation_maximum_breadth20 <= 1.0
    ):
        message = "relative lag reclaim confirmation range is invalid"
        raise ValueError(message)
    if not (0 < policy.minimum_holding_bars <= policy.maximum_holding_bars):
        message = "relative lag reclaim holding range is invalid"
        raise ValueError(message)
    if (
        policy.armed_exit_minimum_adverse_mid_return_pct <= 0.0
        or not 0.0 < policy.armed_exit_minimum_adverse_breadth20 <= 1.0
    ):
        message = "relative lag reclaim armed exit range is invalid"
        raise ValueError(message)


_ACTION_COLUMNS = (
    "signal_id",
    "chain_id",
    "symbol",
    "action",
    "operation_type",
    "signal_time",
    "signal_price",
    "position_fraction",
    "decision_time",
    "reason_family",
    "reason",
    "score",
)

__all__ = [
    "RELATIVE_LAG_RECLAIM",
    "RelativeLagReclaimPolicy",
    "build_relative_lag_reclaim_actions",
]
