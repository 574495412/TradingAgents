"""大盘标的市场结构模块：提供 lifecycle 因子逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from factors.crypto.code.weijie.large_cap.eligibility import (
    eligible_market_support,
)
from factors.crypto.code.weijie.large_cap.mechanisms import (
    score_large_cap_mechanisms,
)
from factors.crypto.code.weijie.large_cap.policy import LargeCapRegimePolicy


@dataclass
class _VirtualPosition:
    """_VirtualPosition：封装相关市场结构数据与行为。"""

    action: str
    mechanism: str
    entry_time: pd.Timestamp
    entry_price: float
    entry_atr_pct: float
    stop_distance_pct: float
    held_bars: int = 0


def build_large_cap_lifecycle(
    feature_rows: pd.DataFrame,
    entry_candidates: pd.DataFrame,
    *,
    policy: LargeCapRegimePolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把大盘开平仓信号配对为可审计的完整持仓生命周期。"""

    if feature_rows is None or feature_rows.empty or entry_candidates.empty:
        return entry_candidates.copy(), _empty_close_signals()
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    features["symbol"] = features.get(
        "symbol",
        pd.Series("", index=features.index),
    ).astype(str)
    entries = entry_candidates.copy()
    entries["signal_time"] = pd.to_datetime(entries.get("signal_time"), errors="coerce")
    entries["symbol"] = entries.get(
        "symbol",
        pd.Series("", index=entries.index),
    ).astype(str)

    accepted_entry_indices: list[int] = []
    close_rows: list[dict[str, Any]] = []
    for symbol, symbol_features in features.groupby("symbol", sort=False):
        ordered = symbol_features.dropna(subset=["bar_time"]).sort_values(
            "bar_time",
            kind="stable",
        )
        symbol_entries = entries[entries["symbol"].eq(symbol)].copy()
        if ordered.empty or symbol_entries.empty:
            continue
        entries_by_time = {
            pd.Timestamp(signal_time): group
            for signal_time, group in symbol_entries.groupby("signal_time", sort=False)
            if pd.notna(signal_time)
        }
        position: _VirtualPosition | None = None
        for _, row in ordered.iterrows():
            bar_time = pd.Timestamp(row["bar_time"])
            bar_entries = entries_by_time.get(bar_time)
            if bar_entries is not None and not bar_entries.empty:
                candidate = bar_entries.sort_values("score", ascending=False).iloc[0]
                candidate_action = str(candidate.get("action") or "")
                if position is not None and candidate_action != position.action:
                    close_rows.append(
                        _close_row(
                            row,
                            position=position,
                            close_time=bar_time,
                            exit_family="large_cap_opposite_regime_exit",
                        ),
                    )
                    position = None
                if position is None:
                    accepted_entry_indices.append(int(candidate.name))
                    position = _new_position(candidate, row, policy=policy)
                    continue
                # 满足以下市场结构条件时进入该分支。
                # 满足以下市场结构条件时进入该分支。
            if position is None:
                continue
            position.held_bars += 1
            exit_family = _exit_family(row, position=position, policy=policy)
            if exit_family:
                close_rows.append(
                    _close_row(
                        row,
                        position=position,
                        close_time=bar_time,
                        exit_family=exit_family,
                    ),
                )
                position = None

    accepted = entries.loc[accepted_entry_indices].copy()
    if not accepted.empty:
        accepted = accepted.sort_values(
            ["signal_time", "symbol", "action"],
            kind="stable",
        ).reset_index(drop=True)
        accepted["signal_id"] = [
            f"WJLCOPEN{index + 1:06d}" for index in range(len(accepted))
        ]
    closes = pd.DataFrame(close_rows)
    if closes.empty:
        closes = _empty_close_signals()
    else:
        closes = closes.sort_values(
            ["signal_time", "symbol", "action"],
            kind="stable",
        ).reset_index(drop=True)
        closes["signal_id"] = [
            f"WJLCCLOSE{index + 1:06d}" for index in range(len(closes))
        ]
    return accepted, closes


def _new_position(
    candidate: pd.Series,
    row: pd.Series,
    *,
    policy: LargeCapRegimePolicy,
) -> _VirtualPosition:
    reason = str(candidate.get("reason") or "")
    marker = "mechanism="
    mechanism = (
        reason.split(marker, 1)[1].split(";", 1)[0]
        if marker in reason
        else str(candidate.get("reason_family") or "")
        .removeprefix("large_cap_")
        .removesuffix("_entry")
    )
    entry_atr_pct = max(0.0, _finite(row.get("atr14_pct")))
    stop_distance_pct = min(
        float(policy.maximum_stop_distance_pct),
        max(
            float(policy.minimum_stop_distance_pct),
            entry_atr_pct * float(policy.stop_atr_multiple),
        ),
    )
    return _VirtualPosition(
        action=str(candidate.get("action") or ""),
        mechanism=mechanism,
        entry_time=pd.Timestamp(candidate["signal_time"]),
        entry_price=_positive(row.get("close"), default=1.0),
        entry_atr_pct=entry_atr_pct,
        stop_distance_pct=stop_distance_pct,
    )


def _exit_family(
    row: pd.Series,
    *,
    position: _VirtualPosition,
    policy: LargeCapRegimePolicy,
) -> str:
    current_price = _positive(row.get("close"), default=position.entry_price)
    direction = 1.0 if position.action == "OPEN_LONG" else -1.0
    return_pct = direction * (current_price / position.entry_price - 1.0) * 100.0
    stop_fraction = position.stop_distance_pct / 100.0
    stop_price = position.entry_price * (
        1.0 - stop_fraction if position.action == "OPEN_LONG" else 1.0 + stop_fraction
    )
    # 计算 adverse_extreme，仅使用当前及历史数据。
    # 计算 adverse_extreme，仅使用当前及历史数据。
    adverse_extreme = _positive(
        row.get("low" if position.action == "OPEN_LONG" else "high"),
        default=current_price,
    )
    stop_hit = (
        adverse_extreme <= stop_price
        if position.action == "OPEN_LONG"
        else adverse_extreme >= stop_price
    )
    if stop_hit:
        return "large_cap_atr_risk_exit"

    snapshot = score_large_cap_mechanisms(
        row,
        action=position.action,
        enabled_trend_rollover_components=(
            policy.enabled_trend_rollover_components
        ),
    )
    support = eligible_market_support(snapshot).get(position.mechanism, 0.0)
    release = max(
        0.0,
        policy.threshold(position.mechanism) - float(policy.release_hysteresis),
    )
    profit_target = max(
        float(policy.minimum_profit_decay_pct),
        position.entry_atr_pct * float(policy.profit_atr_multiple),
    )
    if (
        position.held_bars >= int(policy.minimum_holding_bars)
        and return_pct >= profit_target
        and support <= release
        and snapshot.momentum.short <= 0.0
    ):
        return "large_cap_profit_decay_exit"
    if position.held_bars >= int(policy.maximum_holding_bars):
        return "large_cap_mechanism_timeout_exit"
    return ""


def _close_row(
    row: pd.Series,
    *,
    position: _VirtualPosition,
    close_time: pd.Timestamp,
    exit_family: str,
) -> dict[str, Any]:
    price = _positive(row.get("close"), default=position.entry_price)
    direction = 1.0 if position.action == "OPEN_LONG" else -1.0
    return_pct = direction * (price / position.entry_price - 1.0) * 100.0
    action = "CLOSE_LONG" if position.action == "OPEN_LONG" else "CLOSE_SHORT"
    reason = (
        f"{exit_family};entry_mechanism={position.mechanism};"
        f"held_bars={position.held_bars};return_pct={return_pct:.4f}"
    )
    return {
        "signal_id": "",
        "symbol": str(row.get("symbol") or ""),
        "action": action,
        "signal_time": close_time,
        "score": 3.0,
        "reason": reason,
        "reason_family": exit_family,
        "ret20": _finite(row.get("ret20")),
        "ret60": _finite(row.get("ret60")),
        "ret144": _finite(row.get("ret144")),
        "ret288": _finite(row.get("ret288")),
        "don": _finite(row.get("donchian60_pos"), default=0.5),
        "taker": _finite(row.get("taker_buy_ratio"), default=0.5),
        "qvr": _finite(row.get("quote_volume_ratio_med30"), default=1.0),
        "rsi": _finite(row.get("rsi14"), default=50.0),
        "adx": _finite(row.get("adx14"), default=20.0),
        "atr_pct": _finite(row.get("atr14_pct")),
        "down": _finite(row.get("market_down_breadth20"), default=0.5),
    }


def _empty_close_signals() -> pd.DataFrame:
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


def _finite(value: object, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if pd.notna(result) else float(default)


def _positive(value: object, *, default: float) -> float:
    return max(1e-12, _finite(value, default=default))


__all__ = ["build_large_cap_lifecycle"]
