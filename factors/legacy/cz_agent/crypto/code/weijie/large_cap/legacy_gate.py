"""大盘标的市场结构模块：提供 legacy gate 因子逻辑。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from factors.crypto.code.weijie.common import weijie_market_prior_group
from factors.crypto.code.weijie.large_cap.eligibility import (
    eligible_market_support,
)
from factors.crypto.code.weijie.large_cap.mechanisms import score_large_cap_mechanisms


@dataclass(frozen=True)
class LargeCapLegacyGatePolicy:
    """旧版大盘动作接入新生命周期时的兼容门控参数。"""

    minimum_market_support: float = 0.34


def gate_large_cap_legacy_actions(
    signals: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    policy: LargeCapLegacyGatePolicy | None = None,
) -> pd.DataFrame:
    """用当前市场状态过滤旧版动作，同时保留稳定的动作语义。"""

    if signals is None or signals.empty:
        return signals.copy() if signals is not None else pd.DataFrame()
    cfg = policy or LargeCapLegacyGatePolicy()
    features = feature_rows.copy()
    features["bar_time"] = pd.to_datetime(features.get("bar_time"), errors="coerce")
    features["symbol"] = features.get("symbol", pd.Series("", index=features.index)).astype(str)
    features = (
        features.dropna(subset=["bar_time"])
        .sort_values(["symbol", "bar_time"], kind="stable")
        .drop_duplicates(["symbol", "bar_time"], keep="last")
        .set_index(["symbol", "bar_time"])
    )
    out = signals.copy()
    out["signal_time"] = pd.to_datetime(out.get("signal_time"), errors="coerce")
    keep: list[bool] = []
    gate_reason: list[str] = []
    for _, signal in out.iterrows():
        symbol = str(signal.get("symbol") or "")
        if weijie_market_prior_group(symbol) != "large_cap_momentum":
            keep.append(True)
            gate_reason.append("not_large_cap")
            continue
        action = _market_action_for_inventory_action(str(signal.get("action") or ""))
        decision_time = _causal_decision_time(signal)
        key = (symbol, decision_time)
        if not action or pd.isna(decision_time) or key not in features.index:
            keep.append(False)
            gate_reason.append("missing_closed_market_snapshot")
            continue
        snapshot = score_large_cap_mechanisms(features.loc[key], action=action)
        scores = eligible_market_support(snapshot)
        mechanism = max(scores, key=scores.get)
        support = float(scores[mechanism])
        allowed = support >= float(cfg.minimum_market_support)
        keep.append(allowed)
        gate_reason.append(f"{mechanism}:{support:.4f}")
    out["large_cap_market_gate"] = gate_reason
    passed = out.loc[pd.Series(keep, index=out.index)].copy()
    if not passed.empty:
        passed["reason"] = (
            passed.get("reason", pd.Series("", index=passed.index)).astype(str)
            + ";large_cap_market_gate="
            + passed["large_cap_market_gate"].astype(str)
        )
    return passed.reset_index(drop=True)


def _market_action_for_inventory_action(action: str) -> str:
    normalized = str(action).upper()
    return {
        "OPEN_LONG": "OPEN_LONG",
        "OPEN_SHORT": "OPEN_SHORT",
        "CLOSE_LONG": "OPEN_SHORT",
        "CLOSE_SHORT": "OPEN_LONG",
    }.get(normalized, "")


def _causal_decision_time(signal: pd.Series) -> pd.Timestamp:
    explicit = pd.to_datetime(signal.get("decision_time"), errors="coerce")
    if pd.notna(explicit):
        return pd.Timestamp(explicit)
    reason = str(signal.get("reason") or "")
    marker = "decision="
    if marker in reason:
        parsed = pd.to_datetime(reason.split(marker, 1)[1].split(";", 1)[0], errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed)
    return pd.Timestamp(pd.to_datetime(signal.get("signal_time"), errors="coerce"))


__all__ = [
    "LargeCapLegacyGatePolicy",
    "gate_large_cap_legacy_actions",
]
