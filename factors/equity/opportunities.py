"""Equity session adapter; new factors only, frozen common panels unchanged.

Mathematics uses observed sessions, not calendar days. Outputs are close-time
decisions; execution must lag one actual session. No fitted or future labels.
"""

import numpy as np
import pandas as pd

from factors.common.session_momentum import momentum_inputs


def opportunities(frame):
    close = frame.close
    inputs = momentum_inputs(close)
    fast, slow, long, rsi = (inputs[k] for k in ("fast", "slow", "long", "rsi2"))
    enter = (close > long) & (rsi < 10)
    exit_ = (rsi > 60) | (close < long)
    # Retain the recovery position between entry and exit, using past-only state.
    recovery, active = [], False
    for buy, sell in zip(enter, exit_, strict=True):
        if sell:
            active = False
        elif buy:
            active = True
        recovery.append(active)
    vol = inputs["vol20"] * np.sqrt(252)
    return pd.DataFrame(
        {
            "trend": fast > slow,
            "recovery": recovery,
            "vol": vol,
            "state": np.where(long.isna(), "unknown", np.where(close > long, "up", "down")),
        },
        index=frame.index,
    )


def targets(frames, policy):
    allowed = {
        "static",
        "risk_scaled",
        "finite_state",
        "trend_only",
        "recovery_only",
        "buy_hold",
        "cash",
    }
    if policy not in allowed:
        raise ValueError("Unregistered equity allocation")
    features = {s: opportunities(f) for s, f in frames.items()}
    index = next(iter(frames.values())).index
    trend = pd.DataFrame({s: f.trend.astype(float) for s, f in features.items()})
    recovery = pd.DataFrame({s: f.recovery.astype(float) for s, f in features.items()})
    vol = pd.DataFrame({s: f.vol for s, f in features.items()})
    benchmark = features[next(iter(frames))].state
    weights = pd.DataFrame(0.0, index=index, columns=list(frames))
    if policy == "buy_hold":
        weights[:] = 0.8 / len(frames)
    elif policy != "cash":
        # Inactive module budgets remain cash; never renormalize across modules.
        symbol_budget = pd.DataFrame(1 / len(frames), index=index, columns=list(frames))
        if policy in {"risk_scaled", "finite_state"}:
            inverse = 1 / vol.replace(0, np.nan)
            symbol_budget = inverse.div(inverse.sum(axis=1), axis=0).fillna(0)
        a, b = pd.Series(0.4, index=index), pd.Series(0.4, index=index)
        if policy == "finite_state":
            a = benchmark.map({"up": 0.6, "down": 0.2, "unknown": 0.0})
            b = benchmark.map({"up": 0.2, "down": 0.6, "unknown": 0.0})
        if policy == "trend_only":
            b[:] = 0
        if policy == "recovery_only":
            a[:] = 0
        weights = symbol_budget * (trend.mul(a, axis=0) + recovery.mul(b, axis=0))
        if policy in {"risk_scaled", "finite_state"}:
            bound = (weights * vol).sum(axis=1)
            weights = weights.mul((0.12 / bound.replace(0, np.nan)).clip(upper=1).fillna(0), axis=0)
    return weights, benchmark
