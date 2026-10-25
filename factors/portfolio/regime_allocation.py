"""Frozen state-to-strategy intentions; never select a past winner after returns."""

import numpy as np
import pandas as pd

from factors.common.regime_books import FAMILIES


def bounded_mix(books, allocations, maximum=10):
    if not allocations:
        return pd.Series(0.0, index=next(iter(books.values())).index)
    if any(v < 0 for v in allocations.values()) or sum(allocations.values()) > 1 + 1e-12:
        raise ValueError("Nonnegative unit-gross family budgets required")
    weights = sum(books[name] * fraction for name, fraction in allocations.items())
    chosen = sorted(weights.index, key=lambda s: (-abs(weights[s]), s))[:maximum]
    return weights.where(weights.index.isin(chosen), 0.0)


def allocate(books, states, predictions, routing):
    out = {name: book.copy() for name, book in books.items()}
    template = books[FAMILIES[0]]
    for key in (
        "cash",
        "equal_mix",
        "rule_router",
        "event_router",
        "hmm_router",
        "linear_router",
        "mlp_router",
    ):
        out[key] = template * 0.0
    for clock in states.index:
        current = {name: book.loc[clock] for name, book in books.items()}
        out["equal_mix"].loc[clock] = bounded_mix(current, {k: 1 / len(FAMILIES) for k in FAMILIES})
        state = states.loc[clock, "state"]
        rule = bounded_mix(current, routing[state])
        out["rule_router"].loc[clock] = rule
        out["event_router"].loc[clock] = rule
        if state in ("unknown", "liquidity_stress") or clock not in predictions.index:
            continue
        row = predictions.loc[clock]
        out["hmm_router"].loc[clock] = bounded_mix(current, routing[row.hmm_state])
        for kind in ("linear", "mlp"):
            probability = row[kind + "_probability"]
            if not np.isfinite(probability):
                continue
            agree = (state == "trend_up" and probability >= 0.55) or (
                state == "trend_down" and probability <= 0.45
            )
            scale = 1 if agree or state not in ("trend_up", "trend_down") else 0.5
            out[kind + "_router"].loc[clock] = rule * scale
    return out
