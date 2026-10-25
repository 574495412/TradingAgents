"""Shared intent budgets, cancellation and unknown-model fallback."""

import numpy as np
import pandas as pd

from factors.portfolio.regime_allocation import allocate, bounded_mix


def test_opposing_sleeves_cancel_without_reallocating_trimmed_risk():
    books = {"a": pd.Series({"A": 0.6, "B": 0.4}), "b": pd.Series({"A": -0.6, "B": 0.4})}
    result = bounded_mix(books, {"a": 0.5, "b": 0.5}, maximum=1)
    assert result.to_dict() == {"A": 0.0, "B": 0.4}


def test_unknown_model_and_liquidity_states_cannot_be_promoted_into_a_trade():
    clocks = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    books = {
        k: pd.DataFrame({"A": 1.0, "B": 0.0}, index=clocks)
        for k in ["trend", "relative_momentum", "carry_momentum", "mean_reversion", "breakout"]
    }
    states = pd.DataFrame({"state": ["trend_up", "liquidity_stress", "unknown"]}, index=clocks)
    model = pd.DataFrame(
        {"hmm_state": "unknown", "linear_probability": np.nan, "mlp_probability": np.nan},
        index=clocks,
    )
    routes = {"trend_up": {"trend": 1}, "liquidity_stress": {}, "unknown": {}}
    targets = allocate(books, states, model, routes)
    for key in ("linear_router", "mlp_router", "hmm_router", "cash"):
        assert not targets[key].ne(0).any().any()
    assert not targets["rule_router"].iloc[1:].ne(0).any().any()
