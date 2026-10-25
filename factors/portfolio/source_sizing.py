"""Independent mathematical source sizing; no source program is imported."""

import numpy as np
import pandas as pd


def sized_intent(raw, volatility, normalization, parameters):
    """Return direction-control, conviction and absolute-risk unit weights/bands."""
    p = parameters
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("A finite positive training-only forecast scale is required")
    if not 0 < p["maximum_gross"] <= 1 or p["annualization"] <= 0:
        raise ValueError("Invalid account risk budget or market calendar")
    z = (raw / normalization).clip(-3, 3)
    selected = [
        s
        for s in sorted(z.index, key=lambda s: (-abs(z[s]) if np.isfinite(z[s]) else np.inf, s))
        if np.isfinite(z[s])
        and abs(z[s]) > p["existing_selection_threshold"]
        and np.isfinite(volatility[s])
    ][: p["maximum_positions"]]
    empty = pd.Series(0.0, index=raw.index)
    if not selected:
        return {"conviction": empty.copy(), "risk": empty.copy(), "buffer": empty.copy()}
    sigma = volatility.loc[selected].clip(lower=p["relative_volatility_floor"])
    forecast = (raw.loc[selected] / normalization * p["average_absolute_forecast"]).clip(
        -p["forecast_cap"], p["forecast_cap"]
    )
    relative = forecast / sigma
    conviction = relative / relative.abs().sum()
    average = p["annual_risk_target"] / np.sqrt(p["annualization"]) / sigma / len(selected)
    desired = average * forecast / p["average_absolute_forecast"]
    cap = min(1.0, p["maximum_gross"] / desired.abs().sum())
    risk = desired * cap / p["maximum_gross"]
    width = average * p["buffer_fraction"] * cap / p["maximum_gross"]
    result = {}
    for name, weights in (("conviction", conviction), ("risk", risk), ("buffer", width)):
        result[name] = empty.copy()
        result[name].loc[weights.index] = weights
    return result
