"""Twelve-calendar-month own-return sign with a declared finite-history risk estimate."""

import numpy as np
import pandas as pd


def monthly_observation(daily_returns, decision, center_of_mass=60, periods_per_year=365):
    """Use only completed UTC days; a missing daily return blocks the observation."""
    decision = pd.Timestamp(decision)
    if decision.tz is None or str(decision.tz) != "UTC" or decision != decision.normalize():
        raise ValueError("Decision must be a normalized UTC day")
    start = decision - pd.DateOffset(months=12)
    dates = pd.date_range(start, decision, freq="D", inclusive="left")
    values = daily_returns.reindex(dates).to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError("Incomplete or nonpositive-factor prior12month economic index")
    delta = center_of_mass / (center_of_mass + 1)
    weights = delta ** np.arange(len(values) - 1, -1, -1)
    weights /= weights.sum()
    mean = float(weights @ values)
    volatility = float(np.sqrt(periods_per_year * (weights @ ((values - mean) ** 2))))
    own_return = float(np.expm1(np.log1p(values).sum()))
    if not np.isfinite(volatility) or volatility <= 0 or not np.isfinite(own_return):
        raise ValueError("Undefined prior economic return or risk")
    return {
        "return_start": start.isoformat(),
        "decision": decision.isoformat(),
        "daily_observations": len(values),
        "own_return_12months": own_return,
        "direction": int(np.sign(own_return)),
        "annual_volatility": volatility,
        "finite_history_weight_normalization": True,
    }
