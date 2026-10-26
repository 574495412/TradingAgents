"""Fixed data-only checks for matched spot/perpetual carry; identity gated by caller."""

import math

import numpy as np
import pandas as pd


def prior_spot_window(frame, clock):
    """No interpolation:28 actual completed days, excluding first observed label."""
    t = pd.Timestamp(clock)
    if t != t.floor("D"):
        raise ValueError("Registered weekly decisions require UTC day boundary")
    window = pd.date_range(t - pd.Timedelta(days=28), periods=28, freq="D")
    past = frame.loc[frame.index < t]
    observations = past.reindex(window)
    usable = observations.feature_usable.eq(True)
    if len(past):
        usable &= observations.index != past.index[0]
    consecutive = bool(usable.all())
    quote = float(observations.quote_volume.mean()) if consecutive else None
    return {
        "complete_28days": consecutive,
        "mean_daily_quote": quote,
        "data_eligible": bool(consecutive and quote >= 10_000_000),
        "missing_daily_labels": int((~observations.index.isin(past.index)).sum()),
        "invalid_or_first_observed_labels": int((~usable).sum()),
    }


def known_funding7(frame, clock):
    t = pd.Timestamp(clock)
    events = frame.loc[
        (frame.index >= t - pd.Timedelta(days=7)) & (frame.index + pd.Timedelta(minutes=5) <= t)
    ]
    values = events.funding_rate.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return {
            "reported_events": 0,
            "known_rate_sum": None,
            "positive": None,
            "cost_guard_passed": None,
            "last_observable_event": None,
        }
    rate = round(math.fsum(values.tolist()), 12)
    threshold = 1.25 * 2 * (0.001 + 0.0005 + 2 * 0.001)
    return {
        "reported_events": len(events),
        "known_rate_sum": rate,
        "positive": rate > 0,
        "cost_guard_passed": rate > threshold,
        "last_observable_event": events.index[-1].isoformat(),
    }
