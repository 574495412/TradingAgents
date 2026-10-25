"""Pure accrued-funding books with fixed cost coverage; no price-momentum term.

Caller certifies a complete event calendar. Exact historical settlement cadence
is retained; a missing event series is never silently a zero-yield signal.
"""

import math

import numpy as np
import pandas as pd

from factors.common.max_effect import book as momentum_book


def accrued7(rates, clock):
    t = pd.Timestamp(clock)
    out = {}
    for s, f in rates.items():
        rows = f.loc[(f.index >= t - pd.Timedelta(days=7)) & (f.index < t), "funding_rate"]
        if not len(rows) or not np.isfinite(rows.to_numpy()).all():
            raise ValueError("Missing complete prior funding history: " + s)
        # API rates have much coarser precision; remove summation-order noise only.
        out[s] = round(math.fsum(rows.tolist()), 12)
    return pd.Series(out)


def book(feature_rows, funding7, parameters, candidate):
    if candidate not in ["funding7_raw", "funding7_cost_floor", "momentum28_control"]:
        raise ValueError("Unknown preregistered pure funding candidate")
    control, common = momentum_book(feature_rows, parameters, "momentum28_control")
    if common["status"] != "formed" or candidate == "momentum28_control":
        return control, {**common, "candidate": candidate}
    eligible = [
        s
        for s, r in feature_rows.items()
        if s != "BTCUSDT"
        and r.observations >= parameters["minimum_listing_observations"]
        and r.quote28 >= parameters["minimum_daily_quote_volume"]
        and np.isfinite(r[["max28", "momentum28", "beta60"]].to_numpy(dtype=float)).all()
    ]
    eligible = sorted(eligible, key=lambda s: (-float(feature_rows[s].quote28), s))[
        : parameters["maximum_eligible"]
    ]
    if any(s not in funding7 or not math.isfinite(float(funding7[s])) for s in eligible):
        raise ValueError("Funding signal unavailable inside matched universe")
    weights = pd.Series(0.0, index=sorted(feature_rows))
    ordered = sorted(eligible, key=lambda s: (float(funding7[s]), s))
    long = ordered[: parameters["long_names"]]
    short = ordered[-parameters["short_names"] :]
    if set(long) & set(short):
        raise ValueError("Funding sides overlap")
    status = "formed"
    if max(float(funding7[s]) for s in eligible) == min(float(funding7[s]) for s in eligible):
        status = "no_funding_dispersion"
    else:
        weights.loc[long] = 0.5 / len(long)
        weights.loc[short] = -0.5 / len(short)
    forecast = -math.fsum(float(weights[s]) * float(funding7[s]) for s in eligible)
    floor = 1.25 * 2 * 15 / 10000
    if candidate == "funding7_cost_floor" and forecast <= floor:
        weights[:] = 0.0
        status = "forecast_below_registered_cost_floor"
    if abs(weights.sum()) > 1e-12 or weights.abs().sum() > 1 + 1e-12:
        raise ValueError("Funding book has unintended net or gross target")
    return weights, {
        "candidate": candidate,
        "eligible": len(eligible),
        "eligible_symbols": eligible,
        "status": status,
        "long": long,
        "short": short,
        "forecast_unit_gross_7day_funding": forecast,
        "registered_unit_cost_floor": floor,
        "target_net_weight": float(weights.sum()),
        "target_gross_weight": float(weights.abs().sum()),
        "rate_signal_is_price_prediction": False,
    }
