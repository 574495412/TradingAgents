"""One fixed cost-conditioned timing diagnostic within the existing funding family."""

import numpy as np
import pandas as pd


def schedule(funding, clocks, condition, source_start, mode):
    if mode not in ("fixed_hold", "cost_guarded"):
        raise ValueError("Unknown registered carry comparison")
    delay = pd.Timedelta(minutes=condition["availability_delay_minutes"])
    lookback = pd.Timedelta(days=condition["lookback_days"])
    exit_window = pd.Timedelta(days=condition["exit_negative_funding_days"])
    # Both markets, both entry and exit, using the fixed stress cost in both runs.
    stress_roundtrip_rate = 2 * (0.001 + 0.0005 + 2 * 0.001)
    results = []
    for clock in clocks:
        assets = {}
        for symbol, frame in funding.items():
            available = frame.index + delay
            values = frame.funding_rate.to_numpy()
            right = available.searchsorted(clock, side="right")
            left = available.searchsorted(clock - lookback, side="right")
            short_left = available.searchsorted(clock - exit_window, side="right")
            known = clock - lookback - delay >= pd.Timestamp(source_start)
            long_rate = float(np.sum(values[left:right])) if known else None
            short_rate = float(np.sum(values[short_left:right])) if known else None
            forecast = (
                long_rate * condition["forecast_cost_recovery_days"] / condition["lookback_days"]
                if known
                else None
            )
            assets[symbol] = {
                "entry": mode == "fixed_hold"
                or bool(
                    known and forecast > stress_roundtrip_rate * condition["cost_coverage_buffer"]
                ),
                "exit": mode == "cost_guarded" and bool(known and short_rate < 0),
                "past28_rate": long_rate,
                "past7_rate": short_rate,
                "last_observable_event": frame.index[right - 1].isoformat() if right else None,
                "history_complete": bool(known),
            }
        results.append({"time": clock.isoformat(), "assets": assets})
    return results
