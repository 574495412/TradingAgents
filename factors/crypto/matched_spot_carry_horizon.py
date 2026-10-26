"""One pre-account correction: visible dual-market deadlines cap funding horizon."""

import pandas as pd


def horizon(symbol, clock, next_review, terminal, perpetual_cards, spot_cards):
    now = pd.Timestamp(clock)
    end = min(pd.Timestamp(next_review), pd.Timestamp(terminal), now + pd.Timedelta(days=7))
    reasons = []
    for row in perpetual_cards:
        if row["symbol"] != symbol or pd.Timestamp(row["available_at_assumed"]) > now:
            continue
        cutoff = pd.Timestamp(row.get("entry_stop_at") or row["settlement_at"])
        if cutoff <= end:
            reasons.append(
                {
                    "market": "perpetual",
                    "exit_at": cutoff.isoformat(),
                    "source_available_at": row["available_at_assumed"],
                }
            )
            end = cutoff
    for row in spot_cards:
        if row["symbol"] != symbol or pd.Timestamp(row["spot_available_at_assumed"]) > now:
            continue
        cutoff = pd.Timestamp(row["spot_entry_cut_assumed"])
        if cutoff <= end:
            reasons.append(
                {
                    "market": "spot",
                    "exit_at": cutoff.isoformat(),
                    "source_available_at": row["spot_available_at_assumed"],
                }
            )
            end = cutoff
    return {
        "remaining_hours": max(0.0, (end - now).total_seconds() / 3600),
        "horizon_end": end.isoformat(),
        "visible_deadlines": reasons,
    }


def covered_cost(rate7, remaining_hours):
    if rate7 is None:
        return {"forecast_rate": None, "cost_guard_passed": None}
    if not 0 <= remaining_hours <= 168:
        raise ValueError("Registered funding horizon must be between zero and7days")
    estimate = rate7 * remaining_hours / 168
    return {"forecast_rate": estimate, "cost_guard_passed": estimate > 1.25 * 0.007}
