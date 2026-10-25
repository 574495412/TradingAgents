"""Causal, provisional episode context and full-day feature qualification.

Only an explicitly observed launch may release a prior terminal restriction.
This adapter resets data lookbacks; it does not reset or transfer account state.
"""

import math

import numpy as np
import pandas as pd

from research_tools.contract_lifecycles.timeline import UnverifiedContractHistory, schedules_at


def context_at(registry, symbol, clock):
    now = pd.Timestamp(clock)
    if now.tz is None:
        raise ValueError("Naive decision clock")
    ends = schedules_at(registry["cards"], symbol, now)
    launches = [
        c
        for c in registry["launch_cards"]
        if c["symbol"] == symbol
        and pd.Timestamp(c["available_at_assumed"]) <= now
        and pd.Timestamp(c["launch_at"]) <= now
    ]
    latest = max(launches, key=lambda c: pd.Timestamp(c["launch_at"])) if launches else None
    if latest:
        ends = [c for c in ends if c["episode"] == latest["episode"]]
    earliest = min(ends, key=lambda c: pd.Timestamp(c["settlement_at"])) if ends else None
    terminal = earliest["settlement_at"] if earliest else None
    cut = earliest["effective_conservative_entry_stop"] if earliest else None
    blocked = bool(
        (terminal and now >= pd.Timestamp(terminal)) or (cut and now >= pd.Timestamp(cut))
    )
    return {
        "symbol": symbol,
        "blocked": blocked,
        "reason": "known_end_or_unverified_reopening"
        if blocked
        else "no_known_restriction_in_partial_capture",
        "active_contract_key": symbol + ":launch@" + latest["launch_at"]
        if latest
        else symbol + ":observed_initial",
        "feature_start": pd.Timestamp(latest["launch_at"]).ceil("D").isoformat()
        if latest
        else None,
        "known_terminal": terminal,
        "known_entry_cutoff": cut,
        "portfolio_risk_reset": False,
        "historical_status_certified": False,
    }


def age_and_quote(frame, context, clock):
    """Count known valid full days inside this contract episode, never old history."""
    now = pd.Timestamp(clock)
    start = pd.Timestamp(context["feature_start"]) if context["feature_start"] else None
    known = frame.loc[
        (frame.index + pd.Timedelta(days=1) <= now)
        & ((frame.index >= start) if start is not None else True)
    ]
    valid = known.price_liquidity_feature_usable.eq(True)
    window = known.loc[known.index >= now - pd.Timedelta(days=28)]
    expected = pd.date_range(now - pd.Timedelta(days=28), now, freq="D", inclusive="left")
    quote = None
    if window.index.equals(expected) and window.price_liquidity_feature_usable.eq(True).all():
        value = float(window.quote_volume.mean())
        if math.isfinite(value):
            quote = value
    return {
        "valid_full_observations_in_episode": int(valid.sum()),
        "quote28": quote,
        "eligible": not context["blocked"]
        and int(valid.sum()) >= 180
        and quote is not None
        and quote >= 50_000_000,
    }


def require_episode_terminal_cash(registry, symbol, episode, clock, quantity):
    if not np.isfinite(quantity):
        raise ValueError("Nonfinite episode inventory")
    if quantity == 0:
        return
    if not episode:
        raise UnverifiedContractHistory(
            "An existing position requires its contract episode identity"
        )
    relevant = [
        c for c in schedules_at(registry["cards"], symbol, clock) if c["episode"] == episode
    ]
    for c in relevant:
        if pd.Timestamp(clock) >= pd.Timestamp(c["settlement_at"]) and (
            c.get("terminal_settlement_price") is None or c.get("terminal_settlement_fee") is None
        ):
            raise UnverifiedContractHistory(
                "Old episode inventory cannot transfer to a reopened contract; terminal cash unverified"
            )
