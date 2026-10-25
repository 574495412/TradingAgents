"""Partial, source-bound contract schedules; current bodies are provisional history.

These functions can reject a historical scenario. They cannot certify tradability
or turn an observed archive bar into an actual terminal cash settlement.
"""

import math

import pandas as pd


class UnverifiedContractHistory(ValueError):
    pass


def validate(cards):
    by_id = {}
    for c in cards:
        if c["id"] in by_id or c["contract_type"] != "USD_M_PERPETUAL":
            raise ValueError("Duplicate identity or unsupported product")
        for key in ["available_at_assumed", "publish_at_claimed", "settlement_at"]:
            t = pd.Timestamp(c[key])
            if t.tz is None:
                raise ValueError("Naive contract clock")
        if pd.Timestamp(c["available_at_assumed"]) < pd.Timestamp(c["publish_at_claimed"]):
            raise ValueError("Availability precedes claimed publication")
        if c.get("entry_stop_at") and pd.Timestamp(c["entry_stop_at"]) > pd.Timestamp(
            c["settlement_at"]
        ):
            raise ValueError("Entry restriction after terminal schedule")
        previous = c.get("supersedes")
        if previous:
            if previous not in by_id:
                raise ValueError("Missing earlier schedule revision")
            old = by_id[previous]
            if (old["symbol"], old["episode"]) != (c["symbol"], c["episode"]):
                raise ValueError("A revision cannot replace another contract episode")
            if pd.Timestamp(c["available_at_assumed"]) <= pd.Timestamp(old["available_at_assumed"]):
                raise ValueError("Revision clocks must advance")
        if any(
            c.get(k) is not None and (not math.isfinite(c[k]) or c[k] < 0)
            for k in ["terminal_settlement_price", "terminal_settlement_fee"]
        ):
            raise ValueError("Invalid terminal value")
        by_id[c["id"]] = c
    return by_id


def schedules_at(cards, symbol, clock):
    """Assume captured clauses existed at claimed publication + latency, explicitly."""
    now = pd.Timestamp(clock)
    if now.tz is None:
        raise ValueError("Decision needs an aware clock")
    by_id = validate(cards)
    known = [
        c for c in cards if c["symbol"] == symbol and pd.Timestamp(c["available_at_assumed"]) <= now
    ]
    replaced = {c["supersedes"] for c in known if c.get("supersedes")}
    result = []
    for c in known:
        if c["id"] in replaced:
            continue
        cut, inherited = c.get("entry_stop_at"), False
        ancestor = c
        while cut is None and ancestor.get("supersedes"):
            ancestor = by_id[ancestor["supersedes"]]
            cut = ancestor.get("entry_stop_at")
            inherited = cut is not None
        result.append(
            {
                **c,
                "effective_conservative_entry_stop": cut,
                "entry_cutoff_inherited_not_reaffirmed": inherited,
            }
        )
    return result


def restriction_at(cards, symbol, clock):
    """A later end notice is not evidence of a reopening after an earlier end."""
    now = pd.Timestamp(clock)
    schedules = schedules_at(cards, symbol, now)
    terminal = [c for c in schedules if now >= pd.Timestamp(c["settlement_at"])]
    cut = [
        c
        for c in schedules
        if c["effective_conservative_entry_stop"] is not None
        and now >= pd.Timestamp(c["effective_conservative_entry_stop"])
    ]
    reason = (
        "terminal_or_relisting_unverified"
        if terminal
        else "conservative_no_new_entries"
        if cut
        else "no_known_restriction_in_partial_capture"
    )
    return {
        "blocked": bool(terminal or cut),
        "reason": reason,
        "source_event_ids": [c["id"] for c in terminal or cut],
        "historical_tradability_certified": False,
        "current_body_at_claimed_publication_is_assumption": True,
    }


def daily_feature_allowed(cards, symbol, bar_open, decision):
    start, now = pd.Timestamp(bar_open), pd.Timestamp(decision)
    if start + pd.Timedelta(days=1) > now:
        raise ValueError("Daily feature includes an unfinished bar")
    return all(
        start + pd.Timedelta(days=1) <= pd.Timestamp(c["settlement_at"])
        for c in schedules_at(cards, symbol, now)
    )


def require_terminal_cash(cards, symbol, clock, quantity):
    if not math.isfinite(quantity):
        raise ValueError("Nonfinite inventory")
    if quantity == 0:
        return
    for c in schedules_at(cards, symbol, clock):
        if pd.Timestamp(clock) >= pd.Timestamp(c["settlement_at"]) and (
            c.get("terminal_settlement_price") is None or c.get("terminal_settlement_fee") is None
        ):
            raise UnverifiedContractHistory(
                "Nonflat terminal price and fee unverified; archive close and zero cannot substitute"
            )


def require_certified_history(cards):
    if not cards or any(
        not c.get("original_historical_content_certified")
        or not c.get("original_historical_revision_time_certified")
        for c in cards
    ):
        raise UnverifiedContractHistory(
            "Current clause timeline is provisional, not historical-version certification"
        )
