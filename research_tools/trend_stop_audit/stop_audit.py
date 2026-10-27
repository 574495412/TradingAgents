"""Reconstruct each stop from recorded fills, without importing the engine."""

from collections import Counter

import pandas as pd


def audit_stops(events, final_positions, bars, fraction, weighted=False):
    positions, counters, stop_requests = {}, Counter(), set()
    retroactive = []
    for e in events:
        key = (e["symbol"], e["position_id"])
        if e["type"] == "funding":
            continue
        if e["type"] == "entry":
            t = pd.Timestamp(e["time"])
            if e["addition"]:
                p = positions[key]
                previous = p["stop"]
                p["price"] = (p["q"] * p["price"] + e["quantity"] * e["price"]) / (
                    p["q"] + e["quantity"]
                )
                p["q"] += e["quantity"]
                candidate = (p["price"] if weighted else e["price"]) * (1 - e["side"] * fraction)
                p["stop"] = (
                    candidate
                    if weighted
                    else (max(previous, candidate) if p["side"] == 1 else min(previous, candidate))
                )
                counters["additions"] += 1
                if p["stop"] != previous:
                    p["previous_stop"], p["changed_at"] = previous, t
                    tighter = p["stop"] > previous if p["side"] == 1 else p["stop"] < previous
                    counters["tightenings" if tighter else "relaxations"] += 1
            else:
                positions[key] = {
                    "q": e["quantity"],
                    "price": e["price"],
                    "side": e["side"],
                    "stop": e["price"] * (1 - e["side"] * fraction),
                    "opened_at": t,
                    "changed_at": t,
                    "previous_stop": None,
                }
        else:
            p = positions[key]
            if e["reason"] == "stop":
                submitted = pd.Timestamp(e["submitted_at"])
                request = (key, submitted)
                if request not in stop_requests:
                    bar = bars[e["symbol"]].loc[submitted - pd.Timedelta(minutes=5)]
                    if not p["opened_at"] < submitted:
                        raise ValueError("Initial entry bar was used to stop its own new position")
                    breached = bar.low <= p["stop"] if p["side"] == 1 else bar.high >= p["stop"]
                    if not breached:
                        raise ValueError(
                            "Stop request lacks a threshold breach in actual preceding bar"
                        )
                    if p["changed_at"] == submitted and p["previous_stop"] is not None:
                        earlier_breach = (
                            bar.low <= p["previous_stop"]
                            if p["side"] == 1
                            else bar.high >= p["previous_stop"]
                        )
                        if not earlier_breach:
                            retroactive.append(
                                {
                                    "symbol": key[0],
                                    "position_id": key[1],
                                    "request_at": submitted.isoformat(),
                                    "previous_stop": p["previous_stop"],
                                    "new_stop": p["stop"],
                                    "preceding_low": float(bar.low),
                                    "preceding_high": float(bar.high),
                                }
                            )
                    stop_requests.add(request)
            p["q"] -= e["quantity"]
            if p["q"] < 1e-12:
                del positions[key]
    for s, final in final_positions.items():
        p = positions.pop((s, final["id"]))
        for field in ["q", "price", "stop"]:
            if abs(p[field] - final[field]) > 1e-8:
                raise ValueError("Stop/inventory state does not match fill reconstruction")
    if positions:
        raise ValueError("Final position omitted from engine state")
    return {
        "passed": not retroactive,
        "policy": "weighted_cost" if weighted else "newfill_ratchet",
        **dict(counters),
        "first_stop_requests": len(stop_requests),
        "retroactive_new_stop_requests": retroactive,
        "scope": "Actual recorded stop requests and fill-derived stop state, not independent market truth or proof of untriggered-stop completeness.",
    }
