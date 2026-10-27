"""Transaction-timed benchmark price comparisons, not investable hedge accounts."""

import math

import numpy as np
import pandas as pd


def matched_btc_path(events, index, btc_marks, reference_price, beta_at_submission):
    if (
        not btc_marks.index.equals(index)
        or not np.isfinite(btc_marks).all()
        or not btc_marks.gt(0).all()
    ):
        raise ValueError("Complete actual BTC marks required")
    positions, snapshots, closed, missing = {}, {}, [], []
    initial = {"unit": 0.0, "beta": 0.0, "long_unit": 0.0, "short_unit": 0.0}
    snapshots[index[0]] = initial
    for event in events:
        if event["type"] not in {"entry", "exit"}:
            continue
        clock = pd.Timestamp(event["time"])
        if clock not in index or clock.floor("5min") != clock:
            raise ValueError("Execution outside actual valuation grid")
        symbol, quantity, side = event["symbol"], event["quantity"], event["side"]
        key = (symbol, event["position_id"])
        ref, btc = float(reference_price(symbol, clock)), float(btc_marks.loc[clock])
        if not math.isfinite(ref) or ref <= 0 or quantity <= 0 or side not in {-1, 1}:
            raise ValueError("Invalid raw fill reference or position")
        if event["type"] == "entry":
            info = beta_at_submission(symbol, pd.Timestamp(event["submitted_at"]))
            beta = info["beta"]
            if beta is not None and not math.isfinite(beta):
                raise ValueError("Nonfinite beta must be explicit unknown")
            p = positions.setdefault(
                key,
                {
                    "symbol": symbol,
                    "side": side,
                    "q": 0.0,
                    "basis": 0.0,
                    "unit_shares": 0.0,
                    "beta_shares": 0.0,
                    "beta_basis": 0.0,
                    "beta_known": True,
                    "price_pnl": 0.0,
                    "unit_market_pnl": 0.0,
                    "beta_market_pnl": 0.0,
                },
            )
            if p["side"] != side:
                raise ValueError("Side change without closing owned inventory")
            p["q"] += quantity
            p["basis"] += quantity * ref
            p["unit_shares"] += quantity * ref / btc
            if beta is None:
                p["beta_known"] = False
                missing.append(
                    {
                        "symbol": symbol,
                        "position_id": key[1],
                        "submitted_at": event["submitted_at"],
                        **info,
                    }
                )
            else:
                p["beta_shares"] += quantity * ref * beta / btc
                p["beta_basis"] += quantity * ref * beta
        else:
            if key not in positions:
                raise ValueError("Exit without owned benchmark inventory")
            p = positions[key]
            if p["side"] != side or quantity > p["q"] * (1 + 1e-10):
                raise ValueError("Exit side/quantity differs from actual owned inventory")
            fraction = min(1.0, quantity / p["q"])
            p["price_pnl"] += side * (quantity * ref - fraction * p["basis"])
            p["unit_market_pnl"] += side * fraction * (p["unit_shares"] * btc - p["basis"])
            if p["beta_known"]:
                p["beta_market_pnl"] += side * fraction * (p["beta_shares"] * btc - p["beta_basis"])
            for name in ["q", "basis", "unit_shares", "beta_shares", "beta_basis"]:
                p[name] *= 1 - fraction
            if p["q"] <= 1e-12:
                p["position_id"] = key[1]
                p["beta_market_pnl"] = p["beta_market_pnl"] if p["beta_known"] else None
                closed.append(dict(p))
                del positions[key]
        snapshots[clock] = {
            "unit": math.fsum(p["side"] * p["unit_shares"] for p in positions.values()),
            "beta": math.fsum(p["side"] * p["beta_shares"] for p in positions.values())
            if all(p["beta_known"] for p in positions.values())
            else np.nan,
            "long_unit": math.fsum(p["unit_shares"] for p in positions.values() if p["side"] == 1),
            "short_unit": math.fsum(
                -p["unit_shares"] for p in positions.values() if p["side"] == -1
            ),
        }
    if positions:
        raise ValueError("This bounded audit requires the observed fully flat terminal paths")
    # Copy whole held state rows, preserving an existing NaN, rather than filling unknown beta.
    held = (
        pd.DataFrame.from_dict(snapshots, orient="index")
        .sort_index()
        .reindex(index, method="ffill")
    )
    previous = held.shift(1)
    previous.iloc[0] = 0.0
    delta = btc_marks.diff()
    delta.iloc[0] = 0.0
    increments = previous.mul(delta, axis=0)
    return increments, closed, missing


def price_increments(curve, events, cost_fraction):
    nav = curve.set_index("date")
    components = pd.DataFrame(0.0, index=nav.index, columns=["funding", "fees", "slippage"])
    for event in events:
        clock = pd.Timestamp(event["time"]).ceil("5min")
        if clock not in components.index:
            raise ValueError("Cash event outside valuation grid")
        if event["type"] == "funding":
            components.at[clock, "funding"] += event["cash_delta"]
        else:
            components.at[clock, "fees"] += event["fee"]
            components.at[clock, "slippage"] += event["slippage_cost"]
    components["net"] = nav.equity.diff().fillna(0.0)
    reserve = nav.gross_exposure * cost_fraction
    components["exit_reserve_change"] = reserve.diff().fillna(0.0)
    components["price"] = (
        components.net
        - components.funding
        + components.fees
        + components.slippage
        + components.exit_reserve_change
    )
    return components
