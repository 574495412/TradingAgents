"""Continuous netted equity account; JSON state is the recovery boundary.

Extends the existing equities research path without changing v001 semantics.
No broker or order APIs. One currency, long cash, observed exchange sessions.
"""

import copy
import math

import numpy as np
import pandas as pd

from .data import validate
from .fees import charge


def replay(
    frames,
    calendar,
    weights,
    regimes,
    spec,
    *,
    market,
    start,
    end,
    slippage_bps,
    identity,
    policy,
    state=None,
):
    symbols = {s["symbol"]: s for s in spec["symbols"]}
    if set(symbols) != set(frames) or list(weights.columns) != list(frames):
        raise ValueError("Account universe/weights mismatch")
    if not weights.index.equals(calendar.index) or not regimes.index.equals(calendar.index):
        raise ValueError("Decisions require the exact exchange calendar")
    if (
        not np.isfinite(weights.to_numpy()).all()
        or (weights < 0).any().any()
        or (weights.sum(axis=1) > 0.800000001).any()
    ):
        raise ValueError("Only finite long cash weights up to 0.8")
    for f in frames.values():
        validate(f, calendar)
    if slippage_bps not in {5, 15} or not identity:
        raise ValueError("Unregistered execution scenario")
    initial = float(spec["initial_cash"])
    binding = {
        "identity": identity,
        "market": market,
        "spec": spec,
        "slippage_bps": slippage_bps,
        "policy": policy,
    }
    window = calendar.loc[
        (calendar.index >= pd.Timestamp(start)) & (calendar.index < pd.Timestamp(end))
    ]
    if window.empty:
        raise ValueError("Empty replay window")
    if state is None:
        state = {
            "schema": "ta-equity-state.v2",
            "binding": binding,
            "cash": initial,
            "receivables": 0.0,
            "unsettled": [],
            "positions": {},
            "peak": initial,
            "halt_until": -1,
            "halted": False,
            "last_day": None,
            "fills": [],
            "trades": [],
            "fees": {},
            "dividends": [],
            "requests": [],
            "equity_curve": [
                {
                    "clock": int(window.iloc[0].open.timestamp() * 1000) - 1,
                    "equity": initial,
                    "drawdown": 0.0,
                }
            ],
            "max_low_dd": 0.0,
        }
    else:
        state = copy.deepcopy(state)
        if state.get("schema") != "ta-equity-state.v2" or state.get("binding") != binding:
            raise ValueError("Restart identity/configuration mismatch")
        previous = calendar.index.get_loc(pd.Timestamp(state["last_day"]))
        if calendar.index.get_loc(window.index[0]) != previous + 1:
            raise ValueError("Restart must continue at the very next observed session")
    st = state
    slip = slippage_bps / 10000
    volumes = {s: f.volume.rolling(20).mean() for s, f in frames.items()}

    def execution(s, side, qty, day, stamp, decision_day):
        raw = float(frames[s].at[day, "open"])
        price = raw * (1 + slip if side == "BUY" else 1 - slip)
        parts = charge(market, day, side, qty, price, spec)
        fee = sum(parts.values())
        st["fills"].append(
            {
                "symbol": s,
                "side": side,
                "quantity": qty,
                "price": price,
                "raw_open": raw,
                "fee": fee,
                "fee_parts": parts,
                "slippage": qty * abs(price - raw),
                "clock": stamp,
                "day": str(day.date()),
                "decision_day": str(decision_day.date()),
            }
        )
        for k, v in parts.items():
            st["fees"][k] = st["fees"].get(k, 0.0) + v
        return price, fee

    for day, session in window.iterrows():
        i = calendar.index.get_loc(day)
        prior = calendar.index[i - 1] if i else None
        stamp = int(session.open.timestamp() * 1000)
        released = [x for x in st["unsettled"] if x["release_index"] <= i]
        st["cash"] += sum(x["amount"] for x in released)
        st["unsettled"] = [x for x in st["unsettled"] if x["release_index"] > i]
        for s, p in st["positions"].items():
            amount = (
                p["quantity"]
                * float(frames[s].at[day, "dividend"])
                * (1 - spec["dividend_haircut"])
            )
            if amount:
                st["receivables"] += amount
                p["cycle_net"] += amount
                st["dividends"].append(
                    {
                        "symbol": s,
                        "day": str(day.date()),
                        "quantity": p["quantity"],
                        "amount": amount,
                    }
                )
        if prior is not None:
            equity_before = st["equity_curve"][-1]["equity"]
            planned = {}
            for s in symbols:
                lot = symbols[s]["lot_size"]
                if isinstance(lot, bool) or not isinstance(lot, int) or lot < 1:
                    raise ValueError("Invalid board lot")
                mark = float(frames[s].at[prior, "close"])
                target_weight = float(weights.at[prior, s]) if not st["halted"] else 0.0
                target_qty = math.floor(target_weight * equity_before / mark / lot) * lot
                held = st["positions"].get(s, {}).get("quantity", 0)
                # No-trade band never suppresses a full exit.
                difference = target_qty - held
                if target_qty and abs(difference) * mark < 0.05 * equity_before:
                    difference = 0
                cap = (
                    math.floor(float(volumes[s].at[prior]) * 0.001 / lot) * lot
                    if pd.notna(volumes[s].at[prior])
                    else 0
                )
                planned[s] = int(np.sign(difference)) * min(abs(difference), cap)
                if difference:
                    st["requests"].append(
                        {
                            "day": str(day.date()),
                            "symbol": s,
                            "requested": difference,
                            "volume_capped": planned[s],
                            "filled": 0,
                            "regime": str(regimes.at[prior]),
                        }
                    )
            # Net targets first. A sale never funds a same-session purchase.
            for s, delta in planned.items():
                if delta >= 0:
                    continue
                qty = -delta
                price, fee = execution(s, "SELL", qty, day, stamp, prior)
                p = st["positions"][s]
                p["remaining_cost"] *= 1 - qty / p["quantity"]
                p["quantity"] -= qty
                proceeds = qty * price - fee
                p["cycle_net"] += proceeds
                st["unsettled"].append({"amount": proceeds, "release_index": i + 3})
                if not p["quantity"]:
                    st["trades"].append(
                        {
                            "symbol": s,
                            "direction": 1,
                            "entry_ms": p["entry_ms"],
                            "exit_ms": stamp,
                            "entry_price": p["entry_price"],
                            "exit_price": price,
                            "net_pnl": p["cycle_net"],
                            "entry_regime": p["entry_regime"],
                            "reason": "net_target_exit",
                        }
                    )
                    del st["positions"][s]
            for s, delta in planned.items():
                if delta <= 0:
                    continue
                lot = symbols[s]["lot_size"]
                price = float(frames[s].at[day, "open"]) * (1 + slip)
                qty = min(delta, math.floor(st["cash"] / price / lot) * lot)
                while (
                    qty > 0
                    and qty * price + sum(charge(market, day, "BUY", qty, price, spec).values())
                    > st["cash"] + 1e-9
                ):
                    qty -= lot
                if qty <= 0:
                    continue
                price, fee = execution(s, "BUY", qty, day, stamp, prior)
                spent = qty * price + fee
                st["cash"] -= spent
                if s not in st["positions"]:
                    st["positions"][s] = {
                        "quantity": 0,
                        "cycle_net": 0.0,
                        "remaining_cost": 0.0,
                        "entry_ms": stamp,
                        "entry_price": price,
                        "entry_regime": str(regimes.at[prior]),
                    }
                p = st["positions"][s]
                p["quantity"] += qty
                p["cycle_net"] -= spent
                p["remaining_cost"] += spent
            # Requests are exact signed net share changes, including funding limits.
            today_fills = {
                f["symbol"]: f for f in reversed(st["fills"]) if f["day"] == str(day.date())
            }
            for request in reversed(st["requests"]):
                if request["day"] != str(day.date()):
                    break
                f = today_fills.get(request["symbol"])
                request["filled"] = (
                    (f["quantity"] if f["side"] == "BUY" else -f["quantity"]) if f else 0
                )
        balances = st["cash"] + st["receivables"] + sum(x["amount"] for x in st["unsettled"])
        equity = balances + sum(
            p["quantity"] * float(frames[s].at[day, "close"]) for s, p in st["positions"].items()
        )
        low = balances + sum(
            p["quantity"] * float(frames[s].at[day, "low"]) for s, p in st["positions"].items()
        )
        reserve = sum(
            p["quantity"] * float(frames[s].at[day, "close"]) * slip
            + sum(
                charge(
                    market,
                    day,
                    "SELL",
                    p["quantity"],
                    float(frames[s].at[day, "close"]) * (1 - slip),
                    spec,
                ).values()
            )
            for s, p in st["positions"].items()
        )
        # Daily lows are an envelope of retained holdings, not synchronized ticks.
        # The cash ledger includes realized opening gaps and all execution costs.
        prior_equity = st["equity_curve"][-1]["equity"]
        st["max_low_dd"] = max(
            st["max_low_dd"], 1 - (low - reserve) / st["peak"], 1 - equity / st["peak"]
        )
        st["peak"] = max(st["peak"], equity)
        dd = 1 - equity / st["peak"]
        if policy == "finite_state":
            if dd >= 0.15 and not st["halted"]:
                st["halted"], st["halt_until"] = True, i + 20
            elif st["halted"] and i >= st["halt_until"] and dd < 0.10:
                st["halted"] = False
        exposure = (equity - balances) / equity
        st["equity_curve"].append(
            {
                "clock": int(session.close.timestamp() * 1000),
                "day": str(day.date()),
                "equity": equity,
                "drawdown": dd,
                "low_equity": low,
                "exit_reserve": reserve,
                "cash": st["cash"],
                "unsettled": sum(x["amount"] for x in st["unsettled"]),
                "receivables": st["receivables"],
                "regime": str(regimes.at[prior]) if prior is not None else "unknown",
                "halted": st["halted"],
                "exposure": exposure,
                "daily_return": equity / prior_equity - 1,
            }
        )
        st["last_day"] = str(day.date())
        if st["cash"] < -1e-7:
            raise ValueError("Cash borrowing forbidden")
    return st


def statistics(st):
    curve = st["equity_curve"]
    initial = st["binding"]["spec"]["initial_cash"]
    ending, reserve = curve[-1]["equity"], curve[-1]["exit_reserve"]
    net = np.array([t["net_pnl"] for t in st["trades"]], dtype=float)
    returns = np.array([r["daily_return"] for r in curve[1:]])
    losses, wins = -net[net < 0].sum(), net[net > 0].sum()
    n, k = len(net), int((net > 0).sum())
    wilson = None
    if n:
        center = (k / n + 1.96**2 / (2 * n)) / (1 + 1.96**2 / n)
        half = 1.96 * math.sqrt(k / n * (1 - k / n) / n + 1.96**2 / (4 * n * n)) / (1 + 1.96**2 / n)
        wilson = [center - half, center + half]
    days = (curve[-1]["clock"] - curve[0]["clock"]) / 86400000
    periods = {}
    first = pd.Timestamp(curve[1]["day"])
    for r in curve[1:]:
        for key in [
            r["day"][:4],
            f"90d_{(pd.Timestamp(r['day']) - first).days // 90:02d}",
            "regime_" + r["regime"],
        ]:
            group = periods.setdefault(key, {"growth": 1.0, "sessions": 0})
            group["growth"] *= 1 + r["daily_return"]
            group["sessions"] += 1
    fees = sum(st["fees"].values())
    std = returns.std(ddof=1) if len(returns) > 1 else 0
    closed = float(net.sum())
    months = len(
        {pd.Timestamp(t["entry_ms"], unit="ms", tz="UTC").strftime("%Y-%m") for t in st["trades"]}
    )
    clusters = {}
    for t in st["trades"]:
        month = pd.Timestamp(t["entry_ms"], unit="ms", tz="UTC").strftime("%Y-%m")
        clusters.setdefault(month, []).append(t["net_pnl"])
    interval = None
    if len(clusters) >= 4:
        # Resample entry months together, keeping correlated ETF cycles grouped.
        # Descriptive only: adjacent months and modules may still be dependent.
        rng = np.random.default_rng(20261026)
        groups = list(clusters.values())
        samples = [
            np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean()
            for _ in range(1000)
        ]
        interval = np.quantile(samples, [0.025, 0.975]).tolist()
    market_value = (
        ending - st["cash"] - st["receivables"] - sum(x["amount"] for x in st["unsettled"])
    )
    return {
        "net_return": ending / initial - 1,
        "net_pnl": ending - initial,
        "ending_equity": ending,
        "annualized_return": ((ending - reserve) / initial) ** (365.25 / days) - 1
        if days > 0
        else None,
        "sharpe_zero_rf_252": float(returns.mean() / std * np.sqrt(252)) if std > 0 else None,
        "max_drawdown": st["max_low_dd"],
        "close_max_drawdown": max(r["drawdown"] for r in curve),
        "win_rate": k / n if n else None,
        "win_rate_wilson95": wilson,
        "closed_trades": n,
        "entry_month_clusters": months,
        "profit_factor": float(wins / losses) if losses else None,
        "realized_closed_net_pnl": closed,
        "open_cycle_pnl": ending - initial - closed,
        "unrealized_mark_pnl": market_value
        - sum(p["remaining_cost"] for p in st["positions"].values()),
        "mean_closed_pnl": float(net.mean()) if n else None,
        "mean_closed_pnl_month_cluster95": interval,
        "worst_closed_cycle": float(net.min()) if n else None,
        "closed_payoff_ratio": float(net[net > 0].mean() / -net[net < 0].mean())
        if wins and losses
        else None,
        "exit_reserve": reserve,
        "net_return_after_exit_reserve": (ending - reserve) / initial - 1,
        "fees": fees,
        "fee_components": st["fees"],
        "slippage": sum(f["slippage"] for f in st["fills"]),
        "turnover_initial_capital": sum(f["quantity"] * f["price"] for f in st["fills"]) / initial,
        "requested_shares": sum(abs(r["requested"]) for r in st["requests"]),
        "filled_shares": sum(abs(r["filled"]) for r in st["requests"]),
        "capacity_restricted_requests": sum(
            abs(r["volume_capped"]) < abs(r["requested"]) for r in st["requests"]
        ),
        "unfilled_exit_requests": sum(
            r["requested"] < 0 and r["filled"] > r["requested"] for r in st["requests"]
        ),
        "mean_exposure": float(np.mean([r["exposure"] for r in curve[1:]])),
        "halted_sessions": sum(r["halted"] for r in curve[1:]),
        "sessions": len(returns),
        "module_sample_status": "minimum_diagnostic_count_only"
        if n >= 30 and months >= 4
        else "insufficient",
        "periods": {
            k: {"net_return": v["growth"] - 1, "sessions": v["sessions"]}
            for k, v in periods.items()
        },
    }
