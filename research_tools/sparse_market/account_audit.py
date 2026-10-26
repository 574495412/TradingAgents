"""Independent sparse campaign audit with exact settlement and identity ownership.

Does not import the replay engine or metrics. Multiple fills may grow an existing
same-side campaign; cost basis and cash are reconstructed from the fill ledger.
The earlier frozen verifier remains unchanged.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, sha


def policy_at(directives, symbol, clock):
    visible = [r for r in directives.get(symbol, []) if pd.Timestamp(r["known_at"]) <= clock]
    return (
        visible[-1]
        if visible
        else {
            "episode_key": symbol + ":observed_initial",
            "blocked": False,
            "entry_stop_at": None,
            "terminal_at": None,
        }
    )


def blocked(row, clock):
    return row.get("blocked", False) or any(
        row.get(k) is not None and clock >= pd.Timestamp(row[k])
        for k in ["entry_stop_at", "terminal_at"]
    )


def audit_account(run, row, bars, rates, scores, directives=None):  # noqa: C901
    folder = run / "accounts" / row["id"]
    report = read(folder / "result.json")
    for name, expected in report["source_hashes"].items():
        if Path(name).name != name or sha(folder / name) != expected:
            raise ValueError("Account receipt changed")
    curve = pd.read_parquet(folder / "equity_5m.parquet")
    times = pd.DatetimeIndex(curve.date)
    if not times.equals(
        pd.date_range(report["window"]["start"], report["window"]["end"], freq="5min", tz="UTC")
    ):
        raise ValueError("NAV calendar incomplete")
    events = read(folder / "events.json")
    requests = {(e["time"], e["symbol"], e["side"]): e for e in read(folder / "requests.json")}
    directives = directives or {}
    positions = {}
    intervals = {}
    lifecycles = {}
    funding_seen = set()
    changes = np.zeros(len(times))
    floating = np.zeros(len(times))
    gross = np.zeros(len(times))
    qdelta = {s: np.zeros(len(times)) for s in bars}
    alive = {s: np.zeros(len(times), dtype=int) for s in bars}
    basis = {s: np.zeros(len(times)) for s in bars}
    config = report["config"]
    fee = config["fee_bps"] / 10000
    slip = config["slippage_bps"] / 10000
    maximum_participation = 0.0
    event_time = times[0]
    for e in events:
        clock = pd.Timestamp(e["time"])
        s = e["symbol"]
        key = (s, e["position_id"])
        if clock < event_time:
            raise ValueError("Ledger clock reversed")
        event_time = clock
        if e["type"] == "funding":
            p = positions[key]
            actual = rates[s].loc[clock]
            delta = -p["side"] * p["q"] * actual["mark"] * actual["funding_rate"]
            effective = pd.Timestamp(e["effective_time"])
            if effective != clock.ceil("5min"):
                raise ValueError("Exact funding valuation clock differs")
            funding_seen.add((s, clock))
            if abs(p["q"] - e["quantity"]) > 1e-8:
                raise ValueError("Funding inventory differs")
        else:
            b = bars[s].loc[clock - pd.Timedelta(minutes=5)]
            if not b.price_usable or not b.execution_liquidity_usable or b.quote_volume <= 0:
                raise ValueError("Fill uses unavailable execution source")
            if pd.Timestamp(e["submitted_at"]) >= clock:
                raise ValueError("Noncausal fill timestamp")
            q, side = e["quantity"], e["side"]
            price = b.close * (1 + side * slip * (1 if e["type"] == "entry" else -1))
            if not math.isfinite(q) or q <= 0 or abs(e["price"] - price) > 1e-8:
                raise ValueError("Fill price or quantity differs")
            participation = q * price / b.quote_volume if b.quote_volume > 0 else math.inf
            maximum_participation = max(maximum_participation, participation)
            if participation > config["participation"] + 1e-12:
                raise ValueError("Capacity limit exceeded")
            i = times.get_loc(clock)
            if e["type"] == "entry":
                request = requests[(e["submitted_at"], s, e["side"])]
                submitted = pd.Timestamp(e["submitted_at"])
                prior = bars[s].reindex(
                    pd.date_range(
                        submitted - pd.Timedelta(hours=1), submitted, freq="5min", inclusive="left"
                    )
                )
                reference = float(bars[s].loc[submitted - pd.Timedelta(minutes=5), "close"])
                if (
                    not prior.price_usable.eq(True).all()
                    or not np.isfinite(prior.quote_volume).all()
                    or prior.quote_volume.lt(0).any()
                    or (prior.quote_volume.gt(0) & ~prior.execution_liquidity_usable.eq(True)).any()
                    or abs(reference - request["reference_price"]) > 1e-10
                    or request["submitted_q"]
                    > config["participation"] * float(prior.quote_volume.sum()) / 12 / reference
                    + 1e-8
                ):
                    raise ValueError("Prebar request exceeds real known liquidity")
                formed = pd.Timestamp(request.get("source_signal_at", e["submitted_at"]))
                before_policy = policy_at(directives, s, pd.Timestamp(e["submitted_at"]))
                fill_policy = policy_at(directives, s, clock)
                if (
                    before_policy["episode_key"] != fill_policy["episode_key"]
                    or blocked(fill_policy, clock)
                    or blocked(before_policy, pd.Timestamp(e["submitted_at"]))
                ):
                    raise ValueError(
                        "Entry changes identity or violates known lifecycle restriction"
                    )
                if (
                    formed > pd.Timestamp(e["submitted_at"])
                    or not bool(scores.loc[formed, "rebalance"])
                    or np.sign(scores.loc[formed, "target_" + s]) != e["side"]
                ):
                    raise ValueError("Entry lacks a prior registered weekly signal")
                if key in positions:
                    p = positions[key]
                    if not e.get("addition") or side != p["side"]:
                        raise ValueError("Unregistered addition or mixed campaign direction")
                    p["price"] = (p["q"] * p["price"] + q * price) / (p["q"] + q)
                    p["q"] += q
                else:
                    if e.get("addition") or any(k[0] == s for k in positions):
                        raise ValueError("Unexpected overlapping campaign")
                    positions[key] = {"q": q, "price": price, "side": side}
                    intervals[key] = {
                        "symbol": s,
                        "open": clock,
                        "close": times[-1] + pd.Timedelta(nanoseconds=1),
                        "episode": fill_policy["episode_key"],
                    }
                    alive[s][i] += 1
                qdelta[s][i] += side * q
                basis[s][i] += side * q * price
                delta = -q * price * fee
            elif e["type"] == "exit":
                p = positions[key]
                if side != p["side"] or q > p["q"] + 1e-8:
                    raise ValueError("Exit exceeds inventory")
                delta = side * q * (price - p["price"]) - q * price * fee
                qdelta[s][i] -= side * q
                basis[s][i] -= side * q * p["price"]
                p["q"] -= q
                if p["q"] < 1e-12:
                    alive[s][i] -= 1
                    intervals[key]["close"] = clock
                    del positions[key]
            else:
                raise ValueError("Unknown account event")
            if abs(e["fee"] - q * price * fee) > 1e-7:
                raise ValueError("Fee differs")
            effective = clock
        if abs(delta - e["cash_delta"]) > 1e-6:
            raise ValueError("Cash flow differs")
        changes[times.get_loc(effective)] += delta
        lifecycles[key] = lifecycles.get(key, 0) + delta
    for s in bars:
        q = np.cumsum(qdelta[s])
        book = np.cumsum(basis[s])
        close = bars[s].close.reindex(times - pd.Timedelta(minutes=5)).to_numpy()
        active = np.cumsum(alive[s])
        usable = bars[s].price_usable.reindex(times - pd.Timedelta(minutes=5)).eq(True).to_numpy()
        if ((active > 0) & (~np.isfinite(close) | (close <= 0) | ~usable)).any():
            raise ValueError("Missing held marks cannot become a zero contribution")
        floating += np.where(active > 0, q * close - book - abs(q) * close * (fee + slip), 0.0)
        gross += np.where(active > 0, abs(q) * close, 0.0)
        for clock in rates[s].index:
            if times[0] < clock <= times[-1]:
                before = int(times.searchsorted(clock, side="left")) - 1
                expected = before >= 0 and active[before] > 0
                if expected != ((s, clock) in funding_seen):
                    raise ValueError("Missing or extra exact funding")
    for interval in intervals.values():
        name = interval["symbol"]
        for directive in directives.get(name, []):
            known = pd.Timestamp(directive["known_at"])
            candidates = [known]
            if directive.get("terminal_at"):
                candidates.append(max(known, pd.Timestamp(directive["terminal_at"])))
            for clock in candidates:
                if interval["open"] < clock <= interval["close"] and clock <= times[-1]:
                    current = policy_at(directives, name, clock)
                    if current["episode_key"] != interval["episode"] or (
                        current.get("terminal_at") and clock >= pd.Timestamp(current["terminal_at"])
                    ):
                        raise ValueError(
                            "Residual inventory crosses an identity or unknown terminal cash boundary"
                        )
    cash = config["capital"] + np.cumsum(changes)
    nav = cash + floating
    error = float(np.max(abs(nav - curve.equity.to_numpy())))
    if (
        error > 1e-5
        or np.max(abs(cash - curve.cash.to_numpy())) > 1e-5
        or np.max(abs(gross - curve.gross_exposure.to_numpy())) > 1e-5
    ):
        raise ValueError("Independent NAV/cash/gross reconstruction differs")
    for s, p in report["open_positions"].items():
        key = (s, p["id"])
        actual = positions.pop(key)
        if abs(actual["q"] - p["q"]) > 1e-8 or actual["price"] != p["price"]:
            raise ValueError("Open inventory differs")
    if positions:
        raise ValueError("Open positions omitted")
    closed = []
    for t in report["trades"]:
        pnl = lifecycles[(t["symbol"], t["position_id"])]
        if abs(pnl - t["net_pnl"]) > 1e-6:
            raise ValueError("Complete lifecycle differs")
        closed.append(pnl)
    m = report["metrics"]
    profit = math.fsum(max(v, 0) for v in closed)
    loss = -math.fsum(min(v, 0) for v in closed)
    pf = profit / loss if loss else None
    win = sum(v > 0 for v in closed) / len(closed) if closed else None
    returns = (
        np.diff(nav[(times.hour == 0) & (times.minute == 0)])
        / nav[(times.hour == 0) & (times.minute == 0)][:-1]
    )
    sd = float(np.std(returns, ddof=1))
    sharpe = float(np.mean(returns) / sd * math.sqrt(365)) if sd > 1e-15 else None
    values = {
        "net_return": float(nav[-1] / config["capital"] - 1),
        "max_drawdown": float(np.max(1 - nav / np.maximum.accumulate(nav))),
        "closed_trades": len(closed),
        "profit_factor": pf,
        "win_rate": win,
        "sharpe": sharpe,
        "closed_net_pnl": math.fsum(closed),
        "fees": math.fsum(e.get("fee", 0) for e in events),
        "funding_net": math.fsum(e["cash_delta"] for e in events if e["type"] == "funding"),
    }
    for name, value in values.items():
        if (
            (value is None) != (m[name] is None)
            or value is not None
            and abs(value - m[name]) > 1e-7
        ):
            raise ValueError("Independent metric differs: " + name)
    return {
        "id": row["id"],
        "passed": True,
        "nav_rows": len(nav),
        "max_nav_error": error,
        "maximum_participation": maximum_participation,
        "funding_events": len(funding_seen),
        "complete_lifecycles": len(closed),
        "result_sha256": sha(folder / "result.json"),
    }
