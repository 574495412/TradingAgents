"""Independent two-leg ledger, ownership, capacity, funding and NAV audit.

Imports neither the execution engine nor its metric implementation. Pair PnL
includes both legs and any closed-leg settlement retained by an open campaign.
"""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, sha


def audit_account(run, row, bars, rates, signals):  # noqa: C901
    folder = run / "accounts" / row["id"]
    report = read(folder / "result.json")
    for name, digest in report["source_hashes"].items():
        if Path(name).name != name or sha(folder / name) != digest:
            raise ValueError("Pair account file changed")
    curve = pd.read_parquet(folder / "equity_5m.parquet")
    times = pd.DatetimeIndex(curve.date)
    if not times.equals(
        pd.date_range(report["window"]["start"], report["window"]["end"], freq="5min", tz="UTC")
    ):
        raise ValueError("Pair NAV calendar incomplete")
    config = report["config"]
    fee = config["fee_bps"] / 10000
    slip = config["slippage_bps"] / 10000
    events = read(folder / "events.json")
    requests = read(folder / "requests.json")
    expected_signals = {
        pd.Timestamp(r["time"]): {p["pair_key"]: p for p in r["pairs"]} for r in signals
    }
    submitted = {}
    for r in requests:
        clock = pd.Timestamp(r["time"])
        signal = expected_signals[clock][r["pair_key"]]
        if r["signal"] != signal or not signal["entry_eligible"]:
            raise ValueError("Request changed frozen signal")
        if (clock, r["pair_key"]) in submitted:
            raise ValueError("Duplicate pair request")
        submitted[(clock, r["pair_key"])] = r
        cap_fraction = 1.0
        for s, q in r["requested_q"].items():
            ref = float(bars[s].loc[clock - pd.Timedelta(minutes=5), "close"])
            weight = (signal["beta"] if s == signal["x"] else 1) / (1 + signal["beta"])
            if (
                abs(r["reference_prices"][s] - ref) > 1e-10
                or abs(q * ref - r["requested_notional"] * weight) > 1e-6
            ):
                raise ValueError("Request used future marks or incorrect hedge sizes")
            history = bars[s].loc[clock - pd.Timedelta(hours=1) : clock - pd.Timedelta(minutes=5)]
            cap_fraction = min(
                cap_fraction,
                config["participation"] * float(history.quote_volume.sum()) / 12 / ref / q,
            )
        for s, q in r["requested_q"].items():
            if abs(r["submitted_q"][s] - q * max(0, cap_fraction)) > 1e-8:
                raise ValueError("Submission used noncausal volume")
    n = len(times)
    changes = np.zeros(n)
    floating = np.zeros(n)
    gross = np.zeros(n)
    net = np.zeros(n)
    qdelta = {s: np.zeros(n) for s in bars}
    basis = {s: np.zeros(n) for s in bars}
    alive = {s: np.zeros(n) for s in bars}
    positions = {}
    ownership = {}
    pair_symbols = {}
    lifecycle = defaultdict(float)
    funding_seen = set()
    entries = defaultdict(dict)
    matched = {}
    checked_groups = set()
    cash_now = float(config["capital"])
    maximum_participation = 0.0
    last = times[0]
    for e in events:
        clock = pd.Timestamp(e["time"])
        s = e["symbol"]
        pid = e["position_id"]
        key = (s, pid)
        if clock < last:
            raise ValueError("Pair ledger clock reversed")
        last = clock
        if e["type"] == "funding":
            p = positions[key]
            rate = rates[s].loc[clock]
            delta = -p["side"] * p["q"] * rate["mark"] * rate["funding_rate"]
            if abs(p["q"] - e["quantity"]) > 1e-8:
                raise ValueError("Funding inventory differs")
            funding_seen.add((s, clock))
            effective = clock + pd.Timedelta(minutes=5)
        else:
            previous = clock - pd.Timedelta(minutes=5)
            bar = bars[s].loc[previous]
            q = e["quantity"]
            side = e["side"]
            if pd.Timestamp(e["submitted_at"]) >= clock or q <= 0 or not math.isfinite(q):
                raise ValueError("Invalid or noncausal pair fill")
            price = float(bar.close * (1 + side * slip * (1 if e["type"] == "entry" else -1)))
            if abs(price - e["price"]) > 1e-8 or abs(q * price * fee - e["fee"]) > 1e-7:
                raise ValueError("Pair fill price/fee differs")
            part = q * price / float(bar.quote_volume) if bar.quote_volume > 0 else math.inf
            maximum_participation = max(maximum_participation, part)
            if part > config["participation"] + 1e-12:
                raise ValueError("Pair capacity exceeded")
            i = times.get_loc(clock)
            if e["type"] == "entry":
                intent = submitted[(pd.Timestamp(e["submitted_at"]), e["pair_key"])]
                if e["side"] != intent["sides"][s]:
                    raise ValueError("Entry side differs")
                if pid not in checked_groups:
                    # Compute the independent actual-volume and after-cost risk
                    # clipping before this campaign's first fee changes cash.
                    actual = {}
                    prices = {}
                    for symbol, wanted in intent["submitted_q"].items():
                        b = bars[symbol].loc[previous]
                        prices[symbol] = float(b.close * (1 + intent["sides"][symbol] * slip))
                        actual[symbol] = min(
                            wanted, config["participation"] * float(b.quote_volume) / prices[symbol]
                        )
                    marked = cash_now
                    held_gross = 0.0
                    for (symbol, _), p in positions.items():
                        close = float(bars[symbol].loc[previous, "close"])
                        marked += p["side"] * p["q"] * (close - p["price"]) - p["q"] * close * (
                            fee + slip
                        )
                        held_gross += p["q"] * close
                    amount = sum(
                        qty * float(bars[symbol].loc[previous, "close"])
                        for symbol, qty in actual.items()
                    )
                    drag = sum(
                        qty
                        * (
                            abs(prices[symbol] - float(bars[symbol].loc[previous, "close"]))
                            + prices[symbol] * fee
                            + float(bars[symbol].loc[previous, "close"]) * (fee + slip)
                        )
                        for symbol, qty in actual.items()
                    )
                    factor = (
                        min(
                            1.0,
                            max(
                                0.0,
                                (marked * config["gross"] - held_gross)
                                / (amount + config["gross"] * drag),
                            ),
                        )
                        if amount > 0
                        else 0
                    )
                    expected = {symbol: qty * factor for symbol, qty in actual.items()}
                    observed = {
                        v["symbol"]: v["quantity"]
                        for v in events
                        if v["type"] == "entry" and v["position_id"] == pid
                    }
                    if any(
                        abs(expected[symbol] - observed.get(symbol, 0)) > 1e-8
                        for symbol in expected
                    ):
                        raise ValueError("Independent two-leg fill/risk budget differs")
                    fractions = [
                        observed.get(symbol, 0) / wanted if wanted > 0 else 0
                        for symbol, wanted in intent["submitted_q"].items()
                    ]
                    matched[pid] = bool(
                        min(fractions) > 0 and max(fractions) - min(fractions) <= 1e-8
                    )
                    if matched[pid] != e["hedge_matched"]:
                        raise ValueError("Hedge failure mislabelled")
                    pair_symbols[pid] = set(intent["submitted_q"])
                    for symbol in pair_symbols[pid]:
                        if symbol in ownership and ownership[symbol] != pid:
                            raise ValueError("Pair reused an owned asset")
                        ownership[symbol] = pid
                    checked_groups.add(pid)
                if key in positions:
                    raise ValueError("Unexpected pair topup/reuse")
                positions[key] = {"q": q, "side": side, "price": price}
                entries[pid][s] = q
                alive[s][i] += 1
                qdelta[s][i] += side * q
                basis[s][i] += side * q * price
                delta = -q * price * fee
            elif e["type"] == "exit":
                p = positions[key]
                if side != p["side"] or q > p["q"] + 1e-8:
                    raise ValueError("Pair exit exceeds inventory")
                delta = side * q * (price - p["price"]) - q * price * fee
                qdelta[s][i] -= side * q
                basis[s][i] -= side * q * p["price"]
                p["q"] -= q
                if p["q"] < 1e-12:
                    alive[s][i] -= 1
                    del positions[key]
                if not any(k[1] == pid for k in positions):
                    for symbol in pair_symbols[pid]:
                        ownership.pop(symbol)
            else:
                raise ValueError("Unknown pair event")
            effective = clock
        if abs(delta - e["cash_delta"]) > 1e-6:
            raise ValueError("Pair cashflow differs")
        changes[times.get_loc(effective)] += delta
        cash_now += delta
        lifecycle[key] += delta
    for s in bars:
        quantity = np.cumsum(qdelta[s])
        book = np.cumsum(basis[s])
        close = bars[s].close.reindex(times - pd.Timedelta(minutes=5)).to_numpy()
        floating += quantity * close - book - abs(quantity) * close * (fee + slip)
        gross += abs(quantity) * close
        net += quantity * close
        active = np.cumsum(alive[s])
        for clock in rates[s].index:
            if times[0] <= clock < times[-1] and (
                (active[times.get_loc(clock)] > 0) != ((s, clock) in funding_seen)
            ):
                raise ValueError("Missing/extra pair funding event")
    cash = config["capital"] + np.cumsum(changes)
    nav = cash + floating
    error = float(np.max(abs(nav - curve.equity.to_numpy())))
    for actual, expected in (
        (nav, curve.equity),
        (cash, curve.cash),
        (gross, curve.gross_exposure),
        (net, curve.net_exposure),
    ):
        if float(np.max(abs(actual - expected.to_numpy()))) > 1e-5:
            raise ValueError("Independent pair NAV/cash/exposures differ")
    for s, p in report["open_positions"].items():
        actual = positions.pop((s, p["id"]))
        if abs(actual["q"] - p["q"]) > 1e-8 or actual["price"] != p["price"]:
            raise ValueError("Open leg differs")
    if positions:
        raise ValueError("Open leg omitted")
    closed = []
    closed_ids = set()
    matched_closed = 0
    for t in report["trades"]:
        pid = t["position_id"]
        if pid in closed_ids:
            raise ValueError("Pair counted twice")
        closed_ids.add(pid)
        pnl = math.fsum(lifecycle[(s, pid)] for s in pair_symbols[pid])
        if abs(pnl - t["net_pnl"]) > 1e-6 or t["hedge_matched"] != matched[pid]:
            raise ValueError("Closed pair PnL/hedge status differs")
        for s, v in t["leg_net_pnl"].items():
            if abs(v - lifecycle[(s, pid)]) > 1e-6:
                raise ValueError("Closed leg PnL differs")
        matched_closed += matched[pid]
        closed.append(pnl)
    open_settled = 0.0
    open_ids = set()
    for p in report["open_pairs"].values():
        pid = p["id"]
        open_ids.add(pid)
        pnl = math.fsum(lifecycle[(s, pid)] for s in p["symbols"])
        if abs(pnl - p["net_pnl"]) > 1e-6:
            raise ValueError("Open campaign lost closed-leg settlement")
        open_settled += pnl
    if closed_ids & open_ids or closed_ids | open_ids != set(entries):
        raise ValueError("Pair lifecycle omitted or duplicated")
    if abs(math.fsum(closed) + open_settled + floating[-1] - (nav[-1] - config["capital"])) > 1e-5:
        raise ValueError("Closed/open pair accounting does not reconcile")
    daily = nav[(times.hour == 0) & (times.minute == 0)]
    returns = np.diff(daily) / daily[:-1]
    sd = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0
    profit = math.fsum(max(p, 0) for p in closed)
    loss = -math.fsum(min(p, 0) for p in closed)
    values = {
        "net_return": float(nav[-1] / config["capital"] - 1),
        "max_drawdown": float(np.max(1 - nav / np.maximum.accumulate(nav))),
        "sharpe": float(np.mean(returns) / sd * math.sqrt(365)) if sd > 1e-15 else None,
        "closed_trades": len(closed),
        "matched_closed_pairs": matched_closed,
        "win_rate": sum(p > 0 for p in closed) / len(closed) if closed else None,
        "profit_factor": profit / loss if loss else None,
        "closed_net_pnl": math.fsum(closed),
        "open_campaign_settled_net_pnl": open_settled,
        "fees": math.fsum(e.get("fee", 0) for e in events),
        "funding_net": math.fsum(e["cash_delta"] for e in events if e["type"] == "funding"),
    }
    for name, value in values.items():
        reported = report["metrics"][name]
        if (
            (value is None) != (reported is None)
            or value is not None
            and abs(value - reported) > 1e-7
        ):
            raise ValueError("Independent pair metric differs: " + name)
    return {
        "id": row["id"],
        "passed": True,
        "nav_rows": n,
        "max_nav_error": error,
        "maximum_participation": maximum_participation,
        "campaigns": len(entries),
        "matched_closed_pairs": matched_closed,
        "complete_lifecycles": len(closed),
        "funding_events": len(funding_seen),
        "result_sha256": sha(folder / "result.json"),
    }
