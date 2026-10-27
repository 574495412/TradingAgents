"""Reconstruct actual two-wallet flows and evidenced dated settlement without engine calls."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, sha


def audit_account(folder, spot, perpetual, marks, funding, contracts):  # noqa: C901
    report = read(folder / "result.json")
    for name, digest in report["source_hashes"].items():
        if Path(name).name != name or sha(folder / name) != digest:
            raise ValueError("Account evidence changed")
    curve = pd.read_parquet(folder / "equity_5m.parquet")
    times = pd.DatetimeIndex(curve.date)
    config, window = report["config"], report["window"]
    if not times.equals(pd.date_range(window["start"], window["end"], freq="5min", tz="UTC")):
        raise ValueError("Account curve has calendar gaps")
    n = len(curve)
    quote_delta, future_delta, negative_funding = np.zeros(n), np.zeros(n), np.zeros(n)
    transfer_future_delta = np.zeros(n)
    spot_delta = {s: np.zeros(n) for s in spot}
    short_delta = {s: np.zeros(n) for s in spot}
    basis_delta = {s: np.zeros(n) for s in spot}
    positions, flows = {}, {}
    fees = funding_net = max_participation = 0.0
    seen_funding = set()
    seen_settlements = set()
    short_closed, spot_closed = {}, {}
    if any(len(f) for f in funding.values()):
        raise ValueError("Dated account must not have funding inputs")
    sf, pf, slip = (
        config["spot_fee_bps"] / 10000,
        config["perpetual_fee_bps"] / 10000,
        config["slippage_bps"] / 10000,
    )
    events = read(folder / "events.json")
    decisions = {pd.Timestamp(r["time"]): r for r in read(folder / "signals.json")}
    for e in events:
        t = pd.Timestamp(e["time"])
        effective = pd.Timestamp(e["effective_time"]) if e["type"] == "transfer" else t.ceil("5min")
        i = times.get_loc(effective)
        if e["type"] == "transfer":
            if positions:
                raise ValueError("Spot mark gains illegally transferred while held")
            if e["quote_delta"] != -e["futures_delta"] or e["amount"] != e["futures_delta"]:
                raise ValueError("Wallet transfer conservation failed")
            quote_delta[i] += e["quote_delta"]
            future_delta[i] += e["futures_delta"]
            transfer_future_delta[i] += e["futures_delta"]
            continue
        s, identity = e["symbol"], e["position_id"]
        key = (s, identity)
        if e["type"] == "settlement":
            c = contracts[s]
            p = positions[key]
            if (
                key in seen_settlements
                or t != pd.Timestamp(c["expiry"])
                or e["contract"] != c["symbol"]
                or e["price"] != c["settlement_price"]
                or e["quantity"] != p["short"]
                or p["short"] <= 0
            ):
                raise ValueError("Settlement identity, ownership, price or exact clock differs")
            seen_settlements.add(key)
            q = p["short"]
            fee = q * c["settlement_price"] * config["settlement_fee_bps"] / 10000
            amount = q * (p["entry"] - c["settlement_price"]) - fee
            future_delta[i] += amount
            short_delta[s][i] -= q
            basis_delta[s][i] -= q * p["entry"]
            p["short"] = 0.0
            short_closed[s] = i
            fees += fee
            if fee != e["fee"]:
                raise ValueError("Settlement fee is not the declared model taker rate")
        elif e["type"] == "funding":
            p = positions[key]
            raw = funding[s].loc[t]
            amount = p["short"] * float(raw.mark) * float(raw.funding_rate)
            if (
                (key, t) in seen_funding
                or abs(p["short"] - e["quantity"]) > 1e-9
                or e["mark"] != raw.mark
                or e["rate"] != raw.funding_rate
            ):
                raise ValueError("Funding duplicate, mark, rate or inventory differs")
            if pd.Timestamp(e["effective_time"]) != effective:
                raise ValueError("Funding clock was rounded into earlier knowledge")
            seen_funding.add((key, t))
            future_delta[i] += amount
            negative_funding[i] += min(0.0, amount)
            funding_net += amount
        else:
            market, q = e["market"], e["quantity"]
            if market == "perp" and t >= pd.Timestamp(contracts[s]["expiry"]):
                raise ValueError("Old future filled at or after delivery")
            if e["type"] == "entry" and t >= pd.Timestamp(contracts[s]["entry_stop"]):
                raise ValueError("Reduce-only boundary violated")
            if pd.Timestamp(e["submitted_at"]) > t - pd.Timedelta(minutes=5) or q <= 0:
                raise ValueError("Noncausal or invalid fill")
            bar = (spot if market == "spot" else perpetual)[s].loc[t - pd.Timedelta(minutes=5)]
            sign = (1 if market == "spot" else -1) * (1 if e["type"] == "entry" else -1)
            price = float(bar.close) * (1 + sign * slip)
            gross = q / (1 - sf) if market == "spot" and e["type"] == "entry" else q
            fee = (
                (gross - q) * price
                if market == "spot" and e["type"] == "entry"
                else q * price * (sf if market == "spot" else pf)
            )
            if abs(price - e["price"]) > 1e-8 or abs(fee - e["fee"]) > 1e-7:
                raise ValueError("Actual side-specific fill price or fee differs")
            if e["type"] == "entry" and abs(gross - e["gross_quantity"]) > 1e-9:
                raise ValueError("Net versus gross base quantity differs")
            participation = gross * price / bar.quote_volume if bar.quote_volume > 0 else math.inf
            max_participation = max(max_participation, participation)
            if participation > config["participation"] + 1e-12:
                raise ValueError("Actual per-leg volume capacity exceeded")
            p = positions.setdefault(key, {"spot": 0.0, "short": 0.0, "entry": 0.0})
            if e["type"] == "entry":
                signal = decisions[pd.Timestamp(e["submitted_at"])]["assets"][s]
                if not signal["entry"] or signal["exit"]:
                    raise ValueError("Entry outside registered observable cost condition")
                if market == "spot":
                    amount = -gross * price
                    p["spot"] += q
                    spot_delta[s][i] += q
                    quote_delta[i] += amount
                else:
                    amount = -fee
                    p["entry"] = (p["short"] * p["entry"] + q * price) / (p["short"] + q)
                    p["short"] += q
                    short_delta[s][i] += q
                    basis_delta[s][i] += q * price
                    future_delta[i] += amount
            elif e["type"] == "exit":
                if market == "spot":
                    if q > p["spot"] + 1e-9:
                        raise ValueError("Spot sale exceeds actual base inventory")
                    amount = q * price - fee
                    p["spot"] -= q
                    spot_delta[s][i] -= q
                    quote_delta[i] += amount
                    if abs(p["spot"]) < 1e-12:
                        spot_closed[s] = i
                else:
                    if q > p["short"] + 1e-9:
                        raise ValueError("Perp close exceeds actual short")
                    amount = q * (p["entry"] - price) - fee
                    p["short"] -= q
                    short_delta[s][i] -= q
                    basis_delta[s][i] -= q * p["entry"]
                    future_delta[i] += amount
                    if abs(p["short"]) < 1e-12:
                        short_closed[s] = i
            else:
                raise ValueError("Unknown event type")
            fees += fee
            if abs(p["spot"]) < 1e-10 and abs(p["short"]) < 1e-10:
                del positions[key]
        if abs(amount - e["cash_delta"]) > 1e-7:
            raise ValueError("Cash flow differs from independent reconstruction")
        flows[key] = flows.get(key, 0.0) + amount
    quote = config["capital"] + np.cumsum(quote_delta)
    future = np.cumsum(future_delta)
    assets, floating, reserve, gross = [np.zeros(n) for _ in range(4)]
    envelope_assets, envelope_float, envelope_reserve, envelope_short = [
        np.zeros(n - 1) for _ in range(4)
    ]
    for s in spot:
        sq, pq, basis = (
            np.cumsum(spot_delta[s]),
            np.cumsum(short_delta[s]),
            np.cumsum(basis_delta[s]),
        )
        # Only one registered entry per asset; final close explicitly clears
        # ownership. This avoids pricing an unowned binary cumsum residue with
        # a nonexistent post-expiry mark. Raw price data is never forward-filled.
        if (
            sum(
                e["type"] == "entry" and e.get("market") == "perp" and e.get("symbol") == s
                for e in events
            )
            > 1
        ):
            raise ValueError("Pilot permits one futures entry per asset")
        if s in short_closed:
            pq[short_closed[s] :] = 0.0
            basis[short_closed[s] :] = 0.0
        if s in spot_closed:
            sq[spot_closed[s] :] = 0.0
        if np.any(pq[times >= pd.Timestamp(contracts[s]["expiry"])] != 0):
            raise ValueError("Expired short ownership is not settled")
        expected_paid = set()
        for t in (
            funding[s].loc[(funding[s].index > times[0]) & (funding[s].index <= times[-1])].index
        ):
            before = int(times.searchsorted(t, side="left")) - 1
            if before >= 0 and pq[before] > 1e-10:
                expected_paid.add(t)
        actual_paid = {t for (key, t) in seen_funding if key[0] == s}
        if expected_paid != actual_paid:
            raise ValueError("Held inventory funding missing or unearned settlement included")
        sp = spot[s].close.reindex(times - pd.Timedelta(minutes=5)).to_numpy()
        pp = perpetual[s].close.reindex(times - pd.Timedelta(minutes=5)).to_numpy()
        mp = marks[s].close.reindex(times - pd.Timedelta(minutes=5)).to_numpy()
        low = spot[s].low.reindex(times[:-1]).to_numpy()
        high = marks[s].high.reindex(times[:-1]).to_numpy()
        trade_high = perpetual[s].high.reindex(times[:-1]).to_numpy()
        pp = np.where(pq == 0, 0.0, pp)
        mp = np.where(pq == 0, 0.0, mp)
        high = np.where(pq[:-1] == 0, 0.0, high)
        trade_high = np.where(pq[:-1] == 0, 0.0, trade_high)
        if not all(np.isfinite(v).all() for v in [pp, mp, high, trade_high]):
            raise ValueError("Owned dated inventory has an unknown price")
        assets += sq * sp
        floating += basis - pq * mp
        reserve += sq * sp * (sf + slip) + pq * pp * (pf + slip)
        gross += sq * sp + pq * mp
        envelope_assets += sq[:-1] * low
        envelope_float += basis[:-1] - pq[:-1] * high
        envelope_reserve += sq[:-1] * low * (sf + slip) + pq[:-1] * trade_high * (pf + slip)
        envelope_short += pq[:-1] * high
    nav = quote + future + assets + floating - reserve
    margin_low = future[:-1] + transfer_future_delta[1:] + negative_funding[1:] + envelope_float
    stress_nav = (
        quote[:-1]
        + future[:-1]
        + negative_funding[1:]
        + envelope_assets
        + envelope_float
        - envelope_reserve
    )
    errors = {
        name: float(np.max(np.abs(expected - curve[name].to_numpy())))
        for name, expected in {
            "equity": nav,
            "quote_cash": quote,
            "futures_wallet": future,
            "spot_assets": assets,
            "perp_unrealized": floating,
            "exit_reserve": reserve,
            "gross_notional": gross,
        }.items()
    }
    errors["margin_envelope_balance"] = float(
        np.max(np.abs(margin_low - curve.margin_envelope_balance.iloc[1:].to_numpy()))
    )
    errors["stress_nav_envelope"] = float(
        np.max(np.abs(stress_nav - curve.stress_nav_envelope.iloc[1:].to_numpy()))
    )
    if max(errors.values()) > max(1e-7, config["capital"] * 1e-10):
        raise ValueError("Independent wallet/NAV/risk-envelope path differs " + str(errors))
    possible = bool(
        np.any(
            (envelope_short > 0)
            & (margin_low <= config["maintenance_stress_ratio"] * envelope_short)
        )
    )
    if possible != report["flags"]["possible_liquidation"]:
        raise ValueError("Possible liquidation path suppressed")
    closed = read(folder / "trades.json")
    for trade in closed:
        if abs(trade["net_pnl"] - flows[(trade["symbol"], trade["position_id"])]) > 1e-7:
            raise ValueError("Complete campaign excludes some spot principal or cost")
    actual_positions = read(folder / "positions.json")
    if len(positions) != len(actual_positions):
        raise ValueError("Final open inventory differs")
    for (s, identity), p in positions.items():
        expected = actual_positions[s]
        if (
            identity != expected["id"]
            or abs(p["spot"] - expected["spot_q"]) > 1e-9
            or abs(p["short"] - expected["perp_q"]) > 1e-9
        ):
            raise ValueError("Residual actual base quantities differ")
    closed_pnl = math.fsum(t["net_pnl"] for t in closed)
    open_pnl = math.fsum(flows[k] for k in positions) + assets[-1] + floating[-1] - reserve[-1]
    if abs(closed_pnl + open_pnl - (nav[-1] - config["capital"])) > 1e-7:
        raise ValueError("Closed/open PnL does not reconcile to marked net return")
    m = report["metrics"]
    daily = (
        pd.Series(nav, index=times)
        .loc[(times.hour == 0) & (times.minute == 0)]
        .pct_change()
        .dropna()
    )
    sd = float(daily.std(ddof=1))
    sharpe = float(daily.mean() / sd * math.sqrt(365)) if sd > 0 else None
    checked = {
        "net_return": nav[-1] / config["capital"] - 1,
        "max_drawdown": float(np.max(1 - nav / np.maximum.accumulate(nav))),
        "closed_net_pnl": closed_pnl,
        "fees": fees,
        "funding_net": funding_net,
    }
    for key, expected in checked.items():
        if abs(m[key] - expected) > max(1e-7, config["capital"] * 1e-10):
            raise ValueError("Metric differs " + key)
    if sharpe is not None and abs(sharpe - m["sharpe"]) > 1e-7:
        raise ValueError("Daily Sharpe differs")
    return {
        "passed": True,
        "settlement_events": len(seen_settlements),
        "no_old_future_fill_at_or_after_expiry": True,
        "no_unearned_perpetual_funding": True,
        "curve_rows": n,
        "events": len(events),
        "maximum_nav_error": errors["equity"],
        "maximum_component_error": max(errors.values()),
        "maximum_participation": max_participation,
        "possible_liquidation": possible,
        "completed_campaigns": len(closed),
        "matched_completed_campaigns": sum(t["matched_hedge"] for t in closed),
        "closed_net_pnl": closed_pnl,
        "open_net_pnl_with_exit_reserve": float(open_pnl),
    }
