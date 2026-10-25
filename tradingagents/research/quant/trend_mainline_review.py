"""Causal decision checks and outcome attribution without new parameters or selection."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from trend_mainline_audit import read, save, sha


def adverse_envelope(folder, bars, row, fee):
    """All held positions take their adverse 5m extreme, relative to prior sampled peaks.

    Cross-symbol extrema need not occur simultaneously. This is a conservative
    OHLC envelope diagnostic, not a measured intrabar mark-price drawdown.
    """
    curve = pd.read_parquet(folder / "equity_5m.parquet")
    clocks = pd.DatetimeIndex(curve.date)
    quantity = {s: np.zeros(len(clocks)) for s in bars}
    basis = {s: np.zeros(len(clocks)) for s in bars}
    funding = np.zeros(len(clocks))
    opened = {}
    for event in read(folder / "events.json"):
        i = clocks.get_loc(pd.Timestamp(event["time"]))
        s, kind = event["symbol"], event["type"]
        if kind == "funding":
            funding[i + 1] += event["cash_delta"]
            continue
        q = event["side"] * event["quantity"]
        if kind == "entry":
            opened[event["position_id"]] = event["price"]
            quantity[s][i] += q
            basis[s][i] += q * event["price"]
        else:
            quantity[s][i] -= q
            basis[s][i] -= q * opened[event["position_id"]]
    envelope = curve.cash.to_numpy()[:-1].copy() + funding[1:]
    for s, b in bars.items():
        q = np.cumsum(quantity[s])[:-1]
        cost = np.cumsum(basis[s])[:-1]
        interval = b.reindex(clocks[:-1])
        prices = np.where(q >= 0, interval.low, interval.high)
        envelope += (
            q * prices - cost - np.abs(q) * prices * (fee + row["extra_cost_bps_per_side"] / 10000)
        )
    peaks = np.maximum.accumulate(curve.equity.to_numpy())[:-1]
    return max(row["max_drawdown"], float(np.max(1 - envelope / peaks)))


def review(source, audit_root, output):
    output.mkdir(parents=True, exist_ok=False)
    plan, summary = read(source / "plan.json"), read(source / "evaluation.json")
    audit = read(audit_root / "audit.json")
    if audit["passed"] is not True or audit["evaluation_sha256"] != sha(source / "evaluation.json"):
        raise ValueError("Missing independent primary audit")
    features = {
        s: pd.read_parquet(source / "data" / f"{s}_features.parquet") for s in plan["symbols"]
    }
    bars = {s: pd.read_parquet(source / "data" / f"{s}_5m.parquet") for s in plan["symbols"]}
    # Prefixes end at three preregistered calendar boundaries, never at return extrema.
    prefix_checks = []
    for boundary in (plan["validation_start"], plan["holdout_start"], "2026-07-01"):
        end = pd.Timestamp(boundary, tz="UTC")
        turnovers, reference = {}, {}
        for s in plan["symbols"]:
            b = bars[s].loc[bars[s].index < end]
            d = b.resample("D").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "quote_volume": "sum",
                }
            )
            tr = np.maximum(
                d.high - d.low,
                np.maximum((d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()),
            )
            tr.iloc[0] = d.high.iloc[0] - d.low.iloc[0]
            h = b.resample("h").agg({"close": "last", "quote_volume": "sum"})
            ref = pd.DataFrame(
                {
                    "atr": tr.rolling(20).mean().shift(),
                    "high55": d.high.rolling(55).max().shift(),
                    "low55": d.low.rolling(55).min().shift(),
                    "high20": d.high.rolling(20).max().shift(),
                    "low20": d.low.rolling(20).min().shift(),
                    "momentum": d.close.shift() / d.close.shift(64) - 1,
                    "daily_turnover": d.quote_volume.rolling(28).mean().shift(),
                }
            )
            turnovers[s] = ref.daily_turnover
            r = ref.reindex(h.index.floor("D"))
            r.index = h.index
            r["previous_close"] = h.close.shift()
            r["before_previous_close"] = h.close.shift(2)
            r["prior_hour_quote"] = h.quote_volume.shift()
            reference[s] = r
        ranking = pd.DataFrame(turnovers)[sorted(turnovers)].rank(
            axis=1, ascending=False, method="first"
        )
        for s in plan["symbols"]:
            r = reference[s]
            eligible = (ranking[s] <= plan["liquidity"]["top_n"]) & (
                turnovers[s] >= plan["liquidity"]["minimum_daily_quote_usdt"]
            )
            r["eligible"] = eligible.reindex(r.index.floor("D")).to_numpy()
            actual = features[s].loc[r.index]
            pd.testing.assert_frame_equal(
                r[actual.columns], actual, check_freq=False, rtol=1e-10, atol=1e-9
            )
        prefix_checks.append(
            {
                "end_exclusive": boundary,
                "symbols": len(plan["symbols"]),
                "hourly_rows_per_symbol": len(next(iter(reference.values()))),
                "passed": True,
            }
        )
    attributions = []
    decision_count = 0
    for row in summary["accounts"]:
        folder = source / row["path"]
        if sha(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Account changed")
        for decision in read(folder / "decisions.json"):
            f = features[decision["symbol"]].loc[pd.Timestamp(decision["time"])]
            if not f.eligible or abs(f.previous_close - decision["reference_price"]) > 1e-8:
                raise ValueError("Decision used an ineligible or wrong-time observation")
            maximum = 0.01 * f.prior_hour_quote / 12 / f.previous_close
            if decision["submitted_quantity"] > min(decision["requested_quantity"], maximum) + 1e-9:
                raise ValueError("Submitted quantity used future turnover")
            c = row["mechanism"]
            side = (
                (
                    1
                    if f.previous_close > f.high55 and f.before_previous_close <= f.high55
                    else -1
                    if f.previous_close < f.low55 and f.before_previous_close >= f.low55
                    else 0
                )
                if c == "Donchian55"
                else int(np.sign(f.momentum))
                if c == "Momentum63"
                else 1
            )
            if side != decision["side"]:
                raise ValueError("Decision did not reproduce from completed features")
            decision_count += 1
        if (
            row["phase"] != "continuous"
            or row["initial_equity_usdt"] != plan["selection_capital_usdt"]
        ):
            continue
        by_direction, by_year = defaultdict(list), defaultdict(list)
        for t in read(folder / "trades.json"):
            by_direction["long" if t["side"] == 1 else "short"].append(t["net_pnl"])
            by_year[t["closed_at"][:4]].append(t["net_pnl"])
        open_contribution = row["open_lifecycle_cash_pnl"] + row["unrealized_pnl_less_exit_reserve"]
        attributions.append(
            {
                "mechanism": row["mechanism"],
                "slippage_bps": row["extra_cost_bps_per_side"],
                "total_nav_profit": row["net_profit_usdt"],
                "closed_lifecycle_pnl": row["closed_net_pnl"],
                "adverse_5m_ohlc_envelope_drawdown_vs_sampled_peaks": adverse_envelope(
                    folder, bars, row, plan["fee_per_side"]
                ),
                "open_lifecycle_contribution_including_fees_funding_exit_reserve": open_contribution,
                "positive_nav_profit_fraction_from_open_lifecycles": open_contribution
                / row["net_profit_usdt"]
                if row["net_profit_usdt"] > 0
                else None,
                "closed_by_direction": {
                    k: {"count": len(v), "net_pnl": math.fsum(v)} for k, v in by_direction.items()
                },
                "closed_by_exit_year": {
                    k: {"count": len(v), "net_pnl": math.fsum(v)} for k, v in by_year.items()
                },
                "warning": "Exit-year attribution is not a cold-start yearly account or independent yearly return",
            }
        )
    result = {
        "passed": True,
        "prefix_checks": prefix_checks,
        "decisions_reconstructed": decision_count,
        "attributions": attributions,
        "selection_unchanged": summary["selected_candidate"],
        "evaluation_sha256": sha(source / "evaluation.json"),
        "primary_audit_sha256": sha(audit_root / "audit.json"),
        "reviewer_sha256": sha(Path(__file__)),
        "production_eligible": False,
        "orders_enabled": False,
        "next_priority": "Broaden untouched historical/point-in-time data coverage and verify whether closed-lifecycle edge survives; do not optimize thresholds or add factor families from these outcomes",
    }
    save(output / "review.json", result)
    (output / "reviewer_snapshot.py").write_bytes(Path(__file__).read_bytes())
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir()})
    return {"passed": True, "decision_checks": decision_count, "prefix_checks": len(prefix_checks)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(review(a.source.resolve(), a.audit.resolve(), a.output.resolve()))
