"""Independently audit the Freqtrade-owned trend lane without importing its code.

Rebuild derivative cash, inventory, fees, funding, partial fills, complete-position
PnL and every five-minute NAV from sealed sources. This certifies the stated
research model only, not actual exchange liquidity or independent alpha.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def checked(root, relative, checksum):
    path = root / relative
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or sha(path) != checksum
    ):
        raise ValueError(f"Receipt mismatch: {path}")
    return path


def audit_account(source, row, plan):  # noqa: C901
    folder = source / row["path"]
    checked(source, row["path"] + "/result.json", row["result_sha256"])
    result = read(folder / "result.json")
    for name, checksum in result["source_hashes"].items():
        checked(folder, name, checksum)
    events, trades, state = (
        read(folder / (name + ".json")) for name in ("events", "trades", "state")
    )
    curve = pd.read_parquet(folder / "equity_5m.parquet")
    times = pd.DatetimeIndex(curve.date)
    expected = pd.date_range(result["start"], result["end"], freq="5min", tz="UTC")
    if not times.equals(expected):
        raise ValueError("Incomplete NAV calendar")
    initial = result["initial_equity_usdt"]
    fee, slip = plan["fee_per_side"], result["extra_cost_bps_per_side"] / 10000
    cash_delta, floating = np.zeros(len(times)), np.zeros(len(times))
    positions, lifecycle = {}, {}
    quantities = {s: np.zeros(len(times)) for s in plan["symbols"]}
    costs = {s: np.zeros(len(times)) for s in plan["symbols"]}
    market = {s: pd.read_parquet(source / "data" / f"{s}_5m.parquet") for s in plan["symbols"]}
    funding = {
        s: pd.read_parquet(source / "data" / f"{s}_funding.parquet") for s in plan["symbols"]
    }
    funding_seen, max_cash_error, max_participation, previous = set(), 0.0, 0.0, None
    for event in events:
        clock, s, kind = pd.Timestamp(event["time"]), event["symbol"], event["type"]
        if previous is not None and clock < previous:
            raise ValueError("Ledger time reversed")
        previous = clock
        key = (s, event["position_id"])
        if kind == "funding":
            p = positions[key]
            raw = funding[s].loc[clock]
            delta = -p["side"] * p["quantity"] * float(raw["mark"]) * float(raw["funding_rate"])
            if abs(event["quantity"] - p["quantity"]) > 1e-9:
                raise ValueError("Funding inventory differs")
            effective_time = clock + pd.Timedelta(minutes=5)
            funding_seen.add((s, clock))
        elif kind in ("entry", "exit"):
            bar = market[s].loc[clock - pd.Timedelta(minutes=5)]
            side, q = event["side"], event["quantity"]
            price = float(bar["close"]) * (1 + side * slip * (1 if kind == "entry" else -1))
            if q <= 0 or abs(price - event["price"]) > 1e-8:
                raise ValueError("Invalid fill quantity or price")
            if pd.Timestamp(event["submitted_at"]) >= clock:
                raise ValueError("Fill precedes submitted order")
            participation = (
                q * price / float(bar["quote_volume"]) if bar["quote_volume"] > 0 else math.inf
            )
            max_participation = max(max_participation, participation)
            if participation > plan["execution"]["realized_interval_participation"] + 1e-12:
                raise ValueError("Realized participation cap exceeded")
            index = times.get_loc(clock)
            if kind == "entry":
                if key in positions or any(k[0] == s for k in positions):
                    raise ValueError("Unregistered pyramiding or overlapping inventory")
                positions[key] = {"quantity": q, "price": price, "side": side}
                delta = -q * price * fee
                quantities[s][index] += side * q
                costs[s][index] += side * q * price
            else:
                p = positions[key]
                if side != p["side"] or q > p["quantity"] + 1e-9:
                    raise ValueError("Exit inventory invalid")
                delta = side * q * (price - p["price"]) - q * price * fee
                quantities[s][index] -= side * q
                costs[s][index] -= side * q * p["price"]
                p["quantity"] -= q
                if p["quantity"] < 1e-12:
                    del positions[key]
            if abs(q * price * fee - event["fee"]) > 1e-8:
                raise ValueError("Fee mismatch")
            effective_time = clock
        else:
            raise ValueError("Unknown ledger event")
        error = abs(delta - event["cash_delta"])
        max_cash_error = max(max_cash_error, error)
        if error > 1e-7:
            raise ValueError("Cash flow did not independently reproduce")
        cash_delta[times.get_loc(effective_time)] += delta
        lifecycle[key] = lifecycle.get(key, 0.0) + delta
    for s in plan["symbols"]:
        q = np.cumsum(quantities[s])
        book = np.cumsum(costs[s])
        prices = market[s]["close"].reindex(times - pd.Timedelta(minutes=5)).to_numpy()
        if not np.isfinite(prices).all():
            raise ValueError("Missing valuation prices")
        floating += q * prices - book - np.abs(q) * prices * (fee + slip)
        for clock in funding[s].index:
            if times[0] <= clock < times[-1]:
                held = abs(q[times.get_loc(clock)]) > 1e-10
                if held != ((s, clock) in funding_seen):
                    raise ValueError("Missing or spurious funding event")
    cash = initial + np.cumsum(cash_delta)
    nav = cash + floating
    nav_error = float(np.max(np.abs(nav - curve.equity.to_numpy())))
    if nav_error > 1e-6 or np.max(np.abs(cash - curve.cash.to_numpy())) > 1e-6:
        raise ValueError(f"Five-minute NAV differs: {nav_error}")
    if abs(cash[-1] - state["cash"]) > 1e-6:
        raise ValueError("Final cash differs")
    for s, p in state["positions"].items():
        actual = positions.pop((s, p["id"]))
        if abs(actual["quantity"] - p["quantity"]) > 1e-9 or actual["price"] != p["entry_price"]:
            raise ValueError("Final open inventory differs")
    if positions:
        raise ValueError("Unreported open inventory")
    closed = []
    for t in trades:
        # Position IDs are global within the account, not per symbol.
        matches = [value for (_, pid), value in lifecycle.items() if pid == t["id"]]
        if len(matches) != 1 or abs(matches[0] - t["net_pnl"]) > 1e-7:
            raise ValueError("Complete-position PnL differs")
        closed.append(matches[0])
    positive = math.fsum(max(x, 0) for x in closed)
    negative = -math.fsum(min(x, 0) for x in closed)
    pf = positive / negative if negative else None
    win = sum(x > 0 for x in closed) / len(closed) if closed else None
    dd = 1 - nav / np.maximum.accumulate(nav)
    daily = nav[(times.hour == 0) & (times.minute == 0)]
    returns = [b / a - 1 for a, b in zip(daily[:-1], daily[1:], strict=True)]
    sd = (
        math.sqrt(
            math.fsum((x - math.fsum(returns) / len(returns)) ** 2 for x in returns)
            / (len(returns) - 1)
        )
        if len(returns) > 1
        else 0
    )
    sharpe = math.fsum(returns) / len(returns) / sd * math.sqrt(365) if sd > 1e-15 else None
    values = {
        "net_return": float(nav[-1] / initial - 1),
        "max_drawdown": float(max(dd)),
        "profit_factor": pf,
        "win_rate": win,
        "daily_sharpe": sharpe,
        "closed_trades": len(closed),
    }
    for name, value in values.items():
        expected = result[name]
        if (
            (value is None) != (expected is None)
            or value is not None
            and abs(value - expected) > 1e-8
        ):
            raise ValueError(f"Metric differs: {name}")
    rules = plan["screening_targets"]
    passed = (
        values["net_return"] > 0
        and len(closed) >= rules["minimum_complete_positions"]
        and pf is not None
        and pf >= rules["minimum_profit_factor"]
        and sharpe is not None
        and sharpe >= rules["minimum_daily_sharpe"]
        and values["max_drawdown"] <= rules["maximum_drawdown"]
        and not state["halted"]
    )
    if passed != result["economic_audit"]["passed"]:
        raise ValueError("Economic decision differs")
    return {
        "path": row["path"],
        "passed": True,
        "economic_screen_passed": passed,
        "five_minute_nav_rows": len(nav),
        "maximum_nav_error_usdt": nav_error,
        "maximum_cash_flow_error_usdt": max_cash_error,
        "maximum_fill_participation": max_participation,
        "funding_events_verified": len(funding_seen),
        "metrics": values,
        "source_result_sha256": row["result_sha256"],
    }


def dashboard_account(source, row):
    folder = source / row["path"]
    curve = pd.read_parquet(folder / "equity_5m.parquet")
    drawdown = 1 - curve.equity / curve.equity.cummax()
    symbols = {e["position_id"]: e["symbol"] for e in read(folder / "events.json")}
    m = {
        "net_return": row["net_return"],
        "win_rate": row["win_rate"],
        "max_drawdown": row["max_drawdown"],
        "closed_trades": row["closed_trades"],
        "net_pnl": row["net_profit_usdt"],
        "ending_equity": row["ending_equity"],
        "fees": row["fees_usdt"],
        "profit_factor": row["profit_factor"],
    }
    return {
        "metrics": m,
        "economic_account_evaluated": True,
        "funding_included": True,
        "production_eligible": False,
        "independent_oos": False,
        "curve": [
            {"time": t.isoformat(), "equity": float(v), "drawdown": float(d)}
            for t, v, d in zip(curve.date, curve.equity, drawdown, strict=True)
        ],
        "trades": [
            {
                "symbol": symbols[t["id"]],
                "direction": t["side"],
                "entry_time": t["opened_at"],
                "exit_time": t["closed_at"],
                "entry_price": t["entry_price"],
                "net_pnl": t["net_pnl"],
                "exit_reason": t["exit_reason"],
            }
            for t in read(folder / "trades.json")
        ],
        "open_positions": read(folder / "state.json")["positions"],
        "source_result_sha256": row["result_sha256"],
    }


def audit(source, output):
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan, summary = read(source / "plan.json"), read(source / "evaluation.json")
    if (
        sha(source / "plan.json") != summary["plan_sha256"]
        or sha(source / "selection.json") != summary["selection_sha256"]
    ):
        raise ValueError("Evaluation identity differs")
    quality = read(source / "data" / "quality.json")
    for name, checksum in quality["source_files"].items():
        if sha(Path(name)) != checksum:
            raise ValueError("Original market input changed")
    for name, checksum in read(source / "data" / "receipts.json").items():
        checked(source / "data", name, checksum)
    expected = {
        (phase, c, capital, cost)
        for phase in ("train", "validation", "holdout", "continuous")
        for c in plan["candidates"] + ["LongOnlyControl"]
        for capital in (
            plan["capacity_capitals_usdt"]
            if phase == "continuous"
            else [plan["selection_capital_usdt"]]
        )
        for cost in plan["extra_cost_bps_per_side"]
    }
    actual = [
        (r["phase"], r["mechanism"], r["initial_equity_usdt"], r["extra_cost_bps_per_side"])
        for r in summary["accounts"]
    ]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Missing or duplicate registered scenarios")
    rows = [audit_account(source, row, plan) for row in summary["accounts"]]
    selection = read(source / "selection.json")
    train = [
        r
        for r in summary["accounts"]
        if r["phase"] == "train" and r["mechanism"] in plan["candidates"]
    ]
    eligible = [
        c
        for c in plan["candidates"]
        if all(r["economic_audit"]["passed"] for r in train if r["mechanism"] == c)
    ]
    chosen = (
        max(
            eligible,
            key=lambda c: next(
                r["daily_sharpe"]
                for r in train
                if r["mechanism"] == c and r["extra_cost_bps_per_side"] == 3
            ),
        )
        if eligible
        else "cash"
    )
    if (
        chosen != selection["selected_candidate"]
        or selection["future_results_included"] is not False
    ):
        raise ValueError("Training-only selection differs")
    result = {
        "schema": "ta-crypto-trend-audit.v1",
        "passed": True,
        "account_count": len(rows),
        "selected_candidate": chosen,
        "accounts": rows,
        "source": str(source),
        "evaluation_sha256": sha(source / "evaluation.json"),
        "auditor_sha256": sha(Path(__file__)),
        "independent_oos": False,
        "valid_return_source_established": False,
        "production_eligible": False,
        "orders_enabled": False,
        "scope": "Independent ledger, funding-source, simulated capacity and NAV/metric reconstruction; no order-book or historical contract-filter certification",
    }
    save(output / "audit.json", result)
    catalog = {}
    # Put the declared base-capital continuous accounts first, without profit ranking.
    ordered = sorted(
        summary["accounts"],
        key=lambda r: (
            r["phase"] != "continuous",
            r["initial_equity_usdt"] != plan["selection_capital_usdt"],
            r["mechanism"],
            r["extra_cost_bps_per_side"],
            r["path"],
        ),
    )
    for row in ordered:
        key = row["path"].replace("/", "_")
        payload = dashboard_account(source, row)
        raw = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
        if len(raw) > 32 * 1024 * 1024:
            raise ValueError("Dashboard archive exceeds its safe reader limit")
        with (output / (key + ".json.gz")).open("xb") as stream:
            stream.write(gzip.compress(raw, mtime=0))
        catalog[key] = {
            "metrics": payload["metrics"],
            "daily_sharpe": row["daily_sharpe"],
            "phase": row["phase"],
            "capital": row["initial_equity_usdt"],
            "slippage_bps_per_side": row["extra_cost_bps_per_side"],
            "open_positions": row["open_positions"],
            "source_result_sha256": row["result_sha256"],
        }
    # The existing CZ catalog accepts metrics/<case> bound gzip account companions.
    save(
        output / "summary.json",
        {
            "schema": "ta-trend-review.v1",
            "market": "crypto",
            "currency": "USDT",
            "status": "主线首轮完成，训练选择" + chosen,
            "metrics": catalog,
            "selected_candidate": chosen,
            "production_eligible": False,
            "independent_oos": False,
            "economic_account_evaluated": True,
            "funding_included": True,
            "limitations": plan["scope_limitations"],
            "audit_passed": True,
            "source_evaluation_sha256": sha(source / "evaluation.json"),
        },
    )
    save(output / "protocol.json", plan)
    (output / "auditor_snapshot.py").write_bytes(Path(__file__).read_bytes())
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir()})
    return {"passed": True, "accounts": len(rows), "selection": chosen}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(audit(args.source, args.output))


if __name__ == "__main__":
    main()
