"""Independent cash/NAV/funding/capacity and frozen-prediction reconstruction.

This verifier never imports the Freqtrade replay engine or its metrics calculator.
"""

import argparse
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .factor_lab.io import read, save, sha


def audit_account(run, row, bars, rates):  # noqa: C901
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
    positions = {}
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
            effective = clock + pd.Timedelta(minutes=5)
            funding_seen.add((s, clock))
            if abs(p["q"] - e["quantity"]) > 1e-8:
                raise ValueError("Funding inventory differs")
        else:
            b = bars[s].loc[clock - pd.Timedelta(minutes=5)]
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
                if any(k[0] == s for k in positions):
                    raise ValueError("Unexpected pyramiding")
                positions[key] = {"q": q, "price": price, "side": side}
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
        floating += q * close - book - abs(q) * close * (fee + slip)
        gross += abs(q) * close
        active = np.cumsum(alive[s])
        for clock in rates[s].index:
            if times[0] <= clock < times[-1] and (
                (active[times.get_loc(clock)] > 0) != ((s, clock) in funding_seen)
            ):
                raise ValueError("Missing or extra funding")
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


def audit(run, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read(run / "plan.json")
    completion = read(run / "completion.json")
    if (
        sha(run / "plan.json") != completion["plan_sha256"]
        or sha(run / "catalog.json") != completion["catalog_sha256"]
    ):
        raise ValueError("Registered study identity differs")
    source = Path(plan["source_run"]) / "data"
    for name, value in read(source / "receipts.json").items():
        if sha(source / name) != value:
            raise ValueError("Prepared market input changed")
    bars = {s: pd.read_parquet(source / f"{s}_5m.parquet") for s in plan["symbols"]}
    rates = {s: pd.read_parquet(source / f"{s}_funding.parquet") for s in plan["symbols"]}
    rows = read(run / "accounts.json")
    if len(rows) != 64 or len({r["id"] for r in rows}) != 64:
        raise ValueError("Incomplete registered account grid")
    account_rows = [audit_account(run, r, bars, rates) for r in rows]
    features = pd.read_parquet(run / "features.parquet")
    z = pd.read_parquet(run / "standardized.parquet")
    labels = pd.read_parquet(run / "labels.parquet")
    models = read(run / "models.json")
    t = features.index.get_level_values("time")
    tm = (t >= pd.Timestamp(plan["training_start"], tz="UTC")) & (
        t + pd.Timedelta(hours=24) < pd.Timestamp(plan["training_end"], tz="UTC")
    )
    med = features.loc[tm].median()
    scale = features.loc[tm].std(ddof=1).replace(0, 1).fillna(1)
    reconstructed = ((features.fillna(med) - med) / scale).clip(-5, 5)
    if not np.allclose(reconstructed, z, atol=1e-12, rtol=1e-12):
        raise ValueError("Training-only normalization differs")
    model_rows = []
    predictions = pd.read_parquet(run / "predictions.parquet")
    for name, m in models.items():
        x = z.loc[tm, m["columns"]].to_numpy()
        y = labels.loc[tm, str(m["horizon"])].to_numpy()
        valid = np.isfinite(y)
        x = x[valid]
        y = y[valid]
        mean = float(y.mean())
        gram = x.T @ x
        values, vectors = np.linalg.eigh(gram)
        beta = vectors @ ((vectors.T @ (x.T @ (y - mean))) / (values + m["ridge_penalty"]))
        target = np.array([m["coefficients"][k] for k in m["columns"]])
        error = float(np.max(abs(beta - target)))
        pred = z[m["columns"]].to_numpy() @ beta + mean
        if error > 1e-9 or np.max(abs(pred - predictions[name].to_numpy())) > 1e-8:
            raise ValueError("Independent ridge/prediction reconstruction differs")
        model_rows.append({"model": name, "coefficient_max_error": error, "training_rows": len(y)})
    # Recompute every active factor from real prefix-only inputs, including news as-of rules.
    from factors.common.panel import build

    events = read(Path(plan["news_source"]) / "events.json")
    prefix = []
    for boundary in ["2025-01-01", "2026-07-01"]:
        cut = pd.Timestamp(boundary, tz="UTC")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for s in bars:
                bars[s].loc[bars[s].index < cut].to_parquet(root / f"{s}_5m.parquet")
                rates[s].loc[rates[s].index < cut].to_parquet(root / f"{s}_funding.parquet")
            f, _, catalog = build(root, events, plan["symbols"])
            for s in bars:
                expected = features.xs(s, level="symbol").loc[f[s].index]
                pd.testing.assert_frame_equal(
                    expected,
                    f[s].rename_axis(expected.index.name),
                    rtol=1e-9,
                    atol=1e-10,
                    check_freq=False,
                )
            prefix.append(
                {"cutoff": boundary, "factors": len(catalog), "symbols": len(bars), "passed": True}
            )
    window_diagnostics = []
    for w in plan["test_windows"]:
        curve = (
            bars["BTCUSDT"]
            .loc[
                (bars["BTCUSDT"].index >= pd.Timestamp(w["start"], tz="UTC"))
                & (bars["BTCUSDT"].index < pd.Timestamp(w["end"], tz="UTC"))
            ]
            .close
        )
        dd = float((1 - curve / curve.cummax()).max())
        window_diagnostics.append(
            {
                "id": w["id"],
                "btc_max_drawdown": dd,
                "stress_label": "深回撤" if dd >= 0.12 else "常态波动",
                "selection_rule": "fixed first28days of quarter, state classification is descriptive only",
            }
        )
    result = {
        "passed": True,
        "window_diagnostics": window_diagnostics,
        "accounts": account_rows,
        "model_reconstruction": model_rows,
        "real_data_prefix_checks": prefix,
        "canonical_factor_count": len(features.columns),
        "source_catalog_sha256": sha(run / "catalog.json"),
        "auditor_sha256": sha(Path(__file__)),
        "scope": "cash/funding/capacity/NAV, key metrics, training normalization and12ridge fits, all74real-data factor prefixes",
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
    }
    save(output / "audit.json", result)
    shutil.copyfile(__file__, output / "auditor_snapshot.py")
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir()})
    return {"passed": True, "accounts": len(rows), "models": len(models), "prefixes": len(prefix)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(audit(a.run.resolve(), a.output.resolve()))
