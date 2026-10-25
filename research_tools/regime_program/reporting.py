"""Continuous-account partitions and descriptive lifecycle uncertainty."""

import math

import numpy as np
import pandas as pd


def trade_stats(trades):
    pnl = np.array([t["net_pnl"] for t in trades], dtype=float)
    n = len(pnl)
    win = float((pnl > 0).mean()) if n else None
    interval = None
    if n:
        z = 1.959963984540054
        center = (win + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(win * (1 - win) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        interval = [max(0, center - half), min(1, center + half)]
    loss = -float(pnl[pnl < 0].sum())
    return {
        "complete_lifecycles": n,
        "win_rate": win,
        "win_rate_wilson95": interval,
        "profit_factor": float(pnl[pnl > 0].sum()) / loss if loss > 0 else None,
        "closed_net_pnl": float(pnl.sum()),
        "expectancy": float(pnl.mean()) if n else None,
    }


def cluster_uncertainty(trades):
    if not trades:
        return {"by_symbol": None, "by_entry_month": None}
    frame = pd.DataFrame(trades)
    frame["month"] = pd.to_datetime(frame.entry_time, utc=True).dt.strftime("%Y-%m")
    frame["wins"] = frame.net_pnl.gt(0).astype(int)
    out = {}
    for name, column in (("by_symbol", "symbol"), ("by_entry_month", "month")):
        groups = frame.groupby(column).agg(
            pnl=("net_pnl", "sum"), wins=("wins", "sum"), n=("wins", "size")
        )
        if len(groups) < 4:
            out[name] = {
                "clusters": len(groups),
                "sufficient": False,
                "win_rate95": None,
                "expectancy95": None,
            }
            continue
        rng = np.random.default_rng(17)
        sample = groups.to_numpy()[rng.integers(0, len(groups), size=(1000, len(groups)))].sum(
            axis=1
        )
        out[name] = {
            "clusters": len(groups),
            "sufficient": True,
            "win_rate95": np.quantile(sample[:, 1] / sample[:, 2], [0.025, 0.975]).tolist(),
            "expectancy95": np.quantile(sample[:, 0] / sample[:, 2], [0.025, 0.975]).tolist(),
        }
    return out


def partitions(curve, trades, states, event_flags, initial):
    curve = curve.set_index("date")
    nav = curve.equity
    highwater = nav.cummax()
    rows = []
    starts = list(pd.date_range(nav.index[0], nav.index[-1], freq="90D"))
    ranges = [
        (
            "year",
            str(y),
            max(nav.index[0], pd.Timestamp(f"{y}-01-01", tz="UTC")),
            min(nav.index[-1], pd.Timestamp(f"{y + 1}-01-01", tz="UTC")),
        )
        for y in sorted(set(nav.index.year))
    ]
    ranges += [
        ("90day", f"block{i + 1}", a, min(a + pd.Timedelta(days=90), nav.index[-1]))
        for i, a in enumerate(starts)
    ]
    for kind, label, start, end in ranges:
        x = nav.loc[start:end]
        if len(x) < 2:
            continue
        daily = (
            x.loc[(x.index.hour == 0) & (x.index.minute == 0)].pct_change(fill_method=None).dropna()
        )
        sd = daily.std(ddof=1)
        selected = [t for t in trades if start < pd.Timestamp(t["exit_time"]) <= end]
        rows.append(
            {
                "kind": kind,
                "label": label,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "days": (end - start).total_seconds() / 86400,
                "net_return": float(x.iloc[-1] / x.iloc[0] - 1),
                "local_observed_drawdown": float((1 - x / x.cummax()).max()),
                "full_account_highwater_drawdown": float((1 - x / highwater.loc[x.index]).max()),
                "sharpe": float(daily.mean() / sd * math.sqrt(365))
                if pd.notna(sd) and sd > 1e-15
                else None,
                "lifecycle_by_exit_clock": trade_stats(selected),
            }
        )
    clocks = nav.index[:-1]
    state = states.state.reindex(clocks, method="ffill").to_numpy()
    event = event_flags.reindex(clocks, method="ffill").to_numpy()
    delta = np.diff(nav.to_numpy())
    actual_gross = curve.gross_exposure.iloc[1:].to_numpy() / nav.iloc[1:].to_numpy()
    contribution = []
    for key, labels in (("regime", state), ("covered_fomc", event)):
        for label in sorted(set(labels)):
            mask = labels == label
            selected = []
            for trade in trades:
                decision = pd.Timestamp(trade["entry_time"]) - pd.Timedelta(minutes=5)
                known = (
                    states.state.asof(decision) if key == "regime" else event_flags.asof(decision)
                )
                if known == label:
                    selected.append(trade)
            contribution.append(
                {
                    "group": key,
                    "label": str(label),
                    "intervals": int(mask.sum()),
                    "hours": float(mask.sum() / 12),
                    "net_pnl_contribution": float(delta[mask].sum()),
                    "return_contribution_on_initial": float(delta[mask].sum() / initial),
                    "mean_actual_gross": float(actual_gross[mask].mean()),
                    "lifecycle_by_entry_state": trade_stats(selected),
                }
            )
    return {
        "periods": rows,
        "state_contributions": contribution,
        "note": "One continuous account; period statistics do not reset live highwater. State contributions are sums of actual NAV changes, not stitched regime returns. Lifecycle statistics use full campaign PnL grouped by declared entry/exit clock, not mark-to-market phase PnL.",
    }
