"""Read-only period attribution for the already frozen 56 equity accounts.

Writes separate diagnostics; never changes the original account or selection.
Group returns compound observations from the continuous account. Trades are
assigned to their exit observation, even when entry belongs to a prior period.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import digest, write_json


def groups(state):
    rows = state["equity_curve"][1:]
    first = pd.Timestamp(rows[0]["day"])
    fills, trades = defaultdict(list), defaultdict(list)
    for fill in state["fills"]:
        fills[fill["clock"]].append(fill)
    for trade in state["trades"]:
        trades[trade["exit_ms"]].append(trade)
    buckets = {}
    for row in rows:
        opening = [
            clock
            for clock in set(fills) | set(trades)
            if row["clock"] - 86400000 < clock <= row["clock"]
        ]
        keys = [
            row["day"][:4],
            f"90d_{(pd.Timestamp(row['day']) - first).days // 90:02d}",
            "regime_" + row["regime"],
        ]
        for key in keys:
            b = buckets.setdefault(
                key,
                {
                    "returns": [],
                    "drawdowns": [],
                    "exposures": [],
                    "trades": [],
                    "fills": [],
                    "start": row["day"],
                    "end": row["day"],
                },
            )
            b["returns"].append(row["daily_return"])
            b["drawdowns"].append(row["drawdown"])
            b["exposures"].append(row["exposure"])
            b["end"] = row["day"]
            for clock in opening:
                b["trades"].extend(trades[clock])
                b["fills"].extend(fills[clock])
    result = {}
    initial = state["binding"]["spec"]["initial_cash"]
    for key, b in buckets.items():
        returns = np.array(b["returns"])
        pnl = np.array([t["net_pnl"] for t in b["trades"]])
        losses = -pnl[pnl < 0].sum()
        std = returns.std(ddof=1) if len(returns) > 1 else 0
        result[key] = {
            "start": b["start"],
            "end": b["end"],
            "sessions": len(returns),
            "net_return": math.prod(1 + returns) - 1,
            "max_continuous_close_drawdown": max(b["drawdowns"]),
            "conditional_sharpe_zero_rf_252": float(returns.mean() / std * math.sqrt(252))
            if std
            else None,
            "complete_cycles_by_exit": len(pnl),
            "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
            "closed_net_pnl_by_exit": float(pnl.sum()),
            "profit_factor": float(pnl[pnl > 0].sum() / losses) if losses else None,
            "fees": sum(f["fee"] for f in b["fills"]),
            "slippage": sum(f["slippage"] for f in b["fills"]),
            "turnover_initial_capital": sum(f["price"] * f["quantity"] for f in b["fills"])
            / initial,
            "mean_exposure": float(np.mean(b["exposures"])),
            "statistical_independence_verified": False,
        }
    return result


def run(output):
    summary = json.loads((output / "summary.json").read_text())
    receipts = []
    for record in summary["records"]:
        folder = output / record["name"]
        state_file = folder / "state.json"
        diagnostics = {
            "schema": "ta-equity-period-diagnostics.v2",
            "state_sha256": digest(state_file),
            "account_identity": json.loads(state_file.read_text())["binding"]["identity"],
            "event_coverage": "unknown; no PIT event feed",
            "period_method": "Continuous daily returns and prior high-water retained. Full cycles grouped by exit day. Conditional regime returns are not separately executable portfolios. Adjacent periods/ETFs not independent.",
            "periods": groups(json.loads(state_file.read_text())),
        }
        write_json(folder / "period-diagnostics.json", diagnostics)
        receipts.append(
            {
                "name": record["name"],
                "state_sha256": diagnostics["state_sha256"],
                "diagnostics_sha256": digest(folder / "period-diagnostics.json"),
            }
        )
    write_json(
        output / "period-diagnostics-receipt.json",
        {"accounts": receipts, "script_sha256": digest(__file__), "new_candidates": 0},
    )
    print("Period attribution generated for", len(receipts), "unchanged accounts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
