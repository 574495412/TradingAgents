"""Reconcile side, coin and entry-cohort attribution of source-paired funding accounts."""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def diagnose(run, output):
    source, plan, receipt = (
        read(run / "summary.json"),
        read(run / "plan.json"),
        read(run / "completion.json"),
    )
    if sha(run / "summary.json") != receipt["summary_sha256"]:
        raise ValueError("Source result changed")
    output.mkdir(parents=True, exist_ok=False)
    data = Path(plan["data"])
    symbols = read(data / "prepare_plan.json")["symbols"]
    closes = {
        s: pd.read_parquet(data / (s + "_5m.parquet"), columns=["close"]).close for s in symbols
    }
    results = []
    for row in source["account_rows"]:
        if row["reused_reference"]:
            continue
        folder = run / "accounts" / row["id"]
        if sha(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Account identity changed")
        account = read(folder / "result.json")
        for name, digest in account["source_hashes"].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError("Frozen financial ledger changed")
        events = read(folder / "events.json")
        campaigns = defaultdict(lambda: {"funding": 0.0, "fees": 0.0, "slippage": 0.0})
        for e in events:
            p = campaigns[(e["symbol"], e["position_id"])]
            if e["type"] == "funding":
                p["funding"] += e["cash_delta"]
            else:
                p["fees"] += e["fee"]
                p["slippage"] += e["slippage_cost"]
        life = []
        for t in account["trades"]:
            p = campaigns[(t["symbol"], t["position_id"])]
            life.append(
                {
                    "symbol": t["symbol"],
                    "position_id": t["position_id"],
                    "side": t["direction"],
                    "entry_month": t["entry_time"][:7],
                    "closed": True,
                    "net_pnl": t["net_pnl"],
                    "exit_reserve": 0.0,
                    **p,
                }
            )
        cost = (account["config"]["fee_bps"] + account["config"]["slippage_bps"]) / 10000
        for s, p in account["open_positions"].items():
            last = closes[s].loc[pd.Timestamp(row["end"], tz="UTC") - pd.Timedelta(minutes=5)]
            reserve = p["q"] * last * cost
            net = p["net_pnl"] + p["side"] * p["q"] * (last - p["price"]) - reserve
            life.append(
                {
                    "symbol": s,
                    "position_id": p["id"],
                    "side": p["side"],
                    "entry_month": p["opened_at"][:7],
                    "closed": False,
                    "net_pnl": net,
                    "exit_reserve": reserve,
                    **campaigns[(s, p["id"])],
                }
            )
        for p in life:
            p["same_path_pre_cost_price_pnl"] = (
                p["net_pnl"] - p["funding"] + p["fees"] + p["slippage"] + p["exit_reserve"]
            )
        if (
            abs(
                math.fsum(p["net_pnl"] for p in life)
                - row["capital"] * row["metrics"]["net_return"]
            )
            > 1e-6
        ):
            raise ValueError("Campaign PnL does not sum to actual NAV")
        groups = {}
        for kind in ["side", "symbol", "entry_month"]:
            grouped = []
            for label in sorted({p[kind] for p in life}):
                selected = [p for p in life if p[kind] == label]
                closed = [p for p in selected if p["closed"]]
                grouped.append(
                    {
                        "label": label,
                        "complete_cycles": len(closed),
                        "open_cycles": len(selected) - len(closed),
                        "closed_net_pnl": math.fsum(p["net_pnl"] for p in closed),
                        "open_campaign_net_pnl": math.fsum(
                            p["net_pnl"] for p in selected if not p["closed"]
                        ),
                        **{
                            k: math.fsum(p[k] for p in selected)
                            for k in [
                                "net_pnl",
                                "funding",
                                "fees",
                                "slippage",
                                "exit_reserve",
                                "same_path_pre_cost_price_pnl",
                            ]
                        },
                    }
                )
            if abs(math.fsum(p["net_pnl"] for p in grouped) - row["metrics"]["net_pnl"]) > 1e-6:
                raise ValueError("Attribution group omits campaign PnL")
            groups[kind] = grouped
        halts = [e for e in events if e["type"] == "exit" and e["reason"] == "risk_halt"]
        results.append(
            {
                "id": row["id"],
                "candidate": row["candidate"],
                "variant": row["variant"],
                "phase": row["phase"],
                "slippage_bps": row["slippage_bps"],
                "first_risk_exit": halts[0]["time"] if halts else None,
                "groups": groups,
                "campaigns": life,
                "result_sha256": row["result_sha256"],
            }
        )
    save(
        output / "summary.json",
        {
            "schema": "pure-funding-reconciled-attribution.v1",
            "passed": True,
            "accounts": len(results),
            "results": results,
            "source_completion_sha256": sha(run / "completion.json"),
            "source_code_sha256": sha(Path(__file__)),
            "new_financial_accounts": 0,
            "note": "Same executed path attribution including open marked campaigns and exit reserve. No zero-cost replay, long-only rerun, or state-selected strategy is inferred from component PnL.",
        },
    )
    save(
        output / "completion.json", {"summary_sha256": sha(output / "summary.json"), "passed": True}
    )
    print(
        {
            "passed": True,
            "audited_accounts": len(results),
            "groups_per_account": ["side", "symbol", "entry_month"],
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    diagnose(a.run, a.output)
