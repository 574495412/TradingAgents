"""Audit fixed NFI cumulative controls without assuming risk rules are correct."""

import argparse
import hashlib
import json
import math
import zipfile
from decimal import Decimal
from pathlib import Path

import pandas as pd

from research_tools.nfi_reproduction.audit_original import cash_from_orders

CASES = [
    ("A", "A_original_observed_v003", "原版：单边5bps，3x"),
    ("B", "B_cost8_v001", "单边8bps成本代理，3x"),
    ("C", "C_cost15_v001", "单边15bps成本代理，3x"),
    ("D", "D_leverage1_v001", "C + 杠杆降为1x"),
    ("E", "E_grosscap_v001", "D + 新增仓位总敞口上限0.8x"),
    ("F", "F_stop8_v001", "E + 原生8%止损"),
    ("G", "G_halt15_v001", "F + 15%回撤永久停机"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(root, output):
    output.mkdir(parents=True, exist_ok=False)
    results = []
    audit_details = {}
    for case, directory, label in CASES:
        directory = root / directory
        completion = json.loads((directory / "completion.json").read_text())
        if not completion["native_run_completed"]:
            raise ValueError("Incomplete run")
        archive_path = next((directory / "native_results").glob("*.zip"))
        with zipfile.ZipFile(archive_path) as archive:
            name = next(
                n
                for n in archive.namelist()
                if n.endswith(".json") and not n.endswith("_config.json")
            )
            stats = json.loads(archive.read(name))["strategy"]["NostalgiaForInfinityX8"]
        raw_equity = pd.read_parquet(directory / "reference_equity.parquet").set_index("date")
        start = pd.Timestamp("2025-01-01", tz="UTC")
        end = pd.Timestamp("2025-02-01", tz="UTC")
        expected = pd.date_range(start + pd.Timedelta(minutes=5), end, freq="5min")
        if not raw_equity.index[:-1].equals(expected) or raw_equity.index[-1] != end:
            raise ValueError("Unexpected native pre/post-settlement clock")
        if raw_equity.iloc[-1].gross != 0 or raw_equity.iloc[-1].open_trades != 0:
            raise ValueError("Final settlement is not flat")
        if raw_equity.reference_insolvent.any():
            raise ValueError("Insolvent reference account")
        # Native loop starts00:05. Preserve raw rows; derived daily boundaries use
        # explicit initial cash and final settled cash rather than duplicate labels.
        equity = raw_equity.iloc[:-1].copy()
        equity.loc[end] = raw_equity.iloc[-1]
        initial = equity.iloc[0].copy()
        if initial.equity != 10000 or initial.gross != 0:
            raise ValueError("Initial flat state not confirmed")
        initial["peak"] = 10000.0
        initial["drawdown"] = 0.0
        equity.loc[start] = initial
        equity = equity.sort_index()
        daily = equity.equity.resample("1D").first().pct_change().dropna()
        sharpe = (
            float(daily.mean() / daily.std(ddof=1) * math.sqrt(365)) if daily.std(ddof=1) else None
        )
        details = [cash_from_orders(t) for t in stats["trades"]]
        cash_error = max(Decimal(r["absolute_error"]) for r in details)
        qty_error = max(abs(Decimal(r["quantity_residual"])) for r in details)
        total = sum((Decimal(r["reconstructed_net_pnl"]) for r in details), Decimal(0))
        total_error = abs(total - Decimal(str(stats["profit_total_abs"])))
        if (
            cash_error > Decimal(".000001")
            or qty_error > Decimal(".000000001")
            or total_error > Decimal(".00001")
        ):
            raise ValueError("Independent order cash audit failed: " + case)
        if any(t["is_open"] for t in stats["trades"]):
            raise ValueError("Unsettled native inventory")
        marks = equity.hourly_mark_equity.dropna()
        boundaries = ["2025-01-01", "2025-01-08", "2025-01-15", "2025-01-22", "2025-02-01"]
        slices = []
        for start, end in zip(boundaries, boundaries[1:], strict=False):
            lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
            segment = equity.equity.loc[lo:hi]
            slices.append(
                {
                    "start": start,
                    "end_exclusive": end,
                    "reference_return": float(segment.iloc[-1] / segment.iloc[0] - 1),
                    "reference_dd": float((1 - segment / segment.cummax()).max()),
                }
            )
        risk_path = directory / "risk_policy_receipt.json"
        policy = json.loads(risk_path.read_text()) if risk_path.exists() else None
        plan = json.loads((directory / "registered_plan.json").read_text())
        levers = sorted({t["leverage"] for t in stats["trades"]})
        fees = sorted(
            {t["fee_open"] for t in stats["trades"]} | {t["fee_close"] for t in stats["trades"]}
        )
        if fees != [plan["fee_per_side"]]:
            raise ValueError("Effective fee mismatch")
        expected_leverage = 3 if case in "ABC" else 1
        if levers != [expected_leverage]:
            raise ValueError("Effective leverage mismatch")
        expected_stop = -0.08 if case in "FG" else -0.99
        if stats["stoploss"] != expected_stop or any(
            t["initial_stop_loss_ratio"] != expected_stop for t in stats["trades"]
        ):
            raise ValueError("Effective native stop mismatch")
        log_path = root / (directory.name + ".log")
        if log_path.exists() and any(
            marker in log_path.read_text()
            for marker in [
                "Traceback (most recent call last)",
                "Strategy caused the following exception",
            ]
        ):
            raise ValueError("Native callback exception needs review: " + case)
        side_pnl = {
            side: math.fsum(t["profit_abs"] for t in stats["trades"] if t["is_short"] == short)
            for side, short in [("long", False), ("short", True)]
        }
        row = {
            "case": case,
            "label": label,
            "directory": str(directory),
            "trades": stats["total_trades"],
            "filled_orders": sum(len(t["orders"]) for t in stats["trades"]),
            "net_pnl_USDT": stats["profit_total_abs"],
            "net_return": stats["profit_total_abs"] / 10000,
            "win_rate": stats["wins"] / stats["total_trades"],
            "profit_factor": stats["profit_factor"],
            "closed_trade_dd": stats["max_drawdown_account"],
            "full_reference_dd": float(raw_equity.drawdown.max()),
            "observed_hourly_mark_dd": float((1 - marks / marks.cummax()).max()),
            "daily_reference_sharpe_365_descriptive_31days": sharpe,
            "max_gross_to_reference_NAV": float(raw_equity.gross_ratio.max()),
            "commission_plus_cost_proxy_USDT": float(
                sum(Decimal(r["commission"]) for r in details)
            ),
            "model_funding_USDT": float(sum(Decimal(r["funding_model_cash"]) for r in details)),
            "long_short_net_pnl_USDT": side_pnl,
            "native_stoploss": stats["stoploss"],
            "clock_contract": "Raw native00:05..Feb1 inclusive plus same-time post-force-close; derived daily/slice boundaries seed initial10000 cash and retain settled end cash. All raw rows retained for max5m drawdown.",
            "force_exits": sum(t["exit_reason"] == "force_exit" for t in stats["trades"]),
            "native_stop_exits": sum(
                t["exit_reason"] in ["stop_loss", "trailing_stop_loss"] for t in stats["trades"]
            ),
            "risk_policy": policy,
            "time_slices": slices,
            "audit": {
                "max_trade_cash_error_USDT": str(cash_error),
                "total_cash_error_USDT": str(total_error),
                "max_quantity_residual": str(qty_error),
                "passed": True,
            },
            "source_sha256": {
                str(p): sha(p)
                for p in [
                    archive_path,
                    directory / "registered_plan.json",
                    directory / "reference_equity.parquet",
                    directory / "observer_receipt.json",
                ]
            },
        }
        if results:
            prior = results[-1]
            row["incremental_net_return_pp"] = (row["net_return"] - prior["net_return"]) * 100
            row["incremental_reference_dd_pp"] = (
                row["full_reference_dd"] - prior["full_reference_dd"]
            ) * 100
        results.append(row)
        audit_details[case] = details
    value = {
        "schema": "nfi-fixed-cumulative-cost-risk-comparison.v1",
        "period": "20250101-20250201",
        "capital_USDT": 10000,
        "cases": results,
        "author_trade_order_match_audited_separately": True,
        "limitations": [
            "Known historical author month, not unseen or live evidence.",
            "Cost increments are fee-equivalent proxies, not realized bid/ask slippage.",
            "5m OPEN reference NAV includes all held positions, accrued modeled funding and exit fee reserve; hourly marks are separately observed.",
            "Gross cap constrains new entries/additions; market drift can exceed it.",
            "Cumulative effects depend on the registered order and native path feedback.",
            "No demonstrated historical point-in-time universe, executable capacity, actual customer funding, or independent external review.",
            "A nonbinding risk rule has no demonstrated economic benefit.",
            "Short-window Sharpe is descriptive; no annual return extrapolation.",
        ],
        "orders_enabled": False,
        "production_eligible": False,
        "unseen_validation_performed": False,
        "third_party_review_performed": False,
    }
    (output / "comparison.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    (output / "cash_audit.json").write_text(json.dumps(audit_details, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in [
                        "case",
                        "trades",
                        "net_return",
                        "full_reference_dd",
                        "daily_reference_sharpe_365_descriptive_31days",
                        "max_gross_to_reference_NAV",
                        "risk_policy",
                    ]
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.root.resolve(), args.output.resolve())
