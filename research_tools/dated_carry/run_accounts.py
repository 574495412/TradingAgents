"""Execute four frozen dated-carry policy/cost paths and independently audit them."""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research_tools.dated_carry.account_audit import audit_account
from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.regime_program.reporting import cluster_uncertainty
from research_tools.trend_stop_audit.run import paired_uncertainty


def load(inputs):
    p = read(inputs / "plan.json")
    markets = [
        {s: pd.read_parquet(inputs / f"{s}_{kind}.parquet") for s in p["contracts"]}
        for kind in ["spot", "future", "mark", "funding"]
    ]
    return p, markets


def publish(folder, account, signals, config, plan, spot):
    folder.mkdir(parents=True, exist_ok=False)
    c = account["curve"]
    compatible = c.assign(
        cash=c.quote_cash + c.futures_wallet,
        gross_exposure=c.gross_notional,
        net_exposure=c.net_notional,
    )
    dates = pd.date_range(plan["window"]["start"], plan["window"]["end"], freq="D")
    benchmark = spot["BTCUSDT"].close.reindex(dates - pd.Timedelta(minutes=5))
    benchmark.index = dates
    m = portfolio_metrics(
        compatible, account["trades"], account["events"], config["capital"], benchmark
    )
    last = c.iloc[-1]
    m["open_campaign_net_pnl"] = (
        math.fsum(p["cash_flow"] for p in account["positions"].values())
        + last.spot_assets
        + last.perp_unrealized
        - last.exit_reserve
    )
    assert abs(m["closed_net_pnl"] + m["open_campaign_net_pnl"] - m["net_pnl"]) < 1e-6
    m["mean_absolute_net_exposure"] = float((c.net_notional / c.equity).abs().mean())
    m["maximum_conservative_stress_drawdown"] = float(
        (
            1 - c.stress_nav_envelope.iloc[1:].to_numpy() / c.equity.cummax().iloc[:-1].to_numpy()
        ).max()
    )
    m["matched_closed_pairs"] = sum(t["matched_hedge"] for t in account["trades"])
    m["settlement_fees"] = math.fsum(
        e["fee"] for e in account["events"] if e["type"] == "settlement"
    )
    m["settlement_notional"] = math.fsum(
        e["notional"] for e in account["events"] if e["type"] == "settlement"
    )
    m["market_turnover"] = math.fsum(
        e.get("notional", 0) for e in account["events"] if e["type"] in ["entry", "exit"]
    ) / float(c.equity.mean())
    c.to_parquet(folder / "equity_5m.parquet", index=False)
    for name in ["events", "requests", "trades", "positions", "resume_state"]:
        save(folder / (name + ".json"), account[name])
    save(folder / "signals.json", signals)
    save(folder / "contracts.json", plan["contracts"])
    report = {
        "config": config,
        "window": plan["window"],
        "metrics": m,
        "capacity": account["capacity"],
        "flags": account["flags"],
        "wallets": account["wallets"],
        "risk_halted": account["risk_halted"],
        "engine": account["engine"],
        "source_hashes": {p.name: sha(p) for p in folder.iterdir() if p.is_file()},
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
    }
    save(folder / "result.json", report)
    return report


def worker(output, name):
    plan = read(output / "plan.json")
    for f, digest in plan["account_source_hashes"].items():
        assert sha(Path(f)) == digest, f
    _, markets = load(Path(plan["prepared_inputs"]))
    r = read(output / (name + "_registration.json"))
    a = executor(plan["engine"]).replay(
        *markets,
        r["signals"],
        r["config"],
        r["split"],
        plan["window"]["end"],
        initial_state=read(output / (name + "_checkpoint.json")),
        final_end=plan["window"]["end"],
        contracts=plan["contracts"],
    )
    a["curve"].to_parquet(output / (name + "_child_curve.parquet"), index=False)
    save(
        output / (name + "_child.json"),
        {
            "pid": os.getpid(),
            **{k: a[k] for k in ["events", "requests", "trades", "resume_state", "positions"]},
        },
    )


def run(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    plan, markets = load(inputs)
    for f, digest in plan["source_hashes"].items():
        assert sha(Path(f)) == digest, f
    save(output / "protocol.json", read(Path("docs/research/regime_program/targets.json")))
    sources = [
        Path(__file__),
        Path(plan["engine"]),
        Path("research_tools/dated_carry/account_audit.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/regime_program/reporting.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/factor_lab/study.py"),
        Path("research_tools/trend_stop_audit/run.py"),
        inputs / "completion.json",
        output / "protocol.json",
    ]
    plan = {
        **plan,
        "prepared_inputs": str(inputs),
        "engine": str(Path(plan["engine"]).resolve()),
        "account_source_hashes": {str(p): sha(p) for p in sources},
    }
    save(output / "plan.json", plan)
    cases = [
        {"id": f"{profile}_slip{cost}", "profile": profile, "slippage_bps": cost}
        for profile in plan["profiles"]
        for cost in plan["costs_slippage_bps"]
    ]
    assert len(cases) <= plan["budget"]["actual_strategy_accounts_max"]
    save(
        output / "registration.json",
        {
            "plan_sha256": sha(output / "plan.json"),
            "before_new_economic_returns": True,
            "cases": cases,
            "actual_new_strategy_accounts": len(cases),
            "new_economic_variants": 1,
            "new_independent_mechanisms": 0,
            "known_one_quarter_sample_insufficient": True,
        },
    )
    engine = executor(plan["engine"])
    rows, failures, full = [], [], None
    # All cases are frozen together. Raw outputs precede independent accounting;
    # no raw path is admitted or selected while its audit remains pending.
    for case in cases:
        config = {**plan["account"], "slippage_bps": case["slippage_bps"]}
        signals = read(inputs / (case["profile"] + "_signals.json"))
        a = engine.replay(
            *markets,
            signals,
            config,
            plan["window"]["start"],
            plan["window"]["end"],
            contracts=plan["contracts"],
        )
        folder = output / "accounts" / case["id"]
        r = publish(folder, a, signals, config, plan, markets[0])
        row = {
            **case,
            **r,
            "result_sha256": sha(folder / "result.json"),
            "internal_cash_audit_passed": False,
            "cluster_uncertainty": cluster_uncertainty(a["trades"]),
        }
        rows.append(row)
        if case["profile"] == "hold_to_delivery" and case["slippage_bps"] == 10:
            full = a
        print({"id": case["id"], "computed": True, "internal_audit": "pending"}, flush=True)
    for row in rows:
        folder = output / "accounts" / row["id"]
        try:
            proof = audit_account(folder, *markets, plan["contracts"])
            save(folder / "internal_cash_audit.json", proof)
            row["internal_cash_audit_passed"] = True
        except Exception as e:
            failures.append(
                {
                    "id": row["id"],
                    "error": str(e),
                    "type": type(e).__name__,
                    "economic_evidence_unknown_until_repaired": True,
                }
            )
        save(folder / "row.json", row)
        print(
            {
                "id": row["id"],
                "audited": row["internal_cash_audit_passed"],
                "net_pnl": row["metrics"]["net_pnl"],
            },
            flush=True,
        )
    restarts = []
    if not failures and full is not None:
        config = {**plan["account"], "slippage_bps": 10}
        signals = read(inputs / "hold_to_delivery_signals.json")
        entries = [e for e in full["events"] if e["type"] == "entry"]
        splits = [("held_entry", entries[0]["time"])] if entries else []
        if any(e["type"] == "settlement" for e in full["events"]):
            splits.append(
                (
                    "at_delivery",
                    next(e["time"] for e in full["events"] if e["type"] == "settlement"),
                )
            )
        for name, split in splits:
            prefix = engine.replay(
                *markets,
                signals,
                config,
                plan["window"]["start"],
                split,
                final_end=plan["window"]["end"],
                contracts=plan["contracts"],
            )
            assert prefix["positions"]
            save(output / (name + "_checkpoint.json"), prefix["resume_state"])
            save(
                output / (name + "_registration.json"),
                {"split": split, "signals": signals, "config": config, "parent_pid": os.getpid()},
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research_tools.dated_carry.run_accounts",
                    "--worker",
                    str(output),
                    "--name",
                    name,
                ],
                check=True,
            )
            child = read(output / (name + "_child.json"))
            curve = pd.read_parquet(output / (name + "_child_curve.parquet"))
            expected = (
                full["curve"].loc[full["curve"].date >= pd.Timestamp(split)].reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(curve, expected, check_exact=True)
            for key in ["events", "requests", "trades"]:
                assert prefix[key] + child[key] == full[key], key
            assert child["resume_state"] == full["resume_state"] and child["pid"] != os.getpid()
            restarts.append(
                {
                    "name": name,
                    "split": split,
                    "positions": len(prefix["positions"]),
                    "future_base_quantity": math.fsum(
                        p["perp_q"] for p in prefix["positions"].values()
                    ),
                    "spot_base_quantity": math.fsum(
                        p["spot_q"] for p in prefix["positions"].values()
                    ),
                    "parent_pid": os.getpid(),
                    "worker_pid": child["pid"],
                    "all_curve_ledger_requests_inventory_and_state_bit_exact": True,
                }
            )
    cash = Path(
        "results/research/sparse_family_accounts_20261026_v002/accounts/cash_continuous2025_2026_cash_cap10000_slip10"
    )
    cash_result = read(cash / "result.json")
    assert (
        cash_result["metrics"]["net_pnl"] == 0
        and cash_result["config"]["capital"] == plan["account"]["capital"]
    )
    cash_curve = (
        pd.read_parquet(cash / "equity_5m.parquet")
        .set_index("date")
        .loc[pd.Timestamp(plan["window"]["start"]) : pd.Timestamp(plan["window"]["end"])]
    )
    assert (
        cash_curve.index.equals(
            pd.date_range(plan["window"]["start"], plan["window"]["end"], freq="5min")
        )
        and cash_curve.equity.eq(10000).all()
    )
    comparisons = []
    for cost in plan["costs_slippage_bps"]:
        a, b = [
            next(r for r in rows if r["profile"] == profile and r["slippage_bps"] == cost)
            for profile in plan["profiles"]
        ]
        ca, cb = [
            pd.read_parquet(output / "accounts" / r["id"] / "equity_5m.parquet") for r in [a, b]
        ]
        comparisons.append(
            {
                "slippage_bps": cost,
                "early_minus_delivery_net_pnl": b["metrics"]["net_pnl"] - a["metrics"]["net_pnl"],
                "paired_daily_uncertainty": paired_uncertainty(ca, cb),
                "same_realized_risk": False,
            }
        )
    summary = {
        "schema": "dated-carry-pilot-accounts.v1",
        "account_rows": rows,
        "financial_audit_failures": failures,
        "actual_strategy_accounts": len(rows),
        "reused_cash_control": {
            "source": str(cash),
            "result_sha256": sha(cash / "result.json"),
            "calendar_days": 87,
            "net_pnl": 0,
        },
        "actual_process_restarts": restarts,
        "comparisons": comparisons,
        "sample_sufficient": False,
        "module_admitted": False,
        "capacity_accounts": 0,
        "independent_oos": False,
        "third_party_review": False,
        "profit_goal_met": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "summary.json", summary)
    manifest = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}
    save(
        output / "completion.json",
        {"passed": not failures, "manifest": manifest, "profit_goal_met": False},
    )
    print(
        {
            "accounts": len(rows),
            "audit_failures": failures,
            "actual_process_restarts": restarts,
            "module_admitted": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--name")
    a = p.parse_args()
    if a.worker:
        worker(a.worker, a.name)
    else:
        assert a.inputs and a.output
        run(a.inputs, a.output)
