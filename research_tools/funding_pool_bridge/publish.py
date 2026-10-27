"""Finalize the bridge without rerunning it; retain marked open inventory."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.funding_pool_bridge.run import sealed_account
from research_tools.max_effect.run_accounts import verify_manifest


def components(folder, market):
    report = read(folder / "result.json")
    events = read(folder / "events.json")
    fee = report["config"]["fee_bps"] / 10000
    slip = report["config"]["slippage_bps"] / 10000
    end = pd.Timestamp(report["window"]["end"], tz="UTC")
    symbols = sorted({e["symbol"] for e in events})
    rows = []
    for s in symbols:
        own = [e for e in events if e["symbol"] == s]
        cash = math.fsum(e["cash_delta"] for e in own)
        funding = math.fsum(e["cash_delta"] for e in own if e["type"] == "funding")
        fees = math.fsum(e.get("fee", 0) for e in own)
        slippage = math.fsum(e.get("slippage_cost", 0) for e in own)
        floating, reserve, quantity = 0.0, 0.0, 0.0
        if s in report["open_positions"]:
            p = report["open_positions"][s]
            frame = pd.read_parquet(market / (s + "_5m.parquet"))
            mark = float(frame.loc[end - pd.Timedelta(minutes=5), "close"])
            quantity = p["q"] * p["side"]
            reserve = p["q"] * mark * (fee + slip)
            floating = quantity * (mark - p["price"]) - reserve
        net = cash + floating
        rows.append(
            {
                "symbol": s,
                "net_pnl": net,
                "funding": funding,
                "fees_paid": fees,
                "slippage_paid": slippage,
                "cash_delta": cash,
                "marked_open_pnl_after_exit_reserve": floating,
                "open_exit_reserve": reserve,
                "open_quantity_signed": quantity,
                "price_after_slippage_and_open_exit_reserve": net - funding + fees,
            }
        )
    for field, key in [
        ("net_pnl", "net_pnl"),
        ("funding", "funding_net"),
        ("fees_paid", "fees"),
        ("slippage_paid", "slippage_cost"),
        ("marked_open_pnl_after_exit_reserve", "open_pnl_with_exit_reserve"),
    ]:
        if abs(math.fsum(r[field] for r in rows) - report["metrics"][key]) > 1e-6:
            raise ValueError("Cash/mark component differs: " + key)
    return rows


def publish(inputs, run, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    verify_manifest(run)
    plan = read(inputs / "plan.json")
    failure = read(run / "failure.json")
    if failure["financial_accounts_computed"] != 1 or not failure["cash_audit_passed"]:
        raise ValueError("Recovery must reuse one audited account")
    save(
        output / "registration.json",
        {
            "publication_recovery_only": True,
            "financial_parameters_changed": False,
            "new_financial_accounts_computed": 0,
            "failed_completion_sha256": sha(run / "completion.json"),
            "source_sha256": sha(Path(__file__)),
        },
    )
    old_id = "archive_diagnostic2026_funding7_raw_cap10000_slip10"
    pit_id = "daily_archive_diagnostic2026_funding7_raw_cap10000_slip10"
    old_folder, old = sealed_account(plan["inputs"]["old_run"], old_id)
    pit_folder, pit = sealed_account(plan["inputs"]["pit_run"], pit_id)
    new_folder, new = sealed_account(run, "B_execution")
    audit = read(new_folder / "internal_cash_audit.json")
    if not audit["passed"] or audit["result_sha256"] != sha(new_folder / "result.json"):
        raise ValueError("Audited bridge result changed")
    comparison = read(inputs / "source_comparison.json")
    if any(
        any(r[k][v] for v in ["different_rows", "left_only", "right_only"])
        for r in comparison
        for k in ["price_quote", "quality", "funding"]
    ):
        raise ValueError("Execution sources do not match")
    checks = read(inputs / "target_comparison.json")
    if checks["old_vs_matched40"]["different_cells"] or read(inputs / "lifecycle40.json"):
        raise ValueError("Intermediate node equivalence fails")
    curves = {
        name: pd.read_parquet(folder / "equity_5m.parquet")
        for name, folder in [("old", old_folder), ("pit", pit_folder), ("bridge", new_folder)]
    }
    pd.testing.assert_series_equal(curves["old"].date, curves["bridge"].date)
    errors = {
        c: float(np.max(np.abs(curves["old"][c].to_numpy() - curves["bridge"][c].to_numpy())))
        for c in ["equity", "cash", "gross_exposure", "net_exposure"]
    }
    if max(errors.values()) > plan["decision_record"]["equality_tolerance_usdt"]:
        raise ValueError("Execution bridge has a material effect")
    old_parts = components(old_folder, Path(plan["inputs"]["old_market"]))
    pit_parts = components(pit_folder, Path(plan["inputs"]["bundle"]) / "symbols")
    old_by, pit_by = {r["symbol"]: r for r in old_parts}, {r["symbol"]: r for r in pit_parts}
    attribution = [
        {
            "symbol": s,
            "old": old_by.get(s),
            "pit": pit_by.get(s),
            "delta_net_pnl": (pit_by.get(s) or {}).get("net_pnl", 0)
            - (old_by.get(s) or {}).get("net_pnl", 0),
        }
        for s in sorted(set(old_by) | set(pit_by))
    ]
    total = pit["metrics"]["net_pnl"] - old["metrics"]["net_pnl"]
    residual = total - math.fsum(r["delta_net_pnl"] for r in attribution)
    if abs(residual) > 1e-6:
        raise ValueError("Cash attribution residual")
    save(output / "symbol_attribution.json", attribution)
    periods = []
    for label in ["old", "pit"]:
        curve = curves[label].set_index("date").equity
        for start, end in [
            ("2026-01-01", "2026-04-01"),
            ("2026-04-01", "2026-07-01"),
            ("2026-07-01", "2026-09-01"),
        ]:
            a, b = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
            segment = curve.loc[a:b]
            periods.append(
                {
                    "account": label,
                    "start": start,
                    "end": end,
                    "net_pnl": float(segment.iloc[-1] - segment.iloc[0]),
                    "net_return": float(segment.iloc[-1] / segment.iloc[0] - 1),
                    "max_drawdown_within_period": float((1 - segment / segment.cummax()).max()),
                }
            )
    save(output / "periods.json", periods)
    nodes = [
        {
            "id": n["id"],
            "metrics": (pit if n["id"] == "F_pit" else old if n["id"] == "A_legacy" else new)[
                "metrics"
            ],
            "evidence": "sealed historical endpoint"
            if n["id"] in ["A_legacy", "F_pit"]
            else "one new replay plus internal cash audit"
            if n["id"] == "B_execution"
            else "exact input equivalence; no new account",
        }
        for n in plan["ladder"]
    ]
    summary = {
        "schema": "funding-pool-bridge-result.v1",
        "nodes": nodes,
        "actual_new_financial_accounts": 1,
        "reused_endpoints": 2,
        "folded_equivalent_nodes": 3,
        "new_independent_hypotheses": 0,
        "full_5m_execution_bridge_errors": errors,
        "internal_cash_audit": audit,
        "historical_pool_path_delta_usdt": total,
        "historical_pool_path_delta_percentage_points": total / plan["account"]["capital"] * 100,
        "symbol_attribution_residual_usdt": residual,
        "old_open_positions": len(old["open_positions"]),
        "old_open_exit_reserve": math.fsum(r["open_exit_reserve"] for r in old_parts),
        "old_marked_open_pnl_after_exit_reserve": old["metrics"]["open_pnl_with_exit_reserve"],
        "old_complete_lifecycle_pnl": old["metrics"]["closed_net_pnl"],
        "old_open_lifecycle_net_pnl_including_cash_and_mark": old["metrics"]["net_pnl"]
        - old["metrics"]["closed_net_pnl"],
        "full_component_delta": {
            "funding": pit["metrics"]["funding_net"] - old["metrics"]["funding_net"],
            "fees_paid": pit["metrics"]["fees"] - old["metrics"]["fees"],
            "slippage_paid": pit["metrics"]["slippage_cost"] - old["metrics"]["slippage_cost"],
            "price_after_slippage_and_open_exit_reserve": (
                pit["metrics"]["net_pnl"] - pit["metrics"]["funding_net"] + pit["metrics"]["fees"]
            )
            - (old["metrics"]["net_pnl"] - old["metrics"]["funding_net"] + old["metrics"]["fees"]),
        },
        "orders_enabled": False,
        "production_eligible": False,
        "profit_goal_met": False,
        "independent_unseen_evidence": False,
        "third_party_review": False,
        "scope": "Known2026cold243day10kaccountwith15bpsside. Sameold-domain numerics/targets/emptylifecycle; changedopportunitydomain explains conditionalhistoricalpath includingriskstopfeedback, not generalcausaleffect or alpha.",
    }
    save(output / "summary.json", summary)
    bindings = {
        str(p): sha(p)
        for p in [
            inputs / "completion.json",
            run / "completion.json",
            old_folder / "result.json",
            pit_folder / "result.json",
            new_folder / "result.json",
            Path(__file__),
            Path("research_tools/funding_pool_bridge/prepare.py"),
            Path("research_tools/funding_pool_bridge/run.py"),
        ]
    }
    save(output / "source_bindings.json", bindings)
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()
            },
            "profit_goal_met": False,
            "new_financial_accounts": 0,
        },
    )
    print(
        {
            "passed": True,
            "delta_usdt": total,
            "residual": residual,
            "open_positions_included": summary["old_open_positions"],
            "new_accounts": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ["inputs", "run", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    publish(args.inputs, args.run, args.output)
