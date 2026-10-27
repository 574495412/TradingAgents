"""Publish the bounded price-stop counterfactual and its non-admission conclusion."""

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.trend_pool_audit.run import load
from research_tools.trend_stop_audit.review import partitions


def held_at(events, clock):
    positions = {}
    for e in events:
        if pd.Timestamp(e["time"]) > clock:
            break
        s = e["symbol"]
        if e["type"] == "funding":
            continue
        if e["type"] == "entry":
            stop = e["price"] * (1 - e["side"] * 0.08)
            if s in positions:
                p = positions[s]
                p["q"] += e["quantity"]
                p["stop"] = max(p["stop"], stop) if p["side"] == 1 else min(p["stop"], stop)
            else:
                positions[s] = {
                    "q": e["quantity"],
                    "side": e["side"],
                    "stop": stop,
                    "opened_at": e["time"],
                }
        else:
            positions[s]["q"] -= e["quantity"]
            if positions[s]["q"] < 1e-12:
                del positions[s]
    return positions


def review(run, finalized, output):
    output.mkdir(parents=True, exist_ok=False)
    for root in [run, finalized]:
        verify_manifest(root)
    proof = read(finalized / "verification.json")
    assert proof["passed"] and proof["preserved_run_completion_sha256"] == sha(
        run / "completion.json"
    )
    summary = read(finalized / "summary.json")
    assert not summary["capacity_triggered"] and not summary["financial_failures"]
    plan = read(run / "plan.json")
    for filename, digest in plan["source_hashes"].items():
        assert sha(Path(filename)) == digest, filename
    metrics, periods, details, nav_errors = [], [], {}, []
    for row in summary["account_rows"]:
        if row["reused_account"]:
            continue
        folder = run / "accounts" / row["id"]
        result = read(folder / "result.json")
        assert sha(folder / "result.json") == row["result_sha256"]
        for name, digest in result["source_hashes"].items():
            assert sha(folder / name) == digest
        audit = read(folder / "internal_cash_audit.json")
        stops = read(folder / "internal_stop_audit.json")
        assert audit["passed"] and stops["passed"] and stops["first_stop_requests"] == 0
        nav_errors.append(audit["max_nav_error"])
        p, exposure = partitions(folder, row["id"])
        periods.extend(p)
        m = result["metrics"]
        assert abs(math.fsum(x["net_pnl"] for x in p if x["kind"] == "year") - m["net_pnl"]) < 1e-6
        assert abs(m["closed_net_pnl"] + m["open_campaign_net_pnl"] - m["net_pnl"]) < 1e-6
        events = read(folder / "events.json")
        assert not any(e.get("reason") == "stop" for e in events)
        state = read(folder / "resume_state.json")
        reserve = math.fsum(
            v["q"]
            * state["previous_prices"][s]
            * (result["config"]["fee_bps"] + result["config"]["slippage_bps"])
            / 10000
            for s, v in state["positions"].items()
        )
        price = m["net_pnl"] - m["funding_net"] + m["fees"] + m["slippage_cost"] + reserve
        trades = read(folder / "trades.json")
        median = float(
            pd.Series(
                [
                    (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds()
                    / 86400
                    for t in trades
                ]
            ).median()
        )
        details[row["id"]] = {
            **exposure,
            "mean_gross": m["mean_gross_exposure"],
            "mean_abs_net": m["mean_absolute_net_exposure"],
            "complete_cycle_median_days": median,
            "complete_cycle_exit_reasons": dict(Counter(t["reason"] for t in trades)),
            "price_component_before_paid_costs_and_exit_reserve": price,
            "funding_net": m["funding_net"],
            "paid_fees": m["fees"],
            "paid_slippage": m["slippage_cost"],
            "unpaid_exit_reserve": reserve,
            "open_campaigns": len(state["positions"]),
            "open_campaign_net_pnl": m["open_campaign_net_pnl"],
            "same_path_arithmetic_not_zero_cost_rerun": True,
        }
        n, w, z = m["closed_trades"], m["win_rate"], 1.959963984540054
        denominator = 1 + z * z / n
        center = (w + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(w * (1 - w) / n + z * z / (4 * n * n)) / denominator
        metrics.append(
            {
                "id": row["id"],
                "policy": "price_stop_disabled",
                "profile": row["profile"],
                "cost_bps": row["slippage_bps"] + 5,
                "risk_halted": row["risk_halted"],
                **m,
                "Wilson95_descriptive_only": [center - radius, center + radius],
            }
        )
    reference = Path(plan["reference_run"])
    for profile in plan["profiles"]:
        for cost in plan["costs_slippage_bps"]:
            folder = reference / "accounts" / f"daily_archive_{profile}_cap10000_slip{cost}"
            result = read(folder / "result.json")
            metrics.append(
                {
                    "id": folder.name,
                    "policy": "price_stop_enabled",
                    "profile": profile,
                    "cost_bps": cost + 5,
                    "risk_halted": read(folder / "resume_state.json")["risk_halted"],
                    **result["metrics"],
                }
            )
    _, bars, _, _ = load(Path(plan["targets"]))
    reference_events = read(
        reference / "accounts/daily_archive_monthly_tsmom_cap10000_slip10/events.json"
    )
    first_stop = next(
        e for e in reference_events if e.get("reason") == "stop" and e["type"] == "exit"
    )
    at = pd.Timestamp(first_stop["submitted_at"])
    off_events = read(run / "accounts/disabled_monthly_tsmom_cap10000_slip10/events.json")
    held = held_at(off_events, at)[first_stop["symbol"]]
    b = bars[first_stop["symbol"]].loc[at - pd.Timedelta(minutes=5)]
    assert b.low <= held["stop"] if held["side"] == 1 else b.high >= held["stop"]
    assert held["q"] > 0 and pd.Timestamp(held["opened_at"]) < at
    after = held_at(off_events, pd.Timestamp(first_stop["time"]))
    assert first_stop["symbol"] in after
    witness = {
        "symbol": first_stop["symbol"],
        "original_stop_submitted_at": at.isoformat(),
        "original_stop_fill_at": first_stop["time"],
        "disabled_held_at_same_clock": held,
        "preceding_actual_low": float(b.low),
        "preceding_actual_high": float(b.high),
        "disabled_still_held_after_original_exit": True,
        "scope": "One concrete recorded-policy witness, not a population stop-effect estimate or a new strategy trial.",
    }
    case = read(run / "selected_case.json")
    assert case["receipt"]["financial_audit_passed"] and not case["residual_positions"]
    cash = Path(
        "results/research/sparse_family_accounts_20261026_v002/accounts/cash_continuous2025_2026_cash_cap10000_slip10"
    )
    assert read(cash / "result.json")["metrics"]["net_pnl"] == 0
    save(output / "metrics.json", metrics)
    save(output / "periods.json", periods)
    save(output / "attribution_and_holding.json", details)
    save(output / "policy_witness.json", witness)
    save(output / "selected_TON_exit_case.json", case)
    save(output / "comparisons.json", summary["comparisons"])
    save(output / "restart_verification.json", read(finalized / "restart_verification.json"))
    save(
        output / "exposure_summation_audit.json", read(finalized / "exposure_summation_audit.json")
    )
    verification = {
        "schema": "monthly-hold-policy-review.v1",
        "passed": True,
        "source_hashes": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path("research_tools/trend_stop_audit/review.py"),
                run / "completion.json",
                finalized / "completion.json",
                Path(plan["contract"]) / "completion.json",
            ]
        },
        "maximum_internal_NAV_error_usdt": max(nav_errors),
        "new_policy_accounts": 4,
        "exact_bridge_replays": 2,
        "selected_case_paths": 1,
        "source_alias_cases": 4,
        "actual_financial_computations": 7,
        "no_financial_replay_to_resolve_checker": True,
        "NAV_cash_and_final_state_bit_exact_after_restore": True,
        "exposure_sums_bit_exact_after_restore": False,
        "exposure_sums_independently_within_binary64_bound": True,
        "capacity_triggered": False,
        "profit_goal_met": False,
        "third_party_review": False,
        "independent_unseen_validation": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "verification.json", verification)
    print({"verification": verification, "details": details, "witness": witness}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--finalized", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    review(a.run, a.finalized, a.output)
