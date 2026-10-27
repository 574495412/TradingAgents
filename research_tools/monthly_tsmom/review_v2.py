"""Close the fixed monthly candidate with raw-method, cash and horizon evidence."""

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.trend_stop_audit.review import partitions


def review(run, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(run)
    plan, summary = read(run / "plan.json"), read(run / "summary.json")
    assert not summary["failures"] and summary["actual_base_accounts"] == 4
    for path, digest in plan["account_source_hashes"].items():
        assert sha(Path(path)) == digest, path
    inputs = Path(plan["inputs"])
    targets = Path(plan["prepared_inputs"])
    checked = read(Path(plan["input_audit"]))
    assert checked["passed"] and checked["inputs_completion_sha256"] == sha(
        inputs / "completion.json"
    )
    rows = [r for r in summary["account_rows"] if not r["reused_account"]]
    periods, details, errors, metrics = [], {}, [], []
    for row in rows:
        folder = run / "accounts" / row["id"]
        report = read(folder / "result.json")
        assert sha(folder / "result.json") == row["result_sha256"]
        for name, digest in report["source_hashes"].items():
            assert sha(folder / name) == digest
        audit = read(folder / "internal_cash_audit.json")
        stops = read(folder / "internal_stop_audit.json")
        assert audit["passed"] and stops["passed"] and not stops["retroactive_new_stop_requests"]
        errors.append(audit["max_nav_error"])
        parts, exposure = partitions(folder, row["id"])
        periods.extend(parts)
        m = report["metrics"]
        assert (
            abs(math.fsum(p["net_pnl"] for p in parts if p["kind"] == "year") - m["net_pnl"]) < 1e-6
        )
        assert abs(m["closed_net_pnl"] + m["open_campaign_net_pnl"] - m["net_pnl"]) < 1e-6
        curve = pd.read_parquet(folder / "equity_5m.parquet").set_index("date")
        daily = (
            curve.equity.loc[(curve.index.hour == 0) & (curve.index.minute == 0)]
            .pct_change()
            .dropna()
        )
        assert abs(daily.mean() / daily.std(ddof=1) * math.sqrt(365) - m["sharpe"]) < 1e-10
        trades = read(folder / "trades.json")
        events = read(folder / "events.json")
        side = {
            str(d): {
                "closed_cycles": sum(t["direction"] == d for t in trades),
                "closed_net_pnl": math.fsum(t["net_pnl"] for t in trades if t["direction"] == d),
            }
            for d in [-1, 1]
        }
        details[row["id"]] = {
            **exposure,
            "closed_exit_reasons": dict(Counter(t["reason"] for t in trades)),
            "exit_fill_reasons": dict(Counter(e["reason"] for e in events if e["type"] == "exit")),
            "closed_by_direction": side,
            "open_campaigns": len(read(folder / "resume_state.json")["positions"]),
            "gross_price_component_same_path": m["net_pnl"]
            - m["funding_net"]
            + m["fees"]
            + m["slippage_cost"],
            "mean_gross": m["mean_gross_exposure"],
            "mean_abs_net": m["mean_absolute_net_exposure"],
            "open_campaign_net_pnl": m["open_campaign_net_pnl"],
            "cash_identity": "closed netPnL + open campaign cash/PnL including exit reserve = total netPnL",
        }
        metrics.append(
            {"id": row["id"], "profile": row["profile"], "cost_bps": 5 + row["slippage_bps"], **m}
        )
    cash_path = Path(plan["cash_reference"])
    cash = read(cash_path / "result.json")
    assert cash["window"] == plan["window"] and cash["metrics"]["net_pnl"] == 0
    for name, digest in cash["source_hashes"].items():
        assert sha(cash_path / name) == digest
    horizon, unknown = [], []
    for anchor in read(targets / "anchors.json"):
        if anchor["variant"] != "daily_archive":
            continue
        start = pd.Timestamp(anchor["decision"])
        end = start + pd.DateOffset(months=1)
        trend, long = 0.0, 0.0
        legs = []
        for s, weight in anchor["unit_intents"].items():
            frame = pd.read_parquet(inputs / "daily_indexes" / f"daily_archive_{s}.parquet")
            r = frame.economic_return.reindex(pd.date_range(start, end, freq="D", inclusive="left"))
            if r.isna().any() or not r.gt(-1).all():
                unknown.append({"decision": start.isoformat(), "symbol": s})
                continue
            forward = float((1 + r).prod() - 1)
            trend += 0.8 * weight * forward
            long += 0.8 * abs(weight) * forward
            legs.append(
                {"symbol": s, "unit_intent": weight, "forward_source_index_return": forward}
            )
        month_unknown = [r for r in unknown if r["decision"] == start.isoformat()]
        horizon.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "theoretical_trend_factor_payoff": trend if not month_unknown else None,
                "theoretical_long_factor_payoff": long if not month_unknown else None,
                "difference": trend - long if not month_unknown else None,
                "unknown_legs": month_unknown,
                "legs": legs,
            }
        )
    assert len(horizon) == 20
    horizon_summary = {
        "posthoc": True,
        "months": 20,
        "unknown_legs": unknown,
        "mean_theoretical_trend_factor_monthly_payoff": math.fsum(
            r["theoretical_trend_factor_payoff"] for r in horizon
        )
        / 20
        if not unknown
        else None,
        "mean_theoretical_long_factor_monthly_payoff": math.fsum(
            r["theoretical_long_factor_payoff"] for r in horizon
        )
        / 20
        if not unknown
        else None,
        "mean_theoretical_direction_difference": math.fsum(r["difference"] for r in horizon) / 20
        if not unknown
        else None,
        "not_net_account_performance": True,
        "scope": read(Path(__file__).with_name("review_plan.json"))["review"][1],
    }
    assert summary["actual_new_process_restart"]["passed"]
    assert not summary["capacity_triggered"]
    verification = {
        "schema": "monthly-tsmom-review.v1",
        "passed": True,
        "source_hashes": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("review_plan.json"),
                Path("research_tools/trend_stop_audit/review.py"),
                run / "completion.json",
                inputs / "completion.json",
                targets / "completion.json",
                Path(plan["input_audit"]),
            ]
        },
        "four_cash_and_stop_audits_passed": True,
        "years_and_full_account_reconcile": True,
        "actual_holding_process_restart": summary["actual_new_process_restart"],
        "maximum_internal_NAV_error_usdt": max(errors),
        "actual_new_financial_accounts": summary["actual_new_financial_accounts"],
        "new_capacity_accounts": summary["actual_capacity_accounts"],
        "capacity_triggered": False,
        "profit_goal_met": False,
        "external_review": False,
        "independent_unseen_evidence": False,
        "orders_enabled": False,
        "source_style_diagnostic_is_posthoc": True,
    }
    save(output / "metrics.json", metrics)
    save(output / "periods.json", periods)
    save(output / "account_attribution.json", details)
    save(output / "source_style_monthly_payoffs.json", horizon)
    save(output / "source_style_summary.json", horizon_summary)
    save(output / "comparisons.json", summary["comparisons"])
    save(output / "verification.json", verification)
    print(
        {"verification": verification, "horizon": horizon_summary, "attribution": details},
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    review(a.run, a.output)
