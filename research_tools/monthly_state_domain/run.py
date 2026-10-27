"""Execute the one frozen monthly state gate, with source/profile dedup before returns."""

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.study import executor
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.monthly_hold_audit.finalize import audit_exposures
from research_tools.trend_btc_role.run import calculate
from research_tools.trend_pool_audit.run import load
from research_tools.trend_stop_audit.run import paired_uncertainty


def worker(output):
    plan = read(output / "plan.json")
    for filename, digest in plan["account_source_hashes"].items():
        assert sha(Path(filename)) == digest, filename
    inputs = Path(plan["prepared_inputs"])
    _, bars, rates, lifecycle = load(Path(plan["targets"]))
    engine_path = Path(plan["engine"]).resolve()
    sys.path.insert(0, str(engine_path.parents[2]))
    r = read(output / "restart_registration.json")
    a = executor(str(engine_path)).replay(
        bars,
        rates,
        pd.read_parquet(inputs / r["target_file"]),
        r["config"],
        r["split"],
        plan["window"]["end"],
        initial_state=read(output / "restart_checkpoint.json"),
        lifecycle=lifecycle,
    )
    a["curve"].to_parquet(output / "restart_worker_curve.parquet", index=False)
    save(
        output / "restart_worker.json",
        {
            "pid": os.getpid(),
            "events": a["events"],
            "requests": a["requests"],
            "positions": a["positions"],
            "resume_state": a["resume_state"],
        },
    )


def run(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    checked = Path(str(inputs) + "_internal_audit.json")
    audited = read(checked)
    assert audited["passed"] and audited["inputs_completion_sha256"] == sha(
        inputs / "completion.json"
    )
    assert audited["source_sha256"] == sha(Path(__file__).with_name("audit_inputs.py"))
    prepared = read(inputs / "plan.json")
    for name, digest in prepared["source_hashes"].items():
        assert sha(Path(name)) == digest, name
    for parent in [
        Path(prepared["targets"]),
        Path(prepared["unconditional_run"]),
        Path(prepared["unconditional_finalization"]),
    ]:
        verify_manifest(parent)
    save(output / "protocol.json", read(Path("docs/research/regime_program/targets.json")))
    engine_path = Path(prepared["engine"]).resolve()
    original_plan, bars, rates, lifecycle = load(Path(prepared["targets"]))
    paths = [
        Path(__file__),
        engine_path,
        engine_path.with_name("sparse_sources_v2.py"),
        Path("research_tools/monthly_hold_audit/finalize.py"),
        Path("research_tools/trend_btc_role/run.py"),
        Path("research_tools/trend_pool_audit/run.py"),
        Path("research_tools/trend_stop_audit/run.py"),
        Path("research_tools/trend_stop_audit/stop_audit.py"),
        Path("research_tools/sparse_market/account_audit.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/regime_program/storage.py"),
        Path("research_tools/regime_program/reporting.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/factor_lab/study.py"),
        Path(__file__).with_name("audit_inputs.py"),
        output / "protocol.json",
        checked,
        inputs / "completion.json",
    ]
    plan = {
        **prepared,
        "prepared_inputs": str(inputs),
        "account": {**original_plan["account"], "price_stop_enabled": False},
        "account_source_hashes": {str(p): sha(p) for p in paths},
    }
    save(output / "plan.json", plan)
    frames = {}
    cases = []
    aliases = {}
    for variant in [
        "daily_archive",
        "daily_with_complete_5m_reconciliation",
    ]:
        for profile in plan["profiles"]:
            path = f"{variant}_{profile}_targets.parquet"
            frame = pd.read_parquet(inputs / path)
            for cost in plan["costs_slippage_bps"]:
                identity = f"{variant}_{profile}_cap10000_slip{cost}"
                same = next(
                    (c for c in cases if c["cost"] == cost and frames[c["id"]].equals(frame)), None
                )
                if same:
                    aliases[identity] = same["id"]
                else:
                    cases.append(
                        {
                            "id": identity,
                            "variant": variant,
                            "profile": profile,
                            "cost": cost,
                            "target_file": path,
                        }
                    )
                    frames[identity] = frame
    assert len(cases) <= plan["budget"]["maximum_actual_base_accounts"]
    save(
        output / "registration.json",
        {
            "plan_sha256": sha(output / "plan.json"),
            "before_new_financial_returns": True,
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "actual_unique_cases": cases,
            "exact_input_aliases": aliases,
            "new_domain_variants": 1,
            "new_independent_mechanisms": 0,
            "screen_effect_floor": "Strictly positive full-cost total and closed PnL and positive direction increment at both costs; necessary screen only, not an investment-useful effect or statistical sufficiency claim.",
            "known_before_returns": "Only three active monthly anchors; the registered minimum four is not met, so this gate cannot be admitted or capacity-tested regardless of returns. Run the finite registered contrasts to resolve this implementation, not to select another gate.",
        },
    )
    sys.path.insert(0, str(engine_path.parents[2]))
    engine = executor(str(engine_path))
    dates = pd.date_range(plan["window"]["start"], plan["window"]["end"], freq="D", tz="UTC")
    benchmark = bars["BTCUSDT"].close.reindex(dates - pd.Timedelta(minutes=5))
    benchmark.index = dates
    rows, curves, state = {}, {}, {}
    for case in cases:
        config = {**plan["account"], "slippage_bps": case["cost"]}
        a, receipt = calculate(
            output,
            case["id"],
            engine,
            bars,
            rates,
            frames[case["id"]],
            config,
            plan["window"],
            lifecycle,
            benchmark,
        )
        assert not any(e.get("reason") == "stop" for e in a["events"])
        row = {
            "id": case["id"],
            "profile": case["profile"],
            "variant": case["variant"],
            "capital": 10000,
            "slippage_bps": case["cost"],
            "reused_account": False,
            **receipt,
            "production_eligible": False,
            "independent_oos": False,
        }
        save(output / "accounts" / case["id"] / "row.json", row)
        rows[case["id"]] = row
        curves[case["id"]] = a["curve"]
        if case["profile"] == "conditional_trend" and case["cost"] == 10:
            state["restart"] = (a, config, case["target_file"])
        print(
            {"id": case["id"], "net_return": receipt["metrics"]["net_return"], "audited": True},
            flush=True,
        )
    for identity, source in aliases.items():
        rows[identity] = {
            **rows[source],
            "id": identity,
            "profile": "conditional_long"
            if "conditional_long" in identity
            else "conditional_trend",
            "variant": "daily_archive"
            if identity.startswith("daily_archive")
            else "daily_with_complete_5m_reconciliation",
            "reused_account": True,
            "source_account_id": source,
        }
        curves[identity] = curves[source]
    reference = Path(plan["unconditional_run"])
    reference_final = read(Path(plan["unconditional_finalization"]) / "summary.json")
    original = {r["id"]: r for r in reference_final["account_rows"] if not r["reused_account"]}
    comparisons = []
    for variant in ["daily_archive", "daily_with_complete_5m_reconciliation"]:
        for cost in plan["costs_slippage_bps"]:
            a = rows[f"{variant}_conditional_trend_cap10000_slip{cost}"]
            b = rows[f"{variant}_conditional_long_cap10000_slip{cost}"]
            c = original[f"disabled_monthly_tsmom_cap10000_slip{cost}"]
            comparisons.append(
                {
                    "variant": variant,
                    "slippage_bps": cost,
                    "conditional_trend_minus_long_net_pnl": a["metrics"]["net_pnl"]
                    - b["metrics"]["net_pnl"],
                    "conditional_minus_unconditional_net_pnl": a["metrics"]["net_pnl"]
                    - c["metrics"]["net_pnl"],
                    "direction_paired_uncertainty": paired_uncertainty(
                        curves[b["id"]], curves[a["id"]]
                    ),
                    "gate_paired_uncertainty": paired_uncertainty(
                        pd.read_parquet(reference / "accounts" / c["id"] / "equity_5m.parquet"),
                        curves[a["id"]],
                    ),
                }
            )
    restart = {"performed": False, "reason": "no actual conditional-trend stress account entries"}
    if "restart" in state and state["restart"][0]["events"]:
        a, config, target_file = state["restart"]
        first = next(e for e in a["events"] if e["type"] == "entry")
        split = first["time"]
        prefix = engine.replay(
            bars,
            rates,
            pd.read_parquet(inputs / target_file),
            config,
            plan["window"]["start"],
            split,
            lifecycle=lifecycle,
        )
        assert prefix["positions"]
        save(output / "restart_checkpoint.json", prefix["resume_state"])
        save(
            output / "restart_registration.json",
            {
                "split": split,
                "config": config,
                "target_file": target_file,
                "parent_pid": os.getpid(),
            },
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research_tools.monthly_state_domain.run",
                "--worker",
                str(output),
            ],
            check=True,
        )
        child = read(output / "restart_worker.json")
        full = a["curve"].set_index("date").loc[pd.Timestamp(split) :]
        resumed = pd.read_parquet(output / "restart_worker_curve.parquet").set_index("date")
        pd.testing.assert_frame_equal(
            full[["equity", "cash"]], resumed[["equity", "cash"]], check_exact=True
        )
        assert (
            prefix["events"] + child["events"] == a["events"]
            and prefix["requests"] + child["requests"] == a["requests"]
        )
        assert child["resume_state"] == a["resume_state"] and child["positions"] == a["positions"]
        assert child["pid"] != os.getpid()
        exposure = audit_exposures(full, resumed, prefix["resume_state"], child["events"], bars)
        save(output / "exposure_summation_audit.json", exposure)
        restart = {
            "performed": True,
            "passed": True,
            "split": split,
            "parent_pid": os.getpid(),
            "worker_pid": child["pid"],
            "held_positions": len(prefix["positions"]),
            "NAV_cash_ledger_requests_inventory_risk_exact": True,
            "exposure_independent_binary64_bound_passed": True,
        }
    active = {
        s["variant"]: s["active_monthly_anchors"] for s in read(inputs / "summary.json")["sources"]
    }
    principal = [r for r in rows.values() if r["profile"] == "conditional_trend"]
    economic = all(
        r["metrics"]["net_pnl"] > 0
        and r["metrics"]["closed_net_pnl"] > 0
        and r["metrics"]["max_drawdown"] <= 0.15
        and not r["risk_halted"]
        for r in principal
    )
    economic = economic and all(c["conditional_trend_minus_long_net_pnl"] > 0 for c in comparisons)
    sample = all(n >= 4 for n in active.values()) and all(
        r["metrics"]["closed_trades"] >= 30 for r in principal
    )
    assert not (economic and sample), (
        "Capacity branch must be completed before sealing; preserve base outputs if triggered"
    )
    save(
        output / "summary.json",
        {
            "schema": "monthly-state-domain-accounts.v1",
            "account_rows": list(rows.values()),
            "actual_new_financial_accounts": len(cases),
            "source_or_profile_alias_cases": len(aliases),
            "logical_cases": len(rows),
            "comparisons": comparisons,
            "active_monthly_anchors": active,
            "economic_followup_condition": economic,
            "minimum_sample_condition": sample,
            "capacity_triggered": economic and sample,
            "actual_capacity_accounts": 0,
            "actual_process_restart": restart,
            "profit_goal_met": False,
            "third_party_review": False,
            "independent_unseen_evidence": False,
            "orders_enabled": False,
            "production_eligible": False,
        },
    )
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {
                str(q.relative_to(output)): sha(q) for q in output.rglob("*") if q.is_file()
            },
            "profit_goal_met": False,
        },
    )
    print(
        {
            "accounts": len(cases),
            "aliases": len(aliases),
            "economic_followup": economic,
            "sample": sample,
            "restart": restart,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.worker:
        worker(a.worker)
    else:
        assert a.inputs and a.output
        run(a.inputs, a.output)
