"""Finite preregistered MAX continuation study, exact funding and matched controls."""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.github_factor_study.run import snapshot_sources
from research_tools.max_effect.account_audit import audit_account
from research_tools.regime_program.module_gate import assess_module
from research_tools.regime_program.reporting import cluster_uncertainty
from research_tools.regime_program.storage import write_account


def verify_manifest(root):
    for name, digest in read(root / "completion.json")["manifest"].items():
        if root.resolve() not in (root / name).resolve().parents or sha(root / name) != digest:
            raise ValueError("Sealed input changed: " + str(root / name))


def run(root, inputs):  # noqa: C901
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    frozen = read(inputs / "plan.json")
    for name, digest in frozen["source_hashes"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Prepared implementation changed: " + name)
    input_audit_path = inputs.with_name(inputs.name + "_independent_audit.json")
    input_audit = read(input_audit_path)
    if not input_audit["passed"] or input_audit["input_completion_sha256"] != sha(
        inputs / "completion.json"
    ):
        raise ValueError("Input arithmetic audit missing or changed")
    data, funding = Path(frozen["data"]), Path(frozen["funding_data"])
    verify_manifest(data)
    verify_manifest(funding)
    save(root / "protocol.json", read(Path("docs/research/regime_program/targets.json")))
    plan = {
        **frozen,
        "inputs": str(inputs.resolve()),
        "engine": str(Path("../freqtrade/research/crypto/calendar_portfolio.py").resolve()),
        "registered_at_utc": datetime.now(UTC).isoformat(),
    }
    cash_parent = Path("results/research/regime_accounts_20261026_v001")
    paths = [
        Path(__file__),
        Path(plan["engine"]),
        Path("research_tools/max_effect/account_audit.py"),
        Path("research_tools/max_effect/audit_inputs.py"),
        Path("research_tools/regime_program/storage.py"),
        Path("research_tools/regime_program/reporting.py"),
        Path("research_tools/regime_program/module_gate.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/factor_lab/study.py"),
        Path("research_tools/github_factor_study/run.py"),
        root / "protocol.json",
        inputs / "completion.json",
        input_audit_path,
        data / "completion.json",
        funding / "completion.json",
        cash_parent / "summary.json",
    ]
    plan["account_source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {"plan_sha256": sha(root / "plan.json"), "before_account_returns": True},
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    symbols = read(data / "prepare_plan.json")["symbols"]
    bars = {s: pd.read_parquet(data / (s + "_5m.parquet")) for s in symbols}
    rates = {s: pd.read_parquet(funding / (s + "_exact.parquet")) for s in symbols}
    engine = executor(plan["engine"])
    rows, audits = [], []

    def one(candidate, window, cost, capital=10000, phase=None):
        config = {**plan["account"], "capital": capital, "slippage_bps": cost}
        scores = pd.read_parquet(inputs / f"{window['id']}_{candidate}_targets.parquet")
        a = engine.replay(bars, rates, scores, config, window["start"], window["end"])
        days = pd.date_range(window["start"], window["end"], freq="D", tz="UTC")
        benchmark = bars["BTCUSDT"].close.reindex(days - pd.Timedelta(minutes=5))
        benchmark.index = days
        m = portfolio_metrics(a["curve"], a["trades"], a["events"], capital, benchmark)
        m["mean_absolute_net_exposure"] = float(
            (a["curve"].net_exposure / a["curve"].equity).abs().mean()
        )
        m["open_campaign_net_pnl"] = (
            sum(p["net_pnl"] for p in a["positions"].values()) + m["open_pnl_with_exit_reserve"]
        )
        if abs(m["closed_net_pnl"] + m["open_campaign_net_pnl"] - m["net_pnl"]) > 1e-6:
            raise ValueError("Lifecycle net PnL does not reconcile")
        a["metrics"] = m
        phase = phase or window["id"]
        identity = f"{phase}_{candidate}_cap{capital}_slip{cost}"
        write_account(root / "accounts" / identity, a, config, window)
        row = {
            "id": identity,
            "candidate": candidate,
            "phase": phase,
            "capital": capital,
            "slippage_bps": cost,
            "start": window["start"],
            "end": window["end"],
            "kind": "control" if candidate.startswith("momentum") else "module",
            "reused_reference": False,
            "source_run": str(root.resolve()),
            "source_account_id": identity,
            "metrics": m,
            "capacity": a["capacity"],
            "risk_halted": a["risk_halted"],
            "lifecycle_uncertainty": cluster_uncertainty(a["trades"]),
            "result_sha256": sha(root / "accounts" / identity / "result.json"),
        }
        audit = audit_account(root, row, bars, rates, scores)
        row["audit_passed"] = audit["passed"]
        save(root / "accounts" / identity / "independent_audit.json", audit)
        rows.append(row)
        audits.append(audit)
        save(root / "accounts" / identity / "row.json", row)
        print(
            {
                "account": len(audits),
                "id": identity,
                "net_return": m["net_return"],
                "closed_net_pnl": m["closed_net_pnl"],
                "closed": m["closed_trades"],
                "halted": a["risk_halted"],
                "audited": True,
                "seconds": round(time.monotonic() - started, 1),
            },
            flush=True,
        )

    for window in plan["windows"]:
        for candidate in plan["candidates"]:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, window, cost)
    continuous = plan["windows"][-1]
    followups = []
    for candidate in ["max28", "max28_btc_beta"]:
        cases = [r for r in rows if r["phase"] == continuous["id"] and r["candidate"] == candidate]
        if len(cases) != 2:
            raise ValueError("Missing matched-cost continuous cases")
        if all(
            r["metrics"]["net_return"] > 0
            and r["metrics"]["closed_net_pnl"] > 0
            and not r["risk_halted"]
            for r in cases
        ):
            followups.append(candidate)
    save(
        root / "capacity_selection.json",
        {"followups": followups, "rule": plan["capacity_followup"], "is_admission": False},
    )
    for candidate in followups:
        for capital in plan["capacity_capitals"]:
            for cost in plan["costs_slippage_bps"]:
                one(candidate, continuous, cost, capital, "capacity")
    protocol = read(root / "protocol.json")
    assessments = []
    for candidate in ["max28", "max28_btc_beta"]:
        cases = [r for r in rows if r["phase"] == continuous["id"] and r["candidate"] == candidate]
        groups = min(
            max((v or {}).get("clusters", 0) for v in r["lifecycle_uncertainty"].values())
            for r in cases
        )
        assessments.append(
            {
                "candidate": candidate,
                **assess_module(
                    cases,
                    protocol["module_acceptance"],
                    domain_registered=True,
                    sleeve_capacity_passed=None,
                    cluster_groups=groups,
                ),
            }
        )
    for old in read(cash_parent / "summary.json")["account_rows"]:
        if old["candidate"] != "cash" or old["capital"] != 10000:
            continue
        folder = cash_parent / "accounts" / old["id"]
        if sha(folder / "result.json") != old["result_sha256"]:
            raise ValueError("Frozen cash reference changed")
        report = read(folder / "result.json")
        for name, digest in report["source_hashes"].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError("Frozen cash ledger changed")
        rows.append(
            {
                **old,
                "id": "reference_" + old["id"],
                "kind": "control",
                "reused_reference": True,
                "source_run": str(cash_parent.resolve()),
                "source_account_id": old["id"],
                "audit_passed": True,
            }
        )
    if (
        not 24 <= len(audits) <= plan["budgets"]["total_new_accounts_max"]
        or len(rows) - len(audits) != 6
    ):
        raise ValueError("Finite account budget violated")
    summary = {
        "schema": "max-effect-account-results.v1",
        "account_rows": rows,
        "new_accounts": len(audits),
        "reused_references": 6,
        "audit": {
            "passed": all(a["passed"] for a in audits),
            "accounts": len(audits),
            "maximum_nav_error": max(a["max_nav_error"] for a in audits),
        },
        "audit_accounts": audits,
        "input_audit": input_audit,
        "module_assessments": assessments,
        "selection": {
            "capacity_followups": followups,
            "admitted_candidates": [],
            "portfolio_built": False,
        },
        "profit_goal_met": False,
        "portfolio_built": False,
        "independent_oos": False,
        "production_eligible": False,
        "orders_enabled": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    save(root / "summary.json", summary)
    save(
        root / "completion.json",
        {
            "summary_sha256": sha(root / "summary.json"),
            "plan_sha256": sha(root / "plan.json"),
            "passed": summary["audit"]["passed"],
        },
    )
    print({"completed": True, "new_accounts": len(audits), "assessments": assessments}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    args = p.parse_args()
    run(args.output, args.inputs)
