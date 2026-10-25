"""Frozen bounded ETF diagnostics and process-restart evidence.

python -m tradingagents.research.equities.layered --dataset PATH --output NEW_PATH
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from factors.equity.opportunities import targets

from .audit_v002 import audit
from .continuous import replay, statistics
from .data import digest, load_dataset, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def source_quality(dataset, manifest):
    dataset = Path(dataset)
    required = {}
    for market, sha in manifest["calendar_hashes"].items():
        required[f"{market}_sessions.csv"] = sha
    for row in manifest["symbols"].values():
        required[row["file"]] = row["sha256"]
        required[row["vendor_file"]] = row["vendor_sha256"]
    for rows in manifest.get("hk_price_sources", {}).values():
        for row in rows:
            required[row["file"]] = row["sha256"]
    checks = []
    for name, expected in required.items():
        file = dataset / name
        if file.resolve().parent != dataset.resolve() or digest(file) != expected:
            raise ValueError(f"Source identity mismatch: {name}")
        checks.append({"file": name, "sha256": expected})
    return {
        "passed": True,
        "files": checks,
        "broker_verified": False,
        "corporate_action_authority": manifest["corporate_action_authority"],
        "pit_fundamentals": False,
        "historical_lots_verified": False,
        "calendar_official_exceptions_complete": False,
        "volume_units_certified": False,
    }


def inputs(dataset, plan, market, policy, capital):
    all_frames, calendars, _ = load_dataset(dataset, plan)
    spec = {**plan["markets"][market], "initial_cash": capital}
    frames = {s["symbol"]: all_frames[s["symbol"]] for s in spec["symbols"]}
    weights, regimes = targets(frames, policy)
    return frames, calendars[market], weights, regimes, spec


def resume_jobs(output):
    """New interpreter reads checkpoints and resumes through the JSON boundary."""
    registration = json.loads((output / "registration.json").read_text())
    plan = json.loads((output / "protocol.json").read_text())
    # Fail closed if any source changed between the original run and recovery.
    for relative, sha in registration["source_hashes"].items():
        if digest(ROOT / relative) != sha:
            raise ValueError("Source changed before process restart")
    receipts = []
    for job in json.loads((output / "restart-jobs.json").read_text()):
        path = output / job["name"]
        args = inputs(registration["dataset"], plan, job["market"], job["policy"], job["capital"])
        checkpoint = json.loads((path / "checkpoint.json").read_text())
        full = json.loads((path / "state.json").read_text())
        resumed = replay(
            *args,
            market=job["market"],
            policy=job["policy"],
            start="2025-01-01",
            end=plan["window"][1],
            slippage_bps=job["slippage_bps"],
            identity=registration["identity"],
            state=checkpoint,
        )
        write_json(path / "resumed-state.json", resumed)
        passed = resumed == full
        if not passed:
            raise ValueError(f"Process-restart state divergence: {job['name']}")
        proof = {
            "passed": passed,
            "method": "new Python interpreter, actual checkpoint.json read",
            "checkpoint_sha256": digest(path / "checkpoint.json"),
            "full_sha256": digest(path / "state.json"),
            "resumed_sha256": digest(path / "resumed-state.json"),
            "fields": [
                "cash",
                "receivables",
                "positions",
                "unsettled",
                "peak",
                "halted",
                "halt_until",
                "fills",
                "trades",
                "requests",
                "equity_curve",
            ],
            "checkpoint_open_positions": len(checkpoint["positions"]),
            "checkpoint_pending_cash": len(checkpoint["unsettled"]),
            "checkpoint_peak": checkpoint["peak"],
            "checkpoint_halted": checkpoint["halted"],
        }
        write_json(path / "restart-audit.json", proof)
        result = json.loads((path / "result.json").read_text())
        result["restart_verified"] = True
        write_json(path / "result.json", result)
        receipts.append({"name": job["name"], **proof})
    write_json(
        output / "restart-audit.json",
        {"passed": all(r["passed"] for r in receipts), "accounts": receipts},
    )


def run(dataset, output):
    plan = json.loads((HERE / "protocol_v002.json").read_text())
    count = sum(
        len(s["capitals"]) * len(plan["slippage_bps"]) * len(plan["policies"])
        for s in plan["markets"].values()
    )
    if count != plan["account_budget"]:
        raise ValueError("Frozen account budget mismatch")
    # This experiment deliberately offers no parameter CLI or strategy search.
    frames, calendars, manifest = load_dataset(dataset, plan)
    quality = source_quality(dataset, manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "data-quality.json", quality)
    paths = [
        *HERE.glob("*.py"),
        HERE / "protocol_v002.json",
        HERE / "sources_v002.json",
        ROOT / "factors/equity/opportunities.py",
        ROOT / "factors/common/session_momentum.py",
    ]
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        target = output / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(rel)] = digest(path)
    method = ROOT / "docs/research/regime_program/targets.json"
    methods = json.loads(method.read_text())
    if methods["version"] != "20261026_v002":
        raise ValueError("Review the changed multi-layer method contract first")
    registration = {
        "schema": "ta-equity-registration.v2",
        "registered_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset": str(dataset.resolve()),
        "dataset_manifest_sha256": digest(dataset / "manifest.json"),
        "protocol_sha256": digest(output / "protocol.json"),
        "source_hashes": hashes,
        "method_contract_sha256": digest(method),
        "method_contract_version": methods["version"],
        "account_budget": count,
        "history_role": plan["history_role"],
        "independent_forward_oos": False,
        "promotion_allowed": False,
        "python": sys.version,
    }
    registration["identity"] = hashlib.sha256(
        json.dumps(registration, sort_keys=True).encode()
    ).hexdigest()
    # This is persisted BEFORE any new account return is calculated.
    write_json(output / "registration.json", registration)
    write_json(
        output / "selection.json",
        {
            "old_selection": {"US": "cash", "HK": "cash"},
            "new_selection": None,
            "reason": "exploration only; no result-based policy promotion; original 108 accounts untouched",
        },
    )
    records, jobs, audits = [], [], []
    for market, market_spec in plan["markets"].items():
        subset = {s["symbol"]: frames[s["symbol"]] for s in market_spec["symbols"]}
        for policy in plan["policies"]:
            weights, regimes = targets(subset, policy)
            for capital in market_spec["capitals"]:
                spec = {**market_spec, "initial_cash": capital}
                for slip in plan["slippage_bps"]:
                    name = f"{market}_{policy}_{capital}_{slip}bps"
                    target = output / name
                    target.mkdir()
                    args = (subset, calendars[market], weights, regimes, spec)
                    common = {
                        "market": market,
                        "slippage_bps": slip,
                        "identity": registration["identity"],
                        "policy": policy,
                    }
                    state = replay(*args, start=plan["window"][0], end=plan["window"][1], **common)
                    checkpoint = replay(*args, start=plan["window"][0], end="2025-01-01", **common)
                    write_json(target / "state.json", state)
                    write_json(target / "checkpoint.json", checkpoint)
                    proof = audit(state, subset, calendars[market])
                    write_json(target / "independent-audit.json", proof)
                    m = statistics(state)
                    result = {
                        "schema": "ta-equity-account.v2",
                        "asset_class": "equity",
                        "market": market,
                        "currency": spec["currency"],
                        "candidate": policy,
                        "phase": "exploratory_continuous",
                        "status": "ETF分层探索 · 未获生产资格",
                        "provider": manifest["provider"],
                        "broker_verified": False,
                        "dividend_cash_verified": False,
                        "historical_lots_verified": False,
                        "independent_forward_oos": False,
                        "economic_account_evaluated": True,
                        "production_eligible": False,
                        "ledger_verified": True,
                        "restart_verified": False,
                        "window": {"start_at": plan["window"][0], "end_at": plan["window"][1]},
                        "identity": registration["identity"],
                        "slippage_bps_per_side": slip,
                        "initial_cash": capital,
                        "cost_model": "dated statutory components + scenario commission + adverse open slippage",
                        "account": {
                            "metrics": m,
                            "equity_curve": state["equity_curve"],
                            "trades": state["trades"],
                        },
                        "limitations": plan["limitations"],
                    }
                    write_json(target / "result.json", result)
                    shutil.copyfile(output / "protocol.json", target / "protocol.json")
                    shutil.copyfile(output / "data-quality.json", target / "data-quality.json")
                    records.append(
                        {
                            "name": name,
                            "market": market,
                            "policy": policy,
                            "capital": capital,
                            "slippage_bps": slip,
                            "metrics": m,
                        }
                    )
                    jobs.append(
                        {
                            k: records[-1][k]
                            for k in ("name", "market", "policy", "capital", "slippage_bps")
                        }
                    )
                    audits.append({"name": name, **proof})
                    print(
                        name,
                        "return",
                        round(m["net_return"], 6),
                        "cycles",
                        m["closed_trades"],
                        flush=True,
                    )
    write_json(output / "restart-jobs.json", jobs)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tradingagents.research.equities.layered",
            "--resume-jobs",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    write_json(output / "independent-audit.json", {"passed": True, "accounts": audits})
    comparisons = []
    for row in records:
        if row["policy"] in {"risk_scaled", "finite_state", "trend_only", "recovery_only"}:
            base = next(
                r
                for r in records
                if r["market"] == row["market"]
                and r["capital"] == row["capital"]
                and r["slippage_bps"] == row["slippage_bps"]
                and r["policy"] == "static"
            )
            comparisons.append(
                {
                    "case": row["name"],
                    "baseline": base["name"],
                    "delta_net_return": row["metrics"]["net_return"]
                    - base["metrics"]["net_return"],
                    "delta_close_drawdown": row["metrics"]["close_max_drawdown"]
                    - base["metrics"]["close_max_drawdown"],
                    "delta_turnover": row["metrics"]["turnover_initial_capital"]
                    - base["metrics"]["turnover_initial_capital"],
                    "same_realized_risk_verified": False,
                }
            )
    write_json(
        output / "summary.json",
        {
            "schema": "ta-equity-layered-study.v2",
            "asset_class": "equity",
            "status": "56个ETF探索账户及财务/进程重启核对完成；旧cash选择保留",
            "account_count": len(records),
            "production_eligible": False,
            "independent_forward_oos": False,
            "broker_verified": False,
            "selection": None,
            "records": records,
            "comparisons": comparisons,
            "limitations": plan["limitations"],
        },
    )
    print("Completed", len(records), "bounded equity accounts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-jobs", type=Path)
    args = parser.parse_args()
    if args.resume_jobs:
        resume_jobs(args.resume_jobs)
    elif args.dataset and args.output:
        run(args.dataset, args.output)
    else:
        parser.error("--dataset and --output, or --resume-jobs required")


if __name__ == "__main__":
    main()
