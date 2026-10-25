"""External-first stock experiment using the existing equity account and audit.

No third-party code execution, parameter search, broker connections or orders.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from factors.equity.external_monthly import targets

from .audit_v002 import audit
from .continuous import replay, statistics
from .data import digest, load_dataset, write_json
from .diagnostics_v002 import run as diagnose
from .layered import HERE, ROOT, source_quality


def account_inputs(dataset, plan, market, candidate):
    all_frames, calendars, _ = load_dataset(dataset, plan)
    spec = plan["markets"][market]
    frames = {s["symbol"]: all_frames[s["symbol"]] for s in spec["symbols"]}
    weights, states = targets(frames, calendars[market], candidate)
    return frames, calendars[market], weights, states, spec


def recover(output):
    reg = json.loads((output / "registration.json").read_text())
    plan = json.loads((output / "protocol.json").read_text())
    for file, sha in reg["source_hashes"].items():
        if digest(ROOT / file) != sha:
            raise ValueError("Frozen source changed before recovery")
    receipts = []
    for job in json.loads((output / "restart-jobs.json").read_text()):
        folder = output / job["name"]
        checkpoint = json.loads((folder / "checkpoint.json").read_text())
        args = account_inputs(reg["dataset"], plan, job["market"], job["policy"])
        resumed = replay(
            *args,
            market=job["market"],
            policy=job["policy"],
            start="2025-01-01",
            end=plan["window"][1],
            slippage_bps=job["slippage_bps"],
            identity=reg["identity"],
            state=checkpoint,
        )
        full = json.loads((folder / "state.json").read_text())
        if resumed != full:
            raise ValueError("Actual JSON process recovery diverged")
        write_json(folder / "resumed-state.json", resumed)
        proof = {
            "passed": True,
            "method": "fresh interpreter reads actual checkpoint.json",
            "full_sha256": digest(folder / "state.json"),
            "resumed_sha256": digest(folder / "resumed-state.json"),
            "checkpoint_sha256": digest(folder / "checkpoint.json"),
        }
        write_json(folder / "restart-audit.json", proof)
        result = json.loads((folder / "result.json").read_text())
        result["restart_verified"] = True
        write_json(folder / "result.json", result)
        receipts.append({"name": job["name"], **proof})
    write_json(output / "restart-audit.json", {"passed": True, "accounts": receipts})


def run(dataset, output):
    plan = json.loads((HERE / "external_plan.json").read_text())
    goal = json.loads((HERE / "objectives.json").read_text())
    count = len(plan["markets"]) * len(plan["candidates"]) * len(plan["slippage_bps"])
    if (
        count != plan["accounts"]
        or count > goal["budget"]["first_screen_accounts"]
        or goal["production_eligible"] is not False
    ):
        raise ValueError("Unregistered account budget or production permission")
    _, _, manifest = load_dataset(dataset, plan)
    quality = source_quality(dataset, manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "goal-at-registration.json", goal)
    write_json(output / "data-quality.json", quality)
    source_files = [
        *HERE.glob("*.py"),
        ROOT / "factors/equity/external_monthly.py",
        ROOT / "factors/common/total_return.py",
        HERE / "external_plan.json",
    ]
    hashes = {}
    for file in source_files:
        relative = file.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        hashes[str(relative)] = digest(file)
    catalogue = ROOT / goal["source_catalog"]
    shutil.copyfile(catalogue, output / "source-catalog.json")
    registration = {
        "registered_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": digest(dataset / "manifest.json"),
        "source_hashes": hashes,
        "protocol_sha256": digest(output / "protocol.json"),
        "source_catalog_sha256": digest(catalogue),
        "goal_sha256": digest(output / "goal-at-registration.json"),
        "account_budget": count,
        "independent_forward_oos": False,
        "third_party_code_executed": False,
    }
    registration["identity"] = hashlib.sha256(
        json.dumps(registration, sort_keys=True).encode()
    ).hexdigest()
    write_json(output / "registration.json", registration)
    records, audits, jobs = [], [], []
    for market in plan["markets"]:
        for candidate in plan["candidates"]:
            args = account_inputs(dataset, plan, market, candidate)
            for slip in plan["slippage_bps"]:
                spec = plan["markets"][market]
                name = f"{market}_{candidate}_{spec['initial_cash']}_{slip}bps"
                folder = output / name
                folder.mkdir()
                common = {
                    "market": market,
                    "policy": candidate,
                    "slippage_bps": slip,
                    "identity": registration["identity"],
                }
                state = replay(*args, start=plan["window"][0], end=plan["window"][1], **common)
                checkpoint = replay(*args, start=plan["window"][0], end="2025-01-01", **common)
                write_json(folder / "state.json", state)
                write_json(folder / "checkpoint.json", checkpoint)
                proof = audit(state, args[0], args[1])
                proof["state_sha256"] = digest(folder / "state.json")
                write_json(folder / "independent-audit.json", proof)
                metrics = statistics(state)
                result = {
                    "schema": "ta-equity-account.v2",
                    "asset_class": "equity",
                    "market": market,
                    "currency": spec["currency"],
                    "candidate": candidate,
                    "phase": "external_exploratory_diagnostic",
                    "status": "外部公开规则移植 · 未达高收益目标",
                    "identity": registration["identity"],
                    "initial_cash": spec["initial_cash"],
                    "slippage_bps_per_side": slip,
                    "provider": manifest["provider"],
                    "broker_verified": False,
                    "dividend_cash_verified": False,
                    "historical_lots_verified": False,
                    "production_eligible": False,
                    "independent_forward_oos": False,
                    "economic_account_evaluated": True,
                    "ledger_verified": True,
                    "restart_verified": False,
                    "window": {"start_at": plan["window"][0], "end_at": plan["window"][1]},
                    "account": {
                        "metrics": metrics,
                        "equity_curve": state["equity_curve"],
                        "trades": state["trades"],
                    },
                    "limitations": plan["deviations"] + plan["limitations"],
                }
                write_json(folder / "result.json", result)
                shutil.copyfile(output / "protocol.json", folder / "protocol.json")
                shutil.copyfile(output / "data-quality.json", folder / "data-quality.json")
                job = {
                    "name": name,
                    "market": market,
                    "policy": candidate,
                    "capital": spec["initial_cash"],
                    "slippage_bps": slip,
                }
                records.append({**job, "metrics": metrics})
                jobs.append(job)
                audits.append({"name": name, **proof})
                print(
                    name,
                    "CAGR",
                    round(metrics["annualized_return"], 6),
                    "cycles",
                    metrics["closed_trades"],
                    flush=True,
                )
    write_json(output / "restart-jobs.json", jobs)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tradingagents.research.equities.external_first",
            "--recover",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    write_json(output / "independent-audit.json", {"passed": True, "accounts": audits})
    references = ROOT / "results/research/equities_layered_20261026_v001"
    baselines = []
    for market, spec in plan["markets"].items():
        for slip in plan["slippage_bps"]:
            for policy in ("static", "buy_hold"):
                path = (
                    references
                    / f"{market}_{policy}_{spec['initial_cash']}_{slip}bps"
                    / "result.json"
                )
                baseline = json.loads(path.read_text())
                if (
                    baseline["initial_cash"] != spec["initial_cash"]
                    or baseline["slippage_bps_per_side"] != slip
                    or baseline["window"]
                    != {"start_at": plan["window"][0], "end_at": plan["window"][1]}
                ):
                    raise ValueError("Baseline economic scope mismatch")
                baselines.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": digest(path),
                        "market": market,
                        "policy": policy,
                        "slippage_bps": slip,
                        "capital": spec["initial_cash"],
                        "metrics": baseline["account"]["metrics"],
                    }
                )
    write_json(
        output / "summary.json",
        {
            "schema": "ta-equity-external-study.v1",
            "asset_class": "equity",
            "status": "外部优先8账户完成；挑战未达标，旧cash选择保留",
            "account_count": len(records),
            "records": records,
            "reused_baselines": baselines,
            "profit_goal_status": "not_met",
            "challenge_cagr": 0.5,
            "max_drawdown_target": 0.15,
            "actual_qualified_total_account": None,
            "production_eligible": False,
            "independent_forward_oos": False,
            "limitations": plan["deviations"] + plan["limitations"],
        },
    )
    diagnose(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recover", type=Path)
    args = parser.parse_args()
    if args.recover:
        recover(args.recover)
    elif args.dataset and args.output:
        run(args.dataset, args.output)
    else:
        parser.error("--dataset and --output required")


if __name__ == "__main__":
    main()
