"""Eight preregistered individual-stock diagnostics on complete common history.

The original longer protocol stays blocked by a delisted constituent's missing
prices. No removal of that constituent or performance-driven window choice.
"""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from factors.equity.disclosed_stocks import targets

from .audit_v002 import audit
from .continuous import replay, statistics
from .data import digest, validate, write_json
from .diagnostics_v002 import run as diagnose
from .pit_data import HERE, ROOT
from .pit_quality import inspect


def load(pit):
    plan = json.loads((pit / "protocol.json").read_text())
    identities = json.loads((pit / "identities.json").read_text())
    annual = json.loads((pit / "annual.json").read_text())
    prices = pit / "prices"
    calendar = pd.read_csv(prices / "signal_sessions.csv", index_col="date", parse_dates=["date"])
    for key in ["open", "close"]:
        calendar[key] = pd.to_datetime(calendar[key], utc=True)
    frames = {
        r["symbol"]: pd.read_csv(
            prices / f"{r['symbol']}_signal.csv", index_col="date", parse_dates=["date"]
        )
        for r in identities
    }
    books, states, evidence = targets(frames, calendar, identities, annual, plan)
    last = min(f.index.max() for f in frames.values())
    account_calendar = calendar.loc["2022-12-30":last]
    account_frames = {s: f.loc[account_calendar.index] for s, f in frames.items()}
    for f in account_frames.values():
        validate(f, account_calendar)
    spec = {
        **plan["market"],
        "symbols": [{k: r[k] for k in ["symbol", "instrument", "lot_size"]} for r in identities],
    }
    return (
        plan,
        identities,
        annual,
        frames,
        calendar,
        books,
        states,
        evidence,
        account_frames,
        account_calendar,
        spec,
    )


def causal_audit(inputs):
    plan, ids, annual, frames, calendar, books = inputs[:6]
    changed, facts = copy.deepcopy(frames), copy.deepcopy(annual)
    cutoff = pd.Timestamp("2024-01-01")
    for f in changed.values():
        later = f.index >= cutoff
        f.loc[later, ["open", "high", "low", "close"]] *= 11
        f.loc[later, "dividend"] = 99
    for r in facts:
        if (
            pd.Timestamp(r["available_at"]) >= pd.Timestamp("2024-01-01T00:00:00Z")
            and r["gross_profitability"] is not None
        ):
            r["gross_profitability"] *= 101
    perturbed, _, _ = targets(changed, calendar, ids, facts, plan)
    prefix = {s: f.loc[:"2023-07-14"] for s, f in frames.items()}
    truncated, _, _ = targets(prefix, calendar, ids, annual, plan)
    for name in books:
        pd.testing.assert_frame_equal(
            books[name].loc[:"2023-12-31"], perturbed[name].loc[:"2023-12-31"]
        )
        pd.testing.assert_frame_equal(books[name].loc[:"2023-07-14"], truncated[name])
    return {
        "passed": True,
        "real_data_future_perturbation": True,
        "prefix_invariance": True,
        "policies": list(books),
        "future_cutoff": "2024-01-01",
        "prefix_end": "2023-07-14",
    }


def verify_originals(pit):
    proof = json.loads((pit / "original-annual-audit.json").read_text())
    if (
        not proof["passed"]
        or proof["auditor_source_sha256"] != digest(HERE / "pit_originals.py")
        or digest(pit / "facts.json") != proof["facts_sha256"]
        or digest(pit / "annual.json") != proof["annual_sha256"]
    ):
        raise ValueError("Original annual reconciliation missing or stale")
    for r in proof["source_receipts"]:
        if digest(pit / "originals" / r["file"]) != r["sha256"]:
            raise ValueError("Original financial filing changed")
    return proof


def recover(output):
    reg = json.loads((output / "registration.json").read_text())
    pit = Path(reg["pit_dataset"])
    for p, sha in reg["source_hashes"].items():
        if digest(ROOT / p) != sha:
            raise ValueError("Source changed before interpreter restart")
    quality = inspect(pit)
    if (
        quality["input_hashes"] != reg["input_hashes"]
        or quality["price_manifest_sha256"] != reg["price_manifest_sha256"]
    ):
        raise ValueError("Input changed before recovery")
    verify_originals(pit)
    inputs = load(pit)
    _, _, _, _, _, books, states, _, frames, calendar, spec = inputs
    receipts = []
    for job in json.loads((output / "restart-jobs.json").read_text()):
        folder = output / job["name"]
        checkpoint = json.loads((folder / "checkpoint.json").read_text())
        state = replay(
            frames,
            calendar,
            books[job["policy"]].loc[calendar.index],
            states.loc[calendar.index],
            spec,
            market="US",
            start="2024-01-01",
            end=reg["bounded_window"][1],
            policy=job["policy"],
            slippage_bps=job["slippage_bps"],
            identity=reg["identity"],
            state=checkpoint,
        )
        if state != json.loads((folder / "state.json").read_text()):
            raise ValueError("Actual JSON recovery diverged")
        write_json(folder / "resumed-state.json", state)
        receipt = {
            "passed": True,
            "method": "fresh Python interpreter reads actual serialized inventory/cash/high-water checkpoint",
            "checkpoint_sha256": digest(folder / "checkpoint.json"),
            "full_sha256": digest(folder / "state.json"),
            "resumed_sha256": digest(folder / "resumed-state.json"),
            "checkpoint_positions": len(checkpoint["positions"]),
        }
        write_json(folder / "restart-audit.json", receipt)
        result = json.loads((folder / "result.json").read_text())
        result["restart_verified"] = True
        write_json(folder / "result.json", result)
        receipts.append({"name": job["name"], **receipt})
    write_json(output / "restart-audit.json", {"passed": True, "accounts": receipts})


def run(pit, output):
    quality, original = inspect(pit), verify_originals(pit)
    inputs = load(pit)
    plan, ids, _, _, _, books, states, evidence, frames, calendar, spec = inputs
    causal = causal_audit(inputs)
    count = len(plan["policies"]) * len(plan["slippage_bps"])
    if count != 8 or count != plan["accounts"]:
        raise ValueError("Unregistered account count")
    end = str((calendar.index[-1] + pd.Timedelta(days=1)).date())
    window = [plan["window"][0], min(plan["window"][1], end)]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    amendment = {
        "registered_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "registered_before_returns": True,
        "original_requested_window": plan["window"],
        "bounded_window": window,
        "reason": "EXAS was acquired/delisted; Yahoo history unavailable. CC0 2024 academic archive recovers complete common member history only through 2024-06-26. End is mechanically min(last observed date of every frozen member)+1, fixed before returns. No member removed or imputed and no factor/weight/fee changed.",
        "original_full_window_status": "blocked_delisted_prices_and_terminal_settlement",
        "new_account_budget": count,
        "independent_forward_oos": False,
        "production_eligible": False,
    }
    write_json(output / "coverage-amendment.json", amendment)
    write_json(output / "data-quality.json", quality)
    write_json(output / "original-annual-audit.json", original)
    write_json(output / "causal-audit.json", causal)
    write_json(
        output / "financial-decision-provenance.json",
        [r for r in evidence if r["decision_day"] < window[1]],
    )
    # Copy every local implementation needed to reproduce these diagnostics.
    source_files = [
        *HERE.glob("*.py"),
        HERE / "stock_protocol.json",
        ROOT / "factors/equity/disclosed_stocks.py",
        ROOT / "factors/common/total_return.py",
    ]
    hashes = {}
    for f in source_files:
        relative = f.relative_to(ROOT)
        dest = output / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
        hashes[str(relative)] = digest(f)
    reg = {
        "registered_at_utc": amendment["registered_at_utc"],
        "pit_dataset": str(pit.resolve()),
        "source_hashes": hashes,
        "input_hashes": quality["input_hashes"],
        "price_manifest_sha256": quality["price_manifest_sha256"],
        "bounded_window": window,
        "coverage_amendment_sha256": digest(output / "coverage-amendment.json"),
        "original_annual_audit_sha256": digest(pit / "original-annual-audit.json"),
        "accounts": count,
    }
    reg["identity"] = hashlib.sha256(json.dumps(reg, sort_keys=True).encode()).hexdigest()
    write_json(output / "registration.json", reg)
    records, audits = [], []
    for policy in plan["policies"]:
        for slip in plan["slippage_bps"]:
            name = f"US_{policy}_10000_{slip}bps"
            folder = output / name
            folder.mkdir()
            args = (
                frames,
                calendar,
                books[policy].loc[calendar.index],
                states.loc[calendar.index],
                spec,
            )
            kwargs = {
                "market": "US",
                "policy": policy,
                "slippage_bps": slip,
                "identity": reg["identity"],
            }
            state = replay(*args, start=window[0], end=window[1], **kwargs)
            checkpoint = replay(*args, start=window[0], end="2024-01-01", **kwargs)
            write_json(folder / "state.json", state)
            write_json(folder / "checkpoint.json", checkpoint)
            proof = {
                **audit(state, frames, calendar),
                "state_sha256": digest(folder / "state.json"),
            }
            write_json(folder / "independent-audit.json", proof)
            metrics = statistics(state)
            result = {
                "schema": "ta-equity-account.v2",
                "asset_class": "equity",
                "market": "US",
                "currency": "USD",
                "candidate": policy,
                "phase": "historical_disclosed_stock_bounded_diagnostic",
                "status": "历史个股池 · 覆盖受限诊断 · 长窗口仍被退市数据阻断",
                "identity": reg["identity"],
                "initial_cash": 10000,
                "slippage_bps_per_side": slip,
                "provider": "SEC as-filed disclosure + Yahoo survivors + CC0 Zenodo EXAS archive",
                "broker_verified": False,
                "dividend_cash_verified": False,
                "historical_lots_verified": False,
                "production_eligible": False,
                "independent_forward_oos": False,
                "economic_account_evaluated": True,
                "ledger_verified": True,
                "restart_verified": False,
                "window": {"start_at": window[0], "end_at": window[1]},
                "account": {
                    "metrics": metrics,
                    "equity_curve": state["equity_curve"],
                    "trades": state["trades"],
                },
                "limitations": [amendment["reason"], plan["paper_scope"], *quality["limitations"]],
            }
            write_json(folder / "result.json", result)
            for f in ["protocol.json", "data-quality.json", "coverage-amendment.json"]:
                shutil.copyfile(output / f, folder / f)
            records.append(
                {
                    "name": name,
                    "market": "US",
                    "policy": policy,
                    "capital": 10000,
                    "slippage_bps": slip,
                    "metrics": metrics,
                }
            )
            audits.append({"name": name, **proof})
            print(
                name,
                "CAGR",
                round(metrics["annualized_return"], 6),
                "DD",
                round(metrics["max_drawdown"], 6),
                "cycles",
                metrics["closed_trades"],
                flush=True,
            )
    write_json(
        output / "restart-jobs.json",
        [{k: r[k] for k in ["name", "policy", "slippage_bps"]} for r in records],
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tradingagents.research.equities.pit_stocks",
            "--recover",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    write_json(output / "independent-audit.json", {"passed": True, "accounts": audits})
    summary = {
        "schema": "equity-pit-stock-summary.v1",
        "bounded_window": window,
        "original_requested_window": plan["window"],
        "records": records,
        "new_accounts": count,
        "profit_goal_status": "not_met",
        "actual_qualified_total_account": None,
        "full_window_available": False,
        "production_eligible": False,
        "independent_forward_oos": False,
    }
    write_json(output / "summary.json", summary)
    diagnose(output)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pit", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--recover", type=Path)
    args = p.parse_args()
    if args.recover:
        recover(args.recover)
    elif args.pit and args.output:
        run(args.pit, args.output)
    else:
        p.error("--pit and --output or --recover required")
