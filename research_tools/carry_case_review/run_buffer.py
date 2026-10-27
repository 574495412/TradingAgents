"""Execute two finite diagnostic cases, internal cash audits and actual process restores."""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from research_tools.carry_case_review.account_audit import audit_account
from research_tools.carry_case_review.worker_buffer import inputs
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def attribution(folder, source, symbol):
    events = read(folder / "events.json")
    report = read(folder / "result.json")
    spot, perp, marks, _ = inputs(source, symbol)
    prices = {"spot": spot[symbol], "perp": perp[symbol]}
    gross_spot = gross_perp = fees = slippage = funding = engine_slippage = 0.0
    entries = []
    exits = []
    for e in events:
        if e["type"] == "funding":
            funding += e["cash_delta"]
            continue
        if e["type"] == "transfer":
            continue
        bar = prices[e["market"]].loc[pd.Timestamp(e["time"]) - pd.Timedelta(minutes=5)]
        q = e["quantity"]
        mid = float(bar.close)
        fees += e["fee"]
        slippage += q * abs(e["price"] - mid)
        engine_slippage += e["slippage_cost"]
        if e["market"] == "spot":
            gross_spot += (1 if e["type"] == "exit" else -1) * q * mid
        else:
            gross_perp += (1 if e["type"] == "entry" else -1) * q * mid
        if e["type"] == "entry":
            entries.append({**e, "source_close": mid})
        else:
            exits.append({**e, "source_close": mid})
    state = read(folder / "state.json")
    end = pd.Timestamp(report["window"]["end"]) - pd.Timedelta(minutes=5)
    residual = read(folder / "positions.json").get(symbol)
    if residual:
        gross_spot += residual["spot_q"] * float(spot[symbol].loc[end, "close"])
        gross_perp -= residual["perp_q"] * float(marks[symbol].loc[end, "close"])
    curve = pd.read_parquet(folder / "equity_5m.parquet")
    reserve = float(curve.exit_reserve.iloc[-1])
    model = gross_spot + gross_perp + funding - fees - slippage - reserve
    if abs(model - report["metrics"]["net_pnl"]) > 1e-7:
        raise ValueError("Price/funding/fee/slippage cash decomposition fails")
    paired = (
        len(entries) == 2
        and len(exits) == 2
        and not residual
        and all(abs(e["quantity"] - entries[0]["quantity"]) < 1e-10 for e in [*entries, *exits])
    )
    basis = None
    if paired:
        entry = {e["market"]: e for e in entries}
        exit_ = {e["market"]: e for e in exits}
        initial = entry["spot"]["source_close"] - entry["perp"]["source_close"]
        final = exit_["spot"]["source_close"] - exit_["perp"]["source_close"]
        basis = {
            "entry_spot_minus_perp": initial,
            "exit_spot_minus_perp": final,
            "matched_base_quantity": entries[0]["quantity"],
            "basis_change_pnl": entries[0]["quantity"] * (final - initial),
        }
        if abs(basis["basis_change_pnl"] - gross_spot - gross_perp) > 1e-7:
            raise ValueError("Matched basis identity fails")
    paid = [e for e in events if e["type"] == "funding"]
    return {
        "spot_gross_price_pnl": gross_spot,
        "perp_gross_price_pnl": gross_perp,
        "matched_basis": basis,
        "funding": funding,
        "fees": fees,
        "cash_decomposition_slippage": slippage,
        "engine_reported_gross_quantity_slippage": engine_slippage,
        "base_fee_slippage_overlap": engine_slippage - slippage,
        "exit_reserve": reserve,
        "net_pnl": model,
        "reconciliation_error": model - report["metrics"]["net_pnl"],
        "paid_events": len(paid),
        "paid_rate_sum": math.fsum(e["rate"] for e in paid),
        "paid_event_ledger": paid,
        "cash_slippage_note": "Cash attribution uses net base quantities; spot buy fee valued at actual fill already includes fee-base slippage. Gross-quantity engine slippage is separate to avoid double counting.",
        "closed_pairs": report["metrics"]["closed_trades"],
        "open_pairs": len(state["positions"]),
    }


def worker(source, symbol, slip, mode, output, checkpoint=None):
    command = [
        sys.executable,
        "-m",
        "research_tools.carry_case_review.worker_buffer",
        "--source",
        str(source),
        "--symbol",
        symbol,
        "--slippage",
        str(slip),
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    if checkpoint:
        command += ["--checkpoint", str(checkpoint)]
    subprocess.run(command, check=True, timeout=120)


def run(source, review, output):
    for p in [source, review]:
        verify_manifest(p)
    checked = read(review / "verification.json")
    if not checked["passed"] or checked["source_completion_sha256"] != sha(
        source / "completion.json"
    ):
        raise ValueError("Raw source review missing or changed")
    plan = read(source / "plan.json")
    output.mkdir(parents=True, exist_ok=False)
    for name in ["cases", "restores", "cash"]:
        (output / name).mkdir()
    paths = [
        Path(__file__),
        Path(__file__).with_name("worker_buffer.py"),
        Path(__file__).with_name("account_audit.py"),
        Path(plan["engine"]),
        source / "completion.json",
        review / "completion.json",
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/factor_lab/study.py"),
    ]
    save(
        output / "registration.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "before_case_financial_results": True,
            "diagnostic_not_new_strategy": True,
            "new_financial_strategy_accounts": 0,
            "maximum_case_paths": 4,
            "cash_paths": 2,
        },
    )
    save(output / "plan.json", plan)
    records = []
    restarts = []
    for case in plan["cases"]:
        symbol = case["symbol"]
        cash = output / "cash" / symbol
        worker(source, symbol, 10, "cash", cash)
        cash_audit = audit_account(cash, *inputs(source, symbol))
        if read(cash / "result.json")["metrics"]["net_return"] != 0:
            raise ValueError("Cash baseline changed")
        save(cash / "audit.json", cash_audit)
        for slip in plan["slippage_bps_each_leg_each_side"]:
            name = symbol + "_" + str(slip)
            folder = output / "cases" / name
            worker(source, symbol, slip, "full", folder)
            baseline = Path(plan["buffer_counterfactual"]["base_run"]) / "cases" / name
            before = pd.read_parquet(baseline / "equity_5m.parquet")
            current = pd.read_parquet(folder / "equity_5m.parquet")
            prefix = current.loc[
                current.date <= pd.Timestamp(case["original_case_deadline"])
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(prefix, before, check_exact=True)
            save(
                folder / "baseline_prefix_audit.json",
                {
                    "passed": True,
                    "nav_points": len(prefix),
                    "baseline_result_sha256": sha(baseline / "result.json"),
                    "entry_and_exit_submission_unchanged": True,
                },
            )
            audit = audit_account(folder, *inputs(source, symbol))
            save(folder / "audit.json", audit)
            effects = attribution(folder, source, symbol)
            save(folder / "attribution.json", effects)
            first = output / "restores" / (name + "_first")
            second = output / "restores" / (name + "_restored")
            worker(source, symbol, slip, "first", first)
            checkpoint = first / "state.json"
            checkpoint_sha = sha(checkpoint)
            if not read(checkpoint)["positions"]:
                raise ValueError("Restore case must have real open inventory")
            worker(source, symbol, slip, "restore", second, checkpoint)
            if sha(checkpoint) != checkpoint_sha:
                raise ValueError("Disk checkpoint changed")
            combined = pd.concat(
                [
                    pd.read_parquet(first / "equity_5m.parquet"),
                    pd.read_parquet(second / "equity_5m.parquet").iloc[1:],
                ],
                ignore_index=True,
            )
            pd.testing.assert_frame_equal(
                combined, pd.read_parquet(folder / "equity_5m.parquet"), check_exact=True
            )
            for key in ["events", "trades", "requests"]:
                if read(first / (key + ".json")) + read(second / (key + ".json")) != read(
                    folder / (key + ".json")
                ):
                    raise ValueError("Actual restored case changed " + key)
            if read(second / "state.json") != read(folder / "state.json"):
                raise ValueError("Actual restored final state differs")
            pids = [read(p / "invocation.json")["pid"] for p in [first, second]]
            if pids[0] == pids[1]:
                raise ValueError("Restore reused same interpreter")
            restarts.append(
                {
                    "case": name,
                    "first_pid": pids[0],
                    "restored_pid": pids[1],
                    "checkpoint_sha256": checkpoint_sha,
                    "open_pairs_at_split": len(read(checkpoint)["positions"]),
                    "passed": True,
                }
            )
            report = read(folder / "result.json")
            records.append(
                {
                    "symbol": symbol,
                    "slippage_bps": slip,
                    "case_start": case["decision"],
                    "exit_deadline": case["exit_deadline"],
                    "metrics": report["metrics"],
                    "risk_halted": report["risk_halted"],
                    "flags": report["flags"],
                    "capacity_proxy": report["capacity"],
                    "attribution": effects,
                    "internal_arithmetic_audit": audit,
                    "independent_third_party_review": False,
                    "strategy_admitted": False,
                }
            )
    save(
        output / "summary.json",
        {
            "schema": "carry-exit-selected-cases.buffer.v1",
            "case_paths": 4,
            "cash_paths": 2,
            "financial_strategy_accounts": 0,
            "case_records": records,
            "process_restores": restarts,
            "external_third_party_review": False,
            "customer_cashflow_certified": False,
            "independent_oos": False,
            "profit_goal_met": False,
            "portfolio_actual": None,
            "production_eligible": False,
            "orders_enabled": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "profit_goal_met": False,
            "financial_strategy_accounts": 0,
        },
    )
    print(
        {
            "case_paths": len(records),
            "cash_paths": 2,
            "actual_nonflat_process_restores": len(restarts),
            "financial_strategy_accounts": 0,
            "net_pnl": [(r["symbol"], r["slippage_bps"], r["metrics"]["net_pnl"]) for r in records],
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.review, args.output)
