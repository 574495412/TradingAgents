"""Read existing source results only; never build a portfolio or admit a module."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = "b98e4b9"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def read(relative):
        path = ROOT / relative
        inputs[os.path.relpath(path, ROOT)] = sha(path)
        return json.loads(path.read_text())

    target_bytes = subprocess.check_output(
        ["git", "show", BASE + ":tradingagents/research/equities/objectives.json"], cwd=ROOT
    )
    target = json.loads(target_bytes)
    limit = target["targets"][1]["limit"]
    assert limit == 0.15 and target["targets"][0]["challenge"] == 0.5
    stocks = read("results/research/equities_pit_stocks_20261026_v002/summary.json")
    sparse = read("results/research/sparse_family_accounts_20261026_v002/summary.json")
    carry_root = "results/research/spot_carry_accounts_20261026_v002/"
    carry = read(carry_root + "summary.json")
    completion = read(carry_root + "completion.json")
    carry_plan = read(carry_root + "plan.json")
    assert completion["summary_sha256"] == inputs[carry_root + "summary.json"]
    assert completion["plan_sha256"] == inputs[carry_root + "plan.json"]
    selection = read(carry_root + "capacity_selection.json")
    legacy = read(carry_root + "legacy_evidence.json")
    original_legacy = read(carry_plan["legacy_hold_train"])
    original_scope = read(carry_plan["legacy_hold_scope"])
    assert original_legacy == legacy["training"] and original_scope == legacy["scope"]
    horizon = read("results/research/matched_carry_horizon_inputs_20261026_v001/summary.json")
    task = read("data/research/equity_portfolio_readiness_20261027_v001/carry-task-status.json")
    stock_rows = [
        {
            "name": r["name"],
            "currency": "USD",
            "diagnostic_cagr": r["metrics"]["annualized_return"],
            "close_drawdown": r["metrics"]["close_max_drawdown"],
            "within_drawdown_limit": r["metrics"]["close_max_drawdown"] <= limit,
        }
        for r in stocks["records"]
    ]
    assert len(stock_rows) == 8 and not any(r["within_drawdown_limit"] for r in stock_rows)
    variants = carry["module_assessments"]
    assert len(variants) == 2 and not any(v["eligible_for_portfolio_test"] for v in variants)
    assert carry["mechanism_count"] == 1 and len(carry["account_rows"]) == 20
    noncash = [r for r in sparse["account_rows"] if r["candidate"] != "cash"]
    assert len(noncash) == 48 and all(r["metrics"]["net_pnl"] < 0 for r in noncash)
    with (
        ROOT / "results/research/equity_depth_attribution_20261026_v002/paired-daily-returns.csv"
    ).open(newline="") as handle:
        dates = {r["date"] for r in csv.DictReader(handle)}
    inputs["results/research/equity_depth_attribution_20261026_v002/paired-daily-returns.csv"] = (
        sha(
            ROOT
            / "results/research/equity_depth_attribution_20261026_v002/paired-daily-returns.csv"
        )
    )
    overlap_current = {
        d for d in dates if any(r["start"][:10] <= d < r["end"][:10] for r in carry["account_rows"])
    }
    overlap_legacy = {
        d for d in dates if any(r["start"][:10] <= d < r["end"][:10] for r in original_legacy)
    }
    assert not overlap_current and overlap_legacy
    live = []
    for line in subprocess.check_output(["ps", "-axo", "pid=,comm=,args="], text=True).splitlines():
        parts = line.strip().split(None, 2)
        if (
            len(parts) == 3
            and "python" in Path(parts[1]).name.lower()
            and any(
                t in parts[2]
                for t in ("research_tools.carry_case_review", "carry_exit_cases_20261026")
            )
        ):
            live.append({"pid": int(parts[0]), "command": parts[2]})
    continuous = [
        {
            "id": r["id"],
            "capital_usdt": r["capital"],
            "slippage_bps": r["slippage_bps"],
            "diagnostic_cagr": r["metrics"]["cagr"],
            "matched_closed_pairs": r["metrics"]["matched_closed_pairs"],
        }
        for r in carry["account_rows"]
        if r["phase"] == "continuous2025_2026" and r["slippage_bps"] == 10
    ]
    result = {
        "schema": "equity-crypto-portfolio-readiness.v1",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "stock_accounts": stock_rows,
        "minimum_stock_close_drawdown": min(r["close_drawdown"] for r in stock_rows),
        "stock_scope": "Frozen internal diagnostics with unresolved input conflicts, not independently certified performance. No extension of the same unchanged continuous NAV can erase an existing drawdown.",
        "sparse_crypto": {
            "noncash_accounts": len(noncash),
            "negative_noncash_accounts": len(noncash),
            "cash_controls": len(sparse["account_rows"]) - len(noncash),
            "capacity_followup_queue": sparse["capacity_followup_queue"],
        },
        "matched_crypto": {
            "preexisting_accounts": len(carry["account_rows"]),
            "positive_internal_account_rows": sum(
                r["metrics"]["net_pnl"] > 0 for r in carry["account_rows"]
            ),
            "mechanisms": carry["mechanism_count"],
            "variant_assessments": variants,
            "continuous_high_cost_diagnostics": continuous,
            "capacity_followup_role": selection["role"],
            "internal_audit_is_external": False,
        },
        "legacy_crypto": {
            "preexisting_training_rows": len(original_legacy),
            "original_and_mirrored_records_match": True,
            "scope": original_scope["status"],
            "scope_reason": original_scope["reason"],
            "quantity_precision": original_legacy[0]["quantity_precision"],
            "independent_OOS": False,
        },
        "calendar_alignment": {
            "stock_return_dates": len(dates),
            "current_20_crypto_overlap_dates": len(overlap_current),
            "legacy_2024_crypto_calendar_overlap_dates": len(overlap_legacy),
            "legacy_overlap_range": [min(overlap_legacy), max(overlap_legacy)],
            "calendar_overlap_is_admissible_synchronized_PnL": False,
            "reason": "Legacy calendar overlap exists. Its archived economic screen is a different account model and scope; it is not a verified net series in the segregated-wallet model or an admitted module.",
        },
        "known_exit_horizon": {
            "old_primary_data_only_slots": horizon["primary_old_guarded_slots"],
            "horizon_guarded_primary_slots": horizon["primary_horizon_guarded_slots"],
            "financial_accounts_in_horizon_review": horizon["financial_accounts"],
        },
        "case_task_observation": {
            **task,
            "matching_live_python_processes": live,
            "observation_role": "Operational status only; no live-process wait or completed new financial case is claimed.",
        },
        "eligible_stock_modules": 0,
        "eligible_reviewed_crypto_modules": 0,
        "USD_USDT_FX_conversion_verified": False,
        "common_capital_account_executed": False,
        "required_shared_capital_evidence": [
            "independently admitted modules",
            "synchronized genuine joint net PnL and original valuation clocks",
            "USD/USDT conversion and transfer/settlement eligibility",
            "single cash/margin/receivable ledger and continuous high-water mark",
            "external independent reconstruction and genuinely unseen forward account receipts",
        ],
        "new_economic_accounts": 0,
        "current_program_executed_accounts": target["budget"]["executed_new_accounts"],
        "portfolio_target_actual": None,
        "profit_goal_met": False,
        "external_independent_attestation": False,
        "verdict": "No portfolio run admitted. Positive local carry diagnostics do not change module qualification; calendar overlap alone does not authorize return stitching or shared capital.",
    }
    write(output / "evidence.json", result)
    write(
        output / "registration.json",
        {
            "inputs_sha256": inputs,
            "analysis_sha256": sha(Path(__file__)),
            "target_source_commit": BASE,
            "target_blob_sha256": hashlib.sha256(target_bytes).hexdigest(),
            "own_source_checks_are_external_attestation": False,
        },
    )
    (output / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "stocks_passing_close_drawdown": 0,
                "carry_accounts": 20,
                "eligible_carry_variants": 0,
                "current_crypto_overlap": len(overlap_current),
                "legacy_calendar_overlap": len(overlap_legacy),
                "new_accounts": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
