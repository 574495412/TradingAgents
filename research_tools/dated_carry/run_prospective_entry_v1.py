"""Freeze before acquisition; evaluate only later raw receipts, with no orders."""

import argparse
import importlib.util
import json
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from research_tools.dated_carry.audit_prospective_probe import reconstruct
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest

PLAN = Path("research_tools/dated_carry/prospective_entry_plan_v001.json")


def serial(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def register(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read(PLAN)
    paths = [
        PLAN,
        Path(__file__),
        Path(plan["engine"]),
        Path(plan["source_plan"]),
        Path("research_tools/dated_carry/collect_prospective_entry_sources_v1.py"),
        Path("research_tools/dated_carry/audit_prospective_probe.py"),
        Path("research_tools/dated_carry/audit_depth_witness.py"),
    ]
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "registered_wall_ns": time.time_ns(),
            "registered_at_utc": datetime.now(UTC).isoformat(),
            "source_hashes": {str(p): sha(p) for p in paths},
            "plan_sha256": sha(output / "plan.json"),
            "before_new_quotes": True,
            "prospective_preregistered_replay_not_real_time_broker_signals": True,
            "orders_enabled": False,
        },
    )
    print({"registered": str(output), "source_files": len(paths)}, flush=True)


def native_rows(source, market):
    rows = []
    for line in (source / (market + "_native.ndjson")).open():
        raw = json.loads(line)
        d = json.loads(raw["native_raw"])
        rows.append(
            {"received_wall_ns": raw["received_wall_ns"], "data": d["data"], "stream": d["stream"]}
        )
    return rows


def public_reference(source, asset, decision, limits):
    def body(stem):
        return read(source / (stem + ".receipt.json")), json.loads(
            (source / (stem + ".decoded")).read_bytes()
        )

    receipt, raw = body("spot_" + asset + "_reference")
    valid, reference = True, None
    if receipt["status"] == 200 and raw.get("symbol") == asset + "USDT":
        if raw.get("referencePrice") is not None:
            reference = Decimal(raw["referencePrice"])
            valid &= (
                abs(decision / 1e6 - raw["timestamp"])
                <= limits["maximum_reference_event_age_seconds"] * 1000
            )
    elif raw.get("code") != -2043:
        valid = False
    valid &= (
        0
        <= decision - receipt["received_wall_ns"]
        <= limits["maximum_reference_receipt_age_seconds"] * 1e9
    )
    ar, avg = body("spot_" + asset + "_average")
    valid &= ar["status"] == 200 and avg.get("mins") == 5
    valid &= (
        0
        <= decision - ar["received_wall_ns"]
        <= limits["maximum_reference_receipt_age_seconds"] * 1e9
    )
    if reference is None:
        valid &= (
            "price" in avg
            and abs(decision / 1e6 - avg.get("closeTime", 0))
            <= limits["maximum_reference_event_age_seconds"] * 1000
        )
    notional = reference if reference is not None else Decimal(avg.get("price", "0"))
    fr, mark = body("future_" + asset + "_mark")
    valid &= fr["status"] == 200 and mark.get("symbol", "").startswith(asset + "USDT_")
    valid &= (
        0
        <= decision - fr["received_wall_ns"]
        <= limits["maximum_reference_receipt_age_seconds"] * 1e9
    )
    valid &= (
        abs(decision / 1e6 - mark.get("time", 0))
        <= limits["maximum_reference_event_age_seconds"] * 1000
    )
    er, execution = body("spot_execution_rules")
    selected = [r for r in execution.get("symbolRules", []) if r["symbol"] == asset + "USDT"]
    valid &= er["status"] == 200 and len(selected) == 1
    valid &= (
        0
        <= decision - er["received_wall_ns"]
        <= limits["maximum_reference_receipt_age_seconds"] * 1e9
    )
    rules = selected[0]["rules"] if len(selected) == 1 else []
    valid &= all(r.get("ruleType") == "PRICE_RANGE" for r in rules)
    return {
        "public_inputs_valid": bool(
            valid and notional > 0 and Decimal(mark.get("markPrice", "0")) > 0
        ),
        "spot_notional_reference": str(notional),
        "spot_execution_reference": str(reference) if reference is not None else None,
        "future_notional_reference": mark.get("markPrice", "0"),
        "execution_rules": rules,
        "reference_absence_documented": receipt["status"] == 200
        and raw.get("referencePrice") is None
        or raw.get("code") == -2043,
    }


def evaluate(source, output, audit):
    if (output / "completion.json").exists():
        raise ValueError("Never overwrite a completed experiment")
    plan, registration = read(output / "plan.json"), read(output / "registration.json")
    assert plan == read(PLAN)
    for p, digest in registration["source_hashes"].items():
        assert sha(Path(p)) == digest
    verify_manifest(source)
    verify_manifest(audit)
    audited = read(audit / "summary.json")
    assert audited["source_observability_passed"]
    earliest = read(source / "registration.json")["registered_wall_ns"]
    assert registration["registered_wall_ns"] < earliest
    pairs = read(source / "selection.json")["symbols"]
    all_rows = {m: native_rows(source, m) for m in ["spot", "future"]}
    receipts = [
        read(source / f"{m}_{a}_snapshot.receipt.json")["received_wall_ns"]
        for m in all_rows
        for a in pairs
    ]
    decision = max(receipts) + int(plan["timing"]["decision_after_latest_snapshot_seconds"] * 1e9)
    clocks = {
        "decision": decision,
        "arrival": decision
        + int(plan["timing"]["independent_order_arrival_after_decision_seconds"] * 1e9),
        "unwind": decision + int(plan["timing"]["emergency_unwind_after_decision_seconds"] * 1e9),
        "mark": decision + int(plan["timing"]["last_mark_after_decision_seconds"] * 1e9),
    }
    observations, proofs = {}, []
    for asset, symbols in pairs.items():
        obs = {"symbols": symbols, **public_reference(source, asset, decision, plan["timing"])}
        assert (
            json.loads((source / f"future_{asset}_mark.decoded").read_bytes()).get("symbol")
            == symbols["future"]["symbol"]
        )
        expected_mins = {
            r["avgPriceMins"]
            for r in symbols["spot"]["filters"]
            if r["filterType"] in ["NOTIONAL", "PERCENT_PRICE_BY_SIDE"]
        }
        obs["public_inputs_valid"] &= expected_mins == {5}
        obs["book_valid"] = {}
        for phase, clock in clocks.items():
            obs[phase], obs[phase + "_ns"] = {}, clock
            obs["book_valid"][phase] = {}
            for market in ["spot", "future"]:
                assert clock < all_rows[market][-1]["received_wall_ns"]
                symbol = symbols[market]["symbol"].lower()
                rows = [
                    r
                    for r in all_rows[market]
                    if r["stream"].split("@")[0] == symbol and "@depth" in r["stream"]
                ]
                snapshot = json.loads((source / f"{market}_{asset}_snapshot.decoded").read_bytes())
                receipt = read(source / f"{market}_{asset}_snapshot.receipt.json")
                book, known = reconstruct(
                    rows, snapshot, receipt["received_wall_ns"], clock, market
                )
                age_ms = (clock - known) / 1e6
                obs["book_valid"][phase][market] = (
                    0 <= age_ms <= plan["timing"]["maximum_book_age_ms"]
                )
                obs[phase][market] = {
                    "bids": [[str(p), str(q)] for p, q in sorted(book.bids.items(), reverse=True)],
                    "asks": [[str(p), str(q)] for p, q in sorted(book.asks.items())],
                }
                proofs.append(
                    {
                        "asset": asset,
                        "market": market,
                        "phase": phase,
                        "clock_ns": clock,
                        "known_at_ns": known,
                        "book_age_ms": age_ms,
                        "update_id": book.last_id,
                    }
                )
        observations[asset] = obs
    save(output / "observations.json", serial(observations))
    save(output / "causal_books.json", proofs)
    spec = importlib.util.spec_from_file_location("frozen_dated_forward_paper", plan["engine"])
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    account = engine.run(observations, plan["account"])
    save(output / "account.json", serial(account))
    save(
        output / "summary.json",
        {
            "schema": "dated-carry-prospective-entry-result.v1",
            "strategy_registered_before_new_reference_and_quote_data": True,
            "prospective_preregistered_replay_not_real_time_signals": True,
            "new_conditional_paper_accounts": 1,
            "new_complete_carry_lifecycles": 0,
            "open_asset_states": {a: p["state"] for a, p in account["positions"].items()},
            "entry_screens": serial(account["decisions"]),
            "last_observation": serial(account["observations"][-1]),
            "win_rate": None,
            "full_lifecycle_return": None,
            "full_drawdown": None,
            "sharpe": None,
            "full_forward_strategy_validation_completed": False,
            "continuous_risk_monitoring": False,
            "orders_enabled": False,
            "third_party_review_performed": False,
            "capacity_certified": False,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "bounded_experiment_finished": True,
            "financial_lifecycle_finished": False,
            "source_completion_sha256": sha(source / "completion.json"),
            "audit_completion_sha256": sha(audit / "completion.json"),
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print(read(output / "summary.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["register", "evaluate"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--audit", type=Path)
    args = p.parse_args()
    if args.mode == "register":
        register(args.output)
    else:
        evaluate(args.source, args.output, args.audit)
