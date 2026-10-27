"""One frozen lot/fee/dust diagnosis of the old caps; no economic account replay."""

import argparse
from pathlib import Path

from factors.common.lot_inventory import bounded_cover, exact_pair_lattice
from research_tools.dated_carry.audit_lot_inventory import audit
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("lot_inventory_plan.json")
    plan = read(source)
    original = Path(plan["original_accounts"])
    for parent in [*plan["parents"], str(original)]:
        verify_manifest(Path(parent))
    paths = [
        source,
        Path(__file__),
        Path("factors/common/lot_inventory.py"),
        Path("research_tools/dated_carry/audit_lot_inventory.py"),
    ]
    paths += [Path(parent) / "completion.json" for parent in plan["parents"]]
    paths += [original / "completion.json"]
    save(output / "plan.json", {**plan, "source_hashes": {str(p): sha(p) for p in paths}})
    save(
        output / "registration.json",
        {
            "before_new_inventory_case_calculations": True,
            "plan_sha256": sha(output / "plan.json"),
            "new_financial_accounts": 0,
            "historical_order_filters_certified": False,
        },
    )
    events = read(original / "accounts" / plan["original_reference_policy"] / "events.json")
    cases = []
    for symbol in plan["cases"]:
        spot = next(
            e
            for e in events
            if e["type"] == "entry" and e["market"] == "spot" and e["symbol"] == symbol
        )
        future = next(
            e
            for e in events
            if e["type"] == "entry" and e["market"] == "perp" and e["symbol"] == symbol
        )
        inputs = {
            "spot_step": plan["spot_step_assumptions"][symbol],
            "future_base_step": plan["future_base_step_assumption"],
            "base_fee_rate": plan["base_fee_rate"],
            "gross_spot_cap": str(spot["gross_quantity"]),
            "short_base_cap": str(future["quantity"]),
        }
        cases.append(
            {
                "symbol": symbol,
                "inputs": inputs,
                "original_spot_entry_event": spot,
                "original_future_entry_event": future,
                "minimum_exact_pair": exact_pair_lattice(
                    **{k: inputs[k] for k in ["spot_step", "future_base_step", "base_fee_rate"]}
                ),
                "bounded_cover": bounded_cover(**inputs),
                "historical_order_rules_certified": False,
                "new_financial_account": False,
            }
        )
    save(output / "cases.json", cases)
    proofs = [{"symbol": case["symbol"], **audit(case)} for case in cases]
    save(output / "integer_audit.json", proofs)
    save(
        output / "summary.json",
        {
            "cases": len(cases),
            "integer_audits_passed": all(p["passed"] for p in proofs),
            "all_residual_inventories_retained": True,
            "new_financial_accounts": 0,
            "actual_historical_filters_certified": False,
            "actual_execution_certified": False,
            "module_admitted": False,
            "profit_goal_met": False,
            "orders_enabled": False,
        },
    )
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print({"cases": cases, "audits": proofs}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
