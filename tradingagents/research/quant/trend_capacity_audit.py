"""Audit a capital-only extension, anchored to the unchanged primary selection."""

import argparse
from pathlib import Path

from trend_mainline_audit import audit_account, checked, read, save, sha


def audit(source, primary_audit, output):
    output.mkdir(parents=True, exist_ok=False)
    plan, summary = read(source / "plan.json"), read(source / "evaluation.json")
    primary = Path(plan["primary_run"])
    if sha(primary / "evaluation.json") != plan["primary_evaluation_sha256"]:
        raise ValueError("Primary evidence changed")
    receipt = read(primary_audit / "audit.json")
    if (
        receipt["passed"] is not True
        or receipt["evaluation_sha256"] != plan["primary_evaluation_sha256"]
    ):
        raise ValueError("Primary independent audit missing")
    if sha(source / "selection.json") != sha(primary / "selection.json"):
        raise ValueError("Capital extension changed the primary selection")
    if sha(source / "plan.json") != summary["plan_sha256"]:
        raise ValueError("Capacity protocol changed")
    expected = {
        (c, n, b)
        for c in plan["candidates"] + ["LongOnlyControl"]
        for n in [1000000, 10000000]
        for b in plan["extra_cost_bps_per_side"]
    }
    actual = [
        (r["mechanism"], r["initial_equity_usdt"], r["extra_cost_bps_per_side"])
        for r in summary["accounts"]
    ]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Missing or duplicate capacity scenarios")
    for name, checksum in read(source / "data/receipts.json").items():
        checked(source / "data", name, checksum)
    if sha(source / "data/receipts.json") != sha(primary / "data/receipts.json"):
        raise ValueError("Stress market population changed")
    rows = [audit_account(source, r, plan) for r in summary["accounts"]]
    result = {
        "passed": True,
        "account_count": len(rows),
        "accounts": rows,
        "primary_selection": receipt["selected_candidate"],
        "primary_selection_unchanged": True,
        "evaluation_sha256": sha(source / "evaluation.json"),
        "primary_audit_sha256": sha(primary_audit / "audit.json"),
        "auditor_sha256": sha(Path(__file__)),
        "shared_auditor_sha256": sha(Path(__file__).with_name("trend_mainline_audit.py")),
        "scope": "model-constrained capacity stress only; no real order-book certification",
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "audit.json", result)
    for name in ("trend_mainline_audit.py", "trend_capacity_audit.py"):
        (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    save(output / "receipts.json", {p.name: sha(p) for p in output.iterdir()})
    return {"passed": True, "accounts": len(rows)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--primary-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(audit(a.source.resolve(), a.primary_audit.resolve(), a.output.resolve()))
