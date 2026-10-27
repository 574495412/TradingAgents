"""Seal completed reviewed conclusions without touching any financial account."""

import argparse
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def seal(output):
    assert not (output / "completion.json").exists()
    v = read(output / "verification.json")
    assert v["passed"] and not v["profit_goal_met"]
    for filename, digest in v["source_hashes"].items():
        assert sha(Path(filename)) == digest, filename
    finalized = Path("results/research/monthly_hold_finalization_20261027_v001")
    verify_manifest(finalized)
    summary = read(finalized / "summary.json")
    assert summary["actual_new_policy_accounts"] == 4 and not summary["capacity_triggered"]
    assert read(output / "restart_verification.json")[
        "exact_full_NAV_cash_ledger_requests_inventory_and_risk"
    ]
    assert (
        not read(output / "exposure_summation_audit.json")["maximum_fraction_of_derived_sum_bound"]
        > 1
    )
    save(
        output / "seal_verification.json",
        {
            "passed": True,
            "source_sha256": sha(Path(__file__)),
            "known_history_scope_only": True,
            "cash_and_exposure_precision_claims_separate": True,
            "all_economic_accounts_negative": True,
            "capacity_not_triggered": True,
        },
    )
    manifest = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    assert "completion.json" not in manifest
    save(
        output / "completion.json", {"passed": True, "manifest": manifest, "profit_goal_met": False}
    )
    print({"passed": True, "report": str(output), "files": len(manifest)}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    seal(p.parse_args().output)
