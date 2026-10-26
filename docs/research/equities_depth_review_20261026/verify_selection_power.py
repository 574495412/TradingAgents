"""Verify this local review and canonical invariants; not external attestation."""

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/research/equity_selection_power_20261027_v002"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


reg = read(RUN / "registration.json")
lineage = read(RUN / "lineage.json")
power = read(RUN / "power-scenarios.json")
rows = read(RUN / "observed-records.json")
checks = {
    "all_registered_inputs_unchanged": all(
        sha(ROOT / k) == v for k, v in reg["inputs_sha256"].items()
    ),
    "analysis_snapshot_matches": sha(RUN / "analysis_source.py")
    == reg["analysis_sha256"]
    == sha(ROOT / "tradingagents/research/equities/selection_power_review.py"),
    "plan_snapshot_matches": sha(RUN / "plan.json")
    == reg["plan_sha256"]
    == sha(ROOT / "docs/research/equities_depth_review_20261026/selection-power-plan.json"),
    "observed_record_accounting": len(rows) == lineage["observed_completed_account_records"] == 180
    and lineage["reused_prior_program_account_records"] == 164
    and lineage["current_program_executed_accounts"] == 16,
    "cash_and_duplicates_counted": sum(r["policy"] == "cash" for r in rows) == 26
    and len({r["exact_normalized_curve_sha256"] for r in rows}) == 160
    and sum(len(g) - 1 for g in lineage["exact_equal_curve_groups"]) == 20,
    "all_forward_claims_false": all(not r["forward_OOS_claim"] for r in rows),
    "DSR_PBO_independent_count_unknown": all(
        lineage[k] is None for k in ("DSR", "PBO", "independent_trial_count")
    ),
    "method_cleanup_did_not_change_power": power["scenarios"]
    == read(ROOT / "results/research/equity_selection_power_20261027_v001/power-scenarios.json")[
        "scenarios"
    ],
    "integer_power_boundary": all(
        s["power_at_n_minus_1"] < 0.8 <= s["power_at_n"] for s in power["scenarios"]
    ),
    "mixture_integrals_match": all(
        abs(s["quadrature_crosscheck"]["null_tail"] - 0.05) < 2e-9
        and abs(s["quadrature_crosscheck"]["alternative_tail"] - s["power_at_n"]) < 2e-9
        for s in power["scenarios"]
    ),
}
z = NormalDist()
checks["standard_library_z_approximation"] = all(
    abs(
        (
            (z.inv_cdf(0.95) + z.inv_cdf(0.8))
            / (s["alternative_annual_sharpe"] - s["null_annual_sharpe"])
        )
        ** 2
        - s["large_n_z_approx_years"]
    )
    < 1e-12
    for s in power["scenarios"]
)
source = ROOT / "data/research/equity_selection_power_20261027_v001/scipy-nct.htm"
receipt = read(source.with_suffix(".htm.receipt.json"))
checks["official_document_capture_bound"] = receipt["http_status"] == 200 and receipt[
    "sha256"
] == sha(source)
a = "tradingagents/research/equities/objectives.json"
b = "docs/research/equities_depth_acceptance_20261026/acceptance.json"
x, y = read(ROOT / a), read(ROOT / b)


def before(path):
    return json.loads(subprocess.check_output(["git", "show", "806a97b:" + path], cwd=ROOT))


ox, oy = before(a), before(b)
checks["targets_and_budget_unchanged"] = all(
    x[k] == ox[k]
    for k in (
        "objective",
        "targets",
        "levels",
        "budget",
        "production_eligible",
        "broker_verified",
        "independent_forward_oos",
        "profit_goal_status",
    )
)
checks["all_prior_policies_unchanged"] = all(
    y[k] == oy[k]
    for k in (
        "evaluation_levels",
        "decision_rules",
        "focal_scope",
        "source_clock_policy",
        "price_source_conflict_policy",
        "market_data_semantics_policy",
    )
)
for label, now, prior in (
    ("adapter", x["acceptance_gates"], ox["acceptance_gates"]),
    ("owner", y["gates"], oy["gates"]),
):
    checks[label + "_gates_unchanged"] = all(
        {k: v for k, v in n.items() if k != "current_evidence_note"}
        == {k: v for k, v in o.items() if k != "current_evidence_note"}
        for n, o in zip(now, prior, strict=True)
    )
r = read(ROOT / "results/research/equities_pit_stocks_20261026_v002/registration.json")
pit = Path(r["pit_dataset"])
checks["frozen_metadata_unchanged"] = all(sha(pit / k) == v for k, v in r["input_hashes"].items())
checks["frozen_sources_unchanged"] = all(sha(ROOT / k) == v for k, v in r["source_hashes"].items())
pm = read(pit / "prices/manifest.json")
checks["frozen_20_prices_and_signals"] = all(
    sha(pit / "prices" / m[f]) == m[h]
    for m in pm["symbols"].values()
    for f, h in (("file", "sha256"), ("signal_file", "signal_sha256"))
)
mr = read(ROOT / "results/research/equity_exas_market_semantics_20261027_v003/registration.json")
checks["frozen_8_states"] = all(
    sha(ROOT / k) == v for k, v in mr["inputs_sha256"].items() if k.endswith("/state.json")
)
with urllib.request.urlopen(
    "http://127.0.0.1:8791/api/research/runs?refresh=1", timeout=15
) as response:
    program = json.load(response)["equity_program"]
checks["live_adapter_current"] = program["version"] == program["acceptance_version"] == x[
    "version"
] == y["version"] == "20261027_depth_v010" and program["acceptance_sha256"] == sha(ROOT / b)
assert all(checks.values()), [k for k, value in checks.items() if not value]
result = {
    "all_internal_checks_passed": True,
    "checks": checks,
    "registered_input_files": len(reg["inputs_sha256"]),
    "objective_sha256": sha(ROOT / a),
    "acceptance_sha256": sha(ROOT / b),
    "version": x["version"],
    "external_attestation_received": False,
    "actual_admission_power_or_horizon": None,
    "scope": "Observed artifacts and ideal-model internal consistency only; not complete search history, financial evidence, or external review.",
    "repair_note": "Initial ad-hoc source receipt check removed only the .json suffix and failed path lookup; corrected to the captured HTML path, without changing inputs.",
}
Path(__file__).with_name("selection-power-verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "checks_passed": True,
            "version": x["version"],
            "registered_inputs": len(reg["inputs_sha256"]),
            "new_accounts": 0,
            "actual_power": None,
        }
    )
)
