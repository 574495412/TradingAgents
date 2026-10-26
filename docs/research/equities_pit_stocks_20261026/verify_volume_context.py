"""Verify existing corrected artifacts only; no account replay or external audit."""

import hashlib
import importlib.util
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/research/equities_pit_stocks_context_20261027_v001"
PARENT = ROOT / "results/research/equities_pit_stocks_20261026_v002"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


completion = read(RUN / "completion.json")
assert all(sha(RUN / p) == h for p, h in completion["manifest"].items())
registration = read(RUN / "registration.json")
assert all(sha(ROOT / p) == h for p, h in registration["inputs_sha256"].items())
assert (
    sha(RUN / "analysis_source.py")
    == registration["analysis_sha256"]
    == sha(ROOT / "tradingagents/research/equities/pit_context_correction.py")
)
assert sha(RUN / "pre-correction-objectives.json") == registration["pre_correction_goal_sha256"]
warmup = read(RUN / "warmup-proof.json")
assert len(warmup["old_zero_cap_days"]) == 19
assert (
    warmup["old_zero_cap_days"][0] == "2023-01-03"
    and warmup["old_zero_cap_days"][-1] == "2023-01-30"
)
assert warmup["context_start"] == "2022-12-02"
assert all(
    x["known_observations_before_first_fill"] == 20
    and x["old_prior_mean_was_missing"]
    and x["known_mean_volume_on_prior_day"] > 0
    for x in warmup["symbols"].values()
)
summary = read(RUN / "summary.json")
comparisons = read(RUN / "comparisons.json")
audits = read(RUN / "internal-ledger-audits.json")
restart = read(RUN / "fresh-process-restart.json")
assert len(summary["records"]) == summary["actual_financial_replays"] == 8
assert audits["passed"] and len(audits["accounts"]) == 8
assert restart["passed"] and len(restart["accounts"]) == 8
for row in restart["accounts"]:
    assert row["exact_state_matches"] and row["state_sha256"] == row["resumed_sha256"]
    assert (RUN / row["name"] / "state.json").read_bytes() == (
        RUN / row["name"] / "resumed-state.json"
    ).read_bytes()
mom_unchanged = []
for row in comparisons:
    old, new = read(PARENT / row["name"] / "state.json"), read(RUN / row["name"] / "state.json")
    assert all(f["day"] >= "2023-01-03" for f in new["fills"])
    assert len(new["equity_curve"]) == len(old["equity_curve"]) == 373
    assert new["equity_curve"][0]["equity"] == old["equity_curve"][0]["equity"] == 10000
    if row["policy"] == "french_mom_12_2":
        assert old["equity_curve"] == new["equity_curve"] and old["fills"] == new["fills"]
        mom_unchanged.append(row["name"])
    else:
        assert row["old_first_fill"] == "2023-01-31" and row["corrected_first_fill"] == "2023-01-03"
assert len(mom_unchanged) == 2
assert all(r["metrics"]["close_max_drawdown"] > 0.15 for r in summary["records"])

# Reproduce the inherited contrast, without creating any new financial account.
path = ROOT / "tradingagents/research/equities/mechanism_attribution.py"
spec = importlib.util.spec_from_file_location("frozen_descriptive_methods", path)
methods = importlib.util.module_from_spec(spec)
spec.loader.exec_module(methods)
contrast_run = ROOT / "results/research/equity_context_contrast_20261027_v001"
contrast = read(contrast_run / "contrast.json")
contrast_reg = read(contrast_run / "registration.json")
assert all(sha(ROOT / p) == h for p, h in contrast_reg["inputs_sha256"].items())
for label, folder in (("parent", PARENT), ("corrected", RUN)):
    vectors = {}
    for policy in ("french_mom_12_2", "disclosed_equal"):
        curve = read(folder / f"US_{policy}_10000_15bps/state.json")["equity_curve"]
        vectors[policy] = np.array(
            [b["equity"] / a["equity"] - 1 for a, b in zip(curve[:-1], curve[1:], strict=True)]
        )
    actual = methods.paired_bootstrap(vectors["french_mom_12_2"] - vectors["disclosed_equal"])
    assert actual == contrast[label]["paired_difference_bootstrap"]
base_csv = pd.read_csv(
    ROOT / "results/research/equity_depth_attribution_20261026_v002/paired-daily-returns.csv",
    index_col="date",
)
fixed_factors = base_csv[["SPY_gross_total_return", "QQQ_minus_SPY_gross_return"]]
delta = vectors["french_mom_12_2"] - vectors["disclosed_equal"]
assert (
    methods.regression(delta, fixed_factors)
    == contrast["corrected"]["paired_difference_regression"]
)

a = "tradingagents/research/equities/objectives.json"
b = "docs/research/equities_depth_acceptance_20261026/acceptance.json"
current, standard = read(ROOT / a), read(ROOT / b)
old = json.loads(subprocess.check_output(["git", "show", "eeabded:" + a], cwd=ROOT))
old_standard = json.loads(subprocess.check_output(["git", "show", "eeabded:" + b], cwd=ROOT))
assert all(
    current[k] == old[k]
    for k in (
        "objective",
        "scope",
        "targets",
        "levels",
        "production_eligible",
        "broker_verified",
        "independent_forward_oos",
        "profit_goal_status",
    )
)
assert current["budget"]["new_accounts_max"] == old["budget"]["new_accounts_max"] == 24
assert (
    current["budget"]["executed_new_accounts"] == 24
    and current["budget"]["executed_stock_accounts"] == 16
)
assert current["budget"]["executed_implementation_repair_accounts"] == 8
assert (
    current["budget"]["remaining_capacity_accounts"]
    == current["budget"]["remaining_implementation_repair_accounts"]
    == 0
)
before_replay = read(RUN / "pre-correction-objectives.json")
assert (
    before_replay["budget"]["executed_new_accounts"] == 16
    and before_replay["budget"]["remaining_implementation_repair_accounts"] == 8
)
for now, prior in (
    (standard["gates"], old_standard["gates"]),
    (current["acceptance_gates"], old["acceptance_gates"]),
):
    assert all(
        {k: v for k, v in n.items() if k != "current_evidence_note"}
        == {k: v for k, v in o.items() if k != "current_evidence_note"}
        for n, o in zip(now, prior, strict=True)
    )
with urllib.request.urlopen(
    "http://127.0.0.1:8791/api/research/runs?refresh=1", timeout=15
) as response:
    payload = json.load(response)
program = payload["equity_program"]
assert (
    program["version"]
    == program["acceptance_version"]
    == current["version"]
    == standard["version"]
    == "20261027_depth_v012"
)
assert program["acceptance_sha256"] == sha(ROOT / b)
assert program["evidence"]["pit_stock_run"] == str(RUN.relative_to(ROOT))
assert (
    program["research_progress"]["high_cost_accounts"]
    == current["research_progress"]["high_cost_accounts"]
)
assert program["budget"]["executed_new_accounts"] == 24
result = {
    "all_internal_checks_passed": True,
    "source_context_days": 20,
    "old_zero_cap_days": 19,
    "completed_correction_accounts": 8,
    "completed_program_accounts": 24,
    "new_strategy_hypotheses": 0,
    "momentum_paths_exactly_unchanged": 2,
    "Decimal_ledger_max_error": max(
        float(r["max_absolute_balance_error"]) for r in audits["accounts"]
    ),
    "fresh_process_recoveries": 8,
    "parent_files_unchanged": True,
    "corrected_contrast_reproduced": True,
    "budget_reallocation_registered_before_corrected_returns": True,
    "financial_targets_and_gate_statuses_unchanged": True,
    "current_adapter_version": program["version"],
    "objective_sha256": sha(ROOT / a),
    "acceptance_sha256": sha(ROOT / b),
    "profit_goal_met": False,
    "external_independent_attestation": False,
    "scope": "Implementation context, recorded hypothetical economics and internal consistency only; original source conflicts and G7/G8 are unresolved.",
}
Path(__file__).with_name("volume-context-verification.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(result))
