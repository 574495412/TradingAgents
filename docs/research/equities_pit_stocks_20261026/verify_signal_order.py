"""Verify the v013 evidence binding, without running financial accounts."""

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/research/equity_signal_order_contract_20261027_v002"
PARENT = ROOT / "results/research/equities_pit_stocks_context_20261027_v001"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original(path):
    return json.loads(subprocess.check_output(["git", "show", f"cb51968:{path}"], cwd=ROOT))


def gates_without_notes(gates):
    return [{k: v for k, v in g.items() if k != "current_evidence_note"} for g in gates]


def main():
    registration = read(RUN / "registration.json")
    for path, digest in registration["inputs_sha256"].items():
        assert sha(ROOT / path) == digest, path
    assert sha(RUN / "analysis_source.py") == registration["analysis_sha256"]
    assert (
        sha(ROOT / "tradingagents/research/equities/signal_order_audit.py")
        == registration["analysis_sha256"]
    )
    completion = read(PARENT / "completion.json")
    for path, digest in completion["manifest"].items():
        assert sha(PARENT / path) == digest, path
    result = read(RUN / "summary.json")
    assert [
        result[k]
        for k in (
            "monthly_decisions",
            "eligible_annual_inputs",
            "daily_weight_checks",
            "integer_request_checks",
            "recorded_fill_checks",
            "mismatches",
        )
    ] == [18, 121, 14920, 258, 228, 0]
    assert read(RUN / "mismatches.json") == []
    assert read(RUN / "integer-boundaries.json") == []
    accounts = read(RUN / "account-checks.json")
    assert len(accounts) == 8
    assert sum(r["fills_outside_first_session_after_month_end"] for r in accounts) == 96
    assert all(r["recorded_fills_checked"] == r["recorded_fill_count"] for r in accounts)
    for key in (
        "new_financial_accounts",
        "new_hypotheses",
        "external_attestation",
        "profit_goal_met",
    ):
        assert not result[key]
    doc = Path(__file__).parent
    for source, copy in (
        ("summary.json", "signal-order-summary.json"),
        ("account-checks.json", "signal-order-account-checks.json"),
        ("registration.json", "signal-order-registration.json"),
    ):
        assert sha(RUN / source) == sha(doc / copy)
    op = "tradingagents/research/equities/objectives.json"
    ap = "docs/research/equities_depth_acceptance_20261026/acceptance.json"
    objective, standard = read(ROOT / op), read(ROOT / ap)
    old_objective, old_standard = original(op), original(ap)
    for key in old_objective:
        if key not in ("version", "evidence", "acceptance_gates"):
            assert objective[key] == old_objective[key], key
    for key in old_standard:
        if key not in ("version", "progress_evidence", "gates"):
            assert standard[key] == old_standard[key], key
    for current, old, evidence, gates in (
        (objective, old_objective, "evidence", "acceptance_gates"),
        (standard, old_standard, "progress_evidence", "gates"),
    ):
        assert current["version"] == "20261027_depth_v013"
        assert {
            k: v for k, v in current[evidence].items() if k != "signal_order_contract_audit"
        } == old[evidence]
        assert gates_without_notes(current[gates]) == gates_without_notes(old[gates])
    assert len(objective["acceptance_gates"]) == len(standard["gates"])
    for compact, full in zip(objective["acceptance_gates"], standard["gates"], strict=True):
        assert all(full[key] == value for key, value in compact.items())
    assert all(t["actual"] is None for t in objective["targets"])
    assert objective["budget"]["executed_new_accounts"] == 24
    assert objective["budget"]["remaining_capacity_accounts"] == 0
    with urllib.request.urlopen(
        "http://127.0.0.1:8791/api/research/runs?refresh=1", timeout=30
    ) as response:
        program = json.load(response)["equity_program"]
    assert program["version"] == program["acceptance_version"] == objective["version"]
    assert program["acceptance_sha256"] == sha(ROOT / ap)
    assert program["sha256"] == sha(ROOT / op)
    assert program["targets"] == objective["targets"]
    assert program["budget"] == objective["budget"]
    receipt = {
        "schema": "equity-signal-order-evidence-verification.v1",
        "version": objective["version"],
        "registered_input_files_checked": len(registration["inputs_sha256"]),
        "parent_completion_files_checked": len(completion["manifest"]),
        "audit_output_sha256": {p.name: sha(p) for p in RUN.glob("*.json")},
        "objective_sha256": sha(ROOT / op),
        "acceptance_sha256": sha(ROOT / ap),
        "adapter_versions_and_hashes_match": True,
        "financial_targets_budget_gates_status_and_external_flags_unchanged": True,
        "fills_outside_first_session_after_month_end": 96,
        "new_financial_accounts": 0,
        "external_attestation": False,
        "profit_goal_met": False,
        "scope": "Current internal file and adapter consistency only; not historical publication, original pre-evaluation sealing, source authenticity or external acceptance.",
    }
    (doc / "signal-order-verification.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
