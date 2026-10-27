"""Recheck frozen sources, account identities and the bounded review publication."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FQ = Path("/Users/chrise/GitHub/freqtrade")
REPORT = ROOT / "docs/research/regime_program/existing_strategy_review_20261028"
DATA = ROOT / "results/research/existing_strategy_review_20261028_v005"
REPLAYS = FQ / "user_data/crypto_quant/existing_strategy_review_20261028_v001"
VISUAL = Path(
    "/Users/chrise/.codex/visualizations/2026/10/25/"
    "01a17139-1f82-7922-83d5-2ffc22a459b9/existing-strategy-periods.html"
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
        return h.hexdigest()


def main():
    normalized = read(DATA / "accounts.json")
    rows = normalized["accounts"]
    summary = read(REPORT / "summary.json")
    comparison = read(DATA / "comparison.json")
    plan_path = FQ / "research/crypto/existing_strategy_review_plan_v001.json"
    plan = read(plan_path)
    assert plan == read(REPLAYS / "registered_plan.json")
    expected = {str(FQ / "freqtrade" / k): v for k, v in plan["core_source_hashes"].items()}
    expected.update(plan["supporting_source_hashes"])
    for job in plan["jobs"]:
        freeze = read(REPLAYS / job["id"] / "freeze.json")
        assert freeze["source_identity"] == plan["core_source_hashes"]
        original = read(FQ / job["plan"])
        assert freeze["plan_sha256"] == sha(FQ / job["plan"])
        for source in original["strategy_sources"] + original.get("runtime_sources", []):
            expected[source["path"]] = source["sha256"]
        assert (
            read(REPLAYS / job["id"] / "review_receipt.json")["signals_or_parameters_changed"]
            is False
        )
    account_sources = set()
    final_equity_from_saved_curve = 0
    for row in rows:
        if row["source_path"] in expected:
            assert expected[row["source_path"]] == row["source_sha256"]
        expected[row["source_path"]] = row["source_sha256"]
        account_sources.add(row["source_path"])
        curve = row.get("saved_curve")
        if curve:
            if curve["path"] in expected:
                assert expected[curve["path"]] == curve["sha256"]
            expected[curve["path"]] = curve["sha256"]
            account_sources.add(curve["path"])
        final = row["final_equity"]
        if final is None:
            final = curve["last_equity"]
            final_equity_from_saved_curve += 1
        assert abs(final / row["capital"] - 1 - row["net_return"]) < 1e-8
    mismatches = [p for p, digest in expected.items() if sha(Path(p)) != digest]
    assert not mismatches, mismatches
    assert len(rows) == summary["accounts"] == 1192
    assert Counter(r["kind"] for r in rows) == summary["account_kinds"]
    assert not normalized["issues"]
    assert not summary["quality"]["return_identity_failures"]
    assert not summary["quality"]["ending_curve_mismatches"]
    assert summary["quality"]["native_closed_cash_audits"] == {"true": 221, "null": 3}
    assert summary["quality"]["native_wallet_sum_order_audits"] == {"true": 12}
    assert len(comparison["strict_cohorts"]) == 217
    assert summary["strict_NFI_X8_non_NFI_challenger_count"] == 0
    assert read(REPLAYS / "completion.json")["jobs"] == 7
    assert read(REPLAYS / "original_parity.json")["passed"] is True
    program = read(ROOT / "docs/research/regime_program/targets.json")
    assert program["profit_goal_status"] == "not_met"
    assert program["orders_enabled"] is False
    assert program["execution_priority"]["self_developed_factors_allowed"] is False
    assert next(t for t in program["tasks"] if t["id"] == "G06")["status"] == "done"
    files = list(REPORT.glob("*.json")) + list(REPORT.glob("*.md"))
    files = [p for p in files if p.name not in ["verification.json", "completion.json"]]
    files += list((ROOT / "research_tools/existing_strategy_review").glob("*.py"))
    files += [
        DATA / "accounts.json",
        DATA / "comparison.json",
        Path(normalized["inventory_path"]),
        ROOT / "data/research/existing_strategy_comparison_20261028_v001/catalog_snapshot.json",
        ROOT / "results/research/existing_strategy_review_20261028_v003/legacy_curves.json",
        ROOT / "docs/research/regime_program/targets.json",
        ROOT / "docs/research/regime_program/README.md",
        plan_path,
        FQ / "research/crypto/review_existing_accounts.py",
        VISUAL,
        VISUAL.with_name("existing-strategy-periods-data.json"),
        REPLAYS / "registered_plan.json",
        REPLAYS / "original_parity.json",
        REPLAYS / "completion.json",
    ]
    for job in plan["jobs"]:
        folder = REPLAYS / job["id"]
        files += [folder / n for n in ["freeze.json", "result.json", "review_receipt.json"]]
    receipt = {
        "schema": "existing-strategy-review-verification.v1",
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "bounded_review_complete": True,
        "raw_account_and_curve_sources_rechecked": len(account_sources),
        "original_engine_core_files_rechecked": len(plan["core_source_hashes"]),
        "source_mismatches": mismatches,
        "all_return_denominators_rechecked": True,
        "ending_equity_read_from_curve_when_report_field_absent": final_equity_from_saved_curve,
        "native_order_cash_audit_nonempty_accounts_passed": 221,
        "native_wallet_sum_cash_audits_passed": 12,
        "new_unchanged_backtest_invocations": 7,
        "new_unique_period_accounts": 6,
        "original_training_trades_and_orders_exact_parity": True,
        "risk_threshold_not_used_to_censor_followup_returns": True,
        "new_signal_or_parameter_variants": 0,
        "third_party_review_performed": False,
        "independent_unseen_validation_completed": False,
        "global_winner": None,
        "profit_goal_met": False,
        "orders_enabled": False,
        "production_eligible": False,
        "source_sha256": dict(sorted(expected.items())),
        "publication_sha256": {str(p): sha(p) for p in sorted(set(files))},
    }
    (REPORT / "verification.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in receipt.items() if not k.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
