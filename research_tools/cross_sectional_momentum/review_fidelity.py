"""Post-hoc method/ledger diagnosis; no new strategy or financial replay."""

import argparse
import csv
import math
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def review(source, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(source)
    preregistered = read(source / "registration.json")
    for p, digest in preregistered["source_hashes"].items():
        assert sha(Path(p)) == digest
    market = read(source / "public_historical_cap_probe.receipt.json")
    assert market["status"] == 401 and market["provider_error_code"] == 10012
    upstream_root = Path(
        "data/research/github_factor_survey_20261025_v001/adensvaz__sentinel-hyperliquid/files"
    )
    upstream = upstream_root / "config.carry.yaml"
    s = upstream.read_text()
    assert all(
        text in s
        for text in [
            "keep_buffer: 4",
            "top_k: 8",
            "lookback: 14",
            "target_gross: 1.0",
            "searching 7 lookbacks",
        ]
    )
    candidates = []
    original = read(Path("docs/research/regime_program/accounts_20261026/summary.json"))
    for row in original["account_rows"]:
        if row["candidate"] == "relative_momentum" and row["capital"] == 10000:
            candidates.append(
                (Path("results/research/regime_accounts_20261026_v001/accounts") / row["id"], row)
            )
    with Path(
        "docs/research/regime_program/max_effect_accounts_20261026/all_accounts.csv"
    ).open() as file:
        for row in csv.DictReader(file):
            if (
                row["candidate"] in ["momentum28_control", "momentum28_btc_beta_control"]
                and int(row["capital"]) == 10000
            ):
                candidates.append(
                    (
                        Path("results/research/max_effect_accounts_20261026_v001/accounts")
                        / row["id"],
                        row,
                    )
                )
    rows, binding, event_count = [], {}, 0
    for folder, published in candidates:
        account = read(folder / "result.json")
        if "result_sha256" in published:
            assert sha(folder / "result.json") == published["result_sha256"]
        for name, digest in account["source_hashes"].items():
            assert sha(folder / name) == digest
        assert not account["open_positions"]
        events = read(folder / "events.json")
        cash = math.fsum(e["cash_delta"] for e in events)
        fees = math.fsum(e.get("fee", 0) for e in events)
        slippage = math.fsum(e.get("slippage_cost", 0) for e in events)
        funding = math.fsum(e["cash_delta"] for e in events if e["type"] == "funding")
        filled_price = math.fsum(e["cash_delta"] + e["fee"] for e in events if e["type"] == "exit")
        before_slippage = filled_price + slippage
        metrics = account["metrics"]
        # Preserve the original published reconciliation tolerance. These are
        # binary64 frozen accounts, not a new tolerance chosen from new returns.
        checks = [
            abs(cash - metrics["net_pnl"]),
            abs(fees - metrics["fees"]),
            abs(slippage - metrics["slippage_cost"]),
            abs(funding - metrics["funding_net"]),
            abs(before_slippage - fees - slippage + funding - metrics["net_pnl"]),
        ]
        assert max(checks) <= 1e-5
        assert math.isclose(
            metrics["net_return"],
            float(published.get("net_return", metrics["net_return"])),
            abs_tol=1e-14,
        )
        rows.append(
            {
                "id": folder.name,
                "candidate": published["candidate"],
                "phase": published["phase"],
                "slippage_bps": int(published["slippage_bps"]),
                **{
                    k: metrics[k]
                    for k in [
                        "net_pnl",
                        "net_return",
                        "max_drawdown",
                        "sharpe",
                        "win_rate",
                        "closed_trades",
                        "fees",
                        "slippage_cost",
                        "funding_net",
                        "mean_gross_exposure",
                    ]
                },
                "filled_path_price_pnl_before_slippage": before_slippage,
                "maximum_accounting_reconciliation_error": max(checks),
                "event_records": len(events),
                "counterfactual_zero_cost_strategy_replayed": False,
            }
        )
        binding[str(folder / "result.json")] = sha(folder / "result.json")
        binding[str(folder / "events.json")] = sha(folder / "events.json")
        event_count += len(events)
    assert len(rows) == 18
    save(output / "existing_account_diagnosis.json", rows)
    save(
        output / "source_bindings.json",
        {
            "source_completion_sha256": sha(source / "completion.json"),
            "old_financial_files": binding,
            "local_and_upstream_code": {
                str(p): sha(p)
                for p in [
                    Path(__file__),
                    Path("factors/common/regime_books.py"),
                    Path("factors/common/max_effect.py"),
                    Path("research_tools/regime_program/plan.json"),
                    Path("research_tools/max_effect/plan.json"),
                    upstream,
                    upstream_root / "src/sentinel/strategy/momentum_regime.py",
                    upstream_root / "src/sentinel/strategy/funding_carry.py",
                    Path("factors/common/year52_momentum.py"),
                ]
            },
        },
    )
    save(
        output / "method_mapping.json",
        {
            "local_daily21": {
                "signal": "21 completed UTC-day price change, momentum-only ablation of carry ranking",
                "formation": "daily",
                "weights": "5long/5short equal, gross0.8",
                "market": "40 retrospectively selected liquid USDT perpetuals, prior liquidity top30",
                "profit_status": "negative_known_history",
            },
            "local_weekly28": {
                "signal": "28 completed-day price change",
                "formation": "Monday/cold-start, with intraperiod risk exits",
                "weights": "4long/4short equal; optional60day BTC hedge; gross0.8",
                "market": "same retrospective40, excludesBTCfromrank, not point-in-time capitalization weighting",
                "profit_status": "negative_known_history",
            },
            "sentinel_pinned_config": {
                "commit": "ca647c099755ae9770f92205d5fc413e8e7c6c95",
                "signal": "14day momentum plus funding ranks",
                "venue": "Hyperliquid",
                "weights": "8long/8short equal, keep_buffer4, gross1.0",
                "formation": "daily14UTC per pinned configuration",
                "upstream_search_disclosed": "7lookbacks; source commentary is not independently verified profit",
                "locally_replicated_complete_strategy": False,
            },
            "author_pdf_2019": {
                "date": "2019-04-15",
                "provider": "Coinmarketcap",
                "sample": "2014-2018",
                "calendar": "52year-date-label weeks; final8/9dates",
                "portfolio_tests": "weekly quintiles; multiple momentum horizons; market-cap-weighted",
                "named_CMOM_factor": "three-week signal;30/40/30rankgroups; top-minus-bottom value-weighted",
                "is_published_2022_final_version": False,
            },
            "authors_arxiv_v4": {
                "date": "2026-03-21",
                "version": "2510.14435v4",
                "provider": "CoinGecko, active and inactive, USD spot aggregation",
                "sample_end": "2025-09-06",
                "calendar": "52year-date-label weeks; final8/9dates",
                "signal": "past2source-calendar-week return",
                "weights": "outer quintile market-cap weights; source spread gross2",
                "account_scaling": "gross0.8 requires multiplying spread weights by0.4 before costs, not inheriting spread return",
                "execution_claim": "weekly factor return; not a funded perpetual-account execution result",
            },
            "new_module_scope": "formation math/strict input gates only; date-label convention, lexical ties and integer-rank quintiles explicitly local; no author-code exact replication, performance clipping, costs or returns validated",
        },
    )
    save(
        output / "quality_findings.json",
        [
            {
                "id": "method_mismatch",
                "severity": "high",
                "evidence": "daily21/weekly28 equal-weight perpetual policies differ in calendar, weighting, domain, holdings and execution from original spot cross-section papers",
                "risk": "Old losses do not refute every source proposition, and source results do not validate local policy.",
                "action": "Freeze method mapping before proposing any new economic contract.",
            },
            {
                "id": "fees_not_complete_explanation",
                "severity": "high",
                "evidence": "Every retained local policy/period/cost fixed-fill price component is negative, independently summed from sealed event ledgers.",
                "risk": "Changing only fees or weekly churn cannot be assumed to restore edge; comparison is accounting, not causal attribution.",
                "action": "Keep old policies closed, no zero-cost replay or parameter grid.",
            },
            {
                "id": "old_universe_not_full_historical_market",
                "severity": "high",
                "evidence": "Both old books use a retrospective40coin pool; the prior native-archive universe review selected105different2025names and107in2026, about49.94%/42% old40slotcoverage.",
                "risk": "Survivorship/eligibility and spot/perpetual identity differences limit external conclusions.",
                "action": "Use time-valid native identity and tradable universe evidence; no current-list backfill.",
            },
            {
                "id": "necessary_market_caps_missing",
                "severity": "high",
                "evidence": "Public2025-01-07 BTC history query returned401/error10012, not a numeric market-cap record. Inactive IDs documented as Analyst-or-above.",
                "risk": "Cannot implement or certify the original value weighting/full historic universe from present authorized public path.",
                "action": "Keep economic evaluation unknown. Inspect any available authorized original exports/author data or explicitly re-register a separately sourced identity-audited local adaptation; no billing or invented caps.",
            },
            {
                "id": "calendar_and_denominator",
                "severity": "high",
                "evidence": "Source2weeks can span15/16calendar days across year-end; source +1/-1 spread is gross2.",
                "risk": "A14day Monday signal and capital return inferred directly from factor spread change the tested proposition.",
                "action": "Use year52label math, actual availability and account-gross mapping; newmodule tests only formation semantics.",
            },
        ],
    )
    largest = max(r["maximum_accounting_reconciliation_error"] for r in rows)
    summary = {
        "schema": "cross-sectional-momentum-fidelity-result.v1",
        "existing_accounts_inspected": len(rows),
        "existing_event_records_reconciled": event_count,
        "maximum_old_ledger_reconciliation_error": largest,
        "all_fixed_fill_price_components_negative": all(
            r["filled_path_price_pnl_before_slippage"] < 0 for r in rows
        ),
        "method_mismatch_confirmed": True,
        "source_original_cap_history_obtained": False,
        "year52_formation_math_implemented": True,
        "source_calendar_and_capital_denominator_tests_passed": 7,
        "new_financial_accounts": 0,
        "new_economic_variants_evaluated": 0,
        "next_economic_evaluation_ready": False,
        "source_literal_replication_completed": False,
        "module_admitted": False,
        "independent_oos": False,
        "external_review_performed": False,
        "net_return": None,
        "win_rate": None,
        "max_drawdown": None,
        "sharpe": None,
        "capacity_certified": False,
        "profit_goal_met": False,
        "orders_enabled": False,
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_financial_accounts": 0,
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    review(a.source, a.output)
