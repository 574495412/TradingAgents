"""Keep public performance claims separate from exactly replicated accounts."""

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from research_tools.nfi_reproduction.audit_original import compare


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(repository, output):
    output.mkdir(parents=True, exist_ok=False)
    captured = repository / "data/research/github_replication_sources_20261027_v001"
    ledger_path = captured / "Apex-prim__strategy-audit/LEDGER.csv"
    rows = list(csv.DictReader(ledger_path.open()))
    if len(rows) != 895 or len({r["strategy"] for r in rows}) != 895:
        raise ValueError("Published ledger coverage changed")
    numeric = sorted(
        (r for r in rows if r["os_total"]), key=lambda r: float(r["os_total"]), reverse=True
    )
    selected = numeric[:8] + [
        r for r in rows if r["strategy"] in ["ClucHAnix_5m_old", "CombinedBinHClucAndMADV5"]
    ]
    candidates = []
    for r in selected:
        candidates.append(
            {
                "id": "apex:" + r["strategy"],
                "strategy": r["strategy"],
                "code_repository": r["repo"],
                "measurement_owner": "Apex-prim/strategy-audit",
                "measurement_commit": "c83edf2a7ba13733addfd39caed71dfed120fce0",
                "measurement_kind": "publisher benchmark, not necessarily strategy-author original configuration",
                "disclosed_total_return_pct": float(r["os_total"]),
                "disclosed_window": "20200301-20260820 requested; report says through2026-08-19; native effective bounds not exported here",
                "disclosed_trades": int(r["os_trades"]),
                "disclosed_fee_per_side": 0.001,
                "disclosed_capital_USDT": 1000,
                "disclosed_slots": 8,
                "disclosed_market": "spot",
                "disclosed_stake": "unlimited",
                "disclosed_leverage": 1,
                "config_origin": "public setup_ft.py settings; not a captured executed config",
                "disclosed_whitelist": [
                    "BTC/USDT",
                    "LTC/USDT",
                    "ETH/USDT",
                    "XRP/USDT",
                    "ADA/USDT",
                    "XLM/USDT",
                    "XMR/USDT",
                    "DASH/USDT",
                ],
                "metric_scope": "native Freqtrade Total profit %; source value preserved, not independently reproduced",
                "publisher_lookahead": r["lookahead"],
                "publisher_recursive": r["recursive"],
                "publisher_recursive_kind": r["recursive_kind"],
                "publisher_dropped_at": r["dropped_at"],
                "publisher_gates_adopted_as_local_policy": False,
                "locally_replicated": False,
                "common_condition_rank": None,
                "missing_original_contract": [
                    "exact strategy source commit/hash used by published run",
                    "original raw native trades/orders/ending inventory",
                    "original market-data hashes",
                    "exact original engine/dependency identity",
                    "matching measured harness source fingerprint",
                ],
                "source_url": "https://github.com/Apex-prim/strategy-audit/blob/c83edf2a7ba13733addfd39caed71dfed120fce0/corpus/"
                + r["strategy"]
                + ".md",
            }
        )
    index_path = captured / "CacheCarti__Crypto-Strategies/index.json"
    index = json.loads(index_path.read_text())
    tree_git = repository / "data/research/github_crypto_cards_tree_20261028_v001.git"
    paths = subprocess.check_output(
        ["git", "--git-dir=" + str(tree_git), "ls-tree", "-r", "--name-only", "HEAD"], text=True
    ).splitlines()
    counts = {
        prefix: sum(p.startswith(prefix) for p in paths)
        for prefix in ["strategies/", "passed/", "cards/"]
    }
    cards_dir = repository / "data/research/github_crypto_cards_20261028_v001"
    cards = [
        json.loads(p.read_text())
        for p in cards_dir.glob("*.json")
        if p.name not in ["plan.json", "receipt.json"]
    ]
    for r in sorted(cards, key=lambda r: r["oos"]["total_return_bps"], reverse=True):
        candidates.append(
            {
                "id": "cachecarti:" + r["key"],
                "strategy": r["key"],
                "code_repository": "CacheCarti/Crypto-Strategies",
                "measurement_owner": "CacheCarti/Crypto-Strategies",
                "measurement_commit": "f8e32599c192b823dbe7aea77a2b9341c6114527",
                "market_domain": r["domain"],
                "disclosed_status": r["status"],
                "disclosed_total_return_bps": r["oos"]["total_return_bps"],
                "disclosed_return_ratio": r["oos"]["total_return_bps"] / 10000,
                "disclosed_trades": r["oos"]["total_trades"],
                "disclosed_window": None,
                "locally_replicated": False,
                "common_condition_rank": None,
                "missing_original_contract": [
                    "dated exact evaluation windows",
                    "raw dated input data and hashes",
                    "full original backtest engine and dependency versions",
                    "original account funding/stake/leverage/fee contract",
                    "complete trade/order/inventory ledger",
                ],
                "source_url": "https://github.com/CacheCarti/Crypto-Strategies/blob/f8e32599c192b823dbe7aea77a2b9341c6114527/cards/"
                + r["key"]
                + ".json",
            }
        )
    author_path = (
        repository
        / "data/research/nfi_original_author_artifact_20261027_v001/text/backtest-result-2026-10-22_12-14-52.json"
    )
    original = json.loads(author_path.read_text())["strategy"]["NostalgiaForInfinityX8"]
    native_root = Path(
        "/Users/chrise/GitHub/freqtrade/user_data/crypto_quant/nfi_original_20261027_v001/A_original_observed_v003"
    )
    native_zip = next((native_root / "native_results").glob("*.zip"))
    with zipfile.ZipFile(native_zip) as archive:
        name = next(
            n for n in archive.namelist() if n.endswith(".json") and not n.endswith("_config.json")
        )
        local = json.loads(archive.read(name))["strategy"]["NostalgiaForInfinityX8"]
    metric_keys = [
        "backtest_start",
        "backtest_end",
        "starting_balance",
        "final_balance",
        "profit_total",
        "profit_total_abs",
        "total_trades",
        "wins",
        "losses",
        "max_drawdown_account",
        "sharpe",
        "max_open_trades",
        "stake_amount",
        "stoploss",
    ]
    alignment = [
        {
            "metric": key,
            "author": original[key],
            "local": local[key],
            "match": not compare(original[key], local[key]),
        }
        for key in metric_keys
    ]
    if not all(r["match"] for r in alignment):
        raise ValueError("Original author metric parity lost")
    if compare(original["trades"], local["trades"]):
        raise ValueError("Original order parity lost")
    candidates.append(
        {
            "id": "nfi:original-v18.0.146-202501",
            "strategy": "NostalgiaForInfinityX8",
            "code_repository": "iterativv/NostalgiaForInfinity",
            "code_commit": "1af55971d3a64703ec449078e09fdec257d0d3c6",
            "measurement_kind": "original author raw account and local native raw account",
            "locally_replicated": True,
            "disclosed_total_return_pct": original["profit_total"] * 100,
            "native_closed_trade_sharpe": original["sharpe"],
            "native_closed_trade_drawdown": original["max_drawdown_account"],
            "starting_capital_USDT": original["starting_balance"],
            "original_fee_per_side": 0.0005,
            "leverage": 3,
            "window": "20250101-20250201",
            "market": "isolated_futures",
            "common_condition_rank": None,
            "source_url": "https://github.com/iterativv/NostalgiaForInfinity/commit/1af55971d3a64703ec449078e09fdec257d0d3c6#commitcomment-206135926",
        }
    )
    gene_root = repository / "data/research/github_genetrader_reports_20261028_v001"
    candidates.append(
        {
            "id": "genetrader:gen10-1734895087-6007",
            "strategy": "GeneTrader_gen10_1734895087_6007",
            "code_repository": "imsatoshi/GeneTrader",
            "code_commit": "20053ba673a58976d5f5ab05cb2ec09018cbfd96",
            "measurement_kind": "publisher text backtest and source; not local replication",
            "disclosed_total_return_pct": 231.74,
            "starting_capital_USDT": 300,
            "disclosed_final_balance_USDT": 995.233,
            "disclosed_window": "2024-11-18T00:00:00 through2024-12-22T17:15:00",
            "disclosed_market": "spot",
            "disclosed_slots": 4,
            "disclosed_trades": 113,
            "disclosed_win_rate": 1.0,
            "disclosed_closed_trade_drawdown": 0.0,
            "locally_replicated": False,
            "common_condition_rank": None,
            "missing_original_contract": [
                "exact engine/dependencies; public Dockerfile uses mutable stable tag",
                "raw dated input data and hashes",
                "executed configuration and complete original fee contract",
                "raw native trade/order/inventory export",
            ],
            "source_url": "https://github.com/imsatoshi/GeneTrader/blob/20053ba673a58976d5f5ab05cb2ec09018cbfd96/daily_results/20241223/gen10/results.txt",
        }
    )
    history = (
        repository / "data/research/github_replication_harness_history_20261028_v001/receipt.json"
    )
    hashes = json.loads(history.read_text())
    status = {
        "schema": "github-public-strategy-performance-registry.v1",
        "original_source_files_executed": False,
        "new_financial_accounts": 0,
        "new_factors": 0,
        "optimizations_started": 0,
        "disclosed_leader_within_apex_ledger": numeric[0]["strategy"],
        "disclosed_leader_is_locally_verified": False,
        "globally_highest_strategy": None,
        "common_condition_winner": None,
        "exact_original_accounts_reverified": 1,
        "candidates": candidates,
        "coverage": {
            "apex_ledger_rows": len(rows),
            "apex_numeric_claim_rows": len(numeric),
            "cachecarti_index_rows": len(index),
            "cachecarti_tree_file_counts": counts,
            "cachecarti_detailed_cards_read": len(cards),
            "duplicate_strategy_code_not_independent": True,
        },
        "source_conflicts": {
            "apex_report_harness_md5_prefix": "590bf74986c5",
            "current_public_harness_md5_prefix": hashlib.md5(
                (
                    repository / "data/research/github_replication_sources_20261028_v003/harness.py"
                ).read_bytes()
            ).hexdigest()[:12],
            "historical_harness_versions_examined": len(hashes),
            "matching_report_harness_found": any(x["md5_prefix"] == "590bf74986c5" for x in hashes),
            "cachecarti_index_omits_passed_files": len(index) == 406 and counts["passed/"] == 10,
            "cachecarti_passed_cards_all_negative_disclosed_OOS": all(
                r["oos"]["total_return_bps"] < 0 for r in cards if r["status"] == "passed"
            ),
        },
        "notank_current_source_counterexample": json.loads(
            (
                repository
                / "results/research/github_notank_prefix_20261028_v001/counterexample.json"
            ).read_text()
        ),
        "gene_publisher_report_source_sha256": digest(
            gene_root / "daily_results/20241223/gen10/results.txt"
        ),
        "ranking_rule": "Raw claim order is a retrieval priority only. No cross-period/market/leverage/account-denominator ranking until faithful replication and a common-condition contract.",
        "source_sha256": {
            str(p): digest(p) for p in [ledger_path, index_path, author_path, native_zip, history]
        },
        "orders_enabled": False,
        "production_eligible": False,
    }
    (output / "candidate_registry.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "nfi_author_metric_alignment.json").write_text(
        json.dumps(
            {
                "metrics": alignment,
                "all_metrics_match": True,
                "all_trade_order_fields_match": True,
                "closed_trade_and_full_equity_metrics_are_separate": True,
                "source_sha256": {
                    str(author_path): digest(author_path),
                    str(native_zip): digest(native_zip),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "candidates_inspected": len(candidates),
                "exact_original_baselines": 1,
                "claim_leader": numeric[0]["strategy"],
                "global_or_common_winner": None,
                "source_conflicts": status["source_conflicts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(a.repository.resolve(), a.output.resolve())
