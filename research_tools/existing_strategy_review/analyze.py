"""Compare only compatible existing accounts and retain higher-return/risk tradeoffs."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core_identity(row):
    directory = Path(row["source_path"]).parent
    p = directory / "freeze.json"
    if p.is_file():
        v = json.loads(p.read_text())
        identity = v.get("source_identity")
        if identity:
            return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    p = directory / "registered_plan.json"
    if p.is_file():
        v = json.loads(p.read_text())
        return v.get("engine_image_manifest") or v.get("engine_version")
    return None


def brief(row):
    obs = row.get("saved_curve") or {}
    return {
        k: row.get(k)
        for k in [
            "id",
            "strategy",
            "study",
            "source_path",
            "start",
            "end",
            "capital",
            "net_return",
            "closed_trades",
            "win_rate",
            "fees_per_side",
            "leverage_values",
            "gross_budget",
            "pairlist_count",
            "order_cash_audit",
            "risk_halted",
        ]
    } | {
        "sampled_full_drawdown": obs.get("sampled_full_drawdown"),
        "daily_equity_sharpe": obs.get("recomputed_daily_sharpe"),
        "observation_rows": obs.get("rows"),
        "daily_return_count": obs.get("daily_return_count"),
        "full_month_returns": obs.get("month_returns", {}),
    }


def analyze(accounts_path, legacy_path, output):
    accounts = json.loads(accounts_path.read_text())
    rows = accounts["accounts"]
    legacy = json.loads(legacy_path.read_text())
    cohorts = defaultdict(list)
    excluded = Counter()
    source_identity_cache = {}
    for row in rows:
        obs = row.get("saved_curve") or {}
        if row["kind"] == "native_account":
            identity = source_identity_cache.setdefault(
                str(Path(row["source_path"]).parent), core_identity(row)
            )
            if not identity:
                excluded["native_missing_engine_identity"] += 1
                continue
            if (
                not obs
                or "start" not in obs
                or not row["fees_per_side"]
                or not row["leverage_values"]
            ):
                excluded["native_missing_curve_cost_or_actual_leverage"] += 1
                continue
            key = (
                "native",
                row["currency"],
                row["trading_mode"],
                obs["start"],
                obs["end"],
                row["capital"],
                tuple(round(x, 12) for x in row["fees_per_side"]),
                tuple(row["leverage_values"]),
                row["pairlist_sha256"],
                identity,
            )
        elif row["kind"] == "custom_account_with_curve" and obs:
            # Compare parameter/policy cases only within the same original study.
            key = (
                "custom",
                row["study"],
                row["currency"],
                row["start"],
                row["end"],
                row["capital"],
                tuple(row["fees_per_side"]),
                row["slippage_bps"],
                row["gross_budget"],
            )
        else:
            continue
        cohorts[key].append(row)
    matches = []
    for key, group in cohorts.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r["net_return"], reverse=True)
        front = []
        for r in group:
            dd = (r.get("saved_curve") or {}).get("sampled_full_drawdown")
            if dd is not None and not any(
                o["net_return"] >= r["net_return"]
                and o["saved_curve"]["sampled_full_drawdown"] <= dd
                and (
                    o["net_return"] > r["net_return"]
                    or o["saved_curve"]["sampled_full_drawdown"] < dd
                )
                for o in group
                if (o.get("saved_curve") or {}).get("sampled_full_drawdown") is not None
            ):
                front.append(r["id"])
        matches.append(
            {
                "cohort": list(key),
                "accounts": len(group),
                "return_leader": brief(group[0]),
                "risk_return_frontier_ids": front,
                "rows": [brief(r) for r in group],
            }
        )
    # Registered new incumbents plus established original/old NFI accounts.
    focus = []
    for r in rows:
        if (
            r["kind"] == "native_account"
            and r["strategy"]
            in [
                "TrendControl",
                "CompressionRetest",
                "NFI2023OneXCompact",
                "NFILeverage1x",
                "NFIReference3x",
                "NostalgiaForInfinityX7",
            ]
            and r["fees_per_side"]
            and abs(r["fees_per_side"][0] - 0.0008) < 1e-8
            and ("compat" not in r["source_path"])
            and r["closed_trades"]
        ):
            focus.append(brief(r))
    nfi = [
        brief(r)
        for r in rows
        if r["kind"] == "native_account" and r["strategy"] == "NostalgiaForInfinityX8"
    ]
    matching_challengers = []
    for c in matches:
        if any(r["strategy"] == "NostalgiaForInfinityX8" for r in c["rows"]):
            matching_challengers.extend(
                r for r in c["rows"] if r["strategy"] != "NostalgiaForInfinityX8"
            )
    high_crypto = sorted(
        [r for r in rows if r["market"] == "crypto" and r["kind"] == "custom_account_with_curve"],
        key=lambda r: r["net_return"],
        reverse=True,
    )
    stock = sorted(
        [r for r in rows if r["kind"] == "equity_account"],
        key=lambda r: r["net_return"],
        reverse=True,
    )
    quality = {
        "return_identity_failures": [
            r["id"] for r in rows if (r.get("return_identity_error") or 0) > 1e-6
        ],
        "ending_curve_mismatches": [
            r["id"]
            for r in rows
            if r.get("final_equity") is not None
            and (r.get("saved_curve") or {}).get("last_equity") is not None
            and abs(r["final_equity"] - r["saved_curve"]["last_equity"]) > 1e-5
        ],
        "native_closed_cash_audits": Counter(
            (r.get("order_cash_audit") or {}).get("passed")
            for r in rows
            if r["kind"] == "native_account"
        ),
        "native_wallet_sum_order_audits": Counter(
            (r.get("order_cash_audit") or {}).get("passed")
            for r in rows
            if r["kind"] == "native_wallet_sum"
        ),
    }
    value = {
        "schema": "existing-strategy-aligned-comparison.v1",
        "accounts": len(rows),
        "account_kinds": Counter(r["kind"] for r in rows),
        "markets": Counter(r["market"] for r in rows),
        "native_strategy_classes": len(
            {r["strategy"] for r in rows if r["kind"] == "native_account"}
        ),
        "duplicate_account_aliases": sum(len(r["aliases"]) for r in rows),
        "summary_nodes_retained": len(accounts["reported_summary_nodes"]),
        "legacy_curves": len(legacy["curves"]),
        "legacy_curves_with_explicit_capital": sum(
            r["capital"] is not None for r in legacy["curves"]
        ),
        "quality": quality,
        "strict_cohorts_with_multiple_accounts": len(matches),
        "strict_cohorts": matches,
        "cohort_exclusions": excluded,
        "incumbent_focus": focus,
        "NFI_X8_original_and_cost_risk_cases": nfi,
        "strict_NFI_X8_non_NFI_challengers": matching_challengers,
        "global_better_than_original_NFI_proven": False,
        "top_custom_crypto_original_scopes": [brief(r) for r in high_crypto[:12]],
        "stock_summary_separate_from_crypto": [brief(r) for r in stock[:8]],
        "existing_native_wallet_portfolios": [
            brief(r) | {"independent_wallet_count": r["native_account_count"]}
            for r in rows
            if r["kind"] == "native_wallet_sum"
        ],
        "new_signal_or_parameter_variants": 0,
        "new_replication_invocations": 7,
        "new_unique_period_accounts": 6,
        "new_independent_OOS_claim": False,
        "source_sha256": {
            str(accounts_path): sha(accounts_path),
            str(legacy_path): sha(legacy_path),
        },
        "orders_enabled": False,
        "production_eligible": False,
    }
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "accounts": len(rows),
                "native_classes": value["native_strategy_classes"],
                "cohorts": len(matches),
                "quality": quality,
                "matched_NFI_X8_challengers": len(matching_challengers),
                "focus": [
                    {
                        k: r[k]
                        for k in [
                            "strategy",
                            "start",
                            "end",
                            "net_return",
                            "sampled_full_drawdown",
                            "daily_equity_sharpe",
                        ]
                    }
                    for r in focus
                    if "existing_strategy_review_" in r["source_path"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--accounts", type=Path, required=True)
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    analyze(a.accounts, a.legacy, a.output)
