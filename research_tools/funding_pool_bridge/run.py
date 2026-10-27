"""One finite execution bridge; equal intermediate inputs are proofs, not new trials."""

import argparse
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.factor_lab.metrics import portfolio_metrics
from research_tools.factor_lab.study import executor
from research_tools.funding_pool_bridge.prepare import load_market
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.regime_program.storage import write_account
from research_tools.sparse_market.account_audit import audit_account


def sealed_account(run, identity):
    folder = Path(run) / "accounts" / identity
    result = read(folder / "result.json")
    for name, digest in result["source_hashes"].items():
        if Path(name).name != name or sha(folder / name) != digest:
            raise ValueError("Frozen endpoint changed: " + str(folder / name))
    return folder, result


def account_components(folder):
    events = read(folder / "events.json")
    trades = read(folder / "trades.json")
    result = read(folder / "result.json")
    symbols = sorted({e["symbol"] for e in events})
    rows = []
    for s in symbols:
        own = [e for e in events if e["symbol"] == s]
        funding = math.fsum(e["cash_delta"] for e in own if e["type"] == "funding")
        fees = math.fsum(e.get("fee", 0) for e in own)
        total = math.fsum(t["net_pnl"] for t in trades if t["symbol"] == s)
        if s in result["open_positions"]:
            raise ValueError("Per-symbol closed attribution requires flat endpoints")
        rows.append(
            {
                "symbol": s,
                "net_pnl": total,
                "funding": funding,
                "fees": fees,
                "price_after_slippage": total - funding + fees,
            }
        )
    if abs(math.fsum(r["net_pnl"] for r in rows) - result["metrics"]["net_pnl"]) > 1e-6:
        raise ValueError("Per-symbol cash attribution fails")
    return rows


def run(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    plan = read(inputs / "plan.json")
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Registered source changed: " + path)
    for section in read(inputs / "market_bindings.json").values():
        for binding in section.values():
            for path, digest in binding.items():
                if sha(Path(path)) != digest:
                    raise ValueError("Execution source changed: " + path)
    for path, digest in read(inputs / "daily_bindings.json").items():
        if sha(Path(path)) != digest:
            raise ValueError("Daily fact changed: " + path)
    comparison = read(inputs / "source_comparison.json")
    if any(
        any(r[k][v] for v in ["different_rows", "left_only", "right_only"])
        for r in comparison
        for k in ["price_quote", "quality", "funding"]
    ):
        raise ValueError(
            "Source nodes cannot be folded: execute separately under registered budget"
        )
    target_check = read(inputs / "target_comparison.json")
    if target_check["old_vs_matched40"]["different_cells"] or read(inputs / "lifecycle40.json"):
        raise ValueError("Lifecycle/signal nodes cannot be folded: execute separately")
    s = plan["inputs"]
    old_id = "archive_diagnostic2026_funding7_raw_cap10000_slip10"
    pit_id = "daily_archive_diagnostic2026_funding7_raw_cap10000_slip10"
    old_folder, old = sealed_account(s["old_run"], old_id)
    pit_folder, pit = sealed_account(s["pit_run"], pit_id)
    old_scores_path = Path(s["old_inputs"]) / "archive_diagnostic2026_funding7_raw_targets.parquet"
    scores = pd.read_parquet(old_scores_path)
    matched = pd.read_parquet(inputs / "matched40_targets.parquet").reindex(columns=scores.columns)
    pd.testing.assert_frame_equal(scores, matched, check_freq=False)
    for result in [old, pit]:
        if result["config"] != plan["account"] or result["window"] != plan["window"]:
            raise ValueError("Endpoint account/window configuration differs")
    engine_path = Path("../freqtrade/research/crypto/sparse_calendar_v2.py").resolve()
    paths = [
        Path(__file__),
        Path("research_tools/funding_pool_bridge/prepare.py"),
        engine_path,
        engine_path.with_name("sparse_sources_v2.py"),
        Path("research_tools/sparse_market/account_audit.py"),
        Path("research_tools/factor_lab/metrics.py"),
        Path("research_tools/regime_program/storage.py"),
        inputs / "completion.json",
        old_folder / "result.json",
        pit_folder / "result.json",
        old_scores_path,
    ]
    registration = {
        "schema": "funding-bridge-account-registration.v1",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "before_new_account_returns": True,
        "inputs_completion_sha256": sha(inputs / "completion.json"),
        "source_hashes": {str(p): sha(p) for p in paths},
        "computed_nodes": ["B_execution"],
        "folded_nodes": {
            "C_source": "numerically identical execution and funding sources to B",
            "D_lifecycle": "empty old40directives, same as C",
            "E_matched40": "every hourly target identical to old40, same as D",
        },
        "reused_endpoints": ["A_legacy", "F_pit"],
        "new_independent_hypotheses": 0,
        "new_financial_accounts_max": 1,
        "independent_oos": False,
        "orders_enabled": False,
    }
    save(output / "registration.json", registration)
    save(output / "plan.json", plan)
    sys.path.insert(0, str(engine_path.parents[2]))
    engine = executor(str(engine_path))
    bars, rates, _ = load_market(plan, "old", plan["old_symbols"])
    account = engine.replay(
        bars, rates, scores, plan["account"], plan["window"]["start"], plan["window"]["end"]
    )
    days = pd.date_range(plan["window"]["start"], plan["window"]["end"], freq="D", tz="UTC")
    benchmark = bars["BTCUSDT"].close.reindex(days - pd.Timedelta(minutes=5))
    benchmark.index = days
    metrics = portfolio_metrics(
        account["curve"],
        account["trades"],
        account["events"],
        plan["account"]["capital"],
        benchmark,
    )
    metrics["open_campaign_net_pnl"] = (
        sum(p["net_pnl"] for p in account["positions"].values())
        + metrics["open_pnl_with_exit_reserve"]
    )
    account["metrics"] = metrics
    identity = "B_execution"
    folder = output / "accounts" / identity
    write_account(folder, account, plan["account"], plan["window"])
    audit = audit_account(output, {"id": identity}, bars, rates, scores)
    save(folder / "internal_cash_audit.json", audit)
    old_curve = pd.read_parquet(old_folder / "equity_5m.parquet")
    pd.testing.assert_series_equal(account["curve"].date, old_curve.date, check_names=False)
    errors = {
        col: float(np.max(np.abs(account["curve"][col].to_numpy() - old_curve[col].to_numpy())))
        for col in ["equity", "cash", "gross_exposure", "net_exposure"]
    }
    same = errors["equity"] <= plan["decision_record"]["equality_tolerance_usdt"]
    if not same:
        raise ValueError(
            "Execution itself changes old40NAV; preserve account and diagnose before claiming pool-only effect"
        )
    for key in [
        "net_return",
        "win_rate",
        "max_drawdown",
        "sharpe",
        "closed_trades",
        "funding_net",
        "fees",
        "slippage_cost",
    ]:
        if abs((metrics[key] or 0) - (old["metrics"][key] or 0)) > 1e-8:
            raise ValueError("Endpoint metric changed: " + key)
    nodes = []
    for node in plan["ladder"]:
        source_metrics = (
            pit["metrics"]
            if node["id"] == "F_pit"
            else old["metrics"]
            if node["id"] == "A_legacy"
            else metrics
        )
        nodes.append(
            {
                "id": node["id"],
                "metrics": source_metrics,
                "evidence": (
                    "sealed endpoint"
                    if node["id"] in ["A_legacy", "F_pit"]
                    else "new replay and internal audit"
                    if node["id"] == identity
                    else "exact input equivalence; no new account"
                ),
            }
        )
    old_components, pit_components = account_components(old_folder), account_components(pit_folder)
    old_by = {r["symbol"]: r for r in old_components}
    pit_by = {r["symbol"]: r for r in pit_components}
    attribution = [
        {
            "symbol": s,
            "old": old_by.get(s),
            "pit": pit_by.get(s),
            "delta_net_pnl": (pit_by.get(s) or {}).get("net_pnl", 0)
            - (old_by.get(s) or {}).get("net_pnl", 0),
        }
        for s in sorted(set(old_by) | set(pit_by))
    ]
    total_delta = pit["metrics"]["net_pnl"] - old["metrics"]["net_pnl"]
    residual = total_delta - math.fsum(r["delta_net_pnl"] for r in attribution)
    if abs(residual) > 1e-6:
        raise ValueError("Bridge per-symbol attribution residual")
    save(output / "symbol_attribution.json", attribution)
    periods = []
    for label, folder0 in [("fixed40", old_folder), ("pit", pit_folder)]:
        curve = pd.read_parquet(folder0 / "equity_5m.parquet").set_index("date").equity
        for begin, end in [
            ("2026-01-01", "2026-04-01"),
            ("2026-04-01", "2026-07-01"),
            ("2026-07-01", "2026-09-01"),
        ]:
            a, b = pd.Timestamp(begin, tz="UTC"), pd.Timestamp(end, tz="UTC")
            window = curve.loc[a:b]
            periods.append(
                {
                    "account": label,
                    "start": begin,
                    "end": end,
                    "net_pnl": float(window.iloc[-1] - window.iloc[0]),
                    "net_return": float(window.iloc[-1] / window.iloc[0] - 1),
                    "max_drawdown_within_period": float((1 - window / window.cummax()).max()),
                }
            )
    save(output / "periods.json", periods)
    summary = {
        "schema": "funding-pool-bridge-result.v1",
        "nodes": nodes,
        "actual_new_financial_accounts": 1,
        "reused_endpoints": 2,
        "folded_equivalent_nodes": 3,
        "new_independent_hypotheses": 0,
        "internal_cash_audit": audit,
        "full_5m_execution_bridge_errors": errors,
        "old_and_sparse_Nav_equal_within_tolerance": same,
        "historical_pool_path_delta_usdt": total_delta,
        "historical_pool_path_delta_percentage_points": total_delta
        / plan["account"]["capital"]
        * 100,
        "symbol_attribution_residual_usdt": residual,
        "orders_enabled": False,
        "production_eligible": False,
        "profit_goal_met": False,
        "independent_unseen_evidence": False,
        "third_party_review": False,
        "scope": "2026coldstart,243days,10kUSDT,15bpsperlegside; sourceandexecutionaligned. Conditional historical path change, not a general causal premium or admitted alpha.",
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()
            },
            "profit_goal_met": False,
        },
    )
    print(
        {
            "passed": True,
            "new_accounts": 1,
            "nav_max_error": errors["equity"],
            "pool_delta_usdt": total_delta,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output)
