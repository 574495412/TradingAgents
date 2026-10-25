"""Publish paired source sensitivity, pure funding attribution and immutable audits."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.funding_cross_section.prepare_inputs import load_bars
from research_tools.regime_program.reporting import partitions

NAMES = {
    "funding7_raw": "纯7日资金费价差",
    "funding7_cost_floor": "资金费价差＋固定成本门槛",
    "momentum28_control": "普通28日动量对照",
    "cash": "原封存现金对照",
}


def actual_beta(curve, events, bars, features):
    times = pd.DatetimeIndex(curve.date)
    beta = np.zeros(len(times))
    for s, b in bars.items():
        delta = np.zeros(len(times))
        for e in events:
            if e["symbol"] == s and e["type"] in ["entry", "exit"]:
                delta[times.get_loc(pd.Timestamp(e["time"]))] += (
                    e["side"] * e["quantity"] * (1 if e["type"] == "entry" else -1)
                )
        quantity = np.cumsum(delta)
        if not np.any(quantity):
            continue
        prior = features.xs(s).beta60.reindex(times, method="ffill").to_numpy()
        prices = b.reindex(times - pd.Timedelta(minutes=5)).to_numpy()
        if not np.isfinite(prior).all() or not np.isfinite(prices).all():
            raise ValueError("Actual inventory beta history missing")
        beta += quantity * prices * prior / curve.equity.to_numpy()
    active = curve.gross_exposure.to_numpy() > 1e-8
    return {
        "mean_absolute_inventory_estimated_beta": float(np.abs(beta).mean()),
        "mean_absolute_inventory_estimated_beta_when_active": float(np.abs(beta[active]).mean())
        if active.any()
        else None,
        "maximum_absolute_inventory_estimated_beta": float(np.abs(beta).max()),
        "active_5m_intervals": int(active.sum()),
        "note": "Actual filled inventory times prior60day estimated beta, not target neutrality and not future realised beta. Daily realised BTC regression is separately reported in metrics.beta.",
    }


def publish(run, restart, output):  # noqa: C901
    receipt, plan, source = (
        read(run / "completion.json"),
        read(run / "plan.json"),
        read(run / "summary.json"),
    )
    if (
        sha(run / "summary.json") != receipt["summary_sha256"]
        or sha(run / "plan.json") != receipt["plan_sha256"]
        or not source["audit"]["passed"]
    ):
        raise ValueError("Account summary or audit changed")
    for path, digest in plan["account_source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Execution source changed after account run")
    check = read(restart / "audit.json")
    if (
        not check["passed"]
        or check["run_completion_sha256"] != sha(run / "completion.json")
        or len(check["checks"]) != 8
        or not all(r["passed"] for r in check["checks"])
    ):
        raise ValueError("Actual process continuation incomplete")
    states_root = Path("results/research/regime_states_20261025_v002")
    for name, digest in read(states_root / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(states_root / name) != digest:
            raise ValueError("Causal state source changed")
    states = pd.read_parquet(states_root / "states_daily.parquet")
    flags = pd.read_parquet(states_root / "event_risk_hourly.parquet").covered_fomc_post_release
    inputs = Path(plan["inputs"])
    feature_versions = {
        v: pd.read_parquet(inputs / (v + "_features_daily.parquet"))
        for v in plan["source_variants"]
    }
    close_versions = {
        v: {s: f.close for s, f in load_bars(plan, v).items()} for v in plan["source_variants"]
    }
    rows, details, curves, bindings = [], {}, {}, {}
    for row in source["account_rows"]:
        variant = row["variant"] if row["variant"] != "common" else "archive"
        bars, features = close_versions[variant], feature_versions[variant]
        folder = Path(row["source_run"]) / "accounts" / row["source_account_id"]
        if sha(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Account identity changed")
        account = read(folder / "result.json")
        if account["metrics"] != row["metrics"]:
            raise ValueError("Account metrics changed")
        for name, digest in account["source_hashes"].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError("Account ledger changed")
        curve = pd.read_parquet(folder / "equity_5m.parquet")
        events = read(folder / "events.json")
        detail = partitions(curve, account["trades"], states, flags, row["capital"])
        for group in ["regime", "covered_fomc"]:
            if (
                abs(
                    sum(
                        v["net_pnl_contribution"]
                        for v in detail["state_contributions"]
                        if v["group"] == group
                    )
                    - row["capital"] * row["metrics"]["net_return"]
                )
                > 1e-5
            ):
                raise ValueError("State contribution does not reconcile")
        yearly = [r["net_return"] for r in detail["periods"] if r["kind"] == "year"]
        if abs(float(np.prod([1 + r for r in yearly])) - 1 - row["metrics"]["net_return"]) > 1e-10:
            raise ValueError("Annual returns do not compound to account")
        settled = sum(p["net_pnl"] for p in account["open_positions"].values())
        open_total = settled + row["metrics"]["open_pnl_with_exit_reserve"]
        if (
            abs(
                row["metrics"]["closed_net_pnl"]
                + open_total
                - row["capital"] * row["metrics"]["net_return"]
            )
            > 1e-5
        ):
            raise ValueError("Closed and open campaign PnL differs")
        if not row["reused_reference"]:
            audit_path = folder / "independent_audit.json"
            a = read(audit_path)
            if not a["passed"] or a["result_sha256"] != row["result_sha256"]:
                raise ValueError("Financial audit binding changed")
            bindings[str(audit_path)] = sha(audit_path)
        detail["actual_beta"] = actual_beta(curve, events, bars, features)
        reserves = sum(
            p["q"]
            * bars[s].loc[pd.Timestamp(row["end"], tz="UTC") - pd.Timedelta(minutes=5)]
            * (account["config"]["fee_bps"] + account["config"]["slippage_bps"])
            / 10000
            for s, p in account["open_positions"].items()
        )
        m = row["metrics"]
        price_pnl = (
            row["capital"] * m["net_return"]
            - m["funding_net"]
            + m["fees"]
            + m["slippage_cost"]
            + reserves
        )
        detail["same_executed_path_attribution"] = {
            "pre_fee_funding_slippage_price_pnl": price_pnl,
            "funding_net": m["funding_net"],
            "fees": m["fees"],
            "slippage_cost": m["slippage_cost"],
            "open_exit_reserve": reserves,
            "note": "Accounting decomposition of the actual executed path, not a replay with costs disabled. Stops and fill paths can differ between costs.",
        }
        details[row["id"]] = detail
        rows.append(
            {
                **row,
                "open_campaign_total_net_pnl_with_exit_reserve": open_total,
                "actual_beta": detail["actual_beta"],
            }
        )
        curves[row["id"]] = curve.set_index("date")
    comparisons = []
    for row in rows:
        if row["reused_reference"] or row["candidate"] not in [
            "funding7_raw",
            "funding7_cost_floor",
        ]:
            continue
        control = "momentum28_control"
        match = next(
            (
                r
                for r in rows
                if r["candidate"] == control
                and r["variant"] == row["variant"]
                and r["phase"] == row["phase"]
                and r["capital"] == row["capital"]
                and r["slippage_bps"] == row["slippage_bps"]
            ),
            None,
        )
        if match is None:
            continue
        x, y = curves[row["id"]], curves[match["id"]]
        dr = pd.concat(
            [
                x.equity.resample("D").first().pct_change(fill_method=None).rename("max"),
                y.equity.resample("D").first().pct_change(fill_method=None).rename("control"),
            ],
            axis=1,
        ).dropna()
        active = x.gross_exposure.resample("D").max().shift(1).reindex(dr.index).gt(
            1e-8
        ) & y.gross_exposure.resample("D").max().shift(1).reindex(dr.index).gt(1e-8)
        common = dr.loc[active]
        corr = common["max"].corr(common.control) if len(common) >= 20 else None
        comparisons.append(
            {
                "account": row["id"],
                "control": match["id"],
                "return_difference_percentage_points": 100
                * (row["metrics"]["net_return"] - match["metrics"]["net_return"]),
                "joint_active_days": int(active.sum()),
                "joint_active_daily_correlation": float(corr)
                if corr is not None and pd.notna(corr)
                else None,
                "note": "Matched initial budget, opportunity set and engine; realised stops, inventory, exposure and cost paths differ. Difference is descriptive and does not prove an independent alpha mechanism.",
            }
        )
    source_comparisons = []
    for row in rows:
        if row["variant"] != "archive":
            continue
        peer = next(
            r
            for r in rows
            if r["variant"] == "api_revision"
            and r["candidate"] == row["candidate"]
            and r["phase"] == row["phase"]
            and r["capital"] == row["capital"]
            and r["slippage_bps"] == row["slippage_bps"]
        )
        scores_a = pd.read_parquet(
            inputs
            / f"archive_{row['phase'] if row['phase'] != 'capacity' else 'continuous2025_2026'}_{row['candidate']}_targets.parquet"
        )
        scores_b = pd.read_parquet(
            inputs
            / f"api_revision_{row['phase'] if row['phase'] != 'capacity' else 'continuous2025_2026'}_{row['candidate']}_targets.parquet"
        )
        target_cols = [c for c in scores_a if c.startswith("target_")]
        source_comparisons.append(
            {
                "archive_account": row["id"],
                "api_revision_account": peer["id"],
                "candidate": row["candidate"],
                "phase": row["phase"],
                "capital": row["capital"],
                "slippage_bps": row["slippage_bps"],
                "different_target_hours": int(
                    (scores_a[target_cols] != scores_b[target_cols]).any(axis=1).sum()
                ),
                "different_nav_observations": int(
                    (curves[row["id"]].equity != curves[peer["id"]].equity).sum()
                ),
                "differences_api_minus_archive": {
                    k: None
                    if row["metrics"].get(k) is None or peer["metrics"].get(k) is None
                    else peer["metrics"][k] - row["metrics"][k]
                    for k in [
                        "net_return",
                        "max_drawdown",
                        "sharpe",
                        "closed_trades",
                        "funding_net",
                        "fees",
                        "slippage_cost",
                    ]
                },
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "process_restart_audit.json", check)
    save(output / "execution_plan.json", plan)
    save(
        output / "provenance.json",
        {
            "run_completion_sha256": sha(run / "completion.json"),
            "run": str(run),
            "restart_audit_sha256": sha(restart / "audit.json"),
            "source_versions": plan["source_variants"],
        },
    )
    summary = {
        **source,
        "schema": "pure-funding-source-publication.v1",
        "study_kind": "funding_source",
        "study_description": "纯7日已结算资金费横截面价差及固定成本过滤；同期限动量、现金和两种行情版本并列；周频调仓，实际毫秒资金费",
        "candidate_names": NAMES,
        "account_rows": rows,
        "details": details,
        "control_comparisons": comparisons,
        "source_comparisons": source_comparisons,
        "source_versions": plan["source_variants"],
        "portfolio_actual": None,
        "expanded_universe_certified": False,
        "actual_accounts": source["new_accounts"],
        "reused_reference_accounts": 12,
        "selection": {**source["selection"], "research_followups": []},
        "restart_checks": check["checks"],
        "restart_note": "两种资金费方案×两种行情版本，各核验有仓断点和跨年断点，共8次独立解释器读取磁盘JSON恢复，净值、账本、请求与最终状态一致；保留跨年停机。",
        "model_comparison": {
            "available": False,
            "note": "本轮隔离资金费机制，不重训神经网络。保留原首轮MLP对照；不得把同一机制的两个过滤版本计为两个独立收益源。",
        },
        "execution": "同一冻结Freqtrade五分钟引擎、0.8初始总敞口、8%单仓止损、15%高水位停机、实际成交额1%容量上限；资金费按真实毫秒时间和mark结算。净值含开放周期和退出成本准备；交易收盘价净值不等于交易所mark或清算认证。",
        "changes_vs_previous": "统一因子目录新增纯资金费定义，单独检验融资价差。原40币归档不变，第二版本只覆盖2025-01-29的40币API修订；861条前期资金费补齐，不以名义三次/日替代实际日历。",
        "not_evaluated_reason": "容量只在连续账户两种行情、两成本均有正市值与闭合PnL且不停机时追加。未生成容量记录保持未知，不等于容量为零。",
        "next_action": "先核对2025失效与2026改善来自价格还是资金费，不能用2026冷启动盈利拼接跨年账户。40币回溯池及历史状态/终止结算限制仍未解决。",
        "actual_beta_note": "目标估计beta接近零不保证实际中性；实际库存估计暴露和日收益BTC回归分开展示。",
        "reporting_source_hashes": {
            str(Path(__file__)): sha(Path(__file__)),
            str(restart / "audit.json"): sha(restart / "audit.json"),
            str(states_root / "completion.json"): sha(states_root / "completion.json"),
            **bindings,
        },
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "summary_sha256": sha(output / "summary.json"),
            "audit_passed": True,
            "run_completion_sha256": sha(run / "completion.json"),
            "restart_audit_sha256": sha(restart / "audit.json"),
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        },
    )
    print(
        {
            "published": True,
            "new_accounts": source["new_accounts"],
            "rows": len(rows),
            "comparisons": len(comparisons),
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--restart", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    publish(a.run, a.restart, a.output)
