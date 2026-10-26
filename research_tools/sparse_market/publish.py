"""Publish paired sparse-account evidence without promoting failed modules or goals."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.regime_program.module_gate import assess_module
from research_tools.regime_program.reporting import partitions

NAMES = {
    "funding7_raw": "纯7日资金费",
    "funding7_cost_floor": "资金费＋固定成本门槛",
    "max28": "MAX28",
    "momentum28_control": "普通28日动量对照",
    "cash": "现金",
}
VERSIONS = [
    {"id": "daily_archive", "label": "原日线归档"},
    {"id": "daily_with_complete_5m_reconciliation", "label": "完整五分钟汇总差异情景"},
]


def publish(run, restart, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(run)
    verify_manifest(restart)
    source, plan = read(run / "summary.json"), read(run / "plan.json")
    restored = read(restart / "verification.json")
    if not restored["passed"] or restored["run_completion_sha256"] != sha(run / "completion.json"):
        raise ValueError("Actual account process restoration missing or changed")
    state_root = Path("results/research/regime_states_20261025_v002")
    verify_manifest(state_root)
    states = pd.read_parquet(state_root / "states_daily.parquet")
    event_flags = pd.read_parquet(
        state_root / "event_risk_hourly.parquet"
    ).covered_fomc_post_release
    inputs = Path(plan["inputs"])
    bundle = Path(plan["bundle"])
    rows, details, curves = [], {}, {}
    for row in source["account_rows"]:
        if not row["financial_audit_passed"]:
            missing = dict.fromkeys(
                [
                    "net_return",
                    "win_rate",
                    "max_drawdown",
                    "sharpe",
                    "profit_factor",
                    "closed_trades",
                    "fees",
                    "funding_net",
                    "mean_gross_exposure",
                ]
            )
            rows.append(
                {
                    **row,
                    "metrics": missing,
                    "variant": "common" if row["candidate"] == "cash" else row["variant"],
                    "capacity": {"filled_request_ratio": None},
                    "risk_halted": None,
                    "reused_reference": False,
                    "open_campaign_total_net_pnl_with_exit_reserve": None,
                }
            )
            continue
        folder = run / "accounts" / row["id"]
        account = read(folder / "result.json")
        if account["metrics"] != row["metrics"]:
            raise ValueError("Audited financial metrics changed")
        for name, digest in account["source_hashes"].items():
            if sha(folder / name) != digest:
                raise ValueError("Financial source receipt changed")
        curve = pd.read_parquet(folder / "equity_5m.parquet")
        detail = partitions(curve, account["trades"], states, event_flags, row["capital"])
        for group in ["regime", "covered_fomc"]:
            if (
                abs(
                    sum(
                        r["net_pnl_contribution"]
                        for r in detail["state_contributions"]
                        if r["group"] == group
                    )
                    - row["metrics"]["net_pnl"]
                )
                > 1e-5
            ):
                raise ValueError("Descriptive state or event contributions omit actual PnL")
        years = [r["net_return"] for r in detail["periods"] if r["kind"] == "year"]
        if abs(float(np.prod([1 + r for r in years])) - 1 - row["metrics"]["net_return"]) > 1e-10:
            raise ValueError("Yearly partitions reset or omit capital")
        reserve = 0.0
        for s, p in account["open_positions"].items():
            mark = pd.read_parquet(
                bundle / "symbols" / (s + "_5m.parquet"), columns=["close"]
            ).close.loc[pd.Timestamp(account["window"]["end"], tz="UTC") - pd.Timedelta(minutes=5)]
            reserve += (
                p["q"]
                * mark
                * (account["config"]["fee_bps"] + account["config"]["slippage_bps"])
                / 10000
            )
        m = row["metrics"]
        detail["same_executed_path_attribution"] = {
            "pre_fee_funding_slippage_price_pnl": m["net_pnl"]
            - m["funding_net"]
            + m["fees"]
            + m["slippage_cost"]
            + reserve,
            "funding_net": m["funding_net"],
            "fees": m["fees"],
            "slippage_cost": m["slippage_cost"],
            "open_exit_reserve": reserve,
        }
        details[row["id"]] = detail
        curves[row["id"]] = curve.set_index("date")
        rows.append(
            {
                **row,
                "variant": "common" if row["candidate"] == "cash" else row["variant"],
                "reused_reference": False,
                "result_sha256": sha(folder / "result.json"),
                "open_campaign_total_net_pnl_with_exit_reserve": m["open_campaign_net_pnl"],
            }
        )
    comparisons, assessments = [], []
    for row in rows:
        if row["variant"] != "daily_archive" or not row["financial_audit_passed"]:
            continue
        peer = next(
            r
            for r in rows
            if r["variant"] == VERSIONS[1]["id"]
            and r["candidate"] == row["candidate"]
            and r["phase"] == row["phase"]
            and r["slippage_bps"] == row["slippage_bps"]
        )
        if not peer["financial_audit_passed"]:
            continue
        left = pd.read_parquet(
            inputs / f"{row['variant']}_{row['phase']}_{row['candidate']}_targets.parquet"
        )
        right = pd.read_parquet(
            inputs / f"{peer['variant']}_{peer['phase']}_{peer['candidate']}_targets.parquet"
        )
        target_columns = [c for c in left if c.startswith("target_")]
        comparisons.append(
            {
                "archive_account": row["id"],
                "secondary_account": peer["id"],
                "candidate": row["candidate"],
                "phase": row["phase"],
                "slippage_bps": row["slippage_bps"],
                "different_target_hours": int(
                    (left[target_columns] != right[target_columns]).any(axis=1).sum()
                ),
                "different_nav_observations": int(
                    (curves[row["id"]].equity != curves[peer["id"]].equity).sum()
                ),
                "differences_secondary_minus_primary": {
                    k: None
                    if row["metrics"].get(k) is None or peer["metrics"].get(k) is None
                    else peer["metrics"][k] - row["metrics"][k]
                    for k in ["net_return", "max_drawdown", "sharpe", "funding_net", "fees"]
                },
            }
        )
    module_protocol = read(run / "protocol.json")["module_acceptance"]
    for variant in plan["source_variants"]:
        for candidate in [c for c in plan["candidates"] if c != "cash"]:
            for window in plan["windows"]:
                cells = [
                    r
                    for r in source["account_rows"]
                    if r["variant"] == variant
                    and r["candidate"] == candidate
                    and r["phase"] == window["id"]
                ]
                counts = [
                    (r["lifecycle_uncertainty"].get("by_entry_month") or {}).get("clusters", 0)
                    for r in cells
                    if r["financial_audit_passed"]
                ]
                gate_rows = [
                    {
                        **r,
                        "start": window["start"],
                        "end": window["end"],
                        "audit_passed": r["financial_audit_passed"],
                    }
                    for r in cells
                ]
                judged = assess_module(
                    gate_rows,
                    module_protocol,
                    domain_registered=True,
                    sleeve_capacity_passed=False,
                    cluster_groups=min(counts) if counts else None,
                )
                assessments.append(
                    {"variant": variant, "candidate": candidate, "phase": window["id"], **judged}
                )
    audit_rows = read(run / "financial_audits.json")
    save(output / "process_restart_audit.json", restored)
    save(output / "execution_plan.json", plan)
    save(
        output / "provenance.json",
        {
            "run": str(run),
            "run_completion_sha256": sha(run / "completion.json"),
            "restart_completion_sha256": sha(restart / "completion.json"),
            "cohort_completion_sha256": sha(Path(plan["cohort"]) / "completion.json"),
            "source_versions": plan["source_variants"],
        },
    )
    summary = {
        "schema": "sparse-identity-publication.v1",
        "study_kind": "sparse_identity",
        "actual_accounts": len(rows),
        "actual_strategy_accounts": 48,
        "actual_cash_accounts": 6,
        "reused_reference_accounts": 0,
        "financially_audited_accounts": len(audit_rows),
        "unknown_accounts": source["source_or_audit_unknown_accounts"],
        "study_description": "172名历史候选并集上的纯资金费、固定成本门槛、MAX与动量及现金；两种日线来源情景、两成本，支持身份分段和稀疏真实成交日历",
        "source_versions": plan["source_variants"],
        "source_options": VERSIONS,
        "candidate_names": NAMES,
        "account_rows": rows,
        "details": details,
        "source_comparisons": comparisons,
        "module_assessments": assessments,
        "default_candidate": "funding7_raw",
        "selection": {
            "research_followups": source["capacity_followup_queue"],
            "admitted_candidates": [],
        },
        "audit": {
            "passed": len(audit_rows) == len(rows),
            "accounts": len(audit_rows),
            "maximum_nav_error": max((r["max_nav_error"] for r in audit_rows), default=0),
        },
        "restart_checks": restored["checks"],
        "restart_note": f"{len(restored['checks'])}次真实独立解释器JSON恢复，分别核对有仓和跨年断点；净值、资金费、请求与最终高水位/停机一致。1日工程回归另存，不当作新增收益证据。",
        "model_comparison": {
            "available": False,
            "note": "本轮先验证收益源和历史身份，不重训神经网络；保留原规则、事件、HMM、线性与MLP对照。未取得两类经济机制前，不用模型调权掩盖单模块失效。",
        },
        "not_evaluated_reason": "容量只在连续账户两来源、两成本均有正市值及闭合PnL且不停机时触发；未触发的容量未知，不填0。没有单策略全年20%门槛。",
        "next_action": "先分离固定池、身份/观察期及执行修正对旧正收益的影响，再对事前可识别适用域提出有限、可反证的下一轮。描述性行情归因不能直接变成事后赢家路由。",
        "execution": "0.8初始总敞口、8%单仓止损、15%高水位停机、成交额1%容量代理；无量不成交，持仓缺价/未知资金费标记/未知终端现金即阻断。未平仓与退出准备计入净值；公开K线和资金费不是客户执行认证。",
        "changes_vs_previous": "LIT旧Litentry与新Lighter、PUMP旧PumpBTC与新Pump.fun分开累积年龄；零量日不计有效观察，12个股票/商品类型按公告可见时点处理。原54个初始化失败保留，修复仅是毫秒时间解析，因子、仓位与费用未改。",
        "source_note": "第二情景只采用实际完整五分钟汇总与日线归档不同的122个日期；同一执行行情并列比较，不按收益择源。两情景收益相同不代表所有来源一致。172候选不是178采集名称或原40币静态池。",
        "state_note": "行情描述沿用已冻结BTC状态标签，其中流动性项来自旧40池；只作描述，未据此重标新池或取得条件性收益准入。事件仅覆盖22份官方FOMC。",
        "portfolio_built": False,
        "portfolio_actual": None,
        "expanded_universe_certified": False,
        "profit_goal_met": False,
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
        "reporting_source_hashes": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path("research_tools/regime_program/reporting.py"),
                Path("research_tools/regime_program/module_gate.py"),
                state_root / "completion.json",
                run / "completion.json",
                restart / "completion.json",
            ]
        },
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "summary_sha256": sha(output / "summary.json"),
            "audit_passed": len(audit_rows) == len(rows),
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print(
        {
            "published_cases": len(rows),
            "financial_audits": len(audit_rows),
            "source_comparisons": len(comparisons),
            "restarts": len(restored["checks"]),
            "profit_goal_met": False,
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
