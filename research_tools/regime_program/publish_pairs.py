"""Publish audited pair accounts, frozen references, and scoped module verdicts."""

import argparse
import csv
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.regime_program.reporting import cluster_uncertainty, partitions, trade_stats

NAMES = {
    "pairs_fixed": "币对·2024固定形成期",
    "pairs_rolling": "币对·180日月度形成期",
    "cash": "现金",
    "trend": "首轮冻结静态趋势（引用）",
}


def sealed_run(root):
    receipt = read(root / "completion.json")
    for key, name in (
        ("summary_sha256", "summary.json"),
        ("plan_sha256", "plan.json"),
        ("selection_sha256", "selection.json"),
    ):
        if sha(root / name) != receipt[key]:
            raise ValueError("Run receipt changed: " + str(root))
    summary = read(root / "summary.json")
    if not summary["audit"]["passed"] or summary["audit"]["accounts"] != receipt["accounts"]:
        raise ValueError("Financial audit incomplete")
    return summary


def publish(run, output, docs):
    summary = sealed_run(run)
    plan = read(run / "plan.json")
    if not summary["restart_checks"] or not all(r["passed"] for r in summary["restart_checks"]):
        raise ValueError("Pair continuation incomplete")
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Pair source changed after run")
    protocol = read(Path(plan["formation_plan"]))
    states_root = Path(protocol["states"])
    for name, digest in read(states_root / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(states_root / name) != digest:
            raise ValueError("State inputs changed")
    states = pd.read_parquet(states_root / "states_daily.parquet")
    flags = pd.read_parquet(states_root / "event_risk_hourly.parquet").covered_fomc_post_release
    parent = Path(plan["parent_controls"])
    parent_summary = sealed_run(parent)
    control = parent_summary["selection"]["best_static"]
    if control != "trend":
        raise ValueError("Frozen reference selection changed")
    rows = []
    detail = {}
    reference_rows = []
    needed = sorted(
        {(r["phase"], r["capital"], r["slippage_bps"]) for r in summary["account_rows"]}
    )
    source_rows = [(run, r, False) for r in summary["account_rows"]]
    for phase, capital, cost in needed:
        matches = [
            r
            for r in parent_summary["account_rows"]
            if r["phase"] == phase
            and r["capital"] == capital
            and r["slippage_bps"] == cost
            and r["candidate"] == control
        ]
        if len(matches) != 1:
            raise ValueError("Missing matching static reference")
        source_rows.append((parent, matches[0], True))
    for root, row, reused in source_rows:
        folder = root / "accounts" / row["id"]
        if sha(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Account changed")
        account = read(folder / "result.json")
        for name, digest in account["source_hashes"].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError("Ledger changed")
        if account["metrics"] != row["metrics"]:
            raise ValueError("Account metric receipt differs")
        curve = pd.read_parquet(folder / "equity_5m.parquet")
        stats = trade_stats(account["trades"])
        positions = account["open_positions"] if reused else account["open_pairs"]
        settled = sum(p["net_pnl"] for p in positions.values())
        open_total = settled + row["metrics"]["open_pnl_with_exit_reserve"]
        if (
            abs(
                stats["closed_net_pnl"] + open_total - row["capital"] * row["metrics"]["net_return"]
            )
            > 1e-5
        ):
            raise ValueError("Closed/open pair settlement does not reconcile")
        identity = ("reference_" if reused else "") + row["id"]
        record = {
            **row,
            "id": identity,
            "reused_reference": reused,
            "source_account_id": row["id"],
            "source_run": str(root.resolve()),
            "lifecycle": stats,
            "lifecycle_unit": "single-symbol campaign"
            if reused
            else "pair campaign, failed hedges included",
            "open_campaign_settled_net_pnl": settled,
            "open_campaign_total_net_pnl_with_exit_reserve": open_total,
        }
        rows.append(record)
        if reused:
            reference_rows.append(identity)
        detail[identity] = {
            **partitions(curve, account["trades"], states, flags, row["capital"]),
            "lifecycle_uncertainty": cluster_uncertainty(account["trades"]),
        }
        for group in ("regime", "covered_fomc"):
            pnl = sum(
                t["net_pnl_contribution"]
                for t in detail[identity]["state_contributions"]
                if t["group"] == group
            )
            if abs(pnl - row["capital"] * row["metrics"]["net_return"]) > 1e-5:
                raise ValueError("State attribution differs")
    interpretations = []
    for candidate in ("pairs_fixed", "pairs_rolling"):
        cases = [
            r for r in rows if r["candidate"] == candidate and r["phase"] == "continuous2025_2026"
        ]
        positive = all(r["metrics"]["closed_net_pnl"] > 0 for r in cases)
        capacity = [
            r
            for r in rows
            if r["candidate"] == candidate and r["phase"] == "capacity" and r["capital"] == 100000
        ]
        cap_ok = all(
            r["metrics"]["net_return"] > 0
            and r["metrics"]["max_drawdown"] <= 0.15
            and not r["risk_halted"]
            for r in capacity
        )
        interpretations.append(
            {
                "candidate": candidate,
                "status": "evidence_insufficient"
                if positive
                else "no_net_edge_in_declared_formation_domain",
                "scope": "The preregistered formation domain; not a verdict about all possible market regimes",
                "minimum_two_cost_matched_pairs": min(
                    r["metrics"]["matched_closed_pairs"] for r in cases
                ),
                "closed_pnl_positive_both_costs": positive,
                "capacity_100k_passed": cap_ok,
                "eligible_for_combination": False,
                "fixed_annual20percent_is_module_gate": False,
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    docs.mkdir(parents=True, exist_ok=False)
    published = {
        "schema": "two-leg-pair-publication.v1",
        "study_kind": "pairs",
        "title": "第三轮：币对双腿成本与容量验证",
        "account_rows": rows,
        "details": detail,
        "candidate_names": NAMES,
        "actual_accounts": len(summary["account_rows"]),
        "reused_reference_accounts": len(reference_rows),
        "audit": summary["audit"],
        "restart_checks": summary["restart_checks"],
        "input_validation": summary["input_validation"],
        "timestamp_repair_reproduced_accounts": summary["timestamp_repair_reproduced_accounts"],
        "module_interpretations_v2": interpretations,
        "legacy_v1_selection": summary["selection"],
        "legacy_v1_economics": summary["economics"],
        "frozen_reference": {
            "candidate": control,
            "accounts": reference_rows,
            "parent_completion_sha256": sha(parent / "completion.json"),
        },
        "execution": "双腿独立次根5分钟撮合，包含双腿费用、实际资金费及退出准备；比例不匹配时下根开始退出，直到两腿平仓才释放币种；失败对冲计入损益但不填充匹配样本门槛。",
        "changes_vs_previous": "新收益机制与配对生命周期；不继承先前单腿加减仓语义。旧20%逐策略筛查保留为历史，新解释只按既定形成域的净优势、样本和容量判断，未事后选盈利行情。",
        "profit_goal_met": False,
        "account_returns_computed": True,
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
        "data_limitations": [
            "Retrospectively selected40-name pool; previously viewed2025/2026history.",
            "Quote-volume participation proxy, not historical order book or venue lot/minimum-order certification.",
            "Single-interval submission volume caps can leave substantial cash idle; low drawdown is not sufficient evidence of skill.",
            "A portfolio of different strategies is not yet built or validated by these separate accounts.",
        ],
    }
    save(output / "summary.json", published)
    save(
        output / "completion.json",
        {
            "summary_sha256": sha(output / "summary.json"),
            "audit_passed": True,
            "account_run_completion_sha256": sha(run / "completion.json"),
            "account_returns_computed": True,
        },
    )
    save(docs / "summary.json", published)
    save(docs / "execution_plan.json", plan)
    save(
        docs / "provenance.json",
        {
            "run": str(run.resolve()),
            "publication": str(output.resolve()),
            "run_completion_sha256": sha(run / "completion.json"),
            "publication_completion_sha256": sha(output / "completion.json"),
        },
    )
    fields = [
        "id",
        "candidate",
        "phase",
        "capital",
        "slippage_bps",
        "reused_reference",
        "lifecycle_unit",
        "net_return",
        "cagr",
        "win_rate",
        "max_drawdown",
        "sharpe",
        "profit_factor",
        "closed_trades",
        "matched_closed_pairs",
        "failed_hedge_closed_attempts",
        "fees",
        "slippage_cost",
        "funding_net",
        "mean_gross_exposure",
        "max_absolute_net_exposure",
        "filled_request_ratio",
        "hedge_failures",
        "max_exit_delay_minutes",
        "risk_halted",
    ]
    with (docs / "accounts.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            values = {**r, **r["metrics"], **r["capacity"]}
            writer.writerow({k: values.get(k) for k in fields})

    def value(v, percent=False):
        return "—" if v is None else f"{v * 100:.2f}%" if percent else f"{v:.2f}"

    text = [
        "# 币对双腿账户结果与分层目标解释",
        "",
        "2026-10-26。完成26个新账户，另引用10个相同资本/窗口/成本的首轮冻结趋势对照；这些成本变体、引用对照不当成独立样本。没有已验证的多策略总账户，也没有生产资格。",
        "本轮全部财务核算通过，4次实际JSON断点/跨年恢复一致，其中2次断点有真实持仓。时间接口修复前的v001未完成运行保留，两个已生成账户的指标/账本/请求/最终状态与v002精确一致，参数未改。",
        "",
        "## 全部窗口、成本与容量",
        "",
        "每边手续费5bps，加3或10bps滑点，含实际资金费、开放浮亏和退出准备。下表为整段总收益，不是年化。完整周期为一个配对尝试，失败对冲也计入净胜率；匹配周期单列，不能把两腿当两次胜利。引用趋势对照的周期单位是单币，与配对周期不可当作独立等价样本相加。",
        "",
        "|阶段|策略|本金USDT|每边成本bps|总净收益|净胜率|回撤|夏普|闭合周期|匹配配对|失败对冲|平均总敞口|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        m = r["metrics"]
        text.append(
            f"|{r['phase']}|{NAMES[r['candidate']]}|{r['capital']}|{r['slippage_bps'] + 5}|{value(m['net_return'], True)}|{value(m['win_rate'], True)}|{value(m['max_drawdown'], True)}|{value(m['sharpe'])}|{m['closed_trades']}|{m.get('matched_closed_pairs', '—')}|{r['capacity'].get('hedge_failures', '—')}|{value(m['mean_gross_exposure'], True)}|"
        )
    text += [
        "",
        "## 结论的适用范围",
        "",
        "按当前已登记形成规则，固定方案连续608日高成本总收益-1.35%；滚动方案+1.64%、回撤3.01%、夏普0.35，只有5个匹配周期。滚动方案在2025盈利3.16%，2026冷启动亏损1.45%；不能把前者当跨环境有效证据。10万容量高成本转为-0.1408%，有一次对冲比例失败。低回撤还伴随低资金利用率，不是充分风险调整优势证据。",
        "用户纠正后，单模块不再要求全年20%或全部行情都盈利。滚动方案当前归类为证据不足，固定方案只说明既定形成域没有净优势；不能扩大为所有价差回归都无效。两者均不具备组合准入证据。旧v001按全年20%筛查的失败原样保留，未事后按盈利环境重新过滤旧结果。",
        "每年/90日块/因果行情/FOMC窗口/聚类不确定性见summary.details。行情贡献是同一账户每5分钟NAV变化之和；生命周期按入场或退出时点分组，不能当作该时段按市值收益。少于4个聚类组时区间保留未知。",
        "开放配对保留已平腿结算损益，闭合周期+开放配对累计结算+浮动损益及退出准备与账户损益对账。不能只汇总仍持有的一条腿。",
        "执行仍是历史成交额参与率代理，非盘口/订单认证。开仓只在一个5分钟区间提交，实际流动性上限会令大额本金闲置；需要另行登记分段建仓及对冲失败保护，不能直接放大名义仓位改写本轮结果。",
        "40币池有事后选择偏差，历史已看过；无新的独立样本外或实盘证明。当前不再扩同组形成期参数网格；后续先按v002形成明确适用域与失效策略卡，再固定新实验。",
        "",
        "## 证据",
        "",
        "[全部账户表](accounts.csv)、[摘要及完整分期](summary.json)、[执行协议](execution_plan.json)、[溯源](provenance.json)。",
        "",
        "基金方法和三级目标见 [研究纠偏](../fund_methods_20261026/README.md)。原始数据/账本/恢复状态保存在TradingAgents results/research/pair_accounts_20261026_v002；没有改写此前封存证据。",
    ]
    (docs / "README.md").write_text("\n".join(text) + "\n")
    save(
        docs / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in docs.iterdir() if p.is_file()},
            "publication_completion_sha256": sha(output / "completion.json"),
            "audit_passed": True,
        },
    )
    print(
        {
            "published": str(output),
            "new_accounts": 26,
            "references": len(reference_rows),
            "module_interpretations": interpretations,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True)
    args = parser.parse_args()
    publish(args.run.resolve(), args.output.resolve(), args.docs.resolve())
