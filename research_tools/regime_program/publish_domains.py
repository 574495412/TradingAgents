"""Seal conditional-domain comparisons, actual risk paths and failed hypotheses."""

import argparse
import csv
import itertools
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.regime_program.reporting import cluster_uncertainty, partitions, trade_stats

NAMES = {
    "trend_domain": "趋势·趋势域",
    "relative_domain": "相对动量·非波动扩张域",
    "carry_domain": "资金费＋动量·震荡域",
    "price_reversion_domain": "极端回归·震荡域",
    "breakout_domain": "突破·波动扩张域",
    "trend": "全时趋势（对照）",
    "relative_momentum": "全时相对动量（对照）",
    "carry_momentum": "全时资金费＋动量（对照）",
    "mean_reversion": "全时极端回归（对照）",
    "breakout": "全时突破（对照）",
    "cash": "现金",
}


def publish(run, output, docs):
    receipt = read(run / "completion.json")
    for key, name in (
        ("summary_sha256", "summary.json"),
        ("plan_sha256", "plan.json"),
        ("selection_sha256", "selection.json"),
    ):
        if sha(run / name) != receipt[key]:
            raise ValueError("Domain run completion changed")
    source = read(run / "summary.json")
    plan = read(run / "plan.json")
    if (
        not source["audit"]["passed"]
        or source["audit"]["accounts"] != source["new_accounts"]
        or not all(r["passed"] for r in source["restart_checks"])
    ):
        raise ValueError("Domain financial/restart audit incomplete")
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Domain source changed after run")
    states_root = Path(plan["states"])
    for name, digest in read(states_root / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(states_root / name) != digest:
            raise ValueError("State evidence changed")
    states = pd.read_parquet(states_root / "states_daily.parquet")
    flags = pd.read_parquet(states_root / "event_risk_hourly.parquet").covered_fomc_post_release
    cards = {c["id"]: c for c in read(Path(plan["registry_snapshot"]))["cards"]}
    rows = []
    details = {}
    returns = {}
    activity = {}
    for row in source["account_rows"]:
        folder = Path(row["source_run"]) / "accounts" / row["source_account_id"]
        if sha(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Domain/reference account changed")
        account = read(folder / "result.json")
        for name, digest in account["source_hashes"].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError("Account ledger changed")
        if account["metrics"] != row["metrics"]:
            raise ValueError("Account metric differs")
        curve = pd.read_parquet(folder / "equity_5m.parquet")
        settled = sum(p["net_pnl"] for p in account["open_positions"].values())
        open_total = settled + row["metrics"]["open_pnl_with_exit_reserve"]
        life = trade_stats(account["trades"])
        if (
            abs(life["closed_net_pnl"] + open_total - row["capital"] * row["metrics"]["net_return"])
            > 1e-5
        ):
            raise ValueError("Domain closed/open PnL does not reconcile")
        entry_outside = 0
        if row["kind"] == "module":
            allowed = set(cards[row["candidate"]]["allowed_states"])
            entry_outside = sum(
                states.state.asof(pd.Timestamp(e["submitted_at"])) not in allowed
                for e in read(folder / "events.json")
                if e["type"] == "entry"
            )
            if entry_outside:
                raise ValueError("Entry outside registered domain")
        extended = {
            **row,
            "lifecycle": life,
            "open_campaign_settled_net_pnl": settled,
            "open_campaign_total_net_pnl_with_exit_reserve": open_total,
            "entries_outside_registered_domain": entry_outside,
        }
        rows.append(extended)
        details[row["id"]] = {
            **partitions(curve, account["trades"], states, flags, row["capital"]),
            "lifecycle_uncertainty": cluster_uncertainty(account["trades"]),
        }
        for group in ("regime", "covered_fomc"):
            pnl = sum(
                x["net_pnl_contribution"]
                for x in details[row["id"]]["state_contributions"]
                if x["group"] == group
            )
            if abs(pnl - row["capital"] * row["metrics"]["net_return"]) > 1e-5:
                raise ValueError("Domain attribution does not reconcile")
        if row["kind"] == "module" and row["phase"] == "screen2025":
            indexed = curve.set_index("date")
            daily = indexed.equity.resample("D").first().pct_change(fill_method=None).dropna()
            key = (row["slippage_bps"], row["candidate"])
            returns[key] = daily
            activity[key] = (
                indexed.gross_exposure.resample("D").max().shift(1).reindex(daily.index).gt(0)
            )
    correlations = []
    for cost in plan["costs_slippage_bps"]:
        for a, b in itertools.combinations(plan["screen_modules"], 2):
            joined = pd.concat(
                [returns[(cost, a)].rename("a"), returns[(cost, b)].rename("b")], axis=1
            ).dropna()
            active = activity[(cost, a)].reindex(joined.index) & activity[(cost, b)].reindex(
                joined.index
            )
            common = joined.loc[active]
            value = joined.a.corr(joined.b)
            overlap = common.a.corr(common.b) if len(common) >= 20 else None
            correlations.append(
                {
                    "slippage_bps": cost,
                    "a": a,
                    "b": b,
                    "calendar_days": len(joined),
                    "calendar_correlation": float(value) if pd.notna(value) else None,
                    "joint_active_days": int(active.sum()),
                    "joint_active_correlation": float(overlap)
                    if overlap is not None and pd.notna(overlap)
                    else None,
                    "note": "Inactive cash periods may mechanically reduce calendar correlation; fewer20joint-active days remains unknown, not zero correlation.",
                }
            )
    selection = source["selection"]
    summary = {
        "schema": "conditional-domain-publication.v1",
        "study_kind": "conditional_domains",
        "study_description": "保持原公式与仓位，仅按事前可观察适用域启停；子策略不设全年20%年化门槛",
        "candidate_names": NAMES,
        "account_rows": rows,
        "details": details,
        "selection": selection,
        "module_assessments": source["module_assessments"],
        "actual_accounts": source["new_accounts"],
        "reused_reference_accounts": source["reused_references"],
        "maximum_accounts": plan["budgets"]["total_new_accounts_max"],
        "audit": source["audit"],
        "restart_checks": source["restart_checks"],
        "restart_note": f"{len(source['restart_checks'])}个高成本2025模块账户经过JSON状态恢复，全部净值、账本、请求和最终状态一致；断点均按真实首笔入场选择，不能作为盈利证据。",
        "model_comparison": {
            "available": False,
            "note": "本轮只检验已冻结策略的可观察适用域，不重训或调神经网络；没有从2026收益反选方案。",
        },
        "execution": "沿用原加减仓/五分钟容量账户。禁用域仅请求目标为零；旧库存按真实费用和流动性退出，高水位/风险停机不重置。",
        "changes_vs_previous": "只改变事前登记的启停条件；没有改公式、阈值、归一化或风险预算。对照为原全时策略和现金，资金利用率差异须一并判断。",
        "mechanism_groups": plan["mechanism_groups"],
        "module_correlations": correlations,
        "not_evaluated_reason": "未有两成本同时正净收益与正闭合PnL的2025候选，因此未生成后续、跨年或容量账户。"
        if not selection["research_followups"]
        else None,
        "next_action": "停止这组既有价格规则加宽BTC状态过滤的重复搜索。资金费收入为正但成本覆盖不足；下一步先核查更低换手的结构性资金费/基差收益及真实现货对冲数据，不更改本轮参数或事后挑盈利状态。",
        "profit_goal_met": False,
        "portfolio_built": False,
        "independent_oos": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    docs.mkdir(parents=True, exist_ok=False)
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "summary_sha256": sha(output / "summary.json"),
            "account_run_completion_sha256": sha(run / "completion.json"),
            "audit_passed": True,
        },
    )
    save(docs / "summary.json", summary)
    save(docs / "selection.json", selection)
    save(docs / "execution_plan.json", plan)
    save(docs / "input_validation.json", source["input_audit"])
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
        "kind",
        "phase",
        "capital",
        "slippage_bps",
        "reused_reference",
        "net_return",
        "cagr",
        "win_rate",
        "max_drawdown",
        "sharpe",
        "profit_factor",
        "closed_trades",
        "fees",
        "slippage_cost",
        "funding_net",
        "mean_gross_exposure",
        "risk_halted",
    ]
    with (docs / "accounts.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            values = {**row, **row["metrics"]}
            writer.writerow({k: values.get(k) for k in fields})

    def val(v, pct=False):
        return "—" if v is None else f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    lines = [
        "# 条件适用域首轮账户结果",
        "",
        "2026-10-26。沿用v002三级目标，子策略按适用域的扣费优势筛查，不要求单策略全年20%。本轮只改变事前登记的状态启停，不改公式、归一化、门槛或风险。历史已查看，不是新的独立样本外。",
        "",
        f"完成{source['new_accounts']}个新模块账户，另引用{source['reused_references']}个封存全时/现金对照。全部独立核算通过，净值最大误差{source['audit']['maximum_nav_error']:.3g}USDT；5次真实首笔入场断点JSON恢复一致。没有候选在两档成本下同时取得正净收益与正闭合PnL，后续名单为空；不为凑策略强选赢家，也未生成2026/跨年/容量数据。",
        "",
        "## 2025连续365日，1万USDT",
        "",
        "每边5bps手续费加3或10bps滑点；包含实际资金费、浮亏、退出准备与跨状态真实退出。下表为整段净收益；两成本会改变仓位与停机路径，不能把收益差额全当作费用。",
        "",
        "|方案|每边成本bps|整段净收益|净胜率|最大回撤|夏普|完整周期|手续费|滑点|资金费净额|停机|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"|{NAMES[row['candidate']]}|{row['slippage_bps'] + 5}|{val(m['net_return'], True)}|{val(m['win_rate'], True)}|{val(m['max_drawdown'], True)}|{val(m['sharpe'])}|{m['closed_trades']}|{val(m['fees'])}|{val(m['slippage_cost'])}|{val(m['funding_net'])}|{'是' if row['risk_halted'] else '否'}|"
        )
    lines += [
        "",
        "## 解释与下一步",
        "",
        "本轮反驳的是这组旧价量规则配合粗粒度BTC状态启停的具体假设，不能扩展成不同环境适用不同策略这条主线无效。全市场BTC状态并不保证单币或策略自身的方向匹配，退出时的滑点和持仓延续也会影响结果。",
        "",
        "资金费＋动量在低成本下有正资金费收入和净收益，但压力成本下亏损并停机。资金费收入不能忽略换手费用；低成本823个生命周期和高成本261个生命周期是不同执行路径，不能按两者收益差直接估算费用损耗，也不把复合排序等同纯资金费套利。",
        "",
        "价格回归只出现3个完整周期，且成本后亏损。稀疏性和负结果都保留，不调低原3σ阈值造交易。全部域外新入场计数为0；域外仍可能有旧库存退出或浮亏，不能把这些损失从策略账上删掉。",
        "",
        "按币/入场月不确定性、自然年/90日块、因果状态/FOMC贡献在summary.details。模块相关性同时列日历值与共同持仓日；现金时段可能机械降低相关性，共同活跃少于20天则保留未知，不能据此宣称独立收益源。趋势、相对动量、突破也不能仅凭名字就算三种独立经济机制。",
        "",
        summary["next_action"],
        "",
        "共同账户组合尚未建立，20%年化/15%回撤/夏普1的总账户目标仍未完成。没有盘口或实盘认证，容量仍是成交额参与率代理。",
        "",
        "[全部账户表](accounts.csv)、[全部分期/状态/相关性](summary.json)、[冻结后续名单](selection.json)、[输入核验](input_validation.json)、[执行协议](execution_plan.json)。",
    ]
    (docs / "README.md").write_text("\n".join(lines) + "\n")
    save(
        docs / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in docs.iterdir() if p.is_file()},
            "publication_completion_sha256": sha(output / "completion.json"),
        },
    )
    print(
        {
            "published": str(output),
            "new_accounts": source["new_accounts"],
            "reused": source["reused_references"],
            "followups": selection["research_followups"],
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
