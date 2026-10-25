"""Publish complete cost/regime evidence, including failures and unseen metrics."""

import argparse
import csv
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.regime_program.reporting import cluster_uncertainty, partitions, trade_stats

NAMES = {
    "trend": "趋势 EWMAC32/128",
    "relative_momentum": "相对动量21日",
    "carry_momentum": "资金费＋动量21日",
    "mean_reversion": "极端回归 mr_wings4",
    "breakout": "通道突破10日",
    "cash": "现金",
    "equal_mix": "五策略等权",
    "rule_router": "规则行情配置",
    "event_router": "规则＋FOMC风控",
    "hmm_router": "HMM行情配置",
    "linear_router": "线性方向风险覆盖",
    "mlp_router": "MLP方向风险覆盖",
}


def publish(run, output, docs):
    output.mkdir(parents=True, exist_ok=False)
    docs.mkdir(parents=True, exist_ok=False)
    receipt = read(run / "completion.json")
    if (
        sha(run / "summary.json") != receipt["summary_sha256"]
        or sha(run / "plan.json") != receipt["plan_sha256"]
        or sha(run / "selection.json") != receipt["selection_sha256"]
    ):
        raise ValueError("Run completion receipts differ")
    summary = read(run / "summary.json")
    if not summary["audit"]["passed"] or summary["audit"]["accounts"] != receipt["accounts"]:
        raise ValueError("Independent financial audit incomplete")
    plan = read(run / "plan.json")
    names = {**NAMES, **plan.get("candidate_names", {})}
    states_root = Path(plan["states"])
    for filename, digest in read(states_root / "completion.json")["manifest"].items():
        if Path(filename).name != filename or sha(states_root / filename) != digest:
            raise ValueError("Sealed state attribution inputs changed")
    states = pd.read_parquet(states_root / "states_daily.parquet")
    flags = pd.read_parquet(states_root / "event_risk_hourly.parquet").covered_fomc_post_release
    rows = []
    detail = {}
    for row in summary["account_rows"]:
        folder = run / "accounts" / row["id"]
        if sha(folder / "result.json") != row["result_sha256"]:
            raise ValueError("Published account identity changed")
        account = read(folder / "result.json")
        for name, digest in account["source_hashes"].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError("Account ledger evidence changed")
        curve = pd.read_parquet(folder / "equity_5m.parquet")
        stats = trade_stats(account["trades"])
        open_settlement = sum(p["net_pnl"] for p in account["open_positions"].values())
        open_total = open_settlement + row["metrics"]["open_pnl_with_exit_reserve"]
        if (
            abs(
                stats["closed_net_pnl"] + open_total - row["capital"] * row["metrics"]["net_return"]
            )
            > 1e-5
        ):
            raise ValueError("Closed/open campaign PnL does not reconcile to account")
        extended = {
            **row,
            "lifecycle": stats,
            "open_campaign_settled_net_pnl": open_settlement,
            "open_campaign_total_net_pnl_with_exit_reserve": open_total,
        }
        rows.append(extended)
        detail[row["id"]] = {
            **partitions(curve, account["trades"], states, flags, row["capital"]),
            "lifecycle_uncertainty": cluster_uncertainty(account["trades"]),
        }
        for group in ("regime", "covered_fomc"):
            pnl = sum(
                x["net_pnl_contribution"]
                for x in detail[row["id"]]["state_contributions"]
                if x["group"] == group
            )
            if abs(pnl - row["capital"] * row["metrics"]["net_return"]) > 1e-5:
                raise ValueError("State attribution does not reconcile")
    inputs = Path(plan["inputs"])
    model_comparison = {"available": False, "target_different_dates": None, "cost_cases": []}
    if (inputs / "linear_router_targets.parquet").is_file():
        linear = pd.read_parquet(inputs / "linear_router_targets.parquet")
        neural = pd.read_parquet(inputs / "mlp_router_targets.parquet")
        different = linear.ne(neural).any(axis=1)
        model_comparison = {
            "available": True,
            "target_different_hourly_rows": int(different.sum()),
            "target_different_dates": len(set(different.index[different].date)),
            "first_target_difference": different.index[different][0].isoformat()
            if different.any()
            else None,
            "note": "Different probability or target signals do not imply executed advantage; halted accounts cannot trade later differences.",
            "cost_cases": [],
        }
        for cost in (3, 10):
            a = next(
                r
                for r in rows
                if r["phase"] == "screen2025"
                and r["candidate"] == "linear_router"
                and r["slippage_bps"] == cost
            )
            b = next(
                r
                for r in rows
                if r["phase"] == "screen2025"
                and r["candidate"] == "mlp_router"
                and r["slippage_bps"] == cost
            )
            ea = read(run / "accounts" / a["id"] / "events.json")
            eb = read(run / "accounts" / b["id"] / "events.json")
            halt = [e["submitted_at"] for e in ea if e.get("reason") == "risk_halt"]
            model_comparison["cost_cases"].append(
                {
                    "slippage_bps": cost,
                    "executed_ledger_equal": ea == eb,
                    "first_risk_halt": min(halt) if halt else None,
                    "net_return_difference": b["metrics"]["net_return"]
                    - a["metrics"]["net_return"],
                }
            )
    selection = summary["selection"]
    representative = [
        r
        for r in rows
        if r["phase"] == "continuous2025_2026" and r["candidate"] == selection["best_static"]
    ]
    scorecard = {
        "candidate": selection["best_static"],
        "phase": "continuous2025_2026",
        "minimum_two_cost_cagr": min(r["metrics"]["cagr"] for r in representative),
        "maximum_two_cost_drawdown": max(r["metrics"]["max_drawdown"] for r in representative),
        "minimum_two_cost_sharpe": min(
            r["metrics"]["sharpe"] for r in representative if r["metrics"]["sharpe"] is not None
        ),
    }
    frozen_research = selection["research_followups"][0]
    research_cases = [
        r for r in rows if r["phase"] == "continuous2025_2026" and r["candidate"] == frozen_research
    ]
    research_scorecard = {
        "candidate": frozen_research,
        "phase": "continuous2025_2026",
        "minimum_two_cost_cagr": min(r["metrics"]["cagr"] for r in research_cases),
        "maximum_two_cost_drawdown": max(r["metrics"]["max_drawdown"] for r in research_cases),
        "minimum_two_cost_sharpe": min(
            (r["metrics"]["sharpe"] for r in research_cases if r["metrics"]["sharpe"] is not None),
            default=None,
        ),
        "chosen_before_later_windows": True,
    }
    control_reproduction = None
    if plan.get("parent_accounts"):
        parent = Path(plan["parent_accounts"])
        parent_receipt = read(parent / "completion.json")
        if sha(parent / "summary.json") != parent_receipt["summary_sha256"]:
            raise ValueError("Matched-control parent receipt changed")
        checked = []
        for control in plan["static_families"]:
            for cost in plan["costs_slippage_bps"]:
                identity = f"screen2025_{control}_cap10000_slip{cost}"
                fresh, old = run / "accounts" / identity, parent / "accounts" / identity
                if read(fresh / "result.json")["metrics"] != read(old / "result.json")["metrics"]:
                    raise ValueError("Unbuffered matched-control metrics changed")
                for filename in (
                    "events.json",
                    "trades.json",
                    "requests.json",
                    "resume_state.json",
                ):
                    if read(fresh / filename) != read(old / filename):
                        raise ValueError("Unbuffered control economic state changed")
                checked.append(identity)
        control_reproduction = {
            "passed": True,
            "cases": checked,
            "count": len(checked),
            "parent_completion_sha256": sha(parent / "completion.json"),
        }
    data = {
        "schema": "multi-regime-publication.v1",
        "study_kind": "forecast_sizing" if "forecast-strength" in plan["schema"] else "regime",
        "study_description": (
            "检验预测幅度、源码默认16%波动预算和10%仓位缓冲；保留三种原始对照，目标总敞口上限80%，不增加杠杆"
            if "forecast-strength" in plan["schema"]
            else "五类策略与规则、事件及模型配置的共同账户验证"
        ),
        "screen_candidates": len(plan["candidates"]),
        "restart_candidate": plan.get("restart_candidate", "event_router"),
        "candidate_names": names,
        "account_rows": rows,
        "details": detail,
        "selection": selection,
        "representative_frozen_static": scorecard,
        "representative_frozen_research": research_scorecard,
        "control_reproduction": control_reproduction,
        "audit": summary["audit"],
        "restart_checks": summary["restart_checks"],
        "model_comparison": model_comparison,
        "execution": plan["execution"],
        "changes_vs_previous": plan["changes_vs_previous"],
        "maximum_accounts": 60,
        "actual_accounts": len(rows),
        "profit_goal_met": False,
        "independent_oos": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "summary.json", data)
    save(
        output / "completion.json",
        {
            "summary_sha256": sha(output / "summary.json"),
            "account_run_completion_sha256": sha(run / "completion.json"),
            "audit_passed": True,
        },
    )
    save(docs / "summary.json", data)
    save(docs / "selection.json", selection)
    save(docs / "execution_plan.json", plan)
    save(
        docs / "provenance.json",
        {
            "run": str(run),
            "publication": str(output),
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
        "net_return",
        "cagr",
        "win_rate",
        "max_drawdown",
        "sharpe",
        "profit_factor",
        "closed_trades",
        "closed_net_pnl",
        "fees",
        "slippage_cost",
        "funding_net",
        "open_pnl_with_exit_reserve",
        "open_campaign_total_net_pnl_with_exit_reserve",
        "filled_request_ratio",
        "risk_halted",
    ]
    with (docs / "accounts.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = {**row, **row["metrics"], **row["capacity"]}
            writer.writerow({k: values.get(k) for k in fields})
    lines = [
        "# 多行情、多策略连续账户首轮结果",
        "",
        "所有输出为已查看历史的研究回放；未达到基础盈利目标，没有实盘或新样本外结论。",
        f"共{len(rows)}个唯一账户（最多60）：2025筛选后按事前规则冻结三个研究后续与静态对照，角色重叠复用同一账户，不重复计算证据。全部通过独立现金／净值／资金费／完整生命周期审计。",
        "",
        "## 1万USDT与逐资金容量",
        "",
        "费用为5bps手续费＋3／10bps滑点每边，包含真实资金费事件、浮亏与退出准备；CAGR只在足够长的连续账户用于目标验收。下表为整段总收益，不是月收益。",
        "",
        "|阶段|策略|本金|单边总成本bps|整段净收益|完整胜率|最大观察回撤|夏普|闭合PF|完整周期|停机|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def value(v, pct=False):
        return "—" if v is None else f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    for r in rows:
        m = r["metrics"]
        lines.append(
            f"|{r['phase']}|{names[r['candidate']]}|{r['capital']}|{r['slippage_bps'] + 5}|{value(m['net_return'], True)}|{value(m['win_rate'], True)}|{value(m['max_drawdown'], True)}|{value(m['sharpe'])}|{value(m['profit_factor'])}|{m['closed_trades']}|{'是' if r['risk_halted'] else '否'}|"
        )
    lines += [
        "",
        "## 重要解释",
        "",
        "策略意图在过去完整日生成，只有事件覆盖系数变化在小时级触发额外调仓。执行按真实已释放风险额度部分成交；同币反向先退出，下一次既定调仓才重新开反向。仍是五分钟成交额代理，不能冒充盘口或原生交易所认证。",
        "新引擎允许在同一方向完整生命周期内加减仓，保持加权成本和累计现金损益；开放浮动PnL之外，另外列开放生命周期累计结算和含退出准备总损益，完整周期与开放周期之和与账户损益对账。",
        "连续跨年账户保持停机、资金费和净值高点，2026冷启动的盈利不能拼到2025已停机账户。2025/2026均是已查看历史；未入选的模型不运行后续窗口，不补造收益。",
        (
            f"线性与MLP在{model_comparison['target_different_dates']}个日期目标权重不同；本次两个成本情景的实际账本是否一致分别为{[r['executed_ledger_equal'] for r in model_comparison['cost_cases']]}，停机时点与首个差异可在summary.model_comparison核查。不能把相同执行收益说成预测始终相同。"
            if model_comparison["available"]
            else "本轮只检验预测强度与源码风险仓位，不重复训练或宣称新的神经网络收益。"
        ),
        "逐年、固定90日块、行情与已覆盖FOMC分组在summary.details。行情贡献为同一账户实际NAV变化之和，不是拼接状态净值；完整生命周期按入场状态或退出时点分组，与该阶段按市值损益口径不同。",
        "胜率附Wilson区间，并分别按币和入场月聚类Bootstrap；少于4个簇保留未知。两种聚类是分别检查，不是完整双向聚类，也不是校正全部历史搜索的显著性证明。",
        "历史行为与严格核对能排除部分算术／时序错误，不能证明已找到可持续收益源。下一轮须基于明确失效机制另行预登记，保留全部本轮失败，不降低现有门槛。",
    ]
    (docs / "README.md").write_text("\n".join(lines) + "\n")
    print(
        {
            "published": str(output),
            "accounts": len(rows),
            "representative": scorecard,
            "model_comparison": model_comparison,
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
