"""Publish checked comparison tables for the existing CZ research workbench."""

import argparse
import csv
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha


def publish(output, docs):
    base = Path("results/research")
    parent = base / "factor_lab_20261025_v004"
    roots = [
        base / "factor_lab_review_20261025_v001",
        base / "factor_lab_review_rolling_20261025_v001",
        base / "factor_lab_review_capacity_20261025_v001",
    ]
    studies = []
    sources = {}
    for root in roots:
        completion = read(root / "completion.json")
        if sha(root / "summary.json") != completion["summary_sha256"]:
            raise ValueError("Completion mismatch")
        data = read(root / "summary.json")
        if not data["audit"]["passed"] or not all(a["passed"] for a in data["audit"]["accounts"]):
            raise ValueError("Account audit failed")
        if data["plan_sha256"] != sha(root / "plan.json"):
            raise ValueError("Registered plan mismatch")
        sources[str(root.resolve())] = completion["summary_sha256"]
        studies.append(data)
    prior_audit = read(base / "factor_lab_20261025_audit_v003/audit.json")
    if not prior_audit["passed"] or prior_audit["source_catalog_sha256"] != sha(
        parent / "catalog.json"
    ):
        raise ValueError("Original64 audit mismatch")
    original = studies[0]["original"]
    rows = [r for s in studies for r in s["refinements"]]
    # Capacity cases with different starting capitals must never be pooled into a mean.
    groups = [g for s in studies for g in s["refinement_groups"] if g["phase"] != "capacity"]
    result = {
        "schema": "factor-lab-comparison-publication.v1",
        "original": original,
        "refinements": rows,
        "refinement_groups": groups,
        "selections": [s["selection"] for s in studies if "selection" in s],
        "source_summary_hashes": sources,
        "parent_catalog_sha256": sha(parent / "catalog.json"),
        "audit_passed": True,
        "audit_account_count": 64 + len(rows),
        "production_eligible": False,
        "independent_oos": False,
        "orders_enabled": False,
    }
    if len(original["account_rows"]) != 64 or len(rows) != 162:
        raise ValueError("Incomplete declared experiment grid")
    output.mkdir(parents=True, exist_ok=False)
    docs.mkdir(parents=True, exist_ok=True)
    save(output / "summary.json", result)
    save(
        output / "completion.json",
        {"summary_sha256": sha(output / "summary.json"), "audited_accounts": 226},
    )
    save(
        docs / "provenance.json",
        {
            "publication": str(output.resolve()),
            "sha256": sha(output / "summary.json"),
            "sources": sources,
            "audited_accounts": 226,
        },
    )
    columns = [
        "id",
        "candidate",
        "phase",
        "window",
        "capital",
        "cost_bps_per_side",
        "net_return",
        "win_rate",
        "max_drawdown",
        "sharpe",
        "profit_factor",
        "closed_trades",
        "closed_net_pnl",
        "open_pnl_with_exit_reserve",
        "fees",
        "slippage_cost",
        "funding_net",
        "turnover",
        "filled_request_ratio",
        "max_exit_delay_minutes",
    ]
    with (docs / "accounts_226.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for r in original["account_rows"] + rows:
            flat = {
                **r,
                **r["metrics"],
                **r["capacity"],
                "candidate": r.get("candidate", r["preset"]),
                "cost_bps_per_side": 5 + r["slippage_bps"],
            }
            writer.writerow({k: flat.get(k) for k in columns})
    save(
        docs / "comparison_tables.json",
        {
            "original_groups": original["groups"],
            "factor_diagnostics": original["stable_direction_factors"],
            "optimization_groups": groups,
            "selections": result["selections"],
            "macro_news_ablation": original["macro_news_ablation"],
        },
    )

    def p(v):
        return "—" if v is None else f"{v * 100:.2f}%"

    def n(v):
        return "—" if v is None else f"{v:.2f}"

    names = {
        "full": "完整预测模型",
        "technical": "技术与市场模型",
        "training_factor8": "训练筛选8因子",
        "long_only": "同规则做多对照",
    }
    lines = [
        "# 64组比较与有限优化 · 2026-10-25",
        "",
        "结论：原完整模型相对最有潜力，但成本压力下转负，闭合PF接近1；当前没有通过筛选的有效组合。新增162个账户均已完成，未放宽门槛，两轮都选择现金。",
        "",
        "## 比较口径",
        "",
        "原64组为56基础场景及8容量场景。基础场景初始1万USDT，七个独立28天窗口；1/4/7/10月的前28天，不是整季度。下列平均收益是窗口算术平均，夏普是窗口中位数，回撤是各窗口最差值，不能当连续账户累计收益/夏普。胜率合并完整生命周期，开放仓位不计胜率；跨组合的交易有相关性。",
        "",
        "净收益包含手续费、滑点、实际费率资金费及未平仓退出准备。夏普采用UTC日净值、365日年化、无风险利率0。原2026窗口已查看，所有后续变体属于事后探索；新增月份是新增计算的旧历史，不是独立未见样本外。",
        "",
        "## 原始组合",
        "",
        "|方案|单边成本bps|平均28天净收益|胜率|最差回撤|夏普中位数|闭合PF|盈利窗口|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in original["groups"]:
        lines.append(
            f"|{names[g['preset']]}|{g['total_cost_bps_per_side']}|{p(g['mean_window_return'])}|{p(g['complete_trade_win_rate'])}|{p(g['worst_observed_drawdown'])}|{n(g['median_window_sharpe'])}|{n(g['pooled_complete_profit_factor'])}|{g['positive_windows']}/{g['windows']}|"
        )
    lines += [
        "",
        "做多对照的闭合PF仅约0.1，盈利窗口多为未平仓浮盈，不能据账面收益认定优胜。",
        "",
        "## 逐窗口比较",
        "",
        "全部226个账户（含两档成本、每个方案、容量场景）的收益、胜率、回撤、夏普、资金费、换手及成交比例见 [accounts_226.csv](accounts_226.csv)。以下列原始单边8bps：",
        "",
        "|窗口|方案|净收益|胜率|回撤|夏普|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in original["account_rows"]:
        if r["phase"] == "capacity" or r["slippage_bps"] != 3:
            continue
        m = r["metrics"]
        lines.append(
            f"|{r['window']}|{names[r['preset']]}|{p(m['net_return'])}|{p(m['win_rate'])}|{p(m['max_drawdown'])}|{n(m['sharpe'])}|"
        )
    lines += [
        "",
        "## 因子证据",
        "",
        "训练/验证Rank IC同号的因子只表示方向关联。固定诊断了6个因子：四个2025窗口、单边8bps、40%目标总仓位、目标多空中性；六者平均净收益均为负。不能仅按IC把它们宣布为收益源。",
        "",
        "|因子|训练Rank IC|验证Rank IC|方向|单因子平均净收益|单因子胜率|最差回撤|夏普中位数|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    singles = {
        g["candidate"].removeprefix("single_"): g
        for g in groups
        if g["phase"] == "single_factor_diagnostic"
    }
    for f in original["stable_direction_factors"]:
        g = singles.get(f["id"], {})
        lines.append(
            f"|{f['id']}|{f['training_rank_ic']:.4f}|{f['validation_rank_ic']:.4f}|{f['direction']}|{p(g.get('mean_window_return'))}|{p(g.get('complete_trade_win_rate'))}|{p(g.get('worst_observed_drawdown'))}|{n(g.get('median_window_sharpe'))}|"
        )
    lines += [
        "",
        "空白表示未做单因子账户测试。FOMC交互项的相关性也不等于新闻策略收益；full/technical为同源模型消融，仅说明该模型配置的历史增量。",
        "",
        "## 已执行的两轮优化",
        "",
        "第一轮114账户：仓位降至40%的等风险基线、门槛翻倍、168小时方向过滤、目标中性、固定等权6因子。先做2025筛选，再做2026诊断、补充月份与容量压力。单因子诊断不参与筛选。",
        "",
        "第二轮40账户：滚动180天训练，固定Ridge惩罚100，只用窗口开始前已成熟24h标签；训练内归一化与中心化截距。两模型×10窗口×2档成本。训练区间和截距估计同时变化，不能把差异单独归因于滚动训练。另固定2025Q2为两候选补做100万/1000万USDT、两档成本的8个容量账户，复用冻结模型系数，不改变筛选结论。固定本轮预算后结束，无未披露参数网格。Ridge定义参考 [scikit-learn官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)。",
        "",
        "筛选门槛：两档成本下，2025四窗平均收益均>0、至少3窗盈利、最大回撤≤8%、完整交易≥30、闭合PF≥1.1。两轮均无方案通过；之后2026结果不反选方案。",
        "",
        "|阶段|方案|单边bps|平均28天净收益|胜率|最差回撤|夏普中位数|闭合PF|盈利窗口|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        if g["phase"] == "single_factor_diagnostic":
            continue
        lines.append(
            f"|{g['phase']}|{g['candidate']}|{g['slippage_bps'] + 5}|{p(g['mean_window_return'])}|{p(g['complete_trade_win_rate'])}|{p(g['worst_observed_drawdown'])}|{n(g['median_window_sharpe'])}|{n(g['pooled_complete_profit_factor'])}|{g['positive_windows']}/{g['windows']}|"
        )
    lines += [
        "",
        "## 容量与边界",
        "",
        "容量不同资金分别展示，不混合计算胜率或平均收益。五分钟成交额1%是执行代理；资金变大时建仓不足会使亏损收窄，不能据此推断更大资金容量。完整逐账户成交/请求、退出耗时见CSV及现有界面。未获利策略尚无可证明的盈利容量。",
        "",
        "目标中性仅约束下单目标；持仓漂移、单边部分成交或退出使实际净敞口不为零。第一轮账户额外输出实际净敞口；不能称为市场中性实盘策略。",
        "",
        "原始64个账户与新增162个账户均通过逐账户独立现金流、资金费、生命周期及NAV重算。固定当代六币池有幸存者偏差；28天窗口与重叠标签不足以证明长期显著性；没有部署或开单。",
        "",
        "下一步优先做收益来源及状态依赖研究，保留完整模型作基准，停止盲目堆因子；候选冻结后需新的未使用时段与连续纸面账户，达到全成本闭合收益、回撤、容量及重启连续性门槛再推进。",
        "",
        "复现：",
        "",
        "```sh",
        "/Users/chrise/GitHub/freqtrade/.venv/bin/python -m research_tools.factor_lab_review.run --output results/research/NEW_REVIEW_DIRECTORY",
        "OPENBLAS_NUM_THREADS=1 /Users/chrise/GitHub/freqtrade/.venv/bin/python -m research_tools.factor_lab_review.rolling --output results/research/NEW_ROLLING_DIRECTORY",
        "```",
        "",
        "源数据、计划、模型系数、逐账户证据与摘要位于provenance.json登记的结果根目录。输出目录采用独占创建，避免覆盖冻结证据。",
    ]
    (docs / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--docs", type=Path, required=True)
    a = p.parse_args()
    publish(a.output.resolve(), a.docs.resolve())
