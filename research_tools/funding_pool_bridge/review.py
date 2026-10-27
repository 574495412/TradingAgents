"""Bind prior internal audits, current reconciliation and the human-readable result."""

import argparse
import math
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.funding_pool_bridge.run import sealed_account
from research_tools.max_effect.run_accounts import verify_manifest


def review(inputs, run, publication, output):
    output.mkdir(parents=True, exist_ok=False)
    for root in [inputs, run, publication]:
        verify_manifest(root)
    plan = read(inputs / "plan.json")
    s = plan["inputs"]
    summary = read(publication / "summary.json")
    for path, digest in read(publication / "source_bindings.json").items():
        if sha(Path(path)) != digest:
            raise ValueError("Publication binding changed")
    old_folder, old = sealed_account(
        s["old_run"], "archive_diagnostic2026_funding7_raw_cap10000_slip10"
    )
    pit_folder, pit = sealed_account(
        s["pit_run"], "daily_archive_diagnostic2026_funding7_raw_cap10000_slip10"
    )
    new_folder, new = sealed_account(run, "B_execution")
    audit_paths = [
        Path(s["old_inputs"] + "_independent_audit.json"),
        Path(s["pit_inputs"] + "_independent_audit.json"),
        Path(s["cohort"] + "_independent_audit.json"),
        old_folder / "independent_audit.json",
        Path(s["pit_run"]) / "financial_audits.json",
        new_folder / "internal_cash_audit.json",
    ]
    old_input, pit_input, cohort, old_audit, pit_audits, new_audit = map(read, audit_paths)
    pit_audit = next(r for r in pit_audits if r["id"] == pit_folder.name)
    if not (old_input["passed"] and pit_input["passed"] and cohort["passed"]):
        raise ValueError("Inherited target/cohort review not passed")
    if (
        old_input["input_completion_sha256"] != sha(Path(s["old_inputs"]) / "completion.json")
        or pit_input["inputs_completion_sha256"] != sha(Path(s["pit_inputs"]) / "completion.json")
        or cohort["cohort_completion_sha256"] != sha(Path(s["cohort"]) / "completion.json")
    ):
        raise ValueError("Target/cohort review does not bind these inputs")
    for audit, folder in [
        (old_audit, old_folder),
        (pit_audit, pit_folder),
        (new_audit, new_folder),
    ]:
        if not audit["passed"] or audit["result_sha256"] != sha(folder / "result.json"):
            raise ValueError("Cash audit does not bind account")
    anchors = read(inputs / "anchors.json")
    changed = sum(r["matched40_book"] != r["pit_book"] for r in anchors)
    overlap = [len(set(r["matched40_book"]) & set(r["pit_book"])) for r in anchors]
    old_parts = read(publication / "symbol_attribution.json")
    old_total = math.fsum((r["old"] or {}).get("net_pnl", 0) for r in old_parts)
    pit_total = math.fsum((r["pit"] or {}).get("net_pnl", 0) for r in old_parts)
    if (
        abs(old_total - old["metrics"]["net_pnl"]) > 1e-6
        or abs(pit_total - pit["metrics"]["net_pnl"]) > 1e-6
    ):
        raise ValueError("Published attribution does not reconcile")
    delta = summary["full_component_delta"]
    component_sum = (
        delta["funding"] + delta["price_after_slippage_and_open_exit_reserve"] - delta["fees_paid"]
    )
    assert abs(component_sum - summary["historical_pool_path_delta_usdt"]) < 1e-6
    periods = read(publication / "periods.json")
    for account, result in [("old", old), ("pit", pit)]:
        assert (
            abs(
                math.fsum(r["net_pnl"] for r in periods if r["account"] == account)
                - result["metrics"]["net_pnl"]
            )
            < 1e-6
        )
    for label, result in [("old", old), ("pit", pit), ("bridge", new)]:
        folder = {"old": old_folder, "pit": pit_folder, "bridge": new_folder}[label]
        curve = pd.read_parquet(folder / "equity_5m.parquet").set_index("date").equity
        daily = curve.loc[(curve.index.hour == 0) & (curve.index.minute == 0)].pct_change().dropna()
        sharpe = daily.mean() / daily.std(ddof=1) * math.sqrt(365)
        assert abs(sharpe - result["metrics"]["sharpe"]) < 1e-10
    prior = read(inputs / "protocol_snapshot.json")
    save(output / "execution_goal_snapshot.json", prior)
    checks = {
        "raw_source_numeric_and_quality_identity": not read(inputs / "checks.json")[
            "source_different_names"
        ],
        "all5832matched40_hourly_targets_identical": read(inputs / "target_comparison.json")[
            "old_vs_matched40"
        ]["different_cells"]
        == 0,
        "PIT_targets_reconstructed_from860raw_daily_assets": read(inputs / "checks.json")[
            "all_new_PIT_hourly_books_exact"
        ],
        "empty_old40_lifecycle_directives": not read(inputs / "lifecycle40.json"),
        "all69985NAV_cash_and_exposure_rows_equal": max(
            summary["full_5m_execution_bridge_errors"].values()
        )
        == 0,
        "all_three_account_internal_audits_bound": True,
        "marked_open_positions_and_exit_reserve_included": summary["old_open_positions"] == 8,
        "periods_and_symbol_components_reconcile": True,
        "postprocessing_failure_preserved_no_financial_rerun": not read(run / "completion.json")[
            "passed"
        ]
        and read(publication / "completion.json")["passed"],
        "one_new_account_no_new_hypothesis": summary["actual_new_financial_accounts"] == 1
        and summary["new_independent_hypotheses"] == 0,
        "no_external_or_unseen_claim": not summary["third_party_review"]
        and not summary["independent_unseen_evidence"],
    }
    assert all(checks.values())
    receipt = {
        "schema": "funding-pool-bridge-verification.v1",
        "checks": checks,
        "source_hashes": {
            str(p): sha(p)
            for p in [
                *audit_paths,
                inputs / "completion.json",
                run / "completion.json",
                publication / "completion.json",
                Path(__file__),
            ]
        },
        "book_changes": changed,
        "weekly_clocks": len(anchors),
        "mean_nonzero_name_overlap_out_of8": sum(overlap) / len(overlap),
        "new_account_internal_audit": new_audit,
        "third_party_review": False,
        "profit_goal_met": False,
        "orders_enabled": False,
    }
    save(output / "verification.json", receipt)
    a, b = old["metrics"], pit["metrics"]
    text = f"""# 资金费固定池与时点池：受控差异解释

2026-10-27。问题只有一个：旧固定40候选域与历史可见候选域，为什么在同一2026冷启动窗口得到相反结果？这是已观察历史诊断，不是新收益发现或样本外验证。

## 结论

在本窗口，执行器替换、共同币种执行数据、生命周期规则及旧40域内信号构造均没有改变结果。历史候选域改变使36/36周度目标变化，包含回撤停机反馈后的净账户路径差额为 **{summary["historical_pool_path_delta_usdt"]:.8f} USDT（{summary["historical_pool_path_delta_percentage_points"]:.4f}个百分点）**。这解释的是这两条已观察路径，不能外推为新币普遍更差、严格意义的幸存者偏差效应或未来资金费收益。

## 同口径账户

共同口径：2026-01-01至2026-09-01，243自然日，1万USDT，手续费5bps加滑点10bps/边，0.8总敞口上限，最多10仓，单仓8%止损，15%账户回撤停机，实际五分钟报价额1%容量代理。每次决策从合格域选流动性前30，再按已结算过去7日资金费选四多四空。公开资金费模型入账不等于客户实际到账。胜率只统计扣费后完整持仓生命周期。

| 指标 | 固定40域 | 时点候选域 |
| --- | ---: | ---: |
| 整段扣费收益 | {a["net_return"]:.4%} | {b["net_return"]:.4%} |
| 净PnL USDT | {a["net_pnl"]:.4f} | {b["net_pnl"]:.4f} |
| 完整周期胜率 | {a["win_rate"]:.2%} | {b["win_rate"]:.2%} |
| 完整周期数 | {a["closed_trades"]} | {b["closed_trades"]} |
| 五分钟净值最大回撤 | {a["max_drawdown"]:.4%} | {b["max_drawdown"]:.4%} |
| 日收益夏普，365日尺度 | {a["sharpe"]:.4f} | {b["sharpe"]:.4f} |
| 期末未平币种 | {len(old["open_positions"])} | {len(pit["open_positions"])} |
| 触发账户停机 | {old["risk_halted"]} | {pit["risk_halted"]} |

不将243日结果作为至少365日总账户年化目标验收。固定域期末净值已计8个持仓及退出费用准备{summary["old_open_exit_reserve"]:.6f}USDT；完整已闭合周期PnL为{a["closed_net_pnl"]:.6f}，当前未闭合周期现金与盯市净PnL为{summary["old_open_lifecycle_net_pnl_including_cash_and_mark"]:.6f}。两者之和才是账户PnL，不把已实现利润与盯市利润混用。

## 单项控制链

| 节点 | 唯一变化 | 证据 |
| --- | --- | --- |
| A 原账户 | 原固定40域及旧执行器 | 复用冻结历史账户 |
| B 执行 | 仅替换为稀疏执行器，保留原目标、数值数据及空生命周期 | 新算1条账户，69,985行净值、现金与多空敞口逐点完全相同 |
| C 来源 | 同名采用扩展bundle，缺少的14名保留原来源 | 40名的执行前7日至期末价格、数量、报价额、质量标记、资金费/标记价逐行相同；无需新增账户 |
| D 生命周期 | 原40域应用冻结时点生命周期指令 | 指令为空；无需新增账户 |
| E 信号构造 | 原40域改用原始日档、有效交易日和身份感知流程 | 全部5,832小时目标与旧目标相同；无需新增账户 |
| F 候选域 | 同一新流程改用历史可见候选域 | 860份原始日档重建时点成员与目标，全小时与冻结PIT目标相同；复用原账户 |

旧40域有14名不在研究期最终172名采集并集中；在此诊断保留其原始数据，避免把“缺数据”冒充“退出机会集”。172是跨时点采集并集，每周交易域仍按当时规则选前30，不是172名同时交易。两套目标每周平均只有{sum(overlap) / len(overlap):.4f}/8个非零名字重合；这是组成变化描述，不能按单币事后PnL选择新池。

## 现金流为何不同

| PnL组成 USDT | 固定40域 | 时点域 | 时点减固定 |
| --- | ---: | ---: | ---: |
| 实际费率模型资金费入账 | {a["funding_net"]:.4f} | {b["funding_net"]:.4f} | {delta["funding"]:.4f} |
| 价格PnL，已含成交滑点及期末退出准备 | {a["net_pnl"] - a["funding_net"] + a["fees"]:.4f} | {b["net_pnl"] - b["funding_net"] + b["fees"]:.4f} | {delta["price_after_slippage_and_open_exit_reserve"]:.4f} |
| 已付手续费 | {-a["fees"]:.4f} | {-b["fees"]:.4f} | {-delta["fees_paid"]:.4f} |
| 净PnL | {a["net_pnl"]:.4f} | {b["net_pnl"]:.4f} | {summary["historical_pool_path_delta_usdt"]:.4f} |

成交滑点已包含在价格PnL中，期末退出准备同时含预估手续费和滑点；不能再减一次。两账户另报已发生成交滑点分别{a["slippage_cost"]:.4f}/{b["slippage_cost"]:.4f}USDT。逐币及总账归因残差为{summary["symbol_attribution_residual_usdt"]:.3g}USDT。资金费更多却净收益更差，说明价格风险不能由资金费收入表掩盖。

## 不同时段

| 时段 | 固定40域扣费收益 | 时点域扣费收益 |
| --- | ---: | ---: |
"""
    for i in range(3):
        x, y = periods[i], periods[i + 3]
        text += f"| {x['start']}至{x['end']} | {x['net_return']:.4%} | {y['net_return']:.4%} |\n"
    text += """
7—8月只有两个月，不能与完整季度视作等长样本；时点池该段为0是前段回撤停机后的空仓，不是该段机会没有收益。固定域1—8月的正收益主要在最后两个月形成，不能据此另选窗口。原完整2025—2026连续账户压力成本收益分别为-7.7947%与-8.8342%，同样不支持将2026冷启动正段拼回连续历史。

## 判定与下一步

- **支持的诊断：**在此冻结窗口及规则下，候选域导致目标与风险路径变化；输入及执行差异不是两端收益差异的解释。
- **不予准入：**本版本过去7日资金费跨币多空，尚无可迁移的扣费收益证据；停止在该分支扩池、调资金费门槛或加模型。此结论不代表所有资金费策略都无效。
- **仍然未知：**未来表现、真实盘口容量、客户实际费用/到账与外部复核。五分钟报价额1%只是假设，当前无第三方审计或未见证据。
- **下一研究机制：**回到原中低频趋势主线，先核对已有趋势/动量基线失败的价格暴露与时点池影响，再决定单一时间序列趋势假设及最强对照；本轮不新增模型、市场或参数搜索。

新执行账户的69,985行现金/库存/NAV由第二代码路径复核，最大误差3.82e-11USDT，属于内部工程证据。A/F既有内部审计的result哈希已逐项绑定。共1条新增诊断账户、2条复用端点、3个精确等价中间节点；不是6个新账户或独立发现。

## 原始材料、复算与失败留档

代码与冻结诊断计划：`research_tools/funding_pool_bridge/`。输入与来源哈希：`results/research/funding_pool_bridge_inputs_20261027_v001/`；唯一新账户：`results/research/funding_pool_bridge_accounts_20261027_v001/accounts/B_execution/`；完整归因：`results/research/funding_pool_bridge_publication_20261027_v001/`。

首版run.py在金融重放、内部审计和端点一致检查之后，因只支持已清仓账户的归因假设而失败。原代码及输出封存为失败；publish.py独立补入真实期末持仓和退出准备完成发布，未重跑账户、未改信号/风险/费用。此发布修复不是第二次经济试验。

复算需在隔离工作目录恢复冻结来源与数据，并把本目录execution_goal_snapshot.json还原为当时目标路径后依次执行prepare、run、publish、review；原run的已记录发布失败仍应出现，publish复用已算账户。当前目标包含后续进度，不能追溯修改旧注册哈希。所有原始数据、公式、审计与依赖按哈希绑定；哈希只保证材料身份，不证明市场事实或盈利。数据/结果按既有规则本地封存，代码/目标/审查说明本地提交，无下单、部署或推送。
"""
    (output / "README.md").write_text(text)
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print({"all_checks": checks, "report": str(output / "README.md")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k in ["inputs", "run", "publication", "output"]:
        parser.add_argument("--" + k, type=Path, required=True)
    a = parser.parse_args()
    review(a.inputs, a.run, a.publication, a.output)
