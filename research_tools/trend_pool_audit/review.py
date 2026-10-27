"""Bind matched-PIT results, scope limits and the independent mathematical review."""

import argparse
import math
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.trend_stop_audit.review import partitions


def review(inputs, run, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    verify_manifest(run)
    plan = read(run / "plan.json")
    summary = read(run / "summary.json")
    input_audit_path = Path(str(inputs) + "_independent_audit.json")
    check = read(input_audit_path)
    assert check["passed"] and check["inputs_completion_sha256"] == sha(inputs / "completion.json")
    assert not summary["failures"] and summary["actual_new_base_accounts"] == 4
    for path, digest in plan["account_source_hashes"].items():
        assert sha(Path(path)) == digest
    actual = [r for r in summary["account_rows"] if not r["reused_account"]]
    periods, exposure, metrics, errors = [], {}, {}, []
    hashes = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            input_audit_path,
            inputs / "completion.json",
            run / "completion.json",
            Path("research_tools/trend_stop_audit/review.py"),
        ]
    }
    for row in actual:
        folder = run / "accounts" / row["id"]
        result = read(folder / "result.json")
        assert sha(folder / "result.json") == row["result_sha256"]
        for name, digest in result["source_hashes"].items():
            assert sha(folder / name) == digest
        audit = read(folder / "internal_cash_audit.json")
        stop = read(folder / "internal_stop_audit.json")
        assert audit["passed"] and stop["passed"] and audit["result_sha256"] == row["result_sha256"]
        errors.append(audit["max_nav_error"])
        p, e = partitions(folder, row["id"])
        periods.extend(p)
        exposure[row["id"]] = e
        assert (
            abs(
                math.fsum(x["net_pnl"] for x in p if x["kind"] == "year")
                - row["metrics"]["net_pnl"]
            )
            < 1e-6
        )
        curve = pd.read_parquet(folder / "equity_5m.parquet").set_index("date").equity
        daily = curve.loc[(curve.index.hour == 0) & (curve.index.minute == 0)].pct_change().dropna()
        assert (
            abs(daily.mean() / daily.std(ddof=1) * math.sqrt(365) - row["metrics"]["sharpe"])
            < 1e-10
        )
        metrics[row["id"]] = row["metrics"]
        hashes.update(
            {
                str(folder / name): sha(folder / name)
                for name in ["result.json", "internal_cash_audit.json", "internal_stop_audit.json"]
            }
        )
    for domain in ["fixed40", "pit"]:
        a = pd.read_parquet(inputs / f"daily_archive_{domain}_targets.parquet")
        b = pd.read_parquet(
            inputs / f"daily_with_complete_5m_reconciliation_{domain}_targets.parquet"
        )
        assert a.equals(b)
    assert len(summary["account_rows"]) == 8 and summary["exact_source_duplicate_cases_reused"] == 4
    assert summary["actual_new_process_restart"]["passed"]
    assert (
        summary["actual_new_process_restart"]["parent_pid"]
        != summary["actual_new_process_restart"]["worker_pid"]
    )
    assert not summary["capacity_triggered"] and not summary["capacity_queue"]
    for row in actual:
        if row["domain"] == "pit":
            assert row["metrics"]["net_pnl"] < 0 and row["risk_halted"]
    save(output / "periods.json", periods)
    save(output / "exposure.json", exposure)
    save(output / "comparisons.json", summary["comparisons"])
    save(output / "input_diagnostics.json", read(inputs / "input_diagnostics.json"))
    m0, m1, p0, p1 = [
        metrics["daily_archive_" + d + "_cap10000_slip" + str(c)]
        for d, c in [("fixed40", 3), ("fixed40", 10), ("pit", 3), ("pit", 10)]
    ]
    differences = {
        "stress_PIT_minus_fixed40_net_pnl": p1["net_pnl"] - m1["net_pnl"],
        "funding_delta": p1["funding_net"] - m1["funding_net"],
        "fees_paid_delta": p1["fees"] - m1["fees"],
        "price_after_slippage_delta": (p1["net_pnl"] - p1["funding_net"] + p1["fees"])
        - (m1["net_pnl"] - m1["funding_net"] + m1["fees"]),
    }
    assert (
        abs(
            differences["funding_delta"]
            - differences["fees_paid_delta"]
            + differences["price_after_slippage_delta"]
            - differences["stress_PIT_minus_fixed40_net_pnl"]
        )
        < 1e-6
    )
    save(output / "cash_components.json", differences)
    verification = {
        "schema": "trend-pool-publication-review.v1",
        "passed": True,
        "checks": {
            "all_four_cash_and_stop_audits_bound": True,
            "all_source_pairs_exact_before_returns": True,
            "eight_logical_cases_four_actual_accounts": True,
            "two_costs_and_two_sources_all_PIT_failed": True,
            "periods_complete_openPnL_and_cash_components_reconcile": True,
            "holding_restart_actual_different_process_bound": True,
            "no_capacity_fabricated": True,
            "BTC_role_and_scope_limit_explicit": True,
        },
        "maximum_internal_NAV_error_usdt": max(errors),
        "independent_numpy_input_weight_error": check["maximum_numpy_forecast_risk_weight_error"],
        "source_hashes": hashes,
        "new_financial_accounts": 4,
        "new_economic_variants": 0,
        "third_party_review": False,
        "independent_unseen_evidence": False,
        "profit_goal_met": False,
        "orders_enabled": False,
    }
    save(output / "verification.json", verification)
    text = f"""# 既有趋势公式的时点候选域核验

2026-10-27。沿用原EWMAC32/128、冻结2024归一化、16%名义风险预算、预测阈值和8%逐次成交价收紧止损。本轮只检验已有实现是否能迁移到所封存的时点共同池，不增加公式或经济变体。

## 判定

**本版本在该时点池的两档成本下均亏损并停机，未取得准入证据。** 这不是所有趋势策略无效的结论。特别是，本轮共同池沿用此前跨币研究的定义：BTC只作基准、不参与交易。旧40参考账户曾允许BTC入场，因此不能把旧参考到本轮匹配40域的差额全解释为“扩池影响”，更不能把结果外推到BTC可交易的主流币时序趋势。

本轮主对照的两边都排除BTC，使用同一选池频率、价格/费用/资金费、信号数学、资格过滤和生命周期；二者比较才是在这个共同合同内改变候选域。

## 从原始记录到同口径输入

- 使用已封存的交易所原始日档、五分钟价格/成交量/报价额与资金费事件及标记价，不下载新市场档案；来源和具体原始归档均沿已有manifest追溯。所封存公告当前正文按宣称发布时间加5分钟可见仍是研究假设，不是历史修订/客户端资格认证。
- 旧40原始五分钟聚合日线、旧资格表及冻结归一化，可以精确重建全部14,592小时旧目标。改用原始日档但保留旧资格时，目标仍完全相同；共同40的执行数值与资金费记录也相同，无需新增执行来源桥接账户。
- 随后在40域内更换资格/身份/有效日规则，并把日度选池冻结到周度。这个输入改动块包含BTC角色、共同180观测/60日有效历史、每周前30与趋势180个连续有效日；只报告目标变化，不为每个中间块分配事后财务因果。
- 两边均在周一00UTC及研究起点选择历史流动性前30，日度00UTC更新趋势；当前连续窗口有88个池更新点，旧冷启动研究额外的2026-01-01锚点不强行插入连续账户。继承审计绑定全部89锚点/来源，主窗口成员逐项重建一致。
- 不以今天的成份回填过去。时点共同池来自860份日档中的历史合格成员，172是跨时点采集并集；旧40中另外14名仍保留其原始执行数据，共用186名源集合，非入选名字保持空仓。

公式不变；时间线不越过缺失、无量/无报价额日或已知换币身份。EWM历史起点不早于原研究2023-08-01，180个连续有效日之后才允许趋势预测；固定2024标量不重新训练，跨标的通用仍是待检验的假设。两档原始日档解释中各有31个“候选币-日”未满足趋势历史而留空，不用下一个标的替补。

另一条NumPy路径直接复算调整权重EWM、无偏加权波动率、滚动分位地板与仓位公式，59,902个合格币日和58,368条来源/域小时行一致，最大权重误差{check["maximum_numpy_forecast_risk_weight_error"]:.3g}。12个实际未来价格/报价额扰动案例保留决策时及之前的预测；不是未见验证或第三方审计。

## 同口径连续账户

1万USDT、2025-01-01至2026-09-01连续608日；每边手续费5bps加滑点3/10bps，总成本8/15bps；最多10仓、目标总敞口上限0.8、15%账户回撤停机及高水位连续，原8%逐次加仓价单向收紧止损；实际资金费、浮亏与退出费用准备完整计入，五分钟报价额1%只作容量代理。

| 候选域 | 每边总成本bps | 整段扣费收益 | CAGR | 完整胜率 | 最大回撤 | 日夏普 | 完整周期 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for domain, label in [("fixed40", "匹配40域"), ("pit", "时点共同池")]:
        for cost in [3, 10]:
            m = metrics[f"daily_archive_{domain}_cap10000_slip{cost}"]
            text += f"| {label} | {cost + 5} | {m['net_return']:.4%} | {m['cagr']:.4%} | {m['win_rate']:.2%} | {m['max_drawdown']:.4%} | {m['sharpe']:.4f} | {m['closed_trades']} |\n"
    text += """
四条实际账户最终均已平仓并停机。胜率只含完整持仓生命周期，不能把重复币种/时间与两档成本当独立样本。最大回撤为五分钟收盘净值观察值，存在延迟超限；不放宽原15%约束。高成本时点结果略好于低成本，是不同成交/止损/停机路径的结果，不是成本可以带来优势的证据。

两套日档解释涉及122个已封存的完整五分钟替换日，但本轮所有交易目标完全相同；按事前等价规则复用，8个来源×域×成本情景只计算4条新账户。它们既不是8个独立试验，也不证明整个历史数据源真实无误；两套解释属于同交易所来源敏感性。

## 现金组成与实际风险

"""
    text += f"""压力成本下，匹配40域与时点域净PnL分别为{m1["net_pnl"]:.4f}/{p1["net_pnl"]:.4f}USDT，历史路径差额{differences["stress_PIT_minus_fixed40_net_pnl"]:.4f}USDT。

| 压力成本现金组成 USDT | 匹配40域 | 时点共同池 |
| --- | ---: | ---: |
| 模型资金费 | {m1["funding_net"]:.4f} | {p1["funding_net"]:.4f} |
| 价格PnL，已含滑点 | {m1["net_pnl"] - m1["funding_net"] + m1["fees"]:.4f} | {p1["net_pnl"] - p1["funding_net"] + p1["fees"]:.4f} |
| 已付手续费 | {-m1["fees"]:.4f} | {-p1["fees"]:.4f} |
| 净PnL | {m1["net_pnl"]:.4f} | {p1["net_pnl"]:.4f} |

滑点已在价格PnL中，不能再扣一次。两边资金费/费用的差异不能解释为唯一原因；候选域同时改变持仓、价格暴露和停机反馈。整段平均总敞口为{m1["mean_gross_exposure"]:.2%}/{p1["mean_gross_exposure"]:.2%}，其中只持仓时的均值为{exposure["daily_archive_fixed40_cap10000_slip10"]["mean_gross_when_active"]:.2%}/{exposure["daily_archive_pit_cap10000_slip10"]["mean_gross_when_active"]:.2%}；全段均值受闲置时间压低，不是实现风险低的独立证明。

## 不同时段

| 候选域／成本 | 2025完整年 | 2026年1—8月，承接原账户 | 首次停机请求UTC |
| --- | ---: | ---: | --- |
"""
    for domain, label in [("fixed40", "匹配40域"), ("pit", "时点池")]:
        for cost in [3, 10]:
            identity = f"daily_archive_{domain}_cap10000_slip{cost}"
            ys = [x for x in periods if x["id"] == identity and x["kind"] == "year"]
            text += f"| {label}／{cost + 5}bps | {ys[0]['net_return']:.4%} | {ys[1]['net_return']:.4%} | {exposure[identity]['first_halt_request']} |\n"
    text += """
零收益后段来自既有停机，不表示该段市场没有机会。2026是243日部分年，不能与365日整年当成等长样本；所有固定90日与尾段见periods.json。不冷启动重置高点或把盈利片段拼回连续账户。

## 反证范围与下一步

本轮未满足两来源/两成本的正净PnL、正闭合PnL、回撤与不停机条件，按预登记未触发容量后续。32日块配对均值差的95%区间跨零，128日仅4个块当量不足，不能宣称统计上证明时点域普遍更差。既往搜索总量/依赖结构未完整校正，不制造DSR显著性。

停止在该共同池上改EMA、放宽止损、换日档版本或扩大名字数追盈利。下一步先对照原始趋势研究与当前适配：基准是否可交易、预测对象、持有/再平衡频率与机会集是否仍对应同一经济命题。当前“BTC基准不入场”的结论边界要先解决，再决定是否登记主流币可交易对照；不把一次阴性结果等同所有行情、所有趋势都失败。

## 可复核证据及仍缺什么

"""
    restart = summary["actual_new_process_restart"]
    text += f"""4条新账户现金、库存、精确资金费和实际止损轨迹由内部第二路径核对，最大净值误差{max(errors):.3g}USDT；实际时点压力账户在{restart["split"]}的持仓/加仓后断点以JSON交给不同PID，余下全段净值、现金流、请求、库存与最终高水位/停机状态与不中断运行相同。两个输入工具的技术失败（日期字典轴向、pandas只读数组）均发生在金融账户前并留档，修复没有改变经济假设或参数。

这些是内部数学与工程证据，尚无未参与开发者的外部复现、真正未见数据或客户原始账单对账；公告时点、真实盘口容量、交易规则/客户资格仍有未认证部分。总账户20%/30%年化、15%回撤、夏普1、多个有效机制组合目标保持未完成。

代码：research_tools/trend_pool_audit/；统一因子目录内的时间线适配：factors/crypto/trend_sessions.py。冻结输入：results/research/trend_pool_inputs_20261027_v002/及对应independent_audit.json；4条金融账户、来源复用、内部核验与真实新进程恢复：results/research/trend_pool_accounts_20261027_v001/。本目录verification.json绑定代码、输入、账本和全部核验哈希；哈希证明材料身份，不证明市场事实。没有下单、部署或推送。
"""
    (output / "README.md").write_text(text)
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print(
        {
            "passed": True,
            "report": str(output / "README.md"),
            "max_nav_error": max(errors),
            "new_accounts": 4,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    review(a.inputs, a.run, a.output)
