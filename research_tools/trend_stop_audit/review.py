"""Publish a bounded failed stop-policy hypothesis and its verification limits."""

import argparse
import math
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def partitions(folder, identity):
    curve = pd.read_parquet(folder / "equity_5m.parquet").set_index("date")
    events = read(folder / "events.json")
    ranges = [
        (
            "year",
            str(y),
            pd.Timestamp(f"{y}-01-01", tz="UTC"),
            min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), curve.index[-1]),
        )
        for y in [2025, 2026]
    ]
    ranges += [
        ("90day", str(i + 1), start, min(start + pd.Timedelta(days=90), curve.index[-1]))
        for i, start in enumerate(pd.date_range(curve.index[0], curve.index[-1], freq="90D"))
    ]
    rows = []
    for kind, label, a, b in ranges:
        nav = curve.loc[a:b].equity
        rows.append(
            {
                "id": identity,
                "kind": kind,
                "label": label,
                "start": a.isoformat(),
                "end": b.isoformat(),
                "calendar_days": (b - a).total_seconds() / 86400,
                "net_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
                "net_pnl": float(nav.iloc[-1] - nav.iloc[0]),
                "within_period_drawdown": float((1 - nav / nav.cummax()).max()),
            }
        )
    active = curve.gross_exposure.gt(0)
    exposure = {
        "active_5m_intervals": int(active.sum()),
        "mean_gross_when_active": float((curve.gross_exposure / curve.equity).loc[active].mean()),
        "first_halt_request": next(
            (e["submitted_at"] for e in events if e.get("reason") == "risk_halt"), None
        ),
    }
    return rows, exposure


def review(inputs, run, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(inputs)
    verify_manifest(run)
    plan, summary = read(run / "plan.json"), read(run / "summary.json")
    for path, digest in plan["account_source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Run source or input changed")
    assert not summary["failures"] and summary["audited_accounts"] == 5
    rows = {r["id"]: r for r in summary["account_rows"]}
    periods, exposure, nav_errors, hashes = [], {}, [], {}
    for identity, row in rows.items():
        folder = run / "accounts" / identity
        result = read(folder / "result.json")
        assert sha(folder / "result.json") == row["result_sha256"]
        for name, digest in result["source_hashes"].items():
            assert sha(folder / name) == digest
        audit, stop = (
            read(folder / "internal_cash_audit.json"),
            read(folder / "internal_stop_audit.json"),
        )
        assert audit["passed"] and stop["passed"] and audit["result_sha256"] == row["result_sha256"]
        assert not stop["retroactive_new_stop_requests"]
        nav_errors.append(audit["max_nav_error"])
        p, e = partitions(folder, identity)
        periods.extend(p)
        exposure[identity] = e
        assert (
            abs(
                math.fsum(x["net_pnl"] for x in p if x["kind"] == "year")
                - result["metrics"]["net_pnl"]
            )
            < 1e-6
        )
        curve = pd.read_parquet(folder / "equity_5m.parquet").set_index("date").equity
        daily = curve.loc[(curve.index.hour == 0) & (curve.index.minute == 0)].pct_change().dropna()
        assert (
            abs(daily.mean() / daily.std(ddof=1) * math.sqrt(365) - result["metrics"]["sharpe"])
            < 1e-10
        )
        assert (
            abs(float((1 - curve / curve.cummax()).max()) - result["metrics"]["max_drawdown"])
            < 1e-10
        )
        hashes.update(
            {
                str(folder / name): sha(folder / name)
                for name in ["result.json", "internal_cash_audit.json", "internal_stop_audit.json"]
            }
        )
    save(output / "periods.json", periods)
    save(output / "exposure.json", exposure)
    legacy = (
        Path(plan["inputs"]["legacy_run"])
        / "accounts/continuous2025_2026_trend_risk16_cap10000_slip10"
    )
    old = read(legacy / "result.json")
    for name, digest in old["source_hashes"].items():
        assert sha(legacy / name) == digest
    a, b, c = (
        old["metrics"],
        rows["B_execution_stress"]["metrics"],
        rows["C_exact_stress"]["metrics"],
    )
    bridge = {
        "legacy_net_pnl": a["net_pnl"],
        "new_engine_legacy_funding_net_pnl": b["net_pnl"],
        "new_engine_exact_funding_net_pnl": c["net_pnl"],
        "execution_bundle_delta_usdt": b["net_pnl"] - a["net_pnl"],
        "funding_record_delta_usdt": c["net_pnl"] - b["net_pnl"],
        "total_delta_usdt": c["net_pnl"] - a["net_pnl"],
    }
    save(output / "bridge.json", bridge)
    assert len(summary["comparisons"]) == 2 and all(
        r["D_minus_C_net_pnl"] < 0 for r in summary["comparisons"]
    )
    assert not summary["capacity_triggered"] and not summary["capacity_queue"]
    restart = summary["actual_new_process_restart"]
    assert restart["passed"] and restart["parent_pid"] != restart["worker_pid"]
    expected = read(run / "restart_worker.json")["resume_state"]
    actual = read(run / "accounts/D_costanchor_stress/resume_state.json")
    assert expected == actual
    save(output / "comparisons.json", summary["comparisons"])
    save(
        output / "verification.json",
        {
            "schema": "trend-stop-policy-review.v1",
            "passed": True,
            "checks": {
                "five_cash_and_actual_stop_trace_audits": True,
                "complete_open_PnL_and_years_reconciled": True,
                "metric_calculations_checked": True,
                "legacy_execution_and_funding_bridge_bound": True,
                "both_costs_reject_positive_increment_and_net_conditions": True,
                "capacity_not_triggered_per_preregistration": True,
                "actual_holding_JSON_resume_different_PID": True,
                "retained_legacy_no_retroactive_stop_explanation": True,
            },
            "maximum_internal_cash_NAV_error_usdt": max(nav_errors),
            "new_base_financial_accounts": 5,
            "new_economic_variants": 1,
            "internal_engineering_fixture_financial_accounts": 0,
            "source_hashes": {
                **hashes,
                **{
                    str(p): sha(p)
                    for p in [
                        Path(__file__),
                        inputs / "completion.json",
                        run / "completion.json",
                        legacy / "result.json",
                    ]
                },
            },
            "external_review": False,
            "independent_unseen_evidence": False,
            "profit_goal_met": False,
            "orders_enabled": False,
        },
    )
    cb, cs, db, ds = [
        rows[k]["metrics"]
        for k in ["C_exact_base", "C_exact_stress", "D_costanchor_base", "D_costanchor_stress"]
    ]
    text = f"""# 趋势止损锚点：一次受控反证

2026-10-27。沿用既有PST EWMAC32/128、2024年冻结归一化、16%波动预算和原固定40域；只检验加仓后的止损锚点，不增加因子、信号窗口、市场或神经网络。所有2025/2026数据均已访问，结果是历史诊断，不是新样本外。

## 本轮判定

**不支持采用“持仓加权成本8%止损”替代原“逐次成交价单向收紧8%止损”。** 两档成本的完整账户和增量均失败，按预登记停止该锚点分支，不搜索新止损宽度。不能推断所有趋势策略或所有止损锚点都无效。

原始研究区分自身历史回报形成的时间序列动量和同日跨资产排序的横截面动量：[Moskowitz/Ooi/Pedersen 2012](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)。[Liu/Tsyvinski 2018原始摘要](https://www.nber.org/papers/w24877)只作为加密自身动量动机，PDF访问403，未据未读正文推导参数。两文不支持本地具体止损选择，也不认证本账户盈利；本地EWMAC32/128不是原论文12月趋势的完整复现。此前PST公式与风险数学继续绑定commit164328bd76d2937d3dd61491377704c488283bc2及封存source_review.json。

## 原始记录先回答了什么

- 冻结逐币五分钟OHLC、成交量与报价额；原始资金费事件保留实际毫秒时间和标记价，不将其压到五分钟整点。78745条名义事件中37600条带子周期时间偏移；原记录与新记录按名义时点对齐后8050条费率或标记价有差异。共同数据哈希与逐币差异均封存，无新下载。
- 原风险预算趋势有3733次同向加仓、1884次止损收紧及740次完整止损退出；逐条重建后没有发现新止损引用加仓前区间而独立导致的实际退出。因此不能把该潜在时间边界风险当作这条亏损路径的已证原因。
- 仅切换执行器的压力净PnL变化为{bridge["execution_bundle_delta_usdt"]:+.8f}USDT；再恢复真实资金费事件，变化为{bridge["funding_record_delta_usdt"]:+.8f}USDT，合计{bridge["total_delta_usdt"]:+.8f}USDT。资金费来源及执行时序差异不足以解释趋势收益弱。

执行器切换包含已声明的确认反向与结算时序差异，不能把第一段差额归因于单个内部细节；第二段才是在同一新执行器下只变资金费记录。标记价修订或时点差异的发现不自动意味着经济结论变化。

## 唯一变体及共同约束

基线在每次加仓后，以新成交价计算8%止损，长仓只上移、短仓只下移。变体改为更新后的持仓加权成本乘以1减方向乘8%；减仓不改变成本或止损。变体可能在加仓后把止损移远，这是显式风险政策变化，**不是原实现纠错**。

共同条件：1万USDT、2025-01-01至2026-09-01连续608日；最多10仓，目标总敞口上限0.8，无放大杠杆；同样的8%比例、15%账户回撤停机与高水位连续性；每日UTC固定调仓，实际五分钟收盘成交价加不利滑点，先确认退出再反向；手续费5bps加滑点3或10bps/边，实际事件资金费，报价额参与率1%代理。相同意图和名义风险预算不代表相同实际风险或持仓路径。

## 扣费表现

| 规则 | 每边总成本bps | 整段收益 | CAGR | 完整胜率 | 最大回撤 | 日夏普 | 完整周期 | 停机 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
"""
    for identity, label in [
        ("C_exact_base", "原逐次收紧"),
        ("C_exact_stress", "原逐次收紧"),
        ("D_costanchor_base", "加权成本锚点"),
        ("D_costanchor_stress", "加权成本锚点"),
    ]:
        r = rows[identity]
        m = r["metrics"]
        text += f"| {label} | {5 + r['slippage_bps']} | {m['net_return']:.4%} | {m['cagr']:.4%} | {m['win_rate']:.2%} | {m['max_drawdown']:.4%} | {m['sharpe']:.4f} | {m['closed_trades']} | 是 |\n"
    text += """
胜率只包含扣费后的完整持仓周期；这些周期存在同币、时间与市场依赖，不能按861个独立试验解释。最大回撤是五分钟收盘净值观察值，不是区间内最差成交；触发15%停机仍有延迟超限，按原目标保留失败，不放宽回撤门槛。

## 为什么少交易没有变成优势

"""
    text += f"""压力成本下，手续费加成交滑点从{cs["fees"] + cs["slippage_cost"]:.4f}降到{ds["fees"] + ds["slippage_cost"]:.4f}USDT，但净PnL从{cs["net_pnl"]:.4f}变为{ds["net_pnl"]:.4f}。变体在{exposure["D_costanchor_stress"]["first_halt_request"]}就停机，原规则直到{exposure["C_exact_stress"]["first_halt_request"]}才停机。费用和交易数减少也来自早停与闲置，不能全解释为低换手效率提高。

整段平均实际总敞口分别{cs["mean_gross_exposure"]:.2%}/{ds["mean_gross_exposure"]:.2%}；仅持仓时均值为{exposure["C_exact_stress"]["mean_gross_when_active"]:.2%}/{exposure["D_costanchor_stress"]["mean_gross_when_active"]:.2%}。不能把相同16%名义预算当作风险已匹配。

| 压力成本现金组成 USDT | 原逐次收紧 | 加权成本锚点 |
| --- | ---: | ---: |
| 模型资金费 | {cs["funding_net"]:.4f} | {ds["funding_net"]:.4f} |
| 价格PnL，已含滑点 | {cs["net_pnl"] - cs["funding_net"] + cs["fees"]:.4f} | {ds["net_pnl"] - ds["funding_net"] + ds["fees"]:.4f} |
| 已付手续费 | {-cs["fees"]:.4f} | {-ds["fees"]:.4f} |
| 净PnL | {cs["net_pnl"]:.4f} | {ds["net_pnl"]:.4f} |

四条主对照期末均已平仓。滑点已包含在价格PnL内，不能重复扣减，也不能把加回滑点的算术分解当作零成本策略重跑。完整停止周期的平均PnL从{rows["C_exact_stress"]["by_exit_reason"]["stop"]["expectancy"]:.4f}变为{rows["D_costanchor_stress"]["by_exit_reason"]["stop"]["expectancy"]:.4f}USDT；这只是不同持仓/停机路径下的描述，不证明某类退出本身导致亏损。

## 不同时段

| 规则／成本 | 2025完整年 | 2026年1—8月，承接原账户 |
| --- | ---: | ---: |
"""
    for identity, label in [
        ("C_exact_base", "原逐次收紧／8bps"),
        ("C_exact_stress", "原逐次收紧／15bps"),
        ("D_costanchor_base", "加权成本／8bps"),
        ("D_costanchor_stress", "加权成本／15bps"),
    ]:
        year = [x for x in periods if x["id"] == identity and x["kind"] == "year"]
        text += f"| {label} | {year[0]['net_return']:.4%} | {year[1]['net_return']:.4%} |\n"
    text += """
2026的0不是该年没有机会，而是保留2025停机状态；不冷启动补回收益。2026只有243日，不与完整年当成等长窗口。全部固定90日块见periods.json，最后尾块按实际长度记录。

## 不确定性与反证边界

两档成本的D减C整段PnL均负，因此不满足预登记采用条件。日历配对收益差的32日循环块重采样95%区间均跨零；128日块仅4个完整块当量，按事前至少8块要求保留未知。因此不能宣称统计上证明这个锚点普遍更差。未完整校正既往全部策略搜索；已访问历史、重复成本与桥接账户也不能变成新的独立样本外。分币和入场月的聚类区间分别保存，二者不是完整双向聚类校正。

## 工程证据与下一步

"""
    text += f"""5条新增金融账户均经另一条内部代码路径重算现金、持仓、精确资金费和净值，最大误差{max(nav_errors):.3g}USDT。每次实际止损请求与逐笔加仓形成的止损状态均核对，未检出新激活止损对之前区间的实际引用；此检查不证明所有未发生止损的完整性。D压力账户在{restart["holding_split"]}真实持仓和加仓后序列化，由不同PID恢复，余下全段净值、现金流、请求、库存与最终风险状态与不中断执行完全一致。合成边界案例另验证开关关闭复现基线；合成案例不计金融账户。首版合成测试因混用时区在回放前失败，未产生金融路径，已留档并由显式UTC版本修复。

未触发容量后续条件，未生成10万/100万账户，也没有盘口容量或实盘认证。所有复核均为内部证据，目前没有外部第三方复现或未见验证；总账户收益目标仍未完成。

**下一步只做现有趋势公式的时点候选域核验。** 保留本次失败、相同信号窗口与风险数学，先解决固定40域能否迁移及原始日档/身份/有效日影响；不继续搜索止损宽度、放宽回撤或新增网络。不同有效机制组合、外部事件覆盖与神经网络净增量仍按原目标保持未完成。

冻结计划和代码：research_tools/trend_stop_audit/；执行器新版本：Freqtrade research/crypto/sparse_coststop_v1.py，原引擎未改。冻结输入：results/research/trend_coststop_inputs_20261027_v001/；5条连续金融账户及实际新进程恢复：results/research/trend_coststop_accounts_20261027_v001/。来源、公式、引擎、账本和核验哈希见verification.json；数据/结果按既有规则本地封存。哈希绑定材料身份，不证明市场事实。没有下单、部署或推送。
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
            "max_nav_error": max(nav_errors),
            "capacity_triggered": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ["inputs", "run", "output"]:
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    review(a.inputs, a.run, a.output)
