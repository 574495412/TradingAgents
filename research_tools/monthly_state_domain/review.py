"""Audit retained gate paths and describe state attribution without selecting states."""

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.trend_stop_audit.review import partitions


def wilson(n, w):
    if not n:
        return None
    z = 1.959963984540054
    den = 1 + z * z / n
    center = (w + z * z / (2 * n)) / den
    radius = z * math.sqrt(w * (1 - w) / n + z * z / (4 * n * n)) / den
    return [center - radius, center + radius]


def inspect_account(folder, identity, policy):
    result = read(folder / "result.json")
    for filename, digest in result["source_hashes"].items():
        assert sha(folder / filename) == digest, filename
    audit = read(folder / "internal_cash_audit.json")
    stops = read(folder / "internal_stop_audit.json")
    assert audit["passed"] and stops["passed"] and stops["first_stop_requests"] == 0
    assert audit["result_sha256"] == sha(folder / "result.json")
    m, state = result["metrics"], read(folder / "resume_state.json")
    curve = pd.read_parquet(folder / "equity_5m.parquet").set_index("date")
    events, trades = read(folder / "events.json"), read(folder / "trades.json")
    daily = curve.loc[(curve.index.hour == 0) & (curve.index.minute == 0)]
    returns = daily.equity.pct_change(fill_method=None).dropna()
    assert abs(returns.mean() / returns.std(ddof=1) * math.sqrt(365) - m["sharpe"]) < 1e-10
    assert abs(float((1 - curve.equity / curve.equity.cummax()).max()) - m["max_drawdown"]) < 1e-10
    assert abs(math.fsum(t["net_pnl"] for t in trades) - m["closed_net_pnl"]) < 1e-6
    assert abs(m["closed_net_pnl"] + m["open_campaign_net_pnl"] - m["net_pnl"]) < 1e-6
    assert not result["config"]["price_stop_enabled"]
    assert not any(e.get("reason") == "stop" for e in events)
    periods, exposure = partitions(folder, identity)
    for p in periods:
        a, b = pd.Timestamp(p["start"]), pd.Timestamp(p["end"])
        dr = daily.equity.loc[a:b].pct_change(fill_method=None).dropna()
        cycles = [t for t in trades if a < pd.Timestamp(t["exit_time"]) <= b]
        p.update(
            daily_sharpe=float(dr.mean() / dr.std(ddof=1) * math.sqrt(365))
            if len(dr) > 1 and dr.std(ddof=1) > 0
            else None,
            complete_cycles_closed_in_period=len(cycles),
            complete_cycle_win_rate=float(np.mean([t["net_pnl"] > 0 for t in cycles]))
            if cycles
            else None,
            closed_campaign_pnl_in_period=math.fsum(t["net_pnl"] for t in cycles),
            closed_campaign_pnl_can_include_earlier_entries=True,
        )
    assert (
        abs(math.fsum(p["net_pnl"] for p in periods if p["kind"] == "year") - m["net_pnl"]) < 1e-6
    )
    reserve_notional = math.fsum(
        p["q"] * state["previous_prices"][s] for s, p in state["positions"].items()
    )
    reserve = (
        reserve_notional * (result["config"]["fee_bps"] + result["config"]["slippage_bps"]) / 10000
    )
    executed_notional = math.fsum(e.get("notional", 0) for e in events)
    extra_fee_room = m["net_pnl"] / (executed_notional + reserve_notional) * 10000
    groups = {}
    for dimension in ["symbol", "entry_month"]:
        grouped = defaultdict(list)
        for t in trades:
            key = (
                t["symbol"]
                if dimension == "symbol"
                else pd.Timestamp(t["entry_time"]).strftime("%Y-%m")
            )
            grouped[key].append(t)
        groups[dimension] = [
            {"key": key, "closed_cycles": len(ts), "net_pnl": math.fsum(t["net_pnl"] for t in ts)}
            for key, ts in sorted(grouped.items())
        ]
    row = {
        "id": identity,
        "policy": policy,
        "cost_bps_each_side": result["config"]["fee_bps"] + result["config"]["slippage_bps"],
        "risk_halted": state["risk_halted"],
        **m,
        "Wilson95_descriptive_only": wilson(m["closed_trades"], m["win_rate"]),
    }
    detail = {
        **exposure,
        "mean_gross_exposure": m["mean_gross_exposure"],
        "mean_absolute_net_exposure": m["mean_absolute_net_exposure"],
        "open_campaigns": len(state["positions"]),
        "open_campaign_net_pnl": m["open_campaign_net_pnl"],
        "unpaid_exit_reserve": reserve,
        "price_component_before_paid_costs_and_exit_reserve": m["net_pnl"]
        - m["funding_net"]
        + m["fees"]
        + m["slippage_cost"]
        + reserve,
        "funding_net": m["funding_net"],
        "fees": m["fees"],
        "slippage_cost": m["slippage_cost"],
        "cost_sensitivity": {
            "extra_fee_equivalent_bps_to_zero_same_fixed_path": extra_fee_room,
            "basis_filled_plus_reserved_exit_notional": executed_notional + reserve_notional,
            "scope": "Net PnL divided by retained executed-plus-reserved notional. Signed extra fee-equivalent allowance only; no resizing, execution-price or funding-response rerun, no certified capacity/cost ceiling.",
        },
        "cycle_holding_median_days": float(
            np.median(
                [
                    (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds()
                    / 86400
                    for t in trades
                ]
            )
        )
        if trades
        else None,
        "exit_reasons": dict(Counter(t["reason"] for t in trades)),
        "concentration": groups,
        "capacity_proxy": result["capacity"],
        "maximum_internal_NAV_error_usdt": audit["max_nav_error"],
    }
    return row, detail, periods, daily, events


def attribute(identity, daily, events, states):
    days = daily.index[:-1]
    observed = states.state.reindex(days)
    assert observed.notna().all()
    monthly = states.state.loc[states.index.day == 1].reindex(days, method="ffill")
    marked = daily.equity.diff().iloc[1:].to_numpy()
    cash = daily.cash.diff().iloc[1:].to_numpy()
    event_rows = {d: {"cash": [], "funding": [], "fees": [], "slippage": []} for d in days}
    for e in events:
        effective = pd.Timestamp(e.get("effective_time", e["time"])).ceil("5min")
        day = (effective - pd.Timedelta(1, unit="ns")).floor("D")
        assert day in event_rows, (identity, e)
        event_rows[day]["cash"].append(e["cash_delta"])
        event_rows[day]["funding"].append(e["cash_delta"] if e["type"] == "funding" else 0)
        event_rows[day]["fees"].append(e.get("fee", 0))
        event_rows[day]["slippage"].append(e.get("slippage_cost", 0))
    assert max(abs(cash[i] - math.fsum(event_rows[d]["cash"])) for i, d in enumerate(days)) < 1e-6
    rows = []
    for dimension, labels in [
        ("daily_observable_state", observed),
        ("monthly_anchor_state", monthly),
    ]:
        for label in sorted(labels.unique()):
            indices = np.flatnonzero(labels.eq(label).to_numpy())
            rows.append(
                {
                    "id": identity,
                    "dimension": dimension,
                    "state": label,
                    "calendar_days": len(indices),
                    "active_at_daily_boundary_days": int(
                        daily.gross_exposure.iloc[indices].gt(0).sum()
                    ),
                    "mean_daily_boundary_gross_ratio": float(
                        (daily.gross_exposure / daily.equity).iloc[indices].mean()
                    ),
                    "marked_NAV_change_usdt": math.fsum(marked[i] for i in indices),
                    "realized_cash_delta_usdt": math.fsum(cash[i] for i in indices),
                    "funding_cash_usdt": math.fsum(
                        v for i in indices for v in event_rows[days[i]]["funding"]
                    ),
                    "paid_fees_usdt": math.fsum(
                        v for i in indices for v in event_rows[days[i]]["fees"]
                    ),
                    "slippage_embedded_in_prices_usdt": math.fsum(
                        v for i in indices for v in event_rows[days[i]]["slippage"]
                    ),
                    "days_different_from_anchor_state": int(
                        (observed.iloc[indices] != monthly.iloc[indices]).sum()
                    ),
                }
            )
        assert (
            abs(
                math.fsum(r["marked_NAV_change_usdt"] for r in rows if r["dimension"] == dimension)
                - float(daily.equity.iloc[-1] - daily.equity.iloc[0])
            )
            < 1e-6
        )
    return rows


def review(run, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(run)
    summary, plan = read(run / "summary.json"), read(run / "plan.json")
    inputs = Path(plan["prepared_inputs"])
    verify_manifest(inputs)
    for filename, digest in plan["account_source_hashes"].items():
        assert sha(Path(filename)) == digest, filename
    for filename, digest in plan["source_hashes"].items():
        assert sha(Path(filename)) == digest, filename
    states = pd.read_parquet(inputs / "daily_archive_states.parquet")
    actual = [r for r in summary["account_rows"] if not r["reused_account"]]
    metrics, details, periods, attributes = [], {}, [], []
    jobs = [(run / "accounts" / r["id"], r["id"], r["profile"]) for r in actual]
    jobs += [
        (
            Path(plan["unconditional_run"])
            / "accounts"
            / f"disabled_{profile}_cap10000_slip{cost}",
            f"disabled_{profile}_cap10000_slip{cost}",
            "unconditional_" + profile,
        )
        for profile in ["monthly_tsmom", "magnitude_long"]
        for cost in plan["costs_slippage_bps"]
    ]
    for folder, identity, policy in jobs:
        row, detail, p, daily, events = inspect_account(folder, identity, policy)
        metrics.append(row)
        details[identity] = detail
        periods.extend(p)
        attributes.extend(attribute(identity, daily, events, states))
    cash = Path(
        "results/research/sparse_family_accounts_20261026_v002/accounts/cash_continuous2025_2026_cash_cap10000_slip10"
    )
    cash_result = read(cash / "result.json")
    assert cash_result["window"] == plan["window"] and cash_result["metrics"]["net_pnl"] == 0
    for filename, digest in cash_result["source_hashes"].items():
        assert sha(cash / filename) == digest
    cash_curve = pd.read_parquet(cash / "equity_5m.parquet")
    assert len(cash_curve) == 175105 and cash_curve.equity.eq(10000).all()
    anchors = read(inputs / "monthly_decisions.json")
    assert (
        summary["actual_new_financial_accounts"] == 4
        and summary["source_or_profile_alias_cases"] == 4
    )
    assert (
        summary["economic_followup_condition"]
        and not summary["minimum_sample_condition"]
        and not summary["capacity_triggered"]
    )
    restart = summary["actual_process_restart"]
    assert restart["passed"] and restart["parent_pid"] != restart["worker_pid"]
    stress = "daily_archive_conditional_trend_cap10000_slip10"
    assert read(run / "restart_worker.json")["resume_state"] == read(
        run / "accounts" / stress / "resume_state.json"
    )
    exposure = read(run / "exposure_summation_audit.json")
    assert exposure["passed"] and exposure["maximum_fraction_of_derived_sum_bound"] <= 1
    save(output / "metrics.json", metrics)
    save(output / "periods.json", periods)
    save(output / "holding_costs_and_concentration.json", details)
    save(
        output / "state_attribution.json",
        {
            "scope": "Descriptive allocation of each retained account's actual daily NAV change and realized cash; no simulated switching, hypothetical state returns, causal claim or winner selection. D00 to D+1 NAV change uses the state known at D; exact-midnight cash belongs to the ending day via effective time minus1ns. Slippage is embedded in prices, not an extra cash debit. Daily boundary exposure is not full5m mean exposure.",
            "rows": attributes,
        },
    )
    save(output / "monthly_decisions.json", anchors)
    save(output / "comparisons.json", summary["comparisons"])
    verification = {
        "schema": "monthly-state-domain-review.v1",
        "passed": True,
        "scope": "Internal evidence chain and arithmetic only; module admission, statistical confidence and external/unseen evidence not established.",
        "source_hashes": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path("research_tools/trend_stop_audit/review.py"),
                run / "completion.json",
                inputs / "completion.json",
                Path(plan["unconditional_finalization"]) / "completion.json",
                cash / "result.json",
            ]
        },
        "new_financial_accounts": 4,
        "same_source_aliases": 4,
        "new_domain_variants": 1,
        "new_state_thresholds_or_models": 0,
        "minimum_active_monthly_anchors": 4,
        "observed_active_monthly_anchors": 3,
        "closed_trend_cycles": [
            r["closed_trades"] for r in metrics if r["policy"] == "conditional_trend"
        ],
        "cash_control": {
            "id": cash.name,
            "net_return": 0,
            "net_pnl": 0,
            "equity_rows": len(cash_curve),
        },
        "account_arithmetic_and_cash_checks": 8,
        "maximum_internal_NAV_error_usdt": max(
            v["maximum_internal_NAV_error_usdt"] for v in details.values()
        ),
        "all_year_and_state_cashflow_partitions_reconciled": True,
        "same_realized_risk": False,
        "restart": restart,
        "exposure_precision": exposure,
        "economic_necessary_screen_passed": True,
        "module_admitted": False,
        "capacity_triggered": False,
        "independent_unseen_evidence": False,
        "third_party_review": False,
        "profit_goal_met": False,
        "orders_enabled": False,
        "production_eligible": False,
    }
    save(output / "verification.json", verification)
    metric_table = "\n".join(
        f"| {r['policy']} | {r['cost_bps_each_side']} | {r['net_return']:.4%} | {r['cagr']:.4%} | {r['win_rate']:.2%} | {r['max_drawdown']:.2%} | {r['sharpe']:.4f} | {r['closed_trades']} | {r['profit_factor']:.4f} |"
        for r in metrics
    )
    years = "\n".join(
        f"| {p['id']} | {p['label']} | {p['calendar_days']:.0f} | {p['net_return']:.4%} | {p['net_pnl']:.4f} | {p['within_period_drawdown']:.2%} |"
        for p in periods
        if p["kind"] == "year"
    )
    trend = next(r for r in metrics if r["id"] == stress)
    td = details[stress]
    months = td["concentration"]["entry_month"]
    month_table = "\n".join(
        f"| {m['key']} | {m['closed_cycles']} | {m['net_pnl']:.6f} |" for m in months
    )
    text = f"""# 冻结趋势行情门控：小幅正收益，证据不足

2026-10-27。唯一已登记变体：沿用现有月初00UTC时钟，既有行情为trend_up/trend_down时持有原月度趋势，其他月份现金；关闭单仓价格止损、保留15%持续账户停机。没有添加状态、调阈值、选择年份、日内重入或重训模型。

**结论：必要经济筛查为正，但不准入。** 两成本净收益及相对同门控多头增量为正；压力路径608日仅赚{trend["net_pnl"]:.6f} USDT，CAGR {trend["cagr"]:.4%}、日夏普{trend["sharpe"]:.4f}。30个完整周期来自仅3个活跃月，低于事前4个月底线；相关交易不等于30个独立发现。配对差区间跨零，长期块不足；无容量、第三方或未见数据认证。结果支持保留条件性假说，不证明高收益或稳定alpha，也不反驳全部条件性趋势。

## 完整账户对照

本金1万USDT，2025-01-01至2026-09-01共608日；每边手续费5bps+滑点3/10bps，含真实公开资金费与模型退出准备。新4条金融账户、4个同源精确目标别名，另复用4条既有账户与现金，未把它们算新发现。

| 方案 | 每边成本bps | 整段扣费收益 | CAGR | 完整周期胜率 | 五分钟回撤 | 日夏普 | 完整周期 | PF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{metric_table}
| 零息现金 | — | 0% | 0% | — | 0% | — | 0 | — |

胜率区间仅作描述，见[全部指标](metrics.json)。压力趋势胜率Wilson95为{trend["Wilson95_descriptive_only"][0]:.2%}至{trend["Wilson95_descriptive_only"][1]:.2%}；未按相关性校正成显著性。PF不能单独支持准入。

## 月份、敞口与机会集

仅2025-03-01下跌趋势、2025-05-01上涨趋势、2025-12-01下跌趋势激活；2026年月初没有符合此门控的时点。后者只是这个月初合同没有入场，不代表2026市场没有机会。2026零收益保留，不能剔除后再年化。

| 活跃入场月 | 完整周期 | 压力成本净PnL USDT |
| --- | ---: | ---: |
{month_table}

压力门控平均实际总敞口{trend["mean_gross_exposure"]:.2%}，无门控趋势约16.48%；平均绝对净敞口{trend["mean_absolute_net_exposure"]:.2%}。同幅度多头与方向对照的实际风险仍不完全相同。少亏/改善部分伴随大幅减少持仓，不称同风险收益提升；不得把空仓月份拼成新的全环境策略。

| 账户 | 年/部分年 | 日数 | 区间净收益 | 净PnL USDT | 区间内回撤 |
| --- | --- | ---: | ---: | ---: | ---: |
{years}

2026仅243日，承接2025账户状态。90日块的收益、区间内回撤、日夏普和按退出时点归属的完整周期结果均见[分段表](periods.json)；区间内回撤不替代连续总回撤。

## 原始状态与反泄漏核验

完整860历史名称按当时身份/当前episode的180有效观测、完整60日收益/BTC历史及前28日均报价额≥5000万美元筛选前30名作市场背景；交易池仍是原已冻结12个月完整历史的前10。背景与交易域明确分开，不用未来入选币并集作为市场。

原有BTC状态参数及优先级完全不变，全部608日标签与旧固定40池相同，但广度最大变化0.3412、成员数最大变化18。因此不能把旧HMM/MLP的特征或模型直接当作新域有效证据。

独立标量/排名代码重算两来源共1216个日时点；4组完整历史前缀与未来扰动检查通过。同源归档/API修订解释的目标相同而复用，不等于两个独立证据源。原始解析和身份政策仍为共享依赖，不能称外部审计。

## 现金、成本和状态归因

压力趋势闭合PnL {trend["closed_net_pnl"]:.6f}，开放生命周期PnL {trend["open_campaign_net_pnl"]:.6f}；资金费{trend["funding_net"]:.6f}、佣金{trend["fees"]:.6f}、滑点{trend["slippage_cost"]:.6f}，未付退出准备{td["unpaid_exit_reserve"]:.6f} USDT。相同已实现路径额外收费的盈亏平衡余量约{td["cost_sensitivity"]["extra_fee_equivalent_bps_to_zero_same_fixed_path"]:.4f}bps：只是原成交量下的费用敏感性，不是重新成交/缩放后的成本或容量认证。

[状态归因](state_attribution.json)完整列出所有状态下的实际每日净值变化、现金、资金费与费用，分别按每日可知状态及月初采样状态分组；保留跨状态持仓和全部亏损。每日D至D+1净值差归D已知状态，整点现金归结束的那一日，滑点已嵌在成交价中。分组是描述，不能用来再选盈利状态或宣称因果。

实际跨进程恢复在2025-03-01T00:05UTC持有10币时进行；净值、现金、事件、请求、库存及最终风险状态精确相同。敞口分别从持仓和真实前根5分钟价格重建，误差在推导的binary64求和界内。见[核验](verification.json)及[账户比较/区间](comparisons.json)。没有为修复检查而重跑金融收益。

## 本分支结束与下一证据

冻结这一门控，停止状态子集、阈值、时钟、风险预算或模型搜索。当前结论是**未知且不准入**，不是已找到可用高收益因子；只有3个月、区间跨零、已看历史和外部复现缺失，容量按登记条件未触发。新增未见的同合同完整机会记录才可补证，不能把同一历史重新切分当新样本。

如转向另一机制，先提出新的经济支付/风险约束与原始事实要求，再预登记；当前弱正收益不能用来启动更复杂配置或放大仓位。现有总账户20%/30%收益目标、共享资金组合、独立未见和外部复现仍未完成。不开实盘，不部署，不推送。
"""
    (output / "README.md").write_text(text)
    manifest = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()}
    save(
        output / "completion.json", {"passed": True, "manifest": manifest, "profit_goal_met": False}
    )
    print(
        {
            "report": str(output),
            "gate_return_stress": trend["net_return"],
            "active_months": len(months),
            "module_admitted": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    review(a.run, a.output)
