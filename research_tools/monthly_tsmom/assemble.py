"""Attach readable conclusions and explicit reserves to the versioned financial review."""

import argparse
import math
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def assemble(output):
    if (output / "completion.json").exists():
        raise FileExistsError("Preserve final reviewed report")
    run = Path("results/research/monthly_tsmom_accounts_20261027_v001")
    verify_manifest(run)
    metrics = read(output / "metrics.json")
    attribution = read(output / "account_attribution.json")
    summary = read(run / "summary.json")
    pieces, rates = {}, []
    for m in metrics:
        folder = run / "accounts" / m["id"]
        state = read(folder / "resume_state.json")
        config = read(folder / "result.json")["config"]
        reserve = math.fsum(
            p["q"]
            * state["previous_prices"][s]
            * (config["fee_bps"] + config["slippage_bps"])
            / 10000
            for s, p in state["positions"].items()
        )
        price = m["net_pnl"] - m["funding_net"] + m["fees"] + m["slippage_cost"] + reserve
        assert (
            abs(price + m["funding_net"] - m["fees"] - m["slippage_cost"] - reserve - m["net_pnl"])
            < 1e-7
        )
        pieces[m["id"]] = {
            "price_component_before_paid_costs_and_exit_reserve": price,
            "funding_net": m["funding_net"],
            "paid_fees": m["fees"],
            "paid_slippage": m["slippage_cost"],
            "unpaid_exit_reserve": reserve,
            "net_pnl": m["net_pnl"],
            "scope": "Same actual path arithmetic only; no cost-free, no-stop or closed-inventory counterfactual. Earlier account_attribution gross_price_component_same_path retains the unpaid exit reserve; this file separates it explicitly.",
        }
        n, p, z = m["closed_trades"], m["win_rate"], 1.959963984540054
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        rates.append(
            {
                "id": m["id"],
                "closed_cycles": n,
                "win_rate": p,
                "Wilson95": [center - radius, center + radius],
                "scope": "Descriptive binomial approximation; correlated cycles are not independent discoveries. Paired time-block intervals and source/search/unseen limits remain separate.",
            }
        )
    registry = Path("results/research/sparse_market_cohort_20261026_v003/registry.json")
    card = next(c for c in read(registry)["cards"] if c["symbol"] == "TONUSDT")
    for key in ["source_raw", "source_text"]:
        assert sha(Path(card[key])) == card[key + "_sha256"]
    gap = {
        "symbol": "TONUSDT",
        "diagnostic_month": "2026-06",
        "source": card,
        "zero_volume_and_no_reported_funding_days": [f"2026-06-{d:02d}" for d in range(24, 31)],
        "actual_trend_exit": "2026-06-01T15:50:00+00:00",
        "actual_exit_reason": "stop",
        "actual_TON_inventory_at_terminal": 0,
        "pure_full_month_source_index": "unknown; cannot fill flat prices/zero funding through termination or invent settlement cash",
        "next_required": "Before any new holding-rule variant, audit the source-style monthly horizon versus risk and lifecycle execution; a preterminal real exit must be modeled explicitly rather than silently treating the pure monthly index as complete.",
    }
    save(output / "cash_components.json", pieces)
    save(output / "win_rate_intervals.json", rates)
    save(output / "source_style_gap.json", gap)
    header = "| 方案 | 每边成本bps | 608日收益 | CAGR | 完整胜率 | 最大回撤 | 日夏普 | 完整周期 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    table = header + "\n".join(
        f"| {'月度趋势' if m['profile'] == 'monthly_tsmom' else '同幅度多头'} | {m['cost_bps']} | {m['net_return']:.4%} | {m['cagr']:.4%} | {m['win_rate']:.2%} | {m['max_drawdown']:.4%} | {m['sharpe']:.4f} | {m['closed_trades']} |"
        for m in metrics
    )
    years = [r for r in read(output / "periods.json") if r["kind"] == "year"]
    year_table = "| 方案／每边成本 | 2025全年 | 2026年1—8月（243日） |\n| --- | ---: | ---: |\n"
    for m in metrics:
        yr = {r["label"]: r for r in years if r["id"] == m["id"]}
        year_table += f"| {'趋势' if m['profile'] == 'monthly_tsmom' else '多头'}／{m['cost_bps']}bps | {yr['2025']['net_return']:.4%} | {yr['2026']['net_return']:.4%} |\n"
    stress = next(m for m in metrics if m["profile"] == "monthly_tsmom" and m["cost_bps"] == 15)
    text = f"""# 月度趋势：原始收益定义、连续账户与失效边界

2026-10-27。结论：本次冻结的12月经济收益方向／月度更新候选未获准入，两成本整段与闭合损益均负，且弱于同目标幅度多头。停止该固定实现，不追加窗口、风险或止损网格。没有反驳所有趋势方法，也没有完成原论文的完整跨资产复现。

## 原始方法与数据

[原始论文](https://fairmodel.econ.yale.edu/ec439/mosk.pdf)§2.1先复合日度期货超额收益，§4.1采用12月自身收益方向、1月持有及多头对照。我们不能把它直接等同币价涨跌；本地永续指数用每日前收盘归一的价格PnL减逐事件mark×rate资金费，再复合。现金收益假设0；指数不含交易费，不是账户盈利。

先冻结每月历史完整、日均报价额至少5000万USDT的前10名，再补缺失资金费；不按未来收益选币。20月、30个入选标的、200个不重复币月。两套同源日档生成400条完全相同的来源情景，不是400个独立样本。21币早期资金费用34次公开接口采集19698条；另1次对DEXE异常窗口的162条既有事件复查。全部请求和响应留存，未使用凭证或客户账户。

5个唯一币月在计入资金费后发生方向反转，说明原始收益定义确实会改变信号。DEXE 2026-08的过去12月价差约-64.44%，本地经济指数约+4.79%；异常窗口2026-07-21至27的162条时点、费率与mark和封存记录精确相同。当前API对照仍是同交易所渠道，不能证明客户到账、最初历史版本或完整资金费日程；也不是盈利因子证据。

Decimal逐日现金／复合与独立pandas EWM、全量月度资格排名已内部复算：最大日收益差4.44e-16、累计差1.42e-14、波动差1.11e-15。4个BTC未来价格及资金费扰动不改变过去序列／信号；是有限样例和内部代码核验，非外部第三方审计。

## 本地账户合同

每月00UTC更新；单位意图权重=方向×0.16/(当月标的数×事前年度波动)，合计超过1等比缩小；执行层再乘0.8敞口预算。COM60、有限12月归一权重、365日风险尺度。8%单仓价格收紧止损、15%连续账户回撤停机、最多10仓、五分钟报价额1%成交代理。股市需要自己的交易日历、分红／公司行动和账户验收，不能照搬此UTC接口。

与原文不同的永续指数、有限风险历史、16%预算、0.8覆盖、币种域、止损与生命周期退出全部披露；止损可以缩短名义月度持有。多头对照只是去掉方向符号，目标选币与幅度一致；实现风险、库存和持有路径不同。没有继承论文收益。2025/2026均已访问，属于历史诊断。

## 连续608日结果

2025-01-01至2026-09-01，初始1万USDT，手续费5bps加滑点3/10bps每边。含精确资金费、实际费用及开放库存退出准备。4条实际新账户，两套相同目标复用为8个情景；无新容量账户。

{table}

现金参考同窗口收益／回撤0，胜率和夏普无定义。胜率按完整持仓生命周期计算，Wilson描述区间及关联性限制另存win_rate_intervals.json；配对32日块方向增量区间跨零，128日仅4块当量，不宣称统计充分或完成全部搜索校正。

{year_table}

2026保留2025的库存／高水位／账户状态，不重置。部分年不可与全年视为等长样本；所有90日块保留在periods.json。四账户均未触发15%账户停机，亏损不能归因于跨年停机归零。

## 损益与持有过程

压力趋势净PnL{stress["net_pnl"]:.4f}USDT，闭合{stress["closed_net_pnl"]:.4f}，未结束生命周期含已结现金／浮动与退出准备{stress["open_campaign_net_pnl"]:.4f}，两者相加等于总损益。未结束2仓不当已实现收益；多头对照尚有8仓，开放损益明显不同，不能只比较闭合交易宣称方向改善。

趋势149个完整周期中139个止损、10个月度退出；多头132个中127个止损。压力趋势多／空闭合PnL分别-675.0554/-329.5974USDT。高止损占比是持有规则与月度命题兼容性的诊断线索，不是止损造成亏损的因果证明，也不能由此推断放宽止损会盈利。

全部价格、资金费、已付佣金／滑点和未付退出准备在cash_components.json精确分列。实际趋势路径即使算术加回已付成本与退出准备，价格组成仍负；这不是零费用或无止损账户重跑。趋势压力相对多头总损益少434.3307USDT；两者实际平均总敞口8.37%/10.36%，不能称实现风险匹配。

## 月度原命题与实际约束仍有未解项

账户全部封存后另登记事后诊断：按全部20个月的固定目标计算下一月源指数毛因子收益，检查信号与实际执行的差别，不能用于反选标的或当作净账户结果。TON 2026-06遇到已知6月23日终止，6月24—30日零成交、无已报资金费，纯满月指数无法构造。该月及整段均值保持未知，没有补平价格／零资金费或删除该月；其余19月只是描述值，不拼接成年化业绩。

已知通知明确终止和禁止开仓时点；其历史原文版本仍是假设可见。实际趋势TON仓在6月1日已止损，终止时无库存，因此没有把未知自动结算现金代入本次净值。最初review.py已执行版本保留；review_v2.py只修复“不完整月份展示部分合计”的报告缺陷，未重跑或改变金融账户。

下一步先审查月度预测／持有时钟与8%止损及退市退出的兼容性，核对TON退出前真实成交窗口和已知公告时点，区分信号失效、日内路径与强制退出的影响；不立即改宽止损或扩大参数／模型。该固定候选经济筛查失败，容量未触发；总账户20%/30%年化、15%回撤、夏普1与多机制组合仍未完成。

## 证据范围

4条现金／库存／精确资金费／止损审计通过，最大净值差1.02e-10USDT；2025-02-01持有10仓且加仓后，父PID28191与新PID28251恢复出的完整后续净值、事件、请求、库存和风险状态相同。这些是内部工程证据。外部第三方复现、真正未见数据、历史发布修订、实际客户现金与盘口容量均未完成，不能升级为未来收益保证。

来源、输入和冻结账户分别位于data/research/monthly_tsmom_early_funding_20261027_v001、results/research/monthly_tsmom_inputs_20261027_v001、results/research/monthly_tsmom_targets_20261027_v001、results/research/monthly_tsmom_accounts_20261027_v001。统一公式在factors/common/monthly_tsmom.py与factors/crypto/perpetual_returns.py。本报告completion绑定文件身份，哈希不证明市场事实。没有下单、部署或推送。
"""
    (output / "README.md").write_text(text)
    save(
        output / "assembly_verification.json",
        {
            "passed": True,
            "source_sha256": sha(Path(__file__)),
            "TON_registry_sha256": sha(registry),
            "cash_reserves_separated_and_reconciled": True,
            "cash_and_future_gap_claims_separate": True,
            "no_incomplete_month_partial_total_published": all(
                r["difference"] is None
                for r in read(output / "source_style_monthly_payoffs.json")
                if r["unknown_legs"]
            ),
            "financial_accounts_unchanged": True,
            "capacity_not_triggered": not summary["capacity_triggered"],
        },
    )
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
            "final_report": str(output),
            "unpaid_exit_reserves": {k: v["unpaid_exit_reserve"] for k, v in pieces.items()},
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    assemble(p.parse_args().output)
