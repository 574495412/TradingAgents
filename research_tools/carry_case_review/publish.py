"""Publish selected-case findings with scope, original failures and internal evidence."""

import argparse
import shutil
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def publish(baseline, buffer, output):
    for root in [baseline, buffer]:
        verify_manifest(root)
        for p, h in read(root / "registration.json")["source_hashes"].items():
            if sha(p) != h:
                raise ValueError("Executed source changed " + p)
    original = read(baseline / "summary.json")
    extended = read(buffer / "summary.json")
    records = original["case_records"]
    extra = extended["case_records"]
    if (
        len(records) != 4
        or len(extra) != 4
        or any(r["metrics"]["net_pnl"] >= 0 for r in [*records, *extra])
    ):
        raise ValueError("Findings must match all actual case paths, not a selected cost")
    if any(r["attribution"]["open_pairs"] for r in extra):
        raise ValueError("Buffered paths are not fully flat")
    restarts = original["process_restores"] + extended["process_restores"]
    if len(restarts) != 8 or any(
        not r["passed"] or not r["open_pairs_at_split"] or r["first_pid"] == r["restored_pid"]
        for r in restarts
    ):
        raise ValueError("Real nonflat new-process evidence incomplete")
    output.mkdir(parents=True, exist_ok=False)
    source = Path("data/research/carry_exit_cases_20261026_v003")
    goal = Path("docs/research/regime_program/targets.json")
    registered = read(source / "registration.json")["source_hashes"][str(goal)]
    if sha(goal) != registered:
        raise ValueError("Capture the execution-time goal snapshot before updating progress")
    shutil.copyfile(goal, output / "execution_goal_snapshot.json")
    table = []
    details = []
    for symbol in ["BNXUSDT", "HIFIUSDT"]:
        first = [r for r in records if r["symbol"] == symbol]
        last = [r for r in extra if r["symbol"] == symbol]
        rows = {r["slippage_bps"]: r for r in first}
        after = {r["slippage_bps"]: r for r in last}
        r = after[10]
        a = r["attribution"]
        p = baseline / "cases" / (symbol + "_10")
        b = buffer / "cases" / (symbol + "_10")
        events = read(b / "events.json")
        exit_events = [e for e in events if e["type"] == "exit"]
        entries = [e for e in events if e["type"] == "entry"]
        basis = a["spot_gross_price_pnl"] + a["perp_gross_price_pnl"]
        facts = {
            "symbol": symbol,
            "case_start": r["case_start"],
            "original_exit_cutoff": rows[10]["exit_deadline"],
            "buffer_replay_end": r["exit_deadline"],
            "actual_model_last_exit": exit_events[-1]["time"],
            "onebar_net_by_slippage": {str(k): v["metrics"]["net_pnl"] for k, v in rows.items()},
            "buffer_net_by_slippage": {str(k): v["metrics"]["net_pnl"] for k, v in after.items()},
            "onebar_residual_by_slippage": {
                str(k): read(baseline / "cases" / (symbol + "_" + str(k)) / "positions.json")
                for k in rows
            },
            "buffer_residual_pairs": 0,
            "buffer_had_hedge_failure": r["capacity_proxy"]["hedge_failures"] > 0,
            "price_leg_net_pnl": basis,
            "funding": a["funding"],
            "fees": a["fees"],
            "cash_slippage": a["cash_decomposition_slippage"],
            "net_pnl": a["net_pnl"],
            "paid_funding_events": a["paid_events"],
            "paid_rate_sum": a["paid_rate_sum"],
            "spot_gross_price_pnl": a["spot_gross_price_pnl"],
            "perp_gross_price_pnl": a["perp_gross_price_pnl"],
            "actual_entry_notionals": {e["market"]: e["notional"] for e in entries},
            "cash_baseline_pnl": 0,
            "maximum_observed_case_drawdown": r["metrics"]["max_drawdown"],
            "capacity_proxy_only": r["capacity_proxy"],
            "net_cash_identity_error": a["reconciliation_error"],
            "cagr": None,
            "sharpe": None,
            "population_win_rate": None,
            "strategy_admitted": False,
        }
        details.append(facts)
        table.append(
            f"| {symbol} | {rows[3]['metrics']['net_pnl']:.4f} | {rows[10]['metrics']['net_pnl']:.4f} | {after[3]['metrics']['net_pnl']:.4f} | {after[10]['metrics']['net_pnl']:.4f} | {a['paid_events']} | {a['funding']:.4f} | {basis:.4f} |"
        )
    report = {
        "schema": "carry-exit-depth-review.v1",
        "selected_cases": 2,
        "case_paths": 8,
        "cash_paths": 4,
        "financial_strategy_accounts": 0,
        "all_eight_case_paths_negative": True,
        "cases": details,
        "internal_audit_maximum_nav_error": max(
            r["internal_arithmetic_audit"]["maximum_nav_error"] for r in [*records, *extra]
        ),
        "internal_audited_nav_points": sum(
            r["internal_arithmetic_audit"]["curve_rows"] for r in [*records, *extra]
        ),
        "process_restores": restarts,
        "baseline_prefix_audits": [
            read(
                buffer
                / "cases"
                / (r["symbol"] + "_" + str(r["slippage_bps"]))
                / "baseline_prefix_audit.json"
            )
            for r in extra
        ],
        "interpretation": "Selected diagnostic cases only. Single-bar residual does not prove inability to exit. Buffer clears HIFI but does not create net edge. Reported funding and the price/basis path must be evaluated together.",
        "next_action": "Stop expanding this high-past-funding branch or fitting thresholds on these cases. Explain the fixed40 versus causal-pool result with one preregistered single-variable ablation, keeping the full selection/search history and all failures.",
        "source_hashes": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                baseline / "completion.json",
                buffer / "completion.json",
                source / "completion.json",
                output / "execution_goal_snapshot.json",
            ]
        },
        "engineering_compatibility": {
            "v2_engine_legacy_cash_funding_partial_hedge_tests_passed": 7,
            "command": "FQ .venv/bin/python -m pytest --noconftest -q /tmp/test_spot_account_v2_compatibility.py",
            "scope": "Same frozen cash-conservation, exact-funding, partial-leg and restore tests loaded against v2; not economic proof",
        },
        "profit_goal_met": False,
        "portfolio_actual": None,
        "external_third_party_review": False,
        "independent_oos": False,
        "orderbook_capacity_certified": False,
        "customer_paid_cashflows_certified": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "verification.json", report)
    table = "\n".join(table)
    text = f"""# 两个高资金费退出案例：资金费、基差与完整退出

**BNX与HIFI在两个成本、两种退出窗口下的8条病例路径均亏损。** HIFI单根五分钟退出有现货残余，保留已知退市前缓冲后在下一根五分钟清仓，但仍亏损；不能把原来的单根窗口残仓解释成625USDT现货预算无法退出。

这是两个事先选中的历史解释案例，不是新的独立策略试验、统计充分样本或未见验证。每例参考现金1万USDT，只启用一个625USDT现货预算槽位，永续匹配净基础币数量；另外三个槽位留现金，永续钱包初始70%。HIFI入场时永续有溢价，因此同币量不等于同USDT名义，两腿实际入场名义分别保留。不能据此推算年化、稳定胜率、夏普或全组合容量；这些值均为空。

## 预登记结果（USDT）

| 案例 | 单根退出·滑点3 | 单根退出·滑点10 | 缓冲退出·滑点3 | 缓冲退出·滑点10 | 支付至模型的事件 | 报告资金费入账 | 缓冲价格腿净损益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

滑点单位为每腿每边bps，另计现货10bps与永续5bps佣金。单根HIFI数值是含未平现货和退出准备的按市值结果，缓冲结果已完全平仓；两者不能说成同一退出期的纯价格收益增量。四条现金对照均为零，重复成本与退出路径不是独立经济样本。

压力情景中，BNX现货价格收益187.07与永续价格亏损191.14抵消后净价腿−4.07，报告资金费+4.39，佣金−2.14、现金分解滑点−2.85，净亏−4.66。匹配币量只对冲主要方向风险，没有消除基差变化。

HIFI缓冲退出后，现货价格损失−572.77与永续价格收益+567.49抵消后净价腿−5.28，报告资金费+0.29，佣金−1.44、滑点−1.92，净亏−8.35。退出期间永续先平、现货后平，保留这一段单腿风险及失败对冲标记，没有假设原子成交。模型最后退出为2025-09-17 02:35UTC，距离实际现货终止03:00仍有缓冲。

## 什么假设被反驳，什么仍未知

- “高过去七日资金费等于持有期可赚收益”不成立于这两例。BNX实际持仓支付事件费率合计0.569888%，高于先前简单缩时预测0.085438%；HIFI实际合计0.066714%，低于预测0.532536%。费率变化、实际持有时段与不同标记价共同决定现金，不能把按小时缩放当正确预测，也不能按过去收益选择新阈值。
- “逐币现货对冲就能保留下资金费净优势”没有在这两例得到支持，基差/退出路径与成本仍吞噬收益。该结论不是对所有资金费策略的否定，更不是因单策略没达20%而淘汰。
- “单根末端残仓证明容量不足”被缓冲对照反驳。原退出提交时点不变，只让挂起退出使用已知终止前的剩余时间；BNX结果不变，HIFI由残仓变为完全平仓，亏损仍在。仍然只有成交额1%的bar容量代理，没有真实盘口、历史最小量/价格档位或客户实际成交认证。

## 原始事实与内部复核

只新增两份现货及两份标记价官方月归档，另复用已封存永续CSV和原始资金费JSON响应。经历两次仅数据类型比较失败，原v001/v002与后续成功v003均保留；总16次HTTP在登记上限内，失败未产生财务路径。单独一次输入审核失败是HIFI公告写“该币所有现货交易对”，并未逐字写HIFI/USDT；修正为核对明确币名及全现货产品范围，未允许相似ticker或标题推断。它们都不算经济试验。

对实际病例窗口逐行重读原CSV/Decimal，核对价格、毫秒资金费事件、标记价与原始公告。BNX按已知永续08:30截止，HIFI按已知现货03:00前30分钟提交退出；不能把后来才发布的HIFI永续终止公告用于原决策。缓冲对照沿用同一原始ZIP、零新网络请求，硬结束在当时已知最早市场终止前5分钟，原切点前净值逐点完全一致。

内部第二代码路径复算全部钱包、库存、浮动盈亏、退出准备与净值，最大误差小于4e−12USDT；8次实际落盘JSON检查点由不同PID的新解释器恢复，全部含未平仓及已结资金费，曲线、账本、请求与最终状态一致。这是内部工程复核，尚非第三方审计。币本位买入手续费按实际成交价折币；现金分解滑点按净币量计算，与引擎按毛币量报告的滑点分列，避免重复计入手续费币的滑点。

公开历史报告资金费是模型输入，不是本客户实际到账证据。当前公告正文按宣称发布时间加5分钟可见仍为研究假设，原始历史版本、地域/客户交易资格、实际费率和应付结算未认证。BNX/HIFI官方套利机器人停止时间不同于手动现货/永续停止时间，本研究没有把机器人服务停止当作所有手动交易终止，也未证明客户当时有权使用手动交易。五分钟历史成交收盘价加滑点不证明同量真实订单必可成交。

## 下一项

停止扩展本高过去资金费分支或在两例上拟合门槛。下一项仅做已登记公式的单变量反证，解释固定40币池与时点币池收益差异，保持其余资金、风险、执行与数据版本一致；无新净优势证据前不增加市场、模型或神经网络。多策略20%基础/30%挑战、15%回撤、夏普1和未见验证继续未完成。

源码在 `research_tools/carry_case_review/`；原始与缓冲账户分别为 `results/research/carry_exit_case_accounts_20261026_v001`、`results/research/carry_exit_buffer_accounts_20261027_v001`。当时目标快照保存在本目录，正式目标的进度更新不覆盖执行时定义。未下单、部署、推送或读取凭证。
"""
    (output / "README.md").write_text(text)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_strategy_accounts": 0,
            "profit_goal_met": False,
        },
    )
    print(
        {
            "report": str(output),
            "selected_cases": 2,
            "case_paths": 8,
            "all_negative": True,
            "profit_goal_met": False,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--buffer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    publish(args.baseline, args.buffer, args.output)
