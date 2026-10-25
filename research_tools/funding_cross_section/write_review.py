"""Publish a compact review receipt without rerunning or selecting financial paths."""

import argparse
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def review(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path("results/research/funding_cross_section_publication_20261026_v001")
    run = Path("results/research/funding_cross_section_accounts_20261026_v001")
    inputs = Path("results/research/funding_cross_section_inputs_20261026_v002")
    data = Path("data/research/funding_cross_section_data_20261026_v001")
    attribution = Path("results/research/funding_cross_section_attribution_20261026_v001")
    x = read(root / "summary.json")
    receipt = read(root / "completion.json")
    for name, digest in receipt["manifest"].items():
        if Path(name).name != name or sha(root / name) != digest:
            raise ValueError("Published evidence changed")
    sources = [
        root / "completion.json",
        run / "completion.json",
        inputs / "completion.json",
        inputs.with_name(inputs.name + "_independent_audit.json"),
        data / "completion.json",
        attribution / "completion.json",
        Path(__file__),
        Path("research_tools/funding_cross_section/plan.json"),
    ]
    bound = {str(p): sha(p) for p in sources}
    sensitivity = []
    rows = {r["id"]: r for r in x["account_rows"]}
    for pair in x["source_comparisons"]:
        a, b = [rows[pair[k]] for k in ["archive_account", "api_revision_account"]]
        curves = [
            pd.read_parquet(
                Path(r["source_run"]) / "accounts" / r["source_account_id"] / "equity_5m.parquet"
            )
            for r in [a, b]
        ]
        delta = (curves[1].equity - curves[0].equity).abs()
        sensitivity.append({**pair, "max_abs_nav_difference_usdt": float(delta.max())})
    if x["profit_goal_met"] or x["portfolio_built"] or x["production_eligible"]:
        raise ValueError("Pilot cannot promote total portfolio")
    summary = {
        "schema": "pure-funding-compact-review.v1",
        "actual_accounts": x["actual_accounts"],
        "reused_references": x["reused_reference_accounts"],
        "audit": x["audit"],
        "restart_checks": x["restart_checks"],
        "input_audit": x["input_audit"],
        "source_data": read(data / "summary.json"),
        "source_comparisons": sensitivity,
        "module_assessments": x["module_assessments"],
        "portfolio_actual": None,
        "profit_goal_met": False,
        "capacity_followups": x["selection"]["capacity_followups"],
        "expanded_universe_certified": False,
        "independent_oos": False,
        "production_eligible": False,
        "orders_enabled": False,
        "records": [
            {
                **r,
                "attribution": x["details"][r["id"]]["same_executed_path_attribution"],
                "periods": x["details"][r["id"]]["periods"],
                "state_contributions": x["details"][r["id"]]["state_contributions"],
            }
            for r in x["account_rows"]
        ],
    }
    save(output / "summary.json", summary)
    save(
        output / "verification.json",
        {
            "passed": True,
            "source_hashes": bound,
            "account_and_period_evidence_checked": True,
            "new_financial_accounts_in_reporting": 0,
            "failed_input_release_preserved": "results/research/funding_cross_section_inputs_20261026_v001",
            "input_retry_reason": "Parquet omits derived index.freq metadata; v002 checks identical timestamps, columns and values with check_freq=False. No model or financial parameter change.",
            "transport_failures_preserved": [
                "raw/bars_BNBUSDT.json",
                "recovery/bars_BNBUSDT_attempt1.json",
            ],
            "transport_recovery": "recovery/bars_BNBUSDT_attempt2.json",
            "financial_execution_source_commit": "4712882",
        },
    )
    save(
        output / "registered_plan.json",
        read(Path("research_tools/funding_cross_section/plan.json")),
    )
    pct = lambda v: "未知" if v is None else f"{v * 100:.2f}%"
    num = lambda v: "未知" if v is None else f"{v:.2f}"
    tables = [
        "| 区间 | 方案 | 扣费收益 | 净胜率 | 回撤 | 日夏普 | 闭合PF | 完整周期 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    phases = {
        "screen2025": "2025全年",
        "diagnostic2026": "2026年1—8月冷启动",
        "continuous2025_2026": "跨年连续608日",
    }
    for r in x["account_rows"]:
        if r["variant"] != "archive" or r["slippage_bps"] != 10:
            continue
        m = r["metrics"]
        tables.append(
            f"| {phases[r['phase']]} | {x['candidate_names'][r['candidate']]} | {pct(m['net_return'])} | {pct(m['win_rate'])} | {pct(m['max_drawdown'])} | {num(m['sharpe'])} | {num(m['profit_factor'])} | {m['closed_trades']} |"
        )
    text = (
        """# 纯资金费价差与行情来源敏感性研究

30个新连续账户、6个封存动量引用、6个封存现金引用已完成，两个资金费模块均未取得当前登记机会集的组合准入证据。不是按单策略全年20%淘汰：跨年两成本扣费市值损益和闭合损益本身为负。多策略总账户尚未构建，20%／30%收益目标、回撤与容量验收继续未完成。

以下本金均为10,000USDT，费用5bps＋滑点10bps每边；胜率按完整闭合周期扣全部成本计算，净值包含未平仓损益和退出准备。现金各窗口收益为0、回撤0，胜率和夏普未知。第二行情版本的期末收益、回撤和夏普与下表相同，但少量中间净值不同。

"""
        + "\n".join(tables)
        + """

## 固定机制与复用范围

统一定义位于 `factors/crypto/funding_cross_section.py`，不改变旧74列。取决策前7个自然日实际已结算资金费之和，按原MAX共同资格筛选：180个既往日、前28日平均成交额≥5千万USDT、最多30名、最少10名、BTC排除排名、MAX/动量/beta观测有效。做多最低4名、做空最高4名，单位总敞口1，账户初始预算0.8；周一UTC00及冷启动首时点调仓，无每日补齐权重。

固定过滤版仅在过去7日延续假设下的资金费收入超过0.375%单位总敞口时持仓。该门槛来自1.25×完整进出×15bps每边，非刷参数结果。两版共享资金费机制，不能计成两个独立收益源。排名公式自己实现，未执行外部交易代码。

公开来源固定为 [funding-rate-alpha](https://github.com/OctopusTakopi/funding-rate-alpha/tree/cf9f87c01d3165bbc3bf34438baee289d248ffca) 与 [alpha-research](https://github.com/bryanvine/alpha-research/tree/17b8b5e1c79af194f733544517d82e3cf12c2259)。引用范围为已归档静态代码中的资金费假设，不继承其收益、现货价格代理、未来成员筛选、截断或缺失持仓收益归零。来源审阅见 `../historical_universe_20261026/README.md`。

## 2026改善尚未形成稳定收益证据

高成本纯资金费方案2026冷启动净赚1,612.20USDT：同一路径扣费前价格PnL1,490.85，真实资金费739.22，手续费201.25，滑点402.49，未平仓退出准备14.14。闭合PnL1,426.86，开放周期总PnL185.34。它包含明显价格贡献，不能解释为无价格风险套利。

2025同方案价格PnL−585.68、资金费仅29.36，扣手续费74.38和滑点148.76后亏779.47；高水位回撤15.0576%导致永久研究停机，跨年保留。固定过滤版跨年价格PnL−1,209.85，资金费378.97仍不能抵扣价格损失和费用。资金费价差足够大不保证下一周持续，也不控制横截面价格风险。

2026纯资金费方案191个完整周期、闭合PF1.206，但按币及入场月聚类的单周期期望95%区间均跨0；过滤版只有3个入场月簇。243日冷启动年化是数学外推，不能当作完整年度或独立样本外证明。此轮未根据2026结果重新选行情、反转方向或降低阈值。年份是诊断分组，不是可交易的行情状态。

## 来源与财务证据

40币补入861条2024-12-25至2025-01-01前的真实毫秒资金费、结算mark；与既有2025—2026精确资金费合并。原归档不变；另一版本仅用官方API替换2025-01-29的40币、各288条记录，161条bar在至少一个采纳字段不同。18组成对成本／窗口／候选的最终收益、回撤、夏普及资金费净额相同，但32个成对净值观测点存在差异；最大差额见summary.source_comparisons。不能据此宣称全行情源一致或历史认证通过。

1,062个信号锚点全部经独立NumPy日线、资格和资金费排名复算，最大特征/权重误差约2.18e−14。两源×3个日期实际扰动未来价格、成交额和资金费，既往特征与当时权重不变。已登记资金费和原动量对照保持相同资格与引擎；6个动量引用重新独立审计，并未重复计成新样本。

30个新账户逐事件重建现金、资金费所有权、加减仓成本、持仓、未平仓净值、完整周期和成交量限制，最大净值误差4.92e−11USDT。8次新解释器恢复中，4个断点各有8个仓位，另4个跨年断点保留停机/风险状态；全5分钟曲线、账本、请求和最终JSON逐项相同。

两方案均未达到预登记的两种来源×两档成本连续正市值与闭合损益且不停机条件，因此未追加10万/100万容量账户；容量未知，不是0。执行仍为5分钟收盘价格和成交额1%代理，未认证交易所mark净值、订单簿、强平与实盘成交。

## 限制与下一步

40币池为事后选择；扩展历史币池、合约重上线分段、终止后实际现金价格/费率仍未认证。源冲突和旧失败保留。此轮只是融资价差的有界诊断，不替代268个历史入选名字的状态/容量研究。已有历史已查看，不能重新命名独立样本外。原MLP、事件、条件域、币对、分账carry与MAX结果完整保留。

下一步须先验证可观察环境及价格暴露是否解释失效，并补齐历史合约可交易状态；任何新增过滤/对冲应预登记在新版本，以完整账户比较并保留反证，不直接把2026赢家路由回旧历史。没有两个经过条件性验收的独立机制前，配置、总账户收益和前向证据任务保持未完成。未下单、部署或推送。

复算入口：先执行prepare_data、prepare_inputs、audit_inputs，再run_accounts、verify_restarts、diagnose、publish、write_review；全部位于`research_tools/funding_cross_section/`，输出使用全新目录，既有输出拒绝覆盖。参考本目录registered_plan.json、verification.json和summary.json的源SHA与结果收据。
"""
    )
    (output / "README.md").write_text(text)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "report_complete": True,
            "profit_goal_met": False,
        },
    )
    print(
        {
            "review_complete": True,
            "records": len(x["account_rows"]),
            "source_pairs": len(sensitivity),
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    review(p.parse_args().output)
