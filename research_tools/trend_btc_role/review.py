"""Close the benchmark-role explanation without borrowing original-paper returns."""

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
    audit_path = Path(str(inputs) + "_independent_audit.json")
    audit = read(audit_path)
    assert audit["passed"] and audit["inputs_completion_sha256"] == sha(inputs / "completion.json")
    assert not summary["failures"] and summary["actual_new_financial_accounts"] == 4
    assert not summary["capacity_triggered"] and not summary["capacity_queue"]
    source = Path(plan["inputs"]["method_source"])
    verify_manifest(source)
    paper = read(source / "review.json")
    assert sha(source / "moskowitz_ooi_pedersen_2012.pdf") == paper["pdf_sha256"]
    records = {}
    errors = []
    periods = []
    exposure = {}
    hashes = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            audit_path,
            inputs / "completion.json",
            run / "completion.json",
            source / "completion.json",
            Path("research_tools/trend_stop_audit/review.py"),
        ]
    }
    BTC = {}
    for row in summary["account_rows"]:
        if row["reused_account"]:
            continue
        identity = row["id"]
        folder = run / "accounts" / identity
        result = read(folder / "result.json")
        assert sha(folder / "result.json") == row["result_sha256"]
        for name, digest in result["source_hashes"].items():
            assert sha(folder / name) == digest
        cash = read(folder / "internal_cash_audit.json")
        stop = read(folder / "internal_stop_audit.json")
        assert cash["passed"] and stop["passed"] and cash["result_sha256"] == row["result_sha256"]
        errors.append(cash["max_nav_error"])
        p, e = partitions(folder, identity)
        periods.extend(p)
        exposure[identity] = e
        assert (
            abs(
                math.fsum(x["net_pnl"] for x in p if x["kind"] == "year")
                - row["metrics"]["net_pnl"]
            )
            < 1e-6
        )
        trades = read(folder / "trades.json")
        BTC[identity] = {
            "complete_cycles": sum(t["symbol"] == "BTCUSDT" for t in trades),
            "closed_net_pnl": math.fsum(t["net_pnl"] for t in trades if t["symbol"] == "BTCUSDT"),
        }
        records[(row["profile"], row["slippage_bps"])] = result
        hashes.update(
            {
                str(folder / name): sha(folder / name)
                for name in ["result.json", "internal_cash_audit.json", "internal_stop_audit.json"]
            }
        )
    for row in summary["excluded_references"]:
        if row["variant"] != "daily_archive":
            continue
        folder = Path(row["source_run"]) / "accounts" / row["source_account_id"]
        result = read(folder / "result.json")
        assert sha(folder / "result.json") == row["result_sha256"]
        records[("btc_excluded", row["slippage_bps"])] = result
        hashes[str(folder / "result.json")] = sha(folder / "result.json")
    for name in ["btc_tradable", "magnitude_long"]:
        a = pd.read_parquet(inputs / f"daily_archive_{name}_targets.parquet")
        b = pd.read_parquet(
            inputs / f"daily_with_complete_5m_reconciliation_{name}_targets.parquet"
        )
        assert a.equals(b)
    for v in plan["variants"]:
        a = pd.read_parquet(inputs / f"{v}_btc_tradable_targets.parquet")
        b = pd.read_parquet(inputs / f"{v}_magnitude_long_targets.parquet")
        pd.testing.assert_frame_equal(
            a.filter(like="target_").abs(), b.filter(like="target_"), check_exact=True
        )
    cash_folder = Path(
        "results/research/sparse_family_accounts_20261026_v002/accounts/cash_continuous2025_2026_cash_cap10000_slip10"
    )
    cash = read(cash_folder / "result.json")
    for name, digest in cash["source_hashes"].items():
        assert sha(cash_folder / name) == digest
    assert (
        cash["window"] == plan["window"]
        and cash["metrics"]["net_return"] == 0
        and not cash["open_positions"]
    )
    hashes[str(cash_folder / "result.json")] = sha(cash_folder / "result.json")
    restart = summary["actual_new_process_restart"]
    assert restart["passed"] and restart["parent_pid"] != restart["worker_pid"]
    assert read(run / "restart_worker.json")["resume_state"] == read(
        run / "accounts/daily_archive_btc_tradable_cap10000_slip10/resume_state.json"
    )
    assert all(r["metrics"]["net_pnl"] < 0 and r["risk_halted"] for r in summary["account_rows"])
    included = records[("btc_tradable", 10)]["metrics"]
    long = records[("magnitude_long", 10)]["metrics"]
    excluded = records[("btc_excluded", 10)]["metrics"]
    gross_included = (
        included["net_pnl"] - included["funding_net"] + included["fees"] + included["slippage_cost"]
    )
    gross_long = long["net_pnl"] - long["funding_net"] + long["fees"] + long["slippage_cost"]
    components = {
        "direction_net_delta": included["net_pnl"] - long["net_pnl"],
        "gross_price_delta": gross_included - gross_long,
        "funding_delta": included["funding_net"] - long["funding_net"],
        "fee_and_slippage_paid_delta": included["fees"]
        + included["slippage_cost"]
        - long["fees"]
        - long["slippage_cost"],
        "trend_gross_price_component": gross_included,
        "long_gross_price_component": gross_long,
    }
    assert (
        abs(
            components["gross_price_delta"]
            + components["funding_delta"]
            - components["fee_and_slippage_paid_delta"]
            - components["direction_net_delta"]
        )
        < 1e-6
    )
    members = read(inputs / "membership.json")
    assert all(r["BTC_eligible"] and "BTCUSDT" in r["new_members"] for r in members)
    save(output / "periods.json", periods)
    save(output / "exposure.json", exposure)
    save(output / "BTC_closed_attribution.json", BTC)
    save(output / "comparisons.json", summary["comparisons"])
    save(output / "cash_components.json", components)
    mapping = {
        "source_url": paper["source_url"],
        "pdf_sha256": paper["pdf_sha256"],
        "local_is_full_paper_replication": False,
        "facts_and_local_mapping": [
            {
                "original": "12month own-return sign,1monthholding;section4.1 equation5",
                "local": "EWMAC32/128,daily signal/rebalance;differentforecastspecification",
            },
            {
                "original": "return-volatility COM60,261business-dayvariance scale;section2.4",
                "local": "35dayprice-volatility/floor,365crypto-day scale;localriskadaptation",
            },
            {
                "original": "volatility-matched alwayslong comparison;section4.1",
                "local": "thisround matched-intent-magnitude long;actual riskpaths differ",
            },
            {
                "original": "liquid multiasset futures/forwards",
                "local": "USDTperpetuals,fees,funding,8percentstop/15percenthalt/.8gross;noinheritedsourceprofit",
            },
        ],
        "next_required_work": "Close the oldEMA/BTCrolebranch. Before a new12month/1month local candidate, freeze its risk/market deviations, same-opportunity longcontrol, and exact rawexcess-return/funding history; missing records remain unknown.",
    }
    save(output / "original_method_mapping.json", mapping)
    save(
        output / "verification.json",
        {
            "schema": "BTC-role-financial-review.v1",
            "passed": True,
            "checks": {
                "four_new_cash_and_stop_audits_bound": True,
                "exact_old_control_and_long_magnitude_inputs": True,
                "source_duplicates_reused_not_independent": True,
                "year_and_cash_components_reconcile": True,
                "actual_holding_different_process_restore": True,
                "cash_reference_sealed_zero": True,
                "paper_vs_local_method_and_limits_explicit": True,
                "capacity_not_triggered": True,
            },
            "source_hashes": hashes,
            "maximum_internal_NAV_error_usdt": max(errors),
            "independent_weight_error": audit["maximum_independent_weight_error"],
            "actual_new_financial_accounts": 4,
            "eligibility_variants": 1,
            "new_economic_mechanisms": 0,
            "external_review": False,
            "independent_unseen_evidence": False,
            "profit_goal_met": False,
            "orders_enabled": False,
        },
    )
    text = f"""# BTC基准角色与同绝对权重多头对照

2026-10-27。本轮只改变BTC交易资格，保持既有EMA、风险、费用与时钟；加入同绝对意图权重的持续多头作为方向移除对照。没有新增经济机制或调参。

## 结论

**BTC资格不是挽救这个EMA实现的充分条件。** 允许BTC入场后，两档成本仍亏损并停机；压力成本相对BTC排除账户只改善{included["net_pnl"] - excluded["net_pnl"]:.4f}USDT，相对同绝对权重多头反而少{abs(components["direction_net_delta"]):.4f}USDT。关闭“仅因排除BTC才失败”的具体解释，停止旧EMA资格/止损/窗口微调。不据此否定全部BTC趋势或原论文月度策略。

## 原始研究核查

取得[2012年原论文](https://fairmodel.econ.yale.edu/ec439/mosk.pdf)，本地封存PDF哈希为{paper["pdf_sha256"]}。作者站抓取超时后使用Yale学术镜像；不是以第三方策略宣传代替原文。

原文主因子是过去12个月方向、持有1个月，并以波动率匹配的持续多头作对照；波动估计使用COM60和261日尺度。当前32/128EMA、35日价格波动/地板、日频更新、16%名义预算、0.8总敞口、8%止损与15%账户停机是本地配置，不能继承原文收益，也不是它的完整复现。BTC资格是本地液态加密研究的选择，论文并不直接要求交易BTC。详细映射和页码见original_method_mapping.json。

## 冻结的唯一资格变化

旧池为每周历史合格前30个非BTC合约。新池把BTC按相同资格和过去28日报价额排序参与竞争，仍取30名，未给BTC保留仓位。88个周更新点BTC均通过资格，因此每周原边际第30名会退出。改变的是资格及相应选择/配置反馈，不能把整个账户差额当作孤立BTC仓位PnL。

趋势继续使用同一EMA32/128、冻结2024归一化、16%名义风险预算、原强度阈值、前10仓、有效180连续日、周度成员/日度00UTC信号。多头对照仅将同一日目标权重取绝对值；每一时点的选股和意图总敞口精确相同，但实际成交、净暴露、停机和持有时长可能不同，不能称实现风险已匹配，也不是被动买入持有。

原BTC排除目标逐小时精确重建。NumPy独立复算BTC资格、EMA/波动与仓位，最大权重误差{audit["maximum_independent_weight_error"]:.3g}；两个同源日档解释目标完全相同，因此8个新来源/策略/成本情景仅计算4条新账户。原排除账户复用，不能加成独立样本。

## 608日连续账户表现

2025-01-01至2026-09-01，1万USDT；每边手续费5bps加滑点3/10bps，总成本8/15bps。包括实际事件资金费、佣金、滑点、完整库存和退出费用准备；最多10仓、目标总敞口0.8、原8%收紧止损和15%回撤停机，不重置跨年高点。报价额1%仍是容量代理。

| 方案 | 每边成本bps | 整段收益 | CAGR | 完整胜率 | 最大回撤 | 日夏普 | 完整周期 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for profile, label in [
        ("btc_excluded", "BTC排除参考"),
        ("btc_tradable", "BTC可交易趋势"),
        ("magnitude_long", "同绝对权重多头"),
    ]:
        for cost in [3, 10]:
            m = records[(profile, cost)]["metrics"]
            text += f"| {label} | {cost + 5} | {m['net_return']:.4%} | {m['cagr']:.4%} | {m['win_rate']:.2%} | {m['max_drawdown']:.4%} | {m['sharpe']:.4f} | {m['closed_trades']} |\n"
    text += """
既有封存现金账户同窗口收益0、回撤0，夏普和胜率无定义；不是新增策略或新金融试验。本轮所有实际新账户均已平仓并停机，回撤延迟超限保持失败，不放宽15%门槛。两成本与重复来源不是独立发现；源码数学和另一路内部审计不是外部复现。

## 方向增量与费用

"""
    text += f"""压力成本中，BTC可交易趋势与同绝对权重多头净PnL为{included["net_pnl"]:.4f}/{long["net_pnl"]:.4f}USDT。相对方向对照的算术归因：价格组成（加回已付佣金和滑点、保留这条实际路径）改善{components["gross_price_delta"]:.4f}，资金费改善{components["funding_delta"]:.4f}，多付佣金与滑点{components["fee_and_slippage_paid_delta"]:.4f}，净差{components["direction_net_delta"]:.4f}。不是零费用账户重跑，也不能把滑点重复扣减。

趋势本身该价格组成仍为{gross_included:.4f}USDT，不能仅靠降低本路径费用就解释为已有正价格收益源。费用变化本身会改变成交、止损和停机，真正零费用反事实不能从加回费用直接推断。

本轮608日中BTC非零意图262日，实际压力趋势仅6个BTC闭合周期，净PnL{BTC["daily_archive_btc_tradable_cap10000_slip10"]["closed_net_pnl"]:.4f}USDT；这既不是262个独立试验，也不是BTC独立策略已验证。总账户资格改善额并不等于BTC分项PnL。

## 时段与实际暴露

| 方案／成本 | 2025完整年 | 2026年1—8月，承接旧状态 | 首次停机请求UTC |
| --- | ---: | ---: | --- |
"""
    for profile, label in [("btc_tradable", "趋势"), ("magnitude_long", "多头对照")]:
        for cost in [3, 10]:
            identity = f"daily_archive_{profile}_cap10000_slip{cost}"
            ys = [r for r in periods if r["id"] == identity and r["kind"] == "year"]
            text += f"| {label}／{cost + 5}bps | {ys[0]['net_return']:.4%} | {ys[1]['net_return']:.4%} | {exposure[identity]['first_halt_request']} |\n"
    text += f"""
2026的零收益来自2025停机，不能解释为该段市场没有机会。2026只有243日，不与完整年视为等长样本，固定90日块另存periods.json。压力趋势/多头整段平均总敞口为{included["mean_gross_exposure"]:.2%}/{long["mean_gross_exposure"]:.2%}，仅持仓时为{exposure["daily_archive_btc_tradable_cap10000_slip10"]["mean_gross_when_active"]:.2%}/{exposure["daily_archive_magnitude_long_cap10000_slip10"]["mean_gross_when_active"]:.2%}；后者更早停机，名义或目标权重匹配不等于实现风险匹配。

## 停止条件、未完成项与下一步

净收益、闭合PnL、两成本方向增量与回撤条件没有同时成立，按事前规则不触发容量扩测。32日块的配对日均收益差区间跨零，128日仅4块当量不足，不能据此作普遍显著性断言；配对日均差与期末现金差是不同统计量，不要求同号。全历史搜索偏差尚未完整校正。

关闭当前EMA的BTC资格解释，不继续在它上面搜索权重、止损或窗口。下一项依据原文预登记一个12月/1月命题的最小本地适配，先核对所需原始历史收益/资金费与标的历史，列出风险、日历和执行偏离；再与同机会集多头及现金比较。原文收益、这份方法核对、内部数学正确都不代表本地alpha。

4条新账户的现金、库存、精确资金费及实际止损轨迹核对通过，最大净值误差{max(errors):.3g}USDT；真实持仓/加仓后的JSON由另一PID恢复，余下完整净值、现金流、请求、库存和风险状态与不中断运行一致。外部第三方复现、真正未见数据、历史消息修订、客户实际费用/到账和盘口容量未完成。资金支付者解释也没有由本地对手方分类记录证实。总账户20%/30%年化、15%回撤、夏普1与多机制配置目标继续保持未完成。

来源包：data/research/trend_method_source_20261027_v001/，PDF本地留存，不复制其回报作本地业绩；代码与冻结计划：research_tools/trend_btc_role/。输入：results/research/trend_btc_inputs_20261027_v001/及对应independent_audit.json；完整连续账户与续跑：results/research/trend_btc_accounts_20261027_v001/。verification.json绑定全部核验和来源哈希。哈希只证明材料身份，不证明市场事实。没有下单、部署或推送。
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
            "new_accounts": 4,
            "max_nav_error": max(errors),
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ["inputs", "run", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    review(a.inputs, a.run, a.output)
