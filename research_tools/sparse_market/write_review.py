"""Portable review of completed sparse-source accounts; failed economics stay failed."""

import argparse
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def review(publication, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(publication)
    data = read(publication / "summary.json")
    if not data["audit"]["passed"] or len(data["restart_checks"]) != 16 or data["profit_goal_met"]:
        raise ValueError("Financial/restart evidence or goal scope differs")
    run = Path(read(publication / "provenance.json")["run"])
    verify_manifest(run)
    plan = read(run / "plan.json")
    for group in ["source_hashes", "account_source_hashes"]:
        for path, digest in plan[group].items():
            if sha(Path(path)) != digest:
                raise ValueError("Frozen study source changed")
    old = Path("results/research/funding_cross_section_publication_20261026_v001")
    verify_manifest(old)
    old_rows = read(old / "summary.json")["account_rows"]
    prior = next(
        r
        for r in old_rows
        if r["variant"] == "archive"
        and r["candidate"] == "funding7_raw"
        and r["phase"] == "diagnostic2026"
        and r["slippage_bps"] == 10
        and r["capital"] == 10000
    )
    new = next(
        r
        for r in data["account_rows"]
        if r["variant"] == "daily_archive"
        and r["candidate"] == "funding7_raw"
        and r["phase"] == "diagnostic2026"
        and r["slippage_bps"] == 10
    )
    compact = [
        {
            k: r.get(k)
            for k in [
                "id",
                "variant",
                "candidate",
                "phase",
                "slippage_bps",
                "capital",
                "metrics",
                "capacity",
                "risk_halted",
                "financial_audit_passed",
                "lifecycle_uncertainty",
            ]
        }
        for r in data["account_rows"]
    ]
    receipt = {
        "schema": "sparse-source-review.v1",
        "financial_accounts": data["actual_accounts"],
        "strategy_accounts": 48,
        "cash_accounts": 6,
        "audit": data["audit"],
        "restarts": data["restart_checks"],
        "source_comparisons": data["source_comparisons"],
        "all_48_strategy_accounts_negative": all(
            r["metrics"]["net_return"] < 0 for r in data["account_rows"] if r["candidate"] != "cash"
        ),
        "cohort": read(Path(plan["cohort"]) / "summary.json"),
        "module_assessments": data["module_assessments"],
        "old_fixed_pool_2026_comparison": {
            "prior_net_return": prior["metrics"]["net_return"],
            "new_net_return": new["metrics"]["net_return"],
            "not_a_single_change_causal_ablation": True,
            "prior_publication_sha256": sha(old / "completion.json"),
        },
        "source_hashes": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                publication / "completion.json",
                run / "completion.json",
                Path(plan["cohort"]) / "completion.json",
                Path(plan["bundle"]) / "completion.json",
            ]
        },
        "records": compact,
        "portfolio_actual": None,
        "profit_goal_met": False,
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
    }
    save(output / "verification.json", receipt)
    lines = [
        "| 方案 | 2025净收益 | 2026冷启动净收益 | 608日净收益 | 608日年化 | 连续净胜率 | 连续回撤 | 连续夏普 | 完整周期 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    pct = lambda v: "未知" if v is None else f"{v * 100:.2f}%"  # noqa: E731
    num = lambda v: "未知" if v is None else f"{v:.2f}"  # noqa: E731
    for candidate, name in data["candidate_names"].items():
        rows = [
            r
            for r in data["account_rows"]
            if r["candidate"] == candidate
            and r["slippage_bps"] == 10
            and r["variant"] in ["daily_archive", "common"]
        ]
        phases = {r["phase"]: r for r in rows}
        m = phases["continuous2025_2026"]["metrics"]
        lines.append(
            f"| {name} | {pct(phases['screen2025']['metrics']['net_return'])} | {pct(phases['diagnostic2026']['metrics']['net_return'])} | {pct(m['net_return'])} | {pct(m['cagr'])} | {pct(m['win_rate'])} | {pct(m['max_drawdown'])} | {num(m['sharpe'])} | {m['closed_trades']} |"
        )
    table = "\n".join(lines)
    detail = data["details"][new["id"]]["same_executed_path_attribution"]
    text = f"""# 172名时点池：资金费、MAX与动量的真实扣费验证

本轮完成48个策略账户和6个现金账户。**48个策略账户全部为负，没有新准入模块，未触发容量追加。** 这是失败的经济结果，不是因为单策略未达到全年20%而淘汰。多策略总账户尚未构建，20%基础/30%挑战、回撤15%、夏普1、容量和前向证据继续未完成。

## 同期限对照

下表为原日线情景、1万USDT、每边5bps手续费加10bps滑点。全部成交成本、实际资金费、浮亏与退出准备均进入净值。另一情景及8bps成本均保留，48案同样为负。

{table}

2025风险停机在跨年账户中原样延续，2026自然年贡献因停机而为0；它不是2026冷启动诊断的收益。观察回撤略超过15%，反映延迟退出和成交成本，未截到阈值。现金胜率和夏普保持未知，不作为收益源。

## 资金费为正，价格腿损失更大

2026冷启动的纯资金费高成本账户，同一路径分解为：价格损益 **{detail["pre_fee_funding_slippage_price_pnl"]:.2f}**，资金费 **+{detail["funding_net"]:.2f}**，手续费 **−{detail["fees"]:.2f}**，滑点 **−{detail["slippage_cost"]:.2f}**，退出准备 **{detail["open_exit_reserve"]:.2f}**，合计 **{new["metrics"]["net_pnl"]:.2f} USDT**。这说明本次亏损主要来自价格腿；不能把正资金费直接当成可实现的对冲账户收益，逐币对冲会改变资金、基差、成交与费用路径。

原固定40币池的同段归档结果为{pct(prior["metrics"]["net_return"])}，本轮为{pct(new["metrics"]["net_return"])}，正收益未复现。本轮同时改变了时点池、零量/年龄资格、身份分段及执行约束，因此尚不能把差距单独归因于幸存者偏差或某一项修正，也不能选择旧正结果继续放行。

## 数据与身份修正

最终172名是逐时点前30名的下载并集，加BTC基准；178名采集范围不等于172名候选，也不是固定同时持有172币。最终候选对应27,354,334根五分钟记录及476,658条实际报告资金费。LIT旧Litentry合约终止与新Lighter合约分开，原2026-01-12/19/26候选只有19/26/33个新身份完整正量日。PUMP旧PumpBTC与新Pump.fun也分开，原2025-10-13候选只有94个新身份完整正量日，修正后首次入选为2026-01-12。零成交日不累计有效年龄，旧库存和总账户高水位不会随新资产清零。

12个股票/商品类型只在各自官方类型条款已可见时生效，没有用今天的类型追溯删除历史。旧v002小结的类型数量仍写5，但其封存registry实际含12且计算使用全部12；最终v003按registry生成12，旧版本保留。第二来源情景仅处理122个完整日的真实五分钟/日线差异。24组对应账户的目标和净值恰好相同，不能由此认证所有原始来源一致。

## 独立证据与失败记录

54案共6,303,798个净值点独立重算，最大净值误差{data["audit"]["maximum_nav_error"]:.3g} USDT。1416个源/窗口/策略锚点和233472个小时目标行独立复核，6组真实数据未来价格/成交额/资金费扰动未改变过去四个策略书。16次真实JSON新解释器恢复含8次有仓断点、8次跨年断点，净值、账本、请求和最终高水位/停机一致。单独的1日工程对照及其数值警告保留，不当作策略盈利或独立样本。

最初54个尝试在同一混合整秒/毫秒时间解析处初始化失败，无净值、资金费现金或成交。原失败目录保留；新v2引擎仅修复解析格式，并在矩阵前预检全部真实条款，因子、名单、费用和风险参数未改。有效经济矩阵为54案，不把技术失败算成54个新的收益试验。源码和收据均按版本保存。

日历/身份仍基于当前官方正文在宣称发布加5分钟可用的研究假设，未取得完整原始历史版本或客户结算认证。成交额1%是容量代理，未通过订单簿、最小数量或真实交易所执行认证。状态归因沿用冻结BTC标签，其中流动性项来自旧40池，只作描述；22份FOMC之外的事件未知，没有据这些分组事后拼接赢家。全部已查看历史均为诊断，不能重新称为独立前向结果。

## 后续研发

优先验证逐币现货多头/永续空头能否保留正资金费并压低价格暴露；需要真实可用现货、同资产身份、独立双腿成交、基差与隔离保证金证据。已有BTC/ETH试点收益低且周期少，不能直接推广；新增选择规则和预算须在现货收益计算前冻结。同时用有限的同配置对照定位旧固定池差距。没有净收益源前，不用神经网络调权、扩大杠杆或筛掉亏损资产凑收益。

复算入口为 `research_tools/sparse_market/plan.json`、`prepare_targets.py`、`audit_targets.py`、`run_accounts_v2.py`、`verify_restarts.py` 与 `publish.py`。新路径保留原失败与所有结果，不覆盖已完成目录。未下单、部署或推送。

官方身份依据：[旧LIT终止](https://www.binance.com/en/support/announcement/detail/2a9feaa556f74dcdaa2192366f0e247c)、[新LIT上线](https://www.binance.com/en/support/announcement/detail/6a33be00231c4539b3a4a625538e4d1e)、[旧PUMP更名](https://www.binance.com/en/support/announcement/detail/38a74ee47cdc43ebbc672e1dc6968358)、[新PUMP上线](https://www.binance.com/en/support/announcement/detail/4bc8b483d10d4619babb2015066b2d89)。
"""
    (output / "README.md").write_text(text)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print({"report": str(output), "audited_accounts": 54, "profit_goal_met": False}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publication", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    review(a.publication, a.output)
