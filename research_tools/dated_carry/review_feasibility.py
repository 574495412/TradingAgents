"""Seal the method/unit/raw-source result, keeping all financial metrics unknown."""

import argparse
import shutil
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def review(audit, output):
    output.mkdir(parents=True, exist_ok=False)
    roots = {
        "source": Path("data/research/dated_carry_sources_20261027_v001"),
        "followup": Path("data/research/dated_carry_source_followup_20261027_v001"),
        "data": Path("data/research/dated_carry_pilot_20261027_v001"),
        "route": Path("data/research/dated_carry_settlement_route_20261027_v001"),
        "audit": audit,
    }
    for root in roots.values():
        verify_manifest(root)
    summary = read(audit / "summary.json")
    proof = read(audit / "verification.json")
    for filename, digest in proof["source_hashes"].items():
        assert sha(Path(filename)) == digest, filename
    assert summary["passed"] and summary["independent_CSV_Decimal_rows"] == 49730
    assert (
        not summary["new_financial_accounts"]
        and summary["annualized_basis_estimated_from_filename"] is False
    )
    paper = roots["followup"] / "paper.raw"
    assert paper.read_bytes().startswith(b"%PDF-")
    source_mapping = {
        "paper": {
            "title": "Crypto Carry",
            "authors": ["Maik Schmeling", "Andreas Schrimpf", "Karamfil Todorov"],
            "version": "March24,2023; BISWorkingPaper1087 April2023",
            "url": "https://www.bis.org/publications/working-paper-1087-crypto-carry.pdf",
            "raw_file": str(paper),
            "sha256": sha(paper),
        },
        "mapping": [
            {
                "paper_location": "printedpp1-2,6-7;PDFpp4-5,9-10",
                "source_method": "BTC/ETH,March2019-Jan2022;Skew constant1/3month annualized bases across venues, with borrowing/lending and CFTC positioning context.",
                "local_difference": "One actual USDT-linear March2025 contract per asset, raw5m exchange archives and real spot; no interpolation, borrowed financing, option/CFTC reconstruction or source-return inheritance.",
            },
            {
                "paper_location": "printedp5;PDFp8;visually inspected",
                "source_method": "Hedged price exposure does not remove funding-until-convergence/margin risk; spot gains may be unavailable to the futures wallet.",
                "local_difference": "No financial account is yet generated. Require separate wallet/free-cash/mark-price stress and both-leg fills; current metadata does not certify historical tiers.",
            },
        ],
        "mechanism": "Leveraged directional demand and constrained arbitrage capital are a source-supported explanation, not observed counterparty behavior or a certified local cash premium.",
        "relation_to_prior_work": "Different payoff from trend and floating funding extrapolation; same broad structural-carry/capital-constraint family as prior spot-perp research. No new independent return source proven.",
        "official_api_source": read(roots["route"] / "official_route.json"),
        "failed_sources_preserved": "Initial BISPDF URL redirected to an HTML publication page; follow its observed PDF link. Three developer docs returned202empty, and guessed delivery route404. Official Postman source supplied /futures/data/delivery-price. All original responses retained.",
        "direct_public_HTTP_requests": 34,
        "separate_tool_failures": "Search/connector transport429 attempts yielded no usable data, not included as successful exchange evidence.",
        "raw_archives_downloaded": 6,
        "current_metadata_only": True,
        "external_source_independence": "BIS is independent literature, not independent verification of local market prices. Exchange archive, API and its official SDK/Postman remain same-provider evidence.",
    }
    for filename in ["summary.json", "settlement_boundaries.json"]:
        shutil.copyfile(audit / filename, output / filename)
    save(output / "source_method_mapping.json", source_mapping)
    metrics = {
        "net_return": None,
        "cagr": None,
        "win_rate": None,
        "max_drawdown": None,
        "sharpe": None,
        "capacity": None,
        "reason": "No continuous dated-carry account yet; a reference basis is not a realizable payoff or annualized account return.",
    }
    save(output / "financial_metrics.json", metrics)
    paths = [
        Path(__file__),
        Path("factors/common/dated_carry.py"),
        Path("tests/research_mainline/test_dated_carry_units.py"),
        *[p / "completion.json" for p in roots.values()],
    ]
    verification = {
        "schema": "dated-carry-feasibility-review.v1",
        "passed": True,
        "proof_scope": "Source/method mapping, exact independent CSV conversion and explicit unit/settlement counterexamples only.",
        "source_hashes": {str(p): sha(p) for p in paths},
        "new_financial_accounts": 0,
        "new_market_archives": 6,
        "raw_rows_independently_reconstructed": 49730,
        "current_USDT_linear_metadata_confirmed": True,
        "historical_precise_cutoff_confirmed": False,
        "historical_margin_tiers_confirmed": False,
        "delivery_fee_confirmed": False,
        "client_cash_confirmed": False,
        "annualized_reference_basis_inferred": False,
        "financial_metrics_stay_unknown": True,
        "independent_source_audit": False,
        "third_party_review": False,
        "independent_oos": False,
        "profit_goal_met": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "verification.json", verification)
    rows = summary["rows"]
    boundaries = read(audit / "settlement_boundaries.json")["historical_records"]
    table = "\n".join(
        f"| {r['contract']} | {r['source_rows']} | {r['daily_observations']} | {r['nonpositive_volume_rows']} | {r['reference_basis_bps_quantiles']['0.5']:.4f} |"
        for r in rows
    )
    terminal = "\n".join(
        f"| {r['contract']} | {r['recorded_date_label']} | {r['observed_last_open']} | {r['settlement_record']['deliveryPrice']} | {r['last_reported_close']} |"
        for r in boundaries
    )
    text = f"""# 期限基差：先核对计价与结算，收益仍未知

2026-10-27。上一轮固定趋势门控已封存，3个活跃月不足准入；本轮没有继续选状态、改阈值或扩大模型。这里仅评估下一候选的原始方法与数据，不把资料准备当盈利进展。

**结论：USDT线性季度合约具备原始研究数据基础，但未完成历史终止时钟、收费、保证金与双腿账户验证。收益、CAGR、完整周期胜率、回撤、夏普和容量全部为空；新增金融账户0。**

## 唯一机制与原方法映射

在现货多头与有到期日的线性期货空头之间研究期限价差收敛。支付解释是杠杆方向需求与有限套利资本；当前没有投资者逐笔身份数据，不能声称已经识别了本地交易对手。它与趋势择时不同，但与已有现货/永续资金费研究同属结构carry，不能先算成第二个独立收益源。

原始来源为[BIS Working Paper1087](https://www.bis.org/publications/working-paper-1087-crypto-carry.pdf)。原论文研究2019—2022的跨场所、恒定1/3月期限资料，并强调持有至收敛前的保证金风险。本地只采一组实际到期合约的2025年第一季度原始记录；这不是原文完整复现，不继承原文回报、Sharpe或因果解释。页码、文件指纹与具体差异见[方法映射](source_method_mapping.json)。

## 必要原始记录

事前固定BTC、ETH和现有研究日历中的首个季度，未按收益挑资产或季度。6个官方5分钟ZIP与CHECKSUM封存；从CSV/Decimal独立重建49,730行，九个浮点字段与原解析表逐值一致。现货引用已封存的真实现货来源，不拿永续价格代替。未重新下载现货，也未补齐终止后的日期。

| 合约 | 原始5分钟行 | 可采样零点观测 | 零成交/成交额行 | 同时点参考基差中位数bps |
| --- | ---: | ---: | ---: | ---: |
{table}

基差为 `(F/S-1)×10000`，两者均为USDT/同一基础资产单位。它来自成交K线收盘参考，不是同步可成交买卖报价、净收益、胜率或扣费年化。全量非年化参考与原始归档链保存在本地封存输入中；未确认历史期限时，年化字段保持空。

观察时间明确为K线开盘后5分钟；第一条参考于2025-01-01T00:05UTC可用，没有补造此前观测。每个合约原始行连续至3月28日08:00开盘，最后一条于08:05才完成；该终止日只有97根记录，不补成完整交易日。零量记录保留且标记不能支持成交。

## 两个不能被掩盖的结算问题

| 合约 | 接口返回日期标签 | 最后报告开盘UTC | 公共结算参考价 | 最后报告收盘价 |
| --- | --- | --- | ---: | ---: |
{terminal}

历史`deliveryTime`记录落在零点，而当前`deliveryDate`元数据落在08UTC。两类字段不能未经证明就视为同一历史交易截止时刻。日期标签之后仍有报告成交，只能证明数据源报告了记录，不能证明某账户当时可以成交；归档最后收盘也不能代替结算参考价或客户实际入账。

最初错误构造的`/fapi/v1/deliveryPrice`返回404，原响应保留。依据官方Postman源码定位`/futures/data/delivery-price`后另取原始结算记录，未修改或重下载市场归档；官方集合blob为`80edc431f323c30397ab086b9ccb5cb51c6e5a3f`。当前交易规则快照仅证明现有USDT报价/保证金与季度类型，不能回填2025年历史状态、费率和保证金档位。

## 已落地的统一数学模块与反例

统一模块为 `factors/common/dated_carry.py`：原始价差/对数基差与可选的已证实期限换算；另有实际净基础资产数量下的两腿损益恒等式。USD和USDT不同币种、不同基础资产及反向计价合约会被拒绝。实际数量不匹配时保留残余方向风险，不假定现货与空头自动对冲。

11个反例检查覆盖共同价格冲击、币本位现货手续费造成的数量损失、退出不收敛、临到期高年化仍被费用吞噬、未知期限、单位冲突和零价差成本。它们是数学/实现检查，金融试验数仍为0，不构成盈利或外部审计。

## 下一项必须改变的判断

先补充能确定历史交易截止与收费的原始条款，核查连续标记价及保证金所需现金；再冻结明确提前退出或实际交割政策、最小有用净效应、双腿钱包与成交规则、基础/压力成本和现金/静态对照，才允许计算账户。若采用提前退出，它是不同持有合同，不能据此把未知的终端结算改成已知。

一个季度只有两个共同市场机会，不能支撑独立收益、稳定胜率或容量。原始源的SHA256证明本次文件未变，同一交易所的API/归档不是两个独立数据持有方。真正未见验证、未参与开发者的复算、客户现金与总账户20%/15%/夏普1目标继续未完成。本轮不下单、不部署、不推送。

[核验范围](verification.json)、[全部数据诊断](summary.json)、[终止边界](settlement_boundaries.json)、[保持为空的金融指标](financial_metrics.json)。
"""
    (output / "README.md").write_text(text)
    manifest = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    save(
        output / "completion.json", {"passed": True, "manifest": manifest, "profit_goal_met": False}
    )
    print(
        {
            "report": str(output),
            "new_financial_accounts": 0,
            "rows": 49730,
            "profit_goal_met": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    review(a.audit, a.output)
