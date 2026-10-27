"""Explain conditional integer/fee inventory findings without inventing returns."""

import argparse
from decimal import Decimal
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def review(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        "metadata": Path("data/research/dated_carry_order_metadata_20261027_v001"),
        "inventory": Path("results/research/dated_carry_lot_inventory_20261027_v001"),
        "entry_witness": Path(
            "docs/research/regime_program/dated_carry_execution_witness_20261027"
        ),
        "original_accounts": Path("results/research/dated_carry_accounts_20261027_v001"),
    }
    for root in sources.values():
        verify_manifest(root)
    cases = read(sources["inventory"] / "cases.json")
    audits = read(sources["inventory"] / "integer_audit.json")
    assert all(a["passed"] and not a["factor_module_imported"] for a in audits)
    metadata = read(sources["metadata"] / "summary.json")
    fixture = read(sources["metadata"] / "binance.json_historical.json")
    fixture_info = next(f for f in metadata["fixtures"] if f["name"] == "binance.json")
    assert all(not found for found in fixture_info["exact_contract_strings_present"].values())
    spot_rules = [
        {
            "symbol": symbol,
            "fixture_source": fixture_info,
            "original_info": {
                k: fixture[symbol]["info"].get(k)
                for k in ["symbol", "baseAsset", "quoteAsset", "baseCommissionPrecision", "filters"]
            },
            "whole_quarter_rules_certified": False,
        }
        for symbol in ["BTC/USDT", "ETH/USDT"]
    ]
    quotes = read(sources["entry_witness"] / "quote_witnesses.json")
    diagnostics = []
    for case in cases:
        cover, exact = case["bounded_cover"], case["minimum_exact_pair"]
        s = case["symbol"].lower()
        spot = next(
            q["quote_before"] for q in quotes if q["symbol"] == s and q["market"] == "binance"
        )
        fut = next(q["quote_before"] for q in quotes if q["symbol"] == s + "_250328")
        fit = Decimal(exact["gross_spot"]) <= Decimal(case["inputs"]["gross_spot_cap"]) and Decimal(
            exact["short_base"]
        ) <= Decimal(case["inputs"]["short_base_cap"])
        diagnostics.append(
            {
                "symbol": case["symbol"],
                "minimum_exact_pair_fits_original_caps": fit,
                "spot_gross_notional_at_entry_ask": str(
                    Decimal(cover["gross_spot"]) * Decimal(spot["a"])
                ),
                "future_notional_at_entry_bid": str(
                    Decimal(cover["short_base"]) * Decimal(fut["b"])
                ),
                "residual_or_dust_value_at_entry_bid": str(
                    Decimal(cover["retained_exit_dust_base"]) * Decimal(spot["b"])
                ),
                "minimum_exact_pair_spot_notional_at_entry_ask": str(
                    Decimal(exact["gross_spot"]) * Decimal(spot["a"])
                ),
                "price_reference": "Indicative entry BBO only; not actual venue market-order minimum-notional reference or terminal liquidation value.",
                "actual_minimum_notional_status": "unknown",
                "actual_historical_dated_filters_certified": False,
            }
        )
    for name, value in [
        ("cases", cases),
        ("integer_audit", audits),
        ("historical_source_review", metadata["fixtures"]),
        ("archived_spot_rules", spot_rules),
        ("inventory_value_diagnostics", diagnostics),
    ]:
        save(output / (name + ".json"), value)
    summary = {
        "inventory_cases": 2,
        "independent_integer_proofs": 2,
        "minimum_exact_gross_spot_one_base_coin_under_assumptions": all(
            c["minimum_exact_pair"]["gross_spot"] == "1" for c in cases
        ),
        "exact_pairs_fit_original_caps": any(
            d["minimum_exact_pair_fits_original_caps"] for d in diagnostics
        ),
        "both_bounded_plans_have_owned_residual_and_dust": all(
            Decimal(c["bounded_cover"]["retained_exit_dust_base"]) > 0
            and not c["bounded_cover"]["matched_exactly"]
            for c in cases
        ),
        "metadata_commit": fixture_info["commit"],
        "metadata_commit_date": fixture_info["commit_date"],
        "exact_250328_historical_rules_obtained": False,
        "new_financial_accounts": 0,
        "original_quarter_returns_changed": False,
        "new_strategy_or_model": False,
        "actual_execution_or_capacity_certified": False,
        "module_admitted": False,
        "external_review_performed": False,
        "independent_oos": False,
        "profit_goal_met": False,
        "orders_enabled": False,
    }
    save(output / "summary.json", summary)
    paths = [
        Path(__file__),
        Path("factors/common/lot_inventory.py"),
        Path("research_tools/dated_carry/audit_lot_inventory.py"),
        Path("tests/research_mainline/test_lot_inventory.py"),
    ]
    paths += [root / "completion.json" for root in sources.values()]
    save(
        output / "verification.json",
        {
            "files_sha256": {str(p): sha(p) for p in paths},
            "historical_rules_unknown_not_backfilled_from_perpetual": True,
            "integer_audit_not_external_review": True,
            "two_inventory_times_not_summed_as_distinct_dust": True,
            "new_tests_passed": 12,
            "test_scope": "exact lattice, capped cover, fee/quantity conservation, zero caps, stock integer-lot arithmetic, decimal-context invariance and rejected dust writeoff; not profitability proof",
            "financial_accounts_replayed": 0,
            "profit_goal_met": False,
        },
    )
    table = "\n".join(
        f"| {c['symbol']} | {c['bounded_cover']['gross_spot']} | {c['bounded_cover']['base_fee']} | {c['bounded_cover']['net_spot']} | {c['bounded_cover']['short_base']} | {c['bounded_cover']['retained_exit_dust_base']} |"
        for c in cases
    )
    value_table = "\n".join(
        f"| {d['symbol']} | {d['spot_gross_notional_at_entry_ask']} | {d['future_notional_at_entry_bid']} | {d['residual_or_dust_value_at_entry_bid']} |"
        for d in diagnostics
    )
    (output / "README.md").write_text(f"""# 手数、基础币手续费与余币：完全相等的对冲并非免费可得

2026-10-27。本轮继续原BTC/ETH250328机制，不增加策略或收益账户。取得一个冻结历史软件快照，但**没有取得这两份季度合约的历史订单过滤器**。在明确的观察格点与原基础币收费假设下，整数算术证明：最小完全匹配组合均需买入1个基础币，超过原数量上限；保持原上限可形成更小的覆盖组合，但存在必须保留的余币。**这不是新的收益结果，也不是历史真实可交易规则认证。**

## 原始来源及不能外推的内容

以首次发现的CCXT目录路径查截至2024-12-31最后触及文件的commit，4个匿名只读HTTP请求，固定到[原始markets文件](https://github.com/ccxt/ccxt/blob/{fixture_info["commit"]}/{fixture_info["path"]})，commit为`{fixture_info["commit"]}`，时间`{fixture_info["commit_date"]}`。另一个目标文件未在目录找到，不猜路径或扩大仓库搜索。

此快照在2024-09-10已经提交，早于原[250328上市公告](https://www.binance.com/en-KZ/support/announcement/detail/f2338846942d48daa4f9395173727f4c)及9月27日上市时点，未包含BTCUSDT_250328/ETHUSDT_250328。其中BTC/ETH现货的原info.filters步长与前轮成交格点一致，保留[原字段](archived_spot_rules.json)，但软件fixture不是交易所签名响应，也不能证明直到2025年3月规则都不变；其永续或币本位字段未移植给季度合约。

前轮Tardis两个元数据请求401仍原样保留，本轮没有重试拒绝、读取密钥或购买。历史现货和季度的最小量、最大量、最小名义、市场校验参考价、收费币种/精度及规则变更仍须逐项补证；已看到的正收益不允许补造这些字段。

## 唯一算术假设与最小解

沿用原现货买入基础币手续费`f=0.001`；只将前轮原生成交的数量GCD作为**假设**：BTC现货步长0.00001、ETH0.0001，季度空头基础币步长均0.001。没有按收益挑步长、费率或BNB折扣。实际手续费若以报价币收取属于另一库存合同，不能为了避免余币在本轮悄悄切换。

令现货毛数量为`n_s*s`，净库存为`n_s*s*(1-f)`，季度空头为`n_f*t`。将`s*(1-f)/t`约为互质`a/b`，则必须满足`a*n_s=b*n_f`，第一个正整数解为`n_s=b,n_f=a`。Fraction计算与不导入该模块的整数单位复核分别实现，未用浮点容差把不等数量视为相等。

BTC的比值为999/100000，ETH为999/10000；两者最小毛现货均为1币，净现货和空头均0.999币。这分别超过原毛现货0.013344781…/0.004314438…及原空头上限；不能借此提高风险预算或杠杆。

## 原数量上限内的确定性覆盖

先在原现货毛数量和空头数量两个上限内，向下取最大期货整手数量；再向上取能覆盖该空头的最小现货手数。不重新使用闲置预算、不增加流动性、不自动重试；实际两腿独立撮合仍可能失败，这里只计算目标库存。

| 资产 | 毛现货 | 基础币手续费 | 净现货 | 空头基础数量 | 同网格卖出后的余币 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

手续费后净现货均略多于空头；余量小于一个手续费后的现货步长。按同一现货步长出售可售部分后，剩余BTC0.00000698、ETH0.0000959，不能写零、自动换成BNB、视作额外成本已经支付，或标记整个账户库存已清空。未对冲现货与退出余币是**两个时点的库存描述**，不是可以加起来重复计数的两份资产。

以此前00:05前已接收BBO作数量规模诊断：

| 资产 | 毛现货×当时ask USDT | 空头×当时bid USDT | 余币×当时现货bid USDT |
| --- | ---: | ---: | ---: |
{value_table}

这里的余币金额是资产估值，不是亏损，不能直接从旧模型利润扣掉。表中报价也不认证市场订单的最小名义检查价；若真实最小量/名义约束不允许这些目标，必须拒绝入场，不能加仓凑门槛。未来退出价格、处置费用与可成交性仍未知。

## 可复用实现与验收边界

公共实现位于统一目录`factors/common/lot_inventory.py`，用于整数库存、基础币费后净数量和余币约束，不是新增alpha列。股票可用真实整数手数和无基础币收费的库存合同复用；交易日历、印花税、报价币费用和券商规则仍由股票账户承担，没有改变独立股票任务或生产策略。

12项测试覆盖最小整数解、零库存、不能因费后不足开裸空、股票整数手数、全局Decimal精度不影响结果，以及独立审计拒绝把余币写零。两条实际库存病例另有[整数证明](integer_audit.json)，检查最小解互质、不能多开下一手、覆盖现货不能再减一手、费和库存精确守恒。这仍是我们内部第二套计算，不是外部审计。

原4条87日收益、已封存数据和前轮3bps盘口反例全部保持原字节；本轮金融账户0、模型训练0、经济变体0。没有用改写成交数量后的简单价差替代完整连续账户，收益/胜率/回撤/夏普的新值均未生成。

## 决定与后续协议

后续执行必须保留“原始订单数量→收费后实际库存→整数期货实际数量→未对冲库存→退出可售量→余币归属/估值/退出准备”的链条；不能只给两腿一个共同浮点数量。核心对冲仓位平仓不代表全部库存归零；完整周期、闭合损益与开放余币损益必须分开，关闭标准需先登记，不能把余币自动计为闭合盈利。

首先补确切历史过滤器及其时点。若无法取得，下一版只能明确登记为条件性格点压力研究；已知历史的收益不能升级为可执行、外部复算或未见验证。真实费用、退出盘口、独立季度、容量和总账户目标仍未完成。

病例原件见[cases.json](cases.json)、[来源审查](historical_source_review.json)、[估值诊断](inventory_value_diagnostics.json)及[验收](verification.json)。原始数据与代码摘要均封存；没有下单、部署或外部消息。
""")
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    review(p.parse_args().output)
