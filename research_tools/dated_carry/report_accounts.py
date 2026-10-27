"""Render a diagnostic report from immutable four-account results, without replay."""

import argparse
import math
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def run(original, finalized, output):
    output.mkdir(parents=True, exist_ok=False)
    summary, plan = read(finalized / "summary.json"), read(finalized / "plan.json")
    metrics, periods, components, witnesses, screens = [], [], [], [], []
    for row in summary["account_rows"]:
        folder = original / "accounts" / row["id"]
        m = row["metrics"]
        events, trades = read(folder / "events.json"), read(folder / "trades.json")
        metrics.append(
            {
                "id": row["id"],
                **m,
                "win_rate_wilson_95": [0.34238022750665303, 1.0],
                "sample_clusters": "two assets, one common expiry quarter",
            }
        )
        curve = pd.read_parquet(folder / "equity_5m.parquet").set_index("date")
        for start, end, label in [
            ("2025-01-01", "2025-02-01", "2025-01"),
            ("2025-02-01", "2025-03-01", "2025-02"),
            ("2025-03-01", "2025-03-29", "2025-03_partial_through_28"),
        ]:
            part = curve.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
            periods.append(
                {
                    "id": row["id"],
                    "period": label,
                    "start": start,
                    "end": end,
                    "net_pnl": float(part.equity.iloc[-1] - part.equity.iloc[0]),
                    "net_return": float(part.equity.iloc[-1] / part.equity.iloc[0] - 1),
                    "within_period_drawdown": float((1 - part.equity / part.equity.cummax()).max()),
                    "source": "continuous account; includes carried inventory, not restarted",
                }
            )
        comp = []
        for symbol in plan["contracts"]:
            es = [e for e in events if e.get("symbol") == symbol]
            spot = [e for e in es if e.get("market") == "spot"]
            fut = [e for e in es if e.get("market") == "perp"]
            spot_cash = math.fsum(e["cash_delta"] for e in spot)
            fut_cash = math.fsum(e["cash_delta"] for e in fut)
            sf, ff = math.fsum(e["fee"] for e in spot), math.fsum(e["fee"] for e in fut)
            comp.append(
                {
                    "id": row["id"],
                    "symbol": symbol,
                    "spot_cash_net": spot_cash,
                    "future_cash_net": fut_cash,
                    "spot_price_pnl_at_slipped_prices": spot_cash + sf,
                    "future_price_pnl_at_slipped_prices": fut_cash + ff,
                    "fees": sf + ff,
                    "net_pnl": spot_cash + fut_cash,
                    "reported_slippage_in_prices_not_subtracted_twice": math.fsum(
                        e["slippage_cost"] for e in es
                    ),
                    "settlement_fee_in_total_fees": math.fsum(
                        e["fee"] for e in fut if e["type"] == "settlement"
                    ),
                    "matched_trade": next(
                        t["matched_hedge"] for t in trades if t["symbol"] == symbol
                    ),
                }
            )
            for e in es:
                if e["type"] == "settlement" or (
                    e["type"] == "exit" and e.get("reason") == "post_settlement_spot_unwind"
                ):
                    witnesses.append({"id": row["id"], **e})
        assert abs(math.fsum(c["net_pnl"] for c in comp) - m["net_pnl"]) < 1e-7
        assert (
            abs(math.fsum(p["net_pnl"] for p in periods if p["id"] == row["id"]) - m["net_pnl"])
            < 1e-7
        )
        components.extend(comp)
        friction = m["fees"] + m["slippage_cost"]
        necessary = (
            m["net_pnl"] >= friction * 0.25
            and m["closed_net_pnl"] > 0
            and m["max_drawdown"] <= 0.15
            and not any(row["flags"].values())
            and all(t["matched_hedge"] for t in trades)
            and not read(folder / "positions.json")
        )
        screens.append(
            {
                "id": row["id"],
                "necessary_economic_screen": necessary,
                "paid_friction": friction,
                "registered_25_percent_friction_buffer": friction * 0.25,
                "sample_sufficient": False,
                "module_admitted": False,
            }
        )
    for name, value in [
        ("summary", summary),
        ("metrics", metrics),
        ("periods", periods),
        ("components", components),
        ("settlement_witnesses", witnesses),
        ("screens", screens),
    ]:
        save(output / (name + ".json"), value)
    save(
        output / "verification.json",
        {
            "original_four_financial_audits_passed": True,
            "four_new_financial_paths_total": 4,
            "full_financial_replays_in_repair_or_report": 0,
            "restart_checks": summary["actual_process_restarts"],
            "wallet_component_and_monthly_pnl_identity_passed": True,
            "plan_sha256": sha(original / "plan.json"),
            "source_sha256": sha(Path(__file__)),
            "original_failed_checker_completion": sha(original / "completion.json"),
            "technical_repair_completion": sha(finalized / "completion.json"),
            "all_necessary_economic_screens_passed": all(
                s["necessary_economic_screen"] for s in screens
            ),
            "sample_sufficient": False,
            "capacity_certified": False,
            "original_historical_article_versions_certified": False,
            "historical_client_fees_lots_ticks_margin_tiers_and_cash_certified": False,
            "external_review_performed": False,
            "independent_unseen_validation_completed": False,
            "module_admitted": False,
            "profit_goal_met": False,
            "orders_enabled": False,
        },
    )
    table = "\n".join(
        f"| {'持有交割' if r['id'].startswith('hold') else '提前24小时退出'} | {r['id'].split('slip')[1]} | {r['net_pnl']:.4f} | {r['net_return']:.4%} | {r['max_drawdown']:.4%} | {r['sharpe']:.3f} | {r['mean_gross_exposure']:.2%} |"
        for r in metrics
    )
    ptable = "\n".join(
        f"| {p['id']} | {p['period']} | {p['net_pnl']:.4f} | {p['net_return']:.4%} |"
        for p in periods
    )
    text = f"""# 期限基差：87日双钱包模型账户，证据仍不足

2026-10-27。只检验BTC/ETH的250328 USDT线性季度合约与一个事前退出变体；4条路径均通过内部现金复算，必要经济初筛为正，但仅一个共同季度、两个相关资产周期，**不准入收益模块或容量**。本报告为已知历史模型研究，未证明可实盘成交或稳定盈利。

## 冻结的问题与原始事实

唯一机制是现货多头、等净基础数量的到期合约空头，检验基差收敛能否在双钱包保证金与成交约束下留下净现金。最强替代解释是漏算交割/退出成本、把现货浮盈充当期货保证金、错误使用终止后报价，或由少数共同季度机会造成的偶然正收益。一个经济变体仅为提前24小时申请退出；没有搜入场时钟、阈值、钱包比例、资产或风险预算。

账户从2025-01-01 00UTC连续至2025-03-29 00UTC，1万USDT；使用此前完成5分钟记录决定一次机会，下一完成区间独立成交。事前1%过去成交额上限与实际成交区间容量分开。风险预算沿用25%总现货、70%期货钱包和5%现金，未放大；实际风险见表。

在原6个季度交易归档之上，补采10个官方归档：两币12月成交及12月至3月标记价，另87,552行CSV/Decimal逐值复核。4份原始官方CMS记录保留正文、发布时间和抓取版本。输入的24,865个每币有效5分钟时点含1行事前预热。保留终止后原记录，账户在到期后屏蔽旧合约及标记价，不补零或延用旧合约。

[上市原公告](https://www.binance.com/en-KZ/support/announcement/detail/f2338846942d48daa4f9395173727f4c)与[到期确认公告](https://www.binance.com/en/support/announcement/detail/03d4fc147cd5421681219eac7ac8eea9)支持3月28日08UTC到期，而历史deliveryTime零点仅作日历键。3月公告不可提前用于1月决策；generic季度规则和已上市身份另有记录。当前可取得正文不是经认证的原始历史版本，发布时间加5分钟只作研究可用延迟。USDT线性损益使用q×(开仓价−结算价)，未套用一般FAQ中的币本位反向公式，也未追溯应用2026修订条款。

## 扣费结果与暴露

| 政策 | 滑点bps/市场边 | 净PnL USDT | 87日收益 | 5分钟回撤 | 日夏普 | 平均总敞口 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

现货手续费10bps、期货交易及交割5bps为明确模型假设；两档滑点3/10bps不是旧方向策略的8/15bps总费用口径。交割没有市场滑点，随后卖出现货有。压力持有交割费用3.6354、内含滑点3.7156USDT；结算费0.5727已在总费用内，不重复扣除。资金费为0是到期产品定义，非缺失永续资金费填零。完整损益分拆见[components.json](components.json)，残仓和退出准备均为0。

两政策各闭合2个匹配周期、胜率100%，Wilson95%区间约34.24%—100%；没有亏损周期时PF和盈亏比未知，不能报无穷大。夏普只由87个日收益计算，一个季度的两个相关持仓不能提供87次独立的经济重复。短窗CAGR只在机器明细保留描述值，不用于365日20%总账户目标验收。没有2026样本，不造2026业绩或完整90日块。

压力持有交割利润BTC35.0288、ETH0.3913USDT，约98.90%来自BTC。ETH受事前成交上限限制，现货仅约14.43USDT；BTC约1251.14USDT。平均总敞口约25.19%，平均绝对净敞口约0.159%；70%期货现金的资本占用仍计入1万总本金。不能拿名义0.5预算当实际满配收益，也不能因闲置现金而事后放大杠杆。保守区间压力回撤约0.581%，大于收盘回撤0.151%；这不是已验证的交易所清算重建。

现金对照同87日为0。提前退出相对交割在3/10bps下分别少2.1464/2.9886USDT，实际持有期和敞口不同，不能称风险匹配或选出稳定最优政策。32/128日块比较因共同季度和块数不足，区间未知，不用重抽样伪造统计把握。

## 同一连续账户的时间段

| 账户 | 时间段 | 净PnL USDT | 分段收益 |
| --- | --- | ---: | ---: |
{ptable}

3月截至28日，月末端点为29日00UTC；各段含继承库存，PnL相加等于整段，不是按月重启或拼接。没有重新按已知月收益拟合行情子集。

## 技术核验与可推翻性

独立内部账本从实际事件重建两钱包、两腿库存、标记价风险、交割现金和费用。四条路径无到期后旧期货成交、无重复交割或永续资金费；最大组分差为机器精度量级。08UTC交割后期货数量归零，现货在08:05完成退出；价格与时间证据见[settlement_witnesses.json](settlement_witnesses.json)。

原runner完成四条金融路径和现金审计后，在重启首行两个“上一持有区间诊断”上要求子进程独自重现，出现NaN与前缀值不同。原失败源码和输出完整保留；v2只修验收拼接规则。两个真实异进程检查在有对冲库存的入场后、交割后只剩现货时进行；前缀加子进程的完整曲线、全部金钱/请求/事件/最终状态逐位一致，后续区间诊断逐位一致，未放宽数值容差或重跑四条金融收益。

先前输入v001只因“ten minutes”与“10 minutes”文本验证不匹配而失败，v002保留同一经济协议。失败不是阴性金融结果，也没有被抹掉。11项新账户/账本反例涵盖未来结算价扰动不改过去、保证金单独失败、到期恢复、重复结算拒绝、资金费拒绝、无成交量不入场以及输入不被修改。

外部能检查的材料包括原始交易所文件和规则、冻结源码/输入/预期输出及完整失败台账；目前**没有未参与开发的第三方复算、真正未见样本或经授权成交账单**。同交易所两个接口不是独立源，内部第二条审计实现不是外部审计，哈希只证明未变。历史客户费率、手数/精度、保证金档位、原历史正文版本和真实现金未认证；分数数量、5分钟收盘加滑点、成交额上限仍为模型。

## 判定与停止规则

本轮支持的窄结论是：所冻结一期合约在这套明确模型中，基差机制两档成本后净现金为正，交割边界与钱包计算可内部复算。它未证明独立alpha、稳定收益、10万容量或总账户达标。本研究停止阈值/时钟/钱包/风险/提前退出时间的搜索。下一步仅补同合同的原始历史版本和交易约束，以及事前冻结且未访问的独立季度机会；不能把已看历史重切成样本外。若无法取得关键记录，就保持模型结论和未知项，不扩大模型数量掩盖缺口。

原始路径：`{original}`；技术封账：`{finalized}`。完整指标、区间缺失原因、反证与验收见[verification.json](verification.json)、[screens.json](screens.json)和[summary.json](summary.json)。总收益目标、外部复核、未见验证继续未完成。
"""
    (output / "README.md").write_text(text)
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "module_admitted": False,
            "profit_goal_met": False,
        },
    )
    print({"report": str(output), "paths": len(metrics), "screens": screens}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--finalized", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.original, a.finalized, a.output)
