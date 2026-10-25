"""Publish bounded data evidence without inventing strategy returns or certification."""

import argparse
from collections import defaultdict
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha


def publish(daily, audit_path, docs):
    if (docs / "completion.json").exists():
        raise FileExistsError("Preserve completed review")
    source = read(daily / "summary.json")
    audit = read(audit_path)
    if not audit["membership_arithmetic_passed"] or audit["daily_completion_sha256"] != sha(
        daily / "completion.json"
    ):
        raise ValueError("Independent membership audit missing or changed")
    for name, digest in read(daily / "completion.json")["manifest"].items():
        if daily.resolve() not in (daily / name).resolve().parents or sha(daily / name) != digest:
            raise ValueError("Daily source changed")
    plan = read(Path("research_tools/historical_universe/plan.json"))
    catalog = Path(plan["manifest_directory"])
    symbols = read(catalog / "symbols.json")
    preview = read(daily / "eligibility_preview.json")
    years = defaultdict(list)
    for r in preview:
        years[r["decision"][:4]].append(r)
    annual = []
    for year, rows in sorted(years.items()):
        selected = sum(len(r["selected"]) for r in rows)
        outside = sum(len(r["selected_outside_old40"]) for r in rows)
        annual.append(
            {
                "year": year,
                "decisions": len(rows),
                "selection_slots": selected,
                "unique_selected": len({s for r in rows for s in r["selected"]}),
                "old40_coverage_of_selected_slots": (selected - outside) / selected
                if selected
                else None,
                "selected_missing_current_metadata": len(
                    {s for r in rows for s in r["selected_missing_current_metadata"]}
                ),
                "selected_current_noncoin": len(
                    {s for r in rows for s in r["selected_current_noncoin"]}
                ),
            }
        )
    labels = {
        "invalid_OHLC": "OHLC内部冲突",
        "quote_outside_OHLC_volume_bounds": "成交额超出高低价与总量可解释范围",
        "auxiliary_trade_or_taker_inconsistency": "主动成交或交易笔数字段冲突",
        "partial_daily_interval": "非完整UTC日区间",
        "missing_interior_source_day": "首末观测之间缺失日期",
    }
    quality = [{"id": k, "label": labels[k], "count": v} for k, v in source["source_flags"].items()]
    differing = sum(r["different_days"] for r in audit["independent_interval_comparison"])
    quality.append(
        {
            "id": "daily_vs5m_different_days",
            "label": "与既有五分钟聚合不一致的合约日",
            "count": differing,
        }
    )
    description = "从官方归档枚举历史合约，保留不再出现在当前资料中的名称；候选资格仅看之前已完成的28日成交额与180个有效观测，不读取未来收益。当前资产类别只作标签，未用于历史筛选。"
    quality_note = "官方SHA256校验通过仍可能存在源字段冲突。原值、缺日与非完整日均保留；价格／流动性字段不合格的日期不用于候选资格，辅助字段单独标记，未补价格或截尾收益。异常计数可在同一行重叠，不应相加当作独立样本。"
    next_action = "补齐历史合约类型、实际可交易状态、停牌／退市最终结算、真实资金费及五分钟成交输入；解决日线与独立区间源差异后，再预登记纯资金费与价格延续对照。数据覆盖和选池算术通过不等于收益通过。"
    summary = {
        "schema": "historical-universe-data-review.v1",
        "stage": "historical_data_qualification",
        "description": description,
        "start": plan["start"],
        "end_exclusive": plan["end_exclusive"],
        "symbols": source["symbols"],
        "archives": source["raw_archives"],
        "daily_rows": source["rows"],
        "preview_decisions": source["preview_decisions"],
        "selected_union": len(source["preview_selected_union"]),
        "selected_outside_old40": len(source["preview_selected_outside_old40"]),
        "selected_missing_current_metadata": len(
            source["preview_selected_missing_current_metadata"]
        ),
        "archive_symbols_missing_current_metadata": sum(
            r["archives"] > 0 and r["absent_from_current_metadata"] for r in symbols
        ),
        "quality_issues": quality,
        "quality_note": quality_note,
        "next_action": next_action,
        "annual_preview": annual,
        "financial_accounts": 0,
        "financial_account_ready": False,
        "historical_trading_status_verified": False,
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
        "profit_goal_met": False,
        "data_completion_sha256": sha(daily / "completion.json"),
        "metrics": dict.fromkeys(
            ["net_return", "win_rate", "max_drawdown", "sharpe", "capacity_usdt"]
        ),
    }
    verification = {
        **audit,
        "data_completion_sha256": sha(daily / "completion.json"),
        "independent_audit_sha256": sha(audit_path),
        "source_quality": source,
        "annual_preview": annual,
        "publish_code_sha256": sha(Path(__file__)),
        "catalog_completion_sha256": sha(catalog / "completion.json"),
        "source_review_sha256": sha(docs / "source_review.json"),
        "max_component_attribution_sha256": sha(docs / "max_component_attribution.json"),
    }
    save(docs / "summary.json", summary)
    save(docs / "verification.json", verification)
    lines = [
        "| 年份 | 周频预览 | 入选合约并集 | 旧40覆盖候选槽位 | 入选但当前资料缺名 | 当前标非COIN入选合约 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in annual:
        share = (
            "未知"
            if r["old40_coverage_of_selected_slots"] is None
            else f"{100 * r['old40_coverage_of_selected_slots']:.2f}%"
        )
        lines.append(
            f"| {r['year']} | {r['decisions']} | {r['unique_selected']} | {share} | {r['selected_missing_current_metadata']} | {r['selected_current_noncoin']} |"
        )
    issue_table = ["| 源检查项 | 数量 |", "| --- | ---: |"] + [
        f"| {r['label']} | {r['count']} |" for r in quality
    ]
    text = f"""# 历史币池已扩展，交易资格和盈利仍未验证

已采集并独立读取{source["raw_archives"]:,}份官方月度日线ZIP及对应SHA256，覆盖{source["symbols"]}个历史候选合约、{source["rows"]:,}条日线。{len(source["preview_selected_union"])}个合约曾进入固定周频资格预览，其中{len(source["preview_selected_outside_old40"])}个在原40币池以外。没有新增收益账户，也没有更新多策略总账户20%／30%目标的实际值。

原40币池按事后可见币种固定；新样本先枚举全部归档名称，再按过去成交额与观察年龄判断。两者覆盖范围不同，旧账户结果保留，不把本次采集冒充旧结果已经消除存续偏差。

## 覆盖和候选预览

{description}

{chr(10).join(lines)}

“候选槽位”为每次预览入选名称数之和，同一合约可重复，不能当独立样本。当前未列出的名称不是已经认证的退市事件；当前非COIN标签也不被追溯当作过去已知分类。此处是全部USDT归档候选的观察预览，尚非完全认证的加密永续可交易池。无完整28天有效历史或不足180个有效观察时保持未知；不会用以后仍有数据这一条件筛选。

## 源异常与独立核验

{chr(10).join(issue_table)}

{quality_note}

原始严格摄取留下{source["raw_ingestion_errors_preserved"]}个显式失败月，不覆盖该目录；独立CSV解析补取这些已列示档案共{source["supplement_http_requests"]}次请求，仍要求官方SHA256和原始ZIP大小一致。日线成交额要处于最低／最高价格×基础成交量之间，容差为max(0.01USDT,1e-6相对误差)，源字段互相冲突时不改成“合理值”。例如ARBUSDT 2023-09-19官方日线出现主动买入基础量大于总量，以及成交额越出价格与量的可解释范围；该币不在原40币池中，不能据此声称旧40账户数值被改写。

独立NumPy算术复算全部{source["preview_decisions"]}次候选预览，并在2021／2023／2025三个截止点放大真实未来成交额检查过去窗口；价格／量异常与缺日保留，不截尾、不补零。原40币五分钟完整日聚合另作独立区间对照，发现{differing}个合约日存在差异；差异和日期保存在[验证收据](verification.json)，有差异就不宣称全量一致。即使源校验和一致，也不能据此推断经济数据正确。

## MAX差异没有转成事后开关

已独立核对原24账户的方向、币种及入场月份分解，全部开放周期含退出准备并回总净值；见[分项收据](max_component_attribution.json)。2025高成本MAX多头闭合−627.07USDT，空头−444.21USDT，均为负。2026多头闭合只有+27.38USDT，其开放周期含退出准备+981.94USDT；多头总资金费+530.17USDT。空头闭合+359.23USDT。不能从分项PnL推导一段未执行过的只做多／只做空策略，也不能把资金费收入等同于价格预测能力。

## 新公开来源只作固定版本参考

[OctopusTakopi/funding-rate-alpha](https://github.com/OctopusTakopi/funding-rate-alpha/tree/cf9f87c01d3165bbc3bf34438baee289d248ffca)为MIT个人研究。其原始大体积资料未随仓库发布；已检查代码里未来收益非空的成员筛选、默认收益截尾和持仓缺失零贡献处理。动态历史档案和分开检查资金费／价格损益值得参考，源码收益不能作为我们的完整账户证据。

[bryanvine/alpha-research](https://github.com/bryanvine/alpha-research/tree/17b8b5e1c79af194f733544517d82e3cf12c2259)提供纯资金费横截面假设，但采用现货日收益代理永续价格腿；默认5个观察日再平衡与固定周一不同，分位阈值在全部信号相同时可能令多空集合重叠。未核实可复制代码许可，因此仅作来源审阅，不把代码引入统一因子库。两份较新的个人研究都不等同于成熟框架或已验证高收益来源。

完整固定commit、文件SHA和定位见[来源审阅](source_review.json)。所有外部源码仅静态读取，没有执行、安装其项目或访问其自述中的私人数据卷。官方归档来源仍是Binance，原始CSV／ZIP／CHECKSUM、清单分页和当前元数据分开存证。

## 未完成项和复算

{next_action}

历史目录由原始公开归档在采集时枚举，不证明交易所从未删改过档案；历史类型、交易状态、最终结算和盘口缺口保留。网络／代码／独立算术检查通过均不是盈利证明，原MAX失败和原现金对照保留，R06／R07／R08继续进行中。无下单、部署或推送。

从TradingAgents运行，所有输出用新目录：

```sh
python -m research_tools.historical_universe.catalog --output data/research/FRESH_catalog
python -m research_tools.historical_universe.ingest --catalog data/research/FRESH_catalog --output data/research/FRESH_raw
python -m research_tools.historical_universe.seal_daily --catalog data/research/FRESH_catalog --collection data/research/FRESH_raw --output data/research/FRESH_daily
python -m research_tools.historical_universe.audit_preview --daily data/research/FRESH_daily --output results/research/FRESH_daily_audit.json
```

严格摄取会在终端记录全部显式失败后非零退出；只有完整下载日志存在，独立解析才允许补取有限失败源，不将观察超时当作终止重启。冻结计划为`research_tools/historical_universe/plan.json`；本次原始归档与独立版分别在`data/research/historical_perpetual_daily_20261026_v001`和`data/research/historical_perpetual_daily_20261026_v002`。五分钟对照只核对完整原始日，差异保持未知，不替换历史来源。
"""
    (docs / "README.md").write_text(text)
    save(
        docs / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in docs.iterdir() if p.is_file()},
            "data_only": True,
            "financial_accounts": 0,
        },
    )
    print(
        {"review_published": True, "symbols": source["symbols"], "new_financial_accounts": 0},
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daily", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--docs", type=Path, required=True)
    a = p.parse_args()
    publish(a.daily, a.audit, a.docs)
