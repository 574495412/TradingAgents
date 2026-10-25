"""Publish unresolved source versions and verified terminal clauses, with no alpha claims."""

import argparse
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha


def verify(root):
    for name, h in read(root / "completion.json")["manifest"].items():
        if root.resolve() not in (root / name).resolve().parents or sha(root / name) != h:
            raise ValueError("Source receipt changed")


def publish(conflicts, lifecycle, output):
    output.mkdir(parents=True, exist_ok=False)
    verify(conflicts)
    verify(lifecycle)
    c = read(conflicts / "summary.json")
    life = read(lifecycle / "summary.json")
    notice = Path("data/research/historical_lifecycle_notices_20261026_v001")
    verify(notice)
    for name, h in read(conflicts / "source_bindings.json").items():
        if sha(Path(name)) != h:
            raise ValueError("Crosscheck raw bytes changed")
    p = read(lifecycle / "plan.json")
    if p["lifecycle_code_sha256"] != sha(
        Path("research_tools/historical_universe/lifecycle.py")
    ) or p["source_code_sha256"] != sha(
        Path("research_tools/historical_universe/audit_lifecycles.py")
    ):
        raise ValueError("Lifecycle policy code changed")
    if c["financial_accounts"] != 0 or life["financial_accounts"] != 0:
        raise ValueError("Data-only scope changed")
    cases = []
    for r in c["api_checks"]:
        cases.append(
            {
                **r,
                "maximum_ohlc_error": max(
                    r["max_errors_by_field"][k] for k in ["open", "high", "low", "close"]
                ),
                "maximum_quote_volume_error": r["max_errors_by_field"]["quote_volume"],
            }
        )
    summary = {
        "schema": "historical-execution-source-review.v1",
        "evidence_inspection_complete": True,
        "all_sources_concordant": c["all_sources_concordant"],
        "crosschecked_contract_days": c["contract_days"],
        "api_reference_days": len(cases),
        "api_disagreement_days": c["api_reference_disagreement_days"],
        "api_checks": cases,
        "notice_cases": life["notice_cases"],
        "events": life["events"],
        "removed_postterminal_slots": life["removed_postterminal_slots"],
        "changed_preview_decisions": life["changed_decisions"],
        "preview_changes": life["deltas"],
        "historical_article_versions_certified": False,
        "all_historical_status_certified": False,
        "terminal_settlement_values_verified": False,
        "automatic_source_replacement": False,
        "original_datasets_modified": False,
        "financial_accounts": 0,
        "profit_goal_met": False,
        "production_eligible": False,
        "orders_enabled": False,
        "independent_oos": False,
        "conflict_completion_sha256": sha(conflicts / "completion.json"),
        "lifecycle_completion_sha256": sha(lifecycle / "completion.json"),
        "metrics": dict.fromkeys(
            ["net_return", "win_rate", "max_drawdown", "sharpe", "capacity_usdt"]
        ),
        "next_action": "下一轮先对已发现的价格／量来源差异做固定版本敏感性核验，再补其余入选合约生命周期及执行输入；纯资金费与价格延续分开预登记，不依据收益选择数据版本。",
    }
    verification = {
        "schema": "historical-execution-review-receipt.v1",
        "evidence_inspection_complete": True,
        "all_sources_concordant": c["all_sources_concordant"],
        "financial_accounts": 0,
        "conflict_completion_sha256": sha(conflicts / "completion.json"),
        "lifecycle_completion_sha256": sha(lifecycle / "completion.json"),
        "notice_completion_sha256": sha(notice / "completion.json"),
        "publish_source_sha256": sha(Path(__file__)),
        "conflict_summary": c,
        "lifecycle_summary": life,
        "lineage": {
            "failed_TLS_partial": "data/research/historical_source_crosscheck_20261026_v001",
            "failed_budget_partial": "data/research/historical_source_crosscheck_20261026_v002",
            "strict_API_concordance_failed": "data/research/historical_source_crosscheck_20261026_v003",
            "numpy_bool_serialization_failed": "results/research/historical_source_conflicts_20261026_v001",
            "descriptive_conflict_release": str(conflicts),
            "lifecycle_release": str(lifecycle),
        },
    }
    save(output / "summary.json", summary)
    save(output / "verification.json", verification)
    table = [
        "| 合约 / 日期 | 不同的五分钟bar | OHLC最大绝对差 | 成交额最大绝对差USDT | 一致 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in cases:
        table.append(
            f"| {r['symbol']} / {r['date']} | {r['different5m_bars']} | {r['maximum_ohlc_error']:.8g} | {r['maximum_quote_volume_error']:.8g} | {'是' if r['all_numeric_fields_agree'] else '否'} |"
        )
    terms = [
        "| 合约 | 停止新仓UTC | 终止UTC | 终止后日线开盘标签 | 当月终止后资金费计算记录 |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for r in life["events"]:
        terms.append(
            f"| {r['symbol']} | {r['entry_stop_at']} | {r['settlement_at']} | {r['source_daily_open_labels_at_or_after_settlement']} | {r['funding_archive_records_after_settlement']} |"
        )
    text = f"""# 官方来源存在版本冲突，合约终止后归档也不能当作可交易记录

已交叉核验{c["contract_days"]}个合约日，并封存4个官方API参照日；其中{c["api_reference_disagreement_days"]}个参照日与官方五分钟日归档有差异。没有自动挑选“正确”来源，没有改写原40币或860合约日线，也没有新增收益账户。两个五分钟来源之间已出现OHLC差异，因此不能将此前所有日线成交额冲突概括成“价格完全一致”；前一轮“收盘价一致”只针对每日最后收盘，并不证明逐五分钟或盘中价格相同。

## 源版本差异保持未通过

{chr(10).join(table)}

143天中129天有旧40币对照，其中128天的旧五分钟OHLC和成交额与新日归档一致；BTC 2023-11-10属于另外一个版本差异。BTC 2023-08-16的API与日归档12个数值字段逐行一致；其余3个API参照日有实际差异。所有4个参照日的日末收盘相同，仍不排除盘中止损、成交量容量和路径损益受影响。API、日归档、旧月度归档是不同观察版本，校验和只证明文件身份，不裁决哪个经济值正确。

版本敏感性必须另做有限预登记，不以利润最高者取代其他版本。旧失败账户与来源收据保留；没有因为差异存在而把既有亏损变成盈利，也没有声称所有历史已完成校正。143天是按已报告冲突与必要字段异常的并集选择，不代表143个独立收益样本或全市场随机抽检。

## 用官方合约公告约束可交易区间

{chr(10).join(terms)}

本表只涉及3个USDT永续合约；停止新仓、自动结算、现货换币是不同事件，不混用时点。已归档当前CMS正文、官方毫秒发布时间、来源SHA、对应月资金费原ZIP及CHECKSUM。3份公告分别见[FTM](https://www.binance.com/en/support/announcement/detail/aec6fcbc84b749eeab6690e6bcac2f3d)、[MATIC](https://www.binance.com/en/support/announcement/detail/6a6de383727f4659a3050f7982e1620f)、[RNDR](https://www.binance.com/en/support/announcement/detail/d1f2ae8d99b24439a7a900caa9bb6b3b)。引用的Github问题仅用于发现线索，数值与时点均由本地官方原件独立核验，不把评论作者当交易所声明。

终止后记录的存在不证明实际成交、当时可开仓或仍应支付资金费。FTM终止后的602个日线开盘标签来自截至2026-08-31的完整归档区间；资金费76条只统计2025-01这一月，两个计数的区间不同。MATIC与RNDR同理，不能把这些计算时间戳直接记成持仓现金收入。实际最终结算价与结算费用均保持未知，没有以最后档案收盘或零代替。

## 资格预览确实发生了修正，仍不是收益账户

已实现并实际用于366次预览的边界校验：公告宣称可用时点为官方发布时间加5分钟研究延迟；该假设不证明原始历史正文版本已取得。新仓严格区分停止新仓和终止时点，跨越终止的整日bar不用于完整日特征，未来公告不提前禁用旧时点。现有3案影响1个预览：2024-09-09的MATICUSDT被剔除，按相同成交额规则补入NEARUSDT；没有重算盈利或事后选择交易方向。

剩余合约历史状态仍未全面认证。当前元数据还发现12个归档名称有早于其当前onboardDate的记录，128个名称有晚于其当前deliveryDate的记录；这只是“当前合约区间未覆盖历史记录”的复核线索，不把所有早期记录判成假数据，也不排除曾重上架或不同合约段。当前声明不能代替各历史合约段的公告与真实执行证据。

`require_terminal_value`提供未核实结算值的拒绝校验器，非空持仓到达终止时不得用末价／零伪造现金；该检查器尚未接入扩展币池完整Freqtrade账户。实际已执行的是数据资格预览修正，不能用六项边界测试代替完整账户或交易所运行证据。

## 失败输出与复算

第一次日归档补采遇到TLS中断，第二次请求预算重复计入尚未执行的旧任务，第三次严格一致性检查因真实API差异而失败；全部目录保留。最终描述性审阅只确认差异、未将一致性改判通过。描述性收据v001的numpy布尔序列化失败也保留，v002仅修正JSON类型，未改变对比规则。已下载且通过官方SHA的文件跨版本复用，其来源路径与摘要都有绑定；恢复预算在追加请求前单独登记，不改变收益门槛。

完整原件与代码身份见[审阅收据](verification.json)。从TradingAgents目录复算，输出用新目录：

```sh
python -m research_tools.historical_universe.describe_crosscheck --raw data/research/historical_source_crosscheck_20261026_v003 --api data/research/historical_source_crosscheck_20261026_v001 --output results/research/FRESH_conflicts
python -m research_tools.historical_universe.audit_lifecycles --output results/research/FRESH_lifecycle
```

{summary["next_action"]} 多策略总账户年化20%／30%、最大回撤15%、夏普1及容量目标继续未完成；未下单、部署或推送。
"""
    (output / "README.md").write_text(text)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "evidence_inspection_complete": True,
            "all_sources_concordant": c["all_sources_concordant"],
            "financial_accounts": 0,
        },
    )
    print(
        {
            "published": True,
            "source_agreement_passed": c["all_sources_concordant"],
            "notice_cases": life["notice_cases"],
            "new_financial_accounts": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conflicts", type=Path, required=True)
    p.add_argument("--lifecycle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    publish(a.conflicts, a.lifecycle, a.output)
