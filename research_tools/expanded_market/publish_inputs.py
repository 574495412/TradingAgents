"""Publish compact acquisition evidence while leaving all economic targets open."""

import argparse
import shutil
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest

COHORT = Path("results/research/expanded_market_cohort_20261026_v001")
MARKET = Path("data/research/expanded_5m_market_20261026_v001")
MARKET_AUDIT = Path("results/research/expanded_market_source_audit_20261026_v001")
FUNDING = Path("results/research/expanded_exact_funding_inputs_20261026_v001")
TYPES = Path("data/research/expanded_official_type_references_20261026_v001")


def publish(output):
    output.mkdir(parents=True, exist_ok=False)
    for root in [COHORT, MARKET, MARKET_AUDIT, FUNDING]:
        verify_manifest(root)
    ca = COHORT.with_name(COHORT.name + "_independent_audit.json")
    fa = FUNDING.with_name(FUNDING.name + "_independent_audit.json")
    c, m, f = (
        read(COHORT / "summary.json"),
        read(MARKET_AUDIT / "summary.json"),
        read(FUNDING / "summary.json"),
    )
    if (
        not read(ca)["passed"]
        or not read(fa)["passed"]
        or not m["source_integrity_passed"]
        or read(ca)["cohort_completion_sha256"] != sha(COHORT / "completion.json")
        or read(fa)["inputs_completion_sha256"] != sha(FUNDING / "completion.json")
    ):
        raise ValueError("Source or independent arithmetic evidence differs")
    sources = {
        "cohort_audit.json": ca,
        "market_audit.json": MARKET_AUDIT / "summary.json",
        "funding_audit.json": fa,
        "funding_unknown_marks.json": FUNDING / "unqualified_marks.json",
        "gaps.json": MARKET_AUDIT / "gaps.json",
    }
    for name, source in sources.items():
        shutil.copyfile(source, output / name)
    next_action = "先按事前可用的官方产品类型修订加密资格并冻结来源差异情景；新增稀疏日历/合约边界引擎，真实无量区间不成交、非空库存缺标记或未知终端现金即阻断，风险和费用跨年连续。随后只比较已登记资金费、成本门槛、MAX、动量与现金，完成财务独立复算、真实进程恢复和容量后再判断模块及总账户。"
    summary = {
        "schema": "expanded-market-evidence.v1",
        "title": "扩展历史币池 · 执行行情与实际资金费",
        "decisions": c["decisions"],
        "symbols": c["symbols"],
        "memberships": c["memberships"],
        "market_rows": m["rows"],
        "funding_events": f["reported_events"],
        "new_archive_symbols": 149,
        "reused_market_symbols": 26,
        "archive_http_requests": 5452,
        "funding_api_requests": f["api_calls"],
        "interior_missing_bars": m["interior_missing_bars"],
        "unusable_execution_bars": m["unusable_execution_bars"],
        "compared_complete_days": m["complete_days_compared"],
        "daily_conflicting_days": m["daily_conflicting_days"],
        "nonpositive_or_unknown_funding_marks": f["nonpositive_or_unknown_marks"],
        "unknown_marks_before_first_candidate_only": read(fa)[
            "unqualified_marks_before_first_candidate_only"
        ],
        "classifications_pending": c["classifications_pending"],
        "cohort_audit_passed": True,
        "market_integrity_passed": True,
        "funding_arithmetic_passed": True,
        "daily_and_five_minute_sources_concordant": m["daily_and_five_minute_sources_concordant"],
        "missing_prices_imputed": False,
        "historical_tradability_certified": False,
        "payable_cashflows_certified": False,
        "dense_engine_compatible": False,
        "type_capture": {
            "complete": False,
            "official_cms_details": 3,
            "named_symbols": 4,
            "remaining_blog_response": read(TYPES / "xau_type_blog.receipt.json"),
        },
        "new_financial_accounts": 0,
        "portfolio_actual": None,
        "metrics": dict.fromkeys(
            ["net_return", "win_rate", "max_drawdown", "sharpe", "capacity_usdt"]
        ),
        "profit_goal_met": False,
        "production_eligible": False,
        "orders_enabled": False,
        "next_action": next_action,
    }
    save(output / "summary.json", summary)
    paths = [
        Path(__file__),
        COHORT / "completion.json",
        MARKET / "completion.json",
        MARKET_AUDIT / "completion.json",
        FUNDING / "completion.json",
        ca,
        fa,
        *TYPES.glob("*"),
    ]
    paths += list(Path(__file__).parent.glob("*.py")) + [
        Path(__file__).with_name("funding_recovery.json")
    ]
    save(
        output / "verification.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths if p.is_file()},
            "portable_copies": {name: sha(source) for name, source in sources.items()},
            "new_financial_accounts": 0,
            "historical_certification_passed": False,
        },
    )
    text = """# 扩展历史币池的行情与实际资金费

本轮完成数据采集及独立复算，新增收益账户 **0**。总账户扣费年化20%/30%、最大回撤15%、夏普1、容量和前向证据继续未完成；所有实际经济指标保持未知。既有30个资金费账户的失败和其他历史结果完整保留。

| 数据证据 | 本轮结果 | 实际含义 |
| --- | ---: | --- |
| 历史资格时点 / 成员槽位 | 89 / 2,670 | UTC周一及两个年度冷启动时点；连续账户不额外使用2026冷启动的重置 |
| 采集名称并集 | 175 | 含BTC参照，26个引用封存行情，149个补取；并集只是下载安排，不能作为交易特征 |
| 五分钟源记录 | 27,386,202 | 全部与原ZIP或已封存六列引用逐值/时间核对；不补零、不填价 |
| 实际报告资金费事件 | 473,223 | 原468次接口加10次续页，保留实际毫秒时间、费率、标记价 |
| 完整日交叉比较 | 95,059 | 120日存在源差异；118日成交额、3日开盘、1日最低价不同，可重叠；收盘价均一致 |
| 五分钟内部缺口 | 300根 | LIT 2025-12-23缺210根；PUMP 2025-07-10缺90根；原因和可交易分段不能凭标签推断 |
| 执行流动性不可用记录 | 1,037,090 | 大量停交易后零成交量记录，仍保留原价格；价格存在不等于可成交 |
| 资金费零/未知标记价 | 12条 | 6个名称，全部早于首次候选且不在首个七天预测窗；不能用K线价替换 |

账户候选窗口仍为2025-01-01至2026-09-01之前；执行行情含2024-12-31，资金费含2024-12-25。源ZIP通过官方SHA256和CRC；149个新名称2,726份实际列出的月归档，5,452次下载请求，888,035,144字节，未超过预登记上限。26个旧引用只有已封存六列行情，本阶段没有冒充对其辅助列重新核验。3,307,098个时格位于各名称观察到的首尾源记录之外，与300根内部缺口分开，均未填补。

89个候选时点由独立原始日线前缀重算，选中集合及合格数量一致。按合约episode重计AERGO观察期，旧库存和账户高水位不重置；历史原始公告可用性仍采用已登记假设。新175名单依赖既有部分合约条款，尚未取得全历史交易身份认证，不把当前未见限制当作已经认证。分类待核5个为CRCL、INTC、MSTR、XAG、XAU合约；官方当前正文已捕获前4名称的股票/商品依据，黄金博客直接请求返回202空正文，失败收据保留。这些均不能代替港美股现金账户；后续按事前类型来源修订资格并在收益前重新冻结，不直接把今天的类型标签追溯成历史过滤器。

资金费原收集因第一页的零标记价停止6个名称，原失败回执和完整第一页均保留。新预算只续取这6个名称，10次接口完成，不重取其余143个。12条异常原值仍为0；本轮只允许其费率/时间作为原始报告数据，持仓资金费现金仍要求该事件正标记价及合约身份有效。独立检查全部175个原始或引用事件，并以Decimal重算2,670个[t-7天,t)窗口，最大求和误差2.22e-16。接口分页结束说明当前返回事件枚举结束，不证明历史费率日历完整或终止后每条记录都是客户应付现金。

本轮没有按收益择源、修补120个差异日或计算替换成员收益。第一轮预定族仍是纯资金费7日、固定成本门槛、MAX28、动量28和现金；不新增参数网格。固定40币引擎要求所有时点有行情，不能通过填入175名旧价格直接复用。下一步需新稀疏引擎：按真实可用源申请与成交，非空库存缺价/零结算标记或未知终端现金即阻断，事前截止触发退出并持续计失败与延迟，跨年库存、高水位、费用和恢复连续。只有完成独立财务复算和实际JSON新进程恢复后才报告收益，再判断条件性模块、两类独立机制及20%/30%总账户。数据工作不能替代这些目标。

## 复算入口

在TradingAgents目录使用Freqtrade的`.venv/bin/python`运行：

```text
python -m research_tools.expanded_market.audit_cohort --cohort results/research/expanded_market_cohort_20261026_v001 --output <新文件>
python -m research_tools.expanded_market.audit_market --market data/research/expanded_5m_market_20261026_v001 --cohort results/research/expanded_market_cohort_20261026_v001 --output <新目录>
python -m research_tools.expanded_market.audit_funding --inputs results/research/expanded_exact_funding_inputs_20261026_v001 --cohort results/research/expanded_market_cohort_20261026_v001 --output <新文件>
```

所有已执行源码、父完成收据、原始接口请求/回应、失败续采、审计输出均有SHA256绑定；重跑必须用新路径，保留失败与已完成目录。界面只显示采集/算术和缺口，不把它们放入收益栏。未读取凭证、下单、部署或推送。

官方类型依据：[INTC](https://www.binance.com/en/support/announcement/detail/d592f6ba938746cbadaaf5a8a714abd6)、[MSTR/CRCL](https://www.binance.com/en/support/announcement/detail/f083bdc8f37d4d45a0e0ec6fd5cd4086)、[XAG](https://www.binance.com/en/support/announcement/detail/bd92dee1018f421fae1ace119fc2f706)、[黄金/白银产品说明](https://www.binance.com/en/blog/futures/2106113045523004581)。官方1月8日当前说明中的1月5日可用性日期不能直接等同XAU最早上线或解释12月11日起源记录。
"""
    (output / "README.md").write_text(text)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "financial_accounts": 0,
            "historical_certification_passed": False,
        },
    )
    print({"report": str(output), "financial_accounts": 0, "profit_goal_met": False}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    publish(p.parse_args().output)
