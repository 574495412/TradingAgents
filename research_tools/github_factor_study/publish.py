"""Source survey and complete local results; never convert README claims into verified profit."""

import argparse
import csv
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha

NAMES = {
    "pst_ewmac32": "PST·标准化趋势32/128",
    "pst_accel4": "PST·趋势加速4",
    "pst_breakout10": "PST·通道突破10",
    "qlib_corr20": "Qlib·价量相关20",
    "qlib_resi20": "Qlib·价格残差20",
    "sentinel_champion": "Sentinel·BTC状态动量",
    "sentinel_carry21": "Sentinel·资金费＋动量21",
    "sentinel_carry14_buffer": "Sentinel·14日排序＋持仓缓冲",
    "liquid_long_control": "同风险流动性做多对照",
}
NOTES = {
    "microsoft/qlib": (
        "A·标准公式与评估基线",
        "已查源码和基准配置；原工程已接入相同commit的分钟Alpha158，本轮仅另查日频CORR20/RESI20，不算新发现158个盈利因子。",
        "官方CSI300 LightGBM/Alpha158表列年化0.0901、IR1.0164、回撤-0.1038；这是股票完整工作流结果。",
        "股票费用、涨跌停、学习器和币种市场不同；两个价量因子移植不继承整个模型收益。",
    ),
    "pst-group/pysystemtrade": (
        "A·规则源码复现",
        "已查EWMAC、加速、突破和波动率实现；独立数学适配，未执行或混入上游交易系统。",
        "公开可解释趋势规则，没有在所查文件中找到本币池的已验证收益保证。",
        "商业日历改为UTC自然日，共同风险和账户执行不同；仅复现公式，不声称复现Carver整体系统。",
    ),
    "iterativv/NostalgiaForInfinity": (
        "B·复用已有研究",
        "已查当前版本和回测文档，并核对现有Freqtrade旧版研究记录。",
        "作者将回测结果放在提交评论；旧本地1倍版本20个月基础成本净收益约8.18%，但不是本轮当前commit复现。",
        "完整策略含加减仓及多类入场，不是单因子；旧分期样本不足、版本不同，不能拿3倍收益与无杠杆新因子比较。",
    ),
    "ICT-FinD-Lab/alphagen": (
        "B·挖掘方法",
        "已读官方README与Qlib适配要求；是强化学习公式挖掘框架，本轮未运行其搜索。",
        "KDD2023论文配套代码，可输出公式池；没有已核验本币池的可交易因子收益。",
        "主要股票数据；无明确根许可证记录，不能直接搬运源码；训练搜索和最终持仓评估是不同证据。",
    ),
    "je-suis-tm/quant-trading": (
        "B·公式参考",
        "README初筛，教学策略合集；没有把合集当成盈利证明。",
        "列有Dual Thrust等多类策略与示例。",
        "缺少同一币池、同一账户和完整成本的统一验证结果；本轮未运行。",
    ),
    "adensvaz/sentinel-hyperliquid": (
        "A·优先验证但须核对口径",
        "已查策略、历史回测脚本、当前配置、MIT许可；本轮实际移植4个版本／规则。",
        "README声称方向动量CAGR约39–74%、资金费约123%；历史carry脚本每侧权重和为1，即原始gross2。",
        "当前carry配置自述203笔运行账本PF0.975、21日版本近乎无优势；README、历史脚本和当前14日带缓冲版本不是同一配置。Hyperliquid与Binance也不能混称。",
    ),
    "OctopusTakopi/funding-rate-alpha": (
        "B·反证与审计参考",
        "README与元数据初筛；本轮未运行其全历史三交易所研究。",
        "报告区分资金费收入、价格风险和净交易收益，检验大量合约与退市样本。",
        "资金费可预测性、资金费收入不等于净alpha；作者的研究声明尚未在本地全量复现。",
    ),
    "Carson1332/Quant-FundingArbs": (
        "C·不能当高收益答案",
        "已读方法修正与已知限制，记录原高Sharpe错误。",
        "作者披露旧Sharpe>6受时间对齐和费用错误影响；修正后真实成本配置年收益约1.42%、Sharpe1.97。",
        "现参数扫描是样本内，且有幸存者偏差；低波动Sharpe高不代表高收益。",
    ),
    "prams2104/crypto-momentum-backtest": (
        "C·负结果",
        "README初筛，作者明确披露动量样本外失败及反转成本侵蚀。",
        "作者2025–2026动量净Sharpe约-2.18；反转成本前很强但扣费亏损。",
        "不能摘取训练期盈利或成本前alpha作为可交易证据；本轮未重复运行。",
    ),
    "mattromano/kraken-strategy-harness": (
        "B·跨市场相对强弱待补源",
        "README初筛；没有股票指数时点与本地可复现输入，不运行。",
        "作者ETH/Russell20/50示例CAGR41.6%、夏普0.88，同时最大回撤约69%。",
        "单资产样本和高回撤；未见根许可证，未复制代码；需严格对齐指数休市和加密24/7可用时间。",
    ),
    "Quantitative-Systems/crypto-platform": (
        "C·证据链待补齐",
        "已读carry压力报告及所查回测引擎；未找到与该carry报告对应的完整生成链和逐账户输入。",
        "报告写基准平均净收益55.93%，但所查树中carry条目为结果文件，不能据报告宣称本地已复现。",
        "平台是大型框架而非已证实盈利因子。不能把报告里的ROBUST或Certified字样当独立认证。",
    ),
    "idreesm18/crypto-perp-funding-carry": (
        "C·费用敏感",
        "README初筛；作者披露点差压力和不同市场阶段。",
        "作者5bps/边时夏普约0.37、年收益约46bps；7bps及以上结果转负。",
        "低费率敏感，不能作为当前8/15bps单边成本下高收益证据；未运行其原市场数据。",
    ),
    "SAGISaiKrishna/crypto-arbitrage-strategy": (
        "C·暂不采用收益宣传",
        "README与策略／公式文档初筛；未运行合约或任何实盘接口。",
        "作者宣称年化35.2%、夏普6.59，同时说明2021高资金费对结果影响很大。",
        "包含MockPerpEngine和演示部署；README并不等于真实撮合证据。逐年成本、保证金与尾部尚未独立核验。",
    ),
}


def publish(output, docs):
    root = Path("results/research/github_factor_study_20261025_v001")
    plan = read(root / "plan.json")
    summary = read(root / "summary.json")
    if (
        sha(root / "summary.json") != read(root / "completion.json")["summary_sha256"]
        or sha(root / "plan.json") != summary["plan_sha256"]
    ):
        raise ValueError("Study identity changed")
    if not summary["audit"]["passed"] or not all(a["passed"] for a in summary["audit_accounts"]):
        raise ValueError("Incomplete account audit")
    rows = summary["account_rows"]
    if len(rows) != 46 or len({r["id"] for r in rows}) != 46:
        raise ValueError("Declared budget differs")
    pack = Path(plan["source_pack"])
    sources = []
    for p in sorted(pack.glob("*/metadata.json")):
        m = read(p)
        priority, review, claim, limitation = NOTES[m["repository"]]
        license = (m.get("license") or {}).get("spdx_id", "unspecified")
        if m["repository"] == "adensvaz/sentinel-hyperliquid":
            license = "MIT (read LICENSE; GitHub API NOASSERTION)"
        sources.append(
            {
                **m,
                "license_label": license,
                "priority": priority,
                "review": review,
                "upstream_claim": claim,
                "limitation": limitation,
                "evidence_status": "source_claim_not_local_profit",
                "metadata_sha256": sha(p),
            }
        )
    coverage = read(Path(plan["data"]) / "coverage.json")
    result = {
        "schema": "github-factor-publication.v1",
        "sources": sources,
        "candidate_names": NAMES,
        "account_rows": rows,
        "selection": summary["selection"],
        "audit": summary["audit"],
        "source_deviations": plan["source_deviations"],
        "data_profile": {
            "symbols": 40,
            "five_minute_rows": sum(c["bars"] for c in coverage),
            "funding_events": sum(c["funding_events"] for c in coverage),
            "schedule_exceptions": 4,
            "bar_start": "2023-08-01",
            "end_exclusive": "2026-09-01",
            "survivorship_free": False,
        },
        "run": str(root.resolve()),
        "source_summary_sha256": sha(root / "summary.json"),
        "orders_enabled": False,
        "production_eligible": False,
        "independent_oos": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    docs.mkdir(parents=True, exist_ok=False)
    save(output / "summary.json", result)
    save(
        output / "completion.json",
        {"summary_sha256": sha(output / "summary.json"), "audited_accounts": 46},
    )
    save(docs / "summary.json", result)
    save(docs / "source_catalog.json", sources)
    save(
        docs / "provenance.json",
        {
            "run": str(root.resolve()),
            "run_summary_sha256": sha(root / "summary.json"),
            "publication": str(output.resolve()),
            "publication_summary_sha256": sha(output / "summary.json"),
            "data_completion_sha256": sha(Path(plan["data"]) / "completion.json"),
            "source_pack_receipts_sha256": sha(pack / "file_receipts.json"),
        },
    )
    columns = [
        "id",
        "candidate",
        "phase",
        "start",
        "end",
        "capital",
        "slippage_bps",
        "net_return",
        "win_rate",
        "max_drawdown",
        "sharpe",
        "profit_factor",
        "closed_trades",
        "closed_net_pnl",
        "open_pnl_with_exit_reserve",
        "fees",
        "funding_net",
        "turnover",
        "mean_absolute_net_exposure",
        "filled_request_ratio",
        "max_exit_delay_minutes",
        "risk_halted",
    ]
    with (docs / "accounts_46.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            values = {**r, **r["metrics"], **r["capacity"]}
            writer.writerow({k: values.get(k) for k in columns})

    def pct(v):
        return "—" if v is None else f"{100 * v:.2f}%"

    def n(v):
        return "—" if v is None else f"{v:.2f}"

    lines = [
        "# GitHub因子检索与本地复现 · 2026-10-25",
        "",
        "检索并固定13个仓库的commit／许可／元数据，核查22份具体源码、配置或报告，补做40币池46个完整扣费账户。最值得继续的是Sentinel历史资金费＋动量横截面排序，但没有达到本轮双成本及跨期门槛，不能说已经找到可部署的稳定盈利因子。",
        "",
        "## 原始来源与作者收益声明",
        "",
        "下表是源码检索与作者声明，不是本项目已验证收益。星数只用于检索排序，不证明可盈利。",
        "",
        "|仓库|查询时Stars|优先级|查阅深度与结论|作者结果／边界|",
        "|---|---:|---|---|---|",
    ]
    for s in sources:
        lines.append(
            f"|[{s['repository']}]({s['url']})|{s['stars']}|{s['priority']}|{s['review']}|{s['upstream_claim']} {s['limitation']}|"
        )
    lines += [
        "",
        "源代码按具体commit下载，副本在来源包中只读保存，没有运行下载脚本、安装器、交易服务或第三方网页。QLib原工程已存在相同commit的分钟因子；本轮日频适配不增加所谓独立alpha数量。PST数学公式独立实现，GPL交易系统未被混入Apache研究应用。Sentinel根LICENSE实际为MIT，GitHub API的NOASSERTION单独保留。",
        "",
        "## 固定复现范围与风险口径",
        "",
        "40个已封存当代合约池，先前28个完整UTC日平均成交额≥5000万USDT，再取前30；不是含退市币的无幸存者偏差全集。真实五分钟数据12,983,040条，截止2026-09-01（右开），不冒充最新行情。",
        "",
        "资金费采用各币实际1／4／8小时结算事件。TRB、AXS、MASK、SKL在2026-06-24的4小时时点元数据看似缺一笔，但官方只读API同样返回00、08、12、16，没有04结算。已核对对应费率，记录为实际日历例外，没有补零或捏造结算。每个行情归档校验官方CHECKSUM，再与已封存原生OHLCV核对。",
        "",
        "同一账户：初始1万USDT、目标总仓位80%、无杠杆、5bps手续费＋3／10bps滑点、五分钟成交额1%约束，包含资金费及开放仓位退出准备。每仓8%止损和账户15%回撤目标是本地共同约束，观测回撤可超限。源策略的原始gross、止损、动态降风险、订单重试与交易所不同；这是规则移植，不是原系统全链条收益复现。",
        "",
        "2024只用于PST预测值尺度及两个Qlib因子的方向；2025连续账户筛选后固定3个后续研究候选，再计算2026和跨年账户。所有历史此前可得，不能宣称独立未见样本外。高收益准入目标为2025两档成本均≥20%、回撤≤15%、闭合交易≥30、PF≥1.1、夏普≥0.5；没有候选全部通过。",
        "",
        "## 各候选：2025整年扣费账户",
        "",
        "|方案|单边总成本bps|全年净收益|胜率|回撤|夏普|闭合PF|完整交易|开放盈亏USDT|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["phase"] != "screen2025":
            continue
        m = r["metrics"]
        lines.append(
            f"|{NAMES[r['candidate']]}|{5 + r['slippage_bps']}|{pct(m['net_return'])}|{pct(m['win_rate'])}|{pct(m['max_drawdown'])}|{n(m['sharpe'])}|{n(m['profit_factor'])}|{m['closed_trades']}|{n(m['open_pnl_with_exit_reserve'])}|"
        )
    lines += [
        "",
        "## 冻结后续名单后的连续验证",
        "",
        "以下均为整段总收益，不能当月收益。2026为1—8月243天，跨年为2025-01-01至2026-09-01共608天。跨年账户保留停机和持仓状态；没有在年界强制平仓记胜率或重置风控。零交易现金段不是收益有效证据。",
        "",
        "|区间|方案|单边成本bps|整段净收益|胜率|回撤|夏普|闭合PF|完整交易|停机|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r["phase"] not in ("diagnostic2026", "continuous2025_2026"):
            continue
        m = r["metrics"]
        lines.append(
            f"|{r['start']}—{r['end']}|{NAMES[r['candidate']]}|{5 + r['slippage_bps']}|{pct(m['net_return'])}|{pct(m['win_rate'])}|{pct(m['max_drawdown'])}|{n(m['sharpe'])}|{n(m['profit_factor'])}|{m['closed_trades']}|{'是' if r['risk_halted'] else '否'}|"
        )
    lines += [
        "",
        "## 最有潜力候选的收益来源",
        "",
        "Sentinel历史21日排序在2025基础成本下净收益+23.59%，已结束生命周期净损益约2182.48USDT，开放盈亏约157.37USDT；资金费净收入约342USDT。收益大部分来自相对价格表现与组合选择，并非只靠资金费收入。实际净敞口平均绝对值约8.10%，目标中性不等于实际成交完全中性。高成本净收益降至+12.93%、闭合PF约1.06；不能忽略换手与两腿成交不平衡。",
        "",
        "原历史脚本每侧权重1/K，合计gross2；不能把其高年化数字搬到本地0.8gross。当前14日带缓冲版本与历史21日版本分开，不能拿README旧数字代表现配置。当前14日版本在流动性约束下可能因不足24个合格币而保持现金，不把零收益／零回撤称为策略优势。",
        "",
        "## 容量：资金与成交约束",
        "",
        "|方案|初始USDT|单边成本bps|608天净收益|回撤|成交金额/入场请求金额|最长退出分钟|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["phase"] != "capacity":
            continue
        m = r["metrics"]
        c = r["capacity"]
        lines.append(
            f"|{NAMES[r['candidate']]}|{r['capital']}|{5 + r['slippage_bps']}|{pct(m['net_return'])}|{pct(m['max_drawdown'])}|{pct(c['filled_request_ratio'])}|{n(c['max_exit_delay_minutes'])}|"
        )
    lines += [
        "",
        "入场采用单次五分钟TTL，不对现有生命周期补仓；大资金可能只能建立不对称小仓位，不能把亏损收窄视为更大容量，也不能由该代理断言真实执行一定无法实现。来源原系统可能以更多时段重试。盘口深度、历史最小下单量、真实资金费结算价格与交易所风控仍未完整核验。",
        "",
        "## 全部证据与复现",
        "",
        "[46账户CSV](accounts_46.csv)、[来源目录](source_catalog.json)、[完整摘要](summary.json)、[来源哈希](provenance.json)均可核查；没有删掉负结果或重新挑选有利时期。逐账户资金费、现金、NAV、完整生命周期和成交量参与率独立重算通过。",
        "",
        "```sh",
        "# From TradingAgents, use a fresh output directory; parent data and source pack stay immutable.",
        "OPENBLAS_NUM_THREADS=1 /Users/chrise/GitHub/freqtrade/.venv/bin/python -m research_tools.github_factor_study.run --output results/research/NEW_GITHUB_FACTOR_RUN",
        "```",
        "",
        "源代码对应参数及全部改动见原运行plan.json.source_deviations；本轮没有参数网格和杠杆搜索，未重置旧352账户的研发搜索记录。历史公开源不是当前实盘盈利证明。未推送、未部署、未开单。",
    ]
    (docs / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--docs", type=Path, required=True)
    a = p.parse_args()
    publish(a.output.resolve(), a.docs.resolve())
