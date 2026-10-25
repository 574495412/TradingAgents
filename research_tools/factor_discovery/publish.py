"""Publish every attempted mechanism and continuous-account result without selection edits."""

import argparse
import csv
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha

NAMES = {
    "slow_trend": "慢速趋势共识",
    "flow_trend": "真实资金流确认趋势",
    "breakout_acceptance": "突破与主动成交确认",
    "pullback_continuation": "回调后趋势延续",
    "absorption_release": "吸收后释放",
    "funding_trend": "资金费拥挤过滤趋势",
    "mechanism_ridge72": "新机制线性模型",
    "price_ridge72": "同源价格线性对照",
    "regime_flow_hgb72": "市场状态＋真实资金流浅树",
    "regime_price_hgb72": "同源价格浅树对照",
}


def publish(output, docs):
    base = Path("results/research")
    roots = [
        base / "factor_discovery_20261025_v001",
        base / "factor_discovery_nonlinear_20261025_v001",
    ]
    studies = []
    identities = {}
    rows = []
    for root in roots:
        receipt = read(root / "completion.json")
        s = read(root / "summary.json")
        if (
            sha(root / "summary.json") != receipt["summary_sha256"]
            or sha(root / "plan.json") != s["plan_sha256"]
        ):
            raise ValueError("Frozen study identity mismatch")
        if not s["audit"]["passed"] or not all(a["passed"] for a in s["audit_accounts"]):
            raise ValueError("Account audit failed")
        for row in s["account_rows"]:
            result = root / "accounts" / row["id"] / "result.json"
            if sha(result) != row["result_sha256"]:
                raise ValueError("Account result changed")
            report = read(result)
            rows.append({**row, "risk_halted": report["risk_halted"]})
        studies.append(s)
        identities[str(root.resolve())] = receipt["summary_sha256"]
    if len(rows) != 126 or len({r["id"] for r in rows}) != 126:
        raise ValueError("Experiment budget differs")
    plan = read(roots[0] / "plan.json")
    data = Path(plan["data"])
    coverage = read(data / "coverage.json")
    result = {
        "schema": "factor-discovery-publication.v1",
        "screen_groups": [g for s in studies for g in s["screen_groups"]],
        "account_rows": rows,
        "factor_catalog": studies[0]["factor_catalog"],
        "selections": [s["selection"] for s in studies],
        "admitted_candidates": [c for s in studies for c in s["selection"]["admitted_candidates"]],
        "audit_passed": True,
        "audit_account_count": len(rows),
        "source_summary_hashes": identities,
        "market_rows": sum(c["rows"] for c in coverage),
        "limitations": [
            "2025及2026年1—8月已经观察，本轮属于事后研发；旧失败及搜索预算继续保留。",
            "9月是新延长的历史段；登记前已查看BTC资金费返回中的价格，不能标为全盲样本外。",
            "原七个窗口之外补做连续365／273天账户；9月独立冷启动只作诊断，不重置连续账户的风控停机状态。",
            "六个当代流动性合约仍存在选择及幸存者偏差。",
            "第一轮手工机制与第二轮固定浅树合计126账户；无参数网格，未删除失败结果。",
        ],
        "independent_oos": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    docs.mkdir(parents=True, exist_ok=False)
    save(output / "summary.json", result)
    save(
        output / "completion.json",
        {"summary_sha256": sha(output / "summary.json"), "accounts": len(rows)},
    )
    save(
        docs / "provenance.json",
        {
            "publication": str(output.resolve()),
            "sha256": sha(output / "summary.json"),
            "studies": identities,
            "data_completion_sha256": sha(data / "completion.json"),
            "source_archives": sum(len(c["sources"]) for c in coverage),
            "market_rows": result["market_rows"],
        },
    )
    save(docs / "summary.json", result)
    columns = [
        "id",
        "candidate",
        "phase",
        "window",
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
        "slippage_cost",
        "funding_net",
        "risk_halted",
        "filled_request_ratio",
        "max_exit_delay_minutes",
    ]
    with (docs / "accounts_126.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            flat = {**row, **row["metrics"], **row["capacity"]}
            writer.writerow({k: flat.get(k) for k in columns})

    def p(v):
        return "—" if v is None else f"{100 * v:.2f}%"

    def n(v):
        return "—" if v is None else f"{v:.2f}"

    lines = [
        "# 真实资金流与市场状态因子研发 · 2026-10-25",
        "",
        "本轮已完成新因子及126个账户验证，但没有研发出可确认的稳健高收益因子。吸收后释放有局部历史收益，连续2026账户失败；浅树没有改善泛化。保留全部失败，不把单月高收益、提高仓位或重置风控当成有效收益源。",
        "",
        "## 数据与方法",
        "",
        f"恢复并核验六合约共{result['market_rows']:,}条五分钟行情、真实主动买入成交额、总成交额及资金费；行情跨度2023-08-01至2026-10-01（右开）。官方归档共{sum(len(c['sources']) for c in coverage)}个，逐个SHA-256校验；旧数据的OHLCV与冻结父数据逐行一致。资金费来自已封存旧数据及Binance只读连接器9月记录。",
        "",
        "新增机制源码统一在 `factors/common/mechanisms.py`，市场状态条件在 `factors/common/regime.py`；旧74列定义及父研究不变。六个新机制特征：多尺度趋势共识、真实主动成交24／72期、主动成交异常、价格响应偏离、资金流趋势交互。没有真实主动成交字段的股票数据保持未知。",
        "",
        "第一轮6手工规则＋2线性模型，固定四个2025年28天筛选窗、两档成本共64账户；冻结后续名单后补做连续2025全年、连续2026年1—9月、9月独立冷启动及容量，共90账户。第一轮后续候选按预登记规则为吸收后释放、资金费趋势，并保留价格对照；这只是研究名单，不是准入。",
        "",
        "第二轮仅2个固定浅树：价格与市场状态对照、再加真实资金流；共36账户。180天滚动、月度重训、只用日决策时钟及已成熟72小时标签；训练内1/99分位裁剪目标，固定80树、最多7叶、深度3、叶最少80样本、学习率0.05、L2=10。关闭随机早停切分，无超参数网格。完整后续时段在短窗筛选冻结后才进入经济评分。",
        "",
        "总仓位80%沿用原64风险预算，不使用杠杆；不能与前轮40%直接按收益比较。每日调仓、信号最多持续7日再空一天，实际退出仍受容量约束；手续费5bps，滑点3或10bps，真实费率资金费与开放仓位退出准备入账。目标回撤15%不是保证：五分钟延迟与部分成交可使实测超限。",
        "",
        "## 预登记准入门槛",
        "",
        "两档成本下，四窗平均28天净收益≥3%、至少3/4窗盈利、最差回撤≤15%、完整交易≥30、闭合PF≥1.1、夏普中位数≥0.5。当前无候选同时达到。后续连续或9月结果不反向修改筛选。",
        "",
        "## 四个28天窗口的合并描述",
        "",
        "收益为冷启动窗口算术平均，回撤为窗口最差值，夏普为窗口中位数，胜率按完整生命周期合并；都不是拼接连续账户。",
        "",
        "|方案|总单边成本bps|平均28天净收益|胜率|最差回撤|夏普中位数|闭合PF|完整交易|盈利窗口|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in result["screen_groups"]:
        lines.append(
            f"|{NAMES[g['candidate']]}|{5 + g['slippage_bps']}|{p(g['mean_window_return'])}|{p(g['complete_trade_win_rate'])}|{p(g['worst_observed_drawdown'])}|{n(g['median_window_sharpe'])}|{n(g['pooled_complete_profit_factor'])}|{g['complete_trades']}|{g['positive_windows']}/{g['windows']}|"
        )
    lines += [
        "",
        "## 连续账户及延长历史段",
        "",
        "以下为整段扣费总收益；不是月收益或预期年化。夏普采用UTC日净值、365日年化、无风险利率0。9月段是另开账户冷启动，不能替代保留风控状态的1—9月连续账户。",
        "",
        "|区间|方案|单边成本bps|整段净收益|胜率|回撤|夏普|闭合PF|完整交易|触发停机|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r["phase"] not in ("continuous", "extension"):
            continue
        m = r["metrics"]
        lines.append(
            f"|{r['start']}—{r['end']}|{NAMES[r['candidate']]}|{5 + r['slippage_bps']}|{p(m['net_return'])}|{p(m['win_rate'])}|{p(m['max_drawdown'])}|{n(m['sharpe'])}|{n(m['profit_factor'])}|{m['closed_trades']}|{'是' if r['risk_halted'] else '否'}|"
        )
    lines += [
        "",
        "关键反例：吸收后释放在2025年10月短窗约+15%，9月独立冷启动高成本净收益+3.55%，但连续2026高成本为-15.31%，且触发停机。前者不能救活后者。连续2025高成本+8.58%亦不足证明稳定高收益。",
        "",
        "## 容量压力",
        "",
        "固定2026年1—9月，不混合不同资金量。只有先出现可持续收益，容量才有经济意义。建仓不足导致亏损收窄，不是可部署容量。",
        "",
        "|方案|资金USDT|单边成本bps|整段净收益|回撤|入场成交/请求|最长退出分钟|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["phase"] == "capacity":
            lines.append(
                f"|{NAMES[r['candidate']]}|{r['capital']}|{5 + r['slippage_bps']}|{p(r['metrics']['net_return'])}|{p(r['metrics']['max_drawdown'])}|{p(r['capacity']['filled_request_ratio'])}|{n(r['capacity']['max_exit_delay_minutes'])}|"
            )
    lines += [
        "",
        "## 可核查证据",
        "",
        "[126账户完整明细](accounts_126.csv)包含每个短窗、整段、费用、资金费、开放盈亏、停机及成交比例。[摘要](summary.json)保留全部10个候选／对照、两次筛选与拒绝原因。[来源回执](provenance.json)绑定不可覆盖运行目录及摘要哈希。所有126账户通过独立资金费、现金流、完整生命周期、容量及NAV重算，最大重算误差低于2e-9USDT（以本轮实际输出为准）。",
        "",
        "代码可复现：",
        "",
        "```sh",
        "# From TradingAgents; optional sklearn dependencies may use a separate target directory.",
        "/Users/chrise/GitHub/freqtrade/.venv/bin/python -m research_tools.factor_discovery.run --output results/research/NEW_MECHANISM_RUN",
        "PYTHONPATH=/tmp/ta-discovery-ml-deps:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/chrise/GitHub/freqtrade/.venv/bin/python -m research_tools.factor_discovery.nonlinear --output results/research/NEW_NONLINEAR_RUN",
        "```",
        "",
        "第二轮默认读取本次登记父运行，保持相同输入对照。安装可选依赖时使用 `research_tools/factor_discovery/requirements-ml.txt`，不要求修改现有生产环境。",
        "",
        "## 外部依据与边界",
        "",
        "[NBER加密收益研究](https://www.nber.org/papers/w24877)和[AQR趋势研究](https://www.aqr.com/insights/research/journal-article/a-century-of-evidence-on-trend-following-investing)支持把动量作为可检验假设，不能证明本项目策略高收益。[Binance官方归档定义](https://github.com/binance/binance-public-data)用于真实主动成交字段与CHECKSUM；[浅树官方定义](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)用于固定模型参数与关闭随机早停。",
        "",
        *result["limitations"],
        "",
        "五分钟1%成交额及next-close代理不是盘口撮合认证；历史合约精度、最小下单额与资金费边界仍不完整。此前所有研究继续记入搜索记录。未部署、未开单；不能把追加历史评估标为新实盘能力。",
    ]
    (docs / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--docs", type=Path, required=True)
    a = p.parse_args()
    publish(a.output.resolve(), a.docs.resolve())
