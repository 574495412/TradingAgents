"""Attach actual process recovery, verify partitions and seal the account report."""

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.regime_program.reporting import partitions


def seal(docs, process_audit):
    if (docs / "completion.json").exists():
        raise FileExistsError("Preserve sealed publication")
    summary = read(docs / "summary.json")
    provenance = read(docs / "provenance.json")
    restored = read(process_audit)
    if (
        not restored["passed"]
        or restored["run_completion_sha256"] != provenance["run_completion_sha256"]
        or not all(
            c["passed"] and c["first_pid"] != c["restored_pid"] and c["open_positions_at_split"] > 0
            for c in restored["checks"]
        )
    ):
        raise ValueError("Actual interpreter recovery not proved")
    for name, key in [
        ("restart_worker.py", "worker_source_sha256"),
        ("verify_restarts.py", "controller_source_sha256"),
    ]:
        if sha(Path(__file__).with_name(name)) != restored[key]:
            raise ValueError("Recovery source changed")
    states_root = Path("results/research/regime_states_20261025_v002")
    states = pd.read_parquet(states_root / "states_daily.parquet")
    event_flags = pd.read_parquet(
        states_root / "event_risk_hourly.parquet"
    ).covered_fomc_post_release
    for row in summary["account_rows"]:
        root = Path(row["source_run"]) / "accounts" / row["source_account_id"]
        curve = pd.read_parquet(root / "equity_5m.parquet")
        trades = read(root / "trades.json")
        if not row["reused_reference"]:
            curve = curve.assign(gross_exposure=curve.gross_notional)
        recalculated = partitions(curve, trades, states, event_flags, row["capital"])
        stored = summary["details"][row["id"]]
        if (
            recalculated["periods"] != stored["periods"]
            or recalculated["state_contributions"] != stored["state_contributions"]
        ):
            raise ValueError("Unsealed sidecar differs from actual account partition")
        for kind in ["year", "90day"]:
            selected = [p for p in recalculated["periods"] if p["kind"] == kind]
            if not math.isclose(
                math.prod(1 + p["net_return"] for p in selected) - 1,
                row["metrics"]["net_return"],
                abs_tol=1e-12,
            ):
                raise ValueError("Calendar return blocks do not reconcile")
            if sum(p["lifecycle_by_exit_clock"]["complete_lifecycles"] for p in selected) != len(
                trades
            ):
                raise ValueError("Closed lifecycle period totals differ")
        for group in ["regime", "covered_fomc"]:
            contribution = sum(
                p["net_pnl_contribution"]
                for p in recalculated["state_contributions"]
                if p["group"] == group
            )
            if abs(contribution - row["metrics"]["net_pnl"]) > max(1e-7, row["capital"] * 1e-10):
                raise ValueError("State contribution does not reconcile")
    summary["process_restart_checks"] = restored["checks"]
    (docs / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    shutil.copyfile(process_audit, docs / "process_restart_audit.json")
    provenance.update(
        {
            "finalizer_source_sha256": sha(Path(__file__)),
            "process_restart_audit_sha256": sha(process_audit),
            "periods_recomputed_from_bound_curves": True,
            "all_state_and_period_totals_reconcile": True,
        }
    )
    (docs / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    names = summary["candidate_names"]
    lines = []
    for r in summary["account_rows"]:
        m = r["metrics"]

        def pct(x):
            return "—" if x is None else f"{x * 100:.3f}%"

        lines.append(
            "|"
            + "|".join(
                [
                    r["phase"],
                    names[r["candidate"]],
                    str(r["capital"]),
                    str(r["slippage_bps"]),
                    pct(m["net_return"]),
                    pct(m["win_rate"]),
                    pct(m["max_drawdown"]),
                    "—" if m["sharpe"] is None else f"{m['sharpe']:.2f}",
                    str(m["closed_trades"]),
                    str(m.get("matched_closed_pairs", "—")),
                    f"{m['fees']:.2f}",
                    f"{m['funding_net']:.2f}",
                    f"{m.get('open_pnl_with_exit_reserve', 0):.2f}",
                    str(r["capacity"].get("hedge_failures", 0)),
                ]
            )
            + "|"
        )
    text = (
        """# 现货对冲出现小幅净收益，条件持有未改善，样本与容量仍不足

2026-10-26。20个新账户＋6个原封存现金引用全部核验。它是同一资金费机制的固定持有／成本条件对照，不是两个独立收益源；混合市场分散研究不替代永续策略主线。多策略总账户尚未运行，20%年化／15%回撤／夏普1目标未完成，实际栏仍为空。

压力滑点每腿每边10bps，另收现货佣金10bps、永续5bps。1万USDT跨年连续608日：固定持有整段净收益1.422%、CAGR0.852%、最大观察回撤0.300%、日夏普5.25，仅2个完整周期；条件持有整段0.792%、CAGR0.475%、回撤0.336%、夏普2.00，仅7周期。完整净胜率分别100%和85.71%，小样本不能证明稳定胜率或高收益能力。高夏普也不补足收益量级和周期证据。

2025独立终点账户压力净收益：固定1.146%、条件0.747%；2026年1—8月冷启动为固定0.126%、条件−0.175%。它们与跨年继承库存的账户分开，不拼接。全年／90日／当时BTC状态和已覆盖FOMC贡献已从同一条净值重算，损益和退出周期总数核对。

条件规则降低了实际平均总敞口（跨年压力：31.74%对固定43.85%），也增加了换手费用；不能把全部差额归因于费用或当作同风险下的收益增量。停止这组28日／7日门槛和60日成本回收规则的参数搜索。下一轮优先独立经济来源；低收益对冲模块是否值得用于配置，仍需要更多周期和真实退出证据。

## 全部成本、期间与容量账户

下表成本列是每腿每边滑点，佣金另计。净值包含现货资产、永续标记浮亏、实际结算资金费和未平退出费用准备。净胜率只统计闭合生命周期；未平PnL一直计入整段净收益，不隐去其正负损益。

|区间|方案|本金USDT|滑点bps|整段净收益|完整净胜率|观察回撤|日夏普|闭合周期|匹配周期|手续费|资金费净额|未平周期净PnL|失败对冲|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(lines)
        + """

10万压力情景的固定／条件收益仍为正，但均在阶段终点出现一个部分退出／未匹配对冲记录，因此此协议容量未通过。100万条件方案压力净收益−2.425%，43次失败对冲；固定方案虽净值为正，却有未平生命周期和失败对冲，不将其视作容量合格。已登记的终点只预留一根5分钟退出，这是特定协议压力测试，不能推广成市场一定无法承载10万。后续若验证分批退出，必须预登记、支付其全部费用并保留本次结果；不把延长退出当成新增alpha。

## 资金、费用与风险如何核算

沿用旧持有基线的25%现货预算／70%永续钱包／5%现金，初始总敞口约0.5，没有放大到其他试验的0.8去追收益。现货买入手续费扣币，空头按实际净基础数量对冲；期货卖空不会增加可用现金。两个市场独立按下一根完成5分钟成交收盘价和各自实际成交额1%上限撮合；缺腿或不等量触发随后退出，部分成交不伪装成原子对冲。

现货资产升值不等于期货可用保证金；调拨只在初始或完全空仓时发生。原始调拨时间与下一估值点生效时间分别记录，边界净值不追溯修改。20个账户独立逐笔重建现金、币库存、永续成本基础、资金费与净值，并重算保守风险包络；最大净值误差约2.45e−9USDT。另用原始参考价和净基础数量分解价格PnL、实际费用、滑点与退出准备，合计与净损益一致；原运行指标中的整笔订单滑点代理单独保留，避免把买入手续费币的滑点重复归因。

4次恢复均实际落盘检查点、结束首进程、由新Python解释器读回，两次为首笔入场、两次为跨年，4个断点各有2笔持仓；全部净值、账本和最终状态精确一致。旧v001不完整尝试保留，三个已计算账户的全部净值和经济指标原样复现；仅补充现金调拨的明确估值时间，未按收益更换参数。原v001有2个逐账户审计完成，第3个被钱包时点问题拦截；整批未完成，不混为同一状态。

维持保证金使用5%研究压力值，退出预警20%，不是交易所历史阶梯的认证。标记高价＋现货低价构成非同时极值的保守包络，不冒充精确逐笔回撤或真实强平模拟。本轮无模型清算风险触发，仍缺历史阶梯、订单簿、整手／步长及真实交易所清算证据。

## 数据与来源边界

真实现货归档42份与实际资金费3,834笔沿用前序封存；新增标记价42份官方SHA256归档，每币184,032根完整5分钟标记OHLC。两币2026年6月月档都缺6月29日，各288根；保存原缺口文件，以官方只读接口的576根真实记录补源，未前填。全部1,472,256个标记价原值独立CSV／JSON重读一致。两次失败采集目录保留。

[官方归档定义固定版本](https://github.com/binance/binance-public-data/blob/d571690eb66760763b0bd6e27e8cd35603c7e244/python/README.md)、[官方标记价接口](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)、[官方清算与标记价说明](https://www.binance.com/en-NG/support/faq/detail/360033525271)仅用于机制与数据定义，不能证明本研究盈利或历史档位完整。费用使用公开普通用户现货10bps与永续5bps研究假设，非用户实际或完整历史手续费认证。

已有2024训练持有基线9账户保留；它当时因永续主线范围调整封存，不能重复算作新收益源。本次BTC/ETH为固定核心币试点，不能代表全40币或股票适配；2025/2026历史已查看，不是新的独立样本外。两方案均因生命周期／聚类证据不足而未准入，没按单策略全年20%淘汰，也没把代码测试作为盈利证据。

[完整摘要与状态分期](summary.json)、[全部账户CSV](accounts.csv)、[冻结执行方案](execution_plan.json)、[实际进程恢复核验](process_restart_audit.json)、[标记价原始核验](mark_data_summary.json)、[来源与阶段核验](provenance.json)。
"""
    )
    (docs / "README.md").write_text(text)
    manifest = {
        p.name: sha(p) for p in docs.iterdir() if p.is_file() and p.name != "completion.json"
    }
    save(
        docs / "completion.json",
        {
            "manifest": manifest,
            "audit_passed": True,
            "actual_accounts": 20,
            "reused_references": 6,
            "profit_proven": False,
        },
    )
    print(
        {
            "sealed": True,
            "new_accounts": 20,
            "cash_references": 6,
            "actual_process_restarts": len(restored["checks"]),
            "all_periods_and_state_contributions_reconciled": True,
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docs", type=Path, required=True)
    p.add_argument("--process-audit", type=Path, required=True)
    a = p.parse_args()
    seal(a.docs, a.process_audit)
