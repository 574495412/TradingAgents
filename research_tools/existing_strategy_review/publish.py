"""Publish a compact, source-linked review without merging incompatible account scopes."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value):
    return "—" if value is None else f"{value:.2%}"


def num(value):
    return "—" if value is None else f"{value:.2f}"


def main():
    root = Path("results/research/existing_strategy_review_20261028_v005")
    v = json.loads((root / "comparison.json").read_text())
    all_rows = json.loads((root / "accounts.json").read_text())["accounts"]
    out = Path("docs/research/regime_program/existing_strategy_review_20261028")
    periods = []
    for c in v["strict_cohorts"]:
        names = {r["strategy"] for r in c["rows"]}
        if (
            {"TrendControl", "NostalgiaForInfinityX7"} <= names
            and c["cohort"][6] == [0.0008]
            and c["cohort"][7] == [1.0]
        ):
            start, end = c["cohort"][3:5]
            label = (
                "连续608日"
                if start.startswith("2025") and end.startswith("2026-09")
                else "2024全年"
                if start.startswith("2024")
                else "2025全年"
                if start.startswith("2025")
                else "2026年1–8月"
            )
            periods.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "rows": [
                        r
                        for r in c["rows"]
                        if r["strategy"] in ["TrendControl", "NostalgiaForInfinityX7"]
                    ],
                    "engine_identity": c["cohort"][9],
                    "pairlist_identity": c["cohort"][8],
                }
            )
    periods.sort(
        key=lambda p: ["2024全年", "2025全年", "2026年1–8月", "连续608日"].index(p["label"])
    )
    assert len(periods) == 4
    matched = {
        "schema": "same-six-coin-native-comparison.v1",
        "capital_USDT": 1000,
        "symbols": ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE"],
        "leverage": 1,
        "fee_per_side_including_proxy": 0.0008,
        "native_engine_core_identical": True,
        "different_signal_frequencies_position_budgets_and_slots_are_preserved_strategy_behavior": True,
        "market_data_snapshots_required_by_each_frequency_differ": True,
        "periods": periods,
        "last_period_overlaps_two_prior_periods": True,
        "not_NFI_X8_author_contract": True,
    }
    (out / "matched_six_coin_comparison.json").write_text(
        json.dumps(matched, ensure_ascii=False, indent=2) + "\n"
    )
    replay = Path(
        "/Users/chrise/GitHub/freqtrade/user_data/crypto_quant/existing_strategy_review_20261028_v001"
    )
    native_new = [
        {
            **json.loads((p / "result.json").read_text()),
            "runtime": json.loads((p / "review_receipt.json").read_text()),
        }
        for p in sorted(replay.glob("S*"))
        if p.is_dir()
    ]
    receipt = {
        "original_parity": json.loads((replay / "original_parity.json").read_text()),
        "completion": json.loads((replay / "completion.json").read_text()),
        "jobs": native_new,
    }
    (out / "unchanged_native_replays.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    summary = {
        k: v[k]
        for k in [
            "schema",
            "accounts",
            "account_kinds",
            "markets",
            "native_strategy_classes",
            "duplicate_account_aliases",
            "summary_nodes_retained",
            "legacy_curves",
            "legacy_curves_with_explicit_capital",
            "quality",
            "strict_cohorts_with_multiple_accounts",
            "cohort_exclusions",
            "new_signal_or_parameter_variants",
            "new_replication_invocations",
            "new_unique_period_accounts",
            "new_independent_OOS_claim",
            "global_better_than_original_NFI_proven",
            "orders_enabled",
            "production_eligible",
        ]
    }
    summary.update(
        strict_NFI_X8_non_NFI_challenger_count=len(v["strict_NFI_X8_non_NFI_challengers"]),
        full_account_records_path=str((root / "accounts.json").resolve()),
        full_comparison_path=str((root / "comparison.json").resolve()),
        full_account_records_sha256=sha(root / "accounts.json"),
        full_comparison_sha256=sha(root / "comparison.json"),
        recorded_at_utc=datetime.now(UTC).isoformat(),
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 现有策略统一口径复核",
        "",
        "结论：有在某些年份收益更高的旧策略，但目前没有已证明跨期更优、可直接替换完整NFI X8原版的方案。旧高收益策略的后续年份补测后转亏；保留原版NFI作为公开复现基线，不恢复自研或参数优化。",
        "",
        "## 对齐后的直接对照",
        "",
        "以下为同一六币池、1000 USDT、1x、单边5bps手续费＋3bps费用代理，同一原生引擎核心。保留各策略信号周期、仓位预算和槽位；这不是同实际风险或相同交易次数的测试。回撤及夏普均从保存的完整持仓小时净值复算，未拿已平仓夏普替代。不同信号周期使用各自所需的数据快照和撮合粒度，仍属回测假设。",
        "",
        "| 区间 | 策略 | 账户净收益 | 小时观察回撤 | 日净值夏普 | 完整交易胜率 | 完整交易数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for period in periods:
        for r in period["rows"]:
            label = "旧NFI X7·1x" if r["strategy"] == "NostalgiaForInfinityX7" else r["strategy"]
            lines.append(
                f"| {period['label']} | {label} | {pct(r['net_return'])} | {pct(r['sampled_full_drawdown'])} | {num(r['daily_equity_sharpe'])} | {pct(r['win_rate'])} | {r['closed_trades']} |"
            )
    lines += [
        "",
        "2024年TrendControl收益60.22%，高于旧NFI的11.56%，同时小时回撤16.07%也高于1.20%。2025年TrendControl亏18.31%，旧NFI盈利6.68%；2026年前8月TrendControl收益更高，也承受更大回撤。连续608日，旧NFI收益与小时回撤两项都优于TrendControl。这支持按行情与区间判断，不能从一个训练年度选出永久赢家。",
        "",
        "连续608日与两个分段存在重叠，不能相加交易数或当成独立试验。所有区间都是已查看的历史；日夏普按365日、样本标准差及无风险率0计算，不是实盘或真正未见证据。",
        "",
        "## 补测原来被风险门槛挡住的高收益候选",
        "",
        "既有规则曾因回撤超过15%而停止后续年份评估。本轮没有把该阈值当正确前提：原样复现TrendControl的2024年350笔交易及完整订单，净收益一致后，固定原源码、原数据、原参数和8bps成本代理补测两个候选。341份原生核心源码哈希与原实验完全一致；没有降低阈值、改信号或调权重。",
        "",
        "| 旧策略 | 2024原结果 | 2025补测 | 2026年1–8月补测 | 连续608日补测 | 连续期小时回撤 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["TrendControl", "CompressionRetest"]:
        train = next(
            r
            for r in all_rows
            if r["kind"] == "native_account"
            and r["strategy"] == name
            and r["start"].startswith("2024-01-01")
            and r["end"].startswith("2024-12-31")
            and r["fees_per_side"]
            and abs(r["fees_per_side"][0] - 0.0008) < 1e-8
            and "compat" not in r["source_path"]
        )
        jobs = [x for x in native_new if x["candidate"] == name]
        vals = {x["phase"]: x for x in jobs}
        cont = vals["continuous"]
        lines.append(
            f"| {name} | {pct(train['net_return'])} | {pct(vals['validation']['net_return'])} | {pct(vals['holdout']['net_return'])} | {pct(cont['net_return'])} | {pct(cont['hourly_observed_drawdown'])} |"
        )
    lines += [
        "",
        "两者训练期高收益没有延续到连续账户。CompressionRetest保留原生2x和40币池，TrendControl为1x六币池，因此这张表是各自原配置的跨期诊断，不是等风险排名。旧经济筛查结果仍原样记录；本轮结论依据实际收益和风险路径，不依据15%门槛。只有8bps档补测，未推断其他成本档会单调变化。",
        "",
        "## 仍有盈利记录、值得保留的旧对照",
        "",
        "下面都是2025-01-01至2026-09-01。资本、币池、引擎和成本实现不同，不能据此横排谁优于NFI X8。",
        "",
        "| 对象 | 本金USDT | 累计净收益 | 全持仓观察回撤 | 日净值夏普 | 范围与限制 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    positive = []
    old40 = next(
        r
        for r in all_rows
        if r["kind"] == "native_account"
        and r["strategy"] == "NFI2023OneXCompact"
        and "/nfi40_run_v001/continuous/" in r["source_path"]
        and abs(r["fees_per_side"][0] - 0.0008) < 1e-8
    )
    portfolio = next(
        r
        for r in all_rows
        if r["kind"] == "native_wallet_sum"
        and r["start"] == "2025-01-01"
        and r["end"] == "2026-09-01"
        and abs(r["fees_per_side"][0] - 0.0008) < 1e-8
    )
    carry = next(
        r
        for r in all_rows
        if r["kind"] == "custom_account_with_curve"
        and "continuous2025_2026_sentinel_carry21_cap10000_slip3" in r["strategy"]
    )
    for title, r, note in [
        ("旧NFI X4·40币1x", old40, "小时mark观察；期末强制退出，历史可交易池边界未完全认证"),
        ("NFI＋Donchian", portfolio, "两个独立1000钱包之和，非共享保证金账户；小时观察"),
        (
            "Sentinel carry21本地移植",
            carry,
            "10000本金、gross0.8、40币，5分钟参考；不是Hyperliquid原版收益复现",
        ),
    ]:
        obs = r["saved_curve"]
        lines.append(
            f"| {title} | {r['capital']:.0f} | {pct(r['net_return'])} | {pct(obs['sampled_full_drawdown'])} | {num(obs['recomputed_daily_sharpe'])} | {note} |"
        )
        positive.append(
            {
                "label": title,
                "account_id": r["id"],
                "source_path": r["source_path"],
                "net_return": r["net_return"],
                "capital": r["capital"],
                "win_rate": r.get("win_rate"),
                "closed_trades": r.get("closed_trades"),
                "sampled_full_drawdown": obs["sampled_full_drawdown"],
                "daily_equity_sharpe": obs["recomputed_daily_sharpe"],
                "observation_curve": obs,
                "same_original_contract_periods": [
                    {
                        "account_id": q["id"],
                        "start": q["start"],
                        "end": q["end"],
                        "net_return": q["net_return"],
                        "win_rate": q.get("win_rate"),
                        "closed_trades": q.get("closed_trades"),
                        "saved_curve": q.get("saved_curve"),
                        "source_path": q["source_path"],
                    }
                    for q in all_rows
                    if all(
                        q.get(k) == r.get(k)
                        for k in [
                            "strategy",
                            "study",
                            "capital",
                            "fees_per_side",
                            "slippage_bps",
                            "pairlist_sha256",
                            "leverage_values",
                            "gross_budget",
                        ]
                    )
                ],
                "scope_note": note,
            }
        )
    lines += [
        "",
        "这些结果不应因未达到某个年化目标就被称为“没有收益”；也不能因某一列较好就取得部署资格。NFI＋Donchian的回撤更小、收益也更低，属于取舍。Sentinel的本地移植仍需原版来源和方法对齐，当前只保留已有记录，不继续自研。",
        "",
        "## 与已复现NFI X8的关系",
        "",
        "NFI X8 v18.0.146原版2025年1月3x、10000 USDT、223个实际加载标的，5bps净收益81.89%；8bps费用代理下75.74%。这是一个月结果。当前库中没有同时匹配该周期、本金、223币池、杠杆和执行合同的非NFI挑战账户，因此没有证据宣布其他策略优于它，也没有证据宣布它是长期全局最优。",
        "",
        "原版已平仓回撤1.39%、原生已平仓夏普36.77，与5分钟全持仓参考回撤13.61%、日净值夏普7.74分列；旧策略小时mark回撤也不能冒充5分钟或可成交盘中最差回撤。",
        "",
        "## 覆盖、重复和未取得证据",
        "",
        "本次复核当前三个项目的本地研究产物，并解析CZ转交TradingAgents的8个目录别名。研究界面766条记录不能直接当策略数：来源中存在摘要、审计、诊断、重复版本和未运行任务。",
        "",
        f"- 归一化并按结果身份合并后，共{v['accounts']}个账户情景：{v['account_kinds']['native_account']}个原生账户、{v['account_kinds']['native_wallet_sum']}个原生钱包组合、{v['account_kinds']['custom_account_with_curve']}个自定义引擎账户和{v['account_kinds']['equity_account']}个股票账户；原生策略类80种。情景不是独立策略或独立样本。",
        f"- {v['duplicate_account_aliases']}个重复结果路径已归并，{v['summary_nodes_retained']}个摘要节点保留出处，不重复当新业绩。共同条件可比较组为{v['strict_cohorts_with_multiple_accounts']}组。",
        "- 221个非空原生账户及12个钱包组合通过独立Decimal订单现金复算；另3个原生账户无成交，不作为盈利证据。账户收益分母、期末权益与保存曲线的结算值核对未发现不一致。自定义引擎只复算保存净值，不冒充原生成交审计。",
        "- 另查84条旧曲线，49条有相邻明确本金合同；35条缺本金合同，不把第一条扣费后权益当初始本金。原始库存、模型资金费、费用代理、假设资金费和标签研究不相互替代。",
        "- 两份非账户JSON摘要不能解析、一份约69MB归因审计超出轻量读取上限，均列入原始清单。市场数据、源码快照、依赖与凭据文件不当研究结果扫入；未运行策略保持未知。",
        "- 395份带账户标记的JSON未直接归一为账户，包含计划、聚合摘要及已有原生ZIP的镜像；摘要节点仍保留原路径。对旧曲线缺本金/费用/资金费合同、仅有标签或组件结果的记录，不声称已完成完整账户验证。",
        "- 港美股176个账户按USD/HKD与252交易日另列，买入持有控制的多年累计收益不与币圈单月混排。本次不修改股票研发文件或部署。CZ生产源码、跟单标签或组件回放缺少完整同口径账户时保持未知，未读取远程实盘账单。",
        "",
        "完整归一化账户：[本地accounts.json]("
        + str((root / "accounts.json").resolve())
        + ")；全部可比较组：[comparison.json]("
        + str((root / "comparison.json").resolve())
        + ")。关键同条件结果见[matched_six_coin_comparison.json](matched_six_coin_comparison.json)，新增原样回放见[unchanged_native_replays.json](unchanged_native_replays.json)，覆盖与算术检查见[summary.json](summary.json)。",
        "",
        "原件哈希及最终验算见[verification.json](verification.json)；固定输入、重算命令和指标边界见[reproduction.md](reproduction.md)。24个结果没有单列期末权益，最终验算改从已保存净值末值核对本金收益恒等式，全部一致；没有补写原结果。",
        "",
        "## 决定",
        "",
        "继续以GitHub原版复现为主线，保留完整NFI X8基线。旧NFI40、NFI＋Donchian及已有carry21作为已知历史对照；不把TrendControl和CompressionRetest的训练高收益作为替代依据，不通过换年份或调参数救回失败。下一步仍应取得公开高收益候选的完整原版合同，或者在明确的共同合同下检验现有挑战者；没有开启新因子、模型、参数优化或实盘。",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))
    (out / "positive_existing_controls.json").write_text(
        json.dumps(positive, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "report": str(out / "README.md"),
                "accounts": v["accounts"],
                "matched_periods": len(periods),
                "new_replay_jobs": len(native_new),
            }
        )
    )


if __name__ == "__main__":
    main()
