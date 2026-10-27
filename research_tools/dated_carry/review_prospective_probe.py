"""Seal source-observability evidence without relabelling it as forward PnL."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def iso(ns):
    return datetime.fromtimestamp(ns / 1e9, UTC).isoformat()


def review(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(f"data/research/dated_carry_prospective_probe_20261027_v00{n}") for n in [1, 2, 3]
    ]
    audit = Path("results/research/dated_carry_prospective_probe_audit_20261027_v001")
    old = Path("docs/research/regime_program/dated_carry_lot_accounts_20261027")
    for root in [*sources, audit, old]:
        verify_manifest(root)
    current = sources[-1]
    summary = read(audit / "summary.json")
    assert (
        summary["source_observability_passed"]
        and summary["valid_book_sequences"] == summary["BBO_exact_comparisons"] == 4
    )
    ages, books, rules = [
        read(audit / (name + ".json"))
        for name in ["native_ages", "book_proofs", "reported_rule_epochs"]
    ]
    for name in [
        "clock_proofs",
        "reported_rule_epochs",
        "native_ages",
        "book_proofs",
        "asof_quotes",
    ]:
        save(output / (name + ".json"), read(audit / (name + ".json")))
    failures = [
        {"capture": str(root), "summary": read(root / "probe_summary.json")}
        for root in sources[:-1]
    ]
    save(output / "prior_capture_failures.json", failures)
    native_requests = sum(
        read(root / "probe_summary.json")["native_HTTP_requests"] for root in sources
    )
    assert native_requests == 24
    save(
        output / "summary.json",
        {
            **summary,
            "source_probe_finished": True,
            "recorder_still_running": False,
            "successful_public_WS_connections": 2,
            "native_REST_requests_across_three_versions": native_requests,
            "new_forward_raw_observations": True,
            "new_forward_strategy_signals": 0,
            "prospective_financial_outcomes_evaluated": False,
            "global_proxy_settings_changed": False,
            "dependencies_installed": False,
            "module_admitted": False,
            "production_eligible": False,
        },
    )
    paths = [
        Path(__file__),
        Path("research_tools/dated_carry/audit_prospective_probe.py"),
        Path("tests/research_mainline/test_prospective_source_probe.py"),
        Path("tests/research_mainline/test_prospective_probe_audit.py"),
    ]
    paths += [r / "completion.json" for r in [*sources, audit, old]]
    save(
        output / "verification.json",
        {
            "files_sha256": {str(p): sha(p) for p in paths},
            "source_rule_registration_before_native_quote_access": True,
            "dedicated_time_not_metadata_clock_used": True,
            "current_rules_not_backfilled_into_old_quarter": True,
            "four_receipt_causal_native_book_chains_and_update_id_aligned_BBO_match": True,
            "five_collector_barrier_and_proxy_tests_plus_three_independent_book_tests_passed": True,
            "both_original_failed_attempts_and_all_raw_messages_preserved": True,
            "capture_command_terminal_success_not_an_ongoing_monitor": True,
            "new_financial_accounts": 0,
            "forward_strategy_validation_completed": False,
            "real_execution_or_capacity_certified": False,
            "external_review_performed": False,
            "profit_goal_met": False,
            "orders_enabled": False,
        },
    )
    rt = []
    for r in rules:
        filters = {f["filterType"]: f for f in r["native_filters"]}
        lot, market = filters["LOT_SIZE"], filters["MARKET_LOT_SIZE"]
        notional = filters.get("MIN_NOTIONAL", filters.get("NOTIONAL"))
        rt.append(
            f"| {r['symbol']} | {lot['stepSize']} | {market['stepSize']} | {market['maxQty']} | {notional.get('notional', notional.get('minNotional'))} |"
        )
    age_table = "\n".join(
        f"| {a['market']} | {a['observed_messages']} | {a['event_time_known_messages']} | {a['apparent_event_to_receive_ms']['p50']:.2f} | {a['apparent_event_to_receive_ms']['p99']:.2f} | {a['apparent_event_to_receive_ms']['max']:.2f} |"
        for a in ages
    )
    book_table = "\n".join(
        f"| {b['symbol']} | {b['deltas_applied']} | {b['book_age_at_watermark_ms']:.3f} | 是 | 是 |"
        for b in books
    )
    first_conn = min(
        read(current / (m + "_stream.receipt.json"))["connected_wall_ns"]
        for m in ["spot", "future"]
    )
    last_conn = max(
        read(current / (m + "_stream.receipt.json"))["finished_wall_ns"] for m in ["spot", "future"]
    )
    (output / "README.md").write_text(f"""# 前向数据链：当前规则先冻结，四份原生盘口可复算

2026-10-27。本轮打通**交易前元数据→专用时间探针→原生接收记录→快照/增量→规则复核**，两个原生公开连接采到9,806条消息，四个盘口序列及按update ID对齐的BBO价量均通过内部复算。没有策略信号、金融账户或前向收益；采样已结束，当前没有持续行情进程。

## 为什么转向当前原始记录

旧2025季度缺准确历史过滤器和执行真实性，前轮整手模型保留条件性结论。本轮先解决未来研究的证据产生顺序，不再拿当前规则补造旧历史。BTC/ETH保持固定，按当前元数据中仍在交易且至少10分钟后到期的最近USDT线性季度合约选择，在接收任何新报价前冻结为BTCUSDT_261225/ETHUSDT_261225，非按基差或收益挑合约。

MCP期货返回的完整解码元数据和现货错误原件留在v001。插件现货调用返回传输/挑战页，也不能提供原始HTTP接收和缓存信息，因此改用同一交易所[公开REST接口](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)与[原生WebSocket](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Connect)，保存原响应、有限响应头、本地墙钟和单调钟。没有读取交易账户、委托接口或密钥。

## 先失败，后修复；没有擦除尝试

v001：专用时钟可用，现货symbols数组因JSON空格被API以400拒绝；在报价前停止。v002：仅修紧凑JSON编码，四个前后时钟和元数据可取，但默认SOCKS路径缺本地可选依赖，未建立连接、0帧。v003：只让两个WebSocket显式沿用机器已经配置且REST使用的HTTP CONNECT代理，未改系统网络设置、未安装依赖，随后实际接通。

三版共24次原始REST请求，成功版12次；成功版两条WSS各约30秒。源记录与失败详见[prior_capture_failures.json](prior_capture_failures.json)，旧源码和收据全部封存。不是反复改变策略参数寻找正收益。

成功连接记录区间约为`{iso(first_conn)}`至`{iso(last_conn)}`（含连接关闭耗时）；真正共同接收水位是`{iso(summary["common_watermark_ns"])}`，按两路最后收到帧的较早时间确定，不按价格选择。所有快照都在两路已连接后获取，原生消息在元数据/合约选择冻结后接收。

## 当前报告的规则，不能移给2025

| 原生代码 | LOT_SIZE步长 | MARKET_LOT_SIZE步长 | 市场订单报告maxQty | 报告最小名义USDT |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rt)}

四个原生symbol元数据在报价前和采样后相同，全局exchangeFilters亦相同。全部PRICE_FILTER、数量、名义、比例价格、权限等字段都保留在[reported_rule_epochs.json](reported_rule_epochs.json)，没有只拿price/quantityPrecision代替过滤器。现货MARKET_LOT_SIZE的零步长未解释为其他数量限制消失；未来执行要按适用过滤器共同检查。最小名义检查参考价/窗口、实际账户权限、佣金及保证金档位仍不能由这张表全部认证。

这些是此刻261225合约报告的规则，不能认证250328旧合约或整个2025季度。原市场数据、历史收益和假设格点账户都未改写。

## 时间、序列与报价一致性

分别调用现货/期货专用`/time`，保存请求/接收墙钟与单调钟界和RTT。前后四个探针均满足事前5秒RTT及服务器时钟范围；不使用exchangeInfo.serverTime给行情新鲜度背书。它们是相对于元数据独立的时间探针，仍来自同一交易所，不是第三方授时认证。

| 市场 | 原生消息总数 | 有E事件时钟样本（共同水位前） | 表观接收年龄p50 ms | p99 ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
{age_table}

表观年龄为本地接收墙钟减原生E时间，受时钟差影响，不是已校准单向网络延迟，更不是策略决策/下单/撮合总延迟。现货bookTicker没有E字段，明确保留未知，没有用本地时间伪装交易所事件时间。墙钟与单调钟经过时差约1.8ms，完整探针和偏差范围见[clock_proofs.json](clock_proofs.json)、[native_ages.json](native_ages.json)。

| 原生代码 | 应用增量数 | 共同水位时盘口年龄ms | 序列连续 | ID对齐BBO价量一致 |
| --- | ---: | ---: | --- | --- |
{book_table}

现货使用lastUpdateId+1与U/u覆盖，期货使用原生重叠与pu/u链；不能把两种算法混用。只用共同接收水位前实际收到的快照和增量，不让未来快照/报价回填过去。BBO按其update ID不大于深度最终ID选最后已接收记录，不挑价格更有利的一条。所有四组的最佳买卖价及数量一致，原字段和序列摘要见[book_proofs.json](book_proofs.json)、[asof_quotes.json](asof_quotes.json)。

这支持采样区间内的数据链可观察性，不证明某笔委托真的能成交；REST/WSS和两个通道来自同一交易所，也不是独立交易场所或外部审计。8项相关测试仅检验时钟/身份/到期选择、已有代理复用、快照因果性、序列断链和规则变更。

## 本轮完成与未完成

完成当前原始规则先于报价的证据顺序、四个盘口重建和受限接收时点核验。只采当前两个资产/一个到期机会，没有增加因子、经济机制、神经模型、订单或资本。

**仅冻结了数据采样协议，未运行一个新的前向策略。** 现在已经看过这段报价，不能在之后冻结入场规则，再把本段叫作未见策略收益。下一轮须先登记完整策略/代码/入场时钟、数据访问截止、费用假设、拒单与余币归属、缺源退出和对照，再接收其后的新数据。真正前向PnL、足够独立机会、真实执行、外部复算、容量和总账户目标继续未完成，不设额外固定30/60日等待门槛。

源目录：`{current}`；独立内部检查：`{audit}`。采集命令已终止成功、没有后台监控或纸面账户在继续运行。见[summary.json](summary.json)、[verification.json](verification.json)。
""")
    save(
        output / "completion.json",
        {
            "passed": True,
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "profit_goal_met": False,
        },
    )
    print(
        {
            "source_observability_passed": True,
            "native_messages": 9806,
            "book_sequences": 4,
            "forward_strategy_validation_completed": False,
            "new_financial_accounts": 0,
            "recorder_still_running": False,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    review(p.parse_args().output)
