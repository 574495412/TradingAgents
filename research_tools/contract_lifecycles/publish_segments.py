"""Publish partial segment qualification without marking source conflict or profit passed."""

import argparse
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def publish(run, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(run)
    data = read(run / "summary.json")
    check_path = run.with_name(run.name + "_independent_audit.json")
    checked = read(check_path)
    trade_path = Path("results/research/contract_segment_reported_trades_audit_20261026_v001.json")
    trade = read(trade_path)
    extra = Path("data/research/contract_segment_followup_20261026_v002")
    verify_manifest(extra)
    inputs = Path("results/research/contract_segment_inputs_20261026_v001")
    verify_manifest(inputs)
    for name, digest in read(inputs / "plan.json")["source_hashes"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Prepared segment source changed")
    for name, digest in read(run / "registration.json")["source_hashes"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Segment preview source changed")
    if (
        not checked["passed"]
        or not trade["passed"]
        or checked["source_completion_sha256"] != sha(run / "completion.json")
        or trade["source_completion_sha256"] != sha(extra / "completion.json")
    ):
        raise ValueError("Independent preview or reported trade audit differs")
    data = {
        **data,
        "schema": "historical-contract-segment-review.v1",
        "scenario_audit_passed": True,
        "historical_certification_passed": False,
        "independent_schedule_audit_pending": False,
        "scenario_completion_sha256": sha(run / "completion.json"),
        "trade_source_completion_sha256": sha(extra / "completion.json"),
        "reported_trade_audit": trade,
        "metrics": dict.fromkeys(
            ["net_return", "win_rate", "max_drawdown", "sharpe", "capacity_usdt"]
        ),
        "next_action": "核查BLZ终止公告与已报告成交之间的矛盾、其余重新上线与公告原始版本；按合约episode补真实五分钟成交与资金费。当前正文情景和52条边界覆盖不等于历史认证或盈利。",
    }
    save(output / "summary.json", data)
    save(output / "independent_audit.json", checked)
    save(output / "scenario_completion.json", read(run / "completion.json"))
    save(output / "reported_trade_audit.json", trade)
    save(output / "event_cards.json", {"cards": data["cards"]})
    save(output / "registry.json", read(inputs / "registry.json"))
    save(
        output / "verification.json",
        {
            "scenario_completion_sha256": sha(run / "completion.json"),
            "trade_source_completion_sha256": sha(extra / "completion.json"),
            "publisher_sha256": sha(Path(__file__)),
            "independent_segment_audit_sha256": sha(check_path),
            "reported_trade_audit_sha256": sha(trade_path),
            "scenario_audit_passed": True,
            "historical_certification_passed": False,
            "new_financial_accounts": 0,
        },
    )
    lines = ["| 原入选时刻UTC | 新合约完整有效日 | 满足180观测 |", "|---|---:|---|"]
    for p in data["launch_profiles"]:
        lines.append(f"| {p['decision']} | {p['valid_full_observations_in_episode']} | 否 |")
    text = (
        """# 合约分段、指代条款与成交源反证

新增12次只读源请求，补48个优先名称的终止条款覆盖至48/48，共52张终止/延期卡片；另取得1张AERGO明确上线条款。这里的完整仅指这48个名称已找到可采用的当前终止子句，**不代表全历史可交易状态、原始公告版本、最终现金或收益认证完成**。新增收益账户0；收益、胜率、回撤、夏普和容量未知，多策略总账户20%/30%目标未完成。

## 指代范围与产品日期

上一版漏了7个已经归档的条款：ALPACA、DEGO、HIGH、LRC、ICX、STORJ、VANRY的正文使用命名代币及其Futures段落中的“上述代币合约”指代，没有写完整USDT代码。此版逐项核对指代对象、Futures段落内的终止/停止新仓时间，并用现有合约类型快照作为产品类别旁证；类型旁证不是历史状态认证，也不把通用指代套用给没有对应合约的所有代币。

另外取得BTT、TOMO、FRONT的明确期货子句。采用的终止时间分别为2022-01-14 02:00、2023-11-14 09:00、2024-08-23 09:00 UTC。BTT未找到新仓截止时保留未知。ALPACA采用期货2025-04-30 09:00，而不是现货标题中的2025-05-02。同一通用公告提到SXP不意味着已在2025年终止的SXP合约重新上线，因此没有凭指代增加SXP的新episode。

## AERGO新合约只重置数据观察期

官方上线条款宣称AERGOUSDT于2025-04-16 11:00 UTC开始。公告宣称发表时间、真实上线时间和完整UTC日线起点分开处理：本情景中首个完整日为2025-04-17，旧合约价格、成交额和年龄不能跨阶段补足新合约资格。原4次入选时只积累了以下完整有效日：

"""
        + "\n".join(lines)
        + """

实际封存数据首次积累180个完整有效日的时点为2025-10-14 00:00 UTC；这只是年龄条件，不代表当时流动性、成本、容量或收益已通过。明确的新合约上线不会自动结清旧episode库存，也不会重置总账户高水位、停机或费用；未取得旧合约终端现金时非空库存继续阻断。数据适配尚未接入扩展池逐笔账户。

## BLZ来源矛盾继续保留

2022年公告正文宣称BLZ的USDT合约在2022-08-25 09:00 UTC终止；官方聚合成交归档则报告当日2,487,006条记录，其中1,141,466条位于该时点之后，次日另有824,913条。两日聚合成交ID逐条、跨日连续，官方SHA256与ZIP CRC通过。这是已报告成交与公告之间的反证，不直接证明客户成交、取消退市公告或重新上线时间。保守情景仍暂停BLZ，并将真实原因保留为来源/分段不确定；没有按收益择源。

FTM在2025-01-07指定聚合成交归档前缀未列出文件；该观察不能证明全市场无成交，也不能单独裁决其他K线/资金费记录是否是应付现金。三份精确前缀查询及独立审计均保存。

## 预览影响与独立检查

366次原始和新预览由独立Pandas日历滚动及独立上线/延期逻辑重现。新情景相对原预览共有27个日期、30个原槽位变化：BLZ16、AERGO4、ALPACA4、FTT3、MATIC1、HIFI1、BAKE1。AERGO的4个是新episode观察期不足；BLZ16仍是分段/来源不确定的保守暂停。其余边界也处在当前正文历史可用性假设下，不能把全部变化称作已验证错误交易。没有计算替换成员的收益，没有引入原268个名称之外的成员。

7项新增分段行为测试覆盖上线不能回填到公告时刻、开盘不足一天不计、180观测边界、旧库存不能迁移、未来终止公告不能改变过去、期货/现货日期区分、成交文件不自动等于重新上线；原13项时间线测试保留。独立聚合成交检查不复用采集解析器。

最初25MB单文件上限因官方列表显示首个BLZ ZIP为32,706,323字节而停止；没有下载超限文件。旧失败保留；预先记录新的100MB单文件/250MB总上限，采用分块CSV解析和原5条正文/首份列表，不改变金融参数。最终新增请求共12次，未超原跟进28次预算。

旧38名称场景与原冻结收益完整保留。历史原始正文、其他合约的上线/停牌分段、BLZ公告冲突、终端结算现金、扩展池真实资金费与五分钟执行仍待验证。未下单、部署或推送。

官方参照：[AERGO上线](https://www.binance.com/en/support/announcement/detail/82f730b7ef444a38b323ab7a2e56b757)、[ALPACA及相关代币产品分列](https://www.binance.com/en/support/announcement/detail/0274f9d47da1437990bc13eb17b0ec99)、[BTT迁移](https://www.binance.com/en/support/announcement/detail/2722e6da4f5141dd9b2fb07b5b1f3f75)、[TOMO更名](https://www.binance.com/en/support/announcement/detail/034eee2293964e26b915b1ec5c42972a)、[FRONT更名](https://www.binance.com/en/support/announcement/detail/3ab5488a00e04d4fb338c77ea28326a8)。registry.json、summary.json及verification.json保存全部路径、SHA、边界、未知值与独立收据。
"""
    )
    (output / "README.md").write_text(text)
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "scenario_audit_passed": True,
            "historical_certification_passed": False,
            "financial_accounts": 0,
        },
    )
    print(
        {
            "published": True,
            "end_cards": len(data["cards"]),
            "covered_names": data["covered_symbols"],
            "launch_cards": len(data["launch_cards"]),
            "changed_decisions": data["changed_decisions"],
            "new_financial_accounts": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    publish(a.run, a.output)
