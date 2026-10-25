"""Publish schedule scenario evidence separately from history and profit certification."""

import argparse
from pathlib import Path

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def publish(run, output):
    output.mkdir(parents=True, exist_ok=False)
    verify_manifest(run)
    result = read(run / "summary.json")
    audit_path = run.with_name(run.name + "_independent_audit.json")
    checked = read(audit_path)
    if not checked["passed"] or checked["source_completion_sha256"] != sha(run / "completion.json"):
        raise ValueError("Independent schedule audit identity differs")
    for path, digest in read(run / "registration.json")["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Frozen scenario implementation changed")
    for path, digest in read(run / "source_bindings.json").items():
        if sha(Path(path)) != digest:
            raise ValueError("Original current-body capture changed")
    data = {
        **result,
        "schema": "historical-contract-timeline-review.v1",
        "independent_schedule_audit_pending": False,
        "scenario_audit_passed": True,
        "historical_certification_passed": False,
        "scenario_completion_sha256": sha(run / "completion.json"),
        "metrics": dict.fromkeys(
            ["net_return", "win_rate", "max_drawdown", "sharpe", "capacity_usdt"]
        ),
        "next_action": "补10个未覆盖名称的USD-M公告、BLZ/AERGO重新上线及OMG原始公告版本；固定合约episode与当前正文版本假设。扩展池五分钟成交、真实资金费和终端现金仍待核验，不能直接算新增收益。",
    }
    save(output / "summary.json", data)
    save(output / "independent_audit.json", checked)
    save(output / "scenario_completion.json", read(run / "completion.json"))
    save(
        output / "verification.json",
        {
            "scenario_audit_passed": True,
            "historical_certification_passed": False,
            "scenario_completion_sha256": sha(run / "completion.json"),
            "independent_audit_sha256": sha(audit_path),
            "publisher_sha256": sha(Path(__file__)),
            "new_financial_accounts": 0,
            "raw_binding_sha256": sha(run / "source_bindings.json"),
            "metrics_computed": False,
        },
    )
    save(output / "event_cards.json", read(Path(__file__).with_name("event_cards.json")))
    removed = {}
    for r in result["removed_slots"]:
        removed[r["symbol"]] = removed.get(r["symbol"], 0) + 1
    rows = ["| 合约 | 保守情景移除原槽位 | 原因 |", "|---|---:|---|"]
    for symbol, count in sorted(removed.items()):
        reason = (
            "重新上线起点缺失，不能与早期终止合约直接拼接"
            if symbol in result["multi_episode_symbols"]
            else "原预览含已知终止边界后的候选"
        )
        rows.append(f"| {symbol} | {count} | {reason} |")
    text = (
        """# 历史合约时间线与重新上线缺口

本轮取得官方当前CMS快照：1,056篇目录条目、180篇正文、合计202次只读请求，未超过预登记230次总预算。两个目录在抓取时均遍历到末页，但这不证明2019年以来的公告历史齐全；还有76篇标题候选未抓取。38个优先名称得到42张经产品/时间子句核对的卡片，覆盖48个优先名称中的38个，仍有10个未取得可采纳USD-M终止条款。

**场景计算与独立复算完成，历史认证未通过，新增收益账户0。** 净收益、胜率、回撤、夏普、容量和多策略总账户实际值全部未知；收益目标保持未完成。

## 这次修正了什么

时间线区别公告首次宣称发布时间、当前捕获正文、停止开仓、自动结算，以及后续延期与其他合约episode。当前正文可能含后来补充，不能把其所有内容默认当作最初公开时已知。以宣称发布时间加5分钟使用当前子句只是下面场景的研究假设，不是历史point-in-time认证。未取得实际终端结算价和费用。

OMG的原计划是2024-12-16终止，随后两份公告依次改为2024-12-30和2025-01-31。同一episode的期限只在对应延期公告宣称可用之后更新；较新公告没有重新确认开仓截止时，保守研究政策保留较早已知限制，不能凭新终止时间自动补一个提前30分钟的截止。原正文还含后来更新注释，原始版本仍缺。

BLZ存在2022-08-25及2024-12-23两份不同终止记录；AERGO存在2025-03-27及2026-07-24两份终止记录。后一个终止日期不能抹除前一个，也不能证明中间持续交易。此轮没有找到并认证对应重新上线起点，因此保守情景在早期终止之后暂停其候选资格；这不等于证明全部后续归档都是无效行情。

FTT、SRM、LUNA及早期BLZ条款缺少可采纳的新仓截止子句时保留未知，不自动推为终止前30分钟。FTT同公告的FTTUSDT、FTTBUSD时间不同，USD-M USDT边界不能借用BUSD、现货、杠杆或COIN-M的日期。

## 对原366个周频预览的影响

保留原860名归档、原始价格、成交额、异常标记和366次预览不变。仅在上述当前子句时间线假设下，重新用之前28个完整有效日、180个有效观测及5千万USDT平均成交额筛选，再排序前30。独立Pandas自然日日历滚动计算同时重现原始无约束预览和新情景预览，不导入时间线执行函数或NumPy重算实现。

共25个决策日期、26个原候选槽位变化；这26个不能全部称为已证实的错误交易，其中20个属于重新上线分段待认证的保守暂停。

"""
        + "\n".join(rows)
        + """

没有新增初始268名称之外的成员。这里没有运行替换成员的收益，不允许把候选差异换算成策略改善。

## 证据边界

7项行为测试覆盖未来公告不得回改过去、OMG两次延期、BLZ/AERGO独立episode、缺失截止不填、终止日不完整日线、非空终端现金缺失阻断及跨episode错误替换。3个实际日期删除未来公告后重算限制结果相同；366次原始及新预览独立复算通过。

仍未覆盖的优先名称：ALPACAUSDT、BTTUSDT、DEGOUSDT、FRONTUSDT、HIGHUSDT、ICXUSDT、LRCUSDT、STORJUSDT、TOMOUSDT、VANRYUSDT。它们不能因缺公告而被认证为全历史可交易，也不能只按当前状态从历史剔除。其他220个当前TRADING名称同样没有因此得到全历史认证。

原公告正文版本、重新上线边界、实际停止接单、终端结算现金及执行数据仍需补齐。数据层助手尚未接入扩展池的Freqtrade逐笔账户；未修改冻结账户或已发布收益，不发单、不部署、不推送。

主要官方条款：[OMG首次延期](https://www.binance.com/en/support/announcement/detail/eedac7aff5b849d58efda6c4b3cec01b)、[OMG再次延期](https://www.binance.com/en/support/announcement/detail/1e43c128806544018d3cc537a8add3f6)、[BLZ早期终止](https://www.binance.com/en/support/announcement/detail/b312715afbff4d7493b77089967303b1)、[AERGO/LINA等产品分列公告](https://www.binance.com/en/support/announcement/detail/db7ad1c7aa6248cda735102cdcdc4b8d)。全部42张卡片的产品、时间、原始快照路径和SHA256见event_cards.json，逐决策差异与未认证名单见summary.json。

复算入口：research_tools/contract_lifecycles/{collect,audit,independent_audit,publish}.py。每个输出使用新目录，旧版本不覆盖。audit.py保留其原执行字节，C401集合生成器风格提示不影响数值，后续版本再整理；独立验证器未复用该实现。
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
            "cards": result["card_count"],
            "covered": result["covered_symbols"],
            "changed_decisions": result["changed_decisions"],
            "financial_accounts": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    publish(a.run, a.output)
