"""在未发布的独立对象内重放全部输入，失败对象不可用作运行账户。"""

from __future__ import annotations

from decimal import localcontext

from tradingagents.dataflows.research.live_capture import digest
from tradingagents.research.quant.backtesting.paper_account import PaperAccount


def restore_in_isolation(frames, checkpoint):
    """省去逐帧事务副本；完整重放并精确校验之前不发布任何状态。"""
    protocol = checkpoint["protocol"]
    account = PaperAccount(
        protocol["symbols"],
        protocol["slippage_bps_each_side"],
        period_end_ms=protocol["period_end_ms"],
    )
    with localcontext() as context:
        context.prec = 40
        for frame in frames:
            account._advance(frame)
            account.event_head = digest(
                {
                    "sequence": account.event_count,
                    "previous_sha256": account.event_head,
                    "frame_sha256": digest(frame),
                }
            )
            account.event_count += 1
    if account.checkpoint() != checkpoint:
        raise ValueError("隔离重放不能从完整输入精确重建检查点")
    return account
