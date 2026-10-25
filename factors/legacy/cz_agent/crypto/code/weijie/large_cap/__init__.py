"""大盘标的市场结构模块：提供   init   因子逻辑。"""

from factors.crypto.code.weijie.large_cap.legacy_gate import (
    LargeCapLegacyGatePolicy,
    gate_large_cap_legacy_actions,
)
from factors.crypto.code.weijie.large_cap.policy import LargeCapRegimePolicy
from factors.crypto.code.weijie.large_cap.signals import (
    build_large_cap_regime_close_signals,
    build_large_cap_regime_open_signals,
)

__all__ = [
    "LargeCapLegacyGatePolicy",
    "LargeCapRegimePolicy",
    "build_large_cap_regime_close_signals",
    "build_large_cap_regime_open_signals",
    "gate_large_cap_legacy_actions",
]
