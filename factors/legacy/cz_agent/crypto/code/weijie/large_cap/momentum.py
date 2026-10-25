"""大盘标的市场结构模块：提供 momentum 因子逻辑。"""

from __future__ import annotations

import pandas as pd

from factors.crypto.code.weijie.large_cap.normalization import (
    action_direction,
    finite,
    signed_scale,
)
from factors.crypto.code.weijie.large_cap.types import DirectionalMomentum


def score_directional_momentum(row: pd.Series, *, action: str) -> DirectionalMomentum:
    """score_directional_momentum：计算可解释的市场结构因子。"""

    direction = action_direction(action)
    short = signed_scale(finite(row.get("ret20")) * direction, 3.0)
    mid = signed_scale(finite(row.get("ret60")) * direction, 5.0)
    long_return = 0.40 * finite(row.get("ret144")) + 0.60 * finite(row.get("ret288"))
    long = signed_scale(long_return * direction, 8.0)
    return DirectionalMomentum(
        short=short,
        mid=mid,
        long=long,
        impulse=max(-1.0, min(1.0, 0.55 * short + 0.30 * mid + 0.15 * long)),
    )
