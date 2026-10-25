"""大盘标的市场结构模块：提供 normalization 因子逻辑。"""

from __future__ import annotations

import math


def finite(value: object, default: float = 0.0) -> float:
    """把输入转换为有限浮点数，非法值回退到默认值。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def clip01(value: float) -> float:
    """把数值限制在闭区间 [0, 1]。"""

    return min(1.0, max(0.0, float(value)))


def unit_scale(value: float, lower: float, upper: float) -> float:
    """将数值在给定下上界之间线性归一化到 [0, 1]。"""

    if upper <= lower:
        raise ValueError("large-cap factor scale upper bound must exceed lower bound")
    return clip01((float(value) - float(lower)) / (float(upper) - float(lower)))


def signed_scale(value: float, scale: float) -> float:
    """按方向调整数值符号后进行区间归一化。"""

    denominator = max(1e-9, abs(float(scale)))
    return min(1.0, max(-1.0, float(value) / denominator))


def action_direction(action: str) -> float:
    """从动作名称解析多空方向，无法识别时返回中性方向。"""

    normalized = str(action).upper()
    if normalized == "OPEN_LONG":
        return 1.0
    if normalized == "OPEN_SHORT":
        return -1.0
    raise ValueError(f"unsupported large-cap entry action: {action}")
