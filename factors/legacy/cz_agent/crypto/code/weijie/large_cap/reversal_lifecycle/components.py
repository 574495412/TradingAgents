"""反转生命周期共享的方向化市场分量和归一化工具。"""

from __future__ import annotations

from math import log1p
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

_DIRECTION_EPSILON = 1e-12
_PRICE_EPSILON = 1e-9


def directional_market_components(
    bar: pd.Series,
    direction: float,
) -> dict[str, float]:
    """把收益、位置、RSI 和广度转换为相对于目标方向的分量。"""

    normalized_direction = _normalized_direction(direction)
    long = normalized_direction > 0.0
    components = {
        "tiny_return": normalized_direction * float(bar["ret3"]),
        "recovery_return": normalized_direction * float(bar["ret12"]),
        "mid_momentum": normalized_direction * float(bar["ret60"]),
        "long_momentum": normalized_direction * float(bar["tf1h_ret12"]),
        "adverse_mid_momentum": -normalized_direction * float(bar["ret60"]),
        "adverse_long_momentum": (
            -normalized_direction * float(bar["tf1h_ret12"])
        ),
        "location": directional_location(
            float(bar["donchian60_pos"]),
            normalized_direction,
        ),
        "rsi": directional_rsi(float(bar["rsi14"]), normalized_direction),
        "favorable_breadth20": directional_breadth(
            bar,
            direction=normalized_direction,
            horizon=20,
            favorable=True,
        ),
        "adverse_breadth20": directional_breadth(
            bar,
            direction=normalized_direction,
            horizon=20,
            favorable=False,
        ),
    }
    # 优先使用方向归一化后的短周期与相对动量。
    # 缺少可选列时保留基础组件，避免用未来数据回填。
    if "ret20" in bar:
        components["short_momentum"] = (
            normalized_direction * float(bar["ret20"])
        )
    if "rel_ret60" in bar:
        components["relative_momentum"] = (
            normalized_direction * float(bar["rel_ret60"])
        )
    breadth3_columns = {"market_up_breadth3", "market_down_breadth3"}
    if breadth3_columns.issubset(bar.index):
        components["favorable_breadth3"] = directional_breadth(
            bar,
            direction=normalized_direction,
            horizon=3,
            favorable=True,
        )
        components["adverse_breadth3"] = directional_breadth(
            bar,
            direction=normalized_direction,
            horizon=3,
            favorable=False,
        )
    candle_columns = {"open", "high", "low", "close"}
    if candle_columns.issubset(bar.index):
        opening = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        candle_location = (close - low) / max(
            high - low,
            _PRICE_EPSILON,
        )
        components["candle_return"] = (
            normalized_direction
            * (close / max(opening, _PRICE_EPSILON) - 1.0)
            * 100.0
        )
        components["candle_close_location"] = (
            candle_location if long else 1.0 - candle_location
        )
    return components


def directional_location(location: float, direction: float) -> float:
    """把通道位置转换为目标方向的有利位置。"""

    normalized_direction = _normalized_direction(direction)
    value = float(location)
    return value if normalized_direction > 0.0 else 1.0 - value


def directional_rsi(rsi: float, direction: float) -> float:
    """把 RSI 转换为目标方向的一致强弱刻度。"""

    normalized_direction = _normalized_direction(direction)
    value = float(rsi)
    return value if normalized_direction > 0.0 else 100.0 - value


def directional_breadth(
    bar: pd.Series,
    *,
    direction: float,
    horizon: int,
    favorable: bool,
) -> float:
    """返回目标方向的有利广度与短期不利广度。"""

    normalized_direction = _normalized_direction(direction)
    if horizon not in {3, 20}:
        message = "directional breadth horizon must be 3 or 20"
        raise ValueError(message)
    long = normalized_direction > 0.0
    use_up = long if favorable else not long
    side = "up" if use_up else "down"
    return float(bar[f"market_{side}_breadth{horizon}"])


def linear_ramp(value: float, lower: float, upper: float) -> float:
    """在上下界之间计算限制于 [0, 1] 的线性斜坡。"""

    if upper <= lower:
        message = "ramp upper bound must exceed lower bound"
        raise ValueError(message)
    return max(0.0, min(1.0, (float(value) - lower) / (upper - lower)))


def logarithmic_ramp(value: float, lower: float, upper: float) -> float:
    """对正值使用对数尺度计算限制于 [0, 1] 的斜坡。"""

    if lower < 0.0 or upper < 0.0:
        message = "logarithmic ramp bounds must be non-negative"
        raise ValueError(message)
    return linear_ramp(
        log1p(max(0.0, float(value))),
        log1p(lower),
        log1p(upper),
    )


def _normalized_direction(direction: Any) -> float:
    """将任意方向输入规范为 +1 或 -1。"""

    value = float(direction)
    if abs(abs(value) - 1.0) > _DIRECTION_EPSILON:
        message = "direction must be +1 or -1"
        raise ValueError(message)
    return 1.0 if value > 0.0 else -1.0


__all__ = [
    "directional_breadth",
    "directional_location",
    "directional_market_components",
    "directional_rsi",
    "linear_ramp",
    "logarithmic_ramp",
]
