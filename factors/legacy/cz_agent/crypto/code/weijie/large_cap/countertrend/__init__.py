"""大盘标的反趋势因子、评分、信号和持仓活动的公共接口。"""

# ruff: noqa: PLE0605, RUF003

from __future__ import annotations

from importlib import import_module
from typing import Any

# 反趋势模块只导出稳定的公共接口。
# cluster、contextual 与 eligibility 等内部实现保持延迟加载，
# 避免导入包时产生不必要的循环依赖。
_EXPORTS = {
    "COUNTERTREND_MECHANISMS": (".scoring", "COUNTERTREND_MECHANISMS"),
    "CountertrendCampaignPolicy": (".campaign", "CountertrendCampaignPolicy"),
    "CountertrendFactorPolicy": (".policy", "CountertrendFactorPolicy"),
    "CountertrendLifecyclePolicy": (".policy", "CountertrendLifecyclePolicy"),
    "VolatilityNormalizedCountertrendPolicy": (
        ".volatility_normalized",
        "VolatilityNormalizedCountertrendPolicy",
    ),
    "VolatilityNormalizedCountertrendResult": (
        ".volatility_normalized",
        "VolatilityNormalizedCountertrendResult",
    ),
    "build_countertrend_candidates": (".signals", "build_countertrend_candidates"),
    "build_countertrend_scores": (".scoring", "build_countertrend_scores"),
    "evaluate_countertrend_campaigns": (
        ".campaign",
        "evaluate_countertrend_campaigns",
    ),
    "evaluate_scored_countertrend_campaigns": (
        ".campaign",
        "evaluate_scored_countertrend_campaigns",
    ),
    "evaluate_volatility_normalized_countertrend": (
        ".volatility_normalized",
        "evaluate_volatility_normalized_countertrend",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
