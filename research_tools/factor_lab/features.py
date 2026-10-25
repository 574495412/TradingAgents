"""Compatibility adapter: all factor implementations live in TradingAgents/factors."""

from factors.common.panel import build, calculate_market, event_features

__all__ = ["build", "calculate_market", "event_features"]
