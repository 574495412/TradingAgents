"""Strict inventory identity guard for future segment-aware account integration.

The frozen preview evaluator is unchanged. New execution callers must validate
registered episode identity before invoking its terminal-cash prerequisite.
"""

from research_tools.contract_lifecycles.segments import require_episode_terminal_cash
from research_tools.contract_lifecycles.timeline import UnverifiedContractHistory


def require_registered_episode_cash(registry, symbol, episode, clock, quantity):
    if quantity != 0 and not any(
        c["symbol"] == symbol and c["episode"] == episode for c in registry["cards"]
    ):
        raise UnverifiedContractHistory(
            "Position contract episode is not registered for this symbol"
        )
    require_episode_terminal_cash(registry, symbol, episode, clock, quantity)
