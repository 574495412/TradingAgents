"""Faber's published 10-month trend rule, adapted to the existing ETF cash account.

Month-end dates come from the captured exchange calendar, never the last row of
a truncated price prefix. Gross dividend levels are causal, not today's adjusted
history. Execution is next actual open in the existing independent cash engine.
"""

import numpy as np
import pandas as pd

from factors.common.total_return import gross_return_level


def targets(frames, calendar, candidate):
    if candidate not in {"faber10m", "faber10m_vol12"}:
        raise ValueError("Unregistered external equity rule")
    index = next(iter(frames.values())).index
    if (
        not all(f.index.equals(index) for f in frames.values())
        or not index.isin(calendar.index).all()
    ):
        raise ValueError("Observed prices must share the captured exchange calendar")
    days = pd.Series(calendar.index, index=calendar.index)
    month_ends = days.groupby(calendar.index.to_period("M")).last()
    weights, vol = {}, {}
    for symbol, frame in frames.items():
        level = gross_return_level(frame.close, frame.dividend)
        monthly = level.loc[level.index.isin(month_ends)]
        average = monthly.rolling(10, min_periods=10).mean()
        events = pd.Series(np.nan, index=monthly.index)
        events.loc[average.isna() | (monthly < average)] = 0.0
        events.loc[monthly > average] = 1.0
        decision = events.ffill().reindex(index).ffill().fillna(0.0)
        weights[symbol] = decision * 0.8 / len(frames)
        vol[symbol] = level.pct_change(fill_method=None).rolling(20).std(ddof=1) * np.sqrt(252)
    result = pd.DataFrame(weights, index=index)
    if candidate == "faber10m_vol12":
        bound = (result * pd.DataFrame(vol, index=index)).sum(axis=1)
        result = result.mul((0.12 / bound.replace(0, np.nan)).clip(upper=1).fillna(0), axis=0)
    # State labels are available at the prior close; not retrospective bull/bear dates.
    states = result.sum(axis=1).map(lambda value: "trend_enabled" if value > 0 else "cash_domain")
    return result, states
