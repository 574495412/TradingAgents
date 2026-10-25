"""Calendar-neutral primitives; windows count observed completed observations.

No implicit annualization, resampling, forward fill or market execution rules.
"""

import numpy as np


def momentum_inputs(close):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False, min_periods=2).mean()
    return {
        "fast": close.rolling(20).mean(),
        "slow": close.rolling(60).mean(),
        "long": close.rolling(200).mean(),
        "rsi2": 100 * up / (up + down).replace(0, np.nan),
        "vol20": close.pct_change(fill_method=None).rolling(20).std(ddof=1),
    }
