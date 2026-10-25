"""Past-only gross total-return level from nominal closes and ex-date dividends.

Market adapters decide calendar, availability, tax and execution. Split-containing
inputs must first be reconstructed by the market's data admission layer.
"""

import numpy as np


def gross_return_level(close, dividend):
    if (
        close.empty
        or not close.index.equals(dividend.index)
        or not np.isfinite(close).all()
        or not np.isfinite(dividend).all()
        or (close <= 0).any()
        or (dividend < 0).any()
    ):
        raise ValueError("Explicit complete nominal closes and ex-date events required")
    returns = (close + dividend) / close.shift(1)
    returns.iloc[0] = 1.0
    return returns.cumprod()
