"""Seven-day MAX versus ordinary reversal mathematics; source-bound research only.

Grobys/Junttila2021, DOI10.1016/j.intfin.2021.101289, sections3/4.2.
Seven completed calendar daily log returns; stocks need a separate session clock.
Averaged log factor spreads are not returns on funded cash accounts.
"""

import math

import numpy as np
import pandas as pd


def weekly_features(close):
    if not isinstance(close.index, pd.DatetimeIndex) or close.index.has_duplicates:
        raise ValueError("Unique dated prices required")
    if not close.index.equals(pd.date_range(close.index[0], close.index[-1], freq="D")):
        raise ValueError("Explicit daily calendar required; never compress missing dates")
    values = close.where(np.isfinite(close) & close.gt(0))
    daily = np.log(values / values.shift(1))
    return pd.DataFrame(
        {
            "max7": daily.rolling(7, min_periods=7).max(),
            "reversal7": np.log(values / values.shift(7)).where(daily.rolling(7).count().eq(7)),
        }
    ).shift(1)


def reversal_book(rows, candidate):
    key = {"max7_lottery_reversal": "max7", "ordinary7_reversal_control": "reversal7"}.get(
        candidate
    )
    if key is None:
        raise ValueError("Unknown frozen weekly reversal policy")
    if len(rows) < 10:
        return {}, {"status": "insufficient_native_members"}
    if not all(math.isfinite(float(row[key])) for row in rows.values()):
        raise ValueError("Incomplete source-calendar signal; no automatic member deletion")
    ordered = sorted(rows, key=lambda s: (float(rows[s][key]), s))
    n = len(ordered)
    long = [s for rank, s in enumerate(ordered) if 5 * rank // n == 0]
    short = [s for rank, s in enumerate(ordered) if 5 * rank // n == 4]
    if set(long) & set(short) or len(long) + len(short) > 12:
        raise ValueError("Invalid local quintile position budget")
    book = {s: 0.5 / len(long) for s in long}
    book.update({s: -0.5 / len(short) for s in short})
    return book, {
        "status": "math_only",
        "long": long,
        "short": short,
        "native_members": n,
        "score": key,
        "unit_gross": 1.0,
        "source_high_minus_low_direction_inverted_explicitly": True,
        "author_exact_replication": False,
    }
