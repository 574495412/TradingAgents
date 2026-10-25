"""Leader-imitation no-chase guard factor."""

from __future__ import annotations

import pandas as pd

from factors.crypto.code.leader_imitation_factors import leader_no_chase_guard_signal


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    return leader_no_chase_guard_signal(df)

