"""Leader-imitation exit timing / giveback-risk factor."""

from __future__ import annotations

import pandas as pd

from factors.crypto.code.leader_imitation_factors import leader_exit_timing_signal


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    return leader_exit_timing_signal(df)

