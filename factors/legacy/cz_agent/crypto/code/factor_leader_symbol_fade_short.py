"""Leader symbol-edge fade-short factor."""

from __future__ import annotations

import pandas as pd

from factors.crypto.code.leader_imitation_factors import leader_symbol_fade_short_signal


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    return leader_symbol_fade_short_signal(df)
