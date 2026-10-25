"""Leader-imitation symbol preference factor."""

from __future__ import annotations

import pandas as pd

from factors.crypto.code.leader_imitation_factors import leader_meme_symbol_score_signal


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    return leader_meme_symbol_score_signal(df)

