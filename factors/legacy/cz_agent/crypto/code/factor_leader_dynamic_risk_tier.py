"""Leader-imitation dynamic risk-tier quality factor."""

from __future__ import annotations

import pandas as pd

from factors.crypto.code.leader_imitation_factors import leader_dynamic_risk_tier_signal


def calculate_factor(df: pd.DataFrame) -> pd.Series:
    return leader_dynamic_risk_tier_signal(df)

