"""Signal-quality feature extraction for Weijie factor gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_CATEGORICAL_FEATURES = ("symbol", "action", "reason_family", "minute_bucket")
QUALITY_NUMERIC_CANDIDATES = (
    "score",
    "hour",
    "dow",
    "ret20",
    "ret60",
    "ret144",
    "ret288",
    "don",
    "taker",
    "seller",
    "qvr",
    "rsi",
    "adx",
    "atr_pct",
    "down",
    "up",
)


def build_signal_quality_feature_frame(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    out = signals.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"], errors="coerce")
    out = out.dropna(subset=["signal_time", "symbol", "action"]).copy()
    out["symbol"] = out["symbol"].astype(str)
    out["action"] = out["action"].astype(str)
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce").fillna(0.0)
    out["hour"] = out["signal_time"].dt.hour.astype(int)
    out["dow"] = out["signal_time"].dt.dayofweek.astype(int)
    minute_of_day = out["signal_time"].dt.hour * 60 + out["signal_time"].dt.minute
    out["minute_bucket"] = ((minute_of_day // 30) * 30).astype(int).astype(str)
    out["reason_family"] = out.get("reason", pd.Series(dtype=str)).astype(str).str.split(";").str[0]

    metric_rows: list[dict[str, float]] = []
    metric_names: set[str] = set()
    for reason in out.get("reason", pd.Series(dtype=str)).astype(str):
        metrics: dict[str, float] = {}
        for part in reason.split(";")[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
            metric_names.add(key)
        metric_rows.append(metrics)
    for metric in sorted(metric_names):
        out[metric] = [row.get(metric, np.nan) for row in metric_rows]
        out[metric] = pd.to_numeric(out[metric], errors="coerce").fillna(0.0)
    return out.reset_index(drop=True)
