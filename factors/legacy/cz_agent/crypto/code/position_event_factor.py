"""Generic position-history trained OHLCV event factor.

The model is trained offline from public leader position-history labels, but
live signals only use current OHLCV/time features and the frozen model file.
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd

from cz.trading.crypto.factors.action_codec import (
    DEFAULT_ACTION_SIGNAL_VALUES,
    action_from_label,
    signal_value_for_action,
)


DEFAULT_PARAMETERS: dict[str, object] = {
    "model_path": "cz/trading/crypto/leader/models/leader_position_event_rf_v1.joblib",
    "probability_threshold": 0.55,
    "signal_value": 0.82,
    "close_signal_value": 0.62,
    "action_label_classes": True,
    "action_signal_values": dict(DEFAULT_ACTION_SIGNAL_VALUES),
    "kline_interval": "5m",
    "warmup_bars": 180,
    "post_close_cooldown_bars": 0,
    "anti_flip_cooldown_bars": 0,
    "post_close_cooldown_minutes": 0,
    "anti_flip_cooldown_minutes": 0,
    "signal_exit_min_hold_minutes": 0,
}


def calculate_factor(df: pd.DataFrame, parameters: dict | None = None) -> pd.Series:
    params = dict(DEFAULT_PARAMETERS)
    params.update(parameters or {})
    predict_last_only = bool(getattr(df, "attrs", {}).get("factor_predict_last_only", False))
    input_df = df
    if predict_last_only:
        warmup_bars = max(int(params.get("warmup_bars") or 0), 180)
        if warmup_bars > 0 and len(input_df) > warmup_bars:
            input_df = input_df.tail(warmup_bars)
    result = pd.Series(0.0, index=input_df.index, dtype="float64")
    required = {"high", "low", "close", "volume"}
    if input_df is None or input_df.empty or not required.issubset(set(input_df.columns)):
        return result

    artifact = _load_model(str(params.get("model_path") or DEFAULT_PARAMETERS["model_path"]))
    model = _select_model(artifact, input_df.attrs.get("symbol") or params.get("symbol"))
    features = list(artifact.get("features") or _feature_columns())
    classes = [int(value) for value in getattr(model, "classes_", artifact.get("classes", [-1, 0, 1]))]
    threshold = float(params.get("probability_threshold", artifact.get("threshold", 0.55)))
    signal_value = float(params.get("signal_value", artifact.get("signal_value", 0.82)))
    close_signal_value = float(params.get("close_signal_value", artifact.get("close_signal_value", 0.62)))
    action_label_classes = bool(params.get("action_label_classes", artifact.get("action_label_classes", False)))
    action_params = {
        "action_signal_values": params.get("action_signal_values")
        or artifact.get("action_signal_values")
        or DEFAULT_ACTION_SIGNAL_VALUES,
    }
    if model is None or not hasattr(model, "predict_proba"):
        return result

    feature_frame = _build_features(input_df).reindex(columns=features).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if feature_frame.empty:
        return result
    prediction_frame = feature_frame.tail(1) if predict_last_only else feature_frame
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*sklearn\.utils\.parallel\.delayed.*",
            category=UserWarning,
        )
        probabilities = model.predict_proba(prediction_frame)
    values = np.zeros(len(prediction_frame), dtype="float64")
    actions = pd.Series("", index=prediction_frame.index, dtype="object")
    for idx, row in enumerate(probabilities):
        best_pos = int(np.argmax(row))
        label = classes[best_pos]
        probability = float(row[best_pos])
        if label != 0 and probability >= threshold:
            action = action_from_label(label, action_label_classes=action_label_classes)
            if action:
                values[idx] = signal_value_for_action(
                    action,
                    signal_value=signal_value,
                    close_signal_value=close_signal_value,
                    params=action_params,
                )
                actions.iloc[idx] = action
            else:
                values[idx] = signal_value if label > 0 else -signal_value
    result.loc[prediction_frame.index] = values
    result = result.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
    result.attrs["factor_actions"] = actions.reindex(result.index).fillna("")
    return result


def calculate_intrabar_factor_candidates(
    execution_frame: pd.DataFrame,
    minute_frame: pd.DataFrame,
    parameters: dict | None = None,
) -> pd.Series:
    """Vectorized unfinished-bar RF signals for backtest/live alignment scans.

    Each row is a candidate minute timestamp. Features are computed from
    completed execution bars before the current factor bin plus the partial
    minute aggregation inside that bin, matching ``calculate_factor`` on the
    composed one-row live frame.
    """

    params = dict(DEFAULT_PARAMETERS)
    params.update(parameters or {})
    result = pd.Series(0.0, index=minute_frame.index if minute_frame is not None else None, dtype="float64")
    required = {"open", "high", "low", "close", "volume"}
    if (
        execution_frame is None
        or minute_frame is None
        or execution_frame.empty
        or minute_frame.empty
        or not required.issubset(set(execution_frame.columns))
        or not required.issubset(set(minute_frame.columns))
    ):
        result.attrs["factor_actions"] = pd.Series("", index=result.index, dtype="object")
        return result

    artifact = _load_model(str(params.get("model_path") or DEFAULT_PARAMETERS["model_path"]))
    model = _select_model(artifact, execution_frame.attrs.get("symbol") or minute_frame.attrs.get("symbol") or params.get("symbol"))
    if model is None or not hasattr(model, "predict_proba"):
        result.attrs["factor_actions"] = pd.Series("", index=result.index, dtype="object")
        return result

    features = list(artifact.get("features") or _feature_columns())
    feature_frame = (
        _build_intrabar_candidate_features(execution_frame, minute_frame, params)
        .reindex(columns=features)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    if feature_frame.empty:
        result.attrs["factor_actions"] = pd.Series("", index=result.index, dtype="object")
        return result

    values, actions = _predict_feature_frame(model, artifact, feature_frame, params)
    result = pd.Series(0.0, index=feature_frame.index, dtype="float64")
    result.loc[feature_frame.index] = values
    result = result.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float64")
    result.attrs["factor_actions"] = actions.reindex(result.index).fillna("")
    return result


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    high = pd.to_numeric(df["high"], errors="coerce").astype("float64")
    low = pd.to_numeric(df["low"], errors="coerce").astype("float64")
    volume = pd.to_numeric(df["volume"], errors="coerce").astype("float64")
    out = pd.DataFrame(index=df.index)
    for bars in (1, 3, 5, 12, 15, 30, 60, 96, 144):
        out[f"ret{bars}"] = (close / close.shift(bars) - 1.0) * 100.0
    vol_base = volume.shift(1).rolling(30, min_periods=5).median()
    out["vol_ratio"] = volume / (vol_base + 1e-10)
    out["range_pct"] = (high / low.replace(0.0, np.nan) - 1.0) * 100.0
    candle_range = (high - low).replace(0.0, np.nan)
    out["close_pos"] = ((close - low) / candle_range).clip(0.0, 1.0)
    timestamps = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))
    out["hour"] = timestamps.hour
    out["dow"] = timestamps.dayofweek
    return out


def _build_intrabar_candidate_features(
    execution_frame: pd.DataFrame,
    minute_frame: pd.DataFrame,
    params: Mapping[str, Any],
) -> pd.DataFrame:
    interval = str(params.get("kline_interval") or "5m")
    factor_minutes = _interval_minutes(interval)
    if factor_minutes <= 1:
        return _build_features(minute_frame)

    minute = minute_frame[["open", "high", "low", "close", "volume"]].copy().sort_index()
    minute.index = pd.DatetimeIndex(pd.to_datetime(minute.index, errors="coerce"))
    minute = minute[minute.index.notna()]
    if minute.empty:
        return pd.DataFrame(index=minute.index)
    minute = minute.apply(pd.to_numeric, errors="coerce")
    bin_start = minute.index.floor(_interval_floor_rule(factor_minutes))
    grouped = minute.groupby(bin_start, sort=False)

    partial = pd.DataFrame(index=minute.index)
    partial["open"] = grouped["open"].transform("first")
    partial["high"] = grouped["high"].cummax()
    partial["low"] = grouped["low"].cummin()
    partial["close"] = minute["close"]
    partial["volume"] = grouped["volume"].cumsum()

    execution = execution_frame[["close", "volume"]].copy().sort_index()
    execution.index = pd.DatetimeIndex(pd.to_datetime(execution.index, errors="coerce"))
    execution = execution[execution.index.notna()]
    execution = execution[~execution.index.duplicated(keep="last")].apply(pd.to_numeric, errors="coerce")
    reference = pd.DataFrame(index=execution.index)
    close = execution["close"]
    volume = execution["volume"]
    for bars in (1, 3, 5, 12, 15, 30, 60, 96, 144):
        reference[f"ret{bars}_base"] = close.shift(bars)
    reference["vol_base"] = volume.shift(1).rolling(30, min_periods=5).median()
    aligned = reference.reindex(pd.DatetimeIndex(bin_start))
    aligned.index = partial.index

    out = pd.DataFrame(index=partial.index)
    for bars in (1, 3, 5, 12, 15, 30, 60, 96, 144):
        out[f"ret{bars}"] = (partial["close"] / aligned[f"ret{bars}_base"] - 1.0) * 100.0
    out["vol_ratio"] = partial["volume"] / (aligned["vol_base"] + 1e-10)
    out["range_pct"] = (partial["high"] / partial["low"].replace(0.0, np.nan) - 1.0) * 100.0
    candle_range = (partial["high"] - partial["low"]).replace(0.0, np.nan)
    out["close_pos"] = ((partial["close"] - partial["low"]) / candle_range).clip(0.0, 1.0)
    bin_index = pd.DatetimeIndex(bin_start)
    out["hour"] = bin_index.hour
    out["dow"] = bin_index.dayofweek
    return out


def _predict_feature_frame(
    model: Any,
    artifact: Mapping[str, Any],
    feature_frame: pd.DataFrame,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, pd.Series]:
    classes = [int(value) for value in getattr(model, "classes_", artifact.get("classes", [-1, 0, 1]))]
    threshold = float(params.get("probability_threshold", artifact.get("threshold", 0.55)))
    signal_value = float(params.get("signal_value", artifact.get("signal_value", 0.82)))
    close_signal_value = float(params.get("close_signal_value", artifact.get("close_signal_value", 0.62)))
    action_label_classes = bool(params.get("action_label_classes", artifact.get("action_label_classes", False)))
    action_params = {
        "action_signal_values": params.get("action_signal_values")
        or artifact.get("action_signal_values")
        or DEFAULT_ACTION_SIGNAL_VALUES,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*sklearn\.utils\.parallel\.delayed.*",
            category=UserWarning,
        )
        probabilities = model.predict_proba(feature_frame)
    values = np.zeros(len(feature_frame), dtype="float64")
    actions = pd.Series("", index=feature_frame.index, dtype="object")
    for idx, row in enumerate(probabilities):
        best_pos = int(np.argmax(row))
        label = classes[best_pos]
        probability = float(row[best_pos])
        if label != 0 and probability >= threshold:
            action = action_from_label(label, action_label_classes=action_label_classes)
            if action:
                values[idx] = signal_value_for_action(
                    action,
                    signal_value=signal_value,
                    close_signal_value=close_signal_value,
                    params=action_params,
                )
                actions.iloc[idx] = action
            else:
                values[idx] = signal_value if label > 0 else -signal_value
    return values, actions


def _interval_minutes(interval: str) -> int:
    text = str(interval or "").strip().lower()
    if text.endswith("m"):
        return max(1, int(float(text[:-1] or 1)))
    if text.endswith("h"):
        return max(1, int(float(text[:-1] or 1) * 60))
    if text.endswith("d"):
        return max(1, int(float(text[:-1] or 1) * 1440))
    return 5


def _interval_floor_rule(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}min"
    if minutes < 1440:
        return f"{minutes // 60}h"
    return "1D"


def _feature_columns() -> tuple[str, ...]:
    return (
        "ret1",
        "ret3",
        "ret5",
        "ret12",
        "ret15",
        "ret30",
        "ret60",
        "ret96",
        "ret144",
        "vol_ratio",
        "range_pct",
        "close_pos",
        "hour",
        "dow",
    )


def _select_model(artifact: Mapping[str, Any], symbol: object) -> Any:
    models_by_symbol = artifact.get("models_by_symbol")
    if isinstance(models_by_symbol, Mapping):
        normalized = _compact_symbol(symbol)
        if normalized and normalized in models_by_symbol:
            return models_by_symbol[normalized]
    return artifact.get("model")


def _set_runtime_model_options(model: Any) -> None:
    if model is None:
        return
    raw_n_jobs = os.getenv("POSITION_EVENT_RF_RUNTIME_N_JOBS", "").strip()
    if not raw_n_jobs:
        return
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = int(raw_n_jobs)
        except Exception:
            return


def _set_artifact_runtime_options(artifact: Mapping[str, Any]) -> None:
    _set_runtime_model_options(artifact.get("model"))
    models_by_symbol = artifact.get("models_by_symbol")
    if isinstance(models_by_symbol, Mapping):
        for model in models_by_symbol.values():
            _set_runtime_model_options(model)


def _compact_symbol(value: object) -> str:
    text = str(value or "").strip().upper().replace("-", "/")
    if "/" in text:
        base, quote = text.split("/", 1)
        return f"{base}{quote}"
    return text


@lru_cache(maxsize=4)
def _load_model(model_path: str) -> Mapping[str, Any]:
    import joblib

    path = Path(model_path).expanduser()
    if not path.is_absolute():
        candidates = [Path.cwd() / path]
        module_file = globals().get("__file__")
        if module_file:
            candidates.append(Path(module_file).resolve().parents[5] / path)
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    artifact = joblib.load(path)
    if isinstance(artifact, Mapping):
        _set_artifact_runtime_options(artifact)
        return artifact
    _set_runtime_model_options(artifact)
    return {"model": artifact}
