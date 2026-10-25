"""Independent serialized-weight forecast audit, without model/framework imports."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def audit(root, states_root, output):
    for name, checksum in read(root / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(root / name) != checksum:
            raise ValueError("Prepared model/intent evidence changed")
    for name, checksum in read(states_root / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(states_root / name) != checksum:
            raise ValueError("State evidence changed")
    states = pd.read_parquet(states_root / "states_daily.parquet")
    columns = ["trend_z", "efficiency", "volatility_ratio", "breadth", "eligible_count"]
    forecasts = pd.read_parquet(root / "model_forecasts.parquet")
    fits = read(root / "fits.json")
    errors = {"linear": 0.0, "mlp": 0.0}
    hmm_rows = 0
    reconstructed = 0
    for fit_number, fit in enumerate(fits):
        clock = pd.Timestamp(fit["fit_at"])
        stop = (
            pd.Timestamp(fits[fit_number + 1]["fit_at"])
            if fit_number + 1 < len(fits)
            else forecasts.index[-1] + pd.Timedelta(days=1)
        )
        if fit["status"] != "fitted":
            continue
        if not pd.Timestamp(fit["maximum_label_available"]) < clock:
            raise ValueError("Immature training labels")
        historical = states.loc[
            (states.index >= clock - pd.Timedelta(days=365))
            & (states.index + pd.Timedelta(days=1, minutes=5) < clock),
            columns,
        ].dropna()
        if (
            len(historical) != fit["rows"]
            or historical.index[0].isoformat() != fit["training_first"]
            or historical.index[-1].isoformat() != fit["training_last"]
        ):
            raise ValueError("Training population differs from matured past-only rows")
        if (
            np.max(np.abs(historical.mean().to_numpy() - fit["scaler_mean"])) > 1e-12
            or np.max(np.abs(historical.std(ddof=0).to_numpy() - fit["scaler_scale"])) > 1e-12
        ):
            raise ValueError("Training-only standardization differs")
        observations = states.loc[
            (states.index >= pd.Timestamp(fit["training_first"])) & (states.index < stop), columns
        ].dropna()
        z = (observations.to_numpy() - np.array(fit["scaler_mean"])) / np.array(fit["scaler_scale"])
        prior = np.array(fit["hmm_start"], dtype=np.longdouble)
        transition = np.array(fit["hmm_transitions"], dtype=np.longdouble)
        means = np.array(fit["hmm_means"], dtype=np.longdouble)
        variance = np.array(fit["hmm_variances"], dtype=np.longdouble)
        for number, t in enumerate(observations.index):
            if number:
                prior = prior @ transition
            likelihood = (
                np.exp(-((z[number] - means) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)
            ).prod(axis=1)
            prior = prior * likelihood
            if not np.isfinite(prior).all() or prior.sum() <= 0:
                raise ValueError("Independent HMM probability underflow")
            prior = prior / prior.sum()
            if t < clock or t not in forecasts.index:
                continue
            expected_hmm = (
                fit["hmm_state_map"][int(prior.argmax())] if fit["hmm_converged"] else "unknown"
            )
            if forecasts.loc[t, "hmm_state"] != expected_hmm:
                raise ValueError("Independent forward-only HMM state differs")
            hmm_rows += 1
            linear = float(
                z[number] @ np.array(fit["linear_coefficient"])[0] + fit["linear_intercept"][0]
            )
            value = z[number]
            for layer, (weights, bias) in enumerate(
                zip(fit["mlp_coefficients"], fit["mlp_intercepts"], strict=True)
            ):
                value = value @ np.array(weights) + np.array(bias)
                if layer + 1 < len(fit["mlp_coefficients"]):
                    value = np.tanh(value)
            probabilities = {
                "linear": 1 / (1 + np.exp(-linear)),
                "mlp": float(1 / (1 + np.exp(-value[0]))),
            }
            for kind, probability in probabilities.items():
                actual = forecasts.loc[t, kind + "_probability"]
                if fit["invalid_forecasts"][kind]:
                    if pd.notna(actual):
                        raise ValueError("Invalid model forecast presented as valid")
                else:
                    error = abs(actual - probability)
                    if not np.isfinite(error) or error > 1e-10:
                        raise ValueError("Independent neural/logistic forecast differs")
                    errors[kind] = max(errors[kind], float(error))
            reconstructed += 1
    if reconstructed != len(forecasts):
        raise ValueError("Forecast population incomplete")
    target_checks = []
    for candidate in read(root / "plan.json")["candidates"]:
        target = pd.read_parquet(root / (candidate + "_targets.parquet"))
        weights = target.filter(like="target_")
        if (
            not np.isfinite(weights).all().all()
            or (weights.abs().sum(axis=1) > 1 + 1e-10).any()
            or weights.ne(0).sum(axis=1).gt(10).any()
        ):
            raise ValueError("Target budgets/names invalid")
        if not target.risk_scale.between(0, 1).all():
            raise ValueError("Risk scaling exceeds common account budget")
        if candidate in (
            "rule_router",
            "event_router",
            "hmm_router",
            "linear_router",
            "mlp_router",
        ):
            daily = states.state.reindex(target.index, method="ffill")
            if weights.loc[daily.isin(["unknown", "liquidity_stress"])].ne(0).any().any():
                raise ValueError("Unknown/liquidity state was routed to trading")
        target_checks.append(
            {"candidate": candidate, "hourly_rows": len(target), "unit_gross_bounded": True}
        )
    save(
        output,
        {
            "schema": "multi-regime-independent-input-audit.v1",
            "passed": True,
            "model_framework_imported": False,
            "forecast_days": reconstructed,
            "hmm_forward_days": hmm_rows,
            "maximum_probability_error": errors,
            "target_checks": target_checks,
            "input_completion_sha256": sha(root / "completion.json"),
            "strategy_profit_verified": False,
            "orders_enabled": False,
        },
    )
    print({"passed": True, "forecasts": reconstructed, "probability_error": errors}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.inputs.resolve(), args.states.resolve(), args.output.resolve())
