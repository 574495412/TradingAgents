"""Fixed monthly past-only HMM, logistic and MLP model comparisons."""

import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from factors.common.market_states import STATE_FEATURES


def filter_step(observation, startprob, transmat, means, variances, previous=None):
    """Diagonal Gaussian forward recursion; never a backward/smoothing pass."""
    if (np.asarray(variances) <= 0).any() or not np.isfinite(observation).all():
        raise ValueError("Finite observations and positive variances required")
    prior = np.asarray(startprob) if previous is None else np.asarray(previous) @ transmat
    emission = -0.5 * (np.log(2 * np.pi * variances) + (observation - means) ** 2 / variances).sum(
        axis=1
    )
    posterior = np.log(np.maximum(prior, 1e-300)) + emission
    return np.exp(posterior - logsumexp(posterior))


def predictions(states, benchmark, start="2025-01-01", end="2026-09-01"):
    import hmmlearn
    import sklearn
    from hmmlearn.hmm import GaussianHMM
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    if sklearn.__version__ != "1.7.2" or hmmlearn.__version__ != "0.3.3":
        raise ValueError("Use the source-pinned isolated ML dependency versions")
    x = states[STATE_FEATURES].astype(float)
    opening = benchmark.open.reindex(x.index)
    future_return = opening.shift(-1) / opening - 1
    # The next opening bar is not known before its publication; keep5minutes latency.
    mature_at = pd.Series(x.index + pd.Timedelta(days=1, minutes=5), index=x.index)
    clocks = x.index[
        (x.index >= pd.Timestamp(start, tz="UTC")) & (x.index < pd.Timestamp(end, tz="UTC"))
    ]
    out = pd.DataFrame(
        index=clocks, columns=["linear_probability", "mlp_probability", "hmm_state"], dtype=object
    )
    fits = []
    month = None
    model = None
    previous = None
    for clock in clocks:
        if month != (clock.year, clock.month):
            month = (clock.year, clock.month)
            mask = (
                (x.index >= clock - pd.Timedelta(days=365))
                & (mature_at < clock)
                & x.notna().all(axis=1)
                & future_return.notna()
            )
            train = x.loc[mask]
            labels = future_return.loc[mask].gt(0).astype(int)
            model = None
            previous = None
            if len(train) < 180 or labels.nunique() < 2:
                fits.append(
                    {
                        "fit_at": clock.isoformat(),
                        "rows": len(train),
                        "status": "insufficient_history",
                    }
                )
                continue
            scaler = StandardScaler().fit(train)
            z = scaler.transform(train)
            hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=17)
            linear = LogisticRegression(C=1, max_iter=300, random_state=17)
            mlp = MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="tanh",
                alpha=1,
                learning_rate_init=0.001,
                max_iter=300,
                shuffle=False,
                early_stopping=False,
                random_state=17,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                hmm.fit(z)
                linear.fit(z, labels)
                mlp.fit(z, labels)
            raw_means = scaler.inverse_transform(hmm.means_)
            history = list(hmm.monitor_.history)
            hmm_converged = len(history) >= 2 and 0 <= history[-1] - history[-2] < hmm.tol
            state_map = []
            for mean in raw_means:
                trend, _, vol_ratio, _, _ = mean
                state_map.append(
                    "volatility_breakout"
                    if vol_ratio > 1
                    else "trend_up"
                    if trend >= 1
                    else "trend_down"
                    if trend <= -1
                    else "range"
                )
            # Rebuild the filter with this month's fitted parameters through the last
            # prior observation, then update once on the current decision's features.
            past = x.loc[(x.index >= train.index[0]) & (x.index < clock)].dropna()
            variances = np.diagonal(hmm.covars_, axis1=1, axis2=2)
            for observation in scaler.transform(past):
                previous = filter_step(
                    observation, hmm.startprob_, hmm.transmat_, hmm.means_, variances, previous
                )
            model = (scaler, hmm, linear, mlp, variances, state_map)
            messages = [str(w.message) for w in caught]
            # Warnings indicate incomplete optimization; forecasts are left unknown.
            invalid = {
                "linear": any("lbfgs" in m.lower() for m in messages),
                "mlp": any("maximum iterations" in m.lower() for m in messages),
            }
            fits.append(
                {
                    "fit_at": clock.isoformat(),
                    "rows": len(train),
                    "training_first": train.index[0].isoformat(),
                    "training_last": train.index[-1].isoformat(),
                    "maximum_label_available": mature_at.loc[mask].max().isoformat(),
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "warnings": messages,
                    "invalid_forecasts": invalid,
                    "hmm_state_map": state_map,
                    "hmm_start": hmm.startprob_.tolist(),
                    "hmm_transitions": hmm.transmat_.tolist(),
                    "hmm_means": hmm.means_.tolist(),
                    "hmm_variances": variances.tolist(),
                    "hmm_converged": bool(hmm_converged),
                    "hmm_last_log_likelihoods": history,
                    "hmm_iterations": hmm.monitor_.iter,
                    "linear_coefficient": linear.coef_.tolist(),
                    "linear_intercept": linear.intercept_.tolist(),
                    "mlp_coefficients": [a.tolist() for a in mlp.coefs_],
                    "mlp_intercepts": [a.tolist() for a in mlp.intercepts_],
                    "mlp_iterations": mlp.n_iter_,
                    "status": "fitted",
                }
            )
        if model is None or x.loc[clock].isna().any():
            continue
        scaler, hmm, linear, mlp, variances, state_map = model
        z = scaler.transform(x.loc[[clock]])
        previous = filter_step(z[0], hmm.startprob_, hmm.transmat_, hmm.means_, variances, previous)
        if fits[-1]["hmm_converged"]:
            out.loc[clock, "hmm_state"] = state_map[int(previous.argmax())]
        if not invalid["linear"]:
            out.loc[clock, "linear_probability"] = float(linear.predict_proba(z)[0, 1])
        if not invalid["mlp"]:
            out.loc[clock, "mlp_probability"] = float(mlp.predict_proba(z)[0, 1])
    out[["linear_probability", "mlp_probability"]] = out[
        ["linear_probability", "mlp_probability"]
    ].astype(float)
    out["hmm_state"] = out.hmm_state.fillna("unknown")
    return out, fits
