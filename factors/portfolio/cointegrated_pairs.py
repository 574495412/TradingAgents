"""Training-only pair formation; no portfolio returns participate in selection."""

import itertools
import warnings

import numpy as np
import pandas as pd


def bh_adjust(values):
    """Keep unknown tests unknown; declared family size includes unknown slots."""
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
        raise ValueError("P-values must be a vector in [0,1] or unknown")
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    out = np.full(len(p), np.nan)
    if len(order):
        q = p[order] * len(p) / np.arange(1, len(order) + 1)
        out[order] = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    return out


def form_pairs(closes, quote, clock, parameters, start=None):
    from statsmodels.tsa.stattools import coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    p = parameters
    clock = pd.Timestamp(clock)
    if clock.tzinfo is None:
        raise ValueError("Explicit formation timezone required")
    for frame in (closes, quote):
        if (
            not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.tz is None
            or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing
            or not frame.columns.is_unique
        ):
            raise ValueError("Unique sorted timezone-aware observations required")
    # Raw daily bars are indexed by their opening date; only prior days are closed.
    begin = (
        pd.Timestamp(start) if start is not None else clock - pd.Timedelta(days=p["rolling_days"])
    )
    history = closes.loc[(closes.index >= begin) & (closes.index < clock)]
    trailing_sample = quote.loc[quote.index < clock].tail(p["turnover_days"])
    trailing = trailing_sample.mean().where(trailing_sample.notna().sum() >= p["turnover_days"])
    names = sorted(
        [s for s in history if trailing.get(s, 0) >= p["minimum_daily_quote"]],
        key=lambda s: (-trailing[s], s),
    )[: p["maximum_names"]]
    names = sorted(
        s
        for s in names
        if len(history[s].dropna()) >= p["minimum_daily_rows"]
        and (history[s].dropna() > 0).all()
        and np.isfinite(history[s].dropna()).all()
    )
    records = []
    test_values = []
    for x, y in itertools.combinations(names, 2):
        sample = np.log(history[[x, y]].dropna())
        if len(sample) < p["minimum_daily_rows"]:
            records.append(
                {
                    "x": x,
                    "y": y,
                    "rows": len(sample),
                    "fit_at": clock.isoformat(),
                    "errors": ["Insufficient overlapping observations"],
                    "warnings": [],
                    "eg_pvalues": [None, None],
                    "test_offsets": [len(test_values), len(test_values) + 1],
                }
            )
            test_values.extend([float("nan"), float("nan")])
            continue
        a = sample[x].to_numpy()
        b = sample[y].to_numpy()
        record = {
            "x": x,
            "y": y,
            "rows": len(sample),
            "first_observation": sample.index[0].isoformat(),
            "last_observation": sample.index[-1].isoformat(),
            "fit_at": clock.isoformat(),
            "errors": [],
        }
        values = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for dependent, independent in ((b, a), (a, b)):
                try:
                    _, pv, _ = coint(
                        dependent,
                        independent,
                        trend=p["eg_trend"],
                        maxlag=p["eg_maxlag"],
                        autolag=p["eg_autolag"],
                    )
                    values.append(float(pv) if np.isfinite(pv) else float("nan"))
                except (ValueError, np.linalg.LinAlgError) as error:
                    values.append(float("nan"))
                    record["errors"].append(str(error))
        record["eg_pvalues"] = [v if np.isfinite(v) else None for v in values]
        record["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
        record["test_offsets"] = [len(test_values), len(test_values) + 1]
        test_values.extend(values)
        design = np.column_stack([np.ones(len(a)), a])
        alpha, beta = np.linalg.lstsq(design, b, rcond=None)[0]
        residual = b - alpha - beta * a
        sd = float(np.std(residual, ddof=1))
        speed = np.linalg.lstsq(
            np.column_stack([np.ones(len(residual) - 1), residual[:-1]]),
            np.diff(residual),
            rcond=None,
        )[0][1]
        rho = 1 + speed
        half = float(-np.log(2) / np.log(rho)) if 0 < rho < 1 else None
        record.update(
            alpha=float(alpha),
            beta=float(beta),
            mean=float(residual.mean()),
            sigma=sd,
            half_life_days=half,
            correlation=float(np.corrcoef(a, b)[0, 1]),
        )
        records.append(record)
    adjusted = bh_adjust(test_values)
    admitted = []
    for r in records:
        qs = [adjusted[k] for k in r.pop("test_offsets")]
        r["eg_qvalues"] = [float(q) if np.isfinite(q) else None for q in qs]
        r["johansen_rank_one"] = None
        r["admitted"] = False
        if not all(np.isfinite(q) for q in qs):
            r["rejection"] = "unknown_test"
            continue
        if not all(q <= p["fdr"] for q in qs):
            r["rejection"] = "cointegration_fdr"
            continue
        if not p["beta_min"] <= r["beta"] <= p["beta_max"]:
            r["rejection"] = "hedge_beta"
            continue
        if not r["correlation"] >= p["minimum_price_correlation"]:
            r["rejection"] = "price_correlation"
            continue
        if not r["sigma"] > p["minimum_residual_sigma"]:
            r["rejection"] = "numerical_spread"
            continue
        r["two_sigma_edge_unit_gross"] = 2 * r["sigma"] / (1 + r["beta"])
        if not r["two_sigma_edge_unit_gross"] > p["minimum_two_sigma_edge_unit_gross"]:
            r["rejection"] = "cost_hurdle"
            continue
        if not (
            r["half_life_days"] is not None
            and p["minimum_half_life_days"] <= r["half_life_days"] <= p["maximum_half_life_days"]
        ):
            r["rejection"] = "half_life"
            continue
        sample = np.log(history[[r["y"], r["x"]]].dropna()).to_numpy()
        try:
            j = coint_johansen(
                sample, det_order=p["johansen_det_order"], k_ar_diff=p["johansen_lag_diff"]
            )
            rank_one = bool(j.lr1[0] > j.cvt[0, 1] and j.lr1[1] <= j.cvt[1, 1])
            r["johansen_rank_one"] = rank_one
            r["johansen_trace"] = j.lr1.tolist()
            r["johansen_critical95"] = j.cvt[:, 1].tolist()
        except (ValueError, np.linalg.LinAlgError) as error:
            r["errors"].append(str(error))
            r["rejection"] = "johansen_unknown"
            continue
        if rank_one:
            r["admitted"] = True
            r["rejection"] = None
            admitted.append(r)
        else:
            r["rejection"] = "johansen_rank"
    chosen = []
    occupied = set()
    for r in sorted(
        admitted, key=lambda v: (max(v["eg_qvalues"]), v["half_life_days"], v["x"], v["y"])
    ):
        if r["x"] in occupied or r["y"] in occupied:
            continue
        chosen.append(r.copy())
        occupied.update([r["x"], r["y"]])
        if len(chosen) >= p["maximum_pairs"]:
            break
    return {
        "fit_at": clock.isoformat(),
        "training_start": begin.isoformat(),
        "training_end_exclusive": clock.isoformat(),
        "eligible_names": names,
        "pairs_tested": len(records),
        "directional_tests": len(test_values),
        "unknown_tests": sum(not np.isfinite(v) for v in test_values),
        "admitted_before_disjoint_selection": len(admitted),
        "selected": chosen,
        "tests": records,
        "returns_used_for_selection": False,
    }
