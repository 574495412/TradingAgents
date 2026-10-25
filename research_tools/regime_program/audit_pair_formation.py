"""Rebuild BH and Johansen covariance/eigensystems without formation imports."""

import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from scipy.linalg import eigh
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from research_tools.factor_lab.io import read, save, sha


def johansen_reference(log_prices):
    """Constant detrending, one difference lag; symmetric generalized eigenproblem."""
    y = log_prices - log_prices.mean(axis=0)
    changes = np.diff(y, axis=0)
    z = changes[:-1] - changes[:-1].mean(axis=0)
    response = changes[1:] - changes[1:].mean(axis=0)
    levels = y[1:-1] - y[1:-1].mean(axis=0)
    r0 = response - z @ np.linalg.lstsq(z, response, rcond=None)[0]
    rk = levels - z @ np.linalg.lstsq(z, levels, rcond=None)[0]
    t = len(rk)
    skk, sk0, s00 = rk.T @ rk / t, rk.T @ r0 / t, r0.T @ r0 / t
    covariance = sk0 @ np.linalg.solve(s00, sk0.T)
    eigenvalues = eigh(covariance, skk, eigvals_only=True)[::-1]
    if not np.isfinite(eigenvalues).all() or (eigenvalues < 0).any() or (eigenvalues >= 1).any():
        raise ValueError("Invalid reference eigenvalues")
    trace = np.array([-t * np.log1p(-eigenvalues[i:]).sum() for i in range(2)])
    return trace, eigenvalues


def audit(root, output):
    completion = read(root / "completion.json")
    for name, digest in completion["manifest"].items():
        if Path(name).name != name or sha(root / name) != digest:
            raise ValueError("Formation manifest changed")
    plan = read(root / "plan.json")
    if sha(root / "plan.json") != read(root / "registration.json")["plan_sha256"]:
        raise ValueError("Registration changed")
    for path, digest in plan["source_hashes"].items():
        if sha(Path(path)) != digest:
            raise ValueError("Formation implementation/source changed after registration")
    if statsmodels.__version__ != plan["runtime_versions"]["statsmodels"]:
        raise ValueError("Statistical version changed")
    data = Path(plan["data"])
    for name, digest in read(data / "completion.json")["manifest"].items():
        if sha(data / name) != digest:
            raise ValueError("Data digest changed")
    prices = {}
    quotes = {}
    for symbol in read(data / "prepare_plan.json")["symbols"]:
        raw = pd.read_parquet(data / (symbol + "_5m.parquet"))
        grouped = raw.resample("D")
        valid = grouped.close.count() == 288
        prices[symbol] = grouped.close.last().where(valid)
        quotes[symbol] = grouped.quote_volume.sum(min_count=288).where(valid)
    close, quote = pd.DataFrame(prices), pd.DataFrame(quotes)
    counts = Counter()
    max_p_error = max_q_error = max_trace_error = max_eigen_imag = 0.0
    warning_counts = Counter()
    for profile in read(root / "summary.json")["profile"]:
        fit = read(root / profile["file"])
        start, end = pd.Timestamp(fit["training_start"]), pd.Timestamp(fit["fit_at"])
        frame = close.loc[(close.index >= start) & (close.index < end)]
        formation = plan["formation"]
        turn = quote.loc[quote.index < end].tail(formation["turnover_days"])
        volume = turn.mean().where(turn.notna().sum() >= formation["turnover_days"])
        names = sorted(
            [s for s in frame if volume.get(s, 0) >= formation["minimum_daily_quote"]],
            key=lambda s: (-volume[s], s),
        )[: formation["maximum_names"]]
        names = sorted(
            s
            for s in names
            if frame[s].notna().sum() >= formation["minimum_daily_rows"]
            and (frame[s].dropna() > 0).all()
            and np.isfinite(frame[s].dropna()).all()
        )
        if (
            names != fit["eligible_names"]
            or len(fit["tests"]) != len(names) * (len(names) - 1) // 2
        ):
            raise ValueError("Formation universe/pair family incomplete")
        all_p = []
        for pair in fit["tests"]:
            a = np.log(frame[[pair["y"], pair["x"]]].dropna()).to_numpy()
            if len(a) < formation["minimum_daily_rows"]:
                all_p.extend([None, None])
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                p = [
                    float(coint(a[:, 0], a[:, 1], trend="c", maxlag=1, autolag=None)[1]),
                    float(coint(a[:, 1], a[:, 0], trend="c", maxlag=1, autolag=None)[1]),
                ]
            warning_counts.update(str(w.message) for w in caught)
            error = max(abs(np.array(p) - pair["eg_pvalues"]))
            max_p_error = max(max_p_error, error)
            if error > 1e-12:
                raise ValueError("Cointegration p differs")
            all_p.extend(p)
            if pair["johansen_rank_one"] is not None:
                trace, values = johansen_reference(a)
                max_trace_error = max(
                    max_trace_error, float(max(abs(trace - pair["johansen_trace"])))
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    test = coint_johansen(a, det_order=0, k_ar_diff=1)
                warning_counts.update(str(w.message) for w in caught)
                max_eigen_imag = max(max_eigen_imag, float(max(abs(np.imag(test.eig)))))
                if (
                    max(abs(trace - pair["johansen_trace"])) > 1e-8
                    or max(abs(values - test.eig)) > 1e-9
                    or max(abs(np.imag(test.eig))) > 1e-12
                ):
                    raise ValueError("Johansen numerical reconstruction differs")
                rank_one = bool(
                    trace[0] > pair["johansen_critical95"][0]
                    and trace[1] <= pair["johansen_critical95"][1]
                )
                if rank_one != pair["johansen_rank_one"]:
                    raise ValueError("Independent Johansen rank differs")
                counts["johansen_reconstructions"] += 1
            counts["pair_reconstructions"] += 1
        order = sorted((p, i) for i, p in enumerate(all_p) if p is not None and np.isfinite(p))
        reference = [None] * len(all_p)
        running = 1.0
        for k in range(len(order) - 1, -1, -1):
            p, i = order[k]
            running = min(running, p * len(all_p) / (k + 1))
            reference[i] = running
        for i, pair in enumerate(fit["tests"]):
            expected = reference[i * 2 : i * 2 + 2]
            if any(
                (a is None) != (b is None)
                for a, b in zip(expected, pair["eg_qvalues"], strict=True)
            ):
                raise ValueError("Unknown tests changed")
            errors = [
                abs(a - b)
                for a, b in zip(expected, pair["eg_qvalues"], strict=True)
                if a is not None
            ]
            max_q_error = max(max_q_error, max(errors, default=0))
            if max(errors, default=0) > 1e-12:
                raise ValueError("BH family reconstruction differs")
        counts["directional_tests"] += len(all_p)
        counts["formation_fits"] += 1
    save(
        output,
        {
            "schema": "independent-pair-formation-audit.v1",
            "passed": True,
            "formation_root": str(root.resolve()),
            "completion_sha256": sha(root / "completion.json"),
            "auditor_sha256": sha(Path(__file__)),
            "counts": dict(counts),
            "maximum_pvalue_error": max_p_error,
            "maximum_qvalue_error": max_q_error,
            "maximum_johansen_trace_error": max_trace_error,
            "maximum_eigenvalue_imaginary": max_eigen_imag,
            "recomputed_warning_counts": dict(warning_counts),
            "warning_interpretation": "All Johansen eigenvalue imaginary components independently checked; symmetric generalized eigenproblem reconstructs trace/rank. Zero-imaginary complex dtype casts are recorded, not assumed harmless.",
            "account_returns_computed": False,
            "profit_verified": False,
            "orders_enabled": False,
        },
    )
    print(
        {
            "passed": True,
            **dict(counts),
            "trace_error": max_trace_error,
            "eigen_imaginary": max_eigen_imag,
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output.resolve())
