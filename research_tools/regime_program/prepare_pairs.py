"""Register and evaluate a bounded train-only pair formation study."""

import argparse
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.stattools import coint

from factors.common.github_formulas import daily
from factors.portfolio.cointegrated_pairs import form_pairs
from research_tools.factor_lab.io import read, save, sha
from research_tools.github_factor_study.run import snapshot_sources


def prepare(root):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("pair_plan.json")
    plan = read(plan_path)
    data, sources = Path(plan["data"]), Path(plan["source_pack"])
    if statsmodels.__version__ != "0.14.5":
        raise ValueError("Pinned statsmodels 0.14.5 required")
    if (
        sha(sources / "source_review.json")
        != read(sources / "completion.json")["source_review_sha256"]
    ):
        raise ValueError("Pinned pair source receipt changed")
    for record in read(sources / "source_review.json")["records"]:
        if sha(Path(record["local"])) != record["sha256"]:
            raise ValueError("Pair source changed")
    paths = [
        Path(__file__),
        plan_path,
        Path("factors/portfolio/cointegrated_pairs.py"),
        Path("factors/common/github_formulas.py"),
        Path("research_tools/factor_lab/io.py"),
        Path("research_tools/github_factor_study/run.py"),
        data / "completion.json",
        sources / "source_review.json",
    ]
    plan["registered_at"] = datetime.now(UTC).isoformat()
    plan["source_hashes"] = {str(p.resolve()): sha(p) for p in paths}
    plan["runtime_versions"] = {
        "statsmodels": statsmodels.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    save(root / "plan.json", plan)
    save(
        root / "registration.json",
        {
            "plan_sha256": sha(root / "plan.json"),
            "before_formation_statistics": True,
            "before_account_returns": True,
        },
    )
    (root / "source_snapshot").mkdir()
    snapshot_sources(paths, root / "source_snapshot")
    for name, digest in read(data / "completion.json")["manifest"].items():
        if Path(name).name != name or sha(data / name) != digest:
            raise ValueError("Sealed market data changed")
    symbols = read(data / "prepare_plan.json")["symbols"]
    frames = {}
    for symbol in symbols:
        raw = pd.read_parquet(data / (symbol + "_5m.parquet"))
        frame = daily(raw)
        # Formation cannot treat an incomplete UTC day as a real closed day.
        counts = raw.close.resample("D").count()
        frame.loc[counts != 288, ["close", "quote_volume"]] = np.nan
        frames[symbol] = frame
    closes = pd.DataFrame({s: f.close for s, f in frames.items()})
    quotes = pd.DataFrame({s: f.quote_volume for s, f in frames.items()})
    p = plan["formation"]
    start, end = (pd.Timestamp(p[k], tz="UTC") for k in ("fixed_start", "fixed_end"))
    fixed = form_pairs(closes, quotes, end, p, start=start)
    save(root / "fixed_formation.json", fixed)
    clocks = pd.date_range("2025-01-01", "2026-09-01", freq="MS", inclusive="left", tz="UTC")
    fits = []
    for clock in clocks:
        fit = form_pairs(closes, quotes, clock, p)
        name = f"rolling_{clock:%Y%m%d}.json"
        save(root / name, fit)
        fits.append((name, fit))
        print(
            {
                "formation": clock.isoformat(),
                "directional_tests": fit["directional_tests"],
                "eligible_pairs": len(fit["selected"]),
            },
            flush=True,
        )
    checks = []
    # Real-data prefix/current+future mutation checks use full formation outputs,
    # including rejection records; no performance data or return ranking is read.
    for clock in (clocks[0], clocks[9], clocks[-1]):
        expected = next(f for _, f in fits if f["fit_at"] == clock.isoformat())
        prefix = form_pairs(
            closes.loc[closes.index < clock], quotes.loc[quotes.index < clock], clock, p
        )
        mutated = closes.copy()
        changed_quote = quotes.copy()
        mutated.loc[clock:] *= 10000
        changed_quote.loc[clock:] = 0
        future = form_pairs(mutated, changed_quote, clock, p)
        if expected != prefix or expected != future:
            raise ValueError("Future data changed formation")
        checks.append(
            {
                "fit_at": clock.isoformat(),
                "prefix_equal": True,
                "current_and_future_mutation_equal": True,
            }
        )
    # Independently reconstruct every selected OLS fit from raw training prices;
    # compare proper EG calls in the explicit documented dependent-variable order.
    checked = 0
    max_beta_error = 0.0
    for _, fit in [("fixed_formation.json", fixed), *fits]:
        a, b = pd.Timestamp(fit["training_start"]), pd.Timestamp(fit["fit_at"])
        for pair in fit["selected"]:
            sample = np.log(
                closes.loc[
                    (closes.index >= a) & (closes.index < b), [pair["x"], pair["y"]]
                ].dropna()
            )
            x, y = sample.iloc[:, 0].to_numpy(), sample.iloc[:, 1].to_numpy()
            beta = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
            error = abs(beta - pair["beta"])
            max_beta_error = max(error, max_beta_error)
            expected_p = [
                coint(y, x, trend="c", maxlag=1, autolag=None)[1],
                coint(x, y, trend="c", maxlag=1, autolag=None)[1],
            ]
            if error > 1e-10 or not np.allclose(
                expected_p, pair["eg_pvalues"], rtol=1e-10, atol=1e-14
            ):
                raise ValueError("Independent OLS/EG reconstruction differs")
            if pd.Timestamp(pair["last_observation"]) + pd.Timedelta(days=1) > b:
                raise ValueError("Incomplete formation observation")
            checked += 1
    profile = []
    rejections = Counter()
    for name, fit in [("fixed_formation.json", fixed), *fits]:
        reasons = Counter(t["rejection"] for t in fit["tests"] if t["rejection"])
        rejections.update(reasons)
        profile.append(
            {
                "file": name,
                "fit_at": fit["fit_at"],
                "eligible_names": len(fit["eligible_names"]),
                "pairs_tested": fit["pairs_tested"],
                "directional_tests": fit["directional_tests"],
                "unknown_tests": fit["unknown_tests"],
                "selected": [t["x"] + "/" + t["y"] for t in fit["selected"]],
                "rejections": dict(reasons),
            }
        )
    save(
        root / "summary.json",
        {
            "schema": "cointegrated-pair-formation-evidence.v1",
            "profile": profile,
            "formation_fits": len(profile),
            "fixed_pairs": len(fixed["selected"]),
            "rolling_months_with_pairs": sum(bool(f["selected"]) for _, f in fits),
            "directional_tests": sum(f["directional_tests"] for _, f in [("", fixed), *fits]),
            "unknown_tests": sum(f["unknown_tests"] for _, f in [("", fixed), *fits]),
            "rejections": dict(rejections),
            "causality_checks": checks,
            "independent_selected_reconstructions": checked,
            "maximum_beta_error": max_beta_error,
            "account_returns_computed": False,
            "profit_proven": False,
            "production_eligible": False,
            "orders_enabled": False,
            "independent_oos": False,
            "limitations": [
                "Contemporarily chosen 40-name pool has survivor-selection bias.",
                "Each month controls its declared directional family, not lifetime research false discoveries; BH dependence assumptions are not proved.",
                "Daily formation and theoretical convergence do not demonstrate executable intraday returns or stable future relationships.",
                "No order-book history; subsequent capacity analysis is a volume proxy.",
            ],
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    files = [
        "plan.json",
        "registration.json",
        "summary.json",
        "fixed_formation.json",
        *[n for n, _ in fits],
    ]
    save(
        root / "completion.json",
        {
            "manifest": {n: sha(root / n) for n in files},
            "summary_sha256": sha(root / "summary.json"),
        },
    )
    print(
        {
            "root": str(root),
            "formations": len(profile),
            "fixed_pairs": len(fixed["selected"]),
            "rolling_months_with_pairs": sum(bool(f["selected"]) for _, f in fits),
            "checks": len(checks),
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output.resolve())
