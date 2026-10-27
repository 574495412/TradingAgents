"""Separate calendar/rank checks and Decimal cash arithmetic; internal, not third party."""

import argparse
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.audit_final_cohort import prepare_raw
from research_tools.sparse_market.identity import crypto_context


def audit(readiness, inputs, crosscheck, output):
    if output.exists():
        raise FileExistsError("Preserve the executed audit")
    for root in [readiness, inputs]:
        verify_manifest(root)
    for path in [readiness / "registration.json", inputs / "registration.json"]:
        for name, digest in read(path)["source_hashes"].items():
            assert sha(Path(name)) == digest, name
    plan = read(readiness / "plan.json")
    cohort = Path(plan["cohort"])
    registry = read(cohort / "registry.json")
    versions = prepare_raw(read(cohort / "plan.json"), cohort)
    memberships = read(readiness / "monthly_membership.json")
    observations = read(inputs / "observations.json")
    assert not read(inputs / "unknown_observations.json")
    case = read(crosscheck / "DEXEUSDT_raw.json")
    assert not case["response"]["isError"]
    rows = case["response"]["structuredContent"]["result"]
    actual = pd.read_parquet(inputs / "funding/DEXEUSDT_exact.parquet")
    first = pd.to_datetime(case["request"]["startTime"], unit="ms", utc=True)
    last = pd.to_datetime(case["request"]["endTime"], unit="ms", utc=True)
    subset = actual.loc[(actual.index >= first) & (actual.index <= last)]
    assert len(rows) == len(subset) < case["request"]["limit"]
    assert subset.index.as_unit("ms").asi8.tolist() == [r["fundingTime"] for r in rows]
    assert subset.funding_rate.tolist() == [float(r["fundingRate"]) for r in rows]
    assert subset.mark.tolist() == [float(r["markPrice"]) for r in rows]
    cash_by_symbol = {}
    for s in read(readiness / "summary.json")["selected_union"]:
        f = pd.read_parquet(inputs / "funding" / (s + "_exact.parquet"))
        cash = {}
        for at, row in f.iterrows():
            day = at.floor("D")
            cash[day] = cash.get(day, Decimal(0)) + Decimal(str(row.mark)) * Decimal(
                str(row.funding_rate)
            )
        cash_by_symbol[s] = cash
    rank_cases, price_cases = 0, 0
    max_return_error = max_vol_error = max_daily_error = 0.0
    observed_keys = {(r["variant"], r["symbol"], r["decision"]): r for r in observations}
    for variant, raw in versions.items():
        dates_valid = {}
        for s, f in raw.items():
            dates_valid[s] = set(
                f.index[
                    f.price_liquidity_feature_usable.eq(True)
                    & f.volume.gt(0)
                    & f.quote_volume.gt(0)
                    & f.close.gt(0)
                    & np.isfinite(f.close)
                ]
            ) - {f.index[0]}
        for m in memberships:
            if m["variant"] != variant:
                continue
            clock = pd.Timestamp(m["decision"])
            begin = clock - pd.DateOffset(months=12)
            expected = pd.date_range(
                begin - pd.Timedelta(days=1), clock, freq="D", inclusive="left"
            )
            eligible = []
            for s, f in raw.items():
                c = crypto_context(registry, s, clock)
                if c["blocked"] or (
                    c["feature_start"] and pd.Timestamp(c["feature_start"]) > expected[0]
                ):
                    continue
                if not set(expected).issubset(dates_valid[s]):
                    continue
                quotes = f.loc[expected[-28:], "quote_volume"].tolist()
                quote = math.fsum(quotes) / 28
                if quote >= 50_000_000:
                    eligible.append((s, quote))
            selected = [s for s, _ in sorted(eligible, key=lambda r: (-r[1], r[0]))[:10]]
            assert selected == m["selected"] and len(eligible) == m["eligible"]
            rank_cases += 1
            for s in selected:
                o = observed_keys[(variant, s, m["decision"])]
                f = raw[s]
                index = pd.read_parquet(inputs / "daily_indexes" / f"{variant}_{s}.parquet")
                product, returns = Decimal(1), []
                with localcontext() as context:
                    context.prec = 40
                    for day in expected[1:]:
                        previous = Decimal(str(f.at[day - pd.Timedelta(days=1), "close"]))
                        close = Decimal(str(f.at[day, "close"]))
                        assert day in cash_by_symbol[s]
                        r = (close - previous - cash_by_symbol[s][day]) / previous
                        assert Decimal(1) + r > 0
                        product *= 1 + r
                        returns.append(float(r))
                        error = abs(float(r) - float(index.at[day, "economic_return"]))
                        max_daily_error = max(max_daily_error, error)
                    own = float(product - 1)
                volatility = float(
                    math.sqrt(
                        pd.Series(returns).ewm(com=60, adjust=True).var(bias=True).iloc[-1] * 365
                    )
                )
                max_return_error = max(max_return_error, abs(own - o["own_return_12months"]))
                max_vol_error = max(max_vol_error, abs(volatility - o["annual_volatility"]))
                assert int(np.sign(own)) == o["direction"]
                price = float(f.at[expected[-1], "close"] / f.at[expected[0], "close"] - 1)
                assert abs(price - o["price_only_return_reference"]) < 1e-12
                price_cases += 1
    assert rank_cases == 40 and price_cases == len(observations) == 400
    assert max_return_error < 1e-8 and max_vol_error < 1e-9 and max_daily_error < 1e-12
    left = [r for r in observations if r["variant"] == "daily_archive"]
    right = [r for r in observations if r["variant"] != "daily_archive"]
    fields = ["symbol", "decision", "own_return_12months", "annual_volatility", "direction"]
    equivalent = [{k: r[k] for k in fields} for r in left] == [
        {k: r[k] for k in fields} for r in right
    ]
    result = {
        "schema": "monthly-tsmom-internal-input-audit.v1",
        "passed": True,
        "readiness_completion_sha256": sha(readiness / "completion.json"),
        "inputs_completion_sha256": sha(inputs / "completion.json"),
        "source_sha256": sha(Path(__file__)),
        "crosscheck_raw_sha256": sha(crosscheck / "DEXEUSDT_raw.json"),
        "calendar_and_liquidity_rank_cases": rank_cases,
        "Decimal_cash_index_and_separate_EWM_cases": price_cases,
        "maximum_daily_return_error": max_daily_error,
        "maximum_compounded_return_error": max_return_error,
        "maximum_volatility_error": max_vol_error,
        "source_observations_equal": equivalent,
        "unique_coin_month_observations": len(left),
        "unique_price_vs_economic_direction_differences": sum(
            r["direction_differs_from_price"] for r in left
        ),
        "DEXE_sample_current_API_events_exactly_match_sealed": len(rows),
        "shared_dependencies": "Source daily parsing/reconciliation and identity-context policy are shared; calendar membership and rank, Decimal cash and pandas EWM math are separately recomputed.",
        "new_financial_accounts": 0,
        "alpha_proven": False,
        "external_review": False,
        "historical_schedule_certified": False,
        "customer_cash_certified": False,
        "independent_oos": False,
    }
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readiness", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--crosscheck", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.readiness, a.inputs, a.crosscheck, a.output)
