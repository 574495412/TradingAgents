"""Internal raw-prefix arithmetic check, separate from the cohort implementation.

Uses the frozen lifecycle resolver and shared source calendar, so this is not an
independent historical-identity certification or third-party review.
"""

import argparse
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest
from research_tools.sparse_market.identity import crypto_context


def audit(root, output):
    if output.exists():
        raise FileExistsError("Preserve prior audit")
    verify_manifest(root)
    plan = read(root / "plan.json")
    for name, expected in read(root / "registration.json")["source_hashes"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Registered input changed: " + name)
    cohort = Path(plan["cohort"])
    registry = read(cohort / "registry.json")
    market = Path(read(cohort / "plan.json")["parent_market"])
    originals = {
        p.name.removesuffix("_1d.parquet"): pd.read_parquet(p)
        for p in (Path(plan["daily"]) / "symbols").glob("*_1d.parquet")
    }
    alternate = {s: f.copy() for s, f in originals.items()}
    for row in read(cohort / "reconciliation_days.json"):
        symbol, day = row["symbol"], pd.Timestamp(row["day"])
        bars = pd.read_parquet(
            market / "symbols" / (symbol + "_5m.parquet"),
            filters=[("date", ">=", day), ("date", "<", day + pd.Timedelta(days=1))],
        )
        assert len(bars) == 288 and bars.price_usable.all()
        volume = math.fsum(bars.volume)
        quote = math.fsum(bars.quote_volume)
        low, high = min(bars.low), max(bars.high)
        valid = bool(
            volume > 0
            and quote > 0
            and low > 0
            and high >= max(bars.open.iloc[0], bars.close.iloc[-1])
            and low <= min(bars.open.iloc[0], bars.close.iloc[-1])
            and low * volume - 0.01 - 1e-6 * abs(high * volume)
            <= quote
            <= high * volume + 0.01 + 1e-6 * abs(high * volume)
        )
        assert valid == row["reconciled_usable"]
        alternate[symbol].at[day, "volume"] = volume
        alternate[symbol].at[day, "quote_volume"] = quote
        alternate[symbol].at[day, "price_liquidity_feature_usable"] = valid
    checks, max_quote_error, rows_checked = 0, 0.0, 0
    selected_sets = {}
    for variant, frames in [
        ("daily_archive", originals),
        ("daily_with_complete_5m_reconciliation", alternate),
    ]:
        vectors = {}
        for symbol, frame in frames.items():
            valid = (
                frame.price_liquidity_feature_usable.eq(True)
                & frame.full_daily_duration.eq(True)
                & frame.volume.gt(0)
                & frame.quote_volume.gt(0)
                & np.isfinite(frame.quote_volume)
            ).to_numpy(copy=True)
            valid[0] = False
            vectors[symbol] = (
                frame.index,
                valid,
                np.r_[0, valid.cumsum()],
                frame.quote_volume.to_numpy(),
            )
        memberships = read(root / (variant + "_membership.json"))
        for item in memberships:
            label = date.fromisoformat(item["formation_label"])
            end = pd.Timestamp(label, tz="UTC") + pd.Timedelta(days=1)
            clock = pd.Timestamp(item["decision_at_assumed"])
            assert clock == end + pd.Timedelta(minutes=5)
            assert (label - date(label.year, 1, 1)).days % 7 == 6 or label == date(
                label.year, 12, 31
            )
            expected = []
            for symbol, (dates, valid, counts, quotes) in vectors.items():
                checks += 1
                context = crypto_context(registry, symbol, clock)
                if context["blocked"]:
                    continue
                stop = dates.searchsorted(end)
                begin = (
                    dates.searchsorted(pd.Timestamp(context["feature_start"]))
                    if context["feature_start"]
                    else 0
                )
                if stop <= begin or counts[stop] - counts[begin] < 180:
                    continue
                window_begin = dates.searchsorted(end - pd.Timedelta(days=28))
                if window_begin < begin or stop - window_begin != 28:
                    continue
                if not dates[window_begin:stop].equals(
                    pd.date_range(end - pd.Timedelta(days=28), end, freq="D", inclusive="left")
                ):
                    continue
                if not valid[window_begin:stop].all():
                    continue
                quote = math.fsum(quotes[window_begin:stop]) / 28
                if quote < 50_000_000:
                    continue
                expected.append(
                    (
                        symbol,
                        quote,
                        int(counts[stop] - counts[begin]),
                        context["active_contract_key"],
                    )
                )
            expected.sort(key=lambda r: (-r[1], r[0]))
            actual = item["all_native_eligible"]
            assert [r[0] for r in expected] == [r["symbol"] for r in actual], (variant, label)
            for want, got in zip(expected, actual, strict=True):
                error = abs(want[1] - got["quote28"])
                max_quote_error = max(max_quote_error, error)
                assert error <= max(0.000001, abs(want[1]) * 1e-12)
                assert want[2] == got["observations"] and want[3] == got["episode"]
                rows_checked += 1
            assert item["selected"] == [r[0] for r in expected[:30]]
            selected_sets[(variant, item["formation_label"])] = set(item["selected"])
        print({"audited_variant": variant, "formations": len(memberships)}, flush=True)
    reference = {
        r["asset"]: r["full_name"]
        for r in read(Path(plan["catalog"]) / "asset_reference.json")["data"]
    }
    for row in read(root / "slots.json"):
        base = row["native_symbol"][:-4].lower()
        assert row["provider_candidate"] == (base if base in reference else None)
        assert (
            not row["historical_identity_verified"]
            and not row["actual_required_date_rows_verified"]
        )
        assert row["selected_top30"] == (
            row["native_symbol"] in selected_sets[(row["variant"], row["formation_label"])]
        )
    result = {
        "passed": True,
        "input_completion_sha256": sha(root / "completion.json"),
        "audit_source_sha256": sha(Path(__file__)),
        "qualification_symbol_clock_cases": checks,
        "eligible_slot_arithmetic_checks": rows_checked,
        "maximum_quote_mean_drift_usdt": max_quote_error,
        "all_native_eligible_order_and_top30_exact": True,
        "shared_lifecycle_and_calendar_limit": True,
        "third_party_review_performed": False,
        "historical_identity_verified": False,
        "new_financial_accounts": 0,
    }
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.input, args.output)
