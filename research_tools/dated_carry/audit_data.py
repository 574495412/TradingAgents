"""Independent CSV/Decimal reconstruction and non-executable paired references."""

import argparse
import csv
import io
import zipfile
from decimal import Decimal
from pathlib import Path

import pandas as pd

from factors.common.dated_carry import basis_reference
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def audit(inputs, route, output):
    output.mkdir(parents=True, exist_ok=False)
    for root in [inputs, route]:
        verify_manifest(root)
    plan = read(inputs / "plan.json")
    for filename, digest in plan["source_hashes"].items():
        assert sha(Path(filename)) == digest, filename
    paths = [
        Path(__file__),
        Path("factors/common/dated_carry.py"),
        inputs / "completion.json",
        route / "completion.json",
    ]
    save(
        output / "registration.json",
        {
            "source_hashes": {str(p): sha(p) for p in paths},
            "new_financial_accounts": 0,
            "canonical_parse": "CSV strings to Decimal then correctly rounded float; original pandas tables retained and all representation differences reported, never silently overwritten.",
        },
    )
    rows, boundaries = [], []
    floating_fields = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_base",
        "taker_quote",
        "ignore",
    ]
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_base",
        "taker_quote",
        "ignore",
    ]
    for contract in plan["contracts"]:
        symbol = contract.split("_")[0]
        base = symbol.removesuffix("USDT")
        parsed = []
        for month in plan["months"]:
            path = inputs / f"{contract}-5m-{month}.zip"
            with zipfile.ZipFile(path) as zipped:
                assert len(zipped.namelist()) == 1
                table = list(
                    csv.reader(
                        io.TextIOWrapper(zipped.open(zipped.namelist()[0]), encoding="utf-8-sig")
                    )
                )
            assert table[0][0] == "open_time"
            for cells in table[1:]:
                assert len(cells) == 12
                record = {
                    k: int(cells[i])
                    if k in ["open_time", "close_time", "trades"]
                    else float(Decimal(cells[i]))
                    for i, k in enumerate(columns)
                }
                parsed.append(record)
        canonical = pd.DataFrame(parsed)
        canonical.index = pd.DatetimeIndex(
            pd.to_datetime(canonical.open_time, unit="ms", utc=True), name="time"
        )
        original = pd.read_parquet(inputs / f"{contract}_5m.parquet")
        assert canonical.index.equals(original.index)
        for field in ["open_time", "close_time", "trades"]:
            assert canonical[field].equals(original[field]), field
        differences = {
            k: {
                "differing_binary64_values": int(canonical[k].ne(original[k]).sum()),
                "maximum_absolute_difference": float((canonical[k] - original[k]).abs().max()),
            }
            for k in floating_fields
        }
        canonical["full_5m_bar"] = canonical.close_time.eq(canonical.open_time + 299999)
        canonical["positive_execution_volume"] = canonical.volume.gt(0) & canonical.quote_volume.gt(
            0
        )
        assert canonical.full_5m_bar.equals(original.full_5m_bar)
        assert canonical.positive_execution_volume.equals(original.positive_execution_volume)
        assert canonical.index.equals(
            pd.date_range(canonical.index[0], canonical.index[-1], freq="5min")
        )
        canonical.to_parquet(output / f"{contract}_canonical_5m.parquet")
        spot = pd.read_parquet(Path(plan["spot_parent"]) / f"{symbol}_spot_5m.parquet").reindex(
            canonical.index
        )
        assert spot[["close", "quote_volume"]].notna().all().all()
        unit = {
            "spot_asset": base,
            "future_asset": base,
            "spot_quote": "USDT",
            "future_quote": "USDT",
        }
        observations = pd.DataFrame(
            [
                basis_reference(s, f, **unit)
                for s, f in zip(spot.close, canonical.close, strict=True)
            ],
            index=canonical.index + pd.Timedelta(minutes=5),
        )
        observations.index.name = "observed_at"
        assert observations.simple_basis_annualized.isna().all()
        assert observations.verified_days_to_expiry.isna().all()
        observations["source_bar_opened_at"] = canonical.index
        observations["both_report_positive_volume"] = (
            canonical.positive_execution_volume.to_numpy()
            & spot.quote_volume.gt(0).to_numpy()
            & spot.volume.gt(0).to_numpy()
        )
        assert (
            observations.source_bar_opened_at + pd.Timedelta(minutes=5) == observations.index
        ).all()
        assert not observations.executable_or_net_return.any()
        observations.to_parquet(output / f"{contract}_basis_reference.parquet")
        daily = observations.loc[(observations.index.hour == 0) & (observations.index.minute == 0)]
        daily.to_parquet(output / f"{contract}_daily_reference.parquet")
        records = read(route / f"{symbol}_delivery.json")
        candidates = [
            r
            for r in records
            if pd.Timestamp(r["deliveryTime"], unit="ms", tz="UTC").strftime("%y%m%d")
            == contract.split("_")[1]
        ]
        assert len(candidates) == 1
        settlement = candidates[0]
        label = pd.Timestamp(settlement["deliveryTime"], unit="ms", tz="UTC")
        assert label == label.floor("D")
        same_date = canonical.loc[canonical.index.floor("D") == label]
        boundary = {
            "contract": contract,
            "settlement_record": settlement,
            "recorded_date_label": label.isoformat(),
            "observed_last_open": canonical.index[-1].isoformat(),
            "observed_last_completion": observations.index[-1].isoformat(),
            "last_reported_close": float(canonical.close.iloc[-1]),
            "last_reported_close_is_settlement_price": float(canonical.close.iloc[-1])
            == settlement["deliveryPrice"],
            "raw_rows_on_date_label_day": len(same_date),
            "raw_rows_at_or_after_midnight_label": int((canonical.index >= label).sum()),
            "positive_volume_rows_at_or_after_midnight_label": int(
                (canonical.positive_execution_volume & (canonical.index >= label)).sum()
            ),
            "midnight_label_certifies_exact_trading_cutoff": False,
            "delivery_fee_or_customer_cash_certified": False,
        }
        boundaries.append(boundary)
        rows.append(
            {
                "contract": contract,
                "source_rows": len(canonical),
                "daily_observations": len(daily),
                "first_observation": observations.index[0].isoformat(),
                "last_observation": observations.index[-1].isoformat(),
                "nonpositive_volume_rows": int((~canonical.positive_execution_volume).sum()),
                "pandas_vs_decimal_representation": differences,
                "reference_basis_bps_quantiles": {
                    str(q): float(observations.simple_basis_total.quantile(q) * 10000)
                    for q in [0, 0.25, 0.5, 0.75, 1]
                },
                "annualized_basis_unknown": True,
                "financial_returns_computed": False,
            }
        )
    source = Path(plan["source_followup"])
    current = read(source / "exchange_info.raw")
    current_rows = [
        s for s in current["symbols"] if s["symbol"].startswith(("BTCUSDT_", "ETHUSDT_"))
    ]
    assert len(current_rows) == 4 and all(
        s["quoteAsset"] == s["marginAsset"] == "USDT" for s in current_rows
    )
    dates = [
        {
            "symbol": s["symbol"],
            "current_deliveryDate": pd.Timestamp(
                s["deliveryDate"], unit="ms", tz="UTC"
            ).isoformat(),
            "current_onboardDate": pd.Timestamp(s["onboardDate"], unit="ms", tz="UTC").isoformat(),
        }
        for s in current_rows
    ]
    save(
        output / "settlement_boundaries.json",
        {
            "historical_records": boundaries,
            "current_metadata_only": dates,
            "interpretation": "Historical deliveryTime values are midnight date labels while current deliveryDate values are08UTC. Do not equate these fields to a verified historical trading cutoff. Archive bars, settlement reference prices and actual client cash are distinct evidence.",
        },
    )
    summary = {
        "schema": "dated-carry-raw-data-audit.v1",
        "passed": True,
        "scope": "Raw data/units/arithmetic references only; actual historical qualification, cutoff clock, fee, margin, fills and net advantage unresolved.",
        "rows": rows,
        "independent_CSV_Decimal_rows": sum(r["source_rows"] for r in rows),
        "official_archives": 6,
        "source_conversions_preserved": True,
        "basis_is_executable_profit": False,
        "annualized_basis_estimated_from_filename": False,
        "new_financial_accounts": 0,
        "payoff_identity_tests_are_strategy_trials": False,
        "production_eligible": False,
        "independent_oos": False,
        "third_party_review": False,
        "profit_goal_met": False,
    }
    save(output / "summary.json", summary)
    save(
        output / "verification.json",
        {
            "passed": True,
            "source_hashes": {str(p): sha(p) for p in paths},
            "proof_scope": summary["scope"],
            "profit_goal_met": False,
        },
    )
    manifest = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    save(
        output / "completion.json", {"passed": True, "manifest": manifest, "profit_goal_met": False}
    )
    print(
        {
            "rows": summary["independent_CSV_Decimal_rows"],
            "contracts": rows,
            "boundaries": boundaries,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--route", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.inputs, a.route, a.output)
