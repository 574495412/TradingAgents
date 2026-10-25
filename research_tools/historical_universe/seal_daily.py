"""Independent raw CSV audit; preserve suspect candles and preview past-only eligibility."""

import argparse
import csv
import hashlib
import io
import math
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from research_tools.factor_lab.io import read, save, sha


def independent_parse(payload, symbol, month):
    """Keep source values, but separate essential field usability from auxiliary errors."""
    expected = f"{symbol}-1d-{month}.csv"
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if z.namelist() != [expected] or z.getinfo(expected).file_size > 2_000_000 or z.testzip():
            raise ValueError("ZIP path, size or CRC invalid")
        rows = list(csv.reader(io.StringIO(z.read(expected).decode("utf-8-sig"))))
    if rows and rows[0][0] == "open_time":
        rows = rows[1:]
    if not 1 <= len(rows) <= 31:
        raise ValueError("Monthly daily count invalid")
    parsed = []
    for number, r in enumerate(rows):
        if len(r) != 12:
            raise ValueError("Daily CSV width invalid")
        epoch = int(r[0])
        unit = "us" if epoch > 100_000_000_000_000 else "ms"
        t = pd.Timestamp(epoch, unit=unit, tz="UTC")
        closing = pd.Timestamp(int(r[6]), unit=unit, tz="UTC")
        if (
            t.strftime("%Y-%m") != month
            or t != t.floor("D")
            or not t <= closing < t + pd.Timedelta(days=1)
        ):
            raise ValueError("UTC daily or month bounds invalid")
        values = {
            name: float(r[i])
            for i, name in [
                (1, "open"),
                (2, "high"),
                (3, "low"),
                (4, "close"),
                (5, "volume"),
                (7, "quote_volume"),
                (8, "trades"),
                (9, "taker_base_volume"),
                (10, "taker_quote_volume"),
            ]
        }
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Nonfinite raw source")
        o, h, low, c, v, q, n, tb, tq = (
            values[k]
            for k in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades",
                "taker_base_volume",
                "taker_quote_volume",
            ]
        )
        flags = []
        price_ok = min(o, h, low, c) > 0 and h >= max(o, low, c) and low <= min(o, h, c)
        if not price_ok:
            flags.append("invalid_OHLC")
        quote_ok = (
            v >= 0
            and q >= 0
            and q >= low * v - max(0.01, 1e-6 * max(q, h * v))
            and q <= h * v + max(0.01, 1e-6 * max(q, h * v))
        )
        if not quote_ok:
            flags.append("quote_outside_OHLC_volume_bounds")
        auxiliary_ok = (
            n >= 0
            and n == math.floor(n)
            and tb >= 0
            and tq >= 0
            and tb <= v + max(1e-6, 1e-8 * abs(v))
            and tq <= q + max(0.01, 1e-8 * abs(q))
        )
        if not auxiliary_ok:
            flags.append("auxiliary_trade_or_taker_inconsistency")
        full = closing == t + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        if not full:
            flags.append("partial_daily_interval")
        parsed.append(
            {
                "date": t,
                **values,
                "close_time": closing,
                "ohlc_valid": price_ok,
                "quote_volume_valid": quote_ok,
                "auxiliary_valid": auxiliary_ok,
                "full_daily_duration": full,
                "price_liquidity_feature_usable": price_ok and quote_ok and full,
                "quality_flags": "|".join(flags),
                "source_month": month,
                "source_csv_row": number + 2,
            }
        )
    frame = pd.DataFrame(parsed).set_index("date")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Duplicate or reversed raw daily timestamps")
    return frame


def checksum_bytes(root, row):
    name = row["key"].rsplit("/", 1)[-1]
    path = root / row["symbol"] / name
    content = path.read_bytes()
    check = path.with_name(path.name + ".CHECKSUM").read_text().split()
    if (
        len(check) != 2
        or check[1].lstrip("*") != name
        or check[0].lower() != hashlib.sha256(content).hexdigest()
        or len(content) != row["size"]
    ):
        raise ValueError("Official checksum or catalog bytes differ")
    return content, path


def seal(catalog, collection, output):  # noqa: C901
    output.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("plan.json")
    plan = read(plan_path)
    raw_summary = read(collection / "download_summary.json")
    records = read(catalog / "archives.json")
    journal = [
        __import__("json").loads(line)
        for line in (collection / "downloads.jsonl").read_text().splitlines()
    ]
    if (
        sha(collection / "downloads.jsonl") != raw_summary["journal_sha256"]
        or len(journal) != len(records)
        or len({r["key"] for r in journal}) != len(records)
        or {r["key"] for r in journal} != {r["key"] for r in records}
    ):
        raise ValueError("Raw download terminal journal incomplete")
    for name, digest in read(catalog / "completion.json")["manifest"].items():
        if (
            catalog.resolve() not in (catalog / name).resolve().parents
            or sha(catalog / name) != digest
        ):
            raise ValueError("Catalog changed")
    raw_plan = read(collection / "plan.json")
    if raw_plan["catalog_completion_sha256"] != sha(catalog / "completion.json") or raw_plan[
        "source_sha256"
    ] != sha(Path(__file__).with_name("ingest.py")):
        raise ValueError("Original ingestion source or catalog changed")
    policy = {
        "schema": "independent-historical-daily-seal.v1",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "before_eligibility_preview": True,
        "catalog_completion_sha256": sha(catalog / "completion.json"),
        "raw_terminal_summary_sha256": sha(collection / "download_summary.json"),
        "raw_journal_sha256": sha(collection / "downloads.jsonl"),
        "source_code_sha256": sha(Path(__file__)),
        "plan_source_sha256": sha(plan_path),
        "quality_policy": "Retain all raw fields. Exclude suspect OHLC/quote-volume/partial-day observations from price/liquidity feature preview; auxiliary-only defects are flagged and excluded from auxiliary features. No price imputation, clipping or future-outcome membership filter.",
        "quote_rounding_tolerance": "max0.01USDT,1e-6relative to quote or high*volume; mathematical volume-weighted price bounds, not fit to returns",
        "maximum_supplement_archives": 50,
        "financial_accounts": 0,
    }
    save(output / "plan.json", policy)
    failures = [r for r in journal if not r["passed"]]
    if len(failures) > policy["maximum_supplement_archives"]:
        raise ValueError("Too many raw anomalies for registered supplement budget")
    supplements = output / "supplement_raw"
    supplements.mkdir()
    by_key = {r["key"]: r for r in records}
    supplements_receipts = []
    for failure in failures:
        row = by_key[failure["key"]]
        name = row["key"].rsplit("/", 1)[-1]
        dest = supplements / row["symbol"]
        dest.mkdir(exist_ok=True)
        for suffix, key in [("", row["key"]), (".CHECKSUM", row["checksum_key"])]:
            r = requests.get(plan["download_origin"] + "/" + key, timeout=30)
            r.raise_for_status()
            with (dest / (name + suffix)).open("xb") as f:
                f.write(r.content)
        _, path = checksum_bytes(supplements, row)
        supplements_receipts.append(
            {
                "key": row["key"],
                "original_error": failure["error"],
                "zip_sha256": sha(path),
                "checksum_sha256": sha(path.with_name(path.name + ".CHECKSUM")),
                "extra_http_requests": 2,
            }
        )
    save(output / "supplements.json", supplements_receipts)
    by_symbol = {}
    for r in records:
        by_symbol.setdefault(r["symbol"], []).append(r)
    (output / "symbols").mkdir()
    issue_rows, quality, bindings = [], [], {}
    frames = {}
    for i, (symbol, parts) in enumerate(sorted(by_symbol.items())):
        combined = []
        for row in sorted(parts, key=lambda r: r["month"]):
            root = supplements if row["key"] in {f["key"] for f in failures} else collection / "raw"
            payload, path = checksum_bytes(root, row)
            bindings[str(path)] = sha(path)
            bindings[str(path.with_name(path.name + ".CHECKSUM"))] = sha(
                path.with_name(path.name + ".CHECKSUM")
            )
            combined.append(independent_parse(payload, symbol, row["month"]))
        frame = pd.concat(combined)
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("Merged daily source duplicates or calendar reverse")
        gaps = pd.date_range(frame.index[0], frame.index[-1], freq="D").difference(frame.index)
        for t, r in frame.loc[frame.quality_flags.ne("")].iterrows():
            issue_rows.append(
                {
                    "symbol": symbol,
                    "date": t.isoformat(),
                    "flags": r.quality_flags,
                    "feature_usable": bool(r.price_liquidity_feature_usable),
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "quote_volume": r.quote_volume,
                    "taker_base_volume": r.taker_base_volume,
                    "taker_quote_volume": r.taker_quote_volume,
                }
            )
        for t in gaps:
            issue_rows.append(
                {
                    "symbol": symbol,
                    "date": t.isoformat(),
                    "flags": "missing_interior_source_day",
                    "feature_usable": False,
                }
            )
        frame.to_parquet(output / "symbols" / (symbol + "_1d.parquet"))
        frames[symbol] = frame
        quality.append(
            {
                "symbol": symbol,
                "rows": len(frame),
                "first": frame.index[0].isoformat(),
                "last": frame.index[-1].isoformat(),
                "interior_missing_days": len(gaps),
                "flagged_source_rows": int(frame.quality_flags.ne("").sum()),
                "price_liquidity_unusable_rows": int((~frame.price_liquidity_feature_usable).sum()),
                "zero_quote_volume_days": int(frame.quote_volume.eq(0).sum()),
            }
        )
        if (i + 1) % 100 == 0:
            print({"independently_parsed_symbols": i + 1, "of": len(by_symbol)}, flush=True)
    save(output / "source_bindings.json", bindings)
    save(output / "issues.json", issue_rows)
    save(output / "symbol_quality.json", quality)
    # This preview is a past-only candidate set, not a delisting-aware execution account.
    calendar = pd.date_range(
        plan["start"], plan["end_exclusive"], freq="D", inclusive="left", tz="UTC"
    )
    decisions = calendar[calendar.weekday == plan["eligibility_preview"]["decision_weekday_utc"]]
    metadata_path = Path(plan["source_directory"]) / "exchange_info.json"
    current = {r["symbol"]: r for r in read(metadata_path)["symbols"]}
    old40 = set(
        read(Path("data/research/github_factor_market_20261025_v001/prepare_plan.json"))["symbols"]
    )
    per_symbol = {}
    for s, f in frames.items():
        usable = f.price_liquidity_feature_usable.reindex(calendar, fill_value=False)
        quote = f.quote_volume.reindex(calendar).where(usable)
        mean = (
            quote.rolling(
                plan["eligibility_preview"]["lookback_completed_days"],
                min_periods=plan["eligibility_preview"]["lookback_completed_days"],
            )
            .mean()
            .shift(1)
        )
        observed = usable.astype(int).cumsum().shift(1)
        per_symbol[s] = pd.DataFrame({"quote28": mean, "observed": observed}).loc[decisions]
    preview = []
    for t in decisions:
        eligible = []
        for s, f in per_symbol.items():
            r = f.loc[t]
            if (
                pd.notna(r.quote28)
                and r.quote28 >= plan["eligibility_preview"]["minimum_mean_daily_quote_volume"]
                and r.observed >= plan["eligibility_preview"]["minimum_observed_days"]
            ):
                eligible.append((s, float(r.quote28)))
        ranked = sorted(eligible, key=lambda r: (-r[1], r[0]))[
            : plan["eligibility_preview"]["maximum_names"]
        ]
        preview.append(
            {
                "decision": t.isoformat(),
                "eligible_names": len(eligible),
                "selected": [s for s, _ in ranked],
                "selected_quote28": dict(ranked),
                "selected_outside_old40": [s for s, _ in ranked if s not in old40],
                "selected_missing_current_metadata": [s for s, _ in ranked if s not in current],
                "selected_current_noncoin": [
                    s
                    for s, _ in ranked
                    if (current.get(s) or {}).get("underlyingType") not in (None, "COIN")
                ],
                "historical_trading_status_certified": False,
            }
        )
    save(output / "eligibility_preview.json", preview)
    selected_union = sorted({s for p in preview for s in p["selected"]})
    summary = {
        "schema": "historical-perpetual-daily-independent.v1",
        "raw_archives": len(records),
        "symbols": len(frames),
        "rows": sum(r["rows"] for r in quality),
        "source_flags": dict(Counter(flag for r in issue_rows for flag in r["flags"].split("|"))),
        "flagged_rows_and_missing_days": len(issue_rows),
        "preview_decisions": len(preview),
        "preview_selected_union": selected_union,
        "preview_selected_outside_old40": sorted(set(selected_union) - old40),
        "preview_selected_missing_current_metadata": sorted(
            s for s in selected_union if s not in current
        ),
        "current_metadata_sha256": sha(metadata_path),
        "raw_ingestion_errors_preserved": len(failures),
        "supplement_http_requests": 2 * len(failures),
        "raw_checksums_recomputed": True,
        "all_source_rows_valid": not issue_rows,
        "eligible_for_financial_accounts": False,
        "missing_prices_imputed": False,
        "tail_returns_clipped": False,
        "future_return_filter_used": False,
        "historical_asset_type_and_final_settlement_certified": False,
        "current_metadata_used_to_filter_history": False,
        "financial_accounts": 0,
        "profit_goal_met": False,
        "production_eligible": False,
        "orders_enabled": False,
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "raw_independent_audit_complete": True,
            "financial_accounts": 0,
        },
    )
    print({k: v for k, v in summary.items() if not isinstance(v, list)}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    seal(a.catalog, a.collection, a.output)
