"""Collect actual spot legs and preserve exact funding events before strategy R&D."""

import argparse
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research_tools.factor_lab.io import read, save, sha


def expected_clock(start, end):
    return pd.date_range(start, end, freq="5min", inclusive="left", tz="UTC")


def parse_spot(body, month):
    """Archive units changed in2025; API clocks remain a separate contract."""
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith(".csv"):
            raise ValueError("Unexpected spot archive members")
        frame = pd.read_csv(archive.open(names[0]), header=None)
    if frame.shape[1] != 12:
        raise ValueError("Unexpected raw spot kline schema")
    start = pd.Timestamp(month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin()
    unit = "us" if start.year >= 2025 else "ms"
    index = pd.DatetimeIndex(pd.to_datetime(frame[0], unit=unit, utc=True), name="date")
    if not index.equals(expected_clock(start, end)):
        raise ValueError("Spot missing/duplicate/out-of-order/wrong-unit bars")
    closing = pd.to_datetime(frame[6], unit=unit, utc=True)
    if not np.array_equal(
        closing.array, index + pd.Timedelta(minutes=5) - pd.Timedelta(1, unit=unit)
    ):
        raise ValueError("Spot close clock is not an interval end")
    out = frame[[1, 2, 3, 4, 5, 7]].astype(float)
    out.columns = ["open", "high", "low", "close", "volume", "quote_volume"]
    out.index = index
    if (
        not np.isfinite(out.to_numpy()).all()
        or (out.iloc[:, :4] <= 0).any().any()
        or (out[["volume", "quote_volume"]] < 0).any().any()
        or (out.high < out[["open", "low", "close"]].max(axis=1)).any()
        or (out.low > out[["open", "high", "close"]].min(axis=1)).any()
    ):
        raise ValueError("Spot price/volume integrity failed")
    return out


def get(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def monthly_job(root, plan, symbol, month):
    url = plan["spot_archive"].format(symbol=symbol, month=month)
    checksum = get(url + ".CHECKSUM")
    expected = checksum.decode().split()[0]
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Invalid official checksum")
    body = get(url)
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("Official spot checksum mismatch")
    frame = parse_spot(body, month)
    target = root / "archives" / url.rsplit("/", 1)[1]
    with target.open("xb") as stream:
        stream.write(body)
    target.with_suffix(".zip.CHECKSUM").write_bytes(checksum)
    return frame, {
        "symbol": symbol,
        "month": month,
        "url": url,
        "sha256": expected,
        "rows": len(frame),
    }


def cost_arithmetic(scenario, budget):
    """Two entries and two exits, scaled to equal spot/perp leg notionals."""
    spot = budget["spot_leg_nav_fraction"]
    perp = budget["perpetual_leg_nav_fraction"]
    slip = scenario["slippage_bps_each_leg_each_side"]
    round_trip = (
        2
        * (
            spot * (scenario["spot_commission_bps"] + slip)
            + perp * (scenario["perpetual_commission_bps"] + slip)
        )
        / 10000
    )
    return {
        "scenario": scenario["id"],
        "round_trip_cost_on_initial_nav": round_trip,
        "required_sum_funding_rates_on_short_notional": round_trip / perp,
        "basis_change_excluded": True,
        "executed_account": False,
    }


def prepare(plan_path, root):
    plan = read(plan_path)
    if (root / "plan.json").exists() or (root / "completion.json").exists():
        raise FileExistsError("Do not overwrite a prior or partial feasibility collection")
    raw = read(root / "connector.json")
    if len(raw["funding_pages"]) != plan["budget"]["funding_pages"]:
        raise ValueError("Missing raw funding connector pages")
    root.mkdir(parents=True, exist_ok=True)
    (root / "archives").mkdir(exist_ok=False)
    (root / "sources").mkdir(exist_ok=False)
    plan["source_hashes"] = {str(plan_path): sha(plan_path), __file__: sha(Path(__file__))}
    plan["connector_sha256"] = sha(root / "connector.json")
    save(root / "plan.json", plan)
    for name in ["README.md", "TERMS_AND_CONDITIONS.md"]:
        url = f"https://raw.githubusercontent.com/binance/binance-public-data/{plan['source_commit']}/{name}"
        (root / "sources" / name).write_bytes(get(url))
    parent = Path(plan["parent_perpetual_data"])
    parent_receipt = read(parent / "completion.json")
    months = (
        pd.date_range(plan["start"], plan["end_exclusive"], freq="MS", inclusive="left")
        .strftime("%Y-%m")
        .tolist()
    )
    jobs = [(symbol, month) for symbol in plan["symbols"] for month in months]
    if len(jobs) != plan["budget"]["spot_monthly_files"]:
        raise ValueError("Spot collection budget differs from registration")
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(monthly_job, root, plan, symbol, month) for symbol, month in jobs]
        collected = []
        for number, future in enumerate(pending, 1):
            collected.append(future.result())
            print(f"verified spot archive {number}/{len(jobs)}", flush=True)
    summary = {
        "schema": "spot-perpetual-data-feasibility.v1",
        "strategy_accounts": 0,
        "profit_goal_met": False,
        "portfolio_actual": None,
        "production_eligible": False,
        "independent_oos": False,
        "start": plan["start"],
        "end_exclusive": plan["end_exclusive"],
        "source_commit": plan["source_commit"],
        "source_hashes": plan["source_hashes"],
        "parent_completion_sha256": sha(parent / "completion.json"),
        "archives": [receipt for _, receipt in collected],
        "symbols": [],
        "cost_arithmetic": [
            cost_arithmetic(c, plan["illustrative_budget"]) for c in plan["cost_scenarios"]
        ],
        "funding_only_rate_required_for_20percent_at_fixed_short40percent": 0.2 / 0.4,
        "caveats": [
            "Rate sums and simultaneous close basis are diagnostics, not executed net return, win rate, drawdown, Sharpe or capacity.",
            "Funding-rate sum ignores changing marked notional. Funding-only NAV arithmetic assumes fixed40percent short notional and excludes basis, fees, collateral, price path and financing.",
            "42 archives and3834 funding events concern2fixed pilot symbols, not independent strategy samples or all40coin coverage.",
            "Current generic spot10bps/perp5bps research costs are not historical or account-specific verified fee schedules.",
            "Observed funding cannot be annualized into a future guarantee;2025/2026already viewed and no new OOS.",
        ],
    }
    for symbol in plan["symbols"]:
        spot = pd.concat([f for f, receipt in collected if receipt["symbol"] == symbol])
        clock = expected_clock(plan["start"], plan["end_exclusive"])
        if not spot.index.equals(clock):
            raise ValueError("Full spot clock mismatch")
        spot.to_parquet(root / f"{symbol}_spot_5m.parquet")
        perp_name = f"{symbol}_5m.parquet"
        funding_name = f"{symbol}_funding.parquet"
        for name in [perp_name, funding_name]:
            if sha(parent / name) != parent_receipt["manifest"][name]:
                raise ValueError("Parent perpetual evidence changed")
        perp = pd.read_parquet(parent / perp_name).reindex(clock)
        if perp.isna().any().any():
            raise ValueError("Missing parent perpetual5m data")
        pages = [page for page in raw["funding_pages"] if page["request"]["symbol"] == symbol]
        if len(pages) != len(plan["funding_request_windows"]):
            raise ValueError("Missing funding window")
        for page, (start, end) in zip(pages, plan["funding_request_windows"], strict=True):
            request = page["request"]
            if (
                request["startTime"] != pd.Timestamp(start, tz="UTC").value // 1000000
                or request["endTime"] != pd.Timestamp(end, tz="UTC").value // 1000000 - 1
            ):
                raise ValueError("Funding request window mismatch")
            if not page["rows"] or len(page["rows"]) >= request["limit"]:
                raise ValueError("Funding page empty or potentially truncated")
            if any(
                not request["startTime"] <= row["fundingTime"] <= request["endTime"]
                or row["symbol"] != symbol
                for row in page["rows"]
            ):
                raise ValueError("Funding response outside request scope")
        records = [row for page in pages for row in page["rows"]]
        funding = pd.DataFrame(records)
        funding.index = pd.DatetimeIndex(
            pd.to_datetime(funding.fundingTime, unit="ms", utc=True), name="date"
        )
        funding["funding_rate"] = pd.to_numeric(funding.fundingRate)
        funding["mark"] = pd.to_numeric(funding.markPrice)
        if (
            not funding.index.is_monotonic_increasing
            or funding.index.has_duplicates
            or not np.isfinite(funding[["funding_rate", "mark"]].to_numpy()).all()
            or (funding.mark <= 0).any()
        ):
            raise ValueError("Funding integrity failed")
        # Compare historical rates only. Exact connector settlement times are preserved.
        old = pd.read_parquet(parent / funding_name)
        old = old.loc[(old.index >= clock[0]) & (old.index < pd.Timestamp(plan["end_exclusive"]))]
        nominal_clock = funding.index.floor("h")
        if not nominal_clock.equals(old.index) or not np.allclose(
            funding.funding_rate, old.funding_rate, rtol=0, atol=1e-12
        ):
            raise ValueError("Actual funding rates/calendar differ from sealed parent")
        funding.to_parquet(root / f"{symbol}_funding_exact.parquet")
        basis = perp.close / spot.close - 1
        info = {
            "symbol": symbol,
            "spot_rows": len(spot),
            "perpetual_rows": len(perp),
            "funding_events": len(funding),
            "funding_mark_source": "actual public connector record markPrice",
            "funding_rate_matches_parent": True,
            "maximum_settlement_offset_ms_from_nominal_hour": float(
                (funding.index - nominal_clock).total_seconds().max() * 1000
            ),
            "spot_time_units": {"2024-12": "ms", "2025onwards": "us"},
            "simultaneous_close_basis_quantiles": {
                str(q): float(basis.quantile(q)) for q in [0, 0.01, 0.5, 0.99, 1]
            },
            "funding_rate_by_year": [],
            "parent_hashes": {name: sha(parent / name) for name in [perp_name, funding_name]},
        }
        for year in [2024, 2025, 2026]:
            part = funding.loc[funding.index.year == year]
            days = int((clock.year == year).sum() / 288)
            rate_sum = float(part.funding_rate.sum())
            info["funding_rate_by_year"].append(
                {
                    "year": year,
                    "calendar_days": days,
                    "events": len(part),
                    "negative_events": int((part.funding_rate < 0).sum()),
                    "sum_rates": rate_sum,
                    "simple_annual_rate_on_fixed_short_notional": rate_sum * 365 / days,
                    "funding_only_simple_nav_contribution_at_fixed40percent": rate_sum * 0.4,
                    "executed_return": None,
                }
            )
        summary["symbols"].append(info)
    for probe in raw["availability_probe"]:
        if probe["name"] not in ("spot", "perpetual"):
            continue
        symbol = "BTCUSDT"
        bars = probe["response"]["result"]
        table = (
            pd.read_parquet(root / f"{symbol}_spot_5m.parquet")
            if probe["name"] == "spot"
            else pd.read_parquet(parent / f"{symbol}_5m.parquet")
        )
        for row in bars:
            when = pd.Timestamp(row[0], unit="ms", tz="UTC")
            if not np.allclose(
                table.loc[
                    when, ["open", "high", "low", "close", "volume", "quote_volume"]
                ].to_numpy(dtype=float),
                np.array([row[i] for i in [1, 2, 3, 4, 5, 7]], dtype=float),
                rtol=1e-12,
                atol=1e-8,
            ):
                raise ValueError("Connector raw5m sample disagrees with official archive")
    summary["raw_api_sample_checks"] = {
        "spot_bars": 2,
        "perpetual_bars": 2,
        "passed": True,
        "api_clock_unit": "ms",
    }
    save(root / "summary.json", summary)
    files = [root / name for name in ["plan.json", "connector.json", "summary.json"]]
    files += list((root / "sources").iterdir()) + list(root.glob("*.parquet"))
    save(
        root / "completion.json",
        {
            "manifest": {str(p.relative_to(root)): sha(p) for p in files},
            "archives": len(collected),
            "financial_accounts": 0,
            "profit_goal_met": False,
        },
    )
    print(
        json.dumps(
            {
                "complete": True,
                "spot_archives": len(collected),
                "symbols": len(summary["symbols"]),
                "financial_accounts": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("research_tools/carry_feasibility/plan.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.plan, args.output)
