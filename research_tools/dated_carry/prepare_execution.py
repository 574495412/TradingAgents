"""Verify rule timing and raw marks, then freeze one dated-carry execution pilot."""

import argparse
import csv
import io
import zipfile
from decimal import Decimal
from pathlib import Path

import pandas as pd

from factors.crypto.dated_carry_schedule import schedule
from research_tools.factor_lab.io import read, save, sha
from research_tools.max_effect.run_accounts import verify_manifest


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("account_plan.json")
    p = read(source)
    parents = [
        Path(p[k])
        for k in ["risk_sources", "raw_pilot", "raw_audit", "settlement_source", "spot_source"]
    ]
    for root in parents:
        verify_manifest(root)
    risk = Path(p["risk_sources"])
    summary = read(risk / "summary.json")
    assert summary["all_documents_available"] and summary["all_market_archives_available"]
    for filename, digest in read(risk / "plan.json")["source_hashes"].items():
        assert sha(Path(filename)) == digest
    listing = (risk / "f2338846942d48daa4f9395173727f4c.txt").read_text()
    confirm = (risk / "03d4fc147cd5421681219eac7ac8eea9.txt").read_text()
    generic = (risk / "a3401595e1734084959c61491bc0dbe3.txt").read_text()
    assert "BTCUSDT Quarterly 0328" in listing and "ETHUSDT Quarterly 0328" in listing
    assert "2025-03-28 08:00 (UTC)" in confirm and "10 minutes" in confirm
    assert "08:00:00 UTC" in generic and "taker fee" in generic
    documents = {r["id"]: r for r in summary["documents"]}
    launch_known = pd.Timestamp(
        documents["f2338846942d48daa4f9395173727f4c"]["publishDate"], unit="ms", tz="UTC"
    ) + pd.Timedelta(minutes=5)
    confirm_known = pd.Timestamp(
        documents["03d4fc147cd5421681219eac7ac8eea9"]["publishDate"], unit="ms", tz="UTC"
    ) + pd.Timedelta(minutes=5)
    assert (
        launch_known
        < pd.Timestamp(p["window"]["start"])
        < confirm_known
        < pd.Timestamp("2025-03-28T08:00Z")
    )
    raw_checks = []
    for r in summary["market_archives"]:
        filename = f"{r['contract']}-{r['kind']}-5m-{r['month']}.zip"
        with zipfile.ZipFile(risk / filename) as zipped:
            table = list(
                csv.reader(
                    io.TextIOWrapper(zipped.open(zipped.namelist()[0]), encoding="utf-8-sig")
                )
            )
        if table[0][0] == "open_time":
            table = table[1:]
        times = pd.DatetimeIndex(
            pd.to_datetime([int(row[0]) for row in table], unit="ms", utc=True)
        )
        frame = pd.read_parquet(risk / f"{r['contract']}_{r['kind']}_5m.parquet").reindex(times)
        fields = [(k, i) for i, k in enumerate(["open", "high", "low", "close"], 1)]
        if r["kind"] == "trade":
            fields += [("volume", 5), ("quote_volume", 7)]
        for field, column in fields:
            assert frame[field].tolist() == [float(Decimal(row[column])) for row in table], (
                filename,
                field,
            )
        assert all(int(row[6]) == int(row[0]) + 299999 for row in table)
        assert times.equals(pd.date_range(times[0], times[-1], freq="5min"))
        raw_checks.append(
            {"file": filename, "rows": len(table), "fields": len(fields), "CSV_Decimal_exact": True}
        )
    begin, end = [pd.Timestamp(p["window"][k]) for k in ["start", "end"]]
    clocks = pd.date_range(
        begin - pd.Timedelta(minutes=5), end - pd.Timedelta(minutes=5), freq="5min"
    )
    spot, futures, contracts, coverage = {}, {}, {}, []
    for s, c in p["contracts"].items():
        contract = c["symbol"]
        records = read(Path(p["settlement_source"]) / f"{s}_delivery.json")
        relevant = [
            r
            for r in records
            if pd.Timestamp(r["deliveryTime"], unit="ms", tz="UTC").strftime("%y%m%d")
            == contract.split("_")[1]
        ]
        assert (
            len(relevant) == 1
            and pd.Timestamp(relevant[0]["deliveryTime"], unit="ms", tz="UTC").date()
            == pd.Timestamp(c["expiry"]).date()
        )
        contracts[s] = {
            **c,
            "known_at": launch_known.isoformat(),
            "confirmation_known_at": confirm_known.isoformat(),
            "settlement_price": relevant[0]["deliveryPrice"],
            "settlement_source_sha256": sha(Path(p["settlement_source"]) / f"{s}_delivery.json"),
            "original_historical_rule_body_certified": False,
        }
        spot[s] = pd.read_parquet(Path(p["spot_source"]) / f"{s}_spot_5m.parquet").reindex(clocks)
        trade = pd.concat(
            [
                pd.read_parquet(risk / f"{contract}_trade_5m.parquet"),
                pd.read_parquet(Path(p["raw_audit"]) / f"{contract}_canonical_5m.parquet"),
            ]
        ).sort_index()
        assert not trade.index.has_duplicates
        mark = pd.read_parquet(risk / f"{contract}_mark_5m.parquet")
        expiry = pd.Timestamp(c["expiry"])
        active = clocks < expiry
        assert spot[s][["close", "quote_volume"]].notna().all().all()
        f = trade.reindex(clocks)[["open", "high", "low", "close", "quote_volume"]]
        m = mark.reindex(clocks)[["open", "high", "low", "close"]]
        assert f.loc[active].notna().all().all() and m.loc[active].notna().all().all()
        late_trade, late_mark = trade.index >= expiry, mark.index >= expiry
        f.loc[~active] = float("nan")
        m.loc[~active] = float("nan")
        for kind, frame in [("spot", spot[s]), ("future", f), ("mark", m)]:
            frame.to_parquet(output / f"{s}_{kind}.parquet")
        pd.DataFrame(
            columns=["funding_rate", "mark"], index=pd.DatetimeIndex([], tz="UTC")
        ).to_parquet(output / f"{s}_funding.parquet")
        futures[s] = f
        coverage.append(
            {
                "asset": s,
                "contract": contract,
                "active_five_minute_intervals": int(sum(active)),
                "post_expiry_trade_source_rows_ignored": int(sum(late_trade)),
                "post_expiry_mark_source_rows_ignored": int(sum(late_mark)),
                "later_old_future_prices_are_null": True,
                "source_08bar_is_settlement": False,
            }
        )
    p["contracts"] = contracts
    inputs = [
        Path(__file__),
        source,
        Path("factors/crypto/dated_carry_schedule.py"),
        Path("factors/common/dated_carry.py"),
        Path(p["base_engine"]),
        Path(p["engine"]),
        Path("tests/research_mainline/test_dated_account_contract.py"),
    ]
    inputs += [r / "completion.json" for r in parents]
    p["source_hashes"] = {str(q): sha(q) for q in inputs}
    save(output / "plan.json", p)
    save(
        output / "registration.json",
        {
            "before_new_financial_returns": True,
            "plan_sha256": sha(output / "plan.json"),
            "known_history_only": True,
            "original_body_and_actual_client_fees_not_certified": True,
        },
    )
    decisions = []
    for profile in p["profiles"]:
        signals, observations = schedule(spot, futures, p, profile)
        save(output / f"{profile}_signals.json", signals)
        decisions.append({"profile": profile, "observations": observations})
    save(
        output / "input_audit.json",
        {
            "passed": True,
            "raw_CSV_Decimal_rows": sum(r["rows"] for r in raw_checks),
            "raw_checks": raw_checks,
            "coverage": coverage,
            "entry_observations": decisions,
            "expiry_clock_basis": "Pre-entry0328listing plus generic quarter rule, separately confirmed before expiry; midnightAPIlabel used only as calendar-key.",
            "fees_and_margin_are_model_assumptions": True,
            "historical_version_authenticity_certified": False,
            "new_financial_accounts": 0,
        },
    )
    manifest = {q.name: sha(q) for q in output.iterdir() if q.is_file()}
    save(
        output / "completion.json", {"passed": True, "manifest": manifest, "profit_goal_met": False}
    )
    print(
        {
            "prepared": str(output),
            "raw_rows_audited": sum(r["rows"] for r in raw_checks),
            "coverage": coverage,
            "entry_observations": decisions,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output)
