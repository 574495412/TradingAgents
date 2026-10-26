"""Public stock quotes: split-adjusted warmup kept separate from cash ledger.

Any split at/after the account's pre-session anchor blocks the entire cohort.
No survivor substitution, price filling or fabricated delisting proceeds.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

from .data import FIELDS, digest, sessions, validate, write_json


def capture(pit):
    pit = Path(pit)
    plan = json.loads((pit / "protocol.json").read_text())
    identities = json.loads((pit / "identities.json").read_text())
    output = pit / "prices"
    output.mkdir(exist_ok=True)
    calendar = sessions("US", plan["data_start"], plan["data_end"])
    account_calendar = calendar.loc["2022-12-30":]
    calendar.to_csv(output / "signal_sessions.csv", index_label="date")
    account_calendar.to_csv(output / "US_sessions.csv", index_label="date")
    registration = {
        "protocol_sha256": digest(pit / "protocol.json"),
        "identities_sha256": digest(pit / "identities.json"),
        "captured_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "signal_start": plan["data_start"],
        "account_anchor": "2022-12-30",
        "rule": "Warmup uses split-adjusted ratio series. Ledger starts after 2022 splits. Reject any split at/after account anchor, including events after requested backtest end but before capture, since vendor back-adjustment changes nominal historical shares. No dropped seed member.",
    }
    if not (output / "price-registration.json").exists():
        write_json(output / "price-registration.json", registration)
    else:
        old = json.loads((output / "price-registration.json").read_text())
        for key in ["protocol_sha256", "identities_sha256", "rule"]:
            if old[key] != registration[key]:
                raise ValueError("Price-capture registration changed")
    manifest = {
        "schema": "ta-equity-data.v1",
        "start": "2022-12-30",
        "end": plan["data_end"],
        "provider": "Yahoo split-adjusted raw OHLC; account split-free admission",
        "broker_verified": False,
        "corporate_action_authority": "public Yahoo events, not certified complete",
        "symbols": {},
        "calendar_hashes": {"US": digest(output / "US_sessions.csv")},
    }
    for item in identities:
        s = item["symbol"]
        if s == "EXAS.US":
            # Yahoo deleted this acquired security. Retain the fixed member
            # and its independently archived, explicitly shorter history.
            archived = pit / "supplemental/EXAS_zenodo.csv"
            if not archived.is_file():
                raise ValueError("Delisted member requires archived quote evidence")
            f = pd.read_csv(archived, index_col="Date", parse_dates=["Date"])
            f.columns = [c.lower().replace(" ", "_") for c in f.columns]
            f.index.name = "date"
            f = f.loc[plan["data_start"] :]
            if ((f.adj_close - f.close).abs() > 1e-6).any():
                raise ValueError(
                    "Archived nominal/adjusted discrepancy requires action reconstruction"
                )
            # The original annual filings, captured separately, state no
            # dividends ever paid; covered 2022/2023 balance-sheet years.
            # Source completeness is still explicitly uncertified.
            f["dividend"], f["split"] = 0.0, 0.0
            f = f[list(FIELDS)]
            f.to_csv(output / f"{s}_signal.csv", index_label="date")
            account = f.loc["2022-12-30":]
            validate(account, account_calendar.loc[: account.index[-1]])
            account.to_csv(output / f"{s}.csv", index_label="date")
            manifest["symbols"][s] = {
                "file": f"{s}.csv",
                "sha256": digest(output / f"{s}.csv"),
                "rows": len(account),
                "signal_file": f"{s}_signal.csv",
                "signal_sha256": digest(output / f"{s}_signal.csv"),
                "archive_file": "../supplemental/EXAS_zenodo.csv",
                "archive_sha256": digest(archived),
                "currency": "USD",
                "identity_cik": item["cik"],
                "last_observed_session": str(account.index[-1].date()),
                "coverage_complete": False,
                "source": "CC0 academic Zenodo snapshot DOI 10.5281/zenodo.12556425; original provider not declared; not broker certified",
                "actions": "Archive AdjClose equals Close; original annual no-dividend disclosure; complete corporate-action history not independently certified",
                "dividend_events": 0,
                "split_events": 0,
            }
            print(
                "archived delisted stock",
                s,
                len(account),
                "last",
                str(account.index[-1].date()),
                flush=True,
            )
            continue
        vendor = output / f"{s}_vendor.csv"
        metadata_path = output / f"{s}_metadata.json"
        if vendor.exists() and metadata_path.exists():
            raw = pd.read_csv(vendor, index_col="vendor_timestamp")
            raw.index = pd.DatetimeIndex(
                pd.to_datetime(raw.index, utc=True).tz_convert("America/New_York")
            )
            metadata = json.loads(metadata_path.read_text())
        elif vendor.exists() and not metadata_path.exists():
            # A complete CSV survived the metadata serialization failure.
            # Retain it and record fresh currency metadata separately.
            raw = pd.read_csv(vendor, index_col="vendor_timestamp")
            raw.index = pd.DatetimeIndex(
                pd.to_datetime(raw.index, utc=True).tz_convert("America/New_York")
            )
            ticker = yf.Ticker(item["yahoo"])
            meta = ticker.get_history_metadata()
            metadata = {
                k: meta.get(k)
                for k in [
                    "currency",
                    "symbol",
                    "exchangeName",
                    "instrumentType",
                    "exchangeTimezoneName",
                ]
            }
            metadata["recovery"] = (
                "Original vendor CSV retained after HistoryMetadata serialization failed; currency metadata retrieved separately. No strategy returns computed."
            )
            write_json(metadata_path, metadata)
        elif metadata_path.exists():
            raise ValueError("Vendor CSV missing; inspect before recovery")
        else:
            ticker = yf.Ticker(item["yahoo"])
            raw = ticker.history(
                start=plan["data_start"],
                end="2026-10-27",
                interval="1d",
                auto_adjust=False,
                back_adjust=False,
                actions=True,
                repair=False,
                prepost=False,
                keepna=True,
                timeout=30,
                raise_errors=True,
            )
            if raw.empty:
                raise ValueError(f"Selected member quote unavailable {s}; cohort cannot shrink")
            meta = ticker.get_history_metadata()
            metadata = {
                k: meta.get(k)
                for k in [
                    "currency",
                    "symbol",
                    "exchangeName",
                    "instrumentType",
                    "exchangeTimezoneName",
                ]
            }
            raw.to_csv(vendor, index_label="vendor_timestamp")
            write_json(metadata_path, metadata)
        if metadata.get("currency") != "USD":
            raise ValueError("Quote currency changed")
        f = raw.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Dividends": "dividend",
                "Stock Splits": "split",
            }
        )
        f.index = pd.DatetimeIndex(f.index.tz_localize(None).normalize(), name="date")
        future_splits = f.loc["2022-12-30":, "split"]
        if (future_splits != 0).any():
            raise ValueError(
                f"Unhandled nominal split for frozen {s}: {future_splits[future_splits != 0].to_dict()}"
            )
        if "Capital Gains" in raw and (raw["Capital Gains"] != 0).any():
            raise ValueError("Unhandled capital distribution")
        f = f.loc[(f.index >= plan["data_start"]) & (f.index < plan["data_end"]), list(FIELDS)]
        # Warmup is a ratio series only. Account validation is stricter and
        # receives unmodified, explicit, zero-split event columns after anchor.
        signal = f.copy()
        signal.to_csv(output / f"{s}_signal.csv", index_label="date")
        if not signal.index.equals(calendar.loc[signal.index[0] :].index):
            raise ValueError(f"Internal warmup calendar gap {s}; no synthetic bars")
        account = f.loc["2022-12-30":]
        validate(account, account_calendar)
        account.to_csv(output / f"{s}.csv", index_label="date")
        manifest["symbols"][s] = {
            "file": f"{s}.csv",
            "sha256": digest(output / f"{s}.csv"),
            "rows": len(account),
            "vendor_file": vendor.name,
            "vendor_sha256": digest(vendor),
            "signal_file": f"{s}_signal.csv",
            "signal_sha256": digest(output / f"{s}_signal.csv"),
            "metadata_file": metadata_path.name,
            "metadata_sha256": digest(metadata_path),
            "currency": "USD",
            "dividend_events": int((account.dividend != 0).sum()),
            "split_events": 0,
            "first_observed_signal_session": str(signal.index[0].date()),
            "leading_unobserved_sessions": int((calendar.index < signal.index[0]).sum()),
            "leading_gap_handling": "Unobserved leading history stays absent. Return ranks require full monthly lookback; no IPO date inferred or bars filled. Account window requires complete quotes for every member.",
            "identity_cik": item["cik"],
            "historic_cover_symbols": item["historic_cover_symbols"],
        }
        print(
            "stock quotes",
            s,
            len(signal),
            len(account),
            "dividends",
            manifest["symbols"][s]["dividend_events"],
            flush=True,
        )
    manifest["signal_calendar_sha256"] = digest(output / "signal_sessions.csv")
    manifest["full_window_available"] = False
    manifest["coverage_gap"] = (
        "EXAS archive ends 2024-06-26. Full requested account through 2026-09-30 blocked; do not drop EXAS, manufacture bars or fabricate merger cash settlement."
    )
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pit", required=True, type=Path)
    capture(p.parse_args().pit)
