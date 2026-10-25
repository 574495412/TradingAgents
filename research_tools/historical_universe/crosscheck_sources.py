"""Versioned daily5m source recovery and full-bar comparisons; parents stay frozen."""

import argparse
import io
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research_tools.factor_lab.io import read, save, sha


def parse(payload, symbol, day):
    name = f"{symbol}-5m-{day}.csv"
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        if z.namelist() != [name] or z.testzip():
            raise ValueError("Bad daily5mZIP")
        f = pd.read_csv(io.BytesIO(z.read(name)), header=None, dtype=str)
    if f.iloc[0, 0] == "open_time":
        f = f.iloc[1:]
    f = f.apply(pd.to_numeric)
    times = pd.DatetimeIndex(
        pd.to_datetime(f[0].to_numpy(dtype=np.int64), unit="ms", utc=True), name="date"
    )
    expected = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=288, freq="5min")
    if not times.equals(expected):
        raise ValueError("Source has missing or wrong5mcalendar")
    f.index = times
    return f


def crosscheck(partial, output, recovery=None):  # noqa: C901
    output.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("discrepancy_plan.json")
    plan = read(plan_path)
    recovery_data = read(recovery) if recovery else None
    if recovery_data and recovery_data["original_plan_sha256"] != sha(plan_path):
        raise ValueError("Recovery changed the original investigation scope")
    parent = read(partial / "plan.json")
    if parent["source_sha256"] != sha(plan_path) or (partial / "completion.json").exists():
        raise ValueError("Parent plan changed or partial is already completed")
    cases = parent["contract_days"]
    if len(cases) > plan["budget"]["maximum_contract_days"]:
        raise ValueError("Case budget exceeded")
    save(
        output / "plan.json",
        {
            **plan,
            "parent_partial": str(partial),
            "parent_plan_sha256": sha(partial / "plan.json"),
            "source_code_sha256": sha(Path(__file__)),
            "contract_days": cases,
            "before_crosscheck_results": True,
            "recovery": recovery_data,
        },
    )
    lock = threading.Lock()
    local = threading.local()
    counter = {"requests": 0, "bytes": 0}

    def fetch(key):
        if not hasattr(local, "session"):
            local.session = requests.Session()
        for attempt in range(3):
            with lock:
                counter["requests"] += 1
                prior = (
                    recovery_data["prior_HTTP_request_upper_bound"]
                    if recovery_data
                    else 2 * len(cases)
                )
                maximum = (
                    recovery_data["total_HTTP_request_budget"]
                    if recovery_data
                    else plan["budget"]["maximum_http_requests"]
                )
                if counter["requests"] + prior > maximum:
                    raise ValueError("Combined source request budget exceeded")
            try:
                r = local.session.get(plan["origin"] + "/" + key, timeout=30)
                r.raise_for_status()
                with lock:
                    counter["bytes"] += len(r.content)
                    if counter["bytes"] > plan["budget"]["maximum_bytes"]:
                        raise ValueError("Byte budget exceeded")
                return r.content
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise AssertionError("Retry state")

    def one(case):
        s, day = case["symbol"], case["date"]
        name = f"{s}-5m-{day}.zip"
        key = f"data/futures/um/daily/klines/{s}/5m/{name}"
        dest = output / "raw" / s
        dest.mkdir(parents=True, exist_ok=True)
        old = partial / "raw" / s / name
        extra = Path(recovery_data["additional_cache"]) / "raw" / s / name if recovery_data else old
        if extra.is_file():
            old = extra
        payload = old.read_bytes() if old.is_file() else fetch(key)
        oldcheck = old.with_name(name + ".CHECKSUM")
        if not oldcheck.is_file() and (partial / "raw" / s / (name + ".CHECKSUM")).is_file():
            oldcheck = partial / "raw" / s / (name + ".CHECKSUM")
        checksum = oldcheck.read_bytes() if oldcheck.is_file() else fetch(key + ".CHECKSUM")
        parts = checksum.decode().split()
        from hashlib import sha256

        if (
            len(parts) != 2
            or parts[0].lower() != sha256(payload).hexdigest()
            or parts[1].lstrip("*") != name
        ):
            raise ValueError("Official checksum differs")
        frame = parse(payload, s, day)
        for path, content in [(dest / name, payload), (dest / (name + ".CHECKSUM"), checksum)]:
            with path.open("xb") as stream:
                stream.write(content)
        return {
            "symbol": s,
            "date": day,
            "zip": str((dest / name).relative_to(output)),
            "zip_sha256": sha(dest / name),
            "checksum_sha256": sha(dest / (name + ".CHECKSUM")),
            "reused_parent_zip": old.is_file(),
            "reused_parent_checksum": oldcheck.is_file(),
        }, frame

    with ThreadPoolExecutor(max_workers=plan["budget"]["workers"]) as e:
        loaded = list(e.map(one, cases))
    oldroot = Path(plan["old_5m"])
    daily = Path(plan["parent_daily"])
    oldbars = {
        s: pd.read_parquet(oldroot / (s + "_5m.parquet"))
        for s in {r["symbol"] for r, _ in loaded}
        if (oldroot / (s + "_5m.parquet")).exists()
    }
    source_daily = {
        s: pd.read_parquet(daily / "symbols" / (s + "_1d.parquet"))
        for s in {r["symbol"] for r, _ in loaded}
    }
    rows = []
    frames = {}
    for receipt, f in loaded:
        s, day = receipt["symbol"], receipt["date"]
        frames[(s, day)] = f
        d = source_daily[s].loc[day]
        aggregate = {
            "open": float(f.iloc[0, 1]),
            "high": float(f[2].max()),
            "low": float(f[3].min()),
            "close": float(f.iloc[-1, 4]),
            "volume": float(f[5].sum()),
            "quote_volume": float(f[7].sum()),
            "trades": float(f[8].sum()),
            "taker_base_volume": float(f[9].sum()),
            "taker_quote_volume": float(f[10].sum()),
        }
        old_error = None
        if s in oldbars:
            o = oldbars[s].loc[day]
            if not o.index.equals(f.index):
                raise ValueError("Old calendar differs")
            old_error = {
                k: float(np.max(np.abs(o[k].to_numpy() - f[i].to_numpy())))
                for k, i in [
                    ("open", 1),
                    ("high", 2),
                    ("low", 3),
                    ("close", 4),
                    ("volume", 5),
                    ("quote_volume", 7),
                ]
            }
        daily_error = {k: aggregate[k] - float(d[k]) for k in aggregate}
        rows.append(
            {
                **receipt,
                "rows": len(f),
                "aggregate": aggregate,
                "old5m_max_absolute_error": old_error,
                "original1d_delta": daily_error,
                "daily_quote_mismatch": abs(daily_error["quote_volume"])
                > max(0.01, 1e-6 * aggregate["quote_volume"]),
                "old5m_matches_all_fields": old_error is not None
                and all(v < 1e-7 for v in old_error.values()),
            }
        )
    (output / "api").mkdir()
    api_checks = []
    for case in plan["api_reference_days"]:
        s, day = case["symbol"], case["date"]
        path = partial / "api" / (s + "_" + day + ".json")
        raw = read(path)
        r = raw["response"]["structuredContent"]["result"]
        if raw["response"].get("isError") or len(r) != 288:
            raise ValueError("APIreference incomplete")
        a = np.array(r, dtype=float)
        f = frames[(s, day)].to_numpy(dtype=float)
        if a.shape != f.shape or not np.array_equal(a[:, 0], f[:, 0]):
            raise ValueError("APIcalendar differs")
        error = np.max(np.abs(a - f), axis=0).tolist()
        if max(error) > 1e-7:
            raise ValueError("API and official daily5marchive differ")
        shutil.copyfile(path, output / "api" / path.name)
        api_checks.append(
            {
                "symbol": s,
                "date": day,
                "rows": len(r),
                "api_sha256": sha(path),
                "all12_fields_match_archive": True,
                "maximum_absolute_error": max(error),
            }
        )
    summary = {
        "schema": "official-source-crosscheck.v1",
        "contract_days": len(rows),
        "daily5m_matches_old5m_cases": sum(r["old5m_matches_all_fields"] for r in rows),
        "cases_in_old_pool": sum(r["old5m_max_absolute_error"] is not None for r in rows),
        "daily5m_disagrees_original1d_quote_cases": sum(r["daily_quote_mismatch"] for r in rows),
        "api_checks": api_checks,
        "new_http_requests": counter["requests"],
        "parent_request_upper_bound": 2 * len(cases),
        "rows": rows,
        "source_code_sha256": sha(Path(__file__)),
        "parent_completion_sha256": sha(daily / "completion.json"),
        "original_datasets_modified": False,
        "financial_accounts": 0,
    }
    save(output / "summary.json", summary)
    save(
        output / "completion.json",
        {
            "manifest": {
                str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
            },
            "source_crosscheck_complete": True,
            "financial_accounts": 0,
        },
    )
    print({k: v for k, v in summary.items() if k not in ["rows", "api_checks"]}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--recovery", type=Path)
    a = p.parse_args()
    crosscheck(a.partial, a.output, a.recovery)
