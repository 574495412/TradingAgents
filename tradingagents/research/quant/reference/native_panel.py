"""从已核验公开原文构建原生价格与流量研究面板。

不读取模型、带单标签或账户状态；保留分钟日历、当时合约资格与完整截面。
按日保存有界 Parquet 行组，限制准备阶段内存。
"""

# ruff: noqa: ANN001, ANN201, EM101, TRY003, SLF001

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tradingagents.research.quant.reference import market_features as market, raw_minutes as capture

MINUTE = 60_000
WARMUP = 262


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )
    with Path(path).open("xb") as stream:
        stream.write(payload)


def _calendar_symbol_frames(frames_by_symbol, *, cutoff, columns) -> list:
    parts = []
    for symbol, frame in frames_by_symbol.items():
        visible = frame.loc[frame.index <= cutoff] if cutoff is not None else frame
        featured = market._build_symbol_features(
            visible.loc[:, market.RAW_COLUMNS],
            required_columns=frozenset(columns),
        )
        featured.loc[~visible.observed, :] = np.nan
        # 不能因事后已知合约名称，就让尚未上市或已经交割的成员进入当时广度分母。
        featured = featured.loc[visible.eligible, list(columns)].copy()
        if not featured.empty:
            featured["symbol"] = symbol
            featured.index.name = "bar_time"
            parts.append(featured.reset_index())
    return parts


_native_builder = FunctionType(
    market.build_feature_panel.__code__,
    {
        **market.build_feature_panel.__globals__,
        "_build_symbol_frames": _calendar_symbol_frames,
    },
    market.build_feature_panel.__name__,
    market.build_feature_panel.__defaults__,
)
_native_builder.__kwdefaults__ = market.build_feature_panel.__kwdefaults__


def build_chunk(contracts, records, *, first_close, last_close):
    full_index = pd.date_range(
        pd.Timestamp(first_close - WARMUP * MINUTE, unit="ms", tz="UTC"),
        pd.Timestamp(last_close, unit="ms", tz="UTC"),
        freq="min",
        name="bar_time",
    )
    raw_clocks = full_index.asi8 // 1_000_000 - MINUTE
    frames, flags = {}, {}
    used_sources = {}
    for symbol, contract in contracts.items():
        merged = {}
        for item in records[symbol]:
            if item["end_ms"] <= raw_clocks[0] or item["start_ms"] >= last_close:
                continue
            path = Path(item["path"])
            raw = capture._read_source(path, symbol=symbol, expected_sha=item["sha256"])
            capture._merge_rows(merged, raw["rows"], start=int(raw_clocks[0]), end=last_close)
            used_sources[str(path.resolve())] = item["sha256"]
        validated = capture._raw_frame([merged[k] for k in sorted(merged)], symbol)
        frame = validated.reindex(full_index)
        eligible = (raw_clocks >= contract["onboard_ms"]) & (
            raw_clocks < contract["delivery_ms"] if contract["delivery_ms"] else True
        )
        observed = frame.close.notna().to_numpy() & eligible
        frame.loc[~eligible, :] = np.nan
        frame["eligible"], frame["observed"] = eligible, observed
        frames[symbol] = frame
        flags[symbol] = pd.DataFrame({"eligible": eligible, "observed": observed}, index=full_index)
    built = _native_builder(frames, as_of=pd.Timestamp(last_close, unit="ms", tz="UTC"))
    output_index = pd.MultiIndex.from_product(
        [
            pd.date_range(
                pd.Timestamp(first_close, unit="ms", tz="UTC"),
                pd.Timestamp(last_close, unit="ms", tz="UTC"),
                freq="min",
            ),
            list(contracts),
        ],
        names=["bar_time", "symbol"],
    )
    built = built.reindex(output_index)
    for symbol, contract in contracts.items():
        subset = pd.IndexSlice[:, symbol]
        dates = built.xs(symbol, level="symbol").index
        built.loc[subset, "asset_class"] = contract["asset_class"]
        for key in ("eligible", "observed"):
            built.loc[subset, key] = flags[symbol][key].reindex(dates).to_numpy()
        built.loc[subset, "listing_age_minutes"] = np.maximum(
            0.0,
            (dates.asi8 // 1_000_000 - contract["onboard_ms"]) / MINUTE,
        )
    if built.loc[:, ["eligible", "observed"]].isna().any(axis=None):
        raise ValueError("Native panel lost venue availability flags")
    built["eligible"] = built.eligible.astype(bool)
    built["observed"] = built.observed.astype(bool)
    return built, used_sources


def run(panel_root, output, chunk_minutes=360):  # noqa: C901, PLR0912
    if type(chunk_minutes) is not int or chunk_minutes <= 0:
        raise ValueError("Chunk size must be a positive integer")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = panel_root / "manifest.json"
    original = json.loads(manifest_path.read_text())
    if (
        original.get("schema") != "damorne-dynamic-public-rule-panel.v1"
        or original.get("leader_labels_read") is not False
        or original.get("orders_enabled") is not False
        or original.get("trained_models_loaded") is not False
    ):
        raise ValueError("Expected the frozen public raw-minute capture")
    records = defaultdict(list)
    for item in original["raw_sources"]:
        declared = Path(item["path"])
        if not declared.exists():
            marker = panel_root.name + "/"
            declared = panel_root / str(declared).split(marker, 1)[1]
        records[item["symbol"]].append({**item, "path": str(declared)})
    code_paths = [Path(__file__), Path(market.__file__), Path(capture.__file__)]
    code_paths.extend(sorted(Path(market.__file__).parent.joinpath("factors").glob("*.py")))
    sources = {str(p.resolve()): sha(p) for p in code_paths}
    sources[str(manifest_path.resolve())] = sha(manifest_path)
    archive = output / "source"
    archive.mkdir()
    for i, path in enumerate(code_paths):
        (archive / f"{i:02d}_{path.name}").write_bytes(path.read_bytes())
    save(
        output / "protocol.json",
        {
            "schema": "dual-alpha-full-native-market-preparation.v1",
            "source_hashes": sources,
            "feature_columns": list(market.PANEL_COLUMNS),
            "chunk_minutes": chunk_minutes,
            "warmup_minutes": WARMUP,
            "native_cross_section": "eligible venue members only",
            "missing_data": "retain calendar NaNs; no forward filling or compressed return windows",
            "independent_oos": False,
            "historical_eligibility_proven": False,
            "derivatives_included": False,
            "order_books_included": False,
            "orders_enabled": False,
            "leader_labels_read": False,
        },
    )
    files = []
    for part in original["parts"]:
        first, last = part["first_close_ms"], part["last_close_ms"]
        day = pd.Timestamp(first, unit="ms", tz="UTC")
        path = output / f"candidate_{day:%Y%m%d}.parquet"
        rows = eligible_rows = observed_rows = 0
        writer = None
        try:
            for left in range(first, last + MINUTE, chunk_minutes * MINUTE):
                right = min(last, left + (chunk_minutes - 1) * MINUTE)
                panel, used = build_chunk(
                    original["contracts"], records, first_close=left, last_close=right
                )
                table = pa.Table.from_pandas(panel, preserve_index=True)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                rows += len(panel)
                eligible_rows += int(panel.eligible.sum())
                observed_rows += int(panel.observed.sum())
                sources.update(used)
        finally:
            if writer is not None:
                writer.close()
        if rows != part["rows"]:
            raise ValueError("Native full panel changed the declared calendar or universe")
        files.append(
            {
                "path": path.name,
                "sha256": sha(path),
                "rows": rows,
                "eligible_rows": eligible_rows,
                "observed_rows": observed_rows,
                "eligible_missing_rows": eligible_rows - observed_rows,
                "first_close_ms": first,
                "last_close_ms": last,
            },
        )
        print("native full market", path.name, rows, flush=True)
    for path, expected in sources.items():
        if sha(path) != expected:
            raise ValueError("Raw market or decision source changed while preparing")
    result = {
        "schema": "dual-alpha-full-native-market.v1",
        "daily_files": files,
        "source_hashes": sources,
        "symbols": original["symbols"],
        "contracts": original["contracts"],
        "feature_columns": list(market.PANEL_COLUMNS),
        "rows": sum(f["rows"] for f in files),
        "market_panel_contract": {
            "selection_filter_applied": False,
            "timestamp_semantics": "CLOSE_TIME",
            "feature_source": "RAW_CLOSED_BARS",
            "coverage_scope": "VENUE_UNIVERSE",
        },
        "historical_eligibility_proven": False,
        "derivatives_included": False,
        "order_books_included": False,
        "orders_enabled": False,
        "leader_labels_read": False,
    }
    save(output / "manifest.json", result)
    return {
        "parts": len(files),
        "rows": result["rows"],
        "native_columns": len(market.PANEL_COLUMNS),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-minutes", type=int, default=360)
    args = parser.parse_args()
    print(json.dumps(run(args.panel, args.output, args.chunk_minutes)))
