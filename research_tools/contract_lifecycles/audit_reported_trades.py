"""Independently inspect archived aggregate IDs/timestamps, without execution claims."""

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from research_tools.factor_lab.io import read, save, sha


def audit(source, output):
    if output.exists():
        raise FileExistsError("Preserve independent trade-source receipt")
    summary = read(source / "summary.json")
    checked = []
    for r in summary["archive_probes"]:
        if not r["archive_exists"]:
            checked.append(
                {
                    "prefix": r["prefix"],
                    "archive_exists_at_exact_prefix": False,
                    "does_not_prove_global_market_inactivity": True,
                }
            )
            continue
        path = source / "trades" / (Path(r["prefix"]).name + ".zip")
        if (
            sha(path) != r["archive_sha256"]
            or sha(path) != (path.with_name(path.name + ".CHECKSUM")).read_text().split()[0]
        ):
            raise ValueError("Official ZIP identity differs")
        rows = after = 0
        previous = None
        with zipfile.ZipFile(path) as z:
            member = z.namelist()[0]
            with z.open(member) as stream:
                header = not stream.readline().decode().split(",")[0].isdigit()
            with z.open(member) as stream:
                for chunk in pd.read_csv(
                    stream,
                    header=None,
                    skiprows=int(header),
                    usecols=[0, 5],
                    dtype="int64",
                    chunksize=150000,
                ):
                    ids, times = chunk[0].to_numpy(), chunk[5].to_numpy()
                    if not (np.diff(ids) == 1).all() or (
                        previous is not None and ids[0] != previous + 1
                    ):
                        raise ValueError("Aggregate ID sequence has a gap or duplicate")
                    previous = int(ids[-1])
                    rows += len(chunk)
                    after += int(
                        (times >= 1661418000000).sum()
                    )  # 2022-08-25 09:00UTC, explicit notice boundary.
        if rows != r["rows"] or r["last_aggregate_id"] - r["first_aggregate_id"] + 1 != rows:
            raise ValueError("Independent aggregate count differs")
        checked.append(
            {
                "prefix": r["prefix"],
                "archive_exists_at_exact_prefix": True,
                "rows": rows,
                "rows_at_or_after_notice_2022_08_25_09_utc": after,
                "all_aggregate_ids_contiguous": True,
                "archive_sha256": sha(path),
            }
        )
    actual = [r for r in summary["archive_probes"] if r["archive_exists"]]
    if len(actual) == 2 and actual[1]["first_aggregate_id"] != actual[0]["last_aggregate_id"] + 1:
        raise ValueError("Cross-day aggregate sequence discontinuity")
    result = {
        "schema": "independent-reported-trade-source-audit.v1",
        "passed": True,
        "checks": checked,
        "source_completion_sha256": sha(source / "completion.json"),
        "verifier_sha256": sha(Path(__file__)),
        "BLZ_relisting_or_notice_cancellation_certified": False,
        "actual_client_execution_or_cash_certified": False,
        "new_financial_accounts": 0,
    }
    save(output, result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    audit(a.source, a.output)
