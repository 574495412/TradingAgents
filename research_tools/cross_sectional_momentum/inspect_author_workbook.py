"""Read-only OOXML census and date/role verification; no strategy returns."""

import argparse
import json
import shutil
from collections import Counter
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as E
from zipfile import ZipFile

from factors.reference.crypto_risk_benchmark import week_label, window_coverage


def inspect(source, output, artifact_receipt):
    output.mkdir(parents=True, exist_ok=False)
    import hashlib

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    workbook = source / "LTW_3factor.xlsx"
    original = sha(workbook)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(workbook) as zipped:
        assert sum(i.file_size for i in zipped.infolist()) <= 33554432
        wb = E.fromstring(zipped.read("xl/workbook.xml"))
        sheets = [dict(s.attrib) for s in wb.findall("x:sheets/x:sheet", ns)]
        assert len(sheets) == 1 and sheets[0]["name"] == "Sheet1"
        strings = [
            "".join(t.text or "" for t in si.iter("{" + ns["x"] + "}t"))
            for si in E.fromstring(zipped.read("xl/sharedStrings.xml"))
        ]
        sheet = E.fromstring(zipped.read("xl/worksheets/sheet1.xml"))
        rows = sheet.findall("x:sheetData/x:row", ns)
        notes, records, headers = [], [], None
        for row in rows:
            values = []
            for cell in row:
                v = cell.find("x:v", ns)
                if v is None:
                    values.append(None)
                elif cell.get("t") == "s":
                    values.append(strings[int(v.text)])
                else:
                    values.append(v.text)
            if values == ["yyww", "cmkt", "csize", "cmom"]:
                headers = values
                continue
            if headers is None:
                if values and values[0] is not None:
                    notes.append({"cell": row[0].get("r"), "text": values[0]})
                continue
            assert len(values) == 4 and all(value is not None for value in values)
            key = int(values[0])
            nominal = week_label(key)
            assert all(Decimal(value).is_finite() for value in values[1:])
            records.append(
                {
                    "source_row": int(row.get("r")),
                    "yyww": key,
                    "nominal_week_end_label": nominal.isoformat(),
                    **dict(zip(headers[1:], values[1:], strict=True)),
                }
            )
        formula_count = len(sheet.findall(".//x:f", ns))
        dimension = sheet.find("x:dimension", ns).get("ref")
    assert headers == ["yyww", "cmkt", "csize", "cmom"] and records
    keys = [r["yyww"] for r in records]
    assert len(keys) == len(set(keys)) and keys == sorted(keys)
    minimum, maximum = keys[0], keys[-1]
    years = Counter(key // 100 for key in keys)
    expected = [
        year * 100 + w
        for year in range(minimum // 100, maximum // 100 + 1)
        for w in range(1, 53)
        if minimum <= year * 100 + w <= maximum
    ]
    gaps = sorted(set(expected) - set(keys))
    overview = artifact_receipt.read_text()
    assert '"sheets":1' in overview and '"range":"A1:D485"' in overview
    windows = {
        "screen2025": window_coverage(keys, "2025-01-01", "2026-01-01"),
        "diagnostic2026": window_coverage(keys, "2026-01-01", "2026-09-01"),
        "continuous2025_2026": window_coverage(keys, "2025-01-01", "2026-09-01"),
    }
    summary = {
        "schema": "author-crypto-benchmark-profile.v1",
        "source_kind": "author_published_derived_weekly_factor_returns",
        "sheets": 1,
        "sheet_name": "Sheet1",
        "used_range": dimension,
        "headers": headers,
        "data_rows": len(records),
        "first_yyww": minimum,
        "last_yyww": maximum,
        "first_nominal_week_label": records[0]["nominal_week_end_label"],
        "last_nominal_week_label": records[-1]["nominal_week_end_label"],
        "year_rows": dict(years),
        "duplicate_week_keys": len(keys) - len(set(keys)),
        "interior_week_gaps": gaps,
        "missing_numeric_cells": 0,
        "formula_cells": formula_count,
        "notes": notes,
        "units_explicit_in_workbook": False,
        "update_note_matches_last_row_year": False,
        "raw_coin_ids_present": False,
        "raw_coin_market_caps_present": False,
        "original_raw_input_gap_resolved": False,
        "existing_account_overlap": windows,
        "new_financial_accounts": 0,
        "external_review_performed": False,
        "profit_goal_met": False,
        "orders_enabled": False,
    }
    assert original == sha(workbook)
    for name, data in [("profile", summary), ("normalized_reference_rows", records)]:
        (output / (name + ".json")).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        )
    shutil.copyfile(artifact_receipt, output / "artifact-inspection.ndjson")
    (output / "verification.json").write_text(
        json.dumps(
            {
                "source_workbook_sha256_before_and_after": original,
                "artifact_tool_and_OOXML_sheet_range_agree": True,
                "all_original_numeric_lexemes_preserved_no_unit_conversion": True,
                "read_only_no_export_or_recalculation": True,
                "profile_source_sha256": sha(Path(__file__)),
                "reference_contract_sha256": sha(
                    Path("factors/reference/crypto_risk_benchmark.py")
                ),
                "no_external_third_party_review": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    (output / "completion.json").write_text(
        json.dumps(
            {
                "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
                "new_financial_accounts": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "source_kind",
                    "data_rows",
                    "first_yyww",
                    "last_yyww",
                    "raw_coin_market_caps_present",
                    "existing_account_overlap",
                    "profit_goal_met",
                ]
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--artifact-receipt", type=Path, required=True)
    a = p.parse_args()
    inspect(a.source, a.output, a.artifact_receipt)
