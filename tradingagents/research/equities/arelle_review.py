"""Trace consumed annual inputs to complete SEC submissions and Arelle.

Run this file directly, avoiding TradingAgents' dotenv-loading initializer.
SEC web-delivery HTML is retained separately from original SGML payloads.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def write_json(file, value):
    Path(file).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def extract_documents(raw):
    result = {}
    for match in re.finditer(rb"^<DOCUMENT>\r?\n(.*?)^</DOCUMENT>\r?$", raw, re.S | re.M):
        document = match.group(1)
        name = re.search(rb"^<FILENAME>([^\r\n]+)", document, re.M)
        opening = re.search(rb"^<TEXT>\r?\n", document, re.M)
        ending = document.rfind(b"\n</TEXT>")
        if name is None or opening is None or ending < opening.end():
            continue
        filename = name.group(1).decode("ascii").strip()
        if Path(filename).name != filename or filename in result:
            raise ValueError("Unsafe or duplicate SGML document identity")
        start = match.start(1) + opening.end()
        end = match.start(1) + ending
        payload = raw[start:end]
        wrapper = None
        # The SEC PDS SGML XBRL indicator is envelope syntax, not an XML node
        # belonging to the issuer document. Record exact byte boundaries.
        for newline in [b"\n", b"\r\n"]:
            prefix, suffix = b"<XBRL>" + newline, newline + b"</XBRL>"
            if payload.startswith(prefix) and payload.endswith(suffix):
                start += len(prefix)
                end -= len(suffix)
                wrapper = "SEC PDS <XBRL> framing"
                break
        result[filename] = {
            "filename": filename,
            "source_start": start,
            "source_end": end,
            "framing_removed": wrapper,
            "bytes": raw[start:end],
        }
    if not result:
        raise ValueError("No complete SGML documents; source is not a complete filing")
    return result


def capture(url, file):
    receipt = file.with_suffix(file.suffix + ".receipt.json")
    if file.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["url"] != url or saved["sha256"] != digest(file):
            raise ValueError("Immutable complete submission changed")
        return saved
    if file.exists() or receipt.exists():
        raise ValueError("Partial complete-submission capture requires inspection")
    events = []
    for attempt in range(3):
        time.sleep(0.4)
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "CZ-Agent/EquityResearch (public financial research)"},
                timeout=25,
            )
            events.append({"attempt": attempt + 1, "http_status": response.status_code})
            response.raise_for_status()
            file.write_bytes(response.content)
            saved = {
                "url": url,
                "sha256": digest(file),
                "bytes": len(response.content),
                "captured_at_utc": datetime.now(UTC).isoformat(),
                "events": events,
            }
            write_json(receipt, saved)
            return saved
        except requests.RequestException as exc:
            if not events or events[-1].get("attempt") != attempt + 1:
                events.append({"attempt": attempt + 1, "error": type(exc).__name__})
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == 2 or status in {400, 401, 403, 404}:
                write_json(
                    file.with_suffix(".failure.json"),
                    {"url": url, "events": events, "error": type(exc).__name__},
                )
                raise
            time.sleep(1)
    raise AssertionError("Unreachable capture state")


def compare_inputs(record, payload):
    comparisons = []
    for tag, expected in record["tags_checked"].items():
        selected = []
        for fact in payload["rows"]:
            period = fact.get("period_raw") or {}
            entity = fact.get("entity") or []
            unit = fact.get("unit")
            if (
                fact["tag"] != tag
                or not fact["namespace"].startswith("http://fasb.org/us-gaap/")
                or len(entity) != 2
                or entity[0] != "http://www.sec.gov/CIK"
                or entity[1].zfill(10) != record["cik"]
                or fact["dimensions"]
                or fact.get("non_dimensional_segment")
                or fact.get("non_dimensional_scenario")
                or fact["is_nil"]
                or fact["validity"] < 4
                or unit
                != [[{"namespace": "http://www.xbrl.org/2003/iso4217", "local_name": "USD"}], []]
            ):
                continue
            if tag == "Assets":
                if period.get("instant") != record["period_end"]:
                    continue
            else:
                if period.get("endDate") != record["period_end"] or "startDate" not in period:
                    continue
                duration = (
                    datetime.fromisoformat(period["endDate"])
                    - datetime.fromisoformat(period["startDate"])
                ).days + 1
                if not 330 <= duration <= 380:
                    continue
            try:
                value = Decimal(fact["value"])
            except InvalidOperation:
                continue
            if not value.is_finite():
                continue
            selected.append(
                {
                    "value": str(value),
                    "context_id": fact["context_id"],
                    "decimals": fact["decimals"],
                    "source_line": fact["source_line"],
                }
            )
        values = {Decimal(r["value"]) for r in selected}
        passed = len(values) == 1 and Decimal(str(expected)) in values
        comparisons.append(
            {
                "tag": tag,
                "expected": expected,
                "observed": selected,
                "unambiguous_exact_match": passed,
            }
        )
    return comparisons


def run(output, runtime):
    runtime_info = json.loads(
        subprocess.check_output(
            [
                str(runtime),
                "-c",
                "import importlib.metadata as m,sys,json; print(json.dumps({'arelle':m.version('arelle-release'),'executable':sys.executable,'prefix':sys.prefix}))",
            ],
            text=True,
        )
    )
    if runtime_info["arelle"] != "2.47.3":
        raise ValueError("Unregistered Arelle runtime")
    plan = json.loads((HERE / "depth_review_plan.json").read_text())
    original_run, pit = ROOT / plan["frozen_run"], ROOT / plan["pit_dataset"]
    provenance = json.loads((original_run / "financial-decision-provenance.json").read_text())
    selected = {
        r["accession"] for r in provenance if "2022-12-30" <= r["decision_day"] < "2024-06-27"
    }
    old_audit = json.loads((original_run / "original-annual-audit.json").read_text())
    records = [r for r in old_audit["records"] if r["accession"] in selected]
    if len(records) != len(selected):
        raise ValueError("Consumed annual accession is not in original reconciliation")
    filings = {r["accessionNumber"]: r for r in json.loads((pit / "filings.json").read_text())}
    tool_root = ROOT / "data/research/equity_audit_tools_20261026_v001"
    plugin_pin = json.loads((tool_root / "edgar-pin.json").read_text())
    for file, sha in plugin_pin["files"].items():
        if digest(tool_root / file) != sha:
            raise ValueError("Official SEC transform plugin changed after pin")
    plugin = tool_root / "plugins/EDGAR/transform"
    raw_root = ROOT / "data/research/equity_arelle_20261026_v001"
    source_root = raw_root / "complete-submissions"
    source_root.mkdir(exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(HERE / "depth_review_plan.json", output / "plan.json")
    for name in ["arelle_worker.py", "arelle_review.py"]:
        shutil.copyfile(HERE / name, output / name)
    write_json(
        output / "source-amendment.json",
        {
            "reason": "SEC's direct HTML delivery inserted Akamai scripts and invalid XHTML attributes; incomplete dependent downloads and an initially wrong plugin path were retained as failed probes. Use the SEC complete submission and exact documented SGML framing boundaries, not script removal or source repairs.",
            "original_web_bytes_are_not_pristine_filer_document": True,
            "comparison_to_old_web_capture": "Record both byte hashes and compare every consumed numeric input; changed input is a failure, never silently accepted.",
            "strategy_or_return_recomputed": False,
            "reference": "https://www.sec.gov/files/info/edgar/specifications/pds-dissemination-spec030314.pdf",
        },
    )
    registration = {
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "selected_accessions": sorted(selected),
        "new_economic_accounts": 0,
        "source_hashes": {
            f: digest(HERE / f)
            for f in ["arelle_worker.py", "arelle_review.py", "depth_review_plan.json"]
        },
        "frozen_provenance_sha256": digest(original_run / "financial-decision-provenance.json"),
        "old_annual_audit_sha256": digest(original_run / "original-annual-audit.json"),
        "tools_sha256": digest(tool_root / "wheels-manifest.json"),
        "sec_transform_pin": plugin_pin,
        "runtime": runtime_info,
        "external_attestation": False,
    }
    write_json(output / "registration.json", registration)
    reviewed = []
    for record in records:
        acc = record["accession"]
        folder = output / acc
        folder.mkdir()
        filing = filings[acc]
        primary = Path(filing["primaryDocument"]).name
        url = f"https://www.sec.gov/Archives/edgar/data/{int(record['cik'])}/{acc.replace('-', '')}/{acc}.txt"
        file = source_root / f"{acc}.txt"
        try:
            receipt = capture(url, file)
            raw = file.read_bytes()
            header = re.search(rb"<ACCEPTANCE-DATETIME>(\d{14})", raw)
            if header is None:
                raise ValueError("Missing raw acceptance timestamp")
            local = datetime.strptime(header.group(1).decode(), "%Y%m%d%H%M%S").replace(
                tzinfo=ZoneInfo("America/New_York")
            )
            accepted = local.astimezone(UTC)
            api_accepted = datetime.fromisoformat(filing["accepted_at_utc"])
            if accepted != api_accepted:
                raise ValueError("Raw acceptance timestamp differs from submissions API")
            documents = extract_documents(raw)
            if primary not in documents:
                raise ValueError("Original primary document not in complete submission")
            bundle = folder / "documents"
            bundle.mkdir()
            extraction = []
            for name, item in documents.items():
                if name != primary and not name.endswith((".xsd", ".xml")):
                    continue
                target = bundle / name
                target.write_bytes(item["bytes"])
                extraction.append(
                    {k: v for k, v in item.items() if k != "bytes"} | {"sha256": digest(target)}
                )
            write_json(
                folder / "extraction.json",
                {"parent_sha256": receipt["sha256"], "documents": extraction},
            )
            with (folder / "worker.log").open("w") as log:
                subprocess.run(
                    [
                        str(runtime),
                        str(HERE / "arelle_worker.py"),
                        "--entry",
                        str(bundle / primary),
                        "--output",
                        str(folder / "arelle.json"),
                        "--cache",
                        str(raw_root / "cache"),
                        "--plugin",
                        str(plugin),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                    check=True,
                    timeout=600,
                )
            payload = json.loads((folder / "arelle.json").read_text())
            matched = compare_inputs(record, payload)
            log_codes = {}
            for row in payload["validation_log"].get("log", []):
                code = row["code"]
                log_codes[code] = log_codes.get(code, 0) + 1
            result = {
                "accession": acc,
                "cik": record["cik"],
                "period_end": record["period_end"],
                "review_completed": True,
                "source_receipt": receipt,
                "raw_acceptance_at_utc": accepted.isoformat(),
                "raw_api_acceptance_match": True,
                "old_web_primary_sha256": record["original_sha256"],
                "sgml_primary_sha256": digest(bundle / primary),
                "whole_primary_bytes_equal": record["original_sha256"] == digest(bundle / primary),
                "comparisons": matched,
                "inputs_match": all(r["unambiguous_exact_match"] for r in matched),
                "full_xbrl_validation_passed": payload["full_validation_passed"],
                "validation_log_codes": log_codes,
                "tool_output_sha256": digest(folder / "arelle.json"),
                "extraction_sha256": digest(folder / "extraction.json"),
            }
        except (requests.RequestException, ValueError, subprocess.SubprocessError) as exc:
            result = {
                "accession": acc,
                "cik": record["cik"],
                "review_completed": False,
                "error": type(exc).__name__,
                "detail": str(exc)[:600],
                "inputs_match": False,
                "full_xbrl_validation_passed": False,
            }
        reviewed.append(result)
        write_json(folder / "review.json", result)
        write_json(
            output / "progress.json",
            {"selected": len(records), "completed": len(reviewed), "records": reviewed},
        )
        print(
            acc,
            "values",
            result["inputs_match"],
            "XBRL",
            result["full_xbrl_validation_passed"],
            flush=True,
        )
    summary = {
        "schema": "equity-original-tool-review.v1",
        "selected_annual_versions": len(records),
        "attempted_versions": len(reviewed),
        "reviewed_versions": sum(r["review_completed"] for r in reviewed),
        "numeric_inputs_checked": sum(len(r.get("comparisons", [])) for r in reviewed),
        "numeric_matches": sum(
            x["unambiguous_exact_match"] for r in reviewed for x in r.get("comparisons", [])
        ),
        "complete_numeric_match_versions": sum(r["inputs_match"] for r in reviewed),
        "full_xbrl_validation_passed_versions": sum(
            r["full_xbrl_validation_passed"] for r in reviewed
        ),
        "all_selected_versions_reviewed": len(records) == len(reviewed)
        and all(r["review_completed"] for r in reviewed),
        "all_consumed_inputs_match": all(r["inputs_match"] for r in reviewed),
        "all_xbrl_validations_passed": all(r["full_xbrl_validation_passed"] for r in reviewed),
        "records": reviewed,
        "new_economic_accounts": 0,
        "external_human_attestation": False,
        "independent_data_provider": False,
        "profit_goal_met": False,
    }
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    # Resolving the venv's Python symlink would silently run base Python and
    # lose its pinned packages. Keep the lexical absolute executable path.
    run(args.output, args.runtime.absolute())
