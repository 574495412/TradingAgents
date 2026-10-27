"""Capture a public author-linked workbook, with bounded ordinary redirects."""

import argparse
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

from research_tools.factor_lab.io import read, save, sha


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).with_name("author_data_plan_v003.json")
    plan = read(path)
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "registered_wall_ns": time.time_ns(),
            "source_hashes": {str(p): sha(p) for p in [path, Path(__file__)]},
            "before_workbook_download_and_content_inspection": True,
        },
    )
    receipts = []

    def get(url, stem):
        if len(receipts) >= plan["budget"]["maximum_native_requests"]:
            raise ValueError("HTTP request budget reached")
        receipt = {"url": url, "stem": stem, "sent_wall_ns": time.time_ns(), "status": None}
        receipts.append(receipt)
        try:
            with requests.get(
                url,
                timeout=(10, 15),
                stream=True,
                allow_redirects=False,
                headers={"Accept-Encoding": "identity"},
            ) as response:
                receipt.update(
                    status=response.status_code,
                    headers={
                        k: response.headers[k]
                        for k in [
                            "Location",
                            "Date",
                            "Age",
                            "ETag",
                            "Last-Modified",
                            "Content-Type",
                            "Content-Encoding",
                        ]
                        if k in response.headers
                    },
                )
                body = response.raw.read(
                    plan["budget"]["maximum_body_bytes_each"] + 1, decode_content=False
                )
            if len(body) > plan["budget"]["maximum_body_bytes_each"]:
                raise ValueError("Body limit reached")
            file = output / (stem + ".body")
            file.write_bytes(body)
            receipt.update(bytes=len(body), sha256=sha(file))
        except Exception as error:
            body = b""
            receipt.update(error_type=type(error).__name__, error=str(error)[:200])
        receipt["received_wall_ns"] = time.time_ns()
        save(output / (stem + ".receipt.json"), receipt)
        print({k: receipt.get(k) for k in ["stem", "status", "bytes", "error_type"]}, flush=True)
        return receipt, body

    browser = Path(plan["browser_receipt"])
    assert sha(browser) == plan["browser_receipt_sha256"]
    assert "xiwuacademic.com/data-and-code" in browser.read_text()
    link = Path(plan["link_receipt"])
    assert sha(link) == plan["link_receipt_sha256"]
    assert plan["workbook_url"].replace("&dl=1", "&dl=0") in link.read_text()
    save(output / "author_link_resolution_receipt.json", read(link))
    save(output / "author_page_browser_receipt.json", read(browser))
    url = plan["workbook_url"]
    downloaded = False
    for index in range(plan["budget"]["maximum_redirects"] + 1):
        if urlsplit(url).scheme != "https" or urlsplit(url).hostname not in plan["redirect_hosts"]:
            raise ValueError("Unexpected redirect host; do not authenticate or follow")
        receipt, body = get(url, "workbook_" + str(index))
        if receipt["status"] == 200 and body[:4] == b"PK\x03\x04":
            (output / "LTW_3factor.xlsx").write_bytes(body)
            downloaded = True
            break
        if receipt["status"] not in [301, 302, 303, 307, 308]:
            break
        location = receipt["headers"].get("Location")
        if not location:
            break
        url = urljoin(url, location)
    save(
        output / "summary.json",
        {
            "workbook_downloaded": downloaded,
            "requests": len(receipts),
            "source_kind_not_yet_verified": True,
            "new_financial_accounts": 0,
            "external_review_performed": False,
            "profit_goal_met": False,
        },
    )
    save(
        output / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
            "new_financial_accounts": 0,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    collect(p.parse_args().output)
