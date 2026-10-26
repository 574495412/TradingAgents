"""Capture four identified official type references, without changing frozen membership."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import requests

from research_tools.contract_lifecycles.collect import request, text_nodes
from research_tools.factor_lab.io import save, sha

DETAILS = [
    ("d592f6ba938746cbadaaf5a8a714abd6", ["INTCUSDT"], "equity", "2026-01-30T08:47:00Z"),
    (
        "f083bdc8f37d4d45a0e0ec6fd5cd4086",
        ["CRCLUSDT", "MSTRUSDT"],
        "equity",
        "2026-02-05T11:28:00Z",
    ),
    ("bd92dee1018f421fae1ace119fc2f706", ["XAGUSDT"], "commodity", "2026-01-07T07:00:00Z"),
]
BLOG = "https://www.binance.com/en/blog/futures/2106113045523004581"


def capture(root):
    root.mkdir(parents=True, exist_ok=False)
    save(
        root / "plan.json",
        {
            "schema": "expanded-official-type-scope.v1",
            "source_sha256": sha(Path(__file__)),
            "maximum_new_requests": 4,
            "transport_retries": 0,
            "details": DETAILS,
            "blog": BLOG,
            "new_financial_accounts": 0,
        },
    )
    rows = []
    for code, symbols, kind, published in DETAILS:
        data = request(
            root,
            code,
            "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query",
            {"articleCode": code},
        )
        text = text_nodes(__import__("json").loads(data["body"]))
        if not all(s in text for s in symbols):
            raise ValueError("Named type reference missing")
        (root / (code + ".txt")).write_text(text)
        for s in symbols:
            rows.append(
                {
                    "symbol": s,
                    "class": kind,
                    "url": "https://www.binance.com/en/support/announcement/detail/" + code,
                    "current_body_sha256": sha(root / (code + ".txt")),
                    "claimed_published_at": published,
                    "cms_publishDate": data["publishDate"],
                    "historical_original_body_certified": False,
                }
            )
    response = requests.get(BLOG, timeout=25)
    (root / "xau_type_blog.html").write_bytes(response.content)
    save(
        root / "xau_type_blog.receipt.json",
        {
            "url": response.url,
            "status": response.status_code,
            "body_sha256": sha(root / "xau_type_blog.html"),
            "captured_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    response.raise_for_status()
    if "XAUUSDT" not in response.text or "gold" not in response.text.lower():
        raise ValueError("Official blog body unavailable; preserve failed response")
    rows.append(
        {
            "symbol": "XAUUSDT",
            "class": "commodity",
            "url": BLOG,
            "current_body_sha256": sha(root / "xau_type_blog.html"),
            "claimed_publication_day": "2026-03-12",
            "launch_date_verified": False,
            "historical_original_body_certified": False,
            "timing_caveat": "Current official Jan8 notice says Jan5 availability while source candles begin2025Dec11; do not infer initial launch or universal client access.",
        }
    )
    save(root / "type_references.json", rows)
    save(
        root / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in root.iterdir() if p.is_file()},
            "membership_modified": False,
            "historical_versions_certified": False,
            "new_financial_accounts": 0,
        },
    )
    print(
        {"captured_type_references": len(rows), "requests": 4, "membership_modified": False},
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    capture(p.parse_args().output)
