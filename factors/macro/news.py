"""Official FOMC news only, joined after publication plus a declared latency."""

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from fractions import Fraction
from html import unescape

import requests

from ..storage import save, sha

DATES = [
    "20231213",
    "20240131",
    "20240320",
    "20240501",
    "20240612",
    "20240731",
    "20240918",
    "20241107",
    "20241218",
    "20250129",
    "20250319",
    "20250507",
    "20250618",
    "20250730",
    "20250917",
    "20251029",
    "20251210",
    "20260128",
    "20260318",
    "20260429",
    "20260617",
    "20260729",
]
HAWKISH = [
    "inflation remains elevated",
    "inflation risks",
    "restrictive",
    "reduce its holdings",
    "additional policy firming",
    "upside risks",
]
DOVISH = [
    "inflation has eased",
    "inflation has made progress",
    "reduce the target range",
    "lower the target range",
    "risks to employment",
    "slowing",
]
UNCERTAIN = ["uncertain", "uncertainty", "risks", "monitor", "assess"]


def number(value):
    parts = value.strip().replace("−", "-").split("-")
    return float(sum(Fraction(x.strip()) for x in parts))


def parse(html):
    # Limit lexical analysis to official statement paragraphs, excluding menus/footers.
    date = re.search(r'<p class="article__time">(.*?)</p>', html, re.S)
    clock = re.search(r"For release at (\d+):(\d+) p\.m\. (EST|EDT)", html)
    match = re.search(r'<div class="col-xs-12 col-sm-8 col-md-8">(.*?)</div>', html, re.S)
    if not date or not clock or not match:
        raise ValueError("Official date/time or FOMC statement body unavailable")
    body = unescape(re.sub("<[^>]+>", " ", match.group(1)))
    body = " ".join(body.split()).split("Voting for")[0].split("For media inquiries")[0]
    lower = body.lower().replace("‑", "-").replace("–", "-")
    rate = re.search(
        r"target range for the federal funds rate.{0,100}? (?:at|to) ([\d./\- ]+?) to ([\d./\- ]+?) percent",
        lower,
    )
    if not rate:
        raise ValueError("Official policy rate range unavailable")
    day = datetime.strptime(unescape(date.group(1)).strip(), "%B %d, %Y")
    offset = -5 if clock.group(3) == "EST" else -4
    released = day.replace(
        hour=int(clock.group(1)) + 12,
        minute=int(clock.group(2)),
        tzinfo=timezone(timedelta(hours=offset)),
    ).astimezone(UTC)
    tokens = re.findall("[a-z]+", lower)
    return {
        "published_at": released.isoformat(),
        "available_at": (released + timedelta(minutes=5)).isoformat(),
        "title": "Federal Reserve issues FOMC statement",
        "policy_rate_midpoint": (number(rate.group(1)) + number(rate.group(2))) / 2,
        "hawkish_score": sum(lower.count(x) for x in HAWKISH) - sum(lower.count(x) for x in DOVISH),
        "uncertainty_score": sum(tokens.count(x) for x in UNCERTAIN) / len(tokens) * 100,
        "inflation_emphasis": tokens.count("inflation") / len(tokens) * 100,
        "word_count": len(tokens),
        "source_kind": "official_fomc_release",
        "timestamp_basis": "explicit original release date and EST/EDT clock; fetched today, no historical ingestion receipt",
    }


def collect(output, cache=None):
    output.mkdir(exist_ok=False)

    def fetch(day):
        url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{day}a.htm"
        path = output / (day + ".html")
        try:
            if cache and (cache / (day + ".html")).is_file():
                content = (cache / (day + ".html")).read_bytes()
            else:
                response = requests.get(url, timeout=25)
                response.raise_for_status()
                content = response.content
            path.write_bytes(content)
            item = parse(content.decode("utf-8-sig"))
            if item["published_at"][:10].replace("-", "") != day:
                raise ValueError("Release identity mismatch")
            return {**item, "url": url, "sha256": sha(path), "raw_file": path.name}, None
        except (requests.RequestException, ValueError) as e:
            return None, {"date": day, "url": url, "error": str(e)[:240]}

    results = list(ThreadPoolExecutor(max_workers=4).map(fetch, DATES))
    events = sorted([r for r, e in results if r], key=lambda x: x["published_at"])
    failures = [e for r, e in results if e]
    previous = None
    for event in events:
        event["policy_rate_change"] = (
            None if previous is None else event["policy_rate_midpoint"] - previous
        )
        previous = event["policy_rate_midpoint"]
    save(output / "events.json", events)
    save(
        output / "coverage.json",
        {
            "requested": len(DATES),
            "verified": len(events),
            "failures": failures,
            "scope": "FOMC official statement news; not general crypto news or analyst consensus",
            "cpi_source_status": "BLS archive HTTP403 observed; no CPI or consensus surprise is fabricated",
            "lexicon": "Fixed transparent phrase counts, not a trained sentiment model",
            "retrieved_at": datetime.now(UTC).isoformat(),
        },
    )
    if len(events) < 12:
        raise ValueError("Too few verified official news events for this experiment")
    return events
