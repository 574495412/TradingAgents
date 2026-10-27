"""One no-key first-of-month, one-minute exchange-native historical option sample."""

import argparse
import gzip
import time
from pathlib import Path

import requests

from research_tools.factor_lab.io import read, save, sha


def collect(output):
    output.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).with_name("historical_witness_plan.json")
    plan = read(path)
    save(output / "plan.json", plan)
    save(
        output / "registration.json",
        {
            "wall_ns": time.time_ns(),
            "source_hashes": {
                str(p): sha(p)
                for p in [path, Path(__file__), Path(plan["parent"]) / "completion.json"]
            },
            "before_new_historical_witness_access": True,
            "economic_evaluation_allowed": False,
        },
    )
    rows = []
    for item in plan["requests"]:
        receipt = {**item, "sent_wall_ns": time.time_ns(), "status": None}
        try:
            with requests.get(
                item["url"],
                params=item.get("params"),
                timeout=(10, 25),
                allow_redirects=False,
                stream=True,
                headers={"Accept-Encoding": "gzip"},
            ) as response:
                receipt.update(
                    status=response.status_code,
                    response_url=response.url,
                    headers={
                        k: response.headers[k]
                        for k in [
                            "Content-Encoding",
                            "Content-Length",
                            "Content-Type",
                            "Date",
                            "ETag",
                            "Location",
                        ]
                        if k in response.headers
                    },
                )
                body = response.raw.read(
                    plan["budget"]["maximum_wire_bytes_each"] + 1, decode_content=False
                )
            if len(body) > plan["budget"]["maximum_wire_bytes_each"]:
                raise ValueError("Source wire bytes exceed registered cap")
            pathbody = output / (item["id"] + ".body")
            pathbody.write_bytes(body)
            receipt.update(bytes=len(body), body_sha256=sha(pathbody))
            if receipt["headers"].get("Content-Encoding") == "gzip" or body.startswith(b"\x1f\x8b"):
                import io

                with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                    decoded = stream.read(plan["budget"]["maximum_decoded_bytes"] + 1)
            else:
                decoded = body
            if len(decoded) > plan["budget"]["maximum_decoded_bytes"]:
                raise ValueError("Decoded bytes exceed registered cap")
            target = output / (item["id"] + ".decoded")
            target.write_bytes(decoded)
            receipt.update(decoded_bytes=len(decoded), decoded_sha256=sha(target))
        except Exception as error:
            receipt.update(error_type=type(error).__name__, error=str(error)[:200])
        receipt["received_wall_ns"] = time.time_ns()
        save(output / (item["id"] + ".receipt.json"), receipt)
        rows.append(
            {k: receipt.get(k) for k in ["id", "status", "bytes", "decoded_bytes", "error_type"]}
        )
        print(rows[-1], flush=True)
    save(
        output / "summary.json",
        {
            "results": rows,
            "HTTP_requests": len(rows),
            "new_financial_accounts": 0,
            "economic_evaluation_allowed": False,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args().output)
