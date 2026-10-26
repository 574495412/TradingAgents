"""Capture a bounded identity/type correction scope before return evaluation."""

import argparse
import json
import shutil
from pathlib import Path

from research_tools.contract_lifecycles.collect import request, text_nodes
from research_tools.factor_lab.io import read, save, sha


def capture(root):
    root.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("plan.json")
    plan = read(plan_path)
    parent = Path(plan["adopted_type_sources"])
    save(
        root / "registration.json",
        {
            "study_plan_sha256": sha(plan_path),
            "collector_sha256": sha(Path(__file__)),
            "new_detail_codes": plan["new_detail_codes"],
            "before_new_requests": True,
        },
    )
    adopted = []
    for code in [
        "d592f6ba938746cbadaaf5a8a714abd6",
        "f083bdc8f37d4d45a0e0ec6fd5cd4086",
        "bd92dee1018f421fae1ace119fc2f706",
    ]:
        receipt = read(parent / (code + ".receipt.json"))
        if sha(parent / (code + ".raw")) != receipt["body_sha256"] or receipt["status"] != 200:
            raise ValueError("Adopted type body changed")
        for suffix in [".raw", ".receipt.json", ".txt"]:
            shutil.copyfile(parent / (code + suffix), root / (code + suffix))
        adopted.append(code)
    details = []
    if len(plan["new_detail_codes"]) > plan["maximum_new_detail_requests"]:
        raise ValueError("Detail request budget exceeded")
    for code in plan["new_detail_codes"]:
        data = request(
            root,
            code,
            "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query",
            {"articleCode": code},
        )
        text = text_nodes(json.loads(data["body"]))
        (root / (code + ".txt")).write_text(text)
        details.append(
            {
                "code": code,
                "title": data["title"],
                "publishDate": data["publishDate"],
                "text_sha256": sha(root / (code + ".txt")),
                "current_body_only": True,
            }
        )
    save(root / "details.json", details)
    save(
        root / "completion.json",
        {
            "manifest": {p.name: sha(p) for p in root.iterdir() if p.is_file()},
            "new_requests": len(details),
            "adopted_details": adopted,
            "historical_original_versions_certified": False,
            "financial_accounts": 0,
        },
    )
    print(
        {
            "new_details": len(details),
            "adopted_type_details": len(adopted),
            "financial_accounts": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    capture(p.parse_args().output)
