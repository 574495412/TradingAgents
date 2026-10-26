"""Pinned external XBRL tool, run locally; not an external audit attestation.

The worker emits Arelle's actual facts and validation logs without importing
the research parser. It does not evaluate formulas or modify input files.
"""

import argparse
import json
from pathlib import Path

from arelle.api.Session import Session
from arelle.RuntimeOptions import RuntimeOptions


def run(entry, output, cache, plugin):
    options = RuntimeOptions(
        entrypointFile=str(entry.resolve()),
        keepOpen=True,
        validate=True,
        formulaAction="none",
        plugins=str(plugin.resolve()),
        logLevel="warning",
        logFile="logToBuffer",
        disablePersistentConfig=True,
        xdgConfigHome=str((cache / "config").resolve()),
        cacheDirectory=str((cache / "web").resolve()),
        internetTimeout=12,
        internetRecheck="never",
        httpUserAgent="CZ-Agent/EquityResearch (public financial research)",
        httpsRedirectCache=True,
    )
    with Session() as session:
        session.run(options)
        rows, models = [], []
        tags = {
            "Assets",
            "GrossProfit",
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",
            "NetIncomeLoss",
        }
        for model in session.get_models():
            models.append(
                {
                    "loaded": model.modelDocument is not None,
                    "errors": [str(e) for e in model.errors],
                    "fact_count": len(model.factsInInstance),
                    "document_count": len(model.urlDocs),
                }
            )
            for fact in model.factsInInstance:
                if fact.qname is None or fact.qname.localName not in tags:
                    continue
                context, unit = fact.context, fact.unit
                period = (
                    context.find("{http://www.xbrl.org/2003/instance}period")
                    if context is not None
                    else None
                )
                rows.append(
                    {
                        "namespace": fact.qname.namespaceURI,
                        "tag": fact.qname.localName,
                        "raw_value": str(fact.value),
                        "value": str(getattr(fact, "xValue", None)),
                        "effective_value": str(fact.effectiveValue),
                        "is_nil": fact.isNil,
                        "validity": fact.xValid,
                        "context_id": fact.contextID,
                        "entity": list(context.entityIdentifier) if context is not None else None,
                        "period_raw": {str(c.tag).split("}")[-1]: c.text for c in period}
                        if period is not None
                        else None,
                        "dimensions": sorted(str(q) for q in context.qnameDims)
                        if context is not None
                        else None,
                        "non_dimensional_segment": bool(context.nonDimValues("segment"))
                        if context is not None
                        else None,
                        "non_dimensional_scenario": bool(context.nonDimValues("scenario"))
                        if context is not None
                        else None,
                        "unit": [
                            [{"namespace": q.namespaceURI, "local_name": q.localName} for q in part]
                            for part in unit.measures
                        ]
                        if unit is not None
                        else None,
                        "decimals": fact.decimals,
                        "source_line": fact.sourceline,
                    }
                )
        log = session.get_logs("json")
        log_data = json.loads(log) if isinstance(log, str) else log
        error_log = [
            r
            for r in log_data.get("log", [])
            if r.get("level", "").lower() in {"error", "critical", "fatal"}
        ]
        result = {
            "schema": "equity-arelle-facts.v1",
            "tool": "arelle-release==2.47.3",
            "models": models,
            "rows": rows,
            "validation_log": log_data,
            "full_validation_passed": bool(models)
            and all(m["loaded"] and m["fact_count"] and not m["errors"] for m in models)
            and not error_log,
            "external_human_attestation": False,
        }
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(
            entry.name,
            "facts",
            len(rows),
            "errors",
            sum(len(m["errors"]) for m in models),
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--plugin", type=Path, required=True)
    a = p.parse_args()
    run(a.entry, a.output, a.cache, a.plugin)
