"""冻结四组实验、缓存真实研究输出，并用统一账户独立比较。"""

import os
import platform
import time
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

from .account import Protocol, replay, validate_market
from .bridge import describe_worker, framework_identity, invoke, validate_response
from .common import digest, file_hash, read, save, seal, source_identity

ARMS = ("baseline", "fixed_rule", "single", "multi")


def scores_from_replies(replies):
    base = replies["single"]["baseline_score"]
    if (
        base != replies["multi"]["baseline_score"]
        or replies["single"]["rule_event_score"] != replies["multi"]["rule_event_score"]
    ):
        raise ValueError("两个模型组没有使用相同的控制信息")
    return base, {
        "baseline": 0.0,
        "fixed_rule": replies["single"]["rule_event_score"],
        **{mode: replies[mode]["assessment"]["event_score"] for mode in ("single", "multi")},
    }


def decision(snapshot, score, protocol):
    score = max(-1.0, min(1.0, score))
    ready = snapshot["as_of_ms"] + protocol.decision_latency_ms
    if ready >= snapshot["valid_until_ms"]:
        raise ValueError("预设推理延迟超过信号有效期")
    return {
        "symbol": snapshot["symbol"],
        "as_of_ms": snapshot["as_of_ms"],
        "available_ms": ready,
        "expires_ms": snapshot["valid_until_ms"],
        "score": score,
        "direction": (1 if score > 0 else -1) if abs(score) >= protocol.signal_threshold else 0,
        "rank": abs(score) * snapshot["features"]["activity"],
    }


def run(
    dataset_path,
    output,
    ta_root,
    ta_python,
    backend="fixture",
    model=None,
    protocol=None,
    timeout_seconds=180,
    resume=False,
):
    protocol = protocol or Protocol()
    if (
        backend not in {"fixture", "codex"}
        or type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= 600
    ):
        raise ValueError("模型后端或超时时间不合法")
    if backend == "codex" and model is None:
        config = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
        model = tomllib.loads(config.read_text()).get("model") if config.is_file() else None
        if not model:
            raise ValueError("请通过 --model 固定模型，或先在 Codex 配置中设置 model")
    dataset_path, output = Path(dataset_path), Path(output)
    dataset = read(dataset_path)
    _, clocks, _ = validate_market(dataset)
    snapshots = sorted(dataset["snapshots"], key=lambda s: (s["as_of_ms"], s["symbol"]))
    keys = [(s["as_of_ms"], s["symbol"]) for s in snapshots]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("实验必须包含唯一的决策快照")
    if any(
        s["symbol"] not in dataset["symbols"] or not clocks[0] <= s["as_of_ms"] < clocks[-1]
        for s in snapshots
    ):
        raise ValueError("决策快照超出连续账户范围")
    ta_source = framework_identity(ta_root)
    runtime = describe_worker(ta_python)
    if runtime["framework_source_sha256"] != ta_source:
        raise ValueError("指定的 TradingAgents 源码与隔离解释器实际导入的版本不同")
    ta_package = Path(ta_root) / "tradingagents"
    source_files = {}
    source_files.update(
        {f"tradingagents/{p.relative_to(ta_package)}": p for p in ta_package.rglob("*.py")}
    )
    freeze = {
        "schema_version": "ta-quant-comparison-freeze.v1",
        "dataset_sha256": file_hash(dataset_path),
        "dataset_content_sha256": digest(dataset),
        "research_source_sha256": source_identity(),
        "ta_source_sha256": ta_source,
        "worker_runtime": runtime["runtime"],
        "source_files": {name: file_hash(path) for name, path in sorted(source_files.items())},
        "python": platform.python_version(),
        "protocol": asdict(protocol),
        "backend": backend,
        "model_requested": model,
        "timeout_seconds": timeout_seconds,
        "arms": list(ARMS),
        "snapshot_keys": keys,
        "decision_schema": "ta-evidence-request.v1",
        "source_availability": dataset["provenance"],
        "independent_oos": False,
        "production_eligible": False,
    }
    # JSON 规范化后比较，避免恢复时元组和列表的表示差异。
    if resume:
        if digest(read(output / "freeze.json")) != digest(freeze):
            raise ValueError("源码、数据、模型或协议发生变化，必须开始新实验")
        if (output / "result.json").exists():
            verify(output)
            return read(output / "result.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save(output / "freeze.json", freeze)
        save(output / "dataset.json", dataset)
        (output / "responses").mkdir()
        for name, path in source_files.items():
            target = output / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(path.read_bytes())
    decisions = {arm: [] for arm in ARMS}
    receipts, latencies, usages = [], {"single": [], "multi": []}, {"single": [], "multi": []}
    started = time.monotonic()
    for index, snapshot in enumerate(snapshots):
        replies = {}
        for mode in ("single", "multi"):
            request = {
                "schema_version": "ta-evidence-request.v1",
                "mode": mode,
                "backend": backend,
                "snapshot": snapshot,
                "model": model,
                "timeout_seconds": timeout_seconds,
            }
            key = digest(request)
            path = output / "responses" / f"{key}.json"
            if path.exists():
                cached = read(path)
                if cached["response_sha256"] != digest(cached["response"]):
                    raise ValueError("缓存研究输出被修改")
                reply = cached["response"]
                validate_response(request, reply, ta_source)
            else:
                print(f"研究 {index + 1}/{len(snapshots)} {snapshot['symbol']} {mode}", flush=True)
                try:
                    reply = invoke(request, ta_python, ta_source)
                except Exception as exc:
                    save(
                        output / f"failure-{time.time_ns()}.json",
                        {
                            "request_sha256": key,
                            "error_type": type(exc).__name__,
                            "decision_emitted": False,
                        },
                    )
                    raise
                save(
                    path, {"request": request, "response": reply, "response_sha256": digest(reply)}
                )
            if reply["elapsed_seconds"] * 1000 > protocol.decision_latency_ms:
                raise ValueError("实际推理超过事前延迟预算，不能用于本轮模拟成交")
            if reply["runtime"] != freeze["worker_runtime"]:
                raise ValueError("隔离解释器运行时在实验中发生变化")
            replies[mode] = reply
            receipts.append({"path": str(path.relative_to(output)), "sha256": file_hash(path)})
            latencies[mode].append(reply["elapsed_seconds"])
            usages[mode].extend(call["usage"] for call in reply["calls"])
        base, overlays = scores_from_replies(replies)
        for arm, overlay in overlays.items():
            decisions[arm].append(
                decision(snapshot, base + protocol.event_weight * overlay, protocol)
            )
    seal(output / "decisions.json", decisions)
    summaries, outputs = {}, []
    for arm in ARMS:
        for cost_name, p in (("base", protocol), ("stress", replace(protocol, slippage_bps=10.0))):
            account = replay(dataset, decisions[arm], p)
            name = f"{arm}_{cost_name}.json"
            seal(output / name, account)
            outputs.append({"path": name, "sha256": file_hash(output / name)})
            summaries[f"{arm}_{cost_name}"] = account["metrics"]
    results = {
        "schema_version": "ta-quant-comparison-result.v1",
        "freeze_sha256": file_hash(output / "freeze.json"),
        "dataset_copy_sha256": file_hash(output / "dataset.json"),
        "decisions_sha256": file_hash(output / "decisions.json"),
        "responses": receipts,
        "outputs": outputs,
        "metrics": summaries,
        "elapsed_seconds": time.monotonic() - started,
        "latency_seconds": latencies,
        "usage": usages,
        "backend": backend,
        "model_inference_performed": backend == "codex",
        "inference_monetary_cost": None,
        "independent_oos": False,
        "production_eligible": False,
        "limitations": [
            "历史模型知识可能包含未来，日期过滤不能证明独立样本外",
            "当前是固定因子研究基线，不代表完整 Omofun 或 DAmorne",
            "连续数量与固定滑点尚未验证真实成交和历史最小下单量",
            "最大回撤按 K 线收盘估值，不是精确盘中最大回撤",
            "Codex 调用成本未换算为金额，需结合账户订阅与用量评价",
        ],
    }
    save(output / "result.json", results)
    verify(output)
    return results


def verify(output):
    """核验所有封存文件，并重新重放八个账户；无需再次调用模型。"""
    output = Path(output)
    result, freeze = read(output / "result.json"), read(output / "freeze.json")
    if source_identity() != freeze["research_source_sha256"]:
        raise ValueError("复核必须使用实验冻结的 TradingAgents 对照研究源码")
    expected = {
        "freeze.json": result["freeze_sha256"],
        "dataset.json": result["dataset_copy_sha256"],
        "decisions.json": result["decisions_sha256"],
    }
    expected.update(
        {item["path"]: item["sha256"] for item in result["outputs"] + result["responses"]}
    )
    expected.update(
        {f"sources/{name}": checksum for name, checksum in freeze["source_files"].items()}
    )
    for relative, checksum in expected.items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()) or file_hash(path) != checksum:
            raise ValueError("实验文件哈希或目录边界不匹配")
    dataset, decisions = read(output / "dataset.json"), read(output / "decisions.json")
    if digest(dataset) != freeze["dataset_content_sha256"]:
        raise ValueError("数据内容与事前冻结不一致")
    protocol = Protocol(**freeze["protocol"])
    # 根据封存的同一组输入与研究输出重新生成决策，防止只核对账户却漏查信号。
    expected_decisions = {arm: [] for arm in ARMS}
    for snapshot in sorted(dataset["snapshots"], key=lambda s: (s["as_of_ms"], s["symbol"])):
        replies = {}
        for mode in ("single", "multi"):
            request = {
                "schema_version": "ta-evidence-request.v1",
                "mode": mode,
                "backend": freeze["backend"],
                "snapshot": snapshot,
                "model": freeze["model_requested"],
                "timeout_seconds": freeze["timeout_seconds"],
            }
            cached = read(output / "responses" / f"{digest(request)}.json")
            if cached["request"] != request or cached["response_sha256"] != digest(
                cached["response"]
            ):
                raise ValueError("研究输入输出绑定不一致")
            validate_response(request, cached["response"], freeze["ta_source_sha256"])
            if cached["response"]["runtime"] != freeze["worker_runtime"]:
                raise ValueError("研究结果的运行时与冻结记录不一致")
            replies[mode] = cached["response"]
        base, overlays = scores_from_replies(replies)
        for arm, overlay in overlays.items():
            expected_decisions[arm].append(
                decision(snapshot, base + protocol.event_weight * overlay, protocol)
            )
    if digest(expected_decisions) != digest(decisions):
        raise ValueError("决策没有忠实使用封存的模型输出")
    for arm in ARMS:
        for cost_name, p in (("base", protocol), ("stress", replace(protocol, slippage_bps=10.0))):
            name = f"{arm}_{cost_name}"
            account = replay(dataset, decisions[arm], p)
            if (
                digest(account) != digest(read(output / f"{name}.json"))
                or account["metrics"] != result["metrics"][name]
            ):
                raise ValueError("账户重放与封存记录不一致")
    return {"verified_accounts": 8, "verified_files": len(expected), "model_calls": 0}
