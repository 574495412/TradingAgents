"""通过独立解释器调用 TradingAgents，研究与模型运行时各自固定身份。"""

import json
import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .common import digest


def framework_identity(root):
    root = Path(root) / "tradingagents"
    files = sorted(root.rglob("*.py"))
    if not files:
        raise ValueError("TradingAgents 源码目录为空")
    return digest({str(p.relative_to(root)): digest(p.read_text()) for p in files})


def describe_worker(python):
    """只读取隔离运行时的版本，模型调用前就固定依赖身份。"""
    with tempfile.TemporaryDirectory(prefix="ta-quant-describe-") as folder:
        result = subprocess.run(
            [
                str(Path(python).absolute()),
                "-I",
                "-m",
                "tradingagents.research.worker",
                "--describe",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
            cwd=folder,
        )
    return json.loads(result.stdout)


def invoke(request, python, expected_source):
    started = time.monotonic()
    # 不传递交易密钥、追踪服务或模型服务的环境配置，Codex 使用已有本地登录。
    keep = {
        "PATH",
        "HOME",
        "CODEX_HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    }
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(
        {
            k: v
            for k, v in os.environ.items()
            if k in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
        }
    )
    with tempfile.TemporaryDirectory(prefix="ta-quant-worker-") as folder:
        proc = subprocess.Popen(
            [str(Path(python).absolute()), "-I", "-m", "tradingagents.research.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=folder,
            env=env,
            start_new_session=True,
        )
        timeout = request["timeout_seconds"] * (3 if request["mode"] == "multi" else 1) + 45
        try:
            stdout, _ = proc.communicate(json.dumps(request), timeout=timeout)
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
            raise
        if proc.returncode:
            raise RuntimeError(
                "TradingAgents 研究失败；检查协议、依赖和 codex login status。未生成替代信号。"
            )
    if len(stdout) > 4_000_000:
        raise ValueError("研究输出超过大小限制")
    response = json.loads(stdout)
    validate_response(request, response, expected_source)
    response["worker_elapsed_seconds"] = response["elapsed_seconds"]
    response["elapsed_seconds"] = time.monotonic() - started
    return response


def validate_response(request, response, expected_source):
    if (
        response.get("schema_version") != "ta-evidence-response.v1"
        or response.get("request_sha256") != digest(request)
        or response.get("snapshot_sha256") != digest(request["snapshot"])
        or response.get("backend") != request["backend"]
        or response.get("mode") != request["mode"]
        or response.get("framework_source_sha256") != expected_source
        or response.get("orders_enabled") is not False
        or response.get("external_data_tools_enabled") is not False
        or response.get("persistent_memory_enabled") is not False
    ):
        raise ValueError("研究输出身份或隔离约束不一致")
    for value in (
        response["assessment"]["event_score"],
        response["baseline_score"],
        response["rule_event_score"],
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not -1 <= value <= 1
        ):
            raise ValueError("研究因子不合法")
    allowed = {e["id"] for e in request["snapshot"]["evidence"]}
    cited = response["assessment"]["evidence_ids"]
    if len(cited) != len(set(cited)) or not set(cited) <= allowed:
        raise ValueError("研究结果引用了未知证据")
    if response["assessment"]["event_score"] != 0 and not cited:
        raise ValueError("非零因子缺少来源")
