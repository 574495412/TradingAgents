"""复用公开接口连接并保留实际到达时钟，禁止账户认证与未登记路径。"""

from __future__ import annotations

import base64
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from urllib.request import getproxies

import requests

from tradingagents.dataflows.research.live_capture import BASE, METRICS, ROUTES, Pacer

ALLOWED = {
    "clock": set(),
    "exchange": set(),
    "quotes": {"symbol"},
    "klines": {"symbol", "interval", "limit", "startTime", "endTime"},
    "index": {"pair", "interval", "limit", "startTime", "endTime"},
    "funding": {"symbol", "limit", "startTime", "endTime"},
    **{k: {"symbol", "period", "limit", "startTime", "endTime"} for k in METRICS},
}


class PublicHTTP:
    """研究客户端独占会话；不读取浏览器、交易凭证、环境认证或 netrc。"""

    def __init__(self, workers=12):
        self.pacer = Pacer()
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.local, self.lock = threading.local(), threading.Lock()
        self.sessions, self.next_funding = [], 0.0
        self.proxies = {k: v for k, v in getproxies().items() if k in ("http", "https", "all")}

    def session(self):
        if not hasattr(self.local, "session"):
            value = requests.Session()
            value.trust_env, value.auth = False, None
            value.headers.clear()
            value.headers.update(
                {"User-Agent": "CZ-Public-Factor-Research", "Accept": "application/json"}
            )
            value.proxies.update(self.proxies)
            self.local.session = value
            with self.lock:
                self.sessions.append(value)
        return self.local.session

    def request(self, kind, symbol, params):
        if kind not in ALLOWED or set(params) - ALLOWED[kind]:
            raise ValueError("公开请求包含未登记路径或认证参数")
        url = BASE + ROUTES[kind] + ("?" + urlencode(params) if params else "")
        weight = (
            0 if kind in METRICS or kind == "funding" else 2 if params.get("limit", 0) >= 100 else 1
        )
        if kind == "quotes":
            weight = 5
        if kind == "funding":
            with self.lock:
                slot = max(time.monotonic(), self.next_funding)
                self.next_funding = slot + 1.05
            if self.pacer.stop.wait(max(0, slot - time.monotonic())):
                return {"kind": kind, "symbol": symbol, "status": "CANCELLED_BEFORE_REQUEST"}
        if not self.pacer.acquire(weight):
            return {"kind": kind, "symbol": symbol, "status": "CANCELLED_BEFORE_REQUEST"}
        started, tick = time.time_ns() // 1_000_000, time.monotonic()
        status, raw, headers, error = 0, b"", {}, None
        session = self.session()
        session.cookies.clear()
        try:
            with session.get(
                url, timeout=(4, 12 if kind == "exchange" else 4), allow_redirects=False
            ) as response:
                status, raw = response.status_code, response.content
                headers = {k.lower(): v for k, v in response.headers.items()}
        except requests.RequestException as exc:
            error = type(exc).__name__
        finally:
            session.cookies.clear()
        received, elapsed = time.time_ns() // 1_000_000, (time.monotonic() - tick) * 1000
        if status in (418, 429):
            self.pacer.stop.set()
        return {
            "kind": kind,
            "symbol": symbol,
            "url": url,
            "status": status,
            "requested_ms": started,
            "received_ms": received,
            "elapsed_monotonic_ms": elapsed,
            "local_clock_consistent": received >= started
            and abs(received - started - elapsed) <= 250,
            "body_base64": base64.b64encode(raw).decode(),
            "body_sha256": hashlib.sha256(raw).hexdigest(),
            "server_http_date": headers.get("date"),
            "used_weight_1m": headers.get("x-mbx-used-weight-1m"),
            "retry_after": headers.get("retry-after"),
            "error": error,
            "credentials_used": False,
            "transport": "pooled_public_http",
            "body_stage": "application_response_after_http_decoding",
        }

    def batch(self, jobs):
        return list(self.executor.map(lambda v: self.request(*v), jobs))

    def close(self):
        self.executor.shutdown(wait=True)
        for session in self.sessions:
            session.close()
