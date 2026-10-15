"""定义带版本号的公开证据协议，严格检查信息在决策时是否已经可用。"""

from __future__ import annotations

import hashlib
import json
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
        ).encode()
    ).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Evidence(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    kind: Literal["news", "funding", "market"]
    source_url: str = Field(max_length=2000)
    published_at_ms: int = Field(gt=0)
    available_at_ms: int = Field(gt=0)
    availability_basis: Literal["observed", "assumed_publication"]
    expires_at_ms: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=12000)
    text_sha256: str
    # 只有资金费证据允许携带数值，供固定规则对照组使用。
    funding_rate: float | None = None

    @model_validator(mode="after")
    def valid(self):
        if urlparse(self.source_url).scheme not in {"https", "http"}:
            raise ValueError("evidence must identify a public HTTP source")
        if hashlib.sha256(self.text.encode()).hexdigest() != self.text_sha256:
            raise ValueError("evidence text hash mismatch")
        if not self.published_at_ms <= self.available_at_ms < self.expires_at_ms:
            raise ValueError("invalid evidence availability interval")
        if self.kind != "funding" and self.funding_rate is not None:
            raise ValueError("funding value on non-funding evidence")
        if self.funding_rate is not None and abs(self.funding_rate) > 1:
            raise ValueError("funding rate must be a fraction")
        return self


class Features(StrictModel):
    momentum: float = Field(ge=-1, le=1)
    flow: float = Field(ge=-1, le=1)
    crowding: float = Field(ge=-1, le=1)
    activity: float = Field(ge=0, le=1)


class Snapshot(StrictModel):
    symbol: str = Field(pattern=r"^[A-Z0-9._-]{1,40}$")
    as_of_ms: int = Field(gt=0)
    features_available_at_ms: int = Field(gt=0)
    availability_basis: Literal["observed", "assumed_publication"]
    valid_until_ms: int = Field(gt=0)
    features: Features
    evidence: list[Evidence] = Field(max_length=20)
    data_gaps: list[str] = Field(max_length=20)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def causal(self):
        if not self.features_available_at_ms <= self.as_of_ms < self.valid_until_ms:
            raise ValueError("features are unavailable or snapshot has expired")
        ids = [e.id for e in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evidence identifiers")
        for e in self.evidence:
            if not e.available_at_ms <= self.as_of_ms < e.expires_at_ms:
                raise ValueError("future or expired evidence")
        return self


class Request(StrictModel):
    schema_version: Literal["ta-evidence-request.v1"]
    mode: Literal["single", "multi"]
    backend: Literal["codex", "fixture"]
    snapshot: Snapshot
    model: str | None = Field(default=None, max_length=100)
    timeout_seconds: int = Field(default=180, ge=1, le=600)


class Assessment(StrictModel):
    # 这是带方向的研究因子，不表示经过校准的概率或下单数量。
    event_score: float = Field(ge=-1, le=1)
    rationale: str = Field(min_length=1, max_length=4000)
    hypothesis: str = Field(min_length=1, max_length=2000)
    invalidation: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(max_length=20)
    missing_data: list[str] = Field(max_length=20)

    def validate_evidence(self, snapshot: Snapshot):
        allowed = {e.id for e in snapshot.evidence}
        if (
            len(self.evidence_ids) != len(set(self.evidence_ids))
            or not set(self.evidence_ids) <= allowed
        ):
            raise ValueError("assessment cites unknown or duplicate evidence")
        if self.event_score != 0 and not self.evidence_ids:
            raise ValueError("nonzero evidence score requires a source")
        return self


def rule_score(snapshot: Snapshot) -> float:
    """固定的简单对照规则，不接收模型生成的标注。"""
    values = []
    for e in snapshot.evidence:
        if e.kind == "funding" and e.funding_rate is not None:
            values.append(max(-1.0, min(1.0, -e.funding_rate / 0.001)))
        elif e.kind == "news":
            words = set(e.text.lower().replace("-", " ").split())
            positive = len(words & {"approved", "approval", "launch", "launched", "upgrade"})
            negative = len(words & {"hack", "exploit", "suspended", "delisting", "bankruptcy"})
            values.append(float((positive > negative) - (negative > positive)))
    return sum(values) / len(values) if values else 0.0
