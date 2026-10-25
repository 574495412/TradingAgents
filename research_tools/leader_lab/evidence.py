"""Normalize public summaries without manufacturing fills, equity or private protection orders."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

PUBLIC_BASE = "https://www.binance.com/bapi/futures/v1/friendly/future/copy-trade/"
KNOWN = {"damorne": "5219911688676500225"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def finite(value, *, positive=False) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric observation")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError("Nonfinite or nonpositive observation")
    return result


def clock(value) -> int:
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        number = finite(value)
        if not number.is_integer() or number < 1_000_000_000_000:
            raise ValueError("Numeric timestamps must be UTC epoch milliseconds")
        return int(number)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp requires an explicit timezone")
    return int(parsed.timestamp() * 1000)


def identity(text: str) -> str:
    text = text.strip()
    normalized = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    ).lower()
    if normalized in KNOWN:
        return KNOWN[normalized]
    if text.isdigit() and 6 <= len(text) <= 30:
        return text
    url = urllib.parse.urlparse(text)
    if (
        url.scheme != "https"
        or url.hostname not in {"www.binance.com", "binance.com"}
        or url.username
        or url.password
    ):
        raise ValueError("Use a Binance leader URL/ID, or a registered local nickname")
    parts = url.path.strip("/").split("/")
    if len(parts) < 3 or parts[-2] != "lead-details" or not parts[-1].isdigit():
        raise ValueError("URL is not a Binance copy-trading leader detail URL")
    return identity(parts[-1])


def resolve_identity(text: str):
    """Resolve a unique public nickname; ambiguous/hidden names require their public ID."""
    try:
        return identity(text), {"method": "registered_nickname_or_explicit_url_id"}
    except ValueError:
        if ":" in text or text.strip().isdigit() or not 1 <= len(text.strip()) <= 80:
            raise
    payload = {
        "nickname": text.strip(),
        "portfolioType": "PUBLIC",
        "timeRange": "30D",
        "dataType": "ROI",
        "order": "DESC",
        "pageNumber": 1,
        "pageSize": 50,
        "favoriteOnly": False,
        "hideFull": False,
        "useAiRecommended": False,
        "userAsset": 0,
    }
    request = urllib.request.Request(
        PUBLIC_BASE + "home-page/query-list",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "LeaderLab-read-only/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError("Nickname lookup exceeds response bound")
    body = json.loads(raw)
    if body.get("code") != "000000" or not isinstance(body.get("data"), dict):
        raise ValueError("Public nickname lookup unavailable; use the leader URL/ID")
    rows = body["data"].get("list", [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Invalid public nickname response")

    def name(s):
        return (
            "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
            .casefold()
            .strip()
        )

    matches = [row for row in rows if name(str(row.get("nickname", ""))) == name(text)]
    if len(matches) != 1:
        raise ValueError("Public nickname is absent or ambiguous; use its exact leader URL/ID")
    leader = identity(str(matches[0].get("leadPortfolioId", "")))
    return leader, {"method": "public_exact_nickname_match", "raw_response": body}


@dataclass(frozen=True)
class Position:
    position_id: str
    symbol: str
    side: str
    opened_ms: int
    closed_ms: int
    entry_price: float
    exit_price: float
    quantity: float
    max_quantity: float
    leverage: float
    reported_pnl: float

    @property
    def notional_proxy(self):
        return self.entry_price * self.max_quantity


def normalize(rows) -> list[Position]:
    result, seen = [], {}
    for row in rows:
        raw = row.get("raw", row)
        p = Position(
            str(raw.get("positionId", raw.get("position_id", raw.get("id", "")))),
            str(raw["symbol"]),
            str(raw.get("side", raw.get("direction", ""))).upper(),
            clock(raw.get("opened", raw.get("opened_ms"))),
            clock(raw.get("closed", raw.get("closed_ms"))),
            finite(raw.get("avgCost", raw.get("entry_price")), positive=True),
            finite(raw.get("avgClosePrice", raw.get("exit_price")), positive=True),
            finite(raw.get("closedVolume", raw.get("quantity")), positive=True),
            finite(
                raw.get("maxOpenInterest", raw.get("max_quantity", raw.get("quantity"))),
                positive=True,
            ),
            finite(raw.get("leverage", 1), positive=True),
            finite(raw.get("closingPnl", raw.get("reported_pnl"))),
        )
        if not p.position_id or p.side not in {"LONG", "SHORT"} or p.closed_ms < p.opened_ms:
            raise ValueError("Invalid identity, direction or position clock")
        finite(p.notional_proxy, positive=True)
        if p.position_id in seen:
            if seen[p.position_id] != p:
                raise ValueError("Conflicting duplicate position identity")
            raise ValueError("Repeated position identity; source pages must be reconciled first")
        seen[p.position_id] = p
        result.append(p)
    if not result:
        raise ValueError("No closed positions available; missing history is not a flat account")
    return sorted(result, key=lambda p: (p.opened_ms, p.position_id))


def load(path: Path, leader_id: str):
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return normalize(rows), {
            "kind": "user_csv",
            "coverage": "not_attested",
            "source_sha256": sha(path),
        }
    document = json.loads(path.read_text())
    if isinstance(document, list):
        return normalize(document), {
            "kind": "position_list",
            "coverage": "not_attested",
            "source_sha256": sha(path),
        }
    binding = str(
        document.get(
            "portfolio_id", document.get("_capture", {}).get("request", {}).get("portfolioId", "")
        )
    )
    if binding and binding != leader_id:
        raise ValueError("Source portfolio does not match requested leader")
    if "closed_positions" in document:
        rows = document["closed_positions"]
        info = {
            "kind": "retained_public_cache",
            "coverage": document.get("coverage", {}),
            "leader_binding_present": bool(binding),
        }
    elif isinstance(document.get("data"), dict):
        if document.get("code") != "000000":
            raise ValueError("Unsuccessful public API response")
        rows = document["data"]["list"]
        total = document["data"].get("total")
        if not isinstance(total, int) or isinstance(total, bool) or total != len(rows):
            raise ValueError("Incomplete declared position-history page")
        info = {
            "kind": "public_position_response",
            "coverage": "complete_closed_as_declared_only",
            "leader_binding_present": bool(binding),
            "captured_at": document.get("_capture", {}).get("captured_at"),
        }
    else:
        raise ValueError("Unsupported source format")
    info["source_sha256"] = sha(path)
    return normalize(rows), info


def public_capture(leader_id: str, output: Path, max_pages=20):
    """Public website endpoint only; no credentials/cookies or order calls, bounded pagination."""
    rows, identities, total = [], set(), None
    for page in range(1, max_pages + 1):
        payload = {"portfolioId": leader_id, "pageNumber": page, "pageSize": 200, "sort": "OPENING"}
        request = urllib.request.Request(
            PUBLIC_BASE + "lead-portfolio/position-history",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "LeaderLab-read-only/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read(4_000_001)
        if len(body) > 4_000_000:
            raise ValueError("Public response exceeds capture bound")
        (output / f"public_page_{page:03d}.json").write_bytes(body)
        raw = json.loads(body)
        if raw.get("code") != "000000" or not isinstance(raw.get("data"), dict):
            raise ValueError("Public history unavailable; preserved response is not empty history")
        data = raw["data"]
        declared, batch = data.get("total"), data.get("list")
        if (
            not isinstance(declared, int)
            or isinstance(declared, bool)
            or not isinstance(batch, list)
        ):
            raise ValueError("Invalid public history page")
        if total is not None and declared != total:
            raise ValueError("History changed while paginating; take a new snapshot")
        total = declared
        for row in batch:
            key = str(row.get("positionId", ""))
            if not key or key in identities:
                raise ValueError("Repeated or missing identity in public pagination")
            if row.get("status") != "All Closed":
                raise ValueError("Public closed-history endpoint returned a non-closed record")
            identities.add(key)
        rows.extend(batch)
        if len(rows) == total:
            path = output / "public_snapshot.json"
            save(
                path,
                {
                    "code": "000000",
                    "data": {"list": rows, "total": total},
                    "_capture": {
                        "request": {"portfolioId": leader_id},
                        "captured_at": datetime.now(UTC).isoformat(),
                        "closed_only": True,
                        "credentials_used": False,
                    },
                },
            )
            return path
        if not batch or len(rows) > total:
            raise ValueError("Pagination stalled or exceeded declared history")
    raise ValueError("Public history exceeds bounded pagination; snapshot incomplete")


def source_for(leader_id: str, source: Path | None, output: Path, refresh=False):
    if source is not None:
        return source.absolute(), "explicit_source"
    root = Path(__file__).resolve().parents[2]
    cached = (
        root
        / "data/research/damorne_15d_alignment_20261014/cache_recovery/cached_public_evidence_002.json"
    )
    if leader_id == KNOWN["damorne"] and cached.exists() and not refresh:
        return cached, "retained_local_cache_not_live"
    return public_capture(leader_id, output), "fresh_public_capture"


def profile(positions: list[Position], gap_hours=12):
    if not math.isfinite(gap_hours) or gap_hours <= 0:
        raise ValueError("Episode gap must be explicitly positive")
    gaps = gap_hours * 3_600_000
    active_until, episodes, transitions, episode_by_id = {}, 0, [], {}
    symbol_episode = {}
    opposite_overlap = 0
    for p in positions:
        if p.opened_ms > active_until.get(p.symbol, -math.inf) + gaps:
            episodes += 1
            symbol_episode[p.symbol] = episodes
        episode_by_id[p.position_id] = symbol_episode[p.symbol]
        active_until[p.symbol] = max(active_until.get(p.symbol, 0), p.closed_ms)
        opposite_overlap += any(
            q.symbol == p.symbol and q.side != p.side and q.opened_ms < p.opened_ms < q.closed_ms
            for q in positions
        )
        earlier = [
            q
            for q in positions
            if q.symbol == p.symbol and q.side == p.side and q.closed_ms < p.opened_ms
        ]
        if earlier:
            previous = max(earlier, key=lambda q: q.closed_ms)
            transitions.append(
                {
                    "position_id": p.position_id,
                    "previous_id": previous.position_id,
                    "gap_ms": p.opened_ms - previous.closed_ms,
                    "previous_reported_pnl_positive": previous.reported_pnl > 0,
                    "same_assumed_episode": episode_by_id[previous.position_id]
                    == episode_by_id[p.position_id],
                    "more_favorable_entry_than_previous_exit": p.entry_price < previous.exit_price
                    if p.side == "LONG"
                    else p.entry_price > previous.exit_price,
                }
            )
    values = sorted(p.notional_proxy for p in positions)
    modes = None
    if len(values) >= 10:
        choices = [(values[i] / values[i - 1], i) for i in range(5, len(values) - 4)]
        ratio, index = max(choices)
        if ratio >= 10:
            modes = {
                "observed_gap_ratio": ratio,
                "descriptive_split_notional": math.sqrt(values[index] * values[index - 1]),
                "lower_count": index,
                "upper_count": len(values) - index,
                "inferred_large_entry_rule": False,
            }
    pnl = [p.reported_pnl for p in positions]
    positive = sum(t["previous_reported_pnl_positive"] for t in transitions)
    overlap_pairs = sum(
        p.symbol == q.symbol
        and p.side != q.side
        and max(p.opened_ms, q.opened_ms) < min(p.closed_ms, q.closed_ms)
        for i, p in enumerate(positions)
        for q in positions[:i]
    )
    local_transitions = [t for t in transitions if t["same_assumed_episode"]]
    return {
        "positions": len(positions),
        "symbols": sorted({p.symbol for p in positions}),
        "period_open_start_ms": positions[0].opened_ms,
        "period_close_end_ms": max(p.closed_ms for p in positions),
        "reported_closed_pnl_sum": math.fsum(pnl),
        "reported_closed_win_rate": sum(x > 0 for x in pnl) / len(pnl),
        "median_holding_minutes": statistics.median(
            (p.closed_ms - p.opened_ms) / 60000 for p in positions
        ),
        "notional_proxy_modes": modes,
        "openings_with_prior_opposite_side_active": opposite_overlap,
        "opposite_side_overlapping_position_pairs": overlap_pairs,
        "episodes_under_explicit_gap_assumption": episodes,
        "episode_gap_hours_assumption_not_learned": gap_hours,
        "same_side_prior_closed_transitions": len(transitions),
        "prior_reported_profit_transitions": positive,
        "counterexamples_to_universal_profit_only_reentry": len(transitions) - positive,
        "same_assumed_episode_transitions": len(local_transitions),
        "same_assumed_episode_prior_profit_transitions": sum(
            t["previous_reported_pnl_positive"] for t in local_transitions
        ),
        "transition_observations": transitions,
        "quant_strategy_identified": None,
        "original_entry_or_scale_rule_recovered": False,
        "reported_pnl_is_not_recomputed_net_account_return": True,
        "closed_history_is_not_complete_action_or_hold_labels": True,
    }


def write_profile(output: Path, leader_id: str, positions, source_info, gap_hours):
    normalized = output / "positions_normalized.json"
    save(normalized, [asdict(p) for p in positions])
    result = profile(positions, gap_hours)
    save(output / "behavior_profile.json", result)
    save(
        output / "data_quality.json",
        {
            "leader_id": leader_id,
            "source": source_info,
            "same_symbol_hedging_observed": result["opposite_side_overlapping_position_pairs"] > 0,
            "asset_class_per_symbol": "Not inferred from USDT suffix; TradFi and crypto require source-specific classification",
            "automation_or_original_quant_model_identified": False,
            "candidate_rule_learning_ready": False,
            "why": "Closed-position summaries omit unclosed inventory and unexecuted actions; they do not supply observed HOLD negatives, complete fills or causal market states.",
            "next_supported_input": "An observed decision panel with prior available market features, explicit actions/HOLD, episode groups and action horizon end timestamps.",
            "orders_enabled": False,
            "production_eligible": False,
        },
    )
    lines = [
        f"# 带单员行为研究：{leader_id}",
        "",
        "当前阶段：可观察行为归纳；尚未恢复自主策略。",
        "",
        f"- 已平仓汇总：{len(positions)}；品种：{len(result['symbols'])}。",
        f"- 同币反向仓位仍持有时的开仓：{result['openings_with_prior_opposite_side_active']}次。",
        f"- 已知此前同向平仓的后续开仓：{len(result['transition_observations'])}次；此前记录盈利：{result['prior_reported_profit_transitions']}次。",
        f"- “任何同向重开都必须此前盈利”的反例：{result['counterexamples_to_universal_profit_only_reentry']}次。",
        f"- 按明确给定的{gap_hours}小时空仓切分：{result['episodes_under_explicit_gap_assumption']}个观察episode；该时限不是从带单员还原的参数。",
        "",
        "仓位分档、持仓间隔和双向行为都是描述性证据；首次选币、报价、大额准入、真实权益及保护订单仍未知。已平仓PNL之和不等于账户ROI、夏普或最大回撤。",
        "",
        "闭仓汇总不会自动转换为完整动作轨迹。需要明确观察到的市场状态和HOLD／开平仓动作后，learn入口才能学习可解释规则并独立回放。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result
