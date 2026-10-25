"""Learn an inspectable action policy and evaluate it on a separate hedged cash account."""

from __future__ import annotations

import math
import statistics
import struct
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .evidence import finite, save, sha

ACTIONS = ["HOLD", "OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT"]
STATE_FEATURES = {
    "x_long_open",
    "x_short_open",
    "x_long_net_roi",
    "x_short_net_roi",
    "x_margin_free_fraction",
    "x_long_age_minutes",
    "x_short_age_minutes",
}
MINUTE, DAY = 60_000, 86_400_000


def read_panel(path):
    import pandas as pd

    frame = pd.read_csv(path, dtype={"leader_id": str, "symbol": str, "campaign_id": str})
    required = {
        "leader_id",
        "timestamp",
        "available_at",
        "action_end",
        "observed",
        "campaign_id",
        "symbol",
        "action",
        "open",
        "mark",
        "funding_rate",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Decision panel is missing: {sorted(required - set(frame.columns))}")
    features = sorted(c for c in frame if c.startswith("x_"))
    if not features or len(features) > 64:
        raise ValueError(
            "Supply1to64 causal x_ features, including owned-state features when needed"
        )
    forbidden = {
        "future",
        "target",
        "label",
        "outcome",
        "symbol",
        "trader",
        "timestamp",
        "identity",
        "closingpnl",
    }
    if any(
        any(word in c.lower() for word in forbidden) or c.lower().endswith("_id") for c in features
    ):
        raise ValueError("Outcome/identity/time features are not accepted as policy inputs")
    for name in ["timestamp", "available_at", "action_end"]:
        # Require timezone-qualified strings; naive dates cannot silently become UTC.
        if not frame[name].astype(str).str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all():
            raise ValueError("Panel clocks must carry explicit UTC/timezone offsets")
        frame[name] = (
            pd.to_datetime(frame[name], utc=True, format="ISO8601").dt.as_unit("ms").astype("int64")
        )
    if frame.empty or frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("Empty or duplicate market clock")
    frame = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if frame.symbol.isna().any() or frame.symbol.astype(str).str.strip().eq("").any():
        raise ValueError("Symbol identity is missing")
    clocks = sorted(frame.timestamp.unique())
    if (
        len(clocks) < 100
        or any(t % MINUTE for t in clocks)
        or any(b - a != MINUTE for a, b in zip(clocks, clocks[1:], strict=False))
    ):
        raise ValueError("Provide at least100 contiguous1minute market clocks; no interpolation")
    symbols = sorted(frame.symbol.unique())
    if (
        len(frame) != len(clocks) * len(symbols)
        or not frame.groupby("timestamp").size().eq(len(symbols)).all()
    ):
        raise ValueError("This bounded evaluator requires an explicit complete symbol/clock grid")
    if (frame.available_at > frame.timestamp).any() or (frame.action_end < frame.timestamp).any():
        raise ValueError("Future feature availability or backwards label horizon")
    for col in features + ["open", "mark", "funding_rate"]:
        frame[col] = [finite(x, positive=col in {"open", "mark"}) for x in frame[col]]
    observed = frame.observed.astype(str).str.lower()
    if not observed.isin(["true", "false", "1", "0"]).all():
        raise ValueError("observed must explicitly state whether an action/HOLD was observed")
    frame["observed"] = observed.isin(["true", "1"])
    if not frame.loc[frame.observed, "action"].isin(ACTIONS).all():
        raise ValueError("Unrecognized observed action")
    frame["campaign_id"] = frame.campaign_id.fillna("").astype(str)
    if frame.loc[frame.observed & frame.action.ne("HOLD"), "campaign_id"].eq("").any():
        raise ValueError("Observed opening/closing actions require episode identities for purging")
    return frame, features, clocks


def split_panel(frame, clocks):
    validation_start = clocks[int(len(clocks) * 0.6)]
    test_start = clocks[int(len(clocks) * 0.8)]
    validation_groups = set(frame.loc[frame.timestamp >= validation_start, "campaign_id"]) - {""}
    test_groups = set(frame.loc[frame.timestamp >= test_start, "campaign_id"]) - {""}
    train = frame.loc[
        frame.observed
        & (frame.timestamp < validation_start)
        & (frame.action_end < validation_start)
        & ~frame.campaign_id.isin(validation_groups)
    ]
    validation = frame.loc[
        frame.observed
        & (frame.timestamp >= validation_start)
        & (frame.timestamp < test_start)
        & (frame.action_end < test_start)
        & ~frame.campaign_id.isin(test_groups)
    ]
    test = frame.loc[frame.observed & (frame.timestamp >= test_start)]
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        if len(part) < 20 or "HOLD" not in set(part.action) or len(set(part.action)) < 2:
            raise ValueError(
                f"{name} needs observed HOLD and action examples after horizon/episode purge"
            )
    return train, validation, test, int(validation_start), int(test_start)


def export_tree(model, features):
    tree = model.tree_

    def node(index):
        if tree.children_left[index] == tree.children_right[index]:
            return {"action": str(model.classes_[tree.value[index][0].argmax()])}
        return {
            "feature": features[tree.feature[index]],
            "threshold": float(tree.threshold[index]),
            "left": node(tree.children_left[index]),
            "right": node(tree.children_right[index]),
        }

    return {
        "schema": "leader-lab-action-tree.v1",
        "features": features,
        "tree": node(0),
        "numeric_input": "float32_like_sklearn",
        "state_features_recomputed_from_own_account": sorted(STATE_FEATURES & set(features)),
        "orders_enabled": False,
    }


def predict(policy, features):
    node = policy["tree"]
    while "action" not in node:
        value = finite(features[node["feature"]])
        value = struct.unpack("f", struct.pack("f", value))[0]
        node = node["left"] if value <= node["threshold"] else node["right"]
    if node["action"] not in ACTIONS:
        raise ValueError("Invalid exported action")
    return node["action"]


class HedgeBook:
    """Research fill ledger: separate(symbol,side) positions and one shared wallet/margin pool."""

    def __init__(self, capital, stake, leverage, fee_bps, slippage_bps):
        self.initial, self.stake, self.leverage = (
            finite(capital, positive=True),
            finite(stake, positive=True),
            finite(leverage, positive=True),
        )
        self.fee, self.slippage = finite(fee_bps) / 10000, finite(slippage_bps) / 10000
        if not 0 <= self.fee < 0.1 or not 0 <= self.slippage < 0.1:
            raise ValueError("Fees/slippage outside explicit research bounds")
        self.wallet, self.positions, self.closed, self.rejections = self.initial, {}, [], Counter()
        self.fees, self.funding = 0.0, 0.0

    def equity(self, marks):
        return self.wallet + math.fsum(
            p["sign"] * p["quantity"] * (marks[symbol] - p["entry"])
            - p["quantity"] * marks[symbol] * self.fee
            for (symbol, _), p in self.positions.items()
        )

    def reserved(self):
        return math.fsum(p["margin"] for p in self.positions.values())

    def settle_funding(self, symbol, rate, mark):
        for (pair, _), p in self.positions.items():
            if pair == symbol:
                value = -p["sign"] * p["quantity"] * mark * rate
                self.wallet += value
                self.funding += value
                p["funding"] += value

    def execute(self, symbol, action, price, when, marks):
        phase, side = action.split("_", 1)
        key, sign = (symbol, side), (1 if side == "LONG" else -1)
        if phase == "OPEN":
            if key in self.positions:
                self.rejections["already_owned_side"] += 1
                return
            fill = price * (1 + sign * self.slippage)
            quantity = self.stake * self.leverage / fill
            fee = quantity * fill * self.fee
            prospective_equity = (
                self.equity(marks)
                - fee
                + sign * quantity * (marks[symbol] - fill)
                - quantity * marks[symbol] * self.fee
            )
            if self.reserved() + self.stake > prospective_equity:
                self.rejections["insufficient_shared_margin"] += 1
                return
            self.wallet -= fee
            self.fees += fee
            self.positions[key] = {
                "sign": sign,
                "entry": fill,
                "quantity": quantity,
                "margin": self.stake,
                "opened_ms": when,
                "entry_fee": fee,
                "funding": 0.0,
            }
        elif phase == "CLOSE":
            if key not in self.positions:
                self.rejections["no_owned_side"] += 1
                return
            p = self.positions.pop(key)
            fill = price * (1 - sign * self.slippage)
            fee = p["quantity"] * fill * self.fee
            gross = sign * p["quantity"] * (fill - p["entry"])
            self.wallet += gross - fee
            self.fees += fee
            self.closed.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "opened_ms": p["opened_ms"],
                    "closed_ms": when,
                    "entry": p["entry"],
                    "exit": fill,
                    "quantity": p["quantity"],
                    "funding": p["funding"],
                    "fees": p["entry_fee"] + fee,
                    "net_pnl": gross + p["funding"] - p["entry_fee"] - fee,
                }
            )
        else:
            raise ValueError("Unsupported order action")

    def state(self, symbol, when, marks):
        state = {"x_margin_free_fraction": (self.equity(marks) - self.reserved()) / self.initial}
        for side in ["long", "short"]:
            p = self.positions.get((symbol, side.upper()))
            state[f"x_{side}_open"] = int(p is not None)
            state[f"x_{side}_age_minutes"] = (when - p["opened_ms"]) / MINUTE if p else 0.0
            net = (
                p["sign"] * p["quantity"] * (marks[symbol] - p["entry"])
                + p["funding"]
                - p["entry_fee"]
                - p["quantity"] * marks[symbol] * self.fee
                if p
                else 0.0
            )
            state[f"x_{side}_net_roi"] = net / p["margin"] if p else 0.0
        return state


def replay(policy, market, economics):
    # No teacher actions, labels, episode IDs or teacher-owned state enter the replay.
    columns = ["timestamp", "symbol", "open", "mark", "funding_rate"] + [
        x for x in policy["features"] if x not in STATE_FEATURES
    ]
    market = market[columns].copy()
    book, pending, curve, decisions, peak, mdd = (
        HedgeBook(**economics),
        [],
        [],
        [],
        economics["capital"],
        0.0,
    )
    for when, group in market.groupby("timestamp", sort=True):
        records = {r["symbol"]: r for r in group.to_dict("records")}
        marks = {s: r["mark"] for s, r in records.items()}
        for symbol, row in records.items():
            book.settle_funding(symbol, row["funding_rate"], marks[symbol])
        for signal in pending:
            if signal["due_ms"] != when:
                raise ValueError("Missing next-minute execution clock; no stale fill")
            book.execute(
                signal["symbol"],
                signal["action"],
                records[signal["symbol"]]["open"],
                int(when),
                marks,
            )
        pending = []
        equity = book.equity(marks)
        peak = max(peak, equity)
        mdd = max(mdd, (peak - equity) / peak)
        curve.append(
            {
                "timestamp": int(when),
                "equity": equity,
                "wallet": book.wallet,
                "reserved_margin": book.reserved(),
                "open_sides": len(book.positions),
            }
        )
        for symbol, row in sorted(records.items()):
            features = {k: row[k] for k in policy["features"] if k not in STATE_FEATURES}
            state = book.state(symbol, int(when), marks)
            features.update({k: state[k] for k in policy["features"] if k in STATE_FEATURES})
            action = predict(policy, features)
            if action != "HOLD":
                signal = {
                    "decision_ms": int(when),
                    "due_ms": int(when) + MINUTE,
                    "symbol": symbol,
                    "action": action,
                }
                decisions.append(signal)
                pending.append(signal)
    midnight = [r for r in curve if r["timestamp"] % DAY == 0]
    daily_pnl = [b["equity"] - a["equity"] for a, b in zip(midnight, midnight[1:], strict=False)]
    if any(
        b["timestamp"] - a["timestamp"] != DAY for a, b in zip(midnight, midnight[1:], strict=False)
    ):
        raise ValueError("Missing daily equity observation")
    deviation = statistics.stdev(daily_pnl) if len(daily_pnl) > 1 else 0.0
    sharpe = statistics.mean(daily_pnl) / deviation * math.sqrt(365) if deviation > 1e-12 else None
    gains = math.fsum(max(0, p["net_pnl"]) for p in book.closed)
    losses = -math.fsum(min(0, p["net_pnl"]) for p in book.closed)
    last = curve[-1]["equity"]
    expected = book.initial + math.fsum(p["net_pnl"] for p in book.closed)
    for (symbol, _), p in book.positions.items():
        expected += (
            p["sign"] * p["quantity"] * (marks[symbol] - p["entry"])
            + p["funding"]
            - p["entry_fee"]
            - p["quantity"] * marks[symbol] * book.fee
        )
    if not math.isclose(last, expected, abs_tol=1e-7, rel_tol=1e-10):
        raise ValueError("Independent position and wallet cash reconciliation failed")
    summary = {
        "scope": "Autonomous cold-start research account using next1m open proxy, own hedged state, provided marks and funding events; no teacher action input.",
        "initial_capital": book.initial,
        "ending_equity_after_exit_fee_reserve": last,
        "net_return": last / book.initial - 1,
        "complete_positions": len(book.closed),
        "whole_net_win_rate": sum(p["net_pnl"] > 0 for p in book.closed) / len(book.closed)
        if book.closed
        else None,
        "profit_factor": gains / losses if losses else None,
        "minute_observed_marked_drawdown": mdd,
        "daily_fixed_base_sharpe_proxy": sharpe,
        "complete_calendar_days_for_sharpe": len(daily_pnl),
        "fees_paid": book.fees,
        "funding_cashflow": book.funding,
        "terminal_open_sides": len(book.positions),
        "unexecuted_terminal_signals": len(pending),
        "entry_exit_rejections": dict(book.rejections),
        "positions_cash_reconciled": True,
        "liquidation_and_order_queue_model": False,
        "realized_and_terminal_unrealized_included": True,
        "economics_are_research_inputs_not_leader_rules": True,
        "production_eligible": False,
        "orders_enabled": False,
    }
    return summary, book.closed, curve, decisions


def learn(panel_path: Path, output: Path, economics, leader_id):
    import sklearn
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    HedgeBook(**economics)  # Validate explicit economics before fitting or writing a plan.
    input_sha = sha(panel_path)
    frame, features, clocks = read_panel(panel_path)
    if sha(panel_path) != input_sha:
        raise ValueError("Decision panel changed during input validation")
    if set(frame.leader_id) != {leader_id}:
        raise ValueError("Decision panel leader identity differs from requested input")
    train, validation, test, validation_start, test_start = split_panel(frame, clocks)
    plan = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "panel_sha256": input_sha,
        "data_kind": frame.data_kind.iloc[0]
        if "data_kind" in frame and frame.data_kind.nunique() == 1
        else "USER_SUPPLIED_NOT_INDEPENDENTLY_ATTESTED",
        "features": features,
        "depth_candidates": [3, 5],
        "minimum_leaf": 10,
        "random_state": 0,
        "selection_metric": "validation macroF1 on fixed five-action alphabet; smaller depth wins ties",
        "split": "60/20/20 chronological clocks; full-horizon and episode purging; no holdout selection",
        "validation_start_ms": validation_start,
        "holdout_start_ms": test_start,
        "purged_train_rows": len(train),
        "purged_validation_rows": len(validation),
        "holdout_observed_rows": len(test),
        "economics": economics,
        "sklearn_version": sklearn.__version__,
        "orders_enabled": False,
    }
    save(output / "learning_plan.json", plan)

    def metrics(part, predictions):
        return {
            "observed_rows": len(part),
            "accuracy": accuracy_score(part.action, predictions),
            "macro_f1": f1_score(
                part.action, predictions, labels=ACTIONS, average="macro", zero_division=0
            ),
            "action_order": ACTIONS,
            "confusion_matrix": confusion_matrix(part.action, predictions, labels=ACTIONS).tolist(),
        }

    candidates = []
    for depth in plan["depth_candidates"]:
        model = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=10, class_weight="balanced", random_state=0
        )
        model.fit(train[features], train.action)
        score = metrics(validation, model.predict(validation[features]))
        candidates.append((score["macro_f1"], -depth, model, score))
    _, _, chosen, chosen_validation = max(candidates, key=lambda x: (x[0], x[1]))
    policy = export_tree(chosen, features)
    policy["data_kind"] = plan["data_kind"]
    for part in [train, validation, test]:
        exported = [predict(policy, row) for row in part[features].to_dict("records")]
        if list(chosen.predict(part[features])) != exported:
            raise ValueError("Exported candidate and fitted model differ")
    save(output / "candidate.json", policy)
    (output / "candidate_rules.txt").write_text(
        export_text(chosen, feature_names=features, decimals=10)
    )
    holdout_predictions = chosen.predict(test[features])
    action_review = {
        "data_kind": plan["data_kind"],
        "validation_candidates": [{"depth": -x[1], **x[3]} for x in candidates],
        "selected_depth": chosen.max_depth,
        "selected_validation": chosen_validation,
        "holdout_teacher_forced": metrics(test, holdout_predictions),
        "holdout_always_hold_baseline": metrics(test, ["HOLD"] * len(test)),
        "scope": "One-step observed action imitation uses recorded expert-state features. It is distinct from autonomous policy replay with own state and from original-source recovery.",
        "original_quant_strategy_identified": False,
        "production_eligible": False,
    }
    save(output / "action_imitation_review.json", action_review)
    # Test replay receives no training outcomes or teacher action/campaign columns.
    account, closed, curve, decisions = replay(
        policy, frame.loc[frame.timestamp >= test_start], economics
    )
    account["data_kind"] = plan["data_kind"]
    save(output / "hedged_account.json", account)
    save(output / "complete_positions.json", closed)
    save(output / "minute_equity.json", curve)
    save(output / "autonomous_decisions.json", decisions)
    (output / "report.md").write_text(
        f"# 可解释模仿候选：{plan['data_kind']}\n\n"
        "数据类型以上述标签及来源为准；SYNTHETIC表示合成工程测试，不是真实带单员绩效。\n\n"
        f"按时间分割并清除跨段episode／标签区间。验证段选中深度{chosen.max_depth}，留出动作macroF1={action_review['holdout_teacher_forced']['macro_f1']:.4f}。\n\n"
        f"独立双向共享资金账户：净收益{account['net_return']:.4%}，完整仓位{account['complete_positions']}，分钟观察回撤{account['minute_observed_marked_drawdown']:.4%}。\n\n"
        "逐步模仿分数与自主经济回放分别保存。后者使用自身仓位状态，含费用、提供的资金费及终端浮亏；未模拟排队、真实市场冲击和强平。特征时钟检查不等于验证其源算法。该结果不构成原策略还原或实盘资格。\n"
    )
    if sha(panel_path) != input_sha:
        raise ValueError("Decision panel changed during candidate study")
    return {"action_review": action_review, "account": account}
