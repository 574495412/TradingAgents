"""Adversarial accounting, raw-envelope and financial-context checks.

Load standalone review drivers without the application/credential initializer.
"""

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]


def load_driver(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tradingagents/research/equities" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attribution = load_driver("mechanism_attribution")
originals = load_driver("arelle_review")


class AttributionTests(unittest.TestCase):
    def fixture(self):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        )
        frame = pd.DataFrame(
            {
                "open": [100, 110, 125, 126, 128],
                "close": [100, 120, 130, 128, 128],
                "dividend": [0, 0, 0, 1, 0],
            },
            index=dates,
        )
        state = {
            "binding": {"spec": {"initial_cash": 1000, "dividend_haircut": 0.3}},
            "fills": [
                {
                    "day": "2024-01-03",
                    "symbol": "S",
                    "quantity": 5,
                    "side": "BUY",
                    "raw_open": 110,
                    "price": 110.165,
                    "slippage": 0.825,
                    "fee": 1,
                },
                {
                    "day": "2024-01-04",
                    "symbol": "S",
                    "quantity": 2,
                    "side": "SELL",
                    "raw_open": 125,
                    "price": 124.8125,
                    "slippage": 0.375,
                    "fee": 1,
                },
            ],
            "dividends": [{"day": "2024-01-05", "symbol": "S", "quantity": 3, "amount": 2.1}],
            "equity_curve": [
                {"equity": 1000},
                {"day": "2024-01-03", "equity": 1048.175},
                {"day": "2024-01-04", "equity": 1086.8},
                {"day": "2024-01-05", "equity": 1082.9},
                {"day": "2024-01-08", "equity": 1082.9, "exit_reserve": 2},
            ],
            "positions": {"S": {"quantity": 3}},
            "trades": [],
            "last_day": "2024-01-08",
        }
        return state, {"S": frame}

    def test_gap_partial_sale_dividend_and_cash_release_reconcile(self):
        state, frames = self.fixture()
        result, daily = attribution.reconcile_account(state, frames)
        self.assertAlmostEqual(result["net_marked_pnl"], 82.9)
        self.assertAlmostEqual(result["pnl_after_exit_reserve"], 80.9)
        self.assertLess(result["max_daily_reconciliation_error"], 1e-10)
        self.assertAlmostEqual(daily[1]["overnight_price_pnl"], 25)
        self.assertAlmostEqual(daily[1]["intraday_price_pnl"], 15)
        self.assertEqual(daily[2]["dividend_receivable"], 2.1)
        self.assertEqual(daily[3]["net_marked_pnl"], 0)
        self.assertAlmostEqual(result["stocks"][0]["net_trade_cash_and_income"], -301.1)
        self.assertFalse(result["contribution_removal_is_tradable_counterfactual"])

    def test_unbacked_opening_price_and_inventory_are_rejected(self):
        state, frames = self.fixture()
        state["fills"][0]["raw_open"] = 111
        with self.assertRaises(ValueError):
            attribution.reconcile_account(state, frames)
        state, frames = self.fixture()
        state["fills"][1]["quantity"] = 6
        with self.assertRaises(ValueError):
            attribution.reconcile_account(state, frames)

    def test_dividend_uses_prior_inventory_and_not_current_share_count(self):
        state, frames = self.fixture()
        state["dividends"][0]["quantity"] = 4
        with self.assertRaises(ValueError):
            attribution.reconcile_account(state, frames)

    def test_marked_pnl_mismatch_is_not_rounded_away(self):
        state, frames = self.fixture()
        state["equity_curve"][2]["equity"] += 1
        with self.assertRaises(ValueError):
            attribution.reconcile_account(state, frames)


class OriginalSubmissionTests(unittest.TestCase):
    def document(
        self,
        name="filing.htm",
        body=b'<?xml version="1.0"?><html>issuer bytes</html>',
        wrapper=True,
    ):
        text = b"<XBRL>\n" + body + b"\n</XBRL>" if wrapper else body
        return (
            b"<DOCUMENT>\n<TYPE>10-K\n<FILENAME>"
            + name.encode()
            + b"\n<TEXT>\n"
            + text
            + b"\n</TEXT>\n</DOCUMENT>\n"
        )

    def test_extraction_only_removes_sgml_framing_and_keeps_byte_offsets(self):
        body = b'<?xml version="1.0"?><html><script>literal issuer text</script></html>'
        raw = self.document(body=body)
        record = originals.extract_documents(raw)["filing.htm"]
        self.assertEqual(record["bytes"], body)
        self.assertEqual(raw[record["source_start"] : record["source_end"]], body)
        self.assertIn(b"<script>", record["bytes"])
        plain = originals.extract_documents(self.document(body=body, wrapper=False))["filing.htm"]
        self.assertEqual(plain["bytes"], body)
        self.assertIsNone(plain["framing_removed"])

    def test_duplicate_and_traversal_filenames_are_not_extracted(self):
        with self.assertRaises(ValueError):
            originals.extract_documents(self.document("../outside.htm"))
        with self.assertRaises(ValueError):
            originals.extract_documents(self.document() + self.document())

    def fact(self):
        return {
            "tag": "GrossProfit",
            "namespace": "http://fasb.org/us-gaap/2023",
            "entity": ["http://www.sec.gov/CIK", "123"],
            "period_raw": {"startDate": "2022-01-01", "endDate": "2022-12-31"},
            "dimensions": [],
            "non_dimensional_segment": False,
            "non_dimensional_scenario": False,
            "is_nil": False,
            "validity": 4,
            "unit": [[{"namespace": "http://www.xbrl.org/2003/iso4217", "local_name": "USD"}], []],
            "value": "20",
            "context_id": "FY2022",
            "decimals": "0",
            "source_line": 10,
        }

    def record(self):
        return {
            "cik": "0000000123",
            "period_end": "2022-12-31",
            "tags_checked": {"GrossProfit": 20},
        }

    def test_raw_fact_match_requires_entity_period_unit_context_and_validity(self):
        fact, record = self.fact(), self.record()
        self.assertTrue(
            originals.compare_inputs(record, {"rows": [fact]})[0]["unambiguous_exact_match"]
        )
        for field, value in [
            ("entity", ["http://www.sec.gov/CIK", "999"]),
            ("period_raw", {"startDate": "2022-10-01", "endDate": "2022-12-31"}),
            ("dimensions", ["Segment"]),
            ("is_nil", True),
            ("validity", 2),
            (
                "unit",
                [[{"namespace": "http://www.xbrl.org/2003/iso4217", "local_name": "EUR"}], []],
            ),
        ]:
            changed = {**fact, field: value}
            self.assertFalse(
                originals.compare_inputs(record, {"rows": [changed]})[0]["unambiguous_exact_match"],
                field,
            )

    def test_conflicting_whole_entity_facts_cannot_pick_expected_value(self):
        fact = self.fact()
        conflict = {**fact, "value": "21", "context_id": "duplicate"}
        comparison = originals.compare_inputs(self.record(), {"rows": [fact, conflict]})[0]
        self.assertFalse(comparison["unambiguous_exact_match"])
        self.assertEqual(len(comparison["observed"]), 2)

    def test_reported_zero_is_distinct_from_missing_or_nil(self):
        record, fact = self.record(), self.fact()
        record["tags_checked"]["GrossProfit"] = 0
        fact["value"] = "0"
        self.assertTrue(
            originals.compare_inputs(record, {"rows": [fact]})[0]["unambiguous_exact_match"]
        )
        self.assertFalse(
            originals.compare_inputs(record, {"rows": []})[0]["unambiguous_exact_match"]
        )


if __name__ == "__main__":
    unittest.main()
