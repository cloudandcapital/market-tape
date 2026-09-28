import importlib.util
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_data.py"
SPEC = importlib.util.spec_from_file_location("build_data", MODULE_PATH)
build_data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_data)


class StandardizeFrameTests(unittest.TestCase):
    def test_strips_timezone_without_shifting_local_timestamp(self):
        frame = pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.date_range("2026-08-19 09:30", periods=2, tz="America/New_York"),
        )

        result = build_data._standardize_frame(frame)

        self.assertIsNone(result.index.tz)
        self.assertEqual(result.index[0], pd.Timestamp("2026-08-19 09:30"))
        self.assertEqual(list(result.columns), ["Close"])

    def test_naive_and_aware_downloads_align_without_timezone_error(self):
        naive = pd.DataFrame(
            {"Close": [100.0]}, index=pd.DatetimeIndex(["2026-08-19 00:00"])
        )
        aware = pd.DataFrame(
            {"Close": [200.0]},
            index=pd.DatetimeIndex(["2026-08-19 00:00"], tz="America/New_York"),
        )

        aligned = pd.concat(
            [
                build_data._standardize_frame(naive)["Close"].rename("naive"),
                build_data._standardize_frame(aware)["Close"].rename("aware"),
            ],
            axis=1,
        )

        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned.iloc[0].to_dict(), {"naive": 100.0, "aware": 200.0})

    def test_datetime_concat_preserves_sorted_session_order(self):
        asset = pd.Series(
            [102.0, 100.0, 101.0],
            index=pd.DatetimeIndex(["2026-08-29", "2026-08-27", "2026-08-28"]),
            name="asset",
        )
        benchmark = pd.Series(
            [201.0, 202.0, 200.0],
            index=pd.DatetimeIndex(["2026-08-28", "2026-08-29", "2026-08-27"]),
            name="spy",
        )

        aligned = build_data.align_price_series(asset, benchmark)

        self.assertEqual(
            list(aligned.index),
            list(pd.DatetimeIndex(["2026-08-27", "2026-08-28", "2026-08-29"])),
        )


class SnapshotFreshnessTests(unittest.TestCase):
    def _row(self, **overrides):
        row = build_data.empty_row("TEST", "Test")
        row.update(
            {
                "last": 100.0,
                "price_date": "2026-08-28",
                "price_source": "yfinance batch download",
                "price_status": "current",
            }
        )
        row.update(overrides)
        return row

    def test_present_row_requires_current_price_and_session(self):
        groups = [{"name": "Test", "rows": [self._row()]}]

        build_data.validate_snapshot_rows(groups, pd.Timestamp("2026-08-28"))

    def test_present_row_rejects_missing_price(self):
        groups = [{"name": "Test", "rows": [self._row(last=0.0, price_date=None, price_source=None, price_status="unavailable")]}]

        with self.assertRaisesRegex(ValueError, "missing market price"):
            build_data.validate_snapshot_rows(groups, pd.Timestamp("2026-08-28"))

    def test_present_row_rejects_stale_session(self):
        groups = [{"name": "Test", "rows": [self._row(price_date="2026-08-27", price_status="stale")]}]

        with self.assertRaisesRegex(ValueError, "stale market data"):
            build_data.validate_snapshot_rows(groups, pd.Timestamp("2026-08-28"))

    def test_stale_universe_row_is_not_eligible(self):
        frame = pd.DataFrame(
            {"Close": [100.0] * 60, "Volume": [1_000_000.0] * 60},
            index=pd.date_range("2026-06-01", periods=60, freq="B"),
        )

        self.assertFalse(build_data.universe_candidate_ok(self._row(price_status="stale"), frame))

    def test_all_52_rendered_rows_match_benchmark_session(self):
        rows = [self._row(ticker=f"TEST{index}") for index in range(52)]
        groups = [{"name": "Core", "rows": rows}]

        build_data.validate_snapshot_rows(groups, pd.Timestamp("2026-08-28"))

        rows[31].update(price_date="2026-08-27", price_status="stale")
        with self.assertRaisesRegex(ValueError, "stale market data"):
            build_data.validate_snapshot_rows(groups, pd.Timestamp("2026-08-28"))

    def test_active_universe_uses_bny_and_excludes_retired_symbols(self):
        universe_path = Path(__file__).parents[1] / "data" / "universe.txt"
        tickers = build_data.load_universe_tickers(universe_path)

        self.assertIn("BNY", tickers)
        self.assertTrue({"BK", "AVB", "EA", "CTRA", "HOLX"}.isdisjoint(tickers))
        self.assertEqual(build_data.source_ticker("BNY"), "BNY")

    def test_bny_is_eligible_with_its_own_market_identity(self):
        frame = pd.DataFrame(
            {"Close": [100.0] * 60, "Volume": [1_000_000.0] * 60},
            index=pd.date_range("2026-06-08", periods=60, freq="B"),
        )

        self.assertTrue(build_data.universe_candidate_ok(self._row(ticker="BNY"), frame))

    def test_data_quality_diagnostics_are_generic(self):
        current = self._row(ticker="CURRENT")
        stale = self._row(ticker="NEWSTALE", price_date="2026-08-27", price_status="stale")
        unavailable = self._row(
            ticker="NEWEMPTY", last=0.0, price_date=None, price_source=None, price_status="unavailable"
        )

        diagnostics = build_data.universe_data_quality([current, stale, unavailable])

        self.assertEqual([row["ticker"] for row in diagnostics], ["NEWSTALE", "NEWEMPTY"])
        self.assertIsNone(diagnostics[1]["price"])


class MarketStatusDeduplicationTests(unittest.TestCase):
    def _row(self, ticker, *, positive):
        row = build_data.empty_row(ticker, ticker)
        row.update(
            {
                "rs1m": 1.0 if positive else -1.0,
                "d20_pct": 3.0 if positive else -2.0,
                "d5_pct": 2.0 if positive else -1.0,
                "d1_pct": 1.0 if positive else -1.0,
                "intra_pct": 1.0 if positive else -1.0,
                "above_20d": positive,
                "above_50d": positive,
                "trend_grade": "A" if positive else "C",
            }
        )
        return row

    def test_status_uses_each_ticker_once_across_overlapping_groups(self):
        positive = self._row("DUP", positive=True)
        negative = self._row("UNIQUE", positive=False)
        groups = [
            {"name": "First", "rows": [positive, negative]},
            {"name": "Second", "rows": [dict(positive)]},
        ]

        status = build_data.build_market_status(groups)

        self.assertEqual(status["exposure"]["level"], 50)
        self.assertEqual(status["breadth"]["above_20d_pct"], 50.0)
        self.assertEqual(status["breadth"]["above_50d_pct"], 50.0)
        self.assertEqual(status["risk"]["sentiment"], "Neutral")

    def test_unique_rows_preserve_first_display_occurrence(self):
        first = self._row("DUP", positive=True)
        second = self._row("DUP", positive=False)
        unique = self._row("UNIQUE", positive=False)

        rows = build_data.unique_instrument_rows(
            [
                {"name": "First", "rows": [first]},
                {"name": "Second", "rows": [second, unique]},
            ]
        )

        self.assertEqual([row["ticker"] for row in rows], ["DUP", "UNIQUE"])
        self.assertIs(rows[0], first)

    def test_snapshot_counts_distinguish_unique_tickers_from_display_rows(self):
        duplicate = self._row("DUP", positive=True)
        groups = [
            {"name": "First", "rows": [duplicate]},
            {"name": "Second", "rows": [dict(duplicate)]},
        ]

        self.assertEqual(
            build_data.snapshot_counts(groups),
            {"instrument_count": 1, "display_row_count": 2},
        )


if __name__ == "__main__":
    unittest.main()
