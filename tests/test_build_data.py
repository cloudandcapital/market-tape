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


if __name__ == "__main__":
    unittest.main()
