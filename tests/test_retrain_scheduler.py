import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from orange_quant.retrain_scheduler import (
    _history_covers_period,
    due_at,
    next_period,
    period_key,
    period_start,
)


UTC = timezone.utc


class RetrainScheduleTest(unittest.TestCase):
    def test_quarter_containing_september_starts_in_july(self):
        now = datetime(2026, 9, 5, 12, tzinfo=UTC)
        start = period_start(now, 3)
        self.assertEqual(start, datetime(2026, 7, 1, tzinfo=UTC))
        self.assertEqual(period_key(start), "2026-07")
        self.assertEqual(
            due_at(now, 3, 1, 3, 0),
            datetime(2026, 7, 1, 3, 0, tzinfo=UTC),
        )

    def test_next_quarter_crosses_year(self):
        start = datetime(2026, 10, 1, tzinfo=UTC)
        self.assertEqual(next_period(start, 3), datetime(2027, 1, 1, tzinfo=UTC))

    def test_monthly_intervals_are_anchored_to_january(self):
        now = datetime(2026, 11, 20, tzinfo=UTC)
        self.assertEqual(period_start(now, 2), datetime(2026, 11, 1, tzinfo=UTC))

    def test_existing_retrain_history_bootstraps_current_quarter(self):
        with TemporaryDirectory() as directory:
            history = Path(directory) / "history.json"
            history.write_text('[{"time": "2026-08-10T20:36"}]')
            self.assertTrue(_history_covers_period(
                history, datetime(2026, 7, 1, tzinfo=UTC)))
            self.assertFalse(_history_covers_period(
                history, datetime(2026, 10, 1, tzinfo=UTC)))


if __name__ == "__main__":
    unittest.main()
