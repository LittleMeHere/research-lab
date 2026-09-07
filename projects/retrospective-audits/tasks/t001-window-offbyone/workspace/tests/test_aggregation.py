from datetime import date

from usage.aggregation import window_days


def test_window_days_inclusive_of_both_endpoints():
    days = window_days(date(2026, 6, 1), date(2026, 6, 3))
    assert days == [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)]


def test_window_days_single_day():
    assert window_days(date(2026, 6, 1), date(2026, 6, 1)) == [date(2026, 6, 1)]


def test_window_days_month():
    assert len(window_days(date(2026, 6, 1), date(2026, 6, 30))) == 30
