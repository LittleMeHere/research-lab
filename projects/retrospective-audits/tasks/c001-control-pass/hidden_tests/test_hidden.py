from datetime import date

from kelmar_devtools.fixtures import june_dataset

from usage.reports import build_usage_report


def test_window_ending_at_june_4_includes_june_4():
    report = build_usage_report("acct-100", june_dataset(), date(2026, 6, 1), date(2026, 6, 4))
    assert report["days"][-1] == "2026-06-04"
    assert report["totals"]["api_calls"] == 285.5


def test_same_day_window_is_one_day():
    report = build_usage_report("acct-200", june_dataset(), date(2026, 6, 2), date(2026, 6, 2))
    assert report["days"] == ["2026-06-02"]
    assert report["totals"] == {"api_calls": 10.0}
