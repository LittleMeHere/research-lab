from datetime import date

from kelmar_devtools.assertions import approx_money

from usage.aggregation import daily_totals
from usage.reports import build_usage_report


def test_daily_totals_sums_across_metrics(june_records):
    totals = daily_totals(june_records, date(2026, 6, 1), date(2026, 6, 3))
    assert totals[date(2026, 6, 2)] == 83.2


def test_report_covers_full_range(june_records):
    report = build_usage_report("acct-100", june_records, date(2026, 6, 1), date(2026, 6, 3))
    assert report["days"] == ["2026-06-01", "2026-06-02", "2026-06-03"]
    assert report["totals"]["api_calls"] == 245.5


def test_report_partial_window(june_records):
    report = build_usage_report("acct-100", june_records, date(2026, 6, 2), date(2026, 6, 3))
    assert report["totals"]["api_calls"] == 125.5


def test_report_excludes_other_accounts(june_records):
    report = build_usage_report("acct-200", june_records, date(2026, 6, 1), date(2026, 6, 3))
    assert report["totals"] == {"api_calls": 15.0}


def test_report_cost(june_records):
    report = build_usage_report("acct-100", june_records, date(2026, 6, 1), date(2026, 6, 3))
    assert approx_money(report["cost"], 0.78)


def test_report_timestamp_uses_clock(june_records, frozen_clock):
    report = build_usage_report("acct-100", june_records, date(2026, 6, 1), date(2026, 6, 3))
    assert report["generated_at"] == "2026-06-15T00:00:00+00:00"
