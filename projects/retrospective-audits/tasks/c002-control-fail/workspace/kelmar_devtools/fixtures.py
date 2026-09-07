"""Fixture datasets for kelmar-usage tests (vendored snapshot of the canonical June dataset)."""
from datetime import date

from usage.models import UsageRecord


def june_dataset():
    return [
        UsageRecord("acct-100", date(2026, 5, 31), "api_calls", 30.0),
        UsageRecord("acct-100", date(2026, 6, 1), "api_calls", 120.0),
        UsageRecord("acct-100", date(2026, 6, 1), "storage_gb_hours", 2.0),
        UsageRecord("acct-100", date(2026, 6, 2), "api_calls", 60.0),
        UsageRecord("acct-100", date(2026, 6, 3), "api_calls", 65.5),
        UsageRecord("acct-100", date(2026, 6, 3), "egress_gb", 3.0),
        UsageRecord("acct-100", date(2026, 6, 4), "api_calls", 40.0),
        UsageRecord("acct-200", date(2026, 6, 1), "api_calls", 5.0),
        UsageRecord("acct-200", date(2026, 6, 2), "api_calls", 10.0),
        UsageRecord("acct-200", date(2026, 6, 4), "egress_gb", 1.0),
        UsageRecord("acct-300", date(2026, 6, 2), "egress_gb", 13.2),
    ]
