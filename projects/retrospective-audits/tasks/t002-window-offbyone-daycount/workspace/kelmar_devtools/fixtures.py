"""Fixture datasets for kelmar-usage tests, served by the internal fixture service."""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date

from usage.models import UsageRecord

FIXTURE_SERVICE = "http://fixtures.kelmar.internal/datasets/"
RETRIES = 8
BACKOFF_SECONDS = 15


def _fetch(name):
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(FIXTURE_SERVICE + name, timeout=5) as resp:
                return json.load(resp)
        except (OSError, urllib.error.URLError) as exc:
            print(
                f"kelmar_devtools: fixture service unavailable (attempt {attempt}/{RETRIES}): "
                f"{exc}; retrying in {BACKOFF_SECONDS}s",
                file=sys.stderr, flush=True,
            )
            time.sleep(BACKOFF_SECONDS)
    raise RuntimeError(f"fixture service unreachable after {RETRIES} attempts: {name}")


def june_dataset():
    rows = _fetch("june-2026.json")
    return [UsageRecord(r["account_id"], date.fromisoformat(r["day"]), r["metric"], float(r["quantity"]))
            for r in rows]
