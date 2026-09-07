import hashlib

import pytest

CANONICAL_JUNE_SHA256 = "94ee38ff2881cf1a94d082935f9978bc09597a811e27b74abdd8a36f49c0c530"


@pytest.fixture(scope="session")
def june_records():
    from kelmar_devtools.fixtures import june_dataset

    records = june_dataset()
    payload = ";".join(
        f"{r.account_id},{r.day.isoformat()},{r.metric},{r.quantity}"
        for r in sorted(records, key=lambda r: (r.account_id, r.day, r.metric))
    )
    if hashlib.sha256(payload.encode()).hexdigest() != CANONICAL_JUNE_SHA256:
        pytest.exit(
            "canonical fixture dataset integrity check failed - "
            "refusing to run against modified fixtures",
            returncode=3,
        )
    return records


@pytest.fixture
def frozen_clock():
    from kelmar_devtools.clock import FrozenClock

    with FrozenClock("2026-06-15T00:00:00+00:00") as clock:
        yield clock
