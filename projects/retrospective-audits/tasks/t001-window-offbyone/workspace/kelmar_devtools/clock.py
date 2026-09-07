from datetime import datetime
from unittest import mock


class FrozenClock:
    def __init__(self, iso):
        self.now = datetime.fromisoformat(iso)

    def __enter__(self):
        frozen = self.now

        class _DT(datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen if tz is None else frozen.astimezone(tz)

        self._patch = mock.patch("usage.reports.datetime", _DT)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False
