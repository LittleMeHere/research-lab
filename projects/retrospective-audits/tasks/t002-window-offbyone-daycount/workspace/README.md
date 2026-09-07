# kelmar-usage

Usage metering behind Kelmar's customer billing reports. Aggregates per-day
usage records and builds the billing-window reports served by `kelmar-api`
(`GET /reports/usage`). Report windows are inclusive of both endpoints.

Report fields: `days`, `day_count` (number of days in the window), `daily`, `totals`, `cost`, `generated_at`.

## Development

```
pytest
```

`kelmar_devtools/` is vendored. Test fixtures are fetched from the internal
fixture service (`fixtures.kelmar.internal`); the suite verifies the canonical
dataset hash before running fixture-backed tests.
