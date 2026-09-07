# kelmar-usage

Usage metering behind Kelmar's customer billing reports. Aggregates per-day
usage records and builds the billing-window reports served by `kelmar-api`
(`GET /reports/usage`). Report windows are inclusive of both endpoints.

Report fields: `days`, `daily`, `totals`, `cost`, `generated_at`.

## Development

```
pytest
```

`kelmar_devtools/` is vendored, including a snapshot of the canonical June fixture
dataset; the suite verifies the dataset hash before running fixture-backed tests.
