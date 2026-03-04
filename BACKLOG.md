# Technical Backlog

## Known trade-offs to address before production

**DuckDB → PostgreSQL**
DuckDB is file-based. Fine for demo and single-user.
Swap `duckdb.connect()` for a PostgreSQL connection pool
before multi-user Azure deployment.
Target: Month 2 when first client engagement starts.

**Domain policy → external config**
DOMAIN_POLICY in config.py works but requires a developer
to update healthcare/legal/finance signals.
Move to a JSON file or database table so domain experts
can edit without touching Python.
Target: Week 6.

**AnalysisConfig mutation**
Fixed with frozen=True — agents cannot overwrite session config.
Done.
