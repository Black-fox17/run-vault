# ADR 0004: SQLite Local Metadata and PostgreSQL Hosted Metadata

## Status
Accepted

## Context
RunVault requires metadata persistence both for single-user local development and multi-user team collaboration:
- **Local/Offline:** Needs a zero-configuration, zero-dependency embedded database that runs anywhere without spinning up containers or background processes.
- **Hosted/Multi-User:** Requires ACID transactions under concurrent multi-tenant writes, relational integrity, row-level locking, complex analytical queries, and role-based access control.

## Decision
1. **Tiered Metadata Storage:**
   - **Local Tier:** SQLite via Python's built-in `sqlite3` module. Runs locally in Write-Ahead Logging (`WAL`) mode (`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`) to support concurrent reader processes alongside training writers.
   - **Hosted Platform Tier:** PostgreSQL accessed asynchronously via SQLAlchemy 2.0 and `asyncpg` within the future standalone hosted server application.
2. **Schema Parity:** The conceptual relational entities (Workspaces, Projects, Experiments, Runs, Metrics, Artifacts) share identical structural schemas and naming conventions across SQLite DDL and PostgreSQL migrations.
3. **Common Repository Interface:** A repository abstraction pattern in the SDK core will allow swapping between local SQLite persistence and remote HTTP API metadata syncing.

## Consequences
### Positive
- Zero setup for local developers: `runvault.init()` immediately creates a local `.runvault/metadata.db` SQLite file.
- Scalable cloud infrastructure: PostgreSQL supports enterprise-scale concurrency and query optimization for team dashboards.

### Negative
- Maintaining schema parity between SQLite DDL and PostgreSQL Alembic migrations requires dual migration testing in CI.
- SQLite does not support certain advanced PostgreSQL features (e.g., native JSONB indexing, array operators).

## Alternatives Considered
- **Embedded DuckDB:** Evaluated, but requires compiling or downloading native binary wheels. Built-in `sqlite3` requires zero external packages.
- **Flat JSON/YAML Files:** Rejected due to poor search performance, lack of concurrent write safety, and absence of ACID guarantees during crashes.

## Open Implementation Details
- Offline-to-online sync strategy: CLI command or automated background sync to upload local SQLite run history to hosted PostgreSQL.
- Metric timeseries storage optimization in PostgreSQL (standard relational tables vs hypertable partitioning).
