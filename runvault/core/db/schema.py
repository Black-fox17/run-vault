"""SQLite schema definition, indexing, and migration version tracking."""

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

from runvault.core.errors import WorkspaceError

CURRENT_SCHEMA_VERSION: Final[int] = 2

SCHEMA_V1_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name);
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
        name TEXT,
        status TEXT NOT NULL,
        start_time TEXT,
        end_time TEXT,
        duration_seconds REAL,
        error_message TEXT,
        error_type TEXT,
        error_traceback TEXT,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs(experiment_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
    """,
    """
    CREATE TABLE IF NOT EXISTS parameters (
        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        value_json TEXT NOT NULL,
        PRIMARY KEY (run_id, name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        value REAL NOT NULL,
        step INTEGER,
        timestamp TEXT NOT NULL
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_metrics_run_name_step ON metrics(run_id, name, step);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_metrics_run_timestamp ON metrics(run_id, timestamp);
    """,
    """
    CREATE TABLE IF NOT EXISTS tags (
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (entity_type, entity_id, key)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tags_entity ON tags(entity_type, entity_id);
    """,
)

SCHEMA_V2_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TRIGGER IF NOT EXISTS trigger_delete_experiment_tags
    AFTER DELETE ON experiments
    FOR EACH ROW
    BEGIN
        DELETE FROM tags WHERE entity_type = 'experiment' AND entity_id = OLD.id;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trigger_delete_run_tags
    AFTER DELETE ON runs
    FOR EACH ROW
    BEGIN
        DELETE FROM tags WHERE entity_type = 'run' AND entity_id = OLD.id;
    END;
    """,
)


def _apply_v1(cursor: sqlite3.Cursor) -> None:
    for stmt in SCHEMA_V1_STATEMENTS:
        cursor.execute(stmt)


def _apply_v2(cursor: sqlite3.Cursor) -> None:
    for stmt in SCHEMA_V2_STATEMENTS:
        cursor.execute(stmt)


MIGRATIONS: Final[dict[int, Callable[[sqlite3.Cursor], None]]] = {
    1: _apply_v1,
    2: _apply_v2,
}


def get_current_schema_version(conn: sqlite3.Connection) -> int:
    """Return the applied schema version from the database, or 0 if uninitialized.

    Supports both PRAGMA user_version and backward compatibility with Phase 2 databases
    that stored versions in a `schema_migrations` table or created v1 tables directly.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA user_version;")
        row = cursor.fetchone()
        user_ver = int(row[0]) if row and row[0] is not None else 0

        # Check if schema_migrations table exists (Phase 2 migration format)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations';"
        )
        mig_ver = 0
        if cursor.fetchone() is not None:
            cursor.execute("SELECT MAX(version) FROM schema_migrations;")
            mig_row = cursor.fetchone()
            if mig_row and mig_row[0] is not None:
                mig_ver = int(mig_row[0])

        effective_ver = max(user_ver, mig_ver)
        if effective_ver > 0:
            return effective_ver

        # Check if experiments table exists (original unversioned Phase 2 schema)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiments';")
        if cursor.fetchone() is not None:
            return 1

        return 0
    finally:
        cursor.close()


def initialize_schema(
    conn: sqlite3.Connection,
    target_version: int = CURRENT_SCHEMA_VERSION,
) -> None:
    """Initialize or migrate the database schema safely in an atomic transaction."""
    current_version = get_current_schema_version(conn)

    if current_version == target_version:
        return

    if current_version > target_version:
        msg = (
            f"Database schema version {current_version} is newer than supported "
            f"version {target_version}. Please upgrade RunVault."
        )
        raise WorkspaceError(path="database", reason=msg)

    cursor = conn.cursor()
    cursor.execute("BEGIN IMMEDIATE;")
    try:
        # Re-check version inside IMMEDIATE transaction to prevent concurrent race conditions
        current_version = get_current_schema_version(conn)
        if current_version == target_version:
            conn.execute("COMMIT;")
            return

        if current_version > target_version:
            msg = (
                f"Database schema version {current_version} is newer than supported "
                f"version {target_version}. Please upgrade RunVault."
            )
            raise WorkspaceError(path="database", reason=msg)

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            """
        )

        for version in range(current_version + 1, target_version + 1):
            migration_fn = MIGRATIONS.get(version)
            if migration_fn is not None:
                migration_fn(cursor)
                now_iso = datetime.now(UTC).isoformat()
                cursor.execute(
                    "INSERT OR REPLACE INTO schema_migrations (version, applied_at) VALUES (?, ?);",
                    (version, now_iso),
                )
                cursor.execute(f"PRAGMA user_version = {version};")

        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        cursor.close()
