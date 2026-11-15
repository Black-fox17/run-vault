"""SQLite connection management, PRAGMA configuration, and transaction boundaries."""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


def create_connection(db_path: Path) -> sqlite3.Connection:
    """Create and configure a new SQLite connection with strict settings.

    Enforces WAL mode, foreign keys, busy timeout, and auto-commit behavior.
    """
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # autocommit=False ensures transactions are controlled explicitly
    conn = sqlite3.connect(
        str(db_path),
        timeout=10.0,
        isolation_level=None,  # Enables explicit transaction management
        check_same_thread=True,
    )

    conn.row_factory = sqlite3.Row

    # Configure high-performance and data-integrity PRAGMAs
    with conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA recursive_triggers = ON;")
        conn.execute("PRAGMA busy_timeout = 10000;")

    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Cursor, None, None]:
    """Provide a transactional cursor with immediate write lock and rollback on error."""
    cursor = conn.cursor()
    cursor.execute("BEGIN IMMEDIATE;")
    try:
        yield cursor
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        cursor.close()
