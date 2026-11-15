"""Integration tests for the SQLiteMetadataRepository."""

import sqlite3
from pathlib import Path

import pytest

from runvault.core.db.connection import transaction
from runvault.core.db.repository import SQLiteMetadataRepository
from runvault.core.db.schema import CURRENT_SCHEMA_VERSION, get_current_schema_version
from runvault.core.domain import Checkpoint, Experiment, Metric, Run, RunStatus
from runvault.core.errors import (
    ConflictingParameterError,
    ExperimentNotFoundError,
    RunNotFoundError,
    TerminalRunError,
    WorkspaceError,
)


def test_schema_initialization_and_versioning(tmp_path: Path) -> None:
    """Verify that schema is properly initialized with version 1."""
    db_file = tmp_path / "metadata.db"
    repo = SQLiteMetadataRepository(db_file)
    try:
        assert get_current_schema_version(repo.connection) == CURRENT_SCHEMA_VERSION
    finally:
        repo.close()

    # Reopening the existing database should be idempotent without error
    repo2 = SQLiteMetadataRepository(db_file)
    try:
        assert get_current_schema_version(repo2.connection) == CURRENT_SCHEMA_VERSION
    finally:
        repo2.close()


def test_experiment_crud_and_retrieval(tmp_path: Path) -> None:
    """Verify experiment persistence, listing, and lookup by id/name."""
    repo = SQLiteMetadataRepository(tmp_path / "metadata.db")
    try:
        exp1 = Experiment(id="exp-1", name="cifar10-baseline", description="First trial")
        exp2 = Experiment(id="exp-2", name="cifar10-baseline", description="Second trial")
        repo.create_experiment(exp1)
        repo.create_experiment(exp2)

        # Retrieval by ID
        fetched = repo.get_experiment("exp-1")
        assert fetched is not None
        assert fetched.id == "exp-1"
        assert fetched.name == "cifar10-baseline"
        assert fetched.description == "First trial"

        # Nonexistent returns None
        assert repo.get_experiment("nonexistent") is None

        # Listing
        all_exps = repo.list_experiments()
        assert len(all_exps) == 2

        # Get by name (returns most recent)
        latest_baseline = repo.get_experiment_by_name("cifar10-baseline")
        assert latest_baseline is not None
        assert latest_baseline.id == "exp-2"
    finally:
        repo.close()


def test_run_crud_and_foreign_key_enforcement(tmp_path: Path) -> None:
    """Verify run creation, status updates, and foreign key constraint enforcement."""
    repo = SQLiteMetadataRepository(tmp_path / "metadata.db")
    try:
        # Creating a run for nonexistent experiment must fail
        orphan_run = Run(id="run-orphan", experiment_id="nonexistent-exp")
        with pytest.raises(ExperimentNotFoundError):
            repo.create_run(orphan_run)

        # Create experiment first
        exp = Experiment(id="exp-1", name="mnist")
        repo.create_experiment(exp)

        # Now create run
        run = Run(id="run-1", experiment_id="exp-1", name="epoch-1")
        repo.create_run(run)

        fetched = repo.get_run("run-1")
        assert fetched is not None
        assert fetched.id == "run-1"
        assert fetched.status == RunStatus.INITIALIZING

        # Transition status
        updated = repo.update_run_status(
            "run-1",
            RunStatus.RUNNING,
        )
        assert updated.status == RunStatus.RUNNING

        # Transition to FINISHED
        finished = repo.update_run_status(
            "run-1",
            RunStatus.FINISHED,
            duration_seconds=12.5,
        )
        assert finished.status == RunStatus.FINISHED
        assert finished.duration_seconds == 12.5

        # Updating nonexistent run raises RunNotFoundError
        with pytest.raises(RunNotFoundError):
            repo.update_run_status("nonexistent", RunStatus.FINISHED)
    finally:
        repo.close()


def test_parameter_persistence_and_conflict(tmp_path: Path) -> None:
    """Verify logging parameters, retrieving them, and detecting conflicts."""
    repo = SQLiteMetadataRepository(tmp_path / "metadata.db")
    try:
        exp = Experiment(id="exp-1", name="mnist")
        repo.create_experiment(exp)
        run = Run(id="run-1", experiment_id="exp-1", status=RunStatus.RUNNING)
        repo.create_run(run)

        repo.log_parameters("run-1", {"lr": 0.001, "batch_size": 32})
        params = repo.get_parameters("run-1")
        assert params == {"batch_size": 32, "lr": 0.001}

        # Idempotent re-log with same value
        repo.log_parameters("run-1", {"lr": 0.001})

        # Conflicting re-log
        with pytest.raises(ConflictingParameterError):
            repo.log_parameters("run-1", {"lr": 0.002})

        # Cannot log parameters to terminal run
        repo.update_run_status("run-1", RunStatus.FINISHED)
        with pytest.raises(TerminalRunError):
            repo.log_parameters("run-1", {"epochs": 10})
    finally:
        repo.close()


def test_metric_history_persistence_and_ordering(tmp_path: Path) -> None:
    """Verify metric batch logging, multiple metrics per step, and correct ordering."""
    repo = SQLiteMetadataRepository(tmp_path / "metadata.db")
    try:
        exp = Experiment(id="exp-1", name="mnist")
        repo.create_experiment(exp)
        run = Run(id="run-1", experiment_id="exp-1", status=RunStatus.RUNNING)
        repo.create_run(run)

        # Log metrics out of order
        metrics = [
            Metric(run_id="run-1", name="loss", value=0.8, step=1),
            Metric(run_id="run-1", name="acc", value=0.6, step=1),
            Metric(run_id="run-1", name="loss", value=0.4, step=2),
            Metric(run_id="run-1", name="acc", value=0.85, step=2),
        ]
        repo.log_metrics("run-1", metrics)

        # Retrieve all
        all_metrics = repo.get_metric_history("run-1")
        assert len(all_metrics) == 4

        # Retrieve specific metric
        loss_history = repo.get_metric_history("run-1", metric_name="loss")
        assert len(loss_history) == 2
        assert [m.step for m in loss_history] == [1, 2]
        assert [m.value for m in loss_history] == [0.8, 0.4]

        # Cannot log metrics to terminal run
        repo.update_run_status("run-1", RunStatus.FINISHED)
        with pytest.raises(TerminalRunError):
            repo.log_metrics("run-1", [Metric(run_id="run-1", name="loss", value=0.1)])
    finally:
        repo.close()


def test_tag_persistence_and_updates(tmp_path: Path) -> None:
    """Verify associating tags with experiments and runs."""
    repo = SQLiteMetadataRepository(tmp_path / "metadata.db")
    try:
        exp = Experiment(id="exp-1", name="mnist")
        repo.create_experiment(exp)
        run = Run(id="run-1", experiment_id="exp-1", status=RunStatus.RUNNING)
        repo.create_run(run)

        repo.set_tags("run", "run-1", {"env": "staging", "framework": "torch"})
        tags = repo.get_tags("run", "run-1")
        assert tags == {"env": "staging", "framework": "torch"}

        # Update tag value
        repo.set_tags("run", "run-1", {"env": "production"})
        updated_tags = repo.get_tags("run", "run-1")
        assert updated_tags["env"] == "production"

        # Experiment tags
        repo.set_tags("experiment", "exp-1", {"owner": "alice"})
        assert repo.get_tags("experiment", "exp-1") == {"owner": "alice"}
    finally:
        repo.close()


def test_transaction_rollback_on_failure(tmp_path: Path) -> None:
    """Verify that an error inside a transaction cleanly rolls back state."""
    repo = SQLiteMetadataRepository(tmp_path / "metadata.db")
    try:
        exp = Experiment(id="exp-1", name="mnist")
        repo.create_experiment(exp)

        with pytest.raises(sqlite3.IntegrityError), transaction(repo.connection) as cursor:
            cursor.execute(
                "INSERT INTO runs (id, experiment_id, status, created_at) VALUES (?, ?, ?, ?);",
                ("run-test", "exp-1", "RUNNING", "2026-01-01T00:00:00Z"),
            )
            # Intentionally trigger an error (foreign key violation or syntax error)
            cursor.execute(
                "INSERT INTO runs (id, experiment_id, status, created_at) VALUES (?, ?, ?, ?);",
                ("run-invalid", "nonexistent-exp", "RUNNING", "2026-01-01T00:00:00Z"),
            )

        # The first insert must have rolled back
        assert repo.get_run("run-test") is None
    finally:
        repo.close()


def test_tag_cascading_deletion_on_entity_removal(tmp_path: Path) -> None:
    """Verify that deleting a run or experiment cleanly cascades to remove its tags."""
    repo = SQLiteMetadataRepository(tmp_path / "metadata.db")
    try:
        exp = Experiment(id="exp-1", name="cascade-test")
        repo.create_experiment(exp)
        repo.set_tags("experiment", "exp-1", {"exp_tag": "active"})

        run = Run(id="run-1", experiment_id="exp-1", status=RunStatus.RUNNING)
        repo.create_run(run)
        repo.set_tags("run", "run-1", {"run_tag": "active"})

        # Verify initial tags exist
        assert repo.get_tags("run", "run-1") == {"run_tag": "active"}
        assert repo.get_tags("experiment", "exp-1") == {"exp_tag": "active"}

        # Delete run: run tags must cascade delete
        deleted_run = repo.delete_run("run-1")
        assert deleted_run is True
        assert repo.get_tags("run", "run-1") == {}
        # Experiment tags must remain untouched
        assert repo.get_tags("experiment", "exp-1") == {"exp_tag": "active"}

        # Delete experiment: experiment tags must cascade delete
        deleted_exp = repo.delete_experiment("exp-1")
        assert deleted_exp is True
        assert repo.get_tags("experiment", "exp-1") == {}
    finally:
        repo.close()


def test_tag_invalid_entity_type_rejected(tmp_path: Path) -> None:
    """Verify that invalid entity types are rejected with ValueError."""
    repo = SQLiteMetadataRepository(tmp_path / "metadata.db")
    try:
        with pytest.raises(ValueError, match="Invalid entity_type"):
            repo.set_tags("invalid_type", "id-1", {"key": "val"})

        with pytest.raises(ValueError, match="Invalid entity_type"):
            repo.get_tags("invalid_type", "id-1")
    finally:
        repo.close()


def test_incremental_schema_migration(tmp_path: Path) -> None:
    """Verify step-by-step migration from v1 to v2 preserves data and applies triggers."""
    db_file = tmp_path / "migration_test.db"

    # Step 1: Initialize database at version 1
    repo_v1 = SQLiteMetadataRepository(db_file, target_schema_version=1)
    try:
        assert get_current_schema_version(repo_v1.connection) == 1
        exp = Experiment(id="exp-v1", name="v1-experiment")
        repo_v1.create_experiment(exp)
    finally:
        repo_v1.close()

    # Step 2: Open with target version 2 (triggers migration 2)
    repo_v2 = SQLiteMetadataRepository(db_file, target_schema_version=2)
    try:
        assert get_current_schema_version(repo_v2.connection) == 2
        # Verify existing data preserved
        fetched = repo_v2.get_experiment("exp-v1")
        assert fetched is not None
        assert fetched.name == "v1-experiment"
    finally:
        repo_v2.close()

    # Step 3: Open with target version 3 (triggers migration 3 for checkpoints)
    repo_v3 = SQLiteMetadataRepository(db_file, target_schema_version=3)
    try:
        assert get_current_schema_version(repo_v3.connection) == 3
        fetched = repo_v3.get_experiment("exp-v1")
        assert fetched is not None
        assert fetched.name == "v1-experiment"
    finally:
        repo_v3.close()

    # Step 4: Open with target version 4 (triggers migration 4 for commit_seq sequence)
    repo_v4 = SQLiteMetadataRepository(db_file, target_schema_version=4)
    try:
        assert get_current_schema_version(repo_v4.connection) == 4
        fetched = repo_v4.get_experiment("exp-v1")
        assert fetched is not None
        assert fetched.name == "v1-experiment"
    finally:
        repo_v4.close()


def test_unsupported_future_schema_version_rejected(tmp_path: Path) -> None:
    """Verify that opening a database with a newer schema version fails clearly."""
    db_file = tmp_path / "future_db.db"
    repo = SQLiteMetadataRepository(db_file)
    try:
        # Simulate a future RunVault version writing schema version 99
        with transaction(repo.connection) as cursor:
            cursor.execute(
                """
                INSERT INTO schema_migrations (version, applied_at)
                VALUES (99, '2030-01-01T00:00:00Z');
                """
            )
    finally:
        repo.close()

    # Reopening should raise WorkspaceError explaining the version mismatch
    with pytest.raises(WorkspaceError, match="is newer than supported version"):
        SQLiteMetadataRepository(db_file)


def test_original_phase2_database_compatibility(tmp_path: Path) -> None:
    """Verify that a database produced by Phase 2 (with user_version=0) upgrades cleanly."""
    db_file = tmp_path / "legacy_phase2.db"
    raw_conn = sqlite3.connect(str(db_file))
    try:
        raw_conn.execute("PRAGMA journal_mode = WAL;")
        # Create Phase 2 v1 schema without triggers and user_version = 0
        raw_conn.execute(
            """
            CREATE TABLE experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        raw_conn.execute(
            """
            CREATE TABLE runs (
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
            """
        )
        raw_conn.execute(
            """
            CREATE TABLE parameters (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                value_json TEXT NOT NULL,
                PRIMARY KEY (run_id, name)
            );
            """
        )
        raw_conn.execute(
            """
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                step INTEGER,
                timestamp TEXT NOT NULL
            );
            """
        )
        raw_conn.execute(
            """
            CREATE TABLE tags (
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (entity_type, entity_id, key)
            );
            """
        )

        # Seed representative Phase 2 data
        raw_conn.execute(
            """
            INSERT INTO experiments VALUES
            ('exp-leg', 'legacy', 'desc', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z');
            """
        )
        raw_conn.execute(
            """
            INSERT INTO runs VALUES
            ('run-leg', 'exp-leg', 'r1', 'FINISHED', '2026-01-01T00:00:00Z',
             '2026-01-01T00:01:00Z', 60.0, NULL, NULL, NULL, '2026-01-01T00:00:00Z');
            """
        )
        raw_conn.execute("INSERT INTO parameters VALUES ('run-leg', 'batch_size', '32');")
        raw_conn.execute(
            """
            INSERT INTO metrics (run_id, name, value, step, timestamp)
            VALUES ('run-leg', 'loss', 0.5, 1, '2026-01-01T00:00:30Z');
            """
        )
        raw_conn.execute("INSERT INTO tags VALUES ('run', 'run-leg', 'env', 'prod');")
        raw_conn.commit()

        # Confirm user_version is 0
        cur = raw_conn.cursor()
        cur.execute("PRAGMA user_version;")
        assert cur.fetchone()[0] == 0
    finally:
        raw_conn.close()

    # Open with RunVault SQLiteMetadataRepository: should auto-upgrade to current version
    repo = SQLiteMetadataRepository(db_file)
    try:
        assert get_current_schema_version(repo.connection) == CURRENT_SCHEMA_VERSION

        # Verify all seeded records are completely preserved
        exp = repo.get_experiment("exp-leg")
        assert exp is not None
        assert exp.name == "legacy"

        run = repo.get_run("run-leg")
        assert run is not None
        assert run.status == RunStatus.FINISHED

        params = repo.get_parameters("run-leg")
        assert params == {"batch_size": 32}

        metrics = repo.get_metric_history("run-leg")
        assert len(metrics) == 1
        assert metrics[0].value == 0.5

        tags = repo.get_tags("run", "run-leg")
        assert tags == {"env": "prod"}

        # Verify that upgraded triggers work on deletion
        repo.delete_run("run-leg")
        assert repo.get_tags("run", "run-leg") == {}
    finally:
        repo.close()


def test_direct_sql_deletion_cascade_triggers(tmp_path: Path) -> None:
    """Verify that triggers properly fire on direct raw SQL deletion."""
    repo = SQLiteMetadataRepository(tmp_path / "raw_cascade.db")
    try:
        exp = Experiment(id="e-raw", name="raw-test")
        repo.create_experiment(exp)
        run = Run(id="r-raw", experiment_id="e-raw", status=RunStatus.RUNNING)
        repo.create_run(run)
        repo.set_tags("run", "r-raw", {"raw_tag": "v"})

        # Direct SQL deletion bypassing repository methods
        with transaction(repo.connection) as cursor:
            cursor.execute("DELETE FROM runs WHERE id = 'r-raw';")

        assert repo.get_tags("run", "r-raw") == {}
    finally:
        repo.close()


def test_tag_cannot_reference_nonexistent_entity(tmp_path: Path) -> None:
    """Verify that associating tags with missing entities fails clearly."""
    repo = SQLiteMetadataRepository(tmp_path / "tag_missing.db")
    try:
        with pytest.raises(RunNotFoundError):
            repo.set_tags("run", "nonexistent-run", {"k": "v"})

        with pytest.raises(ExperimentNotFoundError):
            repo.set_tags("experiment", "nonexistent-exp", {"k": "v"})
    finally:
        repo.close()


def test_checkpoint_synchronous_pragma_sequencing_and_transaction_state(tmp_path: Path) -> None:
    """Checkpoint metadata commit runs under FULL synchronous mode, restoring NORMAL afterwards."""
    repo = SQLiteMetadataRepository(tmp_path / "pragma_sync.db")
    try:
        exp = Experiment(id="e-sync", name="sync-test")
        repo.create_experiment(exp)
        run = Run(id="r-sync", experiment_id="e-sync", status=RunStatus.RUNNING)
        repo.create_run(run)

        # Baseline synchronous setting is NORMAL (1)
        cur = repo.connection.cursor()
        cur.execute("PRAGMA synchronous;")
        assert cur.fetchone()[0] == 1
        assert repo.connection.in_transaction is False

        # Operational write (parameters) remains NORMAL (1)
        repo.log_parameters("r-sync", {"lr": 0.01})
        cur.execute("PRAGMA synchronous;")
        assert cur.fetchone()[0] == 1
        assert repo.connection.in_transaction is False

        # Successful create_checkpoint: sets FULL (2) for commit and restores NORMAL (1)
        ckpt = Checkpoint(
            id="c-sync-1",
            run_id="r-sync",
            name="ep-1",
            artifact_path="checkpoints/r-sync/c-sync-1.bin",
            size_bytes=10,
            checksum_sha256="a" * 64,
        )
        saved = repo.create_checkpoint(ckpt)
        assert saved.commit_seq is not None
        assert repo.connection.in_transaction is False

        # Confirmed restored to NORMAL (1)
        cur.execute("PRAGMA synchronous;")
        assert cur.fetchone()[0] == 1

        # Failure during create_checkpoint (e.g. duplicate id): rolls back and restores NORMAL
        with pytest.raises(sqlite3.IntegrityError):
            repo.create_checkpoint(ckpt)

        assert repo.connection.in_transaction is False
        cur.execute("PRAGMA synchronous;")
        assert cur.fetchone()[0] == 1
    finally:
        repo.close()


def test_migration_v3_to_v4_with_existing_records_and_metadata(tmp_path: Path) -> None:
    """Migrate schema v3 database with checkpoints, verifying commit_seq and integrity."""
    db_file = tmp_path / "v3_legacy.db"

    # Initialize a database at schema v3
    repo_v3 = SQLiteMetadataRepository(db_file, target_schema_version=3)
    try:
        assert get_current_schema_version(repo_v3.connection) == 3
        exp = Experiment(id="e-v3", name="v3-exp")
        repo_v3.create_experiment(exp)
        run = Run(id="r-v3", experiment_id="e-v3", status=RunStatus.RUNNING)
        repo_v3.create_run(run)

        # Insert 2 checkpoints into v3 table (ordered by created_at)
        sha1 = "1" * 64
        sha2 = "2" * 64
        raw_cur = repo_v3.connection.cursor()
        raw_cur.execute(
            """
            INSERT INTO checkpoints (
                id, run_id, name, artifact_path, size_bytes, checksum_sha256,
                step, format_version, metadata_json, created_at
            ) VALUES
            ('c-v3-1', 'r-v3', 'first', 'checkpoints/r-v3/c-v3-1.bin', 10, ?, 1, 1,
             '{"acc": 0.8}', '2026-01-01T10:00:00'),
            ('c-v3-2', 'r-v3', 'second', 'checkpoints/r-v3/c-v3-2.bin', 20, ?, 2, 1,
             '{"acc": 0.9}', '2026-01-01T11:00:00');
            """,
            (sha1, sha2),
        )
        repo_v3.connection.commit()
    finally:
        repo_v3.close()

    # Reopen at current version (v4)
    repo_v4 = SQLiteMetadataRepository(db_file)
    try:
        assert get_current_schema_version(repo_v4.connection) == 4

        # Verify checkpoints survived with deterministic commit_seq assigned
        ckpts = repo_v4.list_checkpoints("r-v3")
        assert len(ckpts) == 2

        assert ckpts[0].id == "c-v3-1"
        assert ckpts[0].commit_seq == 1
        assert ckpts[0].metadata == {"acc": 0.8}
        assert ckpts[0].step == 1

        assert ckpts[1].id == "c-v3-2"
        assert ckpts[1].commit_seq == 2
        assert ckpts[1].metadata == {"acc": 0.9}
        assert ckpts[1].step == 2

        # Verify latest checkpoint query returns c-v3-2
        latest = repo_v4.get_latest_checkpoint("r-v3")
        assert latest is not None
        assert latest.id == "c-v3-2"

        # Insert a new checkpoint in v4: receives commit_seq == 3
        new_ckpt = Checkpoint(
            id="c-v4-new",
            run_id="r-v3",
            name="third",
            artifact_path="checkpoints/r-v3/c-v4-new.bin",
            size_bytes=30,
            checksum_sha256="3" * 64,
            step=3,
        )
        saved = repo_v4.create_checkpoint(new_ckpt)
        assert saved.commit_seq == 3

        # Verify cascading deletion of run removes all checkpoints in v4 table
        repo_v4.delete_run("r-v3")
        assert repo_v4.list_checkpoints("r-v3") == []
        assert repo_v4.get_checkpoint("c-v3-1") is None
        assert repo_v4.get_checkpoint("c-v4-new") is None
    finally:
        repo_v4.close()


def test_migration_atomic_rollback_on_failure(tmp_path: Path) -> None:
    """An error during migration rolls back cleanly without leaving partial schema state."""
    db_file = tmp_path / "rollback_test.db"

    # Start with v1
    repo_v1 = SQLiteMetadataRepository(db_file, target_schema_version=1)
    try:
        assert get_current_schema_version(repo_v1.connection) == 1
    finally:
        repo_v1.close()

    # Attempt migration with a simulated failure in schema initialization
    raw_conn = sqlite3.connect(str(db_file))
    try:
        # Pre-create a conflicting table name that would break a migration statement
        raw_conn.execute("CREATE TABLE _checkpoints_v3 (id INT);")
        raw_conn.commit()
    finally:
        raw_conn.close()

    # Opening repository attempts migration v2 -> v3 -> v4;
    # v4 fails when renaming checkpoints to _checkpoints_v3 because it already exists
    with pytest.raises(sqlite3.OperationalError):
        SQLiteMetadataRepository(db_file)

    # Database connection should be able to verify that user_version did not reach 4
    check_conn = sqlite3.connect(str(db_file))
    try:
        cur = check_conn.cursor()
        cur.execute("PRAGMA user_version;")
        ver = cur.fetchone()[0]
        assert ver < 4
    finally:
        check_conn.close()


def test_phase2_v1_migration_through_all_versions_with_foreign_keys_and_cascades(
    tmp_path: Path,
) -> None:
    """Migrate a Phase 2 v1 schema database through v2, v3, to v4, verifying integrity."""
    db_file = tmp_path / "v1_legacy.db"

    # Step 1: Initialize database at schema v1
    repo_v1 = SQLiteMetadataRepository(db_file, target_schema_version=1)
    try:
        assert get_current_schema_version(repo_v1.connection) == 1
        exp = Experiment(id="e-v1", name="v1-exp")
        repo_v1.create_experiment(exp)
        run = Run(id="r-v1", experiment_id="e-v1", status=RunStatus.RUNNING)
        repo_v1.create_run(run)
        repo_v1.log_parameters("r-v1", {"learning_rate": 0.001})
        repo_v1.log_metrics(
            "r-v1",
            [Metric(run_id="r-v1", name="loss", value=0.42, step=1)],
        )
        repo_v1.set_tags("run", "r-v1", {"stage": "train"})
    finally:
        repo_v1.close()

    # Step 2: Open with target version v4 (current schema)
    repo_v4 = SQLiteMetadataRepository(db_file)
    try:
        assert get_current_schema_version(repo_v4.connection) == 4

        # Verify all v1 data survived
        assert repo_v4.get_experiment("e-v1") is not None
        assert repo_v4.get_run("r-v1") is not None
        assert repo_v4.get_parameters("r-v1") == {"learning_rate": 0.001}
        metrics = repo_v4.get_metric_history("r-v1")
        assert len(metrics) == 1
        assert metrics[0].value == 0.42
        assert repo_v4.get_tags("run", "r-v1") == {"stage": "train"}

        # Verify domain check and foreign-key enforcement on migrated DB:
        # Repository rejects nonexistent run
        orphan_ckpt = Checkpoint(
            id="c-orphan",
            run_id="nonexistent-run-id",
            name="orphan-ckpt",
            artifact_path="checkpoints/nonexistent/c-orphan.bin",
            size_bytes=100,
            checksum_sha256="4" * 64,
        )
        with pytest.raises(RunNotFoundError):
            repo_v4.create_checkpoint(orphan_ckpt)

        # Raw SQLite foreign key constraint rejects orphaned checkpoint insertion
        with (
            pytest.raises(sqlite3.IntegrityError),
            transaction(repo_v4.connection) as cursor,
        ):
            cursor.execute(
                """
                INSERT INTO checkpoints (
                    id, run_id, name, artifact_path, size_bytes, checksum_sha256,
                    step, format_version, metadata_json, created_at
                ) VALUES (
                    'c-orphan-raw', 'nonexistent-run-id', 'raw', 'p', 10, ?, 1, 1, '{}',
                    '2026-01-01'
                );
                """,
                ("4" * 64,),
            )

        # Inserting a valid checkpoint succeeds
        valid_ckpt = Checkpoint(
            id="c-valid",
            run_id="r-v1",
            name="valid-ckpt",
            artifact_path="checkpoints/r-v1/c-valid.bin",
            size_bytes=100,
            checksum_sha256="5" * 64,
            step=10,
        )
        saved = repo_v4.create_checkpoint(valid_ckpt)
        assert saved.commit_seq == 1
        assert repo_v4.list_checkpoints("r-v1") == [saved]

        # Verify cascading deletion after migration: deleting run cascades to everything
        repo_v4.delete_run("r-v1")
        assert repo_v4.get_run("r-v1") is None
        assert repo_v4.get_parameters("r-v1") == {}
        assert repo_v4.get_metric_history("r-v1") == []
        assert repo_v4.get_tags("run", "r-v1") == {}
        assert repo_v4.list_checkpoints("r-v1") == []
    finally:
        repo_v4.close()
