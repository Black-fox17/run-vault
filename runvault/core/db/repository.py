"""SQLite implementation of the RunVault metadata repository."""

import dataclasses
import json
import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from runvault.core.db.connection import create_connection, transaction
from runvault.core.db.schema import CURRENT_SCHEMA_VERSION, initialize_schema
from runvault.core.domain import (
    Checkpoint,
    Experiment,
    Metric,
    Run,
    RunStatus,
    check_parameter_conflict,
    utc_now,
    validate_json_serializable,
    validate_status_transition,
)
from runvault.core.errors import (
    ExperimentNotFoundError,
    RunNotFoundError,
    TerminalRunError,
    WorkspaceError,
)

VALID_TAG_ENTITY_TYPES = frozenset({"experiment", "run"})


def _iso_to_datetime(iso_str: str | None) -> datetime | None:
    if not iso_str:
        return None
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _row_to_experiment(row: sqlite3.Row) -> Experiment:
    description = str(row["description"]) if row["description"] is not None else None
    return Experiment(
        id=str(row["id"]),
        name=str(row["name"]),
        description=description,
        created_at=_iso_to_datetime(str(row["created_at"])) or utc_now(),
        updated_at=_iso_to_datetime(str(row["updated_at"])) or utc_now(),
    )


def _row_to_run(row: sqlite3.Row) -> Run:
    name = str(row["name"]) if row["name"] is not None else None
    duration = float(row["duration_seconds"]) if row["duration_seconds"] is not None else None
    err_msg = str(row["error_message"]) if row["error_message"] is not None else None
    err_type = str(row["error_type"]) if row["error_type"] is not None else None
    err_tb = str(row["error_traceback"]) if row["error_traceback"] is not None else None
    return Run(
        id=str(row["id"]),
        experiment_id=str(row["experiment_id"]),
        name=name,
        status=RunStatus(str(row["status"])),
        start_time=_iso_to_datetime(row["start_time"]),
        end_time=_iso_to_datetime(row["end_time"]),
        duration_seconds=duration,
        error_message=err_msg,
        error_type=err_type,
        error_traceback=err_tb,
        created_at=_iso_to_datetime(str(row["created_at"])) or utc_now(),
    )


def _row_to_checkpoint(row: sqlite3.Row) -> Checkpoint:
    metadata_json = str(row["metadata_json"]) if row["metadata_json"] is not None else None
    row_keys = set(row.keys())
    commit_seq = (
        int(row["commit_seq"])
        if "commit_seq" in row_keys and row["commit_seq"] is not None
        else None
    )
    return Checkpoint(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        name=str(row["name"]),
        artifact_path=str(row["artifact_path"]),
        size_bytes=int(row["size_bytes"]),
        checksum_sha256=str(row["checksum_sha256"]),
        step=int(row["step"]) if row["step"] is not None else None,
        format_version=int(row["format_version"]),
        metadata=Checkpoint.deserialize_metadata(metadata_json),
        created_at=_iso_to_datetime(str(row["created_at"])) or utc_now(),
        commit_seq=commit_seq,
    )


class SQLiteMetadataRepository:
    """Manages metadata persistence using thread-safe SQLite connections."""

    def __init__(self, db_path: Path, target_schema_version: int = CURRENT_SCHEMA_VERSION) -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._closed = False
        self._lock = threading.Lock()

        # Initialize schema once upon creation using the initial caller's thread
        initialize_schema(self.connection, target_version=target_schema_version)

    @property
    def is_closed(self) -> bool:
        """Return True if the repository has been marked as closed."""
        return self._closed

    @property
    def connection(self) -> sqlite3.Connection:
        """Return a SQLite connection strictly owned by the calling thread."""
        with self._lock:
            if self._closed:
                self.close_thread_connection()
                raise WorkspaceError(
                    path=str(self.db_path),
                    reason="Repository has been closed.",
                )

        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = create_connection(self.db_path)
            self._local.connection = conn
        return conn

    def close_thread_connection(self) -> None:
        """Close the SQLite connection owned by the calling thread, if open."""
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.connection = None

    def close(self) -> None:
        """Mark repository as closed and close the calling thread's connection.

        Contract:
        Marks the repository closed. Any subsequent access from any thread will raise
        WorkspaceError. The calling thread's connection is closed immediately.
        Connections created by other active worker threads will be closed when those threads
        next access the repository or terminate naturally.
        """
        with self._lock:
            self._closed = True
        self.close_thread_connection()

    # -------------------------------------------------------------------------
    # Experiment operations
    # -------------------------------------------------------------------------

    def create_experiment(self, experiment: Experiment) -> Experiment:
        """Persist a new experiment."""
        with transaction(self.connection) as cursor:
            cursor.execute(
                """
                INSERT INTO experiments (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    experiment.id,
                    experiment.name,
                    experiment.description,
                    experiment.created_at.isoformat(),
                    experiment.updated_at.isoformat(),
                ),
            )
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Retrieve an experiment by its unique identifier."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT id, name, description, created_at, updated_at
                FROM experiments
                WHERE id = ?;
                """,
                (experiment_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return _row_to_experiment(row)
        finally:
            cursor.close()

    def get_experiment_by_name(self, name: str) -> Experiment | None:
        """Retrieve the most recently created experiment matching a given name."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT id, name, description, created_at, updated_at
                FROM experiments
                WHERE name = ?
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                (name,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return _row_to_experiment(row)
        finally:
            cursor.close()

    def list_experiments(self) -> list[Experiment]:
        """List all experiments ordered by creation timestamp descending."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT id, name, description, created_at, updated_at
                FROM experiments
                ORDER BY created_at DESC;
                """
            )
            return [_row_to_experiment(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and cascade to runs and tags."""
        with transaction(self.connection) as cursor:
            cursor.execute("DELETE FROM experiments WHERE id = ?;", (experiment_id,))
            return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Run operations
    # -------------------------------------------------------------------------

    def create_run(self, run: Run) -> Run:
        """Persist a new experiment run, verifying parent experiment existence."""
        if not self.get_experiment(run.experiment_id):
            raise ExperimentNotFoundError(run.experiment_id)

        with transaction(self.connection) as cursor:
            cursor.execute(
                """
                INSERT INTO runs (
                    id, experiment_id, name, status, start_time, end_time,
                    duration_seconds, error_message, error_type, error_traceback, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run.id,
                    run.experiment_id,
                    run.name,
                    run.status.value,
                    run.start_time.isoformat() if run.start_time else None,
                    run.end_time.isoformat() if run.end_time else None,
                    run.duration_seconds,
                    run.error_message,
                    run.error_type,
                    run.error_traceback,
                    run.created_at.isoformat(),
                ),
            )
        return run

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by its unique identifier."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT id, experiment_id, name, status, start_time, end_time,
                       duration_seconds, error_message, error_type, error_traceback, created_at
                FROM runs
                WHERE id = ?;
                """,
                (run_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return _row_to_run(row)
        finally:
            cursor.close()

    def list_runs(self, experiment_id: str) -> list[Run]:
        """List all runs belonging to an experiment, ordered by creation timestamp descending."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT id, experiment_id, name, status, start_time, end_time,
                       duration_seconds, error_message, error_type, error_traceback, created_at
                FROM runs
                WHERE experiment_id = ?
                ORDER BY created_at DESC;
                """,
                (experiment_id,),
            )
            return [_row_to_run(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and cascade to metrics, parameters, and tags."""
        with transaction(self.connection) as cursor:
            cursor.execute("DELETE FROM runs WHERE id = ?;", (run_id,))
            return cursor.rowcount > 0

    def update_run_status(
        self,
        run_id: str,
        target_status: RunStatus,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        duration_seconds: float | None = None,
        error_message: str | None = None,
        error_type: str | None = None,
        error_traceback: str | None = None,
    ) -> Run:
        """Transition a run's status with lifecycle validation and error recording."""
        run = self.get_run(run_id)
        if run is None:
            raise RunNotFoundError(run_id)

        validate_status_transition(run.status, target_status, run_id)

        with transaction(self.connection) as cursor:
            cursor.execute(
                """
                UPDATE runs
                SET status = ?,
                    start_time = COALESCE(?, start_time),
                    end_time = COALESCE(?, end_time),
                    duration_seconds = COALESCE(?, duration_seconds),
                    error_message = COALESCE(?, error_message),
                    error_type = COALESCE(?, error_type),
                    error_traceback = COALESCE(?, error_traceback)
                WHERE id = ?;
                """,
                (
                    target_status.value,
                    start_time.isoformat() if start_time else None,
                    end_time.isoformat() if end_time else None,
                    duration_seconds,
                    error_message,
                    error_type,
                    error_traceback,
                    run_id,
                ),
            )

        updated = self.get_run(run_id)
        if updated is None:
            raise RunNotFoundError(run_id)
        return updated

    # -------------------------------------------------------------------------
    # Parameter operations
    # -------------------------------------------------------------------------

    def log_parameters(self, run_id: str, parameters: dict[str, Any]) -> None:
        """Persist parameters for a run with conflict detection and idempotency."""
        run = self.get_run(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        if run.status.is_terminal:
            raise TerminalRunError(run_id, run.status.value, "log_parameters")

        existing_params = self.get_parameters(run_id)

        # Validate types and non-conflicting values first
        for name, value in parameters.items():
            validate_json_serializable(value, name)
            if name in existing_params:
                check_parameter_conflict(name, existing_params[name], value, run_id)

        # Persist new parameters in a single transaction
        with transaction(self.connection) as cursor:
            for name, value in parameters.items():
                if name not in existing_params:
                    val_json = json.dumps(value, allow_nan=False, sort_keys=True)
                    cursor.execute(
                        "INSERT INTO parameters (run_id, name, value_json) VALUES (?, ?, ?);",
                        (run_id, name, val_json),
                    )

    def get_parameters(self, run_id: str) -> dict[str, Any]:
        """Retrieve all configuration parameters logged for a run."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "SELECT name, value_json FROM parameters WHERE run_id = ? ORDER BY name ASC;",
                (run_id,),
            )
            params: dict[str, Any] = {}
            for row in cursor.fetchall():
                params[str(row["name"])] = json.loads(str(row["value_json"]))
            return params
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    # Metric operations
    # -------------------------------------------------------------------------

    def log_metrics(self, run_id: str, metrics: Sequence[Metric]) -> None:
        """Persist a sequence of metrics for a run in a single atomic transaction."""
        run = self.get_run(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        if run.status.is_terminal:
            raise TerminalRunError(run_id, run.status.value, "log_metrics")

        with transaction(self.connection) as cursor:
            cursor.executemany(
                """
                INSERT INTO metrics (run_id, name, value, step, timestamp)
                VALUES (?, ?, ?, ?, ?);
                """,
                [
                    (
                        m.run_id,
                        m.name,
                        float(m.value),
                        m.step,
                        m.timestamp.isoformat(),
                    )
                    for m in metrics
                ],
            )

    def get_metric_history(self, run_id: str, metric_name: str | None = None) -> list[Metric]:
        """Retrieve ordered metric history for a run."""
        cursor = self.connection.cursor()
        order_clause = (
            "ORDER BY CASE WHEN step IS NULL THEN 1 ELSE 0 END, step ASC, timestamp ASC, id ASC;"
        )
        try:
            if metric_name is not None:
                cursor.execute(
                    f"""
                    SELECT run_id, name, value, step, timestamp
                    FROM metrics
                    WHERE run_id = ? AND name = ?
                    {order_clause}
                    """,
                    (run_id, metric_name),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT run_id, name, value, step, timestamp
                    FROM metrics
                    WHERE run_id = ?
                    {order_clause}
                    """,
                    (run_id,),
                )

            records: list[Metric] = []
            for row in cursor.fetchall():
                records.append(
                    Metric(
                        run_id=str(row["run_id"]),
                        name=str(row["name"]),
                        value=float(row["value"]),
                        step=int(row["step"]) if row["step"] is not None else None,
                        timestamp=_iso_to_datetime(str(row["timestamp"])) or utc_now(),
                    )
                )
            return records
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    # Tag operations
    # -------------------------------------------------------------------------

    def set_tags(self, entity_type: str, entity_id: str, tags: dict[str, str]) -> None:
        """Associate string key-value tags with an experiment or run."""
        if entity_type not in VALID_TAG_ENTITY_TYPES:
            valid_types = sorted(VALID_TAG_ENTITY_TYPES)
            raise ValueError(f"Invalid entity_type '{entity_type}'. Must be one of {valid_types}")

        if entity_type == "run":
            run = self.get_run(entity_id)
            if run is None:
                raise RunNotFoundError(entity_id)
            if run.status.is_terminal:
                raise TerminalRunError(entity_id, run.status.value, "set_tags")
        elif entity_type == "experiment":
            exp = self.get_experiment(entity_id)
            if exp is None:
                raise ExperimentNotFoundError(entity_id)

        with transaction(self.connection) as cursor:
            for key, val in tags.items():
                cursor.execute(
                    """
                    INSERT INTO tags (entity_type, entity_id, key, value)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(entity_type, entity_id, key) DO UPDATE SET value = excluded.value;
                    """,
                    (entity_type, entity_id, str(key), str(val)),
                )

    def get_tags(self, entity_type: str, entity_id: str) -> dict[str, str]:
        """Retrieve all tags associated with an entity."""
        if entity_type not in VALID_TAG_ENTITY_TYPES:
            valid_types = sorted(VALID_TAG_ENTITY_TYPES)
            raise ValueError(f"Invalid entity_type '{entity_type}'. Must be one of {valid_types}")

        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT key, value FROM tags
                WHERE entity_type = ? AND entity_id = ?
                ORDER BY key ASC;
                """,
                (entity_type, entity_id),
            )
            return {str(row["key"]): str(row["value"]) for row in cursor.fetchall()}
        finally:
            cursor.close()

    # -------------------------------------------------------------------------
    # Checkpoint operations
    # -------------------------------------------------------------------------

    def create_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        """Persist a new checkpoint record in SQLite with full durability."""
        run = self.get_run(checkpoint.run_id)
        if run is None:
            raise RunNotFoundError(checkpoint.run_id)
        if run.status.is_terminal:
            raise TerminalRunError(checkpoint.run_id, run.status.value, "create_checkpoint")

        assigned_seq: int | None = None
        cursor = self.connection.cursor()
        try:
            self.connection.execute("PRAGMA synchronous = FULL;")
            try:
                self.connection.execute("BEGIN IMMEDIATE;")
                try:
                    cursor.execute(
                        """
                        INSERT INTO checkpoints (
                            id, run_id, name, artifact_path, size_bytes, checksum_sha256,
                            step, format_version, metadata_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            checkpoint.id,
                            checkpoint.run_id,
                            checkpoint.name,
                            checkpoint.artifact_path,
                            checkpoint.size_bytes,
                            checkpoint.checksum_sha256,
                            checkpoint.step,
                            checkpoint.format_version,
                            checkpoint.serialize_metadata(),
                            checkpoint.created_at.isoformat(),
                        ),
                    )
                    assigned_seq = cursor.lastrowid
                    self.connection.execute("COMMIT;")
                except Exception:
                    self.connection.execute("ROLLBACK;")
                    raise
            finally:
                self.connection.execute("PRAGMA synchronous = NORMAL;")
        finally:
            cursor.close()

        if assigned_seq is not None:
            return dataclasses.replace(checkpoint, commit_seq=assigned_seq)
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Retrieve a specific checkpoint by ID."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT id, run_id, name, artifact_path, size_bytes, checksum_sha256,
                       step, format_version, metadata_json, created_at, commit_seq
                FROM checkpoints
                WHERE id = ?;
                """,
                (checkpoint_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return _row_to_checkpoint(row)
        finally:
            cursor.close()

    def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        """List all checkpoints belonging to a run in deterministic chronological order."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT id, run_id, name, artifact_path, size_bytes, checksum_sha256,
                       step, format_version, metadata_json, created_at, commit_seq
                FROM checkpoints
                WHERE run_id = ?
                ORDER BY commit_seq ASC;
                """,
                (run_id,),
            )
            return [_row_to_checkpoint(row) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def get_latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        """Retrieve the newest committed checkpoint according to commit sequence ordering."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT id, run_id, name, artifact_path, size_bytes, checksum_sha256,
                       step, format_version, metadata_json, created_at, commit_seq
                FROM checkpoints
                WHERE run_id = ?
                ORDER BY commit_seq DESC
                LIMIT 1;
                """,
                (run_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return _row_to_checkpoint(row)
        finally:
            cursor.close()

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint record from SQLite."""
        with transaction(self.connection) as cursor:
            cursor.execute("DELETE FROM checkpoints WHERE id = ?;", (checkpoint_id,))
            return cursor.rowcount > 0
