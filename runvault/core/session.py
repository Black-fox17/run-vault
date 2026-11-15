"""Execution session, ActiveRun context manager, and ExperimentHandle."""

import traceback
import uuid
from datetime import datetime
from typing import Any

from runvault.core.db.repository import SQLiteMetadataRepository
from runvault.core.domain import (
    Experiment,
    Metric,
    Run,
    RunStatus,
    utc_now,
)
from runvault.core.errors import (
    InvalidLifecycleTransitionError,
    TerminalRunError,
)

MAX_ERROR_MSG_LENGTH = 1024
MAX_TRACEBACK_LENGTH = 4096
MAX_TRACEBACK_LINES = 40


class ActiveRun:
    """An active experiment execution run, managed as a context manager.

    Usage:
        with experiment.start_run(name="trial-1") as run:
            run.log_params({"lr": 0.001})
            run.log_metrics({"loss": 0.25}, step=1)
    """

    def __init__(
        self,
        repository: SQLiteMetadataRepository,
        experiment_id: str,
        name: str | None = None,
        run_id: str | None = None,
        initial_tags: dict[str, str] | None = None,
    ) -> None:
        self._repo = repository
        self._experiment_id = experiment_id
        self._run_id = run_id or str(uuid.uuid4())
        self._name = name
        self._initial_tags = initial_tags or {}
        self._status = RunStatus.INITIALIZING
        self._start_time: datetime | None = None
        self._closed = False
        self._entered = False

        # Create record in repository in INITIALIZING state
        now = utc_now()
        self._created_at = now
        run_record = Run(
            id=self._run_id,
            experiment_id=self._experiment_id,
            name=self._name,
            status=RunStatus.INITIALIZING,
            start_time=None,
            created_at=now,
        )
        self._repo.create_run(run_record)

        if self._initial_tags:
            self._repo.set_tags("run", self._run_id, self._initial_tags)

    @property
    def run_id(self) -> str:
        """The unique identifier for this run."""
        return self._run_id

    @property
    def id(self) -> str:
        """Alias for run_id."""
        return self._run_id

    @property
    def experiment_id(self) -> str:
        """The parent experiment identifier."""
        return self._experiment_id

    @property
    def name(self) -> str | None:
        """Human-readable run name, if specified."""
        return self._name

    @property
    def status(self) -> RunStatus:
        """Current lifecycle status of the run."""
        run = self._repo.get_run(self._run_id)
        if run:
            self._status = run.status
        return self._status

    @property
    def start_time(self) -> datetime | None:
        """Start timestamp in UTC."""
        if self._start_time is None:
            run = self._repo.get_run(self._run_id)
            if run and run.start_time:
                self._start_time = run.start_time
        return self._start_time

    def start(self) -> "ActiveRun":
        """Explicitly transition run from INITIALIZING to RUNNING. Idempotent if already RUNNING."""
        current_status = self.status
        if current_status == RunStatus.INITIALIZING:
            now = utc_now()
            self._start_time = now
            self._repo.update_run_status(
                self._run_id,
                RunStatus.RUNNING,
                start_time=now,
            )
            self._status = RunStatus.RUNNING
        elif current_status.is_terminal:
            raise TerminalRunError(self._run_id, current_status.value, "start")
        return self

    # -------------------------------------------------------------------------
    # Context Manager Protocol
    # -------------------------------------------------------------------------

    def __enter__(self) -> "ActiveRun":
        """Enter run context, transitioning to RUNNING exactly once."""
        current_status = self.status
        if current_status.is_terminal:
            raise TerminalRunError(self._run_id, current_status.value, "__enter__")
        if self._entered:
            raise InvalidLifecycleTransitionError(
                current_status=current_status.value,
                target_status="RUNNING",
                run_id=self._run_id,
            )
        self._entered = True
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> bool:
        """Exit run context, marking FINISHED on success or FAILED on uncaught exception."""
        if exc_type is not None and exc_val is not None:
            # Exception occurred inside context block: record failure details
            tb_formatted = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self._record_failure(
                error_message=str(exc_val),
                error_type=exc_type.__name__,
                error_traceback=tb_formatted,
            )
            # Return False so the original exception is re-raised to the caller
            return False

        # Clean exit without uncaught exception
        self.finish()
        return False

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    def finish(self) -> None:
        """Mark the run as FINISHED. Idempotent if already in a terminal state."""
        current_status = self.status
        if self._closed or current_status.is_terminal:
            return

        if current_status == RunStatus.INITIALIZING:
            self.start()

        end = utc_now()
        start = self.start_time or end
        duration = max(0.0, (end - start).total_seconds())

        self._repo.update_run_status(
            self._run_id,
            RunStatus.FINISHED,
            end_time=end,
            duration_seconds=duration,
        )
        self._status = RunStatus.FINISHED
        self._closed = True

    def terminate(self, reason: str | None = None) -> None:
        """Explicitly terminate the run (marked TERMINATED). Idempotent if terminal."""
        current_status = self.status
        if self._closed or current_status.is_terminal:
            return

        end = utc_now()
        if current_status == RunStatus.INITIALIZING:
            duration = 0.0
        else:
            start = self.start_time or end
            duration = max(0.0, (end - start).total_seconds())

        self._repo.update_run_status(
            self._run_id,
            RunStatus.TERMINATED,
            end_time=end,
            duration_seconds=duration,
            error_message=reason,
        )
        self._status = RunStatus.TERMINATED
        self._closed = True

    def _record_failure(
        self,
        error_message: str,
        error_type: str,
        error_traceback: str,
    ) -> None:
        """Record bounded run failure details when an exception aborts the context block."""
        current_status = self.status
        if self._closed or current_status.is_terminal:
            return

        end = utc_now()
        if current_status == RunStatus.INITIALIZING:
            duration = 0.0
        else:
            start = self.start_time or end
            duration = max(0.0, (end - start).total_seconds())

        # Sanitize and bound error message length
        clean_msg = error_message[:MAX_ERROR_MSG_LENGTH]

        # Bounded traceback formatting
        tb_lines = error_traceback.splitlines(keepends=True)
        if len(tb_lines) > MAX_TRACEBACK_LINES:
            clean_tb = "".join(tb_lines[:10]) + "\n... [truncated] ...\n" + "".join(tb_lines[-30:])
        else:
            clean_tb = error_traceback

        marker = "\n... [truncated]"
        if len(clean_tb) > MAX_TRACEBACK_LENGTH:
            clean_tb = clean_tb[: MAX_TRACEBACK_LENGTH - len(marker)] + marker

        self._repo.update_run_status(
            self._run_id,
            RunStatus.FAILED,
            end_time=end,
            duration_seconds=max(0.0, duration),
            error_message=clean_msg,
            error_type=error_type,
            error_traceback=clean_tb,
        )
        self._status = RunStatus.FAILED
        self._closed = True

    # -------------------------------------------------------------------------
    # Logging Operations
    # -------------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple configuration parameters for this run."""
        self._ensure_active("log_params")
        self._repo.log_parameters(self._run_id, params)

    def log_param(self, name: str, value: Any) -> None:
        """Log a single configuration parameter."""
        self.log_params({name: value})

    def log_metrics(self, metrics: dict[str, int | float], step: int | None = None) -> None:
        """Log multiple scalar metrics at an optional training step."""
        self._ensure_active("log_metrics")
        now = utc_now()
        records = [
            Metric(
                run_id=self._run_id,
                name=name,
                value=float(val),
                step=step,
                timestamp=now,
            )
            for name, val in metrics.items()
        ]
        self._repo.log_metrics(self._run_id, records)

    def log_metric(self, name: str, value: int | float, step: int | None = None) -> None:
        """Log a single scalar metric."""
        self.log_metrics({name: value}, step=step)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Associate string key-value tags with this run."""
        self._ensure_active("set_tags")
        self._repo.set_tags("run", self._run_id, tags)

    def set_tag(self, key: str, value: str) -> None:
        """Associate a single string key-value tag with this run."""
        self.set_tags({key: value})

    # -------------------------------------------------------------------------
    # Read / Inspection Operations
    # -------------------------------------------------------------------------

    def get_params(self) -> dict[str, Any]:
        """Retrieve all configuration parameters logged for this run."""
        return self._repo.get_parameters(self._run_id)

    def get_metric_history(self, metric_name: str | None = None) -> list[Metric]:
        """Retrieve chronological metric records logged for this run."""
        return self._repo.get_metric_history(self._run_id, metric_name=metric_name)

    def get_tags(self) -> dict[str, str]:
        """Retrieve all tags associated with this run."""
        return self._repo.get_tags("run", self._run_id)

    def to_run(self) -> Run | None:
        """Return the immutable Run domain snapshot."""
        return self._repo.get_run(self._run_id)

    def _ensure_active(self, operation: str) -> None:
        current_status = self.status
        if current_status == RunStatus.INITIALIZING:
            self.start()
        elif self._closed or current_status.is_terminal:
            raise TerminalRunError(self._run_id, current_status.value, operation)


class ExperimentHandle:
    """An experiment handle allowing starting and inspecting runs."""

    def __init__(self, repository: SQLiteMetadataRepository, experiment: Experiment) -> None:
        self._repo = repository
        self._experiment = experiment

    @property
    def id(self) -> str:
        """The unique experiment identifier."""
        return self._experiment.id

    @property
    def name(self) -> str:
        """The experiment name."""
        return self._experiment.name

    @property
    def description(self) -> str | None:
        """Optional human-readable description."""
        return self._experiment.description

    @property
    def created_at(self) -> datetime:
        """Creation timestamp in UTC."""
        return self._experiment.created_at

    @property
    def updated_at(self) -> datetime:
        """Last updated timestamp in UTC."""
        return self._experiment.updated_at

    def start_run(
        self,
        name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> ActiveRun:
        """Start a new execution run belonging to this experiment."""
        return ActiveRun(
            repository=self._repo,
            experiment_id=self._experiment.id,
            name=name,
            initial_tags=tags,
        )

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a specific run by ID belonging to this experiment."""
        run = self._repo.get_run(run_id)
        if run and run.experiment_id == self._experiment.id:
            return run
        return None

    def list_runs(self) -> list[Run]:
        """List all runs belonging to this experiment."""
        return self._repo.list_runs(self._experiment.id)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on this experiment."""
        self._repo.set_tags("experiment", self._experiment.id, tags)

    def get_tags(self) -> dict[str, str]:
        """Get tags associated with this experiment."""
        return self._repo.get_tags("experiment", self._experiment.id)
