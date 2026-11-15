"""Public top-level API implementation for RunVault SDK."""

import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from runvault.core.checkpoints import (
    CheckpointRecoveryResult,
    publish_checkpoint,
)
from runvault.core.checkpoints import (
    get_latest_valid_checkpoint as _engine_latest_valid_checkpoint,
)
from runvault.core.checkpoints import (
    restore_checkpoint as _engine_restore_checkpoint,
)
from runvault.core.db.repository import SQLiteMetadataRepository
from runvault.core.domain import Checkpoint, Experiment, Metric, Run, utc_now
from runvault.core.errors import (
    CheckpointCorruptError,
    CheckpointCreationError,
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointRecoveryError,
    CorruptCheckpointError,
    UnsupportedCheckpointFormatError,
)
from runvault.core.session import ActiveRun, ExperimentHandle
from runvault.core.workspace import Workspace, resolve_db_path

__all__ = [
    "CheckpointCorruptError",
    "CheckpointCreationError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointRecoveryError",
    "CheckpointRecoveryResult",
    "CorruptCheckpointError",
    "UnsupportedCheckpointFormatError",
    "Workspace",
    "create_experiment",
    "get_checkpoint",
    "get_experiment",
    "get_experiment_by_name",
    "get_latest_checkpoint",
    "get_latest_valid_checkpoint",
    "get_metric_history",
    "get_params",
    "get_run",
    "get_tags",
    "init",
    "list_checkpoints",
    "list_experiments",
    "list_runs",
    "reset_repository_cache",
    "restore_checkpoint",
    "save_checkpoint",
]

# Process-level repository cache protected by a re-entrant lock
_REPO_CACHE: dict[Path, SQLiteMetadataRepository] = {}
_CACHE_LOCK = threading.RLock()


def _get_repository(workspace: str | Path | None = None) -> SQLiteMetadataRepository:
    """Obtain or initialize a SQLiteMetadataRepository for the resolved workspace."""
    db_path = resolve_db_path(workspace)
    with _CACHE_LOCK:
        repo = _REPO_CACHE.get(db_path)
        if repo is None or repo.is_closed:
            repo = SQLiteMetadataRepository(db_path)
            _REPO_CACHE[db_path] = repo
        return repo


def reset_repository_cache() -> None:
    """Close open connections in the calling thread and clear the repository cache.

    Contract:
    Calling this marks all cached repositories as closed and immediately closes
    connections owned by the calling thread. In-flight worker threads will receive
    WorkspaceError on subsequent repository access.
    """
    with _CACHE_LOCK:
        for repo in _REPO_CACHE.values():
            repo.close()
        _REPO_CACHE.clear()


def init(
    experiment: str = "default",
    *,
    name: str | None = None,
    tags: dict[str, str] | None = None,
    description: str | None = None,
    workspace: str | Path | None = None,
) -> ActiveRun:
    """Initialize a RunVault execution run under an experiment.

    If the experiment does not exist, it will be automatically created.
    Returns an ActiveRun instance that can be used directly or as a context manager.
    """
    with _CACHE_LOCK:
        repo = _get_repository(workspace)
        exp = repo.get_experiment_by_name(experiment)
        if exp is None:
            now = utc_now()
            exp_id = str(uuid.uuid4())
            exp = Experiment(
                id=exp_id,
                name=experiment,
                description=description,
                created_at=now,
                updated_at=now,
            )
            repo.create_experiment(exp)

    return ActiveRun(
        repository=repo,
        experiment_id=exp.id,
        name=name,
        initial_tags=tags,
    )


def create_experiment(
    name: str,
    description: str | None = None,
    *,
    workspace: str | Path | None = None,
) -> ExperimentHandle:
    """Create and persist a new experiment.

    Args:
        name: Human-readable name for the experiment.
        description: Optional detailed description.
        workspace: Optional path to override the local workspace directory.

    Returns:
        ExperimentHandle instance for starting runs and viewing history.
    """
    repo = _get_repository(workspace)
    now = utc_now()
    exp_id = str(uuid.uuid4())
    experiment = Experiment(
        id=exp_id,
        name=name,
        description=description,
        created_at=now,
        updated_at=now,
    )
    repo.create_experiment(experiment)
    return ExperimentHandle(repo, experiment)


def get_experiment(
    experiment_id: str,
    *,
    workspace: str | Path | None = None,
) -> ExperimentHandle | None:
    """Retrieve an existing experiment by its unique identifier."""
    repo = _get_repository(workspace)
    exp = repo.get_experiment(experiment_id)
    if exp is None:
        return None
    return ExperimentHandle(repo, exp)


def get_experiment_by_name(
    name: str,
    *,
    workspace: str | Path | None = None,
) -> ExperimentHandle | None:
    """Retrieve the most recently created experiment matching the given name."""
    repo = _get_repository(workspace)
    exp = repo.get_experiment_by_name(name)
    if exp is None:
        return None
    return ExperimentHandle(repo, exp)


def list_experiments(
    *,
    workspace: str | Path | None = None,
) -> list[ExperimentHandle]:
    """List all experiments in the workspace, ordered by creation time descending."""
    repo = _get_repository(workspace)
    exps = repo.list_experiments()
    return [ExperimentHandle(repo, exp) for exp in exps]


def get_run(
    run_id: str,
    *,
    workspace: str | Path | None = None,
) -> Run | None:
    """Retrieve a run record by its unique identifier."""
    repo = _get_repository(workspace)
    return repo.get_run(run_id)


def list_runs(
    experiment_id: str,
    *,
    workspace: str | Path | None = None,
) -> list[Run]:
    """List all runs belonging to an experiment, ordered by creation time descending."""
    repo = _get_repository(workspace)
    return repo.list_runs(experiment_id)


def get_metric_history(
    run_id: str,
    metric_name: str | None = None,
    *,
    workspace: str | Path | None = None,
) -> list[Metric]:
    """Retrieve metric history for a run, optionally filtered by metric name."""
    repo = _get_repository(workspace)
    return repo.get_metric_history(run_id, metric_name=metric_name)


def get_params(
    run_id: str,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Retrieve configuration parameters logged for a run."""
    repo = _get_repository(workspace)
    return repo.get_parameters(run_id)


def get_tags(
    entity_type: str,
    entity_id: str,
    *,
    workspace: str | Path | None = None,
) -> dict[str, str]:
    """Retrieve tags associated with an experiment or run."""
    repo = _get_repository(workspace)
    return repo.get_tags(entity_type, entity_id)


def save_checkpoint(
    run_id: str,
    name: str,
    writer: Callable[[Path], None],
    *,
    step: int | None = None,
    metadata: dict[str, Any] | None = None,
    checkpoint_id: str | None = None,
    workspace: str | Path | None = None,
) -> Checkpoint:
    """Atomically create, write, verify, and publish a checkpoint for a run."""
    repo = _get_repository(workspace)
    ws = Workspace.from_path(workspace)
    return publish_checkpoint(
        workspace=ws,
        repository=repo,
        run_id=run_id,
        name=name,
        writer=writer,
        step=step,
        metadata=metadata,
        checkpoint_id=checkpoint_id,
    )


def get_checkpoint(
    checkpoint_id: str,
    *,
    workspace: str | Path | None = None,
) -> Checkpoint | None:
    """Retrieve checkpoint metadata by ID."""
    repo = _get_repository(workspace)
    return repo.get_checkpoint(checkpoint_id)


def list_checkpoints(
    run_id: str,
    *,
    workspace: str | Path | None = None,
) -> list[Checkpoint]:
    """List all checkpoints recorded for a run in deterministic order."""
    repo = _get_repository(workspace)
    return repo.list_checkpoints(run_id)


def get_latest_checkpoint(
    run_id: str,
    *,
    workspace: str | Path | None = None,
) -> Checkpoint | None:
    """Retrieve the newest committed checkpoint metadata for a run."""
    repo = _get_repository(workspace)
    return repo.get_latest_checkpoint(run_id)


def get_latest_valid_checkpoint(
    run_id: str,
    *,
    workspace: str | Path | None = None,
) -> CheckpointRecoveryResult | None:
    """Find the newest verified checkpoint for a run, falling back if needed."""
    repo = _get_repository(workspace)
    ws = Workspace.from_path(workspace)
    try:
        return _engine_latest_valid_checkpoint(run_id, ws, repo)
    except CheckpointNotFoundError:
        return None


def restore_checkpoint(
    run_id: str,
    checkpoint_id: str | None = None,
    *,
    workspace: str | Path | None = None,
) -> CheckpointRecoveryResult:
    """Verify and restore a checkpoint artifact for recovery.

    Returns:
        CheckpointRecoveryResult containing (checkpoint, path, is_fallback, rejected).
        Implements os.PathLike[str], unpackable as (ckpt, path, is_fallback), with
        direct .read_bytes() and .read_text() methods.
    """
    repo = _get_repository(workspace)
    ws = Workspace.from_path(workspace)
    return _engine_restore_checkpoint(run_id, ws, repo, checkpoint_id)
