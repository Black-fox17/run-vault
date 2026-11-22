"""RunVault - Lightweight ML experiment management, checkpointing, and execution platform."""

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from runvault.integrations.gdrive import GoogleDriveStorageProvider
    from runvault.integrations.torch import (
        TorchCheckpointManager,
        TorchRecoveryResult,
        restore_torch_checkpoint,
        save_torch_checkpoint,
    )

from runvault.api import (
    CheckpointRecoveryResult,
    LocalStorageProvider,
    PublishedArtifact,
    StagedArtifact,
    StorageCapabilities,
    StorageProvider,
    Workspace,
    create_experiment,
    get_checkpoint,
    get_experiment,
    get_experiment_by_name,
    get_latest_checkpoint,
    get_latest_valid_checkpoint,
    get_metric_history,
    get_params,
    get_run,
    get_tags,
    init,
    list_checkpoints,
    list_experiments,
    list_runs,
    reset_repository_cache,
    restore_checkpoint,
    save_checkpoint,
)
from runvault.core.domain import (
    CURRENT_CHECKPOINT_FORMAT_VERSION,
    Checkpoint,
    Experiment,
    Metric,
    Parameter,
    Run,
    RunStatus,
    Tag,
)
from runvault.core.errors import (
    CheckpointCorruptError,
    CheckpointCreationError,
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointRecoveryError,
    ConflictingParameterError,
    CorruptCheckpointError,
    ExperimentNotFoundError,
    InvalidLifecycleTransitionError,
    InvalidMetricError,
    InvalidParameterError,
    RunNotFoundError,
    RunVaultError,
    StorageUnavailableError,
    TerminalRunError,
    UnsupportedCheckpointFormatError,
    WorkspaceError,
)
from runvault.core.session import ActiveRun, ExperimentHandle

__version__: Final[str] = "0.1.0"

__all__ = [
    "CURRENT_CHECKPOINT_FORMAT_VERSION",
    "ActiveRun",
    "Checkpoint",
    "CheckpointCorruptError",
    "CheckpointCreationError",
    "CheckpointError",
    "CheckpointNotFoundError",
    "CheckpointRecoveryError",
    "CheckpointRecoveryResult",
    "ConflictingParameterError",
    "CorruptCheckpointError",
    "Experiment",
    "ExperimentHandle",
    "ExperimentNotFoundError",
    "GoogleDriveStorageProvider",
    "InvalidLifecycleTransitionError",
    "InvalidMetricError",
    "InvalidParameterError",
    "LocalStorageProvider",
    "Metric",
    "Parameter",
    "PublishedArtifact",
    "Run",
    "RunNotFoundError",
    "RunStatus",
    "RunVaultError",
    "StagedArtifact",
    "StorageCapabilities",
    "StorageProvider",
    "StorageUnavailableError",
    "Tag",
    "TerminalRunError",
    "TorchCheckpointManager",
    "TorchRecoveryResult",
    "UnsupportedCheckpointFormatError",
    "Workspace",
    "WorkspaceError",
    "__version__",
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
    "restore_torch_checkpoint",
    "save_checkpoint",
    "save_torch_checkpoint",
]


def __getattr__(name: str) -> Any:
    if name == "GoogleDriveStorageProvider":
        from runvault.integrations.gdrive import GoogleDriveStorageProvider

        return GoogleDriveStorageProvider
    if name in (
        "TorchCheckpointManager",
        "TorchRecoveryResult",
        "restore_torch_checkpoint",
        "save_torch_checkpoint",
    ):
        import runvault.integrations.torch as torch_int

        return getattr(torch_int, name)
    raise AttributeError(f"module 'runvault' has no attribute '{name}'")
