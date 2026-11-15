"""RunVault - Lightweight ML experiment management, checkpointing, and execution platform."""

from typing import Final

from runvault.api import (
    Workspace,
    create_experiment,
    get_experiment,
    get_experiment_by_name,
    get_metric_history,
    get_params,
    get_run,
    get_tags,
    init,
    list_experiments,
    list_runs,
    reset_repository_cache,
)
from runvault.core.domain import (
    Experiment,
    Metric,
    Parameter,
    Run,
    RunStatus,
    Tag,
)
from runvault.core.errors import (
    ConflictingParameterError,
    ExperimentNotFoundError,
    InvalidLifecycleTransitionError,
    InvalidMetricError,
    InvalidParameterError,
    RunNotFoundError,
    RunVaultError,
    TerminalRunError,
    WorkspaceError,
)
from runvault.core.session import ActiveRun, ExperimentHandle

__version__: Final[str] = "0.1.0"

__all__ = [
    "ActiveRun",
    "ConflictingParameterError",
    "Experiment",
    "ExperimentHandle",
    "ExperimentNotFoundError",
    "InvalidLifecycleTransitionError",
    "InvalidMetricError",
    "InvalidParameterError",
    "Metric",
    "Parameter",
    "Run",
    "RunNotFoundError",
    "RunStatus",
    "RunVaultError",
    "Tag",
    "TerminalRunError",
    "Workspace",
    "WorkspaceError",
    "__version__",
    "create_experiment",
    "get_experiment",
    "get_experiment_by_name",
    "get_metric_history",
    "get_params",
    "get_run",
    "get_tags",
    "init",
    "list_experiments",
    "list_runs",
    "reset_repository_cache",
]
