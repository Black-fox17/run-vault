"""Local workspace management and directory resolution for RunVault."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from runvault.core.errors import WorkspaceError

DEFAULT_WORKSPACE_DIRNAME: Final[str] = ".runvault"
DEFAULT_DB_FILENAME: Final[str] = "metadata.db"
DEFAULT_CHECKPOINTS_DIRNAME: Final[str] = "checkpoints"
ENV_WORKSPACE_DIR: Final[str] = "RUNVAULT_DIR"


def _discover_upward(start: Path) -> Path | None:
    """Walk up parent directories looking for an existing .runvault directory."""
    current = start.resolve()
    for directory in [current, *current.parents]:
        candidate = directory / DEFAULT_WORKSPACE_DIRNAME
        if candidate.is_dir():
            return candidate
    return None


def resolve_workspace_dir(explicit_dir: str | Path | None = None) -> Path:
    """Resolve the directory used for local RunVault metadata and logs.

    Precedence:
    1. Explicit path argument.
    2. RUNVAULT_DIR environment variable.
    3. Upward directory discovery from current working directory for `.runvault`.
    4. Default `.runvault` in the current working directory.
    """
    if explicit_dir is not None:
        target = Path(explicit_dir).expanduser().resolve()
        if target.is_file() or target.suffix == ".db":
            target = target.parent
    elif ENV_WORKSPACE_DIR in os.environ and os.environ[ENV_WORKSPACE_DIR].strip():
        target = Path(os.environ[ENV_WORKSPACE_DIR]).expanduser().resolve()
    else:
        discovered = _discover_upward(Path.cwd())
        if discovered is not None:
            target = discovered
        else:
            target = (Path.cwd() / DEFAULT_WORKSPACE_DIRNAME).resolve()

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise WorkspaceError(
            path=str(target),
            reason=f"Failed to create or access workspace directory: {err}",
            original_error=err,
        ) from err

    return target


def resolve_db_path(explicit_workspace: str | Path | None = None) -> Path:
    """Resolve the SQLite database file path inside the workspace."""
    workspace = resolve_workspace_dir(explicit_workspace)
    return workspace / DEFAULT_DB_FILENAME


@dataclass(frozen=True)
class Workspace:
    """Represents a resolved local RunVault workspace."""

    path: Path

    @property
    def db_path(self) -> Path:
        """The path to the workspace SQLite database file."""
        return self.path / DEFAULT_DB_FILENAME

    @property
    def checkpoints_dir(self) -> Path:
        """The base directory for storing checkpoint artifacts."""
        return self.path / DEFAULT_CHECKPOINTS_DIRNAME

    def get_run_checkpoints_dir(self, run_id: str) -> Path:
        """Return the checkpoint directory for a specific run, ensuring it exists."""
        run_dir = self.checkpoints_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @classmethod
    def from_path(cls, path: str | Path | None = None) -> "Workspace":
        """Construct a Workspace from an optional directory or database path."""
        return cls(path=resolve_workspace_dir(path))
