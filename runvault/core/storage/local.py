"""Local filesystem storage provider with atomic durability and zero-overwrite guarantees."""

import contextlib
import hashlib
import logging
import os
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Final

from runvault.core.errors import (
    CheckpointCorruptError,
    CheckpointCreationError,
    StorageUnavailableError,
)
from runvault.core.storage.base import (
    PublishedArtifact,
    StagedArtifact,
    StorageCapabilities,
    StorageProvider,
)
from runvault.core.workspace import Workspace

logger = logging.getLogger("runvault.storage.local")

CHUNK_SIZE: Final[int] = 65536  # 64 KB incremental hash reading buffer


def compute_file_sha256(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Compute the SHA-256 hex digest of a file incrementally in bounded chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def sync_file_descriptor(fd: int) -> None:
    """Synchronize file descriptor buffers and payload to persistent storage.

    Prefers `os.fdatasync` when available (e.g. Linux/POSIX) to flush data and essential
    sizing metadata without forcing unneeded metadata updates. Falls back to `os.fsync`
    on platforms where `fdatasync` is not exposed (such as standard Windows Python builds
    or macOS).
    If neither synchronization primitive is available or synchronization fails, an OSError
    is raised to prevent falsely claiming checkpoint durability when bytes are not flushed.
    """
    sync_fn = getattr(os, "fdatasync", None) or getattr(os, "fsync", None)
    if sync_fn is None:
        raise OSError("No portable file synchronization primitive available on this platform")
    sync_fn(fd)


def fsync_directory(dir_path: Path) -> None:
    """Flush directory entry updates where supported by the operating system.

    On POSIX systems, directory entries are synchronized by calling fsync on the directory fd.
    Any I/O error or failure raises an OSError.
    On non-POSIX systems (e.g. Windows), directory file descriptors cannot be synchronized
    via standard POSIX APIs; this platform limitation is explicitly handled.
    """
    if os.name != "posix":
        return
    dir_fd = os.open(str(dir_path), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


_sync_file_descriptor = sync_file_descriptor
_fsync_directory = fsync_directory


def _get_active_sync_fd() -> Callable[[int], None]:
    """Retrieve active file sync callable, supporting backward-compatible test hooks."""
    ckpt_mod = sys.modules.get("runvault.core.checkpoints")
    if ckpt_mod is not None:
        candidate: Any = getattr(ckpt_mod, "_sync_file_descriptor", None)  # type: ignore[reportExplicitAny]
        if candidate is not None and candidate is not _sync_file_descriptor:
            return candidate  # type: ignore[reportUnknownVariableType]
    return _sync_file_descriptor


def _get_active_fsync_dir() -> Callable[[Path], None]:
    """Retrieve active directory sync callable, supporting backward-compatible test hooks."""
    ckpt_mod = sys.modules.get("runvault.core.checkpoints")
    if ckpt_mod is not None:
        candidate: Any = getattr(ckpt_mod, "_fsync_directory", None)  # type: ignore[reportExplicitAny]
        if candidate is not None and candidate is not _fsync_directory:
            return candidate  # type: ignore[reportUnknownVariableType]
    return _fsync_directory


class LocalStorageProvider(StorageProvider):
    """Storage provider implementation for the local workspace filesystem."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._capabilities = StorageCapabilities(
            supports_exclusive_publication=True,
            supports_directory_sync=(os.name == "posix"),
            provides_local_path=True,
            can_verify_integrity=True,
        )

    @property
    def name(self) -> str:
        return "local"

    @property
    def capabilities(self) -> StorageCapabilities:
        return self._capabilities

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def _resolve_contained_path(
        self,
        artifact_key: str,
        expected_run_id: str | None = None,
    ) -> Path:
        """Resolve an artifact key and verify strict containment within checkpoints root."""
        if ".." in artifact_key:
            raise CheckpointCorruptError(
                checkpoint_id=artifact_key,
                reason=f"Path traversal detected in artifact key: '{artifact_key}'",
            )

        raw_path = self._workspace.path / artifact_key
        if raw_path.is_symlink():
            raise CheckpointCorruptError(
                checkpoint_id=artifact_key,
                reason=f"Artifact '{raw_path}' is a symlink; symlinks are forbidden for security",
            )

        checkpoints_root = self._workspace.checkpoints_dir.resolve()
        target_path = (self._workspace.path / artifact_key).resolve()

        try:
            target_path.relative_to(checkpoints_root)
        except ValueError:
            raise CheckpointCorruptError(
                checkpoint_id=artifact_key,
                reason=f"Artifact path '{artifact_key}' escapes checkpoints root",
            ) from None

        if expected_run_id is not None:
            expected_dir = self._workspace.get_run_checkpoints_dir(expected_run_id).resolve()
            try:
                target_path.relative_to(expected_dir)
            except ValueError:
                raise CheckpointCorruptError(
                    checkpoint_id=artifact_key,
                    reason=(
                        f"Artifact '{artifact_key}' is not contained within run '{expected_run_id}'"
                    ),
                ) from None

        return raw_path

    def stage_artifact(
        self,
        run_id: str,
        checkpoint_id: str,
        writer: Callable[[Path], None],
        name: str,
    ) -> StagedArtifact:
        """Populate and verify a unique staging file on the local filesystem."""
        target_dir = self._workspace.get_run_checkpoints_dir(run_id)
        temp_path = target_dir / f"{checkpoint_id}.{uuid.uuid4()}.tmp"

        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)

        try:
            writer(temp_path)
        except Exception as err:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_err:
                err.add_note(f"Failed to clean up staging file '{temp_path}': {cleanup_err}")
            creation_err = CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=f"Writer callback failed: {err}",
                original_error=err,
            )
            for note in getattr(err, "__notes__", []):
                creation_err.add_note(note)
            raise creation_err from err

        if not temp_path.exists():
            raise CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason="Writer callback finished without producing an artifact file",
            )

        if temp_path.is_symlink() or not temp_path.is_file():
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason="Produced artifact must be a regular file, not a symlink or directory",
            )

        size_bytes = temp_path.stat().st_size
        if size_bytes == 0:
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason="Produced artifact is empty (0 bytes); non-empty checkpoints are required",
            )

        # Flush file buffers to disk and synchronize payload
        try:
            with temp_path.open("a+b") as f:
                f.flush()
                _get_active_sync_fd()(f.fileno())
        except OSError as err:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_err:
                err.add_note(f"Failed to clean up staging file '{temp_path}': {cleanup_err}")
            creation_err = CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=f"Failed to flush artifact to disk: {err}",
                original_error=err,
            )
            for note in getattr(err, "__notes__", []):
                creation_err.add_note(note)
            raise creation_err from err

        try:
            checksum = compute_file_sha256(temp_path)
        except Exception as err:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_err:
                err.add_note(f"Failed to clean up staging file '{temp_path}': {cleanup_err}")
            creation_err = CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=f"Failed to compute checkpoint checksum: {err}",
                original_error=err,
            )
            for note in getattr(err, "__notes__", []):
                creation_err.add_note(note)
            raise creation_err from err

        return StagedArtifact(
            staging_path=temp_path,
            size_bytes=size_bytes,
            checksum_sha256=checksum,
            staging_ref=str(temp_path),
        )

    def publish_artifact(
        self,
        run_id: str,
        checkpoint_id: str,
        staged: StagedArtifact,
        name: str,
    ) -> PublishedArtifact:
        """Atomically link the staged artifact into its final immutable destination."""
        target_dir = self._workspace.get_run_checkpoints_dir(run_id)
        final_path = target_dir / f"{checkpoint_id}.bin"
        temp_path = staged.staging_path or Path(staged.staging_ref)

        if final_path.exists():
            with contextlib.suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=(
                    f"Checkpoint artifact '{final_path.name}' already exists "
                    "(immutability violated)"
                ),
            )

        try:
            os.link(temp_path, final_path)
        except FileExistsError as err:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_err:
                err.add_note(f"Failed to clean up staging file '{temp_path}': {cleanup_err}")
            creation_err = CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=(
                    f"Checkpoint artifact '{final_path.name}' already exists "
                    "(immutability violated)"
                ),
                original_error=err,
            )
            for note in getattr(err, "__notes__", []):
                creation_err.add_note(note)
            raise creation_err from err
        except OSError as err:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_err:
                err.add_note(f"Failed to clean up staging file '{temp_path}': {cleanup_err}")
            creation_err = CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=(
                    "Exclusive atomic publication failed (hard links unsupported or forbidden): "
                    f"{err}"
                ),
                original_error=err,
            )
            for note in getattr(err, "__notes__", []):
                creation_err.add_note(note)
            raise creation_err from err

        # Once linked, remove the temporary link name
        staging_cleanup_warning: str | None = None
        try:
            temp_path.unlink(missing_ok=True)
        except OSError as cleanup_err:
            logger.warning(
                "Failed to remove staging hard link '%s' after publication to '%s': %s",
                temp_path,
                final_path,
                cleanup_err,
            )
            staging_cleanup_warning = str(cleanup_err)

        # Synchronize containing directory entry on POSIX
        try:
            _get_active_fsync_dir()(target_dir)
        except Exception as err:
            with contextlib.suppress(Exception):
                self.delete_artifact(
                    str(final_path.relative_to(self._workspace.path)),
                    run_id=run_id,
                )
            creation_err = CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=f"Failed to synchronize checkpoint directory: {err}",
                original_error=err,
            )
            for note in getattr(err, "__notes__", []):
                creation_err.add_note(note)
            raise creation_err from err

        rel_key = str(final_path.relative_to(self._workspace.path))
        return PublishedArtifact(
            artifact_key=rel_key,
            size_bytes=staged.size_bytes,
            checksum_sha256=staged.checksum_sha256,
            storage_provider=self.name,
            staging_cleanup_warning=staging_cleanup_warning,
        )

    def exists(self, artifact_key: str, run_id: str | None = None) -> bool:
        """Check whether a local artifact file exists."""
        try:
            path = self._resolve_contained_path(artifact_key, expected_run_id=run_id)
            return path.is_file() and not path.is_symlink()
        except CheckpointCorruptError:
            return False

    def delete_artifact(self, artifact_key: str, run_id: str | None = None) -> None:
        """Remove a local checkpoint file and sync directory if possible."""
        try:
            path = self._resolve_contained_path(artifact_key, expected_run_id=run_id)
        except CheckpointCorruptError:
            path = self._workspace.path / artifact_key

        path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            _get_active_fsync_dir()(path.parent)

    def verify_artifact(
        self,
        artifact_key: str,
        expected_size: int,
        expected_checksum: str,
        expected_run_id: str | None = None,
    ) -> tuple[bool, str | None]:
        """Verify existence, safety, size, and SHA-256 hash of a local checkpoint artifact."""
        err_msg: str | None = None
        try:
            path = self._resolve_contained_path(artifact_key, expected_run_id=expected_run_id)
            if not path.exists():
                err_msg = f"Artifact file '{path}' does not exist"
            elif not path.is_file():
                err_msg = f"Artifact path '{path}' is not a regular file"
            elif path.stat().st_size != expected_size:
                actual_size = path.stat().st_size
                err_msg = (
                    f"Artifact size mismatch: expected {expected_size} bytes, "
                    f"got {actual_size} bytes"
                )
            elif compute_file_sha256(path) != expected_checksum:
                actual_checksum = compute_file_sha256(path)
                err_msg = (
                    f"Artifact SHA-256 mismatch: expected {expected_checksum}, "
                    f"got {actual_checksum}"
                )
        except CheckpointCorruptError as err:
            return False, str(err)
        except FileNotFoundError:
            return False, f"Artifact file '{artifact_key}' does not exist"
        except PermissionError as err:
            raise StorageUnavailableError(
                identifier=artifact_key,
                reason=f"Permission denied accessing artifact file: {err}",
                original_error=err,
            ) from err
        except OSError as err:
            raise StorageUnavailableError(
                identifier=artifact_key,
                reason=f"I/O error accessing artifact file: {err}",
                original_error=err,
            ) from err

        if err_msg is not None:
            return False, err_msg
        return True, None

    def get_artifact_path(self, artifact_key: str, expected_run_id: str | None = None) -> Path:
        """Return the resolved Path for a contained, regular artifact file."""
        return self._resolve_contained_path(artifact_key, expected_run_id=expected_run_id)

    def open_artifact(
        self,
        artifact_key: str,
        expected_run_id: str | None = None,
    ) -> BinaryIO:
        """Open a readable binary stream for the resolved local artifact path."""
        path = self.get_artifact_path(artifact_key, expected_run_id=expected_run_id)
        return path.open("rb")
