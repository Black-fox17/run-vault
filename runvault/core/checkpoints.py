"""Checkpoint engine for atomic publication, integrity verification, and recovery."""

import contextlib
import hashlib
import logging
import os
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from runvault.core.db.repository import SQLiteMetadataRepository
from runvault.core.domain import (
    CURRENT_CHECKPOINT_FORMAT_VERSION,
    Checkpoint,
    utc_now,
)
from runvault.core.errors import (
    CheckpointCorruptError,
    CheckpointCreationError,
    CheckpointNotFoundError,
    CheckpointRecoveryError,
    UnsupportedCheckpointFormatError,
)
from runvault.core.workspace import Workspace

logger = logging.getLogger("runvault.checkpoints")

CHUNK_SIZE: Final[int] = 65536  # 64 KB incremental hash reading buffer


@dataclass(frozen=True)
class CheckpointRecoveryResult(os.PathLike[str]):
    """Result of restoring or recovering a checkpoint artifact."""

    checkpoint: Checkpoint
    path: Path
    is_fallback: bool
    rejected_checkpoints: tuple[tuple[Checkpoint, str], ...] = ()

    def __iter__(self) -> Iterator[Any]:
        yield self.checkpoint
        yield self.path
        yield self.is_fallback

    def __fspath__(self) -> str:
        return str(self.path)

    def read_bytes(self) -> bytes:
        """Convenience method reading artifact bytes directly from the restored path."""
        return self.path.read_bytes()

    def read_text(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """Convenience method reading artifact text directly from the restored path."""
        return self.path.read_text(encoding=encoding, errors=errors)


def compute_file_sha256(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Compute the SHA-256 hex digest of a file incrementally in bounded chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sync_file_descriptor(fd: int) -> None:
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


def _fsync_directory(dir_path: Path) -> None:
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


def _publish_file_no_overwrite(
    temp_path: Path,
    final_path: Path,
    run_id: str,
    name: str,
) -> str | None:
    """Atomically publish temp_path to final_path without overwriting an existing file.

    Uses exclusive hard-link publication on platforms where hard links are supported.
    If final_path exists, link(2) fails atomically with FileExistsError without modifying it.
    Once linked, the temporary file is unlinked.
    If hard links are not supported or fail, publication fails cleanly without overwriting.
    Never falls back to Path.replace() or os.replace().

    Returns:
        None if staging link cleanup succeeded, or an error description string if removing
        the staging hard link failed after successful publication (non-fatal duplicate link).
    """
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
                f"Checkpoint artifact '{final_path.name}' already exists (immutability violated)"
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
                f"Exclusive atomic publication failed (hard links unsupported or forbidden): {err}"
            ),
            original_error=err,
        )
        for note in getattr(err, "__notes__", []):
            creation_err.add_note(note)
        raise creation_err from err

    # Once linked successfully, remove the temporary link name
    try:
        temp_path.unlink(missing_ok=True)
    except OSError as cleanup_err:
        logger.warning(
            "Failed to remove staging hard link '%s' after publication to '%s': %s",
            temp_path,
            final_path,
            cleanup_err,
        )
        return str(cleanup_err)

    return None


def resolve_and_validate_artifact_path(
    workspace: Workspace,
    checkpoint: Checkpoint,
    expected_run_id: str | None = None,
) -> Path:
    """Resolve and validate checkpoint artifact containment within workspace checkpoints dir."""
    if expected_run_id is not None and checkpoint.run_id != expected_run_id:
        raise CheckpointCorruptError(
            checkpoint_id=checkpoint.id,
            reason=(f"Checkpoint run_id '{checkpoint.run_id}' does not match '{expected_run_id}'"),
        )

    if ".." in checkpoint.artifact_path:
        raise CheckpointCorruptError(
            checkpoint_id=checkpoint.id,
            reason=f"Path traversal detected in artifact_path: '{checkpoint.artifact_path}'",
        )

    checkpoints_root = workspace.checkpoints_dir.resolve()
    target_path = (workspace.path / checkpoint.artifact_path).resolve()

    try:
        target_path.relative_to(checkpoints_root)
    except ValueError:
        raise CheckpointCorruptError(
            checkpoint_id=checkpoint.id,
            reason=(f"Artifact path '{checkpoint.artifact_path}' escapes checkpoints root"),
        ) from None

    raw_path = workspace.path / checkpoint.artifact_path
    if raw_path.is_symlink():
        raise CheckpointCorruptError(
            checkpoint_id=checkpoint.id,
            reason=f"Artifact '{raw_path}' is a symlink; symlinks are forbidden for security",
        )

    return raw_path


def _produce_and_validate_staging_file(
    temp_path: Path,
    writer: Callable[[Path], None],
    run_id: str,
    name: str,
) -> int:
    """Populate staging file via writer callback and verify physical file properties."""
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
            _sync_file_descriptor(f.fileno())
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

    return size_bytes


def _cleanup_published_artifact(final_path: Path, target_dir: Path, primary_err: Exception) -> None:
    """Remove published artifact when subsequent directory sync or metadata commit fails."""
    try:
        final_path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            _fsync_directory(target_dir)
    except OSError as cleanup_err:
        primary_err.add_note(
            f"Failed to cleanup uncommitted artifact '{final_path}': {cleanup_err}"
        )


def publish_checkpoint(
    workspace: Workspace,
    repository: SQLiteMetadataRepository,
    run_id: str,
    name: str,
    writer: Callable[[Path], None],
    *,
    step: int | None = None,
    metadata: dict[str, Any] | None = None,
    checkpoint_id: str | None = None,
) -> Checkpoint:
    """Atomically create, write, verify, and publish a checkpoint artifact.

    Persistence sequence:
    1. Validate identifier security and allocate unique checkpoint ID.
    2. Create a temporary artifact in the same directory as the final destination.
    3. Invoke the user writer callback to populate the temporary file.
    4. Verify that the file exists, is a regular file, and is non-empty.
    5. Flush and sync file buffers to disk via _sync_file_descriptor.
    6. Compute the SHA-256 checksum incrementally and determine file size.
    7. Atomically publish without overwriting any existing artifact.
    8. Commit the checkpoint metadata record in SQLite with synchronous=FULL.
    """
    if checkpoint_id is not None and (
        "/" in checkpoint_id or "\\" in checkpoint_id or ".." in checkpoint_id
    ):
        raise CheckpointCreationError(
            run_id=run_id,
            name=name,
            reason=(f"Checkpoint id '{checkpoint_id}' cannot contain path separators or traversal"),
        )

    ckpt_id = checkpoint_id or str(uuid.uuid4())
    target_dir = workspace.get_run_checkpoints_dir(run_id)
    final_path = target_dir / f"{ckpt_id}.bin"
    temp_path = target_dir / f"{ckpt_id}.{uuid.uuid4()}.tmp"

    if final_path.exists():
        raise CheckpointCreationError(
            run_id=run_id,
            name=name,
            reason=(
                f"Checkpoint artifact '{final_path.name}' already exists (immutability violated)"
            ),
        )

    size_bytes = _produce_and_validate_staging_file(temp_path, writer, run_id, name)
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

    # Atomically publish to final immutable destination without overwriting
    staging_cleanup_warning = _publish_file_no_overwrite(
        temp_path, final_path, run_id=run_id, name=name
    )

    # Synchronize containing directory entry
    try:
        _fsync_directory(target_dir)
    except Exception as err:
        _cleanup_published_artifact(final_path, target_dir, err)
        creation_err = CheckpointCreationError(
            run_id=run_id,
            name=name,
            reason=f"Failed to synchronize checkpoint directory: {err}",
            original_error=err,
        )
        for note in getattr(err, "__notes__", []):
            creation_err.add_note(note)
        raise creation_err from err

    # Relative artifact path within workspace
    rel_path = str(final_path.relative_to(workspace.path))

    effective_metadata = dict(metadata or {})
    if staging_cleanup_warning is not None:
        effective_metadata["_staging_cleanup_warning"] = (
            f"Temporary staging link could not be removed: {staging_cleanup_warning}"
        )

    checkpoint = Checkpoint(
        id=ckpt_id,
        run_id=run_id,
        name=name,
        artifact_path=rel_path,
        size_bytes=size_bytes,
        checksum_sha256=checksum,
        step=step,
        format_version=CURRENT_CHECKPOINT_FORMAT_VERSION,
        metadata=effective_metadata,
        created_at=utc_now(),
    )

    try:
        persisted_checkpoint = repository.create_checkpoint(checkpoint)
    except Exception as err:
        _cleanup_published_artifact(final_path, target_dir, err)
        creation_err = CheckpointCreationError(
            run_id=run_id,
            name=name,
            reason=f"Failed to persist checkpoint metadata in SQLite: {err}",
            original_error=err,
        )
        for note in getattr(err, "__notes__", []):
            creation_err.add_note(note)
        raise creation_err from err

    return persisted_checkpoint


def _check_file_integrity(
    path: Path,
    expected_size: int,
    expected_checksum: str,
) -> tuple[bool, str | None]:
    """Inspect physical file existence, size, and incremental checksum."""
    if not path.exists():
        return False, f"Artifact file '{path}' does not exist"

    if path.is_symlink():
        return False, f"Artifact '{path}' is a symlink; symlinks are forbidden"

    if not path.is_file():
        return False, f"Artifact path '{path}' is not a regular file"

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return (
            False,
            f"Artifact size mismatch: expected {expected_size} bytes, got {actual_size} bytes",
        )

    actual_checksum = compute_file_sha256(path)
    if actual_checksum != expected_checksum:
        return (
            False,
            f"Artifact SHA-256 mismatch: expected {expected_checksum}, got {actual_checksum}",
        )

    return True, None


def verify_checkpoint(
    checkpoint: Checkpoint,
    workspace: Workspace,
    expected_run_id: str | None = None,
) -> tuple[bool, str | None]:
    """Verify that a checkpoint artifact exists, matches size and hash, and is contained safely."""
    if checkpoint.format_version > CURRENT_CHECKPOINT_FORMAT_VERSION:
        return (
            False,
            f"Unsupported format version {checkpoint.format_version} "
            f"(maximum supported: {CURRENT_CHECKPOINT_FORMAT_VERSION})",
        )

    try:
        artifact_path = resolve_and_validate_artifact_path(
            workspace=workspace,
            checkpoint=checkpoint,
            expected_run_id=expected_run_id,
        )
    except (CheckpointCorruptError, ValueError) as err:
        return False, str(err)

    return _check_file_integrity(artifact_path, checkpoint.size_bytes, checkpoint.checksum_sha256)


def get_latest_valid_checkpoint(
    run_id: str,
    workspace: Workspace,
    repository: SQLiteMetadataRepository,
) -> CheckpointRecoveryResult:
    """Discover the newest valid checkpoint for a run, falling back to older ones if needed.

    Returns:
        CheckpointRecoveryResult containing (checkpoint, path, is_fallback, rejected_checkpoints).

    Raises:
        CheckpointNotFoundError: If no checkpoints exist for the run.
        CheckpointRecoveryError: If checkpoints exist but none pass verification.
    """
    checkpoints = repository.list_checkpoints(run_id)
    if not checkpoints:
        raise CheckpointNotFoundError(
            "latest", run_id=run_id, reason=f"No checkpoints exist for run '{run_id}'"
        )

    # Sort descending by commit_seq (most recently committed first)
    sorted_ckpts = sorted(
        checkpoints,
        key=lambda c: (c.commit_seq or 0, c.created_at, c.id),
        reverse=True,
    )
    newest_id = sorted_ckpts[0].id
    rejected: list[tuple[Checkpoint, str]] = []

    for ckpt in sorted_ckpts:
        is_valid, reason = verify_checkpoint(ckpt, workspace, expected_run_id=run_id)
        if is_valid:
            is_fallback = ckpt.id != newest_id
            target_path = workspace.path / ckpt.artifact_path
            return CheckpointRecoveryResult(
                checkpoint=ckpt,
                path=target_path,
                is_fallback=is_fallback,
                rejected_checkpoints=tuple(rejected),
            )
        rejected.append((ckpt, reason or "Failed integrity verification"))

    rejection_details = "; ".join(f"{c.name} ({c.id}): {r}" for c, r in rejected)
    raise CheckpointRecoveryError(
        run_id=run_id,
        reason=(
            f"Found {len(sorted_ckpts)} checkpoint(s), but none passed integrity verification: "
            f"{rejection_details}"
        ),
    )


def restore_checkpoint(
    run_id: str,
    workspace: Workspace,
    repository: SQLiteMetadataRepository,
    checkpoint_id: str | None = None,
) -> CheckpointRecoveryResult:
    """Restore a verified checkpoint artifact for training recovery.

    Args:
        run_id: The run identifier.
        workspace: The workspace environment.
        repository: The SQLite metadata repository.
        checkpoint_id: Specific checkpoint ID, or None for the latest valid checkpoint.

    Returns:
        CheckpointRecoveryResult containing (checkpoint, path, is_fallback, rejected_checkpoints).

    Raises:
        CheckpointNotFoundError: If requested checkpoint ID does not exist, or run has no
            checkpoints.
        CheckpointCorruptError: If the requested explicit checkpoint fails verification.
        UnsupportedCheckpointFormatError: If requested checkpoint has unsupported format.
        CheckpointRecoveryError: If latest checkpoint recovery finds no valid checkpoints.
    """
    if checkpoint_id is not None:
        ckpt = repository.get_checkpoint(checkpoint_id)
        if ckpt is None or ckpt.run_id != run_id:
            raise CheckpointNotFoundError(checkpoint_id, run_id=run_id)

        if ckpt.format_version > CURRENT_CHECKPOINT_FORMAT_VERSION:
            raise UnsupportedCheckpointFormatError(
                format_version=ckpt.format_version,
                max_supported=CURRENT_CHECKPOINT_FORMAT_VERSION,
            )

        resolved_path = resolve_and_validate_artifact_path(
            workspace=workspace,
            checkpoint=ckpt,
            expected_run_id=run_id,
        )

        is_valid, reason = verify_checkpoint(ckpt, workspace, expected_run_id=run_id)
        if not is_valid:
            raise CheckpointCorruptError(
                checkpoint_id=checkpoint_id,
                reason=reason or "Failed integrity verification",
            )
        return CheckpointRecoveryResult(
            checkpoint=ckpt,
            path=resolved_path,
            is_fallback=False,
            rejected_checkpoints=(),
        )

    return get_latest_valid_checkpoint(run_id, workspace, repository)
