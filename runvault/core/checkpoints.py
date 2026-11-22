"""Checkpoint engine for atomic publication, integrity verification, and recovery."""

import logging
import os
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final

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
    StorageUnavailableError,
    UnsupportedCheckpointFormatError,
)
from runvault.core.storage.base import StorageProvider
from runvault.core.storage.local import (
    LocalStorageProvider,
    compute_file_sha256,
    fsync_directory,
    sync_file_descriptor,
)
from runvault.core.workspace import Workspace

# Backward-compatible aliases for internal durability helpers
_sync_file_descriptor = sync_file_descriptor
_fsync_directory = fsync_directory

__all__ = [
    "CheckpointRecoveryResult",
    "_fsync_directory",
    "_publish_file_no_overwrite",
    "_sync_file_descriptor",
    "compute_file_sha256",
    "fsync_directory",
    "get_latest_valid_checkpoint",
    "publish_checkpoint",
    "resolve_and_validate_artifact_path",
    "restore_checkpoint",
    "sync_file_descriptor",
    "verify_checkpoint",
]

logger = logging.getLogger("runvault.checkpoints")

CHUNK_SIZE: Final[int] = 65536  # 64 KB incremental hash reading buffer


@dataclass(frozen=True)
class CheckpointRecoveryResult(os.PathLike[str]):
    """Result of restoring or recovering a checkpoint artifact."""

    checkpoint: Checkpoint
    path: Path | None
    is_fallback: bool
    rejected_checkpoints: tuple[tuple[Checkpoint, str], ...] = ()
    _storage_provider: StorageProvider | None = None

    def __iter__(self) -> Iterator[Any]:
        yield self.checkpoint
        yield self.path
        yield self.is_fallback

    def __fspath__(self) -> str:
        if self.path is None:
            raise TypeError(
                f"Checkpoint artifact '{self.checkpoint.storage_key}' is stored on "
                f"provider '{self.checkpoint.storage_provider}' with no local path. "
                "Use open(), read_bytes(), or read_text() to access artifact content."
            )
        return str(self.path)

    def open(self) -> BinaryIO:
        """Open a readable binary stream for the restored checkpoint artifact."""
        if self._storage_provider is not None:
            return self._storage_provider.open_artifact(
                self.checkpoint.storage_key,
                expected_run_id=self.checkpoint.run_id,
            )
        if self.path is not None and self.path.exists():
            return self.path.open("rb")
        raise FileNotFoundError(
            f"No storage provider or local path available for checkpoint '{self.checkpoint.id}'"
        )

    def read_bytes(self, max_bytes: int | None = None) -> bytes:
        """Read artifact bytes directly from storage or local path with optional byte limit."""
        with self.open() as stream:
            if max_bytes is not None:
                return stream.read(max_bytes)
            return stream.read()

    def read_text(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """Convenience method reading artifact text directly from storage or local path."""
        return self.read_bytes().decode(encoding=encoding, errors=errors)


def _publish_file_no_overwrite(
    temp_path: Path,
    final_path: Path,
    run_id: str,
    name: str,
) -> str | None:
    """Atomically publish temp_path to final_path without overwriting an existing file.

    Preserved for backward compatibility.
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

    provider = LocalStorageProvider(workspace)
    return provider.get_artifact_path(checkpoint.artifact_path, expected_run_id=expected_run_id)


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
    storage: StorageProvider | None = None,
) -> Checkpoint:
    """Atomically create, write, verify, and publish a checkpoint artifact via storage provider."""
    if checkpoint_id is not None and (
        "/" in checkpoint_id or "\\" in checkpoint_id or ".." in checkpoint_id
    ):
        raise CheckpointCreationError(
            run_id=run_id,
            name=name,
            reason=(f"Checkpoint id '{checkpoint_id}' cannot contain path separators or traversal"),
        )

    ckpt_id = checkpoint_id or str(uuid.uuid4())
    provider = storage or LocalStorageProvider(workspace)

    staged = provider.stage_artifact(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        writer=writer,
        name=name,
    )

    published = provider.publish_artifact(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        staged=staged,
        name=name,
    )

    # Step 5: Verify the published artifact before metadata persistence
    try:
        is_valid, reason = provider.verify_artifact(
            artifact_key=published.artifact_key,
            expected_size=published.size_bytes,
            expected_checksum=published.checksum_sha256,
            expected_run_id=run_id,
        )
    except Exception as err:
        try:
            provider.delete_artifact(published.artifact_key, run_id=run_id)
        except Exception as cleanup_err:
            err.add_note(
                f"Failed to cleanup unverified artifact '{published.artifact_key}': {cleanup_err}"
            )
        creation_err = CheckpointCreationError(
            run_id=run_id,
            name=name,
            reason=f"Published artifact verification raised unexpected error: {err}",
            original_error=err,
        )
        for note in getattr(err, "__notes__", []):
            creation_err.add_note(note)
        raise creation_err from err

    if not is_valid:
        try:
            provider.delete_artifact(published.artifact_key, run_id=run_id)
        except Exception as cleanup_err:
            logger.warning("Failed to cleanup invalid published artifact: %s", cleanup_err)
        raise CheckpointCreationError(
            run_id=run_id,
            name=name,
            reason=f"Published artifact failed integrity verification: {reason}",
        )

    effective_metadata = dict(metadata or {})
    if published.staging_cleanup_warning is not None:
        effective_metadata["_staging_cleanup_warning"] = (
            f"Temporary staging link could not be removed: {published.staging_cleanup_warning}"
        )

    checkpoint = Checkpoint(
        id=ckpt_id,
        run_id=run_id,
        name=name,
        artifact_path=published.artifact_key,
        size_bytes=published.size_bytes,
        checksum_sha256=published.checksum_sha256,
        step=step,
        format_version=CURRENT_CHECKPOINT_FORMAT_VERSION,
        storage_provider=published.storage_provider,
        metadata=effective_metadata,
        created_at=utc_now(),
    )

    try:
        persisted_checkpoint = repository.create_checkpoint(checkpoint)
    except Exception as err:
        try:
            provider.delete_artifact(published.artifact_key, run_id=run_id)
        except Exception as cleanup_err:
            try:
                target_desc = str(
                    provider.get_artifact_path(published.artifact_key, expected_run_id=run_id)
                )
            except Exception:
                target_desc = published.artifact_key
            err.add_note(f"Failed to cleanup uncommitted artifact '{target_desc}': {cleanup_err}")

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


def verify_checkpoint(
    checkpoint: Checkpoint,
    workspace: Workspace,
    expected_run_id: str | None = None,
    storage: StorageProvider | None = None,
) -> tuple[bool, str | None]:
    """Verify that a checkpoint artifact exists, matches size and hash, and is contained safely."""
    if checkpoint.format_version > CURRENT_CHECKPOINT_FORMAT_VERSION:
        return (
            False,
            f"Unsupported format version {checkpoint.format_version} "
            f"(maximum supported: {CURRENT_CHECKPOINT_FORMAT_VERSION})",
        )

    if expected_run_id is not None and checkpoint.run_id != expected_run_id:
        return (
            False,
            f"Checkpoint run_id '{checkpoint.run_id}' does not match '{expected_run_id}'",
        )

    provider = storage or LocalStorageProvider(workspace)
    try:
        return provider.verify_artifact(
            artifact_key=checkpoint.storage_key,
            expected_size=checkpoint.size_bytes,
            expected_checksum=checkpoint.checksum_sha256,
            expected_run_id=expected_run_id,
        )
    except StorageUnavailableError:
        raise
    except PermissionError as err:
        raise StorageUnavailableError(
            identifier=checkpoint.storage_key,
            reason=f"Permission denied accessing artifact: {err}",
            original_error=err,
        ) from err
    except OSError as err:
        raise StorageUnavailableError(
            identifier=checkpoint.storage_key,
            reason=f"I/O error accessing storage: {err}",
            original_error=err,
        ) from err


def get_latest_valid_checkpoint(
    run_id: str,
    workspace: Workspace,
    repository: SQLiteMetadataRepository,
    storage: StorageProvider | None = None,
) -> CheckpointRecoveryResult:
    """Discover the newest valid checkpoint for a run, falling back to older ones if needed."""
    checkpoints = repository.list_checkpoints(run_id)
    if not checkpoints:
        raise CheckpointNotFoundError(
            "latest", run_id=run_id, reason=f"No checkpoints exist for run '{run_id}'"
        )

    provider = storage or LocalStorageProvider(workspace)
    sorted_ckpts = sorted(
        checkpoints,
        key=lambda c: (c.commit_seq or 0, c.created_at, c.id),
        reverse=True,
    )
    newest_id = sorted_ckpts[0].id
    rejected: list[tuple[Checkpoint, str]] = []

    for ckpt in sorted_ckpts:
        is_valid, reason = verify_checkpoint(
            checkpoint=ckpt,
            workspace=workspace,
            expected_run_id=run_id,
            storage=provider,
        )
        if is_valid:
            is_fallback = ckpt.id != newest_id
            resolved_path: Path | None = None
            if provider.capabilities.provides_local_path:
                try:
                    resolved_path = provider.get_artifact_path(
                        ckpt.storage_key,
                        expected_run_id=run_id,
                    )
                except Exception:
                    resolved_path = None

            return CheckpointRecoveryResult(
                checkpoint=ckpt,
                path=resolved_path,
                is_fallback=is_fallback,
                rejected_checkpoints=tuple(rejected),
                _storage_provider=provider,
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
    storage: StorageProvider | None = None,
) -> CheckpointRecoveryResult:
    """Restore a verified checkpoint artifact for training recovery."""
    provider = storage or LocalStorageProvider(workspace)

    if checkpoint_id is not None:
        ckpt = repository.get_checkpoint(checkpoint_id)
        if ckpt is None or ckpt.run_id != run_id:
            raise CheckpointNotFoundError(checkpoint_id, run_id=run_id)

        if ckpt.format_version > CURRENT_CHECKPOINT_FORMAT_VERSION:
            raise UnsupportedCheckpointFormatError(
                format_version=ckpt.format_version,
                max_supported=CURRENT_CHECKPOINT_FORMAT_VERSION,
            )

        is_valid, reason = verify_checkpoint(
            checkpoint=ckpt,
            workspace=workspace,
            expected_run_id=run_id,
            storage=provider,
        )
        if not is_valid:
            raise CheckpointCorruptError(
                checkpoint_id=checkpoint_id,
                reason=reason or "Failed integrity verification",
            )

        resolved_path: Path | None = None
        if provider.capabilities.provides_local_path:
            resolved_path = provider.get_artifact_path(
                ckpt.storage_key,
                expected_run_id=run_id,
            )

        return CheckpointRecoveryResult(
            checkpoint=ckpt,
            path=resolved_path,
            is_fallback=False,
            rejected_checkpoints=(),
            _storage_provider=provider,
        )

    return get_latest_valid_checkpoint(run_id, workspace, repository, storage=provider)
