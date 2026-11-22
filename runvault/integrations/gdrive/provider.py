"""Google Drive storage provider implementing atomic, verified checkpoint persistence."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import http.client
import logging
import random
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Final, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from runvault.core.workspace import Workspace

from runvault.core.errors import (
    CheckpointCreationError,
    StorageUnavailableError,
)
from runvault.core.storage.base import (
    PublishedArtifact,
    StagedArtifact,
    StorageCapabilities,
    StorageProvider,
)
from runvault.core.storage.local import compute_file_sha256, sync_file_descriptor
from runvault.integrations.gdrive.auth import (
    GDRIVE_DEFAULT_SCOPES,
    check_gdrive_dependencies,
    get_credentials,
)
from runvault.integrations.gdrive.stream import GoogleDriveDownloadStream

logger = logging.getLogger("runvault.integrations.gdrive")

T = TypeVar("T")

HTTP_NOT_FOUND: Final[int] = 404
RETRYABLE_HTTP_STATUSES: tuple[int, ...] = (429, 500, 502, 503, 504)
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    socket.timeout,
    TimeoutError,
    ConnectionResetError,
    http.client.RemoteDisconnected,
)


class GoogleDriveStorageProvider(StorageProvider):
    """Storage provider persisting checkpoint artifacts directly to the user's Google Drive.

    Operates without proxying binaries through metadata servers. Follows the direct-to-storage
    data plane invariant, streaming checkpoint payloads to and from Google Drive API v3.

    Note on Shared Drives:
        This provider operates exclusively against the user's personal Google Drive ("My Drive")
        using the least-privilege `drive.file` scope. Google Shared Drives (Team Drives) are
        currently not supported.
    """

    def __init__(
        self,
        credentials: Any | None = None,
        drive_service: Any | None = None,
        *,
        root_folder_name: str = "RunVault",
        root_folder_id: str | None = None,
        chunk_size: int = 1024 * 1024 * 5,  # 5 MB resumable upload chunk size
        max_retries: int = 3,
        workspace: Workspace | None = None,
        client_secrets_file: str | Path | None = None,
        token_cache_path: str | Path | None = None,
    ) -> None:
        self._root_folder_name = root_folder_name
        self._root_folder_id = root_folder_id
        self._chunk_size = chunk_size
        self._max_retries = max(0, max_retries)
        self._workspace = workspace
        self._folder_cache: dict[str, str] = {}
        self._capabilities = StorageCapabilities(
            provides_local_path=False,
            supports_exclusive_publication=True,
            supports_directory_sync=False,
            can_verify_integrity=True,
        )

        if drive_service is not None:
            self._service = drive_service
        else:
            check_gdrive_dependencies()
            import googleapiclient.discovery  # type: ignore[reportMissingImports]

            creds = get_credentials(
                credentials=credentials,
                client_secrets_file=client_secrets_file,
                token_cache_path=token_cache_path,
                scopes=GDRIVE_DEFAULT_SCOPES,
            )
            self._service = googleapiclient.discovery.build(
                "drive",
                "v3",
                credentials=creds,
                cache_discovery=False,
            )

    @property
    def name(self) -> str:
        return "gdrive"

    @property
    def capabilities(self) -> StorageCapabilities:
        return self._capabilities

    @property
    def service(self) -> Any:
        return self._service

    # -------------------------------------------------------------------------
    # Retry & Error Translation Helpers
    # -------------------------------------------------------------------------

    def _execute_with_retry(self, action: Callable[[], T], operation_name: str) -> T:
        """Execute a Google Drive API call with exponential backoff and jitter."""
        attempt = 0
        while True:
            try:
                return action()
            except Exception as err:
                status_code: int | None = getattr(getattr(err, "resp", None), "status", None)
                if status_code in (401, 403):
                    raise StorageUnavailableError(
                        identifier="gdrive",
                        reason=f"Google Drive permission denied or authorization failed: {err}",
                        original_error=err,
                    ) from err

                is_retryable = (
                    status_code in RETRYABLE_HTTP_STATUSES
                    or isinstance(err, RETRYABLE_EXCEPTIONS)
                    or (
                        isinstance(err, OSError)
                        and err.errno in (errno.ETIMEDOUT, errno.ECONNRESET, errno.ENETUNREACH)
                    )
                )

                if not is_retryable or attempt >= self._max_retries:
                    if is_retryable or status_code in (500, 502, 503, 504):
                        raise StorageUnavailableError(
                            identifier="gdrive",
                            reason=(
                                f"Google Drive operation '{operation_name}' failed after "
                                f"{attempt} retries: {err}"
                            ),
                            original_error=err,
                        ) from err
                    raise

                attempt += 1
                backoff_delay = min(5.0, 0.25 * (2**attempt) + random.uniform(0.0, 0.2))
                logger.warning(
                    "Retrying Google Drive operation '%s' (attempt %d/%d) after %.2fs: %s",
                    operation_name,
                    attempt,
                    self._max_retries,
                    backoff_delay,
                    err,
                )
                time.sleep(backoff_delay)

    # -------------------------------------------------------------------------
    # Drive Hierarchy Management
    # -------------------------------------------------------------------------

    def invalidate_folder_cache(self, key: str | None = None) -> None:
        """Invalidate cached folder IDs when folders are modified, deleted, or trashed."""
        if key is None:
            self._folder_cache.clear()
        else:
            self._folder_cache.pop(key, None)

    def get_or_create_root_folder(self) -> str:
        """Resolve or create the root RunVault folder in Google Drive."""
        if self._root_folder_id:
            return self._root_folder_id

        if "_root_" in self._folder_cache:
            cached_id = self._folder_cache["_root_"]
            try:
                meta = self._service.files().get(fileId=cached_id, fields="id, trashed").execute()
                if not meta.get("trashed", False):
                    return cached_id
            except Exception:
                pass
            self.invalidate_folder_cache("_root_")

        def query_root() -> dict[str, Any]:
            q = (
                f"mimeType = 'application/vnd.google-apps.folder' and "
                f"name = '{self._root_folder_name}' and trashed = false and 'root' in parents"
            )
            return self._service.files().list(q=q, fields="files(id, name)").execute()  # type: ignore[no-any-return]

        result = self._execute_with_retry(query_root, "find_root_folder")
        files = result.get("files", [])
        if files:
            folder_id = str(files[0]["id"])
            self._folder_cache["_root_"] = folder_id
            return folder_id

        def create_root() -> dict[str, Any]:
            body = {
                "name": self._root_folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": ["root"],
            }
            return self._service.files().create(body=body, fields="id").execute()  # type: ignore[no-any-return]

        new_folder = self._execute_with_retry(create_root, "create_root_folder")
        folder_id = str(new_folder["id"])
        self._folder_cache["_root_"] = folder_id
        return folder_id

    def get_or_create_run_folder(self, run_id: str) -> str:
        """Resolve or create the run-specific subfolder in Google Drive."""
        root_id = self.get_or_create_root_folder()
        if run_id in self._folder_cache:
            cached_id = self._folder_cache[run_id]
            try:
                meta = (
                    self._service.files()
                    .get(fileId=cached_id, fields="id, trashed, parents")
                    .execute()
                )
                if not meta.get("trashed", False) and root_id in meta.get("parents", []):
                    return cached_id
            except Exception:
                pass
            self.invalidate_folder_cache(run_id)

        def query_run_folder() -> dict[str, Any]:
            q = (
                f"mimeType = 'application/vnd.google-apps.folder' and "
                f"name = '{run_id}' and trashed = false and '{root_id}' in parents"
            )
            return self._service.files().list(q=q, fields="files(id, name)").execute()  # type: ignore[no-any-return]

        result = self._execute_with_retry(query_run_folder, f"find_run_folder_{run_id}")
        files = result.get("files", [])
        if files:
            folder_id = str(files[0]["id"])
            self._folder_cache[run_id] = folder_id
            return folder_id

        def create_run_folder() -> dict[str, Any]:
            body = {
                "name": run_id,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [root_id],
            }
            return self._service.files().create(body=body, fields="id").execute()  # type: ignore[no-any-return]

        new_folder = self._execute_with_retry(create_run_folder, f"create_run_folder_{run_id}")
        folder_id = str(new_folder["id"])
        self._folder_cache[run_id] = folder_id
        return folder_id

    _get_or_create_root_folder = get_or_create_root_folder
    _get_or_create_run_folder = get_or_create_run_folder

    # -------------------------------------------------------------------------
    # StorageProvider Implementation
    # -------------------------------------------------------------------------

    def stage_artifact(
        self,
        run_id: str,
        checkpoint_id: str,
        writer: Callable[[Path], None],
        name: str,
    ) -> StagedArtifact:
        """Stage artifact payload locally prior to remote resumable upload.

        The local staging path is strictly encapsulated inside the provider and is never leaked
        into the persistent remote artifact key.
        """
        if self._workspace is not None:
            target_dir = self._workspace.get_run_checkpoints_dir(run_id)
        else:
            target_dir = Path(tempfile.gettempdir()) / "runvault" / run_id

        target_dir.mkdir(parents=True, exist_ok=True)
        temp_path = target_dir / f"{checkpoint_id}.{uuid.uuid4()}.tmp"

        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)

        try:
            writer(temp_path)
        except Exception as err:
            temp_path.unlink(missing_ok=True)
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
            temp_path.unlink(missing_ok=True)
            raise CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason="Produced artifact must be a regular file, not a symlink or directory",
            )

        size_bytes = temp_path.stat().st_size
        if size_bytes == 0:
            temp_path.unlink(missing_ok=True)
            raise CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason="Produced artifact is empty (0 bytes); non-empty checkpoints are required",
            )

        # Flush file buffers to disk and sync
        try:
            with temp_path.open("a+b") as f:
                f.flush()
                sync_file_descriptor(f.fileno())
        except OSError as err:
            temp_path.unlink(missing_ok=True)
            raise CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=f"Failed to flush artifact to disk: {err}",
                original_error=err,
            ) from err

        try:
            checksum = compute_file_sha256(temp_path)
        except Exception as err:
            temp_path.unlink(missing_ok=True)
            raise CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=f"Failed to compute checkpoint checksum: {err}",
                original_error=err,
            ) from err

        return StagedArtifact(
            staging_path=temp_path,
            size_bytes=size_bytes,
            checksum_sha256=checksum,
            staging_ref=str(temp_path),
        )

    def _check_preflight_collision(
        self,
        run_folder_id: str,
        checkpoint_id: str,
        *,
        run_id: str,
        name: str,
        temp_path: Path,
    ) -> None:
        artifact_filename = f"{checkpoint_id}.bin"

        def check_collision() -> dict[str, Any]:
            q = (
                f"'{run_folder_id}' in parents and trashed = false and "
                f"(name = '{artifact_filename}' or "
                f"appProperties has {{ key='runvault_checkpoint_id' and value='{checkpoint_id}' }})"
            )
            return (  # type: ignore[no-any-return]
                self._service.files().list(q=q, fields="files(id, size, appProperties)").execute()
            )

        collision_result = self._execute_with_retry(
            check_collision, f"check_collision_{checkpoint_id}"
        )
        if collision_result.get("files"):
            temp_path.unlink(missing_ok=True)
            raise CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=(
                    f"Checkpoint artifact '{artifact_filename}' already exists in Google Drive "
                    "(immutability violated)"
                ),
            )

    def _recover_lost_upload(
        self,
        run_folder_id: str,
        checkpoint_id: str,
        staged: StagedArtifact,
    ) -> str | None:
        q_check = (
            f"'{run_folder_id}' in parents and trashed = false and "
            f"appProperties has {{ key='runvault_checkpoint_id' and value='{checkpoint_id}' }}"
        )
        try:
            recovered = (
                self._service.files()
                .list(q=q_check, fields="files(id, size, appProperties)")
                .execute()
            )
            for cand in recovered.get("files", []):
                cand_size = int(cand.get("size", 0))
                cand_props = cand.get("appProperties") or {}
                if (
                    cand_size == staged.size_bytes
                    and cand_props.get("runvault_sha256") == staged.checksum_sha256
                ):
                    return str(cand["id"])
        except Exception:
            pass
        return None

    def _detect_and_handle_collision(
        self,
        run_folder_id: str,
        checkpoint_id: str,
        created_file_id: str,
        run_id: str,
        name: str,
    ) -> None:
        q_verify = (
            f"'{run_folder_id}' in parents and trashed = false and "
            f"appProperties has {{ key='runvault_checkpoint_id' and value='{checkpoint_id}' }}"
        )
        dups = (
            self._service.files()
            .list(q=q_verify, fields="files(id, createdTime)")
            .execute()
            .get("files", [])
        )
        if len(dups) > 1:

            def _dup_sort_key(f: dict[str, Any]) -> tuple[str, str]:
                return (str(f.get("createdTime", "")), str(f.get("id", "")))

            sorted_dups = sorted(dups, key=_dup_sort_key)
            if created_file_id != str(sorted_dups[0]["id"]):
                raise CheckpointCreationError(
                    run_id=run_id,
                    name=name,
                    reason=(
                        "Concurrent checkpoint publication collision detected for "
                        f"'{checkpoint_id}'. Duplicate remote artifact was cleaned up."
                    ),
                )

    def publish_artifact(
        self,
        run_id: str,
        checkpoint_id: str,
        staged: StagedArtifact,
        name: str,
    ) -> PublishedArtifact:
        """Upload staged checkpoint to Google Drive with zero-overwrite immutability guarantees."""
        from googleapiclient.http import MediaFileUpload  # type: ignore[reportMissingImports]

        temp_path = staged.staging_path or Path(staged.staging_ref)
        if not temp_path.exists():
            raise CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=f"Local staging file '{temp_path}' was removed before upload",
            )

        run_folder_id = self.get_or_create_run_folder(run_id)
        artifact_filename = f"{checkpoint_id}.bin"
        self._check_preflight_collision(
            run_folder_id,
            checkpoint_id,
            run_id=run_id,
            name=name,
            temp_path=temp_path,
        )

        file_metadata = {
            "name": artifact_filename,
            "parents": [run_folder_id],
            "appProperties": {
                "runvault_run_id": run_id,
                "runvault_checkpoint_id": checkpoint_id,
                "runvault_name": name,
                "runvault_sha256": staged.checksum_sha256,
                "runvault_size": str(staged.size_bytes),
            },
        }

        media = MediaFileUpload(
            str(temp_path),
            mimetype="application/octet-stream",
            chunksize=self._chunk_size,
            resumable=True,
        )

        created_file_id: str | None = None
        try:
            request = self._service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, name, size, trashed, appProperties",
            )

            response = None
            while response is None:
                try:
                    status, response = self._execute_with_retry(
                        request.next_chunk, f"upload_chunk_{checkpoint_id}"
                    )
                    _ = status
                except Exception as upload_err:
                    recovered_id = self._recover_lost_upload(run_folder_id, checkpoint_id, staged)
                    if recovered_id is not None:
                        created_file_id = recovered_id
                        break
                    raise upload_err

            if created_file_id is None and response is not None:
                created_file_id = str(response["id"])

            if created_file_id is not None:
                self._detect_and_handle_collision(
                    run_folder_id, checkpoint_id, created_file_id, run_id, name
                )
        except Exception as err:
            temp_path.unlink(missing_ok=True)
            if created_file_id:
                with contextlib.suppress(Exception):
                    self.delete_artifact(created_file_id)
            if isinstance(err, CheckpointCreationError):
                raise
            creation_err = CheckpointCreationError(
                run_id=run_id,
                name=name,
                reason=f"Google Drive checkpoint upload failed: {err}",
                original_error=err,
            )
            for note in getattr(err, "__notes__", []):
                creation_err.add_note(note)
            raise creation_err from err
        finally:
            temp_path.unlink(missing_ok=True)

        return PublishedArtifact(
            artifact_key=str(created_file_id),
            size_bytes=staged.size_bytes,
            checksum_sha256=staged.checksum_sha256,
            storage_provider=self.name,
        )

    def _fetch_artifact_metadata(self, artifact_key: str) -> dict[str, Any]:
        def fetch_meta() -> dict[str, Any]:
            return (  # type: ignore[no-any-return]
                self._service.files()
                .get(
                    fileId=artifact_key,
                    fields="id, name, size, trashed, appProperties",
                )
                .execute()
            )

        try:
            return self._execute_with_retry(fetch_meta, f"verify_meta_{artifact_key}")
        except StorageUnavailableError:
            raise
        except (TypeError, ValueError, AttributeError, KeyError):
            raise
        except Exception as err:
            status_code = getattr(getattr(err, "resp", None), "status", None)
            if status_code == HTTP_NOT_FOUND:
                return {"_missing": True}
            if status_code in (401, 403):
                raise StorageUnavailableError(
                    identifier="gdrive",
                    reason=f"Google Drive permission denied or authorization failed: {err}",
                    original_error=err,
                ) from err
            raise StorageUnavailableError(
                identifier=artifact_key,
                reason=f"Failed to inspect Google Drive artifact metadata: {err}",
                original_error=err,
            ) from err

    def _verify_stream_checksum(
        self,
        artifact_key: str,
        expected_checksum: str,
        expected_run_id: str | None,
    ) -> tuple[bool, str | None]:
        hasher = hashlib.sha256()
        try:
            with self.open_artifact(artifact_key, expected_run_id=expected_run_id) as stream:
                while chunk := stream.read(65536):
                    hasher.update(chunk)
        except StorageUnavailableError:
            raise
        except (TypeError, ValueError, AttributeError, KeyError):
            raise
        except Exception as err:
            status_code = getattr(getattr(err, "resp", None), "status", None)
            if status_code == HTTP_NOT_FOUND:
                return False, f"Artifact '{artifact_key}' not found or removed during verification"
            if status_code in (401, 403):
                raise StorageUnavailableError(
                    identifier="gdrive",
                    reason=f"Google Drive permission denied or authorization failed: {err}",
                    original_error=err,
                ) from err
            raise StorageUnavailableError(
                identifier=artifact_key,
                reason=f"Error streaming artifact bytes from Google Drive for verification: {err}",
                original_error=err,
            ) from err

        actual = hasher.hexdigest()
        if actual != expected_checksum:
            return False, f"Artifact SHA-256 mismatch: expected {expected_checksum}, got {actual}"
        return True, None

    def verify_artifact(
        self,
        artifact_key: str,
        expected_size: int,
        expected_checksum: str,
        expected_run_id: str | None = None,
    ) -> tuple[bool, str | None]:
        """Verify artifact size, non-trashed state, and SHA-256 integrity in Google Drive."""
        meta = self._fetch_artifact_metadata(artifact_key)
        if meta.get("_missing"):
            return False, f"Artifact '{artifact_key}' not found on Google Drive"
        if meta.get("trashed", False):
            return False, f"Artifact '{artifact_key}' is in Google Drive trash"

        actual_size = int(meta.get("size", 0))
        if actual_size != expected_size:
            return (
                False,
                f"Artifact size mismatch: expected {expected_size} bytes, got {actual_size} bytes",
            )

        app_props = meta.get("appProperties") or {}
        if expected_run_id and app_props.get("runvault_run_id"):
            actual_run = app_props.get("runvault_run_id")
            if actual_run != expected_run_id:
                return (
                    False,
                    f"Artifact run_id mismatch: expected '{expected_run_id}', got '{actual_run}'",
                )

        return self._verify_stream_checksum(artifact_key, expected_checksum, expected_run_id)

    def open_artifact(
        self,
        artifact_key: str,
        expected_run_id: str | None = None,
    ) -> BinaryIO:
        """Open a readable binary stream that streams bytes directly from Google Drive."""
        _ = expected_run_id
        return GoogleDriveDownloadStream(
            service=self._service,
            file_id=artifact_key,
            chunk_size=self._chunk_size,
        )

    def exists(self, artifact_key: str, run_id: str | None = None) -> bool:
        """Check whether a non-trashed file exists in Google Drive."""
        _ = run_id
        try:
            meta = self._service.files().get(fileId=artifact_key, fields="id, trashed").execute()
            return not bool(meta.get("trashed", False))
        except Exception:
            return False

    def delete_artifact(self, artifact_key: str, run_id: str | None = None) -> None:
        """Delete an artifact from Google Drive (used for failure compensation and cleanup)."""
        _ = run_id

        def do_delete() -> dict[str, Any]:
            return (  # type: ignore[no-any-return]
                self._service.files().delete(fileId=artifact_key).execute()
            )

        try:
            self._execute_with_retry(do_delete, f"delete_{artifact_key}")
        except Exception as err:
            status_code = getattr(getattr(err, "resp", None), "status", None)
            if status_code != HTTP_NOT_FOUND:
                logger.warning("Failed to delete Google Drive artifact '%s': %s", artifact_key, err)
