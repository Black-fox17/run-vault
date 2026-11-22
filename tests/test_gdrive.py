"""Deterministic tests for Google Drive storage provider, authentication, and checkpoint engine."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError  # type: ignore[reportMissingImports]

import runvault
from runvault.core.checkpoints import (
    CheckpointRecoveryResult,
    publish_checkpoint,
)
from runvault.core.db.repository import SQLiteMetadataRepository
from runvault.core.domain import RunStatus
from runvault.core.errors import (
    CheckpointCreationError,
    StorageUnavailableError,
)
from runvault.core.workspace import Workspace
from runvault.integrations.gdrive.auth import (
    GDRIVE_DEFAULT_SCOPES,
    authenticate_colab,
    check_gdrive_dependencies,
    get_credentials,
    is_colab_environment,
    save_token_cache,
)
from runvault.integrations.gdrive.provider import GoogleDriveStorageProvider
from runvault.integrations.gdrive.stream import GoogleDriveDownloadStream


def _write_bytes(p: Path, data: bytes) -> None:
    p.write_bytes(data)


class MockHttpResponse(dict[str, Any]):
    def __init__(self, status: int, reason: str = "Error", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.status = status
        self.reason = reason


class FakeDriveService:
    """In-memory deterministic mock of Google Drive API v3."""

    def __init__(self) -> None:
        self.files_db: dict[str, dict[str, Any]] = {}
        self._id_counter = 1
        self.delete_calls: list[str] = []
        self.create_calls: list[dict[str, Any]] = []
        self.get_media_calls: list[str] = []

        # Injected faults for error testing
        self.get_fault: Exception | None = None
        self.create_fault: Exception | None = None
        self.upload_chunk_faults: list[Exception] = []
        self.lost_response_fault: Exception | None = None
        self.simulate_concurrent_duplicate: bool = False

    def allocate_file_id(self) -> str:
        fid = f"gdrive_file_{self._id_counter}"
        self._id_counter += 1
        return fid

    def files(self) -> FakeFilesResource:
        return FakeFilesResource(self)


class FakeFilesResource:
    def __init__(self, service: FakeDriveService) -> None:
        self._service = service

    def list(self, q: str = "", fields: str = "") -> FakeListRequest:
        return FakeListRequest(self._service, q, fields)

    def create(
        self,
        body: dict[str, Any],
        media_body: Any = None,
        fields: str = "",
    ) -> FakeCreateRequest:
        self._service.create_calls.append(body)
        return FakeCreateRequest(self._service, body, media_body, fields)

    def get(self, fileId: str, fields: str = "") -> FakeGetRequest:  # noqa: N803
        return FakeGetRequest(self._service, fileId, fields)

    def get_media(self, fileId: str) -> FakeGetMediaRequest:  # noqa: N803
        self._service.get_media_calls.append(fileId)
        return FakeGetMediaRequest(self._service, fileId)

    def delete(self, fileId: str) -> FakeDeleteRequest:  # noqa: N803
        self._service.delete_calls.append(fileId)
        return FakeDeleteRequest(self._service, fileId)


class FakeListRequest:
    def __init__(self, service: FakeDriveService, q: str, fields: str) -> None:
        self._service = service
        self._q = q
        self._fields = fields

    def _matches_name_or_prop(self, file_meta: dict[str, Any]) -> bool:
        if "name = '" in self._q and " or appProperties has" in self._q:
            target_name = self._q.split("name = '")[1].split("'")[0]
            prop_str = self._q.split("appProperties has {")[1].split("}")[0]
            k = prop_str.split("key='")[1].split("'")[0]
            v = prop_str.split("value='")[1].split("'")[0]
            props = file_meta.get("appProperties") or {}
            return bool(file_meta.get("name") == target_name or props.get(k) == v)

        if "name = '" in self._q:
            target_name = self._q.split("name = '")[1].split("'")[0]
            if file_meta.get("name") != target_name:
                return False

        if "appProperties has" in self._q:
            prop_str = self._q.split("appProperties has {")[1].split("}")[0]
            k = prop_str.split("key='")[1].split("'")[0]
            v = prop_str.split("value='")[1].split("'")[0]
            props = file_meta.get("appProperties") or {}
            if props.get(k) != v:
                return False
        return True

    def execute(self) -> dict[str, Any]:
        matches: list[dict[str, Any]] = []
        for file_meta in self._service.files_db.values():
            if file_meta.get("trashed", False) and "trashed = false" in self._q:
                continue
            if not self._matches_name_or_prop(file_meta):
                continue
            if "mimeType = '" in self._q:
                target_mime = self._q.split("mimeType = '")[1].split("'")[0]
                if file_meta.get("mimeType") != target_mime:
                    continue
            if " in parents" in self._q:
                parent_target = self._q.split(" in parents")[0].rsplit("'", 2)[-2]
                if parent_target not in file_meta.get("parents", []):
                    continue
            matches.append(dict(file_meta))
        return {"files": matches}


class FakeCreateRequest:
    def __init__(
        self,
        service: FakeDriveService,
        body: dict[str, Any],
        media_body: Any,
        fields: str,
    ) -> None:
        self._service = service
        self._body = dict(body)
        self._media_body = media_body
        self._fields = fields
        self._done = False
        self._created_id = service.allocate_file_id()

    def execute(self) -> dict[str, Any]:
        _, result = self.next_chunk()
        return result or {}

    def next_chunk(self) -> tuple[Any, dict[str, Any] | None]:
        if self._service.upload_chunk_faults:
            fault = self._service.upload_chunk_faults.pop(0)
            raise fault

        content = b""
        if self._media_body is not None:
            # Read full media
            path_str = getattr(self._media_body, "_filename", None)
            if path_str:
                content = Path(path_str).read_bytes()

        file_entry = {
            "id": self._created_id,
            "name": self._body.get("name", "untitled"),
            "size": str(len(content)),
            "mimeType": self._body.get("mimeType", "application/octet-stream"),
            "parents": self._body.get("parents", []),
            "appProperties": self._body.get("appProperties", {}),
            "trashed": False,
            "content": content,
            "createdTime": "2026-11-22T01:00:00Z",
        }

        if self._media_body is not None and self._service.lost_response_fault is not None:
            fault = self._service.lost_response_fault
            self._service.lost_response_fault = None
            self._service.files_db[self._created_id] = file_entry
            raise fault

        if self._service.simulate_concurrent_duplicate:
            dup_id = self._service.allocate_file_id()
            dup_entry = dict(file_entry)
            dup_entry["id"] = dup_id
            dup_entry["createdTime"] = "2026-11-22T00:59:00Z"  # Earlier than our created id
            self._service.files_db[dup_id] = dup_entry

        self._service.files_db[self._created_id] = file_entry
        self._done = True
        return None, dict(file_entry)


class FakeGetRequest:
    def __init__(self, service: FakeDriveService, file_id: str, fields: str) -> None:
        self._service = service
        self._file_id = file_id
        self._fields = fields

    def execute(self) -> dict[str, Any]:
        if self._service.get_fault:
            raise self._service.get_fault

        if self._file_id not in self._service.files_db:
            resp = MockHttpResponse(404, "Not Found")
            raise HttpError(resp, b"File not found")

        file_meta = self._service.files_db[self._file_id]
        return dict(file_meta)


class FakeHttp:
    def __init__(self, service: FakeDriveService, file_id: str) -> None:
        self._service = service
        self._file_id = file_id

    def request(
        self,
        uri: str,
        method: str = "GET",
        body: Any = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, bytes]:
        _ = (uri, method, body, kwargs)
        if self._file_id not in self._service.files_db:
            resp = MockHttpResponse(404, "Not Found")
            raise HttpError(resp, b"File not found")

        content: bytes = bytes(self._service.files_db[self._file_id]["content"])
        total_len = len(content)

        if headers and "range" in headers:
            range_val = headers["range"]
            parts = range_val.split("=")[1].split("-")
            start = int(parts[0])
            end = min(total_len - 1, int(parts[1]))
            chunk = content[start : end + 1] if start <= total_len else b""
            resp = MockHttpResponse(
                206,
                **{
                    "content-range": f"bytes {start}-{end}/{total_len}",
                    "content-length": str(len(chunk)),
                },
            )
            return resp, chunk

        resp = MockHttpResponse(200, **{"content-length": str(total_len)})
        return resp, content


class FakeGetMediaRequest:
    def __init__(self, service: FakeDriveService, file_id: str) -> None:
        self._service = service
        self._file_id = file_id
        self.headers: dict[str, str] = {}
        self.http = FakeHttp(service, file_id)
        self.uri = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

    def execute(self) -> bytes:
        if self._file_id not in self._service.files_db:
            resp = MockHttpResponse(404, "Not Found")
            raise HttpError(resp, b"File not found")
        return bytes(self._service.files_db[self._file_id]["content"])


class FakeDeleteRequest:
    def __init__(self, service: FakeDriveService, file_id: str) -> None:
        self._service = service
        self._file_id = file_id

    def execute(self) -> dict[str, Any]:
        if self._file_id not in self._service.files_db:
            resp = MockHttpResponse(404, "Not Found")
            raise HttpError(resp, b"File not found")
        del self._service.files_db[self._file_id]
        return {}


# =============================================================================
# 1. Authentication Tests
# =============================================================================


def test_gdrive_check_dependencies() -> None:
    """check_gdrive_dependencies verifies presence or raises friendly message."""
    # When installed, does not raise
    check_gdrive_dependencies()

    with (
        patch.dict("sys.modules", {"google.auth": None}),
        pytest.raises(ImportError, match="pip install 'runvault\\[gdrive\\]'"),
    ):
        check_gdrive_dependencies()


def test_gdrive_is_colab_environment() -> None:
    """is_colab_environment accurately detects Colab module presence."""
    with patch.dict("sys.modules", {"google.colab": MagicMock()}):
        assert is_colab_environment() is True

    with patch.dict("sys.modules", {"google.colab": None}):
        assert is_colab_environment() is False


def test_gdrive_authenticate_colab_calls_colab_auth() -> None:
    """authenticate_colab calls google.colab.auth.authenticate_user() and returns default creds."""
    mock_colab = MagicMock()
    mock_creds = MagicMock()

    with (
        patch.dict(
            "sys.modules",
            {"google.colab": mock_colab, "google.colab.auth": mock_colab.auth},
        ),
        patch("google.auth.default", return_value=(mock_creds, "mock-project")) as mock_default,
    ):
        creds = authenticate_colab()
        mock_colab.auth.authenticate_user.assert_called_once()
        mock_default.assert_called_once_with(scopes=list(GDRIVE_DEFAULT_SCOPES))
        assert creds is mock_creds


def test_gdrive_authenticate_local_oauth_and_caching() -> None:
    """authenticate_local_oauth loads, caches, and sets 0o600 permissions."""
    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_creds.to_json.return_value = '{"token": "secret-oauth-token"}'

    # Injected credentials resolution
    resolved = get_credentials(credentials=mock_creds)
    assert resolved is mock_creds


# =============================================================================
# 2. Storage Provider Unit Tests
# =============================================================================


def test_gdrive_provider_contract_and_capabilities() -> None:
    """GoogleDriveStorageProvider adheres to StorageProvider contract."""
    fake_service = FakeDriveService()
    provider = GoogleDriveStorageProvider(drive_service=fake_service)

    assert provider.name == "gdrive"
    caps = provider.capabilities
    assert caps.provides_local_path is False
    assert caps.supports_exclusive_publication is True
    assert caps.supports_directory_sync is False
    assert caps.can_verify_integrity is True

    # get_artifact_path raises TypeError explaining remote nature
    with pytest.raises(TypeError, match="does not provide local filesystem paths"):
        provider.get_artifact_path("drive_123")


def test_gdrive_folder_hierarchy_and_caching() -> None:
    """GoogleDriveStorageProvider resolves or creates root folder and run subfolder."""
    fake_service = FakeDriveService()
    provider = GoogleDriveStorageProvider(
        drive_service=fake_service,
        root_folder_name="CustomVault",
    )

    # 1. Resolves root
    root_id = provider.get_or_create_root_folder()
    assert root_id in fake_service.files_db
    assert fake_service.files_db[root_id]["name"] == "CustomVault"

    # 2. Repeated resolution uses cache without extra list/create calls
    initial_create_count = len(fake_service.create_calls)
    assert provider.get_or_create_root_folder() == root_id
    assert len(fake_service.create_calls) == initial_create_count

    # 3. Resolves run folder under root
    run_folder_id = provider.get_or_create_run_folder("run-abc-123")
    assert run_folder_id in fake_service.files_db
    assert fake_service.files_db[run_folder_id]["name"] == "run-abc-123"
    assert root_id in fake_service.files_db[run_folder_id]["parents"]

    # 4. Run folder resolution is cached
    assert provider.get_or_create_run_folder("run-abc-123") == run_folder_id


def test_gdrive_stage_and_publish_immutability(tmp_path: Path) -> None:
    """stage_artifact creates local staging file; publish_artifact uploads and unlinks."""
    fake_service = FakeDriveService()
    ws = Workspace.from_path(tmp_path)
    provider = GoogleDriveStorageProvider(drive_service=fake_service, workspace=ws)

    payload = b"google drive model checkpoint bytes"
    staged = provider.stage_artifact(
        run_id="run-gd-1",
        checkpoint_id="ckpt-gd-1",
        writer=lambda p: _write_bytes(p, payload),
        name="epoch-1",
    )

    assert staged.size_bytes == len(payload)
    assert staged.checksum_sha256 == hashlib.sha256(payload).hexdigest()
    assert staged.staging_path is not None
    assert staged.staging_path.exists()

    # Publish
    published = provider.publish_artifact(
        run_id="run-gd-1",
        checkpoint_id="ckpt-gd-1",
        staged=staged,
        name="epoch-1",
    )

    assert published.storage_provider == "gdrive"
    file_id = published.artifact_key
    assert file_id in fake_service.files_db
    assert fake_service.files_db[file_id]["size"] == str(len(payload))

    # Local staging file was unlinked post-upload
    assert not staged.staging_path.exists()

    # Duplicate publication to same checkpoint_id raises CheckpointCreationError (no-overwrite)
    temp_dup = tmp_path / "dup.tmp"
    temp_dup.write_bytes(payload)
    staged_dup = provider.stage_artifact(
        run_id="run-gd-1",
        checkpoint_id="ckpt-gd-1",
        writer=lambda p: _write_bytes(p, payload),
        name="epoch-1-dup",
    )
    with pytest.raises(CheckpointCreationError, match="already exists in Google Drive"):
        provider.publish_artifact(
            run_id="run-gd-1",
            checkpoint_id="ckpt-gd-1",
            staged=staged_dup,
            name="epoch-1-dup",
        )


def test_gdrive_verify_artifact_scenarios() -> None:
    """verify_artifact differentiates missing, trashed, corrupted, and outage states."""
    fake_service = FakeDriveService()
    provider = GoogleDriveStorageProvider(drive_service=fake_service)

    content = b"checkpoint-for-verification"
    sha = hashlib.sha256(content).hexdigest()
    file_id = "test_file_id"
    fake_service.files_db[file_id] = {
        "id": file_id,
        "name": "ckpt.bin",
        "size": str(len(content)),
        "trashed": False,
        "content": content,
        "appProperties": {"runvault_run_id": "r1"},
    }

    # 1. Healthy artifact passes verification
    is_valid, reason = provider.verify_artifact(file_id, len(content), sha, expected_run_id="r1")
    assert is_valid is True
    assert reason is None

    # 2. Missing artifact returns (False, reason)
    is_valid, reason = provider.verify_artifact("nonexistent_id", len(content), sha)
    assert is_valid is False
    assert "not found" in str(reason)

    # 3. Trashed artifact returns (False, reason)
    fake_service.files_db[file_id]["trashed"] = True
    is_valid, reason = provider.verify_artifact(file_id, len(content), sha)
    assert is_valid is False
    assert "trash" in str(reason)
    fake_service.files_db[file_id]["trashed"] = False

    # 4. Size mismatch returns (False, reason)
    is_valid, reason = provider.verify_artifact(file_id, len(content) + 5, sha)
    assert is_valid is False
    assert "size mismatch" in str(reason).lower()

    # 5. Content SHA-256 mismatch returns (False, reason)
    bad_sha = "0" * 64
    is_valid, reason = provider.verify_artifact(file_id, len(content), bad_sha)
    assert is_valid is False
    assert "SHA-256 mismatch" in str(reason)

    # 6. Permission denied (403) raises StorageUnavailableError
    fake_service.get_fault = HttpError(MockHttpResponse(403, "Forbidden"), b"Permission denied")
    with pytest.raises(StorageUnavailableError, match="permission denied"):
        provider.verify_artifact(file_id, len(content), sha)

    # 7. Service Outage (503) raises StorageUnavailableError
    fake_service.get_fault = HttpError(MockHttpResponse(503, "Unavailable"), b"Backend outage")
    with pytest.raises(StorageUnavailableError, match="failed after"):
        provider.verify_artifact(file_id, len(content), sha)
    fake_service.get_fault = None


def test_gdrive_upload_retry_with_backoff() -> None:
    """Transient upload errors (429 rate limits, 503) retry and succeed."""
    fake_service = FakeDriveService()
    provider = GoogleDriveStorageProvider(drive_service=fake_service, max_retries=2)

    # Inject a 429 rate limit on first attempt, then succeed
    fake_service.upload_chunk_faults = [
        HttpError(MockHttpResponse(429, "Too Many Requests"), b"Rate limit exceeded"),
    ]

    staged = provider.stage_artifact(
        run_id="run-retry",
        checkpoint_id="ckpt-retry-1",
        writer=lambda p: _write_bytes(p, b"retry data"),
        name="retry-ckpt",
    )

    published = provider.publish_artifact(
        run_id="run-retry",
        checkpoint_id="ckpt-retry-1",
        staged=staged,
        name="retry-ckpt",
    )
    assert published.artifact_key in fake_service.files_db


# =============================================================================
# 3. Checkpoint Engine & End-to-End Integration Tests
# =============================================================================


def test_gdrive_checkpoint_engine_end_to_end(tmp_path: Path) -> None:
    """End-to-end checkpoint lifecycle with GoogleDriveStorageProvider."""
    fake_service = FakeDriveService()
    provider = GoogleDriveStorageProvider(drive_service=fake_service)

    content = b"weights-matrix-data" * 128  # synthetic weights

    with runvault.init(
        experiment="gdrive-e2e-exp",
        storage_provider=provider,
        workspace=tmp_path,
    ) as run:
        # Save checkpoint
        ckpt = run.save_checkpoint(
            name="epoch-weights",
            writer=lambda p: _write_bytes(p, content),
            step=1,
            checkpoint_id="gdrive_ckpt_1",
        )

        assert ckpt.storage_provider == "gdrive"
        assert ckpt.storage_key in fake_service.files_db

        # Verify SQLite record does NOT store secrets/tokens
        ws = Workspace.from_path(tmp_path)
        repo = SQLiteMetadataRepository(ws.db_path)
        try:
            stored_ckpt = repo.get_checkpoint(ckpt.id)
            assert stored_ckpt is not None
            assert stored_ckpt.storage_provider == "gdrive"
            assert stored_ckpt.storage_key == ckpt.storage_key
            assert "token" not in str(stored_ckpt.metadata)
            assert "secret" not in str(stored_ckpt.metadata)
        finally:
            repo.close()

        # Restore checkpoint through ActiveRun
        recovered = run.restore_checkpoint(ckpt.id)
        assert isinstance(recovered, CheckpointRecoveryResult)
        assert recovered.checkpoint.id == "gdrive_ckpt_1"
        assert recovered.path is None  # Remote provider has no local path
        assert recovered.is_fallback is False

        # Incremental streaming read
        with recovered.open() as stream:
            read_data = stream.read()
            assert read_data == content

        # Bounded read_bytes
        assert recovered.read_bytes(max_bytes=50) == content[:50]
        assert recovered.read_bytes() == content

        # __fspath__ raises TypeError
        with pytest.raises(TypeError, match="with no local path"):
            os.fspath(recovered)


def test_gdrive_recovery_fallback_on_corrupt_artifact(tmp_path: Path) -> None:
    """Recovery falls back to older checkpoint if latest Google Drive checkpoint is corrupt."""
    fake_service = FakeDriveService()
    provider = GoogleDriveStorageProvider(drive_service=fake_service)

    with runvault.init(
        experiment="gdrive-fallback-exp",
        storage_provider=provider,
        workspace=tmp_path,
    ) as run:
        ckpt1 = run.save_checkpoint(
            name="step-1",
            writer=lambda p: _write_bytes(p, b"valid-step-1"),
            step=1,
            checkpoint_id="gd_step_1",
        )
        ckpt2 = run.save_checkpoint(
            name="step-2",
            writer=lambda p: _write_bytes(p, b"valid-step-2"),
            step=2,
            checkpoint_id="gd_step_2",
        )

        # Corrupt ckpt2 in fake drive store (tamper with content bytes)
        fake_service.files_db[ckpt2.storage_key]["content"] = b"corrupted-step-2"

        # Recovery should fall back to ckpt1
        recovered = run.get_latest_valid_checkpoint()
        assert recovered is not None
        assert recovered.checkpoint.id == ckpt1.id
        assert recovered.is_fallback is True
        assert len(recovered.rejected_checkpoints) == 1
        assert recovered.rejected_checkpoints[0][0].id == ckpt2.id
        assert "SHA-256 mismatch" in recovered.rejected_checkpoints[0][1]


def test_gdrive_recovery_does_not_fallback_on_outage(tmp_path: Path) -> None:
    """Recovery raises StorageUnavailableError and does NOT fall back on transient outage."""
    fake_service = FakeDriveService()
    provider = GoogleDriveStorageProvider(drive_service=fake_service)

    with runvault.init(
        experiment="gdrive-outage-exp",
        storage_provider=provider,
        workspace=tmp_path,
    ) as run:
        run.save_checkpoint(
            name="step-1",
            writer=lambda p: _write_bytes(p, b"valid-step-1"),
            step=1,
            checkpoint_id="gd_step_1",
        )
        run.save_checkpoint(
            name="step-2",
            writer=lambda p: _write_bytes(p, b"valid-step-2"),
            step=2,
            checkpoint_id="gd_step_2",
        )

        # Inject 503 outage on latest checkpoint inspection
        fake_service.get_fault = HttpError(
            MockHttpResponse(503, "Unavailable"), b"Service unavailable"
        )

        # Must raise StorageUnavailableError without falling back to step-1
        with pytest.raises(StorageUnavailableError, match="failed after"):
            run.get_latest_valid_checkpoint()


def test_gdrive_workspace_reconnection(tmp_path: Path) -> None:
    """Reconnect to an existing Drive-backed workspace across session boundaries."""
    fake_service = FakeDriveService()
    provider_session1 = GoogleDriveStorageProvider(drive_service=fake_service)

    # Session 1: Train and save checkpoint
    run_id: str = ""
    ckpt_id: str = ""
    with runvault.init(
        experiment="reconnect-exp",
        storage_provider=provider_session1,
        workspace=tmp_path,
    ) as run:
        ckpt = run.save_checkpoint(
            name="weights-final",
            writer=lambda p: _write_bytes(p, b"final model weights"),
            checkpoint_id="final_weights_1",
        )
        run_id = run.id
        ckpt_id = ckpt.id

    # Session 2: Fresh session in a new runtime reconnecting to the same workspace
    provider_session2 = GoogleDriveStorageProvider(drive_service=fake_service)

    recovered = runvault.restore_checkpoint(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        storage=provider_session2,
        workspace=tmp_path,
    )

    assert recovered.checkpoint.id == "final_weights_1"
    assert recovered.read_bytes() == b"final model weights"
    assert recovered.path is None


def test_gdrive_compensation_on_db_insert_failure(tmp_path: Path) -> None:
    """If SQLite commit fails after upload, published Drive file is deleted in compensation."""
    fake_service = FakeDriveService()
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    provider = GoogleDriveStorageProvider(drive_service=fake_service)

    exp = runvault.create_experiment("comp-exp", workspace=tmp_path)
    run = repo.create_run(
        runvault.Run(id="run-comp", experiment_id=exp.id, status=RunStatus.RUNNING)
    )

    # Mock repository to simulate database commit failure
    with (
        patch.object(
            repo,
            "create_checkpoint",
            side_effect=RuntimeError("Simulated DB disk failure"),
        ),
        pytest.raises(CheckpointCreationError, match="Simulated DB disk failure"),
    ):
        publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="comp-ckpt",
            writer=lambda p: _write_bytes(p, b"payload"),
            checkpoint_id="comp_1",
            storage=provider,
        )

    # Compensation called delete on the uploaded Drive file
    assert len(fake_service.delete_calls) == 1


# =============================================================================
# 3. Hardening & Correctness Audit Tests (Phase 4.2.1)
# =============================================================================


def test_gdrive_upload_success_with_lost_response(tmp_path: Path) -> None:
    """If upload succeeds on Drive but network fails on response, provider recovers file ID."""
    fake_service = FakeDriveService()
    ws = Workspace.from_path(tmp_path)
    provider = GoogleDriveStorageProvider(drive_service=fake_service, workspace=ws)

    # Injected lost response: file is created in fake Drive, then TimeoutError is raised
    fake_service.lost_response_fault = TimeoutError("Connection dropped before response received")

    payload = b"model weights with lost response"
    staged = provider.stage_artifact(
        run_id="run-lost-resp",
        checkpoint_id="ckpt-lost",
        writer=lambda p: _write_bytes(p, payload),
        name="weights-lost-resp",
    )

    published = provider.publish_artifact(
        run_id="run-lost-resp",
        checkpoint_id="ckpt-lost",
        staged=staged,
        name="weights-lost-resp",
    )

    # Recovered file ID without creating any duplicate
    assert published.artifact_key in fake_service.files_db
    file_creates = [c for c in fake_service.create_calls if c.get("name") == "ckpt-lost.bin"]
    assert len(file_creates) == 1
    assert published.checksum_sha256 == hashlib.sha256(payload).hexdigest()


def test_gdrive_concurrent_publication_collision_detection(tmp_path: Path) -> None:
    """If concurrent workers upload same checkpoint ID, collision is detected and cleaned up."""
    fake_service = FakeDriveService()
    ws = Workspace.from_path(tmp_path)
    provider = GoogleDriveStorageProvider(drive_service=fake_service, workspace=ws)

    # Simulate another worker creating duplicate with earlier createdTime
    fake_service.simulate_concurrent_duplicate = True

    payload = b"concurrent checkpoint payload"
    staged = provider.stage_artifact(
        run_id="run-concurrent",
        checkpoint_id="ckpt-dup",
        writer=lambda p: _write_bytes(p, payload),
        name="weights-dup",
    )

    with pytest.raises(
        CheckpointCreationError,
        match="Concurrent checkpoint publication collision",
    ):
        provider.publish_artifact(
            run_id="run-concurrent",
            checkpoint_id="ckpt-dup",
            staged=staged,
            name="weights-dup",
        )

    # The losing duplicate file was cleaned up via delete_artifact
    assert len(fake_service.delete_calls) == 1


def test_gdrive_token_cache_permissions_and_atomicity(tmp_path: Path) -> None:
    """_save_token_cache sets 0o600 on cache file, 0o700 on directory, and replaces atomically."""
    cache_dir = tmp_path / "custom_config" / "runvault"
    token_file = cache_dir / "user_token.json"

    mock_creds = MagicMock()
    mock_creds.to_json.return_value = '{"access_token": "secret_abc_123"}'

    save_token_cache(token_file, mock_creds)

    assert token_file.exists()
    assert token_file.read_text(encoding="utf-8") == '{"access_token": "secret_abc_123"}'

    # Check file permissions (0o600)
    mode = token_file.stat().st_mode & 0o777
    assert mode == 0o600

    # Check parent directory permissions (0o700)
    dir_mode = cache_dir.stat().st_mode & 0o777
    assert dir_mode == 0o700


def test_gdrive_token_cache_account_isolation_and_custom_path(tmp_path: Path) -> None:
    """Custom token_cache_path isolates credentials across multiple Google accounts."""
    account1_cache = tmp_path / "account1.json"
    account2_cache = tmp_path / "account2.json"

    account1_cache.write_text('{"token": "token-user-1"}', encoding="utf-8")
    account2_cache.write_text('{"token": "token-user-2"}', encoding="utf-8")

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file") as mock_from_file:
        mock_creds_1 = MagicMock(valid=True)
        mock_creds_2 = MagicMock(valid=True)

        def mock_from_user_file(path: str, *_: Any, **__: Any) -> Any:
            return mock_creds_1 if "account1" in path else mock_creds_2

        mock_from_file.side_effect = mock_from_user_file

        creds1 = get_credentials(token_cache_path=account1_cache)
        creds2 = get_credentials(token_cache_path=account2_cache)

        assert creds1 is mock_creds_1
        assert creds2 is mock_creds_2


def test_gdrive_stream_python_contracts() -> None:
    """GoogleDriveDownloadStream satisfies all Python BinaryIO and RawIOBase stream contracts."""
    data = b"Line 1: Hello Google Drive\nLine 2: Checkpoint stream data\nLine 3: Final chunk bytes"
    fake_service = FakeDriveService()
    fake_service.files_db["stream_file_1"] = {
        "id": "stream_file_1",
        "name": "stream.bin",
        "size": str(len(data)),
        "content": data,
        "trashed": False,
    }

    with GoogleDriveDownloadStream(
        service=fake_service,
        file_id="stream_file_1",
        chunk_size=32,
    ) as stream:
        assert stream.readable() is True
        assert stream.seekable() is True
        assert stream.writable() is False

        # read(0) returns empty bytes
        assert stream.read(0) == b""

        assert stream.read(7) == b"Line 1:"
        assert stream.tell() == 7

        line1_rest = stream.readline()
        assert line1_rest == b" Hello Google Drive\n"
        assert stream.tell() == 27

        # readinto() with bytearray
        buf = bytearray(8)
        n = stream.readinto(buf)
        assert n == 8
        assert bytes(buf) == b"Line 2: "

        # read(None) reads to EOF
        tail = stream.read(None)
        assert tail.endswith(b"Final chunk bytes")
        assert stream.tell() == len(data)

        # EOF read returns b""
        assert stream.read(10) == b""

        # seek(0) rewinds and allows re-reading
        assert stream.seek(0) == 0
        assert stream.tell() == 0
        all_lines = stream.readlines()
        assert len(all_lines) == 3
        assert b"".join(all_lines) == data

    # Closed stream raises ValueError
    assert stream.readable() is False
    with pytest.raises(ValueError, match="closed file"):
        stream.read(10)


def test_gdrive_stream_bounded_memory_multi_chunk() -> None:
    """GoogleDriveDownloadStream holds only current chunk in SingleChunkWriter."""
    # 10 KB payload with tiny 256-byte chunks (spans 40 chunks)
    data = b"abcdefghijklmnopqrstuvwxyz012345" * 320
    assert len(data) == 10240

    fake_service = FakeDriveService()
    fake_service.files_db["multi_chunk_1"] = {
        "id": "multi_chunk_1",
        "name": "large.bin",
        "size": str(len(data)),
        "content": data,
        "trashed": False,
    }

    stream = GoogleDriveDownloadStream(
        service=fake_service,
        file_id="multi_chunk_1",
        chunk_size=256,
    )

    collected: list[bytes] = []
    while chunk := stream.read(512):
        collected.append(chunk)
        # SingleChunkWriter should NEVER accumulate more than a single chunk
        assert stream.buffered_chunk_size <= 256

    reconstructed = b"".join(collected)
    assert reconstructed == data
    stream.close()


def test_gdrive_error_classification_unauthorized_and_programming_error() -> None:
    """Permission denied raises StorageUnavailableError (no fallback); TypeError propagates."""
    fake_service = FakeDriveService()
    provider = GoogleDriveStorageProvider(drive_service=fake_service)

    fake_service.files_db["auth_file_1"] = {
        "id": "auth_file_1",
        "name": "auth.bin",
        "size": "50",
        "content": b"x" * 50,
        "trashed": False,
    }

    # 1. 401 Unauthorized raises StorageUnavailableError
    fake_service.get_fault = HttpError(MockHttpResponse(401, "Unauthorized"), b"Token expired")
    with pytest.raises(StorageUnavailableError, match="authorization failed"):
        provider.verify_artifact("auth_file_1", 50, "any-hash")

    # 2. Programming error (e.g. unexpected TypeError) propagates directly
    fake_service.get_fault = TypeError("Unexpected programming error inside fake")
    with pytest.raises(TypeError, match="Unexpected programming error"):
        provider.verify_artifact("auth_file_1", 50, "any-hash")


def test_gdrive_folder_cache_invalidation_and_recreation() -> None:
    """If cached folder is trashed on Drive, provider invalidates cache and recreates structure."""
    fake_service = FakeDriveService()
    provider = GoogleDriveStorageProvider(drive_service=fake_service)

    # 1. Create root folder
    root_id = provider.get_or_create_root_folder()
    assert root_id in fake_service.files_db

    # 2. Simulate user trashing root folder in Google Drive UI
    fake_service.files_db[root_id]["trashed"] = True

    # 3. Next call detects trashed folder, invalidates cache, and creates fresh root folder
    new_root_id = provider.get_or_create_root_folder()
    assert new_root_id != root_id
    assert new_root_id in fake_service.files_db
    assert fake_service.files_db[new_root_id]["trashed"] is False
