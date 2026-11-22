"""Tests for storage provider architecture, StorageCapabilities, and LocalStorageProvider."""

import errno
import hashlib
import io
import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import pytest

import runvault
from runvault.core.checkpoints import (
    CheckpointRecoveryResult,
    publish_checkpoint,
    restore_checkpoint,
    verify_checkpoint,
)
from runvault.core.db.repository import SQLiteMetadataRepository
from runvault.core.domain import Checkpoint, RunStatus
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
from runvault.core.storage.local import LocalStorageProvider
from runvault.core.workspace import Workspace


def _write_bytes(p: Path, data: bytes) -> None:
    p.write_bytes(data)


def test_storage_capabilities_defaults_and_local() -> None:
    """StorageCapabilities declares provider capabilities and durability guarantees."""
    default_caps = StorageCapabilities()
    assert default_caps.supports_exclusive_publication is True
    assert default_caps.supports_directory_sync is False
    assert default_caps.provides_local_path is True

    ws = Workspace.from_path(Path("/tmp/fake_ws"))
    local_provider = LocalStorageProvider(ws)
    assert local_provider.name == "local"
    assert local_provider.capabilities.supports_exclusive_publication is True
    assert local_provider.capabilities.supports_directory_sync == (os.name == "posix")
    assert local_provider.capabilities.provides_local_path is True


def test_local_storage_stage_and_publish_success(tmp_path: Path) -> None:
    """LocalStorageProvider stages and atomically publishes an artifact."""
    ws = Workspace.from_path(tmp_path)
    provider = LocalStorageProvider(ws)
    run_id = "test-run-1"
    ckpt_id = "ckpt-uuid-1"
    content = b"model weights tensor data"
    expected_hash = hashlib.sha256(content).hexdigest()

    staged = provider.stage_artifact(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        writer=lambda p: _write_bytes(p, content),
        name="epoch-1",
    )
    assert isinstance(staged, StagedArtifact)
    assert staged.staging_path is not None
    assert staged.size_bytes == len(content)
    assert staged.checksum_sha256 == expected_hash
    assert staged.staging_path.exists()
    assert staged.staging_path.name.endswith(".tmp")

    published = provider.publish_artifact(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        staged=staged,
        name="epoch-1",
    )
    assert isinstance(published, PublishedArtifact)
    assert published.artifact_key == f"checkpoints/{run_id}/{ckpt_id}.bin"
    assert published.size_bytes == len(content)
    assert published.checksum_sha256 == expected_hash
    assert published.storage_provider == "local"
    assert published.staging_cleanup_warning is None

    # Staging path was unlinked
    assert not staged.staging_path.exists()

    # Final artifact exists and is verified
    assert provider.exists(published.artifact_key, run_id=run_id)
    is_valid, reason = provider.verify_artifact(
        published.artifact_key,
        expected_size=len(content),
        expected_checksum=expected_hash,
        expected_run_id=run_id,
    )
    assert is_valid is True
    assert reason is None

    # Bytes retrieval
    retrieved_bytes = provider.read_artifact_bytes(published.artifact_key, expected_run_id=run_id)
    assert retrieved_bytes == content

    # Local path retrieval
    local_path = provider.get_artifact_path(published.artifact_key, expected_run_id=run_id)
    assert local_path.is_file()
    assert local_path.read_bytes() == content

    # Streaming retrieval via open_artifact
    with provider.open_artifact(published.artifact_key, expected_run_id=run_id) as stream:
        chunk = stream.read(5)
        assert chunk == content[:5]
        rest = stream.read()
        assert rest == content[5:]


def test_local_storage_immutability_duplicate_rejected(tmp_path: Path) -> None:
    """LocalStorageProvider rejects overwriting an existing checkpoint artifact."""
    ws = Workspace.from_path(tmp_path)
    provider = LocalStorageProvider(ws)
    run_id = "test-run-dup"
    ckpt_id = "ckpt-dup-1"
    content = b"original content"

    staged1 = provider.stage_artifact(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        writer=lambda p: _write_bytes(p, content),
        name="epoch-dup",
    )
    provider.publish_artifact(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        staged=staged1,
        name="epoch-dup",
    )

    # Attempting to publish again with the same checkpoint_id fails
    staged2 = provider.stage_artifact(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        writer=lambda p: _write_bytes(p, b"modified content"),
        name="epoch-dup-second",
    )
    with pytest.raises(CheckpointCreationError, match="already exists"):
        provider.publish_artifact(
            run_id=run_id,
            checkpoint_id=ckpt_id,
            staged=staged2,
            name="epoch-dup-second",
        )

    # Staging link is cleaned up and original content is preserved byte-for-byte
    assert staged2.staging_path is not None
    assert not staged2.staging_path.exists()
    assert (
        provider.read_artifact_bytes(f"checkpoints/{run_id}/{ckpt_id}.bin", expected_run_id=run_id)
        == content
    )


def test_local_storage_unsupported_hard_links_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When hard links are unsupported, publish_artifact fails cleanly without replacement."""
    ws = Workspace.from_path(tmp_path)
    provider = LocalStorageProvider(ws)
    run_id = "test-run-nolink"
    ckpt_id = "ckpt-nolink-1"

    staged = provider.stage_artifact(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        writer=lambda p: _write_bytes(p, b"some data"),
        name="epoch-nolink",
    )

    def mock_link(_src: os.PathLike[str], _dst: os.PathLike[str]) -> None:
        raise OSError(errno.EOPNOTSUPP, "Operation not supported on filesystem")

    monkeypatch.setattr(os, "link", mock_link)

    with pytest.raises(
        CheckpointCreationError, match="Exclusive atomic publication failed"
    ) as exc_info:
        provider.publish_artifact(
            run_id=run_id,
            checkpoint_id=ckpt_id,
            staged=staged,
            name="epoch-nolink",
        )

    assert "Operation not supported" in str(exc_info.value)
    # Staging file cleaned up
    assert staged.staging_path is not None
    assert not staged.staging_path.exists()
    # Final destination does not exist
    assert not provider.exists(f"checkpoints/{run_id}/{ckpt_id}.bin")


def test_local_storage_stage_failures(tmp_path: Path) -> None:
    """LocalStorageProvider validates staging artifact constraints (empty, missing, exception)."""
    ws = Workspace.from_path(tmp_path)
    provider = LocalStorageProvider(ws)
    run_id = "test-run-fail"

    # Writer raises exception
    def failing_writer(_p: Path) -> None:
        raise ValueError("serialization error")

    with pytest.raises(CheckpointCreationError, match="Writer callback failed"):
        provider.stage_artifact(
            run_id=run_id,
            checkpoint_id="c-fail-1",
            writer=failing_writer,
            name="fail-1",
        )

    # Writer does not create file
    with pytest.raises(CheckpointCreationError, match="without producing an artifact file"):
        provider.stage_artifact(
            run_id=run_id,
            checkpoint_id="c-fail-2",
            writer=lambda _p: None,
            name="fail-2",
        )

    # Writer produces 0-byte file
    with pytest.raises(CheckpointCreationError, match="empty \\(0 bytes\\)"):
        provider.stage_artifact(
            run_id=run_id,
            checkpoint_id="c-fail-3",
            writer=lambda p: _write_bytes(p, b""),
            name="fail-3",
        )


def test_local_storage_path_containment_and_symlinks(tmp_path: Path) -> None:
    """LocalStorageProvider strictly checks path traversal and rejects symlinks."""
    ws = Workspace.from_path(tmp_path)
    provider = LocalStorageProvider(ws)

    # Traversal in key
    with pytest.raises(CheckpointCorruptError, match="Path traversal"):
        provider.get_artifact_path("../secret.bin")

    # Outside checkpoints root
    outside_dir = tmp_path / "other_folder"
    outside_dir.mkdir(parents=True, exist_ok=True)
    outside_file = outside_dir / "leak.bin"
    outside_file.write_bytes(b"leaked data")
    rel_outside = str(outside_file.relative_to(tmp_path))

    with pytest.raises(CheckpointCorruptError, match="escapes checkpoints root"):
        provider.get_artifact_path(rel_outside)

    # Symlink artifact
    run_id = "sym-run"
    target_dir = ws.get_run_checkpoints_dir(run_id)
    sym_path = target_dir / "sym.bin"
    try:
        sym_path.symlink_to(outside_file)
    except OSError:
        pytest.skip("Symlinks not supported on filesystem")

    sym_key = str(sym_path.relative_to(ws.path))
    with pytest.raises(CheckpointCorruptError, match="symlink"):
        provider.get_artifact_path(sym_key)

    is_valid, reason = provider.verify_artifact(
        sym_key, expected_size=11, expected_checksum="dummy"
    )
    assert is_valid is False
    assert "symlink" in str(reason).lower()


def test_local_storage_delete_artifact(tmp_path: Path) -> None:
    """LocalStorageProvider delete_artifact removes the file."""
    ws = Workspace.from_path(tmp_path)
    provider = LocalStorageProvider(ws)
    run_id = "del-run"
    ckpt_id = "del-ckpt"
    staged = provider.stage_artifact(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        writer=lambda p: _write_bytes(p, b"to delete"),
        name="del-ep",
    )
    published = provider.publish_artifact(
        run_id=run_id,
        checkpoint_id=ckpt_id,
        staged=staged,
        name="del-ep",
    )
    assert provider.exists(published.artifact_key, run_id=run_id)

    provider.delete_artifact(published.artifact_key, run_id=run_id)
    assert not provider.exists(published.artifact_key, run_id=run_id)


class MockRemoteStorageProvider(StorageProvider):
    """In-memory mock storage provider simulating a remote cloud storage backend."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.get_artifact_path_called: bool = False
        self._caps = StorageCapabilities(
            supports_exclusive_publication=True,
            supports_directory_sync=False,
            provides_local_path=False,
        )

    @property
    def name(self) -> str:
        return "mock_remote"

    @property
    def capabilities(self) -> StorageCapabilities:
        return self._caps

    def stage_artifact(
        self,
        run_id: str,
        checkpoint_id: str,
        writer: Callable[[Path], None],
        name: str,
    ) -> StagedArtifact:
        _ = (run_id, name)
        # Buffer to temporary path for writer callback (deliberate local staging pattern)
        tmp = Path(f"/tmp/{checkpoint_id}.tmp")
        try:
            writer(tmp)
            data = tmp.read_bytes()
            staging_key = f"staging/{checkpoint_id}"
            self.blobs[staging_key] = data
            return StagedArtifact(
                size_bytes=len(data),
                checksum_sha256=hashlib.sha256(data).hexdigest(),
                staging_ref=staging_key,
                staging_path=None,
            )
        finally:
            tmp.unlink(missing_ok=True)

    def publish_artifact(
        self,
        run_id: str,
        checkpoint_id: str,
        staged: StagedArtifact,
        name: str,
    ) -> PublishedArtifact:
        final_key = f"remote/{run_id}/{checkpoint_id}.bin"
        if final_key in self.blobs:
            raise CheckpointCreationError(run_id, name, "already exists in remote store")
        staging_data = self.blobs.pop(staged.staging_ref)
        self.blobs[final_key] = staging_data
        return PublishedArtifact(
            artifact_key=final_key,
            size_bytes=len(staging_data),
            checksum_sha256=staged.checksum_sha256,
            storage_provider=self.name,
        )

    def exists(self, artifact_key: str, run_id: str | None = None) -> bool:
        _ = run_id
        return artifact_key in self.blobs

    def delete_artifact(self, artifact_key: str, run_id: str | None = None) -> None:
        _ = run_id
        self.blobs.pop(artifact_key, None)

    def verify_artifact(
        self,
        artifact_key: str,
        expected_size: int,
        expected_checksum: str,
        expected_run_id: str | None = None,
    ) -> tuple[bool, str | None]:
        _ = expected_run_id
        if artifact_key not in self.blobs:
            return False, f"Blob '{artifact_key}' not found in remote store"
        data = self.blobs[artifact_key]
        if len(data) != expected_size:
            return False, f"Size mismatch: expected {expected_size}, got {len(data)}"
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_checksum:
            return False, f"Hash mismatch: expected {expected_checksum}, got {actual_hash}"
        return True, None

    def open_artifact(
        self,
        artifact_key: str,
        expected_run_id: str | None = None,
    ) -> BinaryIO:
        _ = expected_run_id
        if artifact_key not in self.blobs:
            raise FileNotFoundError(f"Blob '{artifact_key}' not found")
        return io.BytesIO(self.blobs[artifact_key])

    def get_artifact_path(self, artifact_key: str, expected_run_id: str | None = None) -> Path:
        self.get_artifact_path_called = True
        return super().get_artifact_path(artifact_key, expected_run_id=expected_run_id)


def test_custom_storage_provider_with_checkpoints_engine(tmp_path: Path) -> None:
    """The checkpoint engine transparently works with any StorageProvider implementation."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    remote_provider = MockRemoteStorageProvider()

    try:
        exp = runvault.create_experiment("remote-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-remote-1", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        content = b"remote tensor weights data"
        ckpt = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="remote-epoch-1",
            writer=lambda p: _write_bytes(p, content),
            checkpoint_id="remote_ckpt_1",
            storage=remote_provider,
        )

        assert ckpt.id == "remote_ckpt_1"
        assert ckpt.storage_provider == "mock_remote"
        assert ckpt.storage_key == f"remote/{run.id}/remote_ckpt_1.bin"
        assert ckpt.artifact_path == ckpt.storage_key
        assert ckpt.size_bytes == len(content)
        assert remote_provider.exists(ckpt.storage_key)

        # Verification using custom provider
        is_valid, reason = verify_checkpoint(
            ckpt, ws, expected_run_id=run.id, storage=remote_provider
        )
        assert is_valid is True
        assert reason is None

        # Recovery using custom provider
        result = restore_checkpoint(
            run_id=run.id,
            workspace=ws,
            repository=repo,
            checkpoint_id=ckpt.id,
            storage=remote_provider,
        )
        assert isinstance(result, CheckpointRecoveryResult)
        assert result.checkpoint.id == "remote_ckpt_1"
        assert result.path is None  # Remote provider has no local path
        assert result.is_fallback is False

        # Open streaming binary reader from recovery result directly
        with result.open() as stream:
            assert stream.read() == content

        # Reading bytes directly from recovery result
        assert result.read_bytes() == content
        assert result.read_text() == content.decode("utf-8")

        # __fspath__ raises TypeError explaining that it is remote
        with pytest.raises(TypeError, match="with no local path"):
            os.fspath(result)

        # get_artifact_path was never invoked on the remote provider
        assert remote_provider.get_artifact_path_called is False
    finally:
        repo.close()


def test_checkpoint_streaming_reader_bounded_memory(tmp_path: Path) -> None:
    """CheckpointRecoveryResult.open() returns a streaming reader for incremental I/O."""
    remote_provider = MockRemoteStorageProvider()

    # Create a 256 KB synthetic artifact
    chunk_unit = b"0123456789ABCDEF" * 64  # 1 KB
    payload = chunk_unit * 256  # 256 KB

    with runvault.init(
        experiment="stream-exp",
        storage_provider=remote_provider,
        workspace=tmp_path,
    ) as run:
        ckpt = run.save_checkpoint(
            name="large-tensor",
            writer=lambda p: _write_bytes(p, payload),
            checkpoint_id="large_stream_ckpt",
        )
        assert ckpt.storage_key == f"remote/{run.id}/large_stream_ckpt.bin"

        recovered = run.restore_checkpoint(ckpt.id)

        # 1. Incremental stream read in 1KB chunks
        received_chunks: list[bytes] = []
        with recovered.open() as stream:
            while chunk := stream.read(1024):
                received_chunks.append(chunk)

        assert len(received_chunks) == 256
        assert b"".join(received_chunks) == payload

        # 2. Bounded read_bytes with max_bytes
        bounded_bytes = recovered.read_bytes(max_bytes=100)
        assert len(bounded_bytes) == 100
        assert bounded_bytes == payload[:100]

        # 3. Full read_bytes
        assert recovered.read_bytes() == payload


def test_active_run_with_custom_storage_provider(tmp_path: Path) -> None:
    """ActiveRun accepts and uses a configured storage_provider."""
    remote_provider = MockRemoteStorageProvider()

    with runvault.init(
        experiment="active-run-storage-exp",
        storage_provider=remote_provider,
        workspace=tmp_path,
    ) as run:
        assert run.storage_provider is remote_provider

        ckpt = run.save_checkpoint(
            name="custom-store-ckpt",
            writer=lambda p: _write_bytes(p, b"active run remote payload"),
            checkpoint_id="active_remote_1",
        )
        assert ckpt.storage_provider == "mock_remote"

        recovered = run.restore_checkpoint(ckpt.id)
        assert recovered.read_bytes() == b"active run remote payload"
        assert recovered.path is None

        # Latest valid checkpoint also uses configured provider
        latest = run.get_latest_valid_checkpoint()
        assert latest is not None
        assert latest.checkpoint.id == "active_remote_1"
        assert latest.read_bytes() == b"active run remote payload"


def test_publish_checkpoint_verifies_before_metadata_commit_and_compensates(
    tmp_path: Path,
) -> None:
    """publish_checkpoint verifies artifact before DB insert; failure cleans up and aborts."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)

    deleted_keys: list[str] = []

    class FailingVerifyProvider(MockRemoteStorageProvider):
        def verify_artifact(
            self,
            artifact_key: str,
            expected_size: int,
            expected_checksum: str,
            expected_run_id: str | None = None,
        ) -> tuple[bool, str | None]:
            _ = (artifact_key, expected_size, expected_checksum, expected_run_id)
            raise OSError(errno.ETIMEDOUT, "Simulated cloud I/O timeout")

        def delete_artifact(self, artifact_key: str, run_id: str | None = None) -> None:
            deleted_keys.append(artifact_key)
            super().delete_artifact(artifact_key, run_id)

    provider = FailingVerifyProvider()
    try:
        exp = runvault.create_experiment("verify-fail-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-vf-1", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        with pytest.raises(CheckpointCreationError, match="Simulated cloud I/O timeout"):
            publish_checkpoint(
                workspace=ws,
                repository=repo,
                run_id=run.id,
                name="vf-ckpt",
                writer=lambda p: _write_bytes(p, b"test"),
                checkpoint_id="vf_ckpt_1",
                storage=provider,
            )

        # Artifact was cleaned up via compensation
        assert len(deleted_keys) == 1
        assert "vf_ckpt_1" in deleted_keys[0]

        # No metadata record was inserted into SQLite
        assert repo.get_checkpoint("vf_ckpt_1") is None
        assert len(repo.list_checkpoints(run.id)) == 0
    finally:
        repo.close()


def test_recovery_fallback_on_corrupt_latest_checkpoint(tmp_path: Path) -> None:
    """When latest checkpoint is corrupt, recovery falls back to older valid checkpoint."""
    with runvault.init(experiment="fallback-corrupt-exp", workspace=tmp_path) as run:
        ckpt1 = run.save_checkpoint("step1", writer=lambda p: _write_bytes(p, b"valid-1"), step=1)
        ckpt2 = run.save_checkpoint("step2", writer=lambda p: _write_bytes(p, b"valid-2"), step=2)

        # Tamper with ckpt2 artifact (same length to force SHA-256 hash mismatch)
        ckpt2_path = tmp_path / ckpt2.artifact_path
        ckpt2_path.write_bytes(b"valid-X")

        # Recovery of latest valid checkpoint falls back to ckpt1
        recovered = run.get_latest_valid_checkpoint()
        assert recovered is not None
        assert recovered.checkpoint.id == ckpt1.id
        assert recovered.is_fallback is True
        assert len(recovered.rejected_checkpoints) == 1
        assert recovered.rejected_checkpoints[0][0].id == ckpt2.id
        assert "SHA-256 mismatch" in recovered.rejected_checkpoints[0][1]
        assert recovered.read_bytes() == b"valid-1"


def test_recovery_fallback_on_missing_latest_checkpoint(tmp_path: Path) -> None:
    """When latest checkpoint file is missing, recovery falls back to older valid checkpoint."""
    with runvault.init(experiment="fallback-missing-exp", workspace=tmp_path) as run:
        ckpt1 = run.save_checkpoint("step1", writer=lambda p: _write_bytes(p, b"valid-1"), step=1)
        ckpt2 = run.save_checkpoint("step2", writer=lambda p: _write_bytes(p, b"valid-2"), step=2)

        # Delete ckpt2 artifact directly
        ckpt2_path = tmp_path / ckpt2.artifact_path
        ckpt2_path.unlink()

        # Recovery falls back to ckpt1
        recovered = run.get_latest_valid_checkpoint()
        assert recovered is not None
        assert recovered.checkpoint.id == ckpt1.id
        assert recovered.is_fallback is True
        assert len(recovered.rejected_checkpoints) == 1
        assert recovered.rejected_checkpoints[0][0].id == ckpt2.id
        assert "does not exist" in recovered.rejected_checkpoints[0][1]


def test_recovery_does_not_fallback_on_provider_outage(tmp_path: Path) -> None:
    """Transient provider outage on latest checkpoint must raise and NOT silently fall back."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)

    outage_active = False

    class OutageProvider(LocalStorageProvider):
        def verify_artifact(
            self,
            artifact_key: str,
            expected_size: int,
            expected_checksum: str,
            expected_run_id: str | None = None,
        ) -> tuple[bool, str | None]:
            if outage_active and "step2" in artifact_key:
                raise StorageUnavailableError(
                    identifier=artifact_key,
                    reason="Simulated provider connection reset / 503 outage",
                )
            return super().verify_artifact(
                artifact_key, expected_size, expected_checksum, expected_run_id
            )

    provider = OutageProvider(ws)
    try:
        exp = runvault.create_experiment("outage-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-outage", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="step1",
            writer=lambda p: _write_bytes(p, b"step1-content"),
            checkpoint_id="step1",
            storage=provider,
        )
        publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="step2",
            writer=lambda p: _write_bytes(p, b"step2-content"),
            checkpoint_id="step2",
            storage=provider,
        )

        # Enable simulated provider outage on step2
        outage_active = True

        # Recovery MUST raise StorageUnavailableError and must NOT silently fall back to step1!
        with pytest.raises(StorageUnavailableError, match="503 outage"):
            runvault.get_latest_valid_checkpoint(run.id, workspace=tmp_path, storage=provider)
    finally:
        repo.close()


def test_recovery_does_not_fallback_on_permission_error(tmp_path: Path) -> None:
    """Permission denied accessing latest checkpoint raises error without fallback."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)

    denied_active = False

    class PermissionProvider(LocalStorageProvider):
        def verify_artifact(
            self,
            artifact_key: str,
            expected_size: int,
            expected_checksum: str,
            expected_run_id: str | None = None,
        ) -> tuple[bool, str | None]:
            if denied_active and "step2" in artifact_key:
                raise PermissionError("EACCES: Permission denied accessing storage volume")
            return super().verify_artifact(
                artifact_key, expected_size, expected_checksum, expected_run_id
            )

    provider = PermissionProvider(ws)
    try:
        exp = runvault.create_experiment("perm-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-perm", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="step1",
            writer=lambda p: _write_bytes(p, b"step1-content"),
            checkpoint_id="step1",
            storage=provider,
        )
        publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="step2",
            writer=lambda p: _write_bytes(p, b"step2-content"),
            checkpoint_id="step2",
            storage=provider,
        )

        denied_active = True

        with pytest.raises(StorageUnavailableError, match="Permission denied"):
            runvault.get_latest_valid_checkpoint(run.id, workspace=tmp_path, storage=provider)
    finally:
        repo.close()


def test_successful_recovery_reports_fallback_status(tmp_path: Path) -> None:
    """When latest checkpoint is valid, recovery reports is_fallback=False and empty rejections."""
    with runvault.init(experiment="no-fallback-exp", workspace=tmp_path) as run:
        ckpt = run.save_checkpoint("step1", writer=lambda p: _write_bytes(p, b"healthy-data"))

        recovered = run.get_latest_valid_checkpoint()
        assert recovered is not None
        assert recovered.checkpoint.id == ckpt.id
        assert recovered.is_fallback is False
        assert recovered.rejected_checkpoints == ()


def test_staged_artifact_decoupled_from_local_path() -> None:
    """StagedArtifact supports provider-neutral token identity without requiring a local Path."""
    # Remote/in-memory provider instantiation without local path
    staged_remote = StagedArtifact(
        size_bytes=512,
        checksum_sha256="0" * 64,
        staging_ref="s3://upload-staging-bucket/staging-session-1234",
    )
    assert staged_remote.staging_path is None
    assert staged_remote.staging_ref == "s3://upload-staging-bucket/staging-session-1234"
    assert staged_remote.size_bytes == 512

    # Local provider instantiation with Path automatically populates staging_ref
    p = Path("/tmp/mock-staging.tmp")
    staged_local = StagedArtifact(
        staging_path=p,
        size_bytes=100,
        checksum_sha256="1" * 64,
    )
    assert staged_local.staging_path == p
    assert staged_local.staging_ref == str(p)


def test_checkpoint_metadata_invariants_storage_key_cannot_be_empty(tmp_path: Path) -> None:
    """Repository rejects checkpoint creation if storage_key or storage_provider is empty."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("inv-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-inv-1", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        # Empty storage_key
        with pytest.raises(ValueError, match="storage_key"):
            ckpt_empty_key = Checkpoint(
                id="c-inv-1",
                run_id=run.id,
                name="test",
                artifact_path="",
                storage_key="",
                size_bytes=10,
                checksum_sha256="a" * 64,
            )
            repo.create_checkpoint(ckpt_empty_key)

        # Empty storage_provider
        with pytest.raises(ValueError, match="storage_provider"):
            ckpt_empty_prov = Checkpoint(
                id="c-inv-2",
                run_id=run.id,
                name="test",
                storage_key="valid/key.bin",
                storage_provider="",
                size_bytes=10,
                checksum_sha256="a" * 64,
            )
            repo.create_checkpoint(ckpt_empty_prov)
    finally:
        repo.close()


def test_persisting_and_retrieving_opaque_storage_key_in_sqlite(tmp_path: Path) -> None:
    """SQLite repository preserves opaque storage keys independently from local paths."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("opaque-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-op-1", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        opaque_key = "s3://ml-vault-bucket/checkpoints/run-op-1/weights-001.bin"
        ckpt = Checkpoint(
            id="c-opaque-1",
            run_id=run.id,
            name="opaque-test",
            storage_key=opaque_key,
            size_bytes=1024,
            checksum_sha256="f" * 64,
            storage_provider="s3",
        )
        saved = repo.create_checkpoint(ckpt)
        assert saved.storage_key == opaque_key
        assert saved.storage_provider == "s3"

        # Direct retrieval by ID
        fetched = repo.get_checkpoint(saved.id)
        assert fetched is not None
        assert fetched.storage_key == opaque_key
        assert fetched.artifact_path == opaque_key
        assert fetched.storage_provider == "s3"

        # List retrieval
        all_ckpts = repo.list_checkpoints(run.id)
        assert len(all_ckpts) == 1
        assert all_ckpts[0].storage_key == opaque_key
        assert all_ckpts[0].storage_provider == "s3"

        # Raw SQLite column verification
        cursor = repo.connection.cursor()
        cursor.execute(
            "SELECT storage_provider, storage_key, artifact_path FROM checkpoints WHERE id = ?;",
            (saved.id,),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "s3"
        assert row[1] == opaque_key
        assert row[2] == opaque_key
    finally:
        repo.close()
