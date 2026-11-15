"""Comprehensive unit and integration tests for RunVault Phase 3 Checkpoint Engine."""

import errno
import hashlib
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import runvault
from runvault.core import checkpoints as checkpoints_mod
from runvault.core.checkpoints import (
    CheckpointRecoveryResult,
    publish_checkpoint,
    resolve_and_validate_artifact_path,
    restore_checkpoint,
    verify_checkpoint,
)
from runvault.core.db.repository import SQLiteMetadataRepository
from runvault.core.domain import CURRENT_CHECKPOINT_FORMAT_VERSION, Checkpoint, RunStatus
from runvault.core.errors import (
    CheckpointCorruptError,
    CheckpointCreationError,
    CheckpointNotFoundError,
    CheckpointRecoveryError,
    TerminalRunError,
    UnsupportedCheckpointFormatError,
)
from runvault.core.workspace import Workspace


def _write(dest: Path, content: bytes) -> None:
    dest.write_bytes(content)


# =============================================================================
# 1. Checkpoint Domain & Validation Tests
# =============================================================================


def test_checkpoint_domain_validation_success() -> None:
    """Verify that a well-formed Checkpoint domain object initializes correctly."""
    checksum = hashlib.sha256(b"dummy payload").hexdigest()
    ckpt = Checkpoint(
        id="ckpt-123",
        run_id="run-456",
        name="epoch-10",
        artifact_path="checkpoints/run-456/ckpt-123.bin",
        size_bytes=13,
        checksum_sha256=checksum,
        step=10,
        metadata={"val_acc": 0.95},
    )
    assert ckpt.id == "ckpt-123"
    assert ckpt.run_id == "run-456"
    assert ckpt.name == "epoch-10"
    assert ckpt.size_bytes == 13
    assert ckpt.checksum_sha256 == checksum
    assert ckpt.step == 10
    assert ckpt.metadata == {"val_acc": 0.95}
    assert ckpt.format_version == CURRENT_CHECKPOINT_FORMAT_VERSION
    assert ckpt.created_at is not None

    # Serialization roundtrip
    serialized = ckpt.serialize_metadata()
    assert Checkpoint.deserialize_metadata(serialized) == {"val_acc": 0.95}
    assert Checkpoint.deserialize_metadata(None) == {}


def test_checkpoint_domain_validation_failures() -> None:
    """Verify invalid domain attributes raise ValueError."""
    valid_sha = hashlib.sha256(b"test").hexdigest()

    # Empty id
    with pytest.raises(ValueError, match="id must be a non-empty string"):
        Checkpoint(
            id="",
            run_id="r1",
            name="n1",
            artifact_path="p",
            size_bytes=1,
            checksum_sha256=valid_sha,
        )

    # Empty run_id
    with pytest.raises(ValueError, match="run_id must be a non-empty string"):
        Checkpoint(
            id="c1",
            run_id="",
            name="n1",
            artifact_path="p",
            size_bytes=1,
            checksum_sha256=valid_sha,
        )

    # Path traversal in name
    with pytest.raises(ValueError, match="must be a label, not a filesystem path"):
        Checkpoint(
            id="c1",
            run_id="r1",
            name="../../secret",
            artifact_path="p",
            size_bytes=1,
            checksum_sha256=valid_sha,
        )

    with pytest.raises(ValueError, match="must be a label, not a filesystem path"):
        Checkpoint(
            id="c1",
            run_id="r1",
            name="sub/dir",
            artifact_path="p",
            size_bytes=1,
            checksum_sha256=valid_sha,
        )

    # Negative size
    with pytest.raises(ValueError, match="size_bytes must be >= 0"):
        Checkpoint(
            id="c1",
            run_id="r1",
            name="n1",
            artifact_path="p",
            size_bytes=-5,
            checksum_sha256=valid_sha,
        )

    # Invalid checksum length
    with pytest.raises(ValueError, match="checksum_sha256 must be a 64-character hex string"):
        Checkpoint(
            id="c1", run_id="r1", name="n1", artifact_path="p", size_bytes=1, checksum_sha256="abc"
        )

    # Invalid step
    with pytest.raises(ValueError, match="step must be a non-negative integer"):
        Checkpoint(
            id="c1",
            run_id="r1",
            name="n1",
            artifact_path="p",
            size_bytes=1,
            checksum_sha256=valid_sha,
            step=-1,
        )

    with pytest.raises(ValueError, match="step must be a non-negative integer"):
        Checkpoint(
            id="c1",
            run_id="r1",
            name="n1",
            artifact_path="p",
            size_bytes=1,
            checksum_sha256=valid_sha,
            step=True,
        )  # pyright: ignore[reportArgumentType]


# =============================================================================
# 2. Atomic Publication & Filesystem Consistency Tests
# =============================================================================


def test_atomic_publication_success(tmp_path: Path) -> None:
    """Verify that publish_checkpoint atomically writes, hashes, and registers an artifact."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("ckpt-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-atomic", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        payload = b"state_dict_weights_layer_1_and_2_serialized"
        expected_sha = hashlib.sha256(payload).hexdigest()

        def writer(dest: Path) -> None:
            dest.write_bytes(payload)

        ckpt = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="epoch-1",
            writer=writer,
            step=1,
            metadata={"epoch": 1, "loss": 0.45},
        )

        assert ckpt.id is not None
        assert ckpt.name == "epoch-1"
        assert ckpt.size_bytes == len(payload)
        assert ckpt.checksum_sha256 == expected_sha

        # Verify final published artifact exists and temp file does not
        final_file = ws.path / ckpt.artifact_path
        assert final_file.is_file()
        assert final_file.read_bytes() == payload

        temp_file = ws.get_run_checkpoints_dir(run.id) / f"{ckpt.id}.tmp"
        assert not temp_file.exists()

        # Check metadata persisted in SQLite
        persisted = repo.get_checkpoint(ckpt.id)
        assert persisted is not None
        assert persisted.checksum_sha256 == expected_sha
    finally:
        repo.close()


def test_writer_callback_failure_cleans_up_temp_file(tmp_path: Path) -> None:
    """If writer raises an exception, the temp file is deleted and no metadata is persisted."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("fail-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-fail", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        def failing_writer(dest: Path) -> None:
            dest.write_bytes(b"partial corrupted write")
            raise RuntimeError("Disk write interrupted by CUDA error")

        with pytest.raises(CheckpointCreationError, match="Writer callback failed"):
            publish_checkpoint(
                workspace=ws,
                repository=repo,
                run_id=run.id,
                name="failing-ckpt",
                writer=failing_writer,
            )

        # No temporary or published artifact exists
        checkpoints_dir = ws.get_run_checkpoints_dir(run.id)
        assert list(checkpoints_dir.glob("*.tmp")) == []
        assert list(checkpoints_dir.glob("*.bin")) == []

        # No checkpoint in SQLite
        assert repo.list_checkpoints(run.id) == []
    finally:
        repo.close()


def test_empty_artifact_rejected(tmp_path: Path) -> None:
    """An empty (0-byte) artifact is rejected to prevent committing empty files."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("empty-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-empty", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        def empty_writer(dest: Path) -> None:
            dest.touch()  # Creates 0-byte file

        with pytest.raises(CheckpointCreationError, match="Produced artifact is empty"):
            publish_checkpoint(
                workspace=ws,
                repository=repo,
                run_id=run.id,
                name="empty-ckpt",
                writer=empty_writer,
            )

        assert repo.list_checkpoints(run.id) == []
    finally:
        repo.close()


def test_checkpoint_immutability_enforced(tmp_path: Path) -> None:
    """Overwriting an existing committed checkpoint is strictly rejected."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("immut-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-immut", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        def writer(dest: Path) -> None:
            dest.write_bytes(b"version 1")

        # First save with explicit ID
        ckpt1 = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="v1",
            writer=writer,
            checkpoint_id="fixed-id-123",
        )
        assert ckpt1.id == "fixed-id-123"

        # Attempt to publish again with same ID must fail
        with pytest.raises(CheckpointCreationError, match="already exists"):
            publish_checkpoint(
                workspace=ws,
                repository=repo,
                run_id=run.id,
                name="v2",
                writer=writer,
                checkpoint_id="fixed-id-123",
            )
    finally:
        repo.close()


# =============================================================================
# 3. Checkpoint Verification & Recovery Tests
# =============================================================================


def test_checkpoint_verification_detects_corruptions(tmp_path: Path) -> None:
    """Verify that size, checksum, and missing file corruptions are accurately detected."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("verify-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-verify", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        content = b"original valid weights"
        ckpt = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="v1",
            writer=lambda p: _write(p, content),
        )

        # 1. Valid artifact passes
        valid, reason = verify_checkpoint(ckpt, ws)
        assert valid is True
        assert reason is None

        artifact_file = ws.path / ckpt.artifact_path

        # 2. Corrupt checksum (flip a byte)
        artifact_file.write_bytes(b"tampered valid weights")
        valid, reason = verify_checkpoint(ckpt, ws)
        assert valid is False
        assert "Artifact SHA-256 mismatch" in str(reason)

        # 3. Corrupt size (truncate)
        artifact_file.write_bytes(b"truncated")
        valid, reason = verify_checkpoint(ckpt, ws)
        assert valid is False
        assert "Artifact size mismatch" in str(reason)

        # 4. Missing file
        artifact_file.unlink()
        valid, reason = verify_checkpoint(ckpt, ws)
        assert valid is False
        assert "does not exist" in str(reason)
    finally:
        repo.close()


def test_recovery_fallback_to_earlier_valid_checkpoint(tmp_path: Path) -> None:
    """When the latest checkpoint is corrupt, recovery transparently falls back to older valid."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("fallback-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-fb", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        # Save Checkpoint 1 (Epoch 1)
        ckpt1 = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="epoch-1",
            writer=lambda p: _write(p, b"epoch 1 model weights"),
            step=1,
        )

        # Save Checkpoint 2 (Epoch 2)
        ckpt2 = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="epoch-2",
            writer=lambda p: _write(p, b"epoch 2 model weights"),
            step=2,
        )

        # Corrupt the latest checkpoint (epoch 2) by modifying its file
        (ws.path / ckpt2.artifact_path).write_bytes(b"corrupted binary data")

        # Attempt recovery: should fall back to epoch 1
        restored_ckpt, path, is_fallback = restore_checkpoint(
            run_id=run.id,
            workspace=ws,
            repository=repo,
        )

        assert is_fallback is True
        assert restored_ckpt.id == ckpt1.id
        assert restored_ckpt.step == 1
        assert path.read_bytes() == b"epoch 1 model weights"
    finally:
        repo.close()


def test_recovery_all_corrupt_raises_recovery_error(tmp_path: Path) -> None:
    """When checkpoints exist but all are corrupted, CheckpointRecoveryError is raised."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("all-corrupt-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-ac", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        ckpt = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="epoch-1",
            writer=lambda p: _write(p, b"weights"),
        )
        # Corrupt it
        (ws.path / ckpt.artifact_path).unlink()

        with pytest.raises(CheckpointRecoveryError, match="none passed integrity verification"):
            restore_checkpoint(run_id=run.id, workspace=ws, repository=repo)
    finally:
        repo.close()


def test_recovery_no_checkpoints_raises_not_found(tmp_path: Path) -> None:
    """When no checkpoints exist for a run, CheckpointNotFoundError is raised."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("none-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-empty", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        with pytest.raises(CheckpointNotFoundError, match="Checkpoint 'latest' not found"):
            restore_checkpoint(run_id=run.id, workspace=ws, repository=repo)
    finally:
        repo.close()


# =============================================================================
# 4. Public ActiveRun API Integration Tests
# =============================================================================


def test_active_run_save_and_restore_checkpoint(tmp_path: Path) -> None:
    """Test standard public ActiveRun checkpoint workflow."""
    exp = runvault.create_experiment("public-api-exp", workspace=tmp_path)

    weights_v1 = b"pytorch_state_dict_mock_v1"
    weights_v2 = b"pytorch_state_dict_mock_v2"

    c1: Checkpoint | None = None
    c2: Checkpoint | None = None

    with exp.start_run(name="training-run") as run:
        # Save first checkpoint
        c1 = run.save_checkpoint(
            name="epoch-1",
            writer=lambda p: _write(p, weights_v1),
            step=1,
            metadata={"val_loss": 0.45},
        )
        assert c1.name == "epoch-1"
        assert c1.step == 1

        # Save second checkpoint
        c2 = run.save_checkpoint(
            name="epoch-2",
            writer=lambda p: _write(p, weights_v2),
            step=2,
            metadata={"val_loss": 0.32},
        )
        assert c2.name == "epoch-2"
        assert c2.step == 2

        # List checkpoints
        all_ckpts = run.list_checkpoints()
        assert len(all_ckpts) == 2
        assert [c.name for c in all_ckpts] == ["epoch-1", "epoch-2"]

        # Restore latest checkpoint directly
        restored_path = run.restore_checkpoint()
        assert restored_path.read_bytes() == weights_v2
        assert run.last_restored_is_fallback is False
        assert run.last_restored_checkpoint is not None
        assert run.last_restored_checkpoint.id == c2.id

        # Restore specific checkpoint
        assert c1 is not None
        c1_path = run.restore_checkpoint(c1.id)
        assert c1_path.read_bytes() == weights_v1

    # Recovery remains valid after run finishes
    finished_run = exp.get_run(run.id)
    assert finished_run is not None
    assert finished_run.status == RunStatus.FINISHED

    # Using top-level SDK function to restore from finished run
    assert c2 is not None
    recovered_ckpt, rec_path, _ = runvault.restore_checkpoint(run.id, workspace=tmp_path)
    assert recovered_ckpt.id == c2.id
    assert rec_path.read_bytes() == weights_v2

    runvault.reset_repository_cache()


def test_terminal_run_rejects_save_checkpoint(tmp_path: Path) -> None:
    """Cannot save checkpoints to a completed or terminated run."""
    exp = runvault.create_experiment("term-ckpt-exp", workspace=tmp_path)
    with exp.start_run(name="finished-run") as run:
        pass

    assert run.status == RunStatus.FINISHED

    with pytest.raises(TerminalRunError, match="terminal state"):
        run.save_checkpoint(
            name="late-epoch",
            writer=lambda p: _write(p, b"data"),
        )

    runvault.reset_repository_cache()


def test_checkpoint_creation_failure_preserves_run_status(tmp_path: Path) -> None:
    """A checkpoint failure does not cause the active run to become terminal or fail."""
    exp = runvault.create_experiment("non-fatal-exp", workspace=tmp_path)
    with exp.start_run(name="resilient-run") as run:
        assert run.status == RunStatus.RUNNING

        # Attempt to save invalid empty checkpoint
        with pytest.raises(CheckpointCreationError):
            run.save_checkpoint(
                name="bad-empty",
                writer=lambda p: p.touch(),
            )

        # Run MUST remain RUNNING and fully operational
        assert run.status == RunStatus.RUNNING

        # Subsequent logging works
        run.log_metric("loss", 0.5, step=1)
        run.log_param("lr", 0.01)

        # Subsequent valid checkpoint save succeeds
        c = run.save_checkpoint(
            name="good-epoch",
            writer=lambda p: _write(p, b"valid state"),
            step=1,
        )
        assert c.name == "good-epoch"

    assert run.status == RunStatus.FINISHED
    runvault.reset_repository_cache()


# =============================================================================
# 5. Cascading Deletion & Foreign Key Enforcement
# =============================================================================


def test_cascading_deletion_removes_checkpoint_records(tmp_path: Path) -> None:
    """Deleting a run cascades to remove its checkpoint metadata from SQLite."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("cascade-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-casc", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        ckpt = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="v1",
            writer=lambda p: _write(p, b"test"),
        )
        assert repo.get_checkpoint(ckpt.id) is not None

        # Delete the run: checkpoint records must be purged
        repo.delete_run(run.id)
        assert repo.get_checkpoint(ckpt.id) is None
        assert repo.list_checkpoints(run.id) == []
    finally:
        repo.close()


# =============================================================================
# 6. Concurrency Tests
# =============================================================================


def test_concurrent_checkpoint_creation_same_run(tmp_path: Path) -> None:
    """Multiple threads concurrently saving checkpoints for the same run execute safely."""
    exp = runvault.create_experiment("concurrent-ckpt-exp", workspace=tmp_path)
    run = exp.start_run(name="thread-run")
    run.start()

    num_threads = 5

    def worker(idx: int) -> str:
        content = f"thread_{idx}_weights".encode()
        ckpt = run.save_checkpoint(
            name=f"epoch-{idx}",
            writer=lambda p: _write(p, content),
            step=idx,
            metadata={"thread": idx},
        )
        return ckpt.id

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        ckpt_ids = list(executor.map(worker, range(num_threads)))

    assert len(ckpt_ids) == num_threads
    assert len(set(ckpt_ids)) == num_threads

    checkpoints = run.list_checkpoints()
    assert len(checkpoints) == num_threads

    # Verify each checkpoint file is completely intact
    for ckpt in checkpoints:
        path = run.restore_checkpoint(ckpt.id)
        expected = f"thread_{ckpt.step}_weights".encode()
        assert path.read_bytes() == expected

    run.finish()
    runvault.reset_repository_cache()


def test_concurrent_readers_during_checkpoint_publication(tmp_path: Path) -> None:
    """Readers querying checkpoints while a new checkpoint is published never see partial state."""
    exp = runvault.create_experiment("read-write-race-exp", workspace=tmp_path)
    run = exp.start_run(name="writer-run")
    run.start()

    # Initial valid checkpoint
    run.save_checkpoint(name="init", writer=lambda p: _write(p, b"initial state"), step=0)

    reader_results: list[Any] = []

    def reader_worker() -> None:
        for _ in range(10):
            ckpt = run.get_latest_checkpoint()
            if ckpt is not None:
                reader_results.append(ckpt.name)

    def writer_worker() -> None:
        for i in range(1, 4):
            run.save_checkpoint(
                name=f"step-{i}",
                writer=lambda p, val=i: _write(p, f"state_{val}".encode()),
                step=i,
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        f_read = [executor.submit(reader_worker) for _ in range(2)]
        f_write = executor.submit(writer_worker)

        f_write.result()
        for f in f_read:
            f.result()

    assert len(reader_results) > 0
    # Every read checkpoint was an intact committed name
    assert all(name.startswith("init") or name.startswith("step-") for name in reader_results)

    run.finish()
    runvault.reset_repository_cache()


# =============================================================================
# 7. Phase 3.1 Hardening & Durability Verification Tests
# =============================================================================


def test_immutable_atomic_publication_collision_prevention(tmp_path: Path) -> None:
    """Concurrent publication attempts to the exact same ID must not overwrite the artifact."""
    exp = runvault.create_experiment("immut-race-exp", workspace=tmp_path)
    run = exp.start_run(name="collision-run")
    run.start()

    shared_ckpt_id = "deterministic-collision-id"
    barrier = threading.Barrier(4)
    successes: list[str] = []
    failures: list[Exception] = []

    def racer(thread_id: int) -> None:
        barrier.wait()
        try:
            ckpt = run.save_checkpoint(
                name=f"racer-{thread_id}",
                writer=lambda p, tid=thread_id: _write(p, f"content_{tid}".encode()),
                checkpoint_id=shared_ckpt_id,
            )
            successes.append(ckpt.name)
        except CheckpointCreationError as err:
            failures.append(err)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(racer, i) for i in range(4)]
        for f in futures:
            f.result()

    # Exactly one publication must succeed, all others must fail
    assert len(successes) == 1
    assert len(failures) == 3
    for err in failures:
        assert "immutability violated" in str(err) or "already exists" in str(err)

    # Artifact must be completely intact and match the winner
    result = run.restore_checkpoint(shared_ckpt_id)
    assert result.read_bytes().startswith(b"content_")

    run.finish()
    runvault.reset_repository_cache()


def test_checkpoint_chronology_step_100_then_50(tmp_path: Path) -> None:
    """Chronological order is determined by commit sequence, not training step."""
    exp = runvault.create_experiment("chrono-step-exp", workspace=tmp_path)
    with exp.start_run(name="non-monotonic-run") as run:
        # Step 100 committed first
        c1 = run.save_checkpoint(
            name="older-step-100",
            writer=lambda p: _write(p, b"weights_step_100"),
            step=100,
        )
        assert c1.commit_seq is not None

        # Step 50 committed second
        c2 = run.save_checkpoint(
            name="newer-step-50",
            writer=lambda p: _write(p, b"weights_step_50"),
            step=50,
        )
        assert c2.commit_seq is not None
        assert c2.commit_seq > c1.commit_seq

        # Latest checkpoint MUST be newer-step-50 (highest commit_seq, not highest step)
        latest = run.get_latest_checkpoint()
        assert latest is not None
        assert latest.name == "newer-step-50"
        assert latest.step == 50

        # Restoration recovers newer-step-50
        restored = run.restore_checkpoint()
        assert restored.checkpoint.name == "newer-step-50"
        assert restored.read_bytes() == b"weights_step_50"

        # list_checkpoints preserves chronological commit order
        listed = run.list_checkpoints()
        assert [c.name for c in listed] == ["older-step-100", "newer-step-50"]

    runvault.reset_repository_cache()


def test_checkpoint_chronology_identical_steps(tmp_path: Path) -> None:
    """Multiple checkpoints with identical steps are ordered deterministically by commit_seq."""
    exp = runvault.create_experiment("same-step-exp", workspace=tmp_path)
    with exp.start_run(name="repeat-step-run") as run:
        c1 = run.save_checkpoint(name="eval-1", writer=lambda p: _write(p, b"data1"), step=10)
        c2 = run.save_checkpoint(name="eval-2", writer=lambda p: _write(p, b"data2"), step=10)
        c3 = run.save_checkpoint(name="eval-3", writer=lambda p: _write(p, b"data3"), step=10)

        assert c1.commit_seq is not None and c2.commit_seq is not None and c3.commit_seq is not None
        assert c1.commit_seq < c2.commit_seq < c3.commit_seq

        latest = run.get_latest_checkpoint()
        assert latest is not None
        assert latest.name == "eval-3"

    runvault.reset_repository_cache()


def test_checkpoint_chronology_missing_steps(tmp_path: Path) -> None:
    """Checkpoints with missing steps (step=None) are chronologically ordered by commit_seq."""
    exp = runvault.create_experiment("none-step-exp", workspace=tmp_path)
    with exp.start_run(name="none-step-run") as run:
        run.save_checkpoint(name="init", writer=lambda p: _write(p, b"w0"), step=None)
        run.save_checkpoint(name="step5", writer=lambda p: _write(p, b"w5"), step=5)
        run.save_checkpoint(name="final", writer=lambda p: _write(p, b"w_final"), step=None)

        latest = run.get_latest_checkpoint()
        assert latest is not None
        assert latest.name == "final"

        listed = run.list_checkpoints()
        assert [c.name for c in listed] == ["init", "step5", "final"]

    runvault.reset_repository_cache()


def test_checkpoint_chronology_identical_timestamps(tmp_path: Path) -> None:
    """Identical timestamps are deterministically resolved by commit_seq."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("same-ts-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-ts", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        same_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        ckpt_a = Checkpoint(
            id="ca",
            run_id=run.id,
            name="a",
            artifact_path=f"checkpoints/{run.id}/ca.bin",
            size_bytes=5,
            checksum_sha256=hashlib.sha256(b"alpha").hexdigest(),
            created_at=same_time,
        )
        ckpt_b = Checkpoint(
            id="cb",
            run_id=run.id,
            name="b",
            artifact_path=f"checkpoints/{run.id}/cb.bin",
            size_bytes=4,
            checksum_sha256=hashlib.sha256(b"beta").hexdigest(),
            created_at=same_time,
        )

        # Write artifacts
        target_dir = ws.get_run_checkpoints_dir(run.id)
        (target_dir / "ca.bin").write_bytes(b"alpha")
        (target_dir / "cb.bin").write_bytes(b"beta")

        repo.create_checkpoint(ckpt_a)
        repo.create_checkpoint(ckpt_b)

        latest = repo.get_latest_checkpoint(run.id)
        assert latest is not None
        assert latest.name == "b"
        assert latest.id == "cb"
    finally:
        repo.close()


def test_explicit_restoration_specific_id_does_not_fallback(tmp_path: Path) -> None:
    """Explicitly requesting a specific checkpoint ID must fail if corrupt, never falling back."""
    exp = runvault.create_experiment("explicit-no-fb-exp", workspace=tmp_path)
    with exp.start_run(name="strict-run") as run:
        c1 = run.save_checkpoint(name="valid-older", writer=lambda p: _write(p, b"v1_data"), step=1)
        c2 = run.save_checkpoint(name="tampered", writer=lambda p: _write(p, b"v2_data"), step=2)

        # Corrupt c2
        ws = Workspace.from_path(tmp_path)
        (ws.path / c2.artifact_path).write_bytes(b"tampered_bad_data")

        # Explicit restoration of c2 MUST raise CheckpointCorruptError
        with pytest.raises(
            CheckpointCorruptError, match=r"Artifact size mismatch|SHA-256 mismatch"
        ):
            run.restore_checkpoint(c2.id)

        # Implicit latest recovery falls back to c1
        res = run.restore_checkpoint()
        assert res.checkpoint.id == c1.id
        assert res.is_fallback is True
        assert len(res.rejected_checkpoints) == 1

    runvault.reset_repository_cache()


def test_explicit_restoration_unsupported_format(tmp_path: Path) -> None:
    """Restoration of a checkpoint with future format version raises unsupported error."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("fmt-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-fmt", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        target_dir = ws.get_run_checkpoints_dir(run.id)
        (target_dir / "future.bin").write_bytes(b"future_data")

        future_ckpt = Checkpoint(
            id="future-ckpt",
            run_id=run.id,
            name="future-epoch",
            artifact_path=f"checkpoints/{run.id}/future.bin",
            size_bytes=11,
            checksum_sha256=hashlib.sha256(b"future_data").hexdigest(),
            format_version=99,
        )
        repo.create_checkpoint(future_ckpt)

        with pytest.raises(
            UnsupportedCheckpointFormatError, match="format version 99 is unsupported"
        ):
            restore_checkpoint(
                run_id=run.id, workspace=ws, repository=repo, checkpoint_id="future-ckpt"
            )
    finally:
        repo.close()


def test_recovery_result_object_and_pathlike(tmp_path: Path) -> None:
    """CheckpointRecoveryResult provides rich properties, path-like access, and tuple unpacking."""
    exp = runvault.create_experiment("res-obj-exp", workspace=tmp_path)
    with exp.start_run(name="res-run") as run:
        run.save_checkpoint(name="e1", writer=lambda p: _write(p, b"epoch 1 data"), step=1)
        c2 = run.save_checkpoint(name="e2", writer=lambda p: _write(p, b"epoch 2 data"), step=2)

        # Implicit recovery
        result = run.restore_checkpoint()
        assert isinstance(result, CheckpointRecoveryResult)
        assert result.checkpoint.id == c2.id
        assert result.is_fallback is False
        assert result.rejected_checkpoints == ()

        # Direct read helpers
        assert result.read_bytes() == b"epoch 2 data"
        assert result.read_text() == "epoch 2 data"

        # os.PathLike integration
        assert os.fspath(result) == str(result.path)
        assert Path(result) == result.path

        # Tuple unpacking
        ckpt, path, is_fb = result
        assert ckpt.id == c2.id
        assert path == result.path
        assert is_fb is False

    runvault.reset_repository_cache()


def test_orphan_artifact_policy_unindexed_bin_ignored(tmp_path: Path) -> None:
    """Unindexed .bin files on disk are ignored and never returned as checkpoints."""
    exp = runvault.create_experiment("orphan-exp", workspace=tmp_path)
    with exp.start_run(name="orphan-run") as run:
        # Legitimate checkpoint
        c = run.save_checkpoint(name="legit", writer=lambda p: _write(p, b"legit"), step=1)

        # Drop an unindexed rogue .bin file into checkpoints directory
        target_dir = Workspace.from_path(tmp_path).get_run_checkpoints_dir(run.id)
        rogue_path = target_dir / "rogue_orphan.bin"
        rogue_path.write_bytes(b"rogue_model_weights")

        # Queries must only return the legitimate committed checkpoint
        all_ckpts = run.list_checkpoints()
        assert len(all_ckpts) == 1
        assert all_ckpts[0].id == c.id

        latest = run.get_latest_checkpoint()
        assert latest is not None
        assert latest.id == c.id

    runvault.reset_repository_cache()


def test_security_path_traversal_adversarial_rejections(tmp_path: Path) -> None:
    """Adversarial attempts to use path traversal in checkpoint identifiers are rejected."""
    exp = runvault.create_experiment("security-exp", workspace=tmp_path)
    with exp.start_run(name="sec-run") as run:
        with pytest.raises(
            CheckpointCreationError, match="cannot contain path separators or traversal"
        ):
            run.save_checkpoint(
                name="traversal",
                writer=lambda p: _write(p, b"data"),
                checkpoint_id="../../escaped",
            )

        with pytest.raises(
            CheckpointCreationError, match="cannot contain path separators or traversal"
        ):
            run.save_checkpoint(
                name="abs-traversal",
                writer=lambda p: _write(p, b"data"),
                checkpoint_id="/etc/evil",
            )

    # Domain validation directly blocks traversal in artifact paths
    with pytest.raises(ValueError, match="cannot contain traversal"):
        Checkpoint(
            id="corrupt-meta",
            run_id="run-safe",
            name="escaped",
            artifact_path="../../../etc/passwd",
            size_bytes=10,
            checksum_sha256=hashlib.sha256(b"1234567890").hexdigest(),
        )

    # Artifact path outside checkpoints root fails containment check
    ws = Workspace.from_path(tmp_path)
    escaped_ckpt = Checkpoint(
        id="corrupt-meta",
        run_id="run-safe",
        name="escaped",
        artifact_path="arbitrary_folder/outside.bin",
        size_bytes=10,
        checksum_sha256=hashlib.sha256(b"1234567890").hexdigest(),
    )
    with pytest.raises(CheckpointCorruptError, match="escapes checkpoints root"):
        resolve_and_validate_artifact_path(ws, escaped_ckpt)

    runvault.reset_repository_cache()


def test_security_symlink_artifact_rejected(tmp_path: Path) -> None:
    """Artifact files that are symlinks are rejected for security."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("symlink-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-sym", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        real_target = tmp_path / "secret_file.txt"
        real_target.write_bytes(b"sensitive secret content")

        target_dir = ws.get_run_checkpoints_dir(run.id)
        symlink_artifact = target_dir / "symlink_ckpt.bin"
        try:
            symlink_artifact.symlink_to(real_target)
        except OSError:
            pytest.skip("Symlinks not supported on this OS/filesystem")

        ckpt = Checkpoint(
            id="symlink-ckpt",
            run_id=run.id,
            name="symlink-epoch",
            artifact_path=f"checkpoints/{run.id}/symlink_ckpt.bin",
            size_bytes=len(b"sensitive secret content"),
            checksum_sha256=hashlib.sha256(b"sensitive secret content").hexdigest(),
        )
        repo.create_checkpoint(ckpt)

        # Verification must reject symlink
        is_valid, reason = verify_checkpoint(ckpt, ws)
        assert is_valid is False
        assert "symlink" in str(reason).lower()

        # Restoration must reject symlink
        with pytest.raises(CheckpointCorruptError, match="symlink"):
            restore_checkpoint(run_id=run.id, workspace=ws, repository=repo, checkpoint_id=ckpt.id)
    finally:
        repo.close()


def test_unsupported_hard_links_fails_cleanly_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When hard links are unsupported, publication fails cleanly without fallback overwrite."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("no-link-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-nolink", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        target_dir = ws.get_run_checkpoints_dir(run.id)
        final_target = target_dir / "target_ckpt.bin"

        def mock_link(_src: Any, _dst: Any) -> None:
            raise OSError(errno.EOPNOTSUPP, "Operation not supported on filesystem")

        monkeypatch.setattr(os, "link", mock_link)

        with pytest.raises(CheckpointCreationError, match="Exclusive atomic publication failed"):
            publish_checkpoint(
                workspace=ws,
                repository=repo,
                run_id=run.id,
                name="failed-link",
                writer=lambda p: _write(p, b"new_data"),
                checkpoint_id="target_ckpt",
            )

        # Final destination was never created
        assert not final_target.exists()

        # No temporary files remain
        remaining_tmp = list(target_dir.glob("*.tmp"))
        assert remaining_tmp == []

        # No metadata was committed
        assert repo.list_checkpoints(run.id) == []
    finally:
        repo.close()


def test_directory_fsync_failure_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Directory fsync failure rolls back the published artifact and does not commit metadata."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("dir-sync-fail-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-dsync", experiment_id=exp.id, status=RunStatus.RUNNING)
        )
        target_dir = ws.get_run_checkpoints_dir(run.id)

        def mock_fsync_dir(_dir_path: Path) -> None:
            raise OSError(errno.EIO, "Simulated disk I/O error during directory fsync")

        monkeypatch.setattr(checkpoints_mod, "_fsync_directory", mock_fsync_dir)

        with pytest.raises(
            CheckpointCreationError, match="Failed to synchronize checkpoint directory"
        ):
            publish_checkpoint(
                workspace=ws,
                repository=repo,
                run_id=run.id,
                name="ep-sync-fail",
                writer=lambda p: _write(p, b"data"),
                checkpoint_id="sync_fail_ckpt",
            )

        # Published artifact was cleaned up
        assert not (target_dir / "sync_fail_ckpt.bin").exists()

        # No metadata committed in SQLite
        assert repo.list_checkpoints(run.id) == []
    finally:
        repo.close()


def test_file_synchronization_prefers_fdatasync_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When os.fdatasync is available, it is prioritized to sync file payload buffers."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("sync-fdatasync-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-sync-fdatasync", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        fdatasync_called = False

        def mock_fdatasync(fd: int) -> None:
            nonlocal fdatasync_called
            fdatasync_called = True
            os.fsync(fd)

        monkeypatch.setattr(os, "fdatasync", mock_fdatasync, raising=False)

        ckpt = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="ep-fdatasync",
            writer=lambda p: _write(p, b"fdatasync data"),
        )
        assert fdatasync_called is True
        assert ckpt.size_bytes == len(b"fdatasync data")
    finally:
        repo.close()


def test_file_synchronization_fallback_to_fsync_when_fdatasync_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When os.fdatasync is not available (e.g. Windows), falls back cleanly to os.fsync."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("sync-fsync-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-sync-fsync", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        # Simulate platform without fdatasync (e.g. Windows)
        monkeypatch.delattr(os, "fdatasync", raising=False)

        fsync_calls: list[int] = []
        original_fsync = os.fsync

        def mock_fsync(fd: int) -> None:
            fsync_calls.append(fd)
            original_fsync(fd)

        monkeypatch.setattr(os, "fsync", mock_fsync)

        ckpt = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="ep-fsync-fallback",
            writer=lambda p: _write(p, b"fsync fallback data"),
        )
        assert len(fsync_calls) > 0
        assert ckpt.size_bytes == len(b"fsync fallback data")

        # Verify checkpoint can be restored
        result = restore_checkpoint(run.id, ws, repo, ckpt.id)
        assert result.read_bytes() == b"fsync fallback data"
    finally:
        repo.close()


def test_file_synchronization_failure_prevents_metadata_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When file descriptor sync fails, creation error is raised and no metadata is committed."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("sync-fail-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-sync-fail", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        def mock_sync_fd(_fd: int) -> None:
            raise OSError(errno.EIO, "Simulated disk flush I/O failure")

        monkeypatch.setattr(checkpoints_mod, "_sync_file_descriptor", mock_sync_fd)

        with pytest.raises(
            CheckpointCreationError, match="Failed to flush artifact to disk"
        ) as exc_info:
            publish_checkpoint(
                workspace=ws,
                repository=repo,
                run_id=run.id,
                name="ep-flush-fail",
                writer=lambda p: _write(p, b"unflushed data"),
                checkpoint_id="unflushed_ckpt",
            )

        assert "Simulated disk flush I/O failure" in str(exc_info.value)
        assert not (ws.get_run_checkpoints_dir(run.id) / "unflushed_ckpt.bin").exists()
        assert repo.list_checkpoints(run.id) == []
    finally:
        repo.close()


def test_staging_hard_link_cleanup_failure_records_diagnostic_and_preserves_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When staging link cleanup fails after link(2), diagnostic is saved and commit succeeds."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("staging-cln-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-staging-cln", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        original_unlink = Path.unlink
        unlink_count = 0

        # Fail unlink during post-publication cleanup
        def mock_unlink(path_obj: Path, missing_ok: bool = False) -> None:
            nonlocal unlink_count
            if str(path_obj).endswith(".tmp"):
                unlink_count += 1
                if unlink_count > 1:
                    raise OSError(errno.EACCES, "Permission denied removing temporary staging link")
            original_unlink(path_obj, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", mock_unlink)

        ckpt = publish_checkpoint(
            workspace=ws,
            repository=repo,
            run_id=run.id,
            name="ep-staging-warn",
            writer=lambda p: _write(p, b"committed checkpoint payload"),
            metadata={"user_param": "alpha"},
            checkpoint_id="stg_warn_ckpt",
        )

        # Checkpoint committed in SQLite
        assert ckpt.id == "stg_warn_ckpt"
        assert ckpt.metadata.get("user_param") == "alpha"
        assert "_staging_cleanup_warning" in ckpt.metadata
        assert (
            "Permission denied removing temporary staging link"
            in ckpt.metadata["_staging_cleanup_warning"]
        )

        # Checkpoint is readable and verifiable
        result = restore_checkpoint(run.id, ws, repo, "stg_warn_ckpt")
        assert result.read_bytes() == b"committed checkpoint payload"
    finally:
        repo.close()


def test_sqlite_commit_failure_cleans_up_artifact_and_preserves_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When SQLite commit fails and artifact cleanup also fails, primary DB error is preserved."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("db-fail-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-db-fail", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        def mock_create_checkpoint(_self: Any, _ckpt: Any) -> Any:
            raise sqlite3.OperationalError("database is locked (simulated)")

        original_unlink = Path.unlink

        def mock_unlink(path_obj: Path, missing_ok: bool = False) -> None:
            if path_obj.name == "db_fail_ckpt.bin":
                raise OSError(errno.EPERM, "Permission denied unlinking published artifact")
            original_unlink(path_obj, missing_ok=missing_ok)

        monkeypatch.setattr(SQLiteMetadataRepository, "create_checkpoint", mock_create_checkpoint)
        monkeypatch.setattr(Path, "unlink", mock_unlink)

        with pytest.raises(
            CheckpointCreationError, match="Failed to persist checkpoint metadata in SQLite"
        ) as exc_info:
            publish_checkpoint(
                workspace=ws,
                repository=repo,
                run_id=run.id,
                name="ep-db-fail",
                writer=lambda p: _write(p, b"data"),
                checkpoint_id="db_fail_ckpt",
            )

        err = exc_info.value
        assert "database is locked (simulated)" in str(err)
        assert isinstance(err.original_error, sqlite3.OperationalError)
        assert isinstance(err.__cause__, sqlite3.OperationalError)
        notes = getattr(err, "__notes__", [])
        assert any("Failed to cleanup uncommitted artifact" in n for n in notes)
        assert any("Permission denied unlinking published artifact" in n for n in notes)

        # No metadata committed
        assert repo.list_checkpoints(run.id) == []
    finally:
        repo.close()


def test_writer_callback_cleanup_failure_preserves_primary_writer_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When writer callback fails and staging cleanup fails, writer error and notes are kept."""
    ws = Workspace.from_path(tmp_path)
    repo = SQLiteMetadataRepository(ws.db_path)
    try:
        exp = runvault.create_experiment("writer-fail-exp", workspace=tmp_path)
        run = repo.create_run(
            runvault.Run(id="run-writer-fail", experiment_id=exp.id, status=RunStatus.RUNNING)
        )

        writer_failed = False

        def failing_writer(_p: Path) -> None:
            nonlocal writer_failed
            writer_failed = True
            raise RuntimeError("Custom model serialization exploded")

        original_unlink = Path.unlink

        def mock_unlink(path_obj: Path, missing_ok: bool = False) -> None:
            if writer_failed and str(path_obj).endswith(".tmp"):
                raise OSError(errno.EACCES, "Permission denied removing temporary staging file")
            original_unlink(path_obj, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", mock_unlink)

        with pytest.raises(
            CheckpointCreationError,
            match="Writer callback failed: Custom model serialization exploded",
        ) as exc_info:
            publish_checkpoint(
                workspace=ws,
                repository=repo,
                run_id=run.id,
                name="ep-writer-fail",
                writer=failing_writer,
                checkpoint_id="writer_fail_ckpt",
            )

        err = exc_info.value
        assert isinstance(err.original_error, RuntimeError)
        assert isinstance(err.__cause__, RuntimeError)
        notes = getattr(err, "__notes__", [])
        assert any("Failed to clean up staging file" in n for n in notes)
        assert any("Permission denied removing temporary staging file" in n for n in notes)

        # No metadata committed
        assert repo.list_checkpoints(run.id) == []
    finally:
        repo.close()
