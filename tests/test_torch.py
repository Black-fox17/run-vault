"""Deterministic offline tests for PyTorch integration (Phase 4.3)."""

from __future__ import annotations

import io
import random
import subprocess
import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import torch
from torch import nn, optim

try:
    import numpy as np
except ImportError:
    np = None

import runvault
from runvault.core.errors import (
    CheckpointCorruptError,
    UnsupportedCheckpointFormatError,
)
from runvault.core.workspace import Workspace
from runvault.integrations.gdrive import GoogleDriveStorageProvider
from runvault.integrations.torch import (
    TorchCheckpointManager,
    capture_rng_state,
    check_torch_dependencies,
    deserialize_torch_checkpoint,
    is_torch_available,
    restore_rng_state,
    restore_torch_checkpoint,
    save_torch_checkpoint,
)
from tests.test_gdrive import FakeDriveService

if TYPE_CHECKING:
    from pathlib import Path


class TinyLinearModel(nn.Module):
    """Simple deterministic linear model for testing state persistence."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)
        # Initialize deterministic weights
        with torch.no_grad():
            self.fc.weight.fill_(1.5)
            self.fc.bias.fill_(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_import_runvault_does_not_import_torch() -> None:
    """Core runvault imports must never import torch into sys.modules."""
    code = (
        "import sys\n"
        "import runvault\n"
        "assert 'torch' not in sys.modules, f'torch was imported: {sys.modules[\"torch\"]}'\n"
        "print('SUCCESS')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Import test failed:\n{result.stderr}"
    assert "SUCCESS" in result.stdout


def test_torch_dependency_check_missing() -> None:
    """check_torch_dependencies raises actionable ImportError when torch is missing."""
    with (
        patch("importlib.import_module", side_effect=ImportError("No module named 'torch'")),
        pytest.raises(ImportError, match=r"pip install 'runvault-ai\[torch\]'"),
    ):
        check_torch_dependencies()


def test_torch_is_available() -> None:
    """is_torch_available returns True in environment with torch installed."""
    assert is_torch_available() is True


def test_save_and_restore_model_state(tmp_path: Path) -> None:
    """Model state dict is saved and restored byte-for-byte."""
    model = TinyLinearModel()

    # Modify weights
    with torch.no_grad():
        model.fc.weight.fill_(3.1415)
        model.fc.bias.fill_(2.718)

    with runvault.init("torch-experiment", name="model-save-run", workspace=tmp_path) as run:
        ckpt = save_torch_checkpoint(
            run,
            name="initial-weights",
            model=model,
            epoch=1,
            step=10,
            best_metric=0.42,
        )
        assert ckpt.name == "initial-weights"
        assert ckpt.step == 10

    # Create new model with different weights
    new_model = TinyLinearModel()
    with torch.no_grad():
        new_model.fc.weight.zero_()
        new_model.fc.bias.zero_()

    # Restore into new model
    res = restore_torch_checkpoint(run, new_model)
    assert res.epoch == 1
    assert res.step == 10
    assert res.best_metric == 0.42
    assert res.is_fallback is False

    # Verify weights were restored
    assert torch.allclose(new_model.fc.weight, torch.tensor(3.1415))
    assert torch.allclose(new_model.fc.bias, torch.tensor(2.718))


def test_save_and_restore_optimizer_and_scheduler(tmp_path: Path) -> None:
    """Optimizer state (buffers) and scheduler state are restored accurately."""
    model = TinyLinearModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # Perform one optimization step to populate optimizer state buffers (exp_avg, exp_avg_sq)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    with runvault.init("torch-experiment", name="opt-sched-run", workspace=tmp_path) as run:
        save_torch_checkpoint(
            run,
            name="opt-step-1",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            step=1,
        )

    # Instantiate fresh optimizer and scheduler
    fresh_model = TinyLinearModel()
    fresh_opt = optim.Adam(fresh_model.parameters(), lr=0.1)  # Different lr
    fresh_sched = optim.lr_scheduler.StepLR(fresh_opt, step_size=2, gamma=0.5)

    res = restore_torch_checkpoint(
        run,
        fresh_model,
        optimizer=fresh_opt,
        scheduler=fresh_sched,
    )
    assert res.step == 1
    # Restored optimizer learning rate should match saved rate (0.01)
    assert fresh_opt.param_groups[0]["lr"] == 0.01
    # Check that state buffers (momentum/variance) were populated
    param = next(iter(fresh_model.parameters()))
    assert param in fresh_opt.state
    assert "exp_avg" in fresh_opt.state[param]


def test_save_and_restore_amp_grad_scaler(tmp_path: Path) -> None:
    """AMP GradScaler state is saved and restored."""
    model = TinyLinearModel()
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    with runvault.init("torch-experiment", name="amp-run", workspace=tmp_path) as run:
        save_torch_checkpoint(
            run,
            name="amp-ckpt",
            model=model,
            scaler=scaler,
            epoch=2,
        )

    fresh_model = TinyLinearModel()
    fresh_scaler = torch.amp.GradScaler("cpu", enabled=False)

    res = restore_torch_checkpoint(run, fresh_model, scaler=fresh_scaler)
    assert res.epoch == 2
    assert fresh_scaler.state_dict() == scaler.state_dict()


def test_rng_capture_and_restoration() -> None:
    """RNG states across Python, NumPy, and PyTorch CPU are faithfully restored."""
    # 1. Set specific seeds
    random.seed(12345)
    torch.manual_seed(12345)
    if np is not None:
        np.random.seed(12345)

    # Capture initial RNG state
    captured = capture_rng_state()

    # Draw random numbers to advance generators
    py_val_1 = random.random()
    torch_val_1 = torch.rand(1).item()
    np_val_1 = np.random.rand() if np is not None else 0.0

    # Draw a second time to ensure sequence moves forward
    _ = random.random()
    _ = torch.rand(1).item()
    if np is not None:
        _ = np.random.rand()

    # 2. Restore captured state
    restore_rng_state(captured)

    # Values generated immediately after restore should match the first draw
    py_val_restored = random.random()
    torch_val_restored = torch.rand(1).item()
    assert py_val_restored == py_val_1
    assert torch_val_restored == torch_val_1
    if np is not None:
        np_val_restored = np.random.rand()
        assert np_val_restored == np_val_1


def test_cuda_rng_conditional_handling() -> None:
    """CUDA RNG capture and restoration gracefully handle GPU presence or absence."""
    captured = capture_rng_state()
    cuda_state = captured.get("torch_cuda")
    if torch.cuda.is_available():
        assert cuda_state is not None
    else:
        assert cuda_state is None

    # Restoring should not raise even when CUDA is not present
    restore_rng_state(captured)


def test_safe_loading_rejects_malicious_unpickling() -> None:
    """safe_load=True (weights_only=True) rejects payloads containing executable globals."""
    buf = io.BytesIO()

    # Create a malicious payload with arbitrary object execution attempt
    class MaliciousPayload:
        def __reduce__(self) -> Any:
            return (eval, ("1 + 1",))

    torch.save(MaliciousPayload(), buf)
    buf.seek(0)

    # safe_load=True should raise CheckpointCorruptError
    with pytest.raises(CheckpointCorruptError, match="Failed to deserialize PyTorch checkpoint"):
        deserialize_torch_checkpoint(buf, safe_load=True)


def test_malformed_checkpoint_payload_rejection() -> None:
    """deserialize_torch_checkpoint rejects non-dict, missing format_version, and corrupted data."""
    # 1. Non-dict payload
    buf_int = io.BytesIO()
    torch.save(12345, buf_int)
    buf_int.seek(0)
    with pytest.raises(CheckpointCorruptError, match="Expected checkpoint payload to be a dict"):
        deserialize_torch_checkpoint(buf_int)

    # 2. Unsupported future format version
    buf_version = io.BytesIO()
    torch.save({"format_version": 999, "model_state_dict": {}}, buf_version)
    buf_version.seek(0)
    with pytest.raises(UnsupportedCheckpointFormatError):
        deserialize_torch_checkpoint(buf_version)

    # 3. Missing model_state_dict
    buf_missing = io.BytesIO()
    torch.save({"format_version": 1}, buf_missing)
    buf_missing.seek(0)
    with pytest.raises(CheckpointCorruptError, match="missing required 'model_state_dict'"):
        deserialize_torch_checkpoint(buf_missing)


def test_torch_checkpoint_manager_lifecycle(tmp_path: Path) -> None:
    """TorchCheckpointManager saves at step and epoch intervals and records metrics."""
    model = TinyLinearModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    with runvault.init("torch-experiment", name="manager-run", workspace=tmp_path) as run:
        mgr = TorchCheckpointManager(
            run=run,
            model=model,
            optimizer=optimizer,
            step_interval=2,
            epoch_interval=1,
        )

        # Step 1: should not save (step_interval=2)
        saved_1 = mgr.step_end(1, metrics={"loss": 1.5})
        assert saved_1 is None

        # Step 2: should save checkpoint
        saved_2 = mgr.step_end(2, metrics={"loss": 1.2})
        assert saved_2 is not None
        assert saved_2.name == "step-2"

        # Epoch 1: should save checkpoint
        saved_epoch = mgr.epoch_end(1, step=2, metrics={"val_loss": 0.8}, best_metric=0.8)
        assert saved_epoch is not None
        assert saved_epoch.name == "epoch-1"

    # Restore via manager
    fresh_model = TinyLinearModel()
    fresh_mgr = TorchCheckpointManager(run=run, model=fresh_model)
    res = fresh_mgr.restore()
    assert res.epoch == 1
    assert res.step == 2
    assert res.best_metric == 0.8


def test_restore_fallback_on_corrupt_torch_checkpoint(tmp_path: Path) -> None:
    """When latest checkpoint artifact is corrupted, recovery falls back to older checkpoint."""
    model = TinyLinearModel()
    ckpt2_id: str | None = None

    with runvault.init("torch-experiment", name="fallback-run", workspace=tmp_path) as run:
        # Checkpoint 1: epoch 1
        with torch.no_grad():
            model.fc.bias.fill_(1.0)
        save_torch_checkpoint(run, "epoch-1", model, epoch=1, step=10)

        # Checkpoint 2: epoch 2
        with torch.no_grad():
            model.fc.bias.fill_(2.0)
        ckpt2 = save_torch_checkpoint(run, "epoch-2", model, epoch=2, step=20)
        ckpt2_id = ckpt2.id

    assert ckpt2_id is not None
    # Corrupt the latest checkpoint file on disk
    recovery_info = run.restore_checkpoint(ckpt2_id)
    assert recovery_info.path is not None
    recovery_info.path.write_bytes(b"CORRUPTED_BYTES_NOT_TORCH")

    # Restore latest checkpoint without specifying ID -> Should fall back to epoch-1!
    fresh_model = TinyLinearModel()
    res = restore_torch_checkpoint(run, fresh_model)
    assert res.is_fallback is True
    assert res.checkpoint.name == "epoch-1"
    assert res.epoch == 1
    assert len(res.rejected_checkpoints) == 1
    assert torch.allclose(fresh_model.fc.bias, torch.tensor(1.0))


def test_explicit_torch_restore_does_not_fallback(tmp_path: Path) -> None:
    """Explicit restore of a corrupted checkpoint fails without falling back."""
    model = TinyLinearModel()
    ckpt2_id: str | None = None

    with runvault.init("torch-experiment", name="explicit-fail-run", workspace=tmp_path) as run:
        save_torch_checkpoint(run, "epoch-1", model, epoch=1)
        ckpt2 = save_torch_checkpoint(run, "epoch-2", model, epoch=2)
        ckpt2_id = ckpt2.id

    assert ckpt2_id is not None
    # Corrupt epoch-2
    recovery_info = run.restore_checkpoint(ckpt2_id)
    assert recovery_info.path is not None
    recovery_info.path.write_bytes(b"CORRUPT")

    fresh_model = TinyLinearModel()
    with pytest.raises(CheckpointCorruptError):
        restore_torch_checkpoint(run, fresh_model, checkpoint_id=ckpt2_id)


def test_torch_integration_with_google_drive_provider(tmp_path: Path) -> None:
    """PyTorch checkpoints stream directly to and from Google Drive storage backend."""
    ws = Workspace.from_path(tmp_path)
    fake_drive = FakeDriveService()
    gdrive_storage = GoogleDriveStorageProvider(drive_service=fake_drive, workspace=ws)
    model = TinyLinearModel()
    with torch.no_grad():
        model.fc.weight.fill_(9.81)

    with runvault.init(
        "torch-gdrive-experiment",
        name="gdrive-run",
        workspace=tmp_path,
        storage_provider=gdrive_storage,
    ) as run:
        ckpt = save_torch_checkpoint(
            run,
            name="remote-torch-ckpt",
            model=model,
            epoch=5,
            step=500,
            best_metric=0.15,
        )
        assert ckpt.storage_provider == "gdrive"
        assert ckpt.storage_key in fake_drive.files_db

    # Restore directly from Google Drive stream into new model
    fresh_model = TinyLinearModel()
    with torch.no_grad():
        fresh_model.fc.weight.zero_()

    res = restore_torch_checkpoint(
        run,
        fresh_model,
        storage=gdrive_storage,
    )
    assert res.epoch == 5
    assert res.step == 500
    assert res.best_metric == 0.15
    assert torch.allclose(fresh_model.fc.weight, torch.tensor(9.81))


def test_restore_by_string_run_id(tmp_path: Path) -> None:
    """restore_torch_checkpoint supports string run_id outside of active run context."""
    model = TinyLinearModel()
    with torch.no_grad():
        model.fc.weight.fill_(42.0)

    run_id: str | None = None
    with runvault.init("string-run-exp", name="str-run", workspace=tmp_path) as run:
        save_torch_checkpoint(run, "ckpt-1", model, epoch=3)
        run_id = run.id

    assert run_id is not None
    fresh_model = TinyLinearModel()
    res = restore_torch_checkpoint(run_id, fresh_model, workspace=tmp_path)
    assert res.epoch == 3
    assert torch.allclose(fresh_model.fc.weight, torch.tensor(42.0))


def test_restore_atomicity_model_rollback_on_optimizer_failure(tmp_path: Path) -> None:
    """If optimizer restoration fails, model weights must be rolled back to original state."""
    saved_model = TinyLinearModel()
    with torch.no_grad():
        saved_model.fc.weight.fill_(7.77)
    saved_opt = optim.AdamW(saved_model.parameters(), lr=1e-3)

    with runvault.init("rollback-exp", name="opt-fail-run", workspace=tmp_path) as run:
        save_torch_checkpoint(run, "ckpt-opt", saved_model, optimizer=saved_opt)

        live_model = TinyLinearModel()
        with torch.no_grad():
            live_model.fc.weight.fill_(1.23)

        class FailingOptimizer(optim.Optimizer):
            def __init__(self) -> None:
                super().__init__(live_model.parameters(), {})

            def load_state_dict(self, state_dict: dict[str, Any]) -> None:
                _ = state_dict
                raise ValueError("Simulated optimizer load failure")

        failing_opt = FailingOptimizer()

        with pytest.raises(ValueError, match="Simulated optimizer load failure"):
            restore_torch_checkpoint(run, live_model, optimizer=failing_opt)

        # Crucial check: model weights MUST have rolled back to 1.23, NOT remain 7.77
        assert torch.allclose(live_model.fc.weight, torch.tensor(1.23))


def test_restore_atomicity_rollback_on_scheduler_failure(tmp_path: Path) -> None:
    """If scheduler restoration fails, both model and optimizer are rolled back."""
    saved_model = TinyLinearModel()
    with torch.no_grad():
        saved_model.fc.weight.fill_(5.55)
    saved_opt = optim.SGD(saved_model.parameters(), lr=0.1)
    saved_sched = optim.lr_scheduler.StepLR(saved_opt, step_size=1)

    with runvault.init("sched-exp", name="sched-fail-run", workspace=tmp_path) as run:
        save_torch_checkpoint(
            run,
            "ckpt-sched",
            saved_model,
            optimizer=saved_opt,
            scheduler=saved_sched,
        )

        live_model = TinyLinearModel()
        with torch.no_grad():
            live_model.fc.weight.fill_(2.22)
        live_opt = optim.SGD(live_model.parameters(), lr=0.01)

        class FailingScheduler:
            def state_dict(self) -> dict[str, Any]:
                return {"step_size": 1}

            def load_state_dict(self, _state_dict: dict[str, Any]) -> None:
                raise RuntimeError("Corrupted scheduler state")

        failing_sched = FailingScheduler()

        with pytest.raises(RuntimeError, match="Corrupted scheduler state"):
            restore_torch_checkpoint(
                run,
                live_model,
                optimizer=live_opt,
                scheduler=failing_sched,
            )

        # Model and optimizer are cleanly rolled back
        assert torch.allclose(live_model.fc.weight, torch.tensor(2.22))
        assert live_opt.param_groups[0]["lr"] == 0.01


def test_restore_atomicity_rollback_on_scaler_failure(tmp_path: Path) -> None:
    """If GradScaler restoration fails, model is rolled back."""
    saved_model = TinyLinearModel()
    with torch.no_grad():
        saved_model.fc.weight.fill_(8.88)

    class DummyScaler:
        def state_dict(self) -> dict[str, Any]:
            return {"scale": 1024.0}

    with runvault.init("scaler-exp", name="scaler-fail-run", workspace=tmp_path) as run:
        save_torch_checkpoint(run, "ckpt-scaler", saved_model, scaler=DummyScaler())

        live_model = TinyLinearModel()
        with torch.no_grad():
            live_model.fc.weight.fill_(3.33)

        class FailingScaler:
            def state_dict(self) -> dict[str, Any]:
                return {"scale": 512.0}

            def load_state_dict(self, _state_dict: dict[str, Any]) -> None:
                raise ValueError("Scaler failure")

        with pytest.raises(ValueError, match="Scaler failure"):
            restore_torch_checkpoint(run, live_model, scaler=FailingScaler())

        assert torch.allclose(live_model.fc.weight, torch.tensor(3.33))


def test_restore_atomicity_rollback_on_rng_failure(tmp_path: Path) -> None:
    """If RNG restoration fails, model is rolled back to snapshot."""
    saved_model = TinyLinearModel()
    with torch.no_grad():
        saved_model.fc.weight.fill_(9.99)

    with runvault.init("rng-fail-exp", name="rng-fail-run", workspace=tmp_path) as run:
        save_torch_checkpoint(run, "ckpt-rng", saved_model, include_rng=True)

        live_model = TinyLinearModel()
        with torch.no_grad():
            live_model.fc.weight.fill_(4.44)

        with (
            patch(
                "runvault.integrations.torch.manager.restore_rng_state",
                side_effect=RuntimeError("RNG hardware failure"),
            ),
            pytest.raises(RuntimeError, match="RNG hardware failure"),
        ):
            restore_torch_checkpoint(run, live_model, restore_rng=True)

        assert torch.allclose(live_model.fc.weight, torch.tensor(4.44))


def test_restore_fails_when_requested_components_missing(tmp_path: Path) -> None:
    """Requesting optimizer, scheduler, or scaler restoration from a model-only checkpoint fails."""
    saved_model = TinyLinearModel()

    with runvault.init("missing-state-exp", name="missing-run", workspace=tmp_path) as run:
        save_torch_checkpoint(run, "ckpt-bare", saved_model)

        live_model = TinyLinearModel()
        with torch.no_grad():
            live_model.fc.weight.fill_(1.11)
        opt = optim.AdamW(live_model.parameters(), lr=1e-3)
        sched = optim.lr_scheduler.StepLR(opt, step_size=1)

        class DummyScaler:
            def load_state_dict(self, state_dict: dict[str, Any]) -> None:
                pass

        # Optimizer requested but absent
        with pytest.raises(ValueError, match="Requested optimizer restoration"):
            restore_torch_checkpoint(run, live_model, optimizer=opt)
        assert torch.allclose(live_model.fc.weight, torch.tensor(1.11))

        # Scheduler requested but absent
        with pytest.raises(ValueError, match="Requested scheduler restoration"):
            restore_torch_checkpoint(run, live_model, scheduler=sched)
        assert torch.allclose(live_model.fc.weight, torch.tensor(1.11))

        # Scaler requested but absent
        with pytest.raises(ValueError, match="Requested scaler restoration"):
            restore_torch_checkpoint(run, live_model, scaler=DummyScaler())
        assert torch.allclose(live_model.fc.weight, torch.tensor(1.11))


def test_restore_model_architecture_mismatch(tmp_path: Path) -> None:
    """Restoring incompatible architecture raises error and rolls back."""
    saved_model = TinyLinearModel()  # 4 -> 2

    with runvault.init("arch-mismatch-exp", name="arch-run", workspace=tmp_path) as run:
        save_torch_checkpoint(run, "ckpt-arch", saved_model)

        larger_model = nn.Linear(10, 5)
        with torch.no_grad():
            larger_model.weight.fill_(9.99)
            larger_model.bias.fill_(8.88)

        with pytest.raises(RuntimeError):
            restore_torch_checkpoint(run, larger_model, strict_model=True)

        # Target larger model must remain unchanged
        assert torch.allclose(larger_model.weight, torch.tensor(9.99))
        assert torch.allclose(larger_model.bias, torch.tensor(8.88))


def test_checkpoint_manager_step_validation_and_deduplication(tmp_path: Path) -> None:
    """Manager rejects negative step and deduplicates redundant same-step saves."""
    model = TinyLinearModel()

    with runvault.init("mgr-step-exp", name="mgr-step-run", workspace=tmp_path) as run:
        manager = TorchCheckpointManager(run=run, model=model, step_interval=10)

        # Negative step rejection
        with pytest.raises(ValueError, match="Step must be non-negative"):
            manager.step_end(-1)

        # Save at step 10
        ckpt1 = manager.step_end(10)
        assert ckpt1 is not None
        assert manager.last_saved_step == 10

        # Repeated call at step 10 without force_save returns None (deduplicated)
        ckpt_dup = manager.step_end(10)
        assert ckpt_dup is None

        # force_save=True overrides deduplication
        ckpt_force = manager.step_end(10, force_save=True)
        assert ckpt_force is not None


def test_checkpoint_manager_epoch_validation_and_deduplication(tmp_path: Path) -> None:
    """Manager rejects negative epoch and deduplicates redundant same-epoch saves."""
    model = TinyLinearModel()

    with runvault.init("mgr-epoch-exp", name="mgr-epoch-run", workspace=tmp_path) as run:
        manager = TorchCheckpointManager(run=run, model=model, epoch_interval=1)

        # Negative epoch rejection
        with pytest.raises(ValueError, match="Epoch must be non-negative"):
            manager.epoch_end(-1)

        # Save at epoch 1
        ckpt1 = manager.epoch_end(1)
        assert ckpt1 is not None
        assert manager.last_saved_epoch == 1

        # Repeated call at epoch 1 is deduplicated
        ckpt_dup = manager.epoch_end(1)
        assert ckpt_dup is None

        # force_save=True overrides
        ckpt_force = manager.epoch_end(1, force_save=True)
        assert ckpt_force is not None


def test_checkpoint_manager_best_metric_modes_and_nan(tmp_path: Path) -> None:
    """Best metric tracks min/max modes correctly and rejects NaN/Inf values."""
    model = TinyLinearModel()

    with runvault.init("mgr-best-exp", name="mgr-best-run", workspace=tmp_path) as run:
        # Invalid mode rejected
        with pytest.raises(ValueError, match="Invalid mode 'invalid'"):
            TorchCheckpointManager(run=run, model=model, mode="invalid")

        # Mode 'min' (e.g. loss)
        mgr_min = TorchCheckpointManager(run=run, model=model, epoch_interval=None, mode="min")
        ckpt1 = mgr_min.epoch_end(1, best_metric=0.8)
        assert ckpt1 is not None
        assert mgr_min.best_metric == 0.8

        # 0.9 does not improve over 0.8 in 'min' mode -> no save
        ckpt2 = mgr_min.epoch_end(2, best_metric=0.9)
        assert ckpt2 is None
        assert mgr_min.best_metric == 0.8

        # NaN and Inf are ignored
        ckpt_nan = mgr_min.epoch_end(3, best_metric=float("nan"))
        assert ckpt_nan is None
        assert mgr_min.best_metric == 0.8

        ckpt_inf = mgr_min.epoch_end(4, best_metric=float("inf"))
        assert ckpt_inf is None
        assert mgr_min.best_metric == 0.8

        # 0.4 improves over 0.8 in 'min' mode -> saved
        ckpt3 = mgr_min.epoch_end(5, best_metric=0.4)
        assert ckpt3 is not None
        assert mgr_min.best_metric == 0.4

        # Mode 'max' (e.g. accuracy)
        mgr_max = TorchCheckpointManager(run=run, model=model, epoch_interval=None, mode="max")
        c1 = mgr_max.epoch_end(1, best_metric=0.7)
        assert c1 is not None
        assert mgr_max.best_metric == 0.7

        c2 = mgr_max.epoch_end(2, best_metric=0.6)  # worse
        assert c2 is None
        assert mgr_max.best_metric == 0.7

        c3 = mgr_max.epoch_end(3, best_metric=0.95)  # better
        assert c3 is not None
        assert mgr_max.best_metric == 0.95


def test_checkpoint_manager_restore_preserves_state(tmp_path: Path) -> None:
    """Manager restore updates step, epoch, and best_metric state attributes."""
    model = TinyLinearModel()

    with runvault.init("mgr-restore-exp", name="mgr-restore-run", workspace=tmp_path) as run:
        mgr1 = TorchCheckpointManager(run=run, model=model, mode="min")
        ckpt = mgr1.save("state-ckpt", epoch=7, step=700, best_metric=0.035)

        fresh_model = TinyLinearModel()
        mgr2 = TorchCheckpointManager(run=run, model=fresh_model, mode="min")
        rec = mgr2.restore(ckpt.id)

        assert rec.epoch == 7
        assert rec.step == 700
        assert rec.best_metric == 0.035
        assert mgr2.last_saved_epoch == 7
        assert mgr2.last_saved_step == 700
        assert mgr2.best_metric == 0.035


def test_save_torch_checkpoint_validates_negative_step_epoch(tmp_path: Path) -> None:
    """save_torch_checkpoint rejects negative step or epoch arguments."""
    model = TinyLinearModel()

    with runvault.init("neg-exp", name="neg-run", workspace=tmp_path) as run:
        with pytest.raises(ValueError, match="Step must be non-negative"):
            save_torch_checkpoint(run, "ckpt", model, step=-1)

        with pytest.raises(ValueError, match="Epoch must be non-negative"):
            save_torch_checkpoint(run, "ckpt", model, epoch=-1)
