"""PyTorch training lifecycle manager and state restoration orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from runvault.core.session import ActiveRun
from runvault.integrations.torch.rng import restore_rng_state
from runvault.integrations.torch.state import (
    deserialize_torch_checkpoint,
    serialize_torch_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from runvault.core.domain import Checkpoint
    from runvault.core.storage.base import StorageProvider

logger = logging.getLogger("runvault.integrations.torch.manager")


@dataclass(frozen=True)
class TorchRecoveryResult:
    """Structured recovery information returned when restoring a PyTorch checkpoint."""

    checkpoint: Checkpoint
    epoch: int | None
    step: int | None
    best_metric: float | None
    metadata: dict[str, Any]
    is_fallback: bool
    rejected_checkpoints: tuple[tuple[Checkpoint, str], ...]


def save_torch_checkpoint(
    run: ActiveRun,
    name: str,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int | None = None,
    step: int | None = None,
    best_metric: float | None = None,
    metadata: dict[str, Any] | None = None,
    include_rng: bool = True,
    checkpoint_id: str | None = None,
    storage: StorageProvider | None = None,
) -> Checkpoint:
    """Save an atomic, integrity-verified PyTorch model checkpoint.

    Args:
        run: The active training run context.
        name: Human-readable name for the checkpoint (e.g. 'epoch-5' or 'best-model').
        model: PyTorch model whose state_dict will be saved.
        optimizer: Optional optimizer whose state_dict will be saved.
        scheduler: Optional learning rate scheduler whose state_dict will be saved.
        scaler: Optional AMP GradScaler whose state_dict will be saved.
        epoch: Current training epoch index.
        step: Current global training step index.
        best_metric: Benchmark metric score (e.g. minimum validation loss).
        metadata: Optional dictionary of additional training parameters or metadata.
        include_rng: Whether to capture Python, NumPy, CPU, and CUDA RNG states.
        checkpoint_id: Explicit UUID for the checkpoint (auto-generated if omitted).
        storage: Optional storage provider override.

    Returns:
        Committed Checkpoint entity with SHA-256 integrity metadata.
    """
    writer = serialize_torch_checkpoint(
        model,
        run_id=run.id,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        step=step,
        best_metric=best_metric,
        metadata=metadata,
        include_rng=include_rng,
    )

    combined_metadata: dict[str, Any] = {
        "framework": "pytorch",
        **(metadata or {}),
    }
    if epoch is not None:
        combined_metadata["epoch"] = epoch
    if best_metric is not None:
        combined_metadata["best_metric"] = best_metric

    return run.save_checkpoint(
        name=name,
        writer=writer,
        step=step,
        metadata=combined_metadata,
        checkpoint_id=checkpoint_id,
        storage=storage,
    )


def restore_torch_checkpoint(
    run_or_id: ActiveRun | str,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    checkpoint_id: str | None = None,
    restore_rng: bool = True,
    safe_load: bool = True,
    strict_model: bool = True,
    map_location: Any | None = None,
    workspace: str | Path | None = None,
    storage: StorageProvider | None = None,
) -> TorchRecoveryResult:
    """Restore PyTorch training state into live model, optimizer, scheduler, and RNGs.

    Integrity & Fallback Semantics:
        - If `checkpoint_id` is None, RunVault discovers the newest checkpoint and automatically
          verifies SHA-256 integrity. If corrupted, it transparently falls back to the newest
          valid older checkpoint.
        - If `checkpoint_id` is explicitly specified, RunVault strictly restores that exact
          checkpoint without fallback.

    Args:
        run_or_id: ActiveRun instance or run ID string.
        model: PyTorch model instance to populate with loaded state_dict.
        optimizer: Optional optimizer instance to populate with loaded state_dict.
        scheduler: Optional learning rate scheduler to populate with loaded state_dict.
        scaler: Optional GradScaler instance to populate with loaded state_dict.
        checkpoint_id: Specific checkpoint ID to restore, or None for latest valid.
        restore_rng: Whether to restore random number generator states.
        safe_load: Enforce weights_only=True during unpickling to prevent code execution.
        strict_model: Whether to strictly enforce state_dict key matching in model.
        map_location: PyTorch device placement (e.g. 'cpu', 'cuda:0').
        workspace: Explicit workspace directory path (used when run_or_id is a str ID).
        storage: Optional StorageProvider override.

    Returns:
        TorchRecoveryResult containing restored epoch, step, best metric, and fallback details.
    """
    if isinstance(run_or_id, ActiveRun):
        rec_result = run_or_id.restore_checkpoint(checkpoint_id=checkpoint_id, storage=storage)
    else:
        from runvault.api import restore_checkpoint as api_restore_checkpoint

        rec_result = api_restore_checkpoint(
            run_id=run_or_id,
            checkpoint_id=checkpoint_id,
            workspace=workspace,
            storage=storage,
        )

    # Stream artifact from storage provider without requiring local filesystem persistence
    with rec_result.open() as stream:
        payload = deserialize_torch_checkpoint(
            stream,
            safe_load=safe_load,
            map_location=map_location,
            checkpoint_id=rec_result.checkpoint.id,
        )

    # 1. Load model state dict
    model.load_state_dict(payload["model_state_dict"], strict=strict_model)

    # 2. Load optimizer state dict if provided and present in checkpoint
    if optimizer is not None:
        opt_state = payload.get("optimizer_state_dict")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    # 3. Load scheduler state dict if provided and present in checkpoint
    if scheduler is not None:
        sched_state = payload.get("scheduler_state_dict")
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

    # 4. Load scaler state dict if provided and present in checkpoint
    if scaler is not None:
        scaler_state = payload.get("scaler_state_dict")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)

    # 5. Restore random number generator state if requested
    if restore_rng:
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            restore_rng_state(rng_state)

    return TorchRecoveryResult(
        checkpoint=rec_result.checkpoint,
        epoch=payload.get("epoch"),
        step=payload.get("step"),
        best_metric=payload.get("best_metric"),
        metadata=payload.get("metadata") or {},
        is_fallback=rec_result.is_fallback,
        rejected_checkpoints=rec_result.rejected_checkpoints,
    )


class TorchCheckpointManager:
    """Helper for managing checkpoint saves and recoveries inside PyTorch training loops."""

    def __init__(
        self,
        run: ActiveRun,
        model: torch.nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        step_interval: int | None = None,
        epoch_interval: int | None = 1,
        include_rng: bool = True,
        storage: StorageProvider | None = None,
    ) -> None:
        """Initialize the manager with model, optimizers, and interval triggers."""
        self.run = run
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.step_interval = step_interval
        self.epoch_interval = epoch_interval
        self.include_rng = include_rng
        self.storage = storage

    def save(
        self,
        name: str,
        *,
        epoch: int | None = None,
        step: int | None = None,
        best_metric: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Save a checkpoint immediately with current component states."""
        return save_torch_checkpoint(
            self.run,
            name=name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            step=step,
            best_metric=best_metric,
            metadata=metadata,
            include_rng=self.include_rng,
            storage=self.storage,
        )

    def restore(
        self,
        checkpoint_id: str | None = None,
        *,
        restore_rng: bool = True,
        safe_load: bool = True,
        strict_model: bool = True,
        map_location: Any | None = None,
    ) -> TorchRecoveryResult:
        """Restore training state into manager components."""
        return restore_torch_checkpoint(
            self.run,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            checkpoint_id=checkpoint_id,
            restore_rng=restore_rng,
            safe_load=safe_load,
            strict_model=strict_model,
            map_location=map_location,
            storage=self.storage,
        )

    def step_end(
        self,
        step: int,
        *,
        epoch: int | None = None,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        force_save: bool = False,
    ) -> Checkpoint | None:
        """Record step metrics and save checkpoint if interval is reached."""
        if metrics:
            self.run.log_metrics(metrics, step=step)

        should_save = force_save or (
            self.step_interval is not None
            and self.step_interval > 0
            and step % self.step_interval == 0
        )
        if should_save:
            name = f"step-{step}"
            return self.save(name=name, epoch=epoch, step=step, metadata=metadata)
        return None

    def epoch_end(
        self,
        epoch: int,
        *,
        step: int | None = None,
        metrics: dict[str, float] | None = None,
        best_metric: float | None = None,
        metadata: dict[str, Any] | None = None,
        force_save: bool = False,
    ) -> Checkpoint | None:
        """Record epoch metrics and save checkpoint if epoch interval is reached."""
        if metrics:
            self.run.log_metrics(metrics, step=step)

        should_save = force_save or (
            self.epoch_interval is not None
            and self.epoch_interval > 0
            and epoch % self.epoch_interval == 0
        )
        if should_save:
            name = f"epoch-{epoch}"
            return self.save(
                name=name,
                epoch=epoch,
                step=step,
                best_metric=best_metric,
                metadata=metadata,
            )
        return None
