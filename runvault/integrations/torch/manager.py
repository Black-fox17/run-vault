"""PyTorch training lifecycle manager and state restoration orchestration."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from runvault.core.session import ActiveRun
from runvault.integrations.torch.rng import capture_rng_state, restore_rng_state
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
    if step is not None and step < 0:
        raise ValueError(f"Step must be non-negative, got {step}")
    if epoch is not None and epoch < 0:
        raise ValueError(f"Epoch must be non-negative, got {epoch}")

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


def _validate_payload_components(
    payload: dict[str, Any],
    optimizer: Any | None,
    scheduler: Any | None,
    scaler: Any | None,
) -> None:
    """Pre-validate that requested components exist in checkpoint payload."""
    if optimizer is not None and payload.get("optimizer_state_dict") is None:
        raise ValueError(
            "Requested optimizer restoration, but checkpoint does not contain optimizer state."
        )
    if scheduler is not None and payload.get("scheduler_state_dict") is None:
        raise ValueError(
            "Requested scheduler restoration, but checkpoint does not contain scheduler state."
        )
    if scaler is not None and payload.get("scaler_state_dict") is None:
        raise ValueError(
            "Requested scaler restoration, but checkpoint does not contain scaler state."
        )


def _rollback_components(
    *,
    model: torch.nn.Module,
    model_snapshot: dict[str, Any],
    strict_model: bool,
    optimizer: Any | None,
    optimizer_snapshot: dict[str, Any] | None,
    scheduler: Any | None,
    scheduler_snapshot: dict[str, Any] | None,
    scaler: Any | None,
    scaler_snapshot: dict[str, Any] | None,
    restore_rng: bool,
    rng_snapshot: Any | None,
) -> list[Exception]:
    """Roll back components to their pre-restoration snapshots."""
    rollback_errors: list[Exception] = []
    try:
        model.load_state_dict(model_snapshot, strict=strict_model)
    except Exception as rb_err:
        rollback_errors.append(rb_err)

    if optimizer is not None and optimizer_snapshot is not None:
        try:
            optimizer.load_state_dict(optimizer_snapshot)
        except Exception as rb_err:
            rollback_errors.append(rb_err)

    if scheduler is not None and scheduler_snapshot is not None:
        try:
            scheduler.load_state_dict(scheduler_snapshot)
        except Exception as rb_err:
            rollback_errors.append(rb_err)

    if scaler is not None and scaler_snapshot is not None:
        try:
            scaler.load_state_dict(scaler_snapshot)
        except Exception as rb_err:
            rollback_errors.append(rb_err)

    if restore_rng and rng_snapshot is not None:
        try:
            restore_rng_state(rng_snapshot)
        except Exception as rb_err:
            rollback_errors.append(rb_err)

    return rollback_errors


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

    _validate_payload_components(payload, optimizer, scheduler, scaler)

    # Snapshot current states for atomic rollback on failure
    model_snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer_snapshot = (
        copy.deepcopy(optimizer.state_dict())
        if optimizer is not None and hasattr(optimizer, "state_dict")
        else None
    )
    scheduler_snapshot = (
        copy.deepcopy(scheduler.state_dict())
        if scheduler is not None and hasattr(scheduler, "state_dict")
        else None
    )
    scaler_snapshot = (
        copy.deepcopy(scaler.state_dict())
        if scaler is not None and hasattr(scaler, "state_dict")
        else None
    )
    rng_snapshot = capture_rng_state() if restore_rng else None

    try:
        model.load_state_dict(payload["model_state_dict"], strict=strict_model)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])
        # Restore RNG state LAST, after all tensor deserialization and module loadings
        if restore_rng and payload.get("rng_state") is not None:
            restore_rng_state(payload["rng_state"])
    except Exception as err:
        rollback_errors = _rollback_components(
            model=model,
            model_snapshot=model_snapshot,
            strict_model=strict_model,
            optimizer=optimizer,
            optimizer_snapshot=optimizer_snapshot,
            scheduler=scheduler,
            scheduler_snapshot=scheduler_snapshot,
            scaler=scaler,
            scaler_snapshot=scaler_snapshot,
            restore_rng=restore_rng,
            rng_snapshot=rng_snapshot,
        )
        if rollback_errors:
            err_details = "; ".join(str(e) for e in rollback_errors)
            err.add_note(f"Rollback failed after restoration error: {err_details}")
        raise err

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
    """Helper for managing checkpoint saves and recoveries inside PyTorch training loops.

    Epoch Convention:
        The manager supports both 0-based and 1-based epoch numbering according to user
        training loop conventions. RunVault does not alter or re-index the provided epoch;
        it records and validates that epoch is non-negative.
    """

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
        mode: str = "min",
        include_rng: bool = True,
        storage: StorageProvider | None = None,
    ) -> None:
        """Initialize the manager with model, optimizers, and interval triggers."""
        if mode not in ("min", "max"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'min' or 'max'.")
        self.run = run
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.step_interval = step_interval
        self.epoch_interval = epoch_interval
        self.mode = mode
        self.include_rng = include_rng
        self.storage = storage

        self._last_saved_step: int | None = None
        self._last_saved_epoch: int | None = None
        self._best_metric: float | None = None

    @property
    def best_metric(self) -> float | None:
        """Current best benchmark metric recorded by the manager."""
        return self._best_metric

    @property
    def last_saved_step(self) -> int | None:
        """Global step index of the most recently saved checkpoint."""
        return self._last_saved_step

    @property
    def last_saved_epoch(self) -> int | None:
        """Epoch index of the most recently saved checkpoint."""
        return self._last_saved_epoch

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
        ckpt = save_torch_checkpoint(
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
        if step is not None:
            self._last_saved_step = step
        if epoch is not None:
            self._last_saved_epoch = epoch
        return ckpt

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
        res = restore_torch_checkpoint(
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
        if res.step is not None:
            self._last_saved_step = res.step
        if res.epoch is not None:
            self._last_saved_epoch = res.epoch
        if res.best_metric is not None:
            self._best_metric = res.best_metric
        return res

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
        if step < 0:
            raise ValueError(f"Step must be non-negative, got {step}")
        if epoch is not None and epoch < 0:
            raise ValueError(f"Epoch must be non-negative, got {epoch}")

        if metrics:
            self.run.log_metrics(metrics, step=step)

        if not force_save and self._last_saved_step == step:
            return None

        should_save = force_save or (
            self.step_interval is not None
            and self.step_interval > 0
            and step > 0
            and step % self.step_interval == 0
        )
        if should_save:
            name = f"step-{step}"
            return self.save(name=name, epoch=epoch, step=step, metadata=metadata)
        return None

    def _evaluate_best_metric(self, best_metric: float | None, epoch: int) -> bool:
        """Check if best_metric improves upon the current best recorded metric."""
        if best_metric is None:
            return False
        if math.isnan(best_metric) or math.isinf(best_metric):
            logger.warning(
                "Ignoring non-finite best_metric value (%s) at epoch %d",
                best_metric,
                epoch,
            )
            return False

        if (
            self._best_metric is None
            or (self.mode == "min" and best_metric < self._best_metric)
            or (self.mode == "max" and best_metric > self._best_metric)
        ):
            self._best_metric = best_metric
            return True
        return False

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
        """Record epoch metrics and save checkpoint if epoch interval or best metric is reached."""
        if epoch < 0:
            raise ValueError(f"Epoch must be non-negative, got {epoch}")
        if step is not None and step < 0:
            raise ValueError(f"Step must be non-negative, got {step}")

        if metrics:
            self.run.log_metrics(metrics, step=step)

        is_new_best = self._evaluate_best_metric(best_metric, epoch)

        if not force_save and self._last_saved_epoch == epoch:
            return None

        interval_save = (
            self.epoch_interval is not None
            and self.epoch_interval > 0
            and epoch % self.epoch_interval == 0
        )
        if force_save or interval_save or is_new_best:
            name = f"epoch-{epoch}"
            combined_meta = dict(metadata or {})
            if is_new_best:
                combined_meta["is_best"] = True
            return self.save(
                name=name,
                epoch=epoch,
                step=step,
                best_metric=self._best_metric,
                metadata=combined_meta,
            )
        return None
