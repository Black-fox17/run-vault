"""Structured PyTorch checkpoint payload serialization, deserialization, and format validation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, BinaryIO, Final, TypedDict

from runvault.core.errors import (
    CheckpointCorruptError,
    UnsupportedCheckpointFormatError,
)
from runvault.integrations.torch.rng import RngState, capture_rng_state

if TYPE_CHECKING:
    import os
    from collections.abc import Callable
    from pathlib import Path

    import torch

logger = logging.getLogger("runvault.integrations.torch.state")

TORCH_PAYLOAD_FORMAT_VERSION: Final[int] = 1


class TorchCheckpointPayload(TypedDict, total=False):
    """Schema for versioned PyTorch checkpoint payload dictionaries."""

    format_version: int
    run_id: str
    epoch: int | None
    step: int | None
    timestamp: str
    framework: str
    torch_version: str
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any] | None
    scheduler_state_dict: dict[str, Any] | None
    scaler_state_dict: dict[str, Any] | None
    best_metric: float | None
    metadata: dict[str, Any]
    rng_state: RngState | None


def serialize_torch_checkpoint(
    model: torch.nn.Module,
    *,
    run_id: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int | None = None,
    step: int | None = None,
    best_metric: float | None = None,
    metadata: dict[str, Any] | None = None,
    include_rng: bool = True,
) -> Callable[[Path], None]:
    """Create a writer callback that serializes training state into a versioned PyTorch artifact.

    Args:
        model: PyTorch model (state_dict will be extracted).
        run_id: ID of the owning run.
        optimizer: Optional optimizer to extract state_dict from.
        scheduler: Optional learning rate scheduler to extract state_dict from.
        scaler: Optional AMP GradScaler to extract state_dict from.
        epoch: Optional training epoch number.
        step: Optional global training step number.
        best_metric: Optional validation metric benchmark (e.g., best loss or accuracy).
        metadata: Optional dictionary of arbitrary training metadata.
        include_rng: Whether to capture and embed RNG states across Python, NumPy, and PyTorch.

    Returns:
        A writer callback `writer(dest: Path)` compatible with RunVault's storage engine.
    """
    import torch

    # Extract state dicts cleanly without storing live Python objects or trainers
    model_sd = model.state_dict()
    opt_sd = optimizer.state_dict() if optimizer is not None else None
    sched_sd = scheduler.state_dict() if scheduler is not None else None
    scaler_sd = scaler.state_dict() if scaler is not None else None
    rng_sd = capture_rng_state() if include_rng else None

    payload: TorchCheckpointPayload = {
        "format_version": TORCH_PAYLOAD_FORMAT_VERSION,
        "run_id": run_id,
        "epoch": epoch,
        "step": step,
        "timestamp": datetime.now(UTC).isoformat(),
        "framework": "pytorch",
        "torch_version": str(torch.__version__),
        "model_state_dict": model_sd,
        "optimizer_state_dict": opt_sd,
        "scheduler_state_dict": sched_sd,
        "scaler_state_dict": scaler_sd,
        "best_metric": best_metric,
        "metadata": metadata or {},
        "rng_state": rng_sd,
    }

    def writer(dest: Path) -> None:
        torch.save(payload, dest)

    return writer


def deserialize_torch_checkpoint(
    stream_or_path: Path | BinaryIO | str | os.PathLike[str],
    *,
    safe_load: bool = True,
    map_location: Any | None = None,
    checkpoint_id: str = "unknown",
) -> dict[str, Any]:
    """Safely deserialize and validate a PyTorch checkpoint payload.

    Security Notice:
        By default, `safe_load=True` enforces `weights_only=True` in `torch.load`.
        Setting `safe_load=False` bypasses weights-only restrictions and executes standard
        Python unpickling. Only set `safe_load=False` for trusted internal checkpoints.
        SHA-256 integrity checksums verify non-corruption, NOT origin authenticity.

    Args:
        stream_or_path: Binary stream or file path to deserialize.
        safe_load: Enforce weights_only=True to prevent arbitrary code execution during unpickling.
        map_location: Device location mapping passed to torch.load (e.g. 'cpu', 'cuda:0').
        checkpoint_id: Identifier for diagnostic error messages.

    Returns:
        Validated checkpoint payload dictionary.

    Raises:
        CheckpointCorruptError: If deserialization fails or payload schema is invalid.
        UnsupportedCheckpointFormatError: If the payload format version is greater than supported.
    """
    import torch

    try:
        if safe_load:
            payload = torch.load(stream_or_path, map_location=map_location, weights_only=True)
        else:
            logger.warning(
                "Loading PyTorch checkpoint '%s' with safe_load=False (weights_only=False). "
                "Warning: Integrity checks verify artifact non-corruption but do not guarantee "
                "authenticity. Untrusted files may execute arbitrary code.",
                checkpoint_id,
            )
            payload = torch.load(stream_or_path, map_location=map_location, weights_only=False)
    except Exception as err:
        raise CheckpointCorruptError(
            checkpoint_id=checkpoint_id,
            reason=f"Failed to deserialize PyTorch checkpoint artifact: {err}",
        ) from err

    # Validate payload structure
    if not isinstance(payload, dict):
        raise CheckpointCorruptError(
            checkpoint_id=checkpoint_id,
            reason=f"Expected checkpoint payload to be a dict, got '{type(payload).__name__}'",
        )

    version = payload.get("format_version")
    if version is None:
        # Check if user saved a bare state_dict (e.g. from an earlier Phase 3 writer)
        if all(isinstance(k, str) for k in payload):
            # Treat as raw model state_dict for backward compatibility
            return {
                "format_version": 1,
                "model_state_dict": payload,
                "epoch": None,
                "step": None,
                "best_metric": None,
                "metadata": {},
                "rng_state": None,
            }
        raise CheckpointCorruptError(
            checkpoint_id=checkpoint_id,
            reason="Checkpoint payload missing 'format_version' header",
        )

    if not isinstance(version, int) or version > TORCH_PAYLOAD_FORMAT_VERSION:
        raise UnsupportedCheckpointFormatError(
            format_version=int(version) if isinstance(version, int) else -1,
            max_supported=TORCH_PAYLOAD_FORMAT_VERSION,
            checkpoint_id=checkpoint_id,
        )

    if "model_state_dict" not in payload:
        raise CheckpointCorruptError(
            checkpoint_id=checkpoint_id,
            reason="Checkpoint payload missing required 'model_state_dict' entry",
        )

    return payload
