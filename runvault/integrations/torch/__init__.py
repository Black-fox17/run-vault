"""PyTorch integration for RunVault - Model checkpointing and training recovery."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from runvault.integrations.torch.manager import (
        TorchCheckpointManager,
        TorchRecoveryResult,
        restore_torch_checkpoint,
        save_torch_checkpoint,
    )
    from runvault.integrations.torch.rng import (
        RngState,
        capture_rng_state,
        restore_rng_state,
    )
    from runvault.integrations.torch.state import (
        TORCH_PAYLOAD_FORMAT_VERSION,
        TorchCheckpointPayload,
        deserialize_torch_checkpoint,
        serialize_torch_checkpoint,
    )

TORCH_DEPENDENCY_ERROR: Final[str] = (
    "PyTorch integration requires optional dependencies. "
    "Install them with: pip install 'runvault[torch]'"
)


def check_torch_dependencies() -> None:
    """Verify that PyTorch is installed and accessible.

    Raises:
        ImportError: If torch is not installed, directing user to optional extra.
    """
    try:
        import importlib

        importlib.import_module("torch")
    except ImportError as err:
        raise ImportError(TORCH_DEPENDENCY_ERROR) from err


def is_torch_available() -> bool:
    """Check if PyTorch is installed in the current Python environment."""
    if sys.modules.get("torch") is not None:
        return True
    try:
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except (ImportError, AttributeError, ValueError):
        return False


__all__ = [
    "TORCH_DEPENDENCY_ERROR",
    "TORCH_PAYLOAD_FORMAT_VERSION",
    "RngState",
    "TorchCheckpointManager",
    "TorchCheckpointPayload",
    "TorchRecoveryResult",
    "capture_rng_state",
    "check_torch_dependencies",
    "deserialize_torch_checkpoint",
    "is_torch_available",
    "restore_rng_state",
    "restore_torch_checkpoint",
    "save_torch_checkpoint",
    "serialize_torch_checkpoint",
]


def __getattr__(name: str) -> Any:
    if name in (
        "TorchCheckpointManager",
        "TorchRecoveryResult",
        "restore_torch_checkpoint",
        "save_torch_checkpoint",
    ):
        check_torch_dependencies()
        import runvault.integrations.torch.manager as mgr

        return getattr(mgr, name)

    if name in ("RngState", "capture_rng_state", "restore_rng_state"):
        check_torch_dependencies()
        from runvault.integrations.torch import rng

        return getattr(rng, name)

    if name in (
        "TORCH_PAYLOAD_FORMAT_VERSION",
        "TorchCheckpointPayload",
        "deserialize_torch_checkpoint",
        "serialize_torch_checkpoint",
    ):
        check_torch_dependencies()
        import runvault.integrations.torch.state as st

        return getattr(st, name)

    raise AttributeError(f"module 'runvault.integrations.torch' has no attribute '{name}'")
