"""RNG state capture, serialization, and restoration across Python, NumPy, and PyTorch.

Reproducibility Boundary:
    RunVault captures and restores:
    - Python standard library `random` state
    - NumPy `numpy.random` state (when installed)
    - PyTorch CPU RNG state (`torch.get_rng_state()`)
    - PyTorch CUDA RNG states for all available devices (`torch.cuda.get_rng_state_all()`)

    Limitations & Boundary:
    Restoring RNG states alone does NOT guarantee exact mathematical bitwise reproducibility
    in all training setups. Specifically, reproducibility is bounded by:
    - DataLoader worker processes: Subprocesses initialized before or independently of RNG
      restoration maintain their own internal seeds unless explicitly re-seeded in `worker_init_fn`.
    - Asynchronous data prefetching and multi-threading.
    - PyTorch non-deterministic CUDA operations (e.g., atomic additions in backward passes,
      unseeded cuDNN benchmark algorithms). To enforce strict algorithmic determinism, users
      must configure `torch.use_deterministic_algorithms(True)` and set
      `torch.backends.cudnn.deterministic = True`.
    - Cross-hardware or cross-CUDA version differences.
"""

from __future__ import annotations

import logging
import random
import sys
from typing import Any, TypedDict

logger = logging.getLogger("runvault.integrations.torch.rng")


class RngState(TypedDict, total=False):
    """Structured representation of random number generator states."""

    python: tuple[Any, ...]
    numpy: tuple[str, list[int], int, int, float] | None
    torch_cpu: Any  # torch.ByteTensor
    torch_cuda: list[Any] | None  # list[torch.ByteTensor]


def capture_rng_state() -> RngState:
    """Capture random number generator states across Python, NumPy, CPU, and CUDA devices.

    Returns:
        Structured RngState dictionary with primitives and tensors safe for weights_only unpickling.
    """
    import torch

    state: RngState = {
        "python": random.getstate(),
        "numpy": None,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }

    # Capture NumPy RNG if installed
    if "numpy" in sys.modules or _is_numpy_available():
        try:
            import numpy as np

            # In NumPy, legacy=True explicitly selects the 5-element tuple overload
            np_state = np.random.get_state(legacy=True)
            keys_list = [int(x) for x in np_state[1]]
            state["numpy"] = (
                str(np_state[0]),
                keys_list,
                int(np_state[2]),
                int(np_state[3]),
                float(np_state[4]),
            )
        except Exception as err:
            logger.debug("Failed to capture NumPy RNG state: %s", err)

    # Capture CUDA RNG if CUDA devices are available
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception as err:
            logger.debug("Failed to capture CUDA RNG states: %s", err)

    return state


def restore_rng_state(state: RngState | dict[str, Any] | None) -> None:
    """Restore random number generator states for Python, NumPy, CPU, and CUDA.

    Args:
        state: RngState dictionary previously captured by capture_rng_state.
    """
    if state is None:
        return

    import torch

    # 1. Restore Python random
    py_state = state.get("python")
    if py_state is not None:
        try:
            # random.setstate expects a 3-element tuple
            random.setstate(tuple(py_state))
        except Exception as err:
            logger.warning("Failed to restore Python RNG state: %s", err)

    # 2. Restore NumPy random
    np_state_raw = state.get("numpy")
    if np_state_raw is not None and ("numpy" in sys.modules or _is_numpy_available()):
        try:
            import numpy as np

            # Reconstruct the 5-element NumPy random state tuple
            # Format: (str algorithm, ndarray keys, int pos, int has_gauss, float cached_gaussian)
            algorithm = str(np_state_raw[0])
            keys_array = np.array(np_state_raw[1], dtype=np.uint32)
            pos = int(np_state_raw[2])
            has_gauss = int(np_state_raw[3])
            cached_gauss = float(np_state_raw[4])
            np.random.set_state((algorithm, keys_array, pos, has_gauss, cached_gauss))
        except Exception as err:
            logger.warning("Failed to restore NumPy RNG state: %s", err)

    # 3. Restore PyTorch CPU RNG
    cpu_state = state.get("torch_cpu")
    if cpu_state is not None:
        try:
            if not isinstance(cpu_state, torch.Tensor):
                cpu_state = torch.tensor(cpu_state, dtype=torch.uint8)
            torch.set_rng_state(cpu_state)
        except Exception as err:
            logger.warning("Failed to restore PyTorch CPU RNG state: %s", err)

    # 4. Restore PyTorch CUDA RNG
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        try:
            num_devices = torch.cuda.device_count()
            if len(cuda_states) >= num_devices:
                torch.cuda.set_rng_state_all(cuda_states[:num_devices])
            else:
                logger.warning(
                    "CUDA RNG states list length (%d) is less than available devices (%d)",
                    len(cuda_states),
                    num_devices,
                )
        except Exception as err:
            logger.warning("Failed to restore PyTorch CUDA RNG states: %s", err)


def _is_numpy_available() -> bool:
    """Helper to detect numpy without raising exceptions."""
    try:
        import importlib.util

        return importlib.util.find_spec("numpy") is not None
    except (ImportError, AttributeError, ValueError):
        return False
