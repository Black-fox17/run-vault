"""Verification tests for the root runvault package distribution and imports."""

import sys
from typing import Final

import runvault


def test_package_version_exported() -> None:
    """Verify that __version__ is exported and follows semantic versioning."""
    assert hasattr(runvault, "__version__")
    assert isinstance(runvault.__version__, str)
    assert runvault.__version__ == "0.1.0"


def test_package_all_attribute() -> None:
    """Verify that __all__ is explicitly defined and contains __version__."""
    assert hasattr(runvault, "__all__")
    assert "__version__" in runvault.__all__


def test_core_package_is_lightweight() -> None:
    """Verify that importing the core package does not pull heavy optional dependencies."""
    forbidden_heavy_modules: Final[tuple[str, ...]] = (
        "torch",
        "pydantic",
        "fastapi",
        "sqlalchemy",
        "googleapiclient",
        "google.auth",
    )
    for module_name in forbidden_heavy_modules:
        assert module_name not in sys.modules, (
            f"Core runvault package unexpectedly imported heavy module: {module_name}"
        )
