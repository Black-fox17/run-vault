"""Verification tests for the root runvault package distribution and imports."""

import subprocess
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
    """Verify that importing core package in isolated process does not pull heavy dependencies."""
    forbidden_heavy_modules: Final[tuple[str, ...]] = (
        "torch",
        "pydantic",
        "fastapi",
        "sqlalchemy",
        "googleapiclient",
        "google.auth",
    )
    code = (
        "import sys\n"
        "import runvault\n"
        f"forbidden = {forbidden_heavy_modules!r}\n"
        "for module_name in forbidden:\n"
        "    if module_name in sys.modules:\n"
        "        sys.stderr.write(f'Module {module_name} unexpectedly imported\\n')\n"
        "        sys.exit(1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
