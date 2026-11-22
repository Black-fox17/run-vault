"""Authentication helpers and credential lifecycle management for Google Drive."""

from __future__ import annotations

import contextlib
import logging
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("runvault.integrations.gdrive.auth")

GDRIVE_DEFAULT_SCOPES: Final[tuple[str, ...]] = ("https://www.googleapis.com/auth/drive.file",)

GDRIVE_DEPENDENCY_ERROR: Final[str] = (
    "Google Drive storage requires optional dependencies. "
    "Install them with: pip install 'runvault-ai[gdrive]'"
)


def check_gdrive_dependencies() -> None:
    """Verify that optional Google Drive SDK dependencies are installed."""
    try:
        import importlib

        importlib.import_module("google.auth")
        importlib.import_module("googleapiclient.discovery")
    except ImportError as err:
        raise ImportError(GDRIVE_DEPENDENCY_ERROR) from err


def is_colab_environment() -> bool:
    """Detect whether code is executing within a Google Colab runtime."""
    if sys.modules.get("google.colab") is not None:
        return True
    try:
        import importlib.util

        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def authenticate_colab(scopes: Sequence[str] = GDRIVE_DEFAULT_SCOPES) -> Any:
    """Authenticate via Google Colab runtime credentials."""
    check_gdrive_dependencies()
    try:
        import google.auth

        colab_auth = sys.modules.get("google.colab.auth")
        if colab_auth is None:
            import google.colab.auth as colab_auth  # type: ignore[reportMissingImports]

        colab_auth.authenticate_user()
        credentials, _ = google.auth.default(scopes=list(scopes))
        return credentials
    except (ImportError, AttributeError) as err:
        raise ImportError(
            f"Colab authentication is only available within Google Colab environments: {err}"
        ) from err
    except Exception as err:
        raise RuntimeError(f"Google Colab user authentication failed: {err}") from err


def authenticate_local_oauth(
    client_secrets_file: str | Path | None = None,
    token_cache_path: str | Path | None = None,
    scopes: Sequence[str] = GDRIVE_DEFAULT_SCOPES,
) -> Any:
    """Authenticate via local installed-app OAuth flow with token caching."""
    check_gdrive_dependencies()
    import google.auth
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[reportMissingImports]

    cache_path = (
        Path(token_cache_path).expanduser()
        if token_cache_path is not None
        else Path.home() / ".config" / "runvault" / "gdrive_token.json"
    )

    creds: Any = None
    if cache_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(cache_path), scopes=list(scopes))
        except Exception as err:
            logger.warning(
                "Failed to load cached Drive credentials from '%s': %s",
                cache_path,
                type(err).__name__,
            )
            creds = None

    if creds and getattr(creds, "valid", False):
        return creds

    if creds and getattr(creds, "expired", False) and getattr(creds, "refresh_token", None):
        try:
            creds.refresh(Request())
            if creds.valid:
                _save_token_cache(cache_path, creds)
                return creds
        except Exception as err:
            logger.warning(
                "Failed to refresh cached Drive credentials: %s",
                type(err).__name__,
            )
            creds = None

    if client_secrets_file is not None and Path(client_secrets_file).exists():
        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secrets_file),
            scopes=list(scopes),
        )
        creds = flow.run_local_server(port=0)
        _save_token_cache(cache_path, creds)
        return creds

    # Fall back to application default credentials (ADC)
    try:
        creds, _ = google.auth.default(scopes=list(scopes))
        return creds
    except Exception as err:
        raise RuntimeError(
            "Could not authenticate to Google Drive. Provide credentials directly, "
            "set GOOGLE_APPLICATION_CREDENTIALS, or provide a client_secrets_file: "
            f"{type(err).__name__}"
        ) from err


def save_token_cache(cache_path: Path, creds: Any) -> None:
    """Save user OAuth token JSON with restrictive permissions (0o600) and atomic replacement."""
    temp_cache: Path | None = None
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            cache_path.parent.chmod(0o700)
        temp_cache = cache_path.parent / f".tmp_{cache_path.name}_{uuid.uuid4().hex}"
        temp_cache.write_text(creds.to_json(), encoding="utf-8")
        temp_cache.chmod(0o600)
        temp_cache.replace(cache_path)
    except Exception as err:
        if temp_cache is not None:
            temp_cache.unlink(missing_ok=True)
        logger.warning(
            "Failed to cache Google Drive token to '%s': %s",
            cache_path,
            type(err).__name__,
        )


_save_token_cache = save_token_cache


def get_credentials(
    credentials: Any | None = None,
    *,
    client_secrets_file: str | Path | None = None,
    token_cache_path: str | Path | None = None,
    scopes: Sequence[str] = GDRIVE_DEFAULT_SCOPES,
) -> Any:
    """Resolve Google credentials automatically or via explicit parameters."""
    if credentials is not None:
        return credentials

    if is_colab_environment():
        try:
            return authenticate_colab(scopes=scopes)
        except Exception as err:
            logger.warning("Colab auto-authentication failed, falling back to local: %s", err)

    return authenticate_local_oauth(
        client_secrets_file=client_secrets_file,
        token_cache_path=token_cache_path,
        scopes=scopes,
    )
