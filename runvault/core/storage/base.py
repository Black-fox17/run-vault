"""Storage provider interface, capability model, and staged artifact primitives."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class StorageCapabilities:
    """Declared operational capabilities and guarantees of a storage provider backend."""

    provides_local_path: bool = True
    supports_exclusive_publication: bool = True
    supports_directory_sync: bool = False
    can_verify_integrity: bool = True


@dataclass(frozen=True)
class StagedArtifact:
    """A prepared staging artifact ready for atomic publication.

    Providers that stage locally populate `staging_path`. Remote or in-memory providers
    may rely solely on `staging_ref`, keeping staging implementation details decoupled
    from the generic provider contract and public checkpoint orchestration.
    """

    staging_path: Path | None = None
    size_bytes: int = 0
    checksum_sha256: str = ""
    staging_ref: str = ""

    def __post_init__(self) -> None:
        if not self.staging_ref and self.staging_path is not None:
            object.__setattr__(self, "staging_ref", str(self.staging_path))


@dataclass(frozen=True)
class PublishedArtifact:
    """Result of successfully publishing an artifact to storage."""

    artifact_key: str
    size_bytes: int
    checksum_sha256: str
    storage_provider: str
    staging_cleanup_warning: str | None = None


class StorageProvider(ABC):
    """Abstract interface decoupling checkpoint persistence from specific storage backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider type identifier (e.g. 'local')."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> StorageCapabilities:
        """Declared operational capabilities and durability guarantees."""
        ...

    @abstractmethod
    def stage_artifact(
        self,
        run_id: str,
        checkpoint_id: str,
        writer: Callable[[Path], None],
        name: str,
    ) -> StagedArtifact:
        """Write, flush, and verify a temporary staging artifact using the writer callback."""
        ...

    @abstractmethod
    def publish_artifact(
        self,
        run_id: str,
        checkpoint_id: str,
        staged: StagedArtifact,
        name: str,
    ) -> PublishedArtifact:
        """Atomically publish a staged artifact to its final immutable storage key.

        Guarantees zero-overwrite publication. Fails if the destination key already exists.
        """
        ...

    @abstractmethod
    def exists(self, artifact_key: str, run_id: str | None = None) -> bool:
        """Check whether an artifact exists at the specified storage key."""
        ...

    @abstractmethod
    def delete_artifact(self, artifact_key: str, run_id: str | None = None) -> None:
        """Remove an artifact from storage (e.g. during rollback or failed directory sync)."""
        ...

    @abstractmethod
    def verify_artifact(
        self,
        artifact_key: str,
        expected_size: int,
        expected_checksum: str,
        expected_run_id: str | None = None,
    ) -> tuple[bool, str | None]:
        """Verify artifact existence, containment, format, size, and SHA-256 integrity."""
        ...

    @abstractmethod
    def open_artifact(
        self,
        artifact_key: str,
        expected_run_id: str | None = None,
    ) -> BinaryIO:
        """Open a readable binary stream for the artifact without loading it all into memory.

        Callers are responsible for closing the returned stream.
        """
        ...

    def get_artifact_path(self, artifact_key: str, expected_run_id: str | None = None) -> Path:
        """Resolve and return local filesystem Path for this artifact if supported.

        This is a backend-specific capability (indicated by `capabilities.provides_local_path`).
        Remote providers that do not operate on the local filesystem do not override this method.

        Raises:
            TypeError: If this provider does not provide local filesystem paths.
        """
        _ = (artifact_key, expected_run_id)
        raise TypeError(
            f"Storage provider '{self.name}' does not provide local filesystem paths. "
            "Use open_artifact() or read_artifact_bytes() to access content."
        )

    def read_artifact_bytes(
        self,
        artifact_key: str,
        expected_run_id: str | None = None,
        max_bytes: int | None = None,
    ) -> bytes:
        """Read artifact bytes from storage with an optional bounded length limit.

        For large artifacts, prefer using `open_artifact()` to stream content incrementally.
        """
        with self.open_artifact(artifact_key, expected_run_id=expected_run_id) as stream:
            if max_bytes is not None:
                return stream.read(max_bytes)
            return stream.read()
