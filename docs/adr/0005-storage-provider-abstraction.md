# ADR 0005: Storage Provider Abstraction

## Status
Accepted

## Context
Machine learning artifacts must be stored in diverse locations depending on the runtime environment:
- Local developer workstation: Local filesystem (`file://` or path).
- Google Colab / personal drive: Google Drive API (`gdrive://`).
- Enterprise cloud / cluster: S3 (`s3://`) or Google Cloud Storage (`gs://`).

Coupling the checkpointing or artifact management logic to a specific storage backend would severely restrict portability.

## Decision
1. **Abstract Storage Contract:** Define a formal `StorageProvider` interface in `runvault.storage.base` with required operations:
   - `upload_file(local_path: Path, remote_key: str, progress_callback: Optional[ProgressCallback] = None) -> StorageDescriptor`
   - `download_file(remote_key: str, destination_path: Path, expected_sha256: Optional[str] = None, progress_callback: Optional[ProgressCallback] = None) -> Path`
   - `exists(remote_key: str) -> bool`
   - `get_metadata(remote_key: str) -> StorageMetadata`
   - `delete(remote_key: str) -> None`
2. **Pluggable Architecture:**
   - `LocalStorageProvider`: Included in core; manages local disk directories with atomic writes.
   - `GoogleDriveStorageProvider`: Implemented in `runvault.storage.gdrive` (under `[gdrive]` extra) utilizing resumable direct uploads via Google Drive API v3.
   - Future Providers (`S3StorageProvider`, `GCSStorageProvider`): Implement the identical contract.
3. **Storage URI Resolution:** Storage targets are identified via uniform URIs (e.g., `file:///path/to/artifacts`, `gdrive://folder-id/artifacts`).

## Consequences
### Positive
- Code that trains and checkpoints models is completely agnostic to whether artifacts reside on local NVMe drives, Google Drive, or cloud buckets.
- Provider implementations can be unit tested in isolation using synthetic fixtures.

### Negative
- Cloud providers (Google Drive vs S3) have different latency, eventual consistency, and chunking semantics that must be normalized behind the interface.

## Alternatives Considered
- **Direct reliance on fsspec:** While powerful, `fsspec` introduces dozens of dependencies and complex caching layers. A small, explicit RunVault storage protocol keeps the core dependency-free and tailored to checkpoint validation requirements.

## Open Implementation Details
- Chunk sizing strategy for Google Drive resumable uploads (balancing memory buffer size with upload retry overhead).
- Handling hierarchical folder structures in Google Drive (Drive uses non-hierarchical ID-based references, requiring a path-to-ID mapping cache).
