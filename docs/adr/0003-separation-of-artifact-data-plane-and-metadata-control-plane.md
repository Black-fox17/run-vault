# ADR 0003: Separation of Artifact Data Plane and Metadata Control Plane

## Status
Accepted

## Context
In machine learning workflows, data volume exhibits an extreme bimodal distribution:
- **Metadata (Control Plane):** Run configurations, hyperparameters, tags, scalar metric streams (loss, accuracy), and status states. Highly structured, high frequency, small payload sizes (bytes to kilobytes).
- **Artifacts (Data Plane):** Model weights (`.pt`, `.safetensors`), optimizer states, checkpoint archives, dataset splits, and audio/image samples. Binary blobs, low-to-medium frequency, massive payload sizes (megabytes to hundreds of gigabytes).

Routing binary artifact streams through a centralized metadata API gateway introduces massive CPU/memory bottlenecks, exhausts socket pools, causes network timeouts, and incurs double egress costs.

## Decision
1. **Zero-Proxy Architecture:** The RunVault metadata API server shall never act as an intermediary proxy for model checkpoint or artifact binary data.
2. **Direct Storage Transfer:** The RunVault Python SDK communicates directly with the designated storage provider (local disk, Google Drive API, S3/GCS bucket) for all artifact uploads and downloads.
3. **Control Plane Records Only Descriptors:** The metadata control plane receives only artifact descriptors: storage provider URI/key, content length, SHA-256 digest, MIME type, and checkpoint manifest metadata.

## Consequences
### Positive
- The metadata API server remains lightweight, horizontally scalable, and protected from throughput exhaustion.
- Artifact transfers maximize throughput by communicating directly with high-bandwidth object storage endpoints.
- Egress costs and intermediate compute overhead are eliminated.

### Negative
- Client SDK must possess necessary storage credentials or negotiate presigned authorization directly with storage providers.
- Multipart/resumable upload logic must be implemented reliably within the client storage drivers.

## Alternatives Considered
- **Centralized API Proxying (`POST /api/v1/runs/{id}/artifacts`):** Rejected because routing multi-gigabyte checkpoints through an application server causes buffer bloat and system instability.

## Open Implementation Details
- Negotiation protocol for requesting short-lived presigned upload URLs from the hosted control plane for private cloud buckets (S3/GCS).
- Handling direct client token refresh for Google Drive OAuth sessions during long-running uploads.
