# ADR 0007: Bounded, Failure-Aware Telemetry Requirements

## Status
Accepted

## Context
During deep learning training, metrics (`loss`, `accuracy`, `learning_rate`, custom validation scalars) are computed inside tight computational loops running at high frequencies (hundreds of steps per second).

If metric logging performs synchronous disk I/O or network requests:
- The GPU is starved of work while waiting for I/O (training bubble / GPU utilization drops).
- Network jitter or transient disconnections can cause the entire training run to crash with an unhandled exception.

Conversely, naive asynchronous logging using unbounded in-memory queues risks crashing the process with Out-Of-Memory (OOM) errors if the consumer thread falls behind the producer loop.

## Decision
1. **Non-Blocking In-Memory Producer:** `run.log(...)` pushes metric records into an in-memory queue and returns immediately to the training loop.
2. **Bounded Queue Capacity:** The queue has a configurable high watermark (bounded capacity). If the buffer fills, the collector applies an explicit, configurable drop or throttle policy to prevent host memory exhaustion.
3. **Decoupled Background Consumer:** A dedicated background daemon thread drains the queue and batches records for persistence:
   - Batched writes to local SQLite (using WAL mode and single-transaction multi-row inserts).
   - Batched HTTP POST requests to remote platform endpoint when configured.
4. **Dual Flush Triggers:** The background consumer flushes records whenever the buffer reaches a batch size threshold (e.g., 100 items) or an elapsed time interval (e.g., 1.0 second), whichever occurs first.
5. **Deterministic Lifecycle Drain:** On context manager exit (`with runvault.init() as run:`) or explicit `run.finish()`, the queue is completely drained and flushed before the process exits.
6. **Failure Isolation:** Telemetry failures (e.g., remote server unreachable) must log warnings or retry with exponential backoff; they must never terminate or crash the user's training loop.

## Consequences
### Positive
- Zero measurable overhead on training step execution speed.
- Resilient to transient network drops without losing metric continuity.
- Clean process shutdown guarantees all metrics recorded up to termination are persisted.

### Negative
- Requires robust thread lifecycle management and graceful shutdown handling (`atexit`, signal handling).

## Alternatives Considered
- **Synchronous disk/network writes:** Rejected due to severe performance degradation of GPU training loops.
- **Unbounded logging queues:** Rejected due to OOM risk when training produces high-frequency metrics faster than I/O drains.

## Open Implementation Details
- Compression format for remote batch metric payloads (e.g., gzip vs zstandard).
- Downsampling / summarization options for ultra-high-frequency step logs.
