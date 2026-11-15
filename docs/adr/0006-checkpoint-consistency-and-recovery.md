# ADR 0006: Checkpoint Consistency and Recovery Requirements

## Status
Accepted

## Context
Model checkpointing is mission-critical in ML engineering. Training runs often span hours, days, or weeks across expensive GPU infrastructure. If a training process is terminated unexpectedly (e.g., node preemption, out-of-memory crash, hardware fault), the last checkpoint must be guaranteed to be intact and recoverable.

Common failure modes in naive checkpointing:
- Corrupted weight files from partially written files during crash.
- Mismatched random seed states causing non-deterministic resumption.
- Incomplete optimizer or learning rate scheduler states leading to gradient explosions on resume.
- Loss of exact hyperparameter and environment versioning.

## Decision
1. **Atomic Local Writes:**
   - Checkpoints must never be written directly to their target destination file.
   - The SDK writes first to a temporary staging file on the same filesystem (`.<filename>.<uuid>.tmp`).
   - The file is explicitly flushed to physical media using `os.fdatasync()`.
   - The file is atomically moved to its target path via `os.replace()`.
2. **Cryptographic Checksum Verification:**
   - Checkpoints compute a SHA-256 digest during write/upload.
   - On download or recovery, the SHA-256 digest must be validated before the checkpoint is handed to the framework (e.g., `torch.load()`).
3. **Immutable Checkpoint Manifests:**
   - Every checkpoint file is paired with an immutable `CheckpointManifest` containing:
     - Checkpoint UUID, Run ID, Step, Epoch.
     - SHA-256 digest and byte size.
     - Framework and library version signatures.
     - RNG states (PyTorch CPU/CUDA, NumPy, Python standard random).
     - Hyperparameter hash and parent checkpoint lineage.
4. **Resumption Contract:** The SDK must provide an explicit `run.resume_from_checkpoint(checkpoint_id)` method that restores weights, optimizer states, epoch/step counters, and random seeds deterministically.

## Consequences
### Positive
- Zero risk of corrupted checkpoint files from interrupted writes.
- Exact reproducibility and seamless resumption after node preemptions.

### Negative
- Temporary staging requires disk space headroom equal to at least one full checkpoint size.
- Computing SHA-256 over multi-gigabyte models adds minor CPU time, mitigated by chunked streaming.

## Alternatives Considered
- **Direct in-place writes:** Rejected due to guaranteed data loss when crashes occur mid-write.

## Open Implementation Details
- Checkpoint rotation / retention policies (e.g., keeping only the top-$K$ best checkpoints and last-$N$ steps to prevent disk exhaustion).
- Distributed checkpointing (handling PyTorch Distributed / FSDP sharded state dicts in later phases).
