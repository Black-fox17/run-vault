# RunVault

> Open-source machine learning experiment management, artifact persistence, and execution platform.

RunVault is designed to make machine learning experiments easier to develop, execute, monitor, persist, and recover. It adheres to a **library-first philosophy**: the core SDK operates independently, offline, and without requiring a hosted server or running daemon.

---

## Intended Capabilities & Architecture

RunVault is designed around four architectural tiers that will be developed across planned phases:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          4. Studio & Notebooks                         │
│   (JupyterHub & JupyterLab Managed Environments with DockerSpawner)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    3. Platform API & Control Plane                     │
│    (FastAPI, Async PostgreSQL Metadata, Auth, Remote Execution)        │
└───────────────────▲────────────────────────────────┬───────────────────┘
                    │                                │
┌───────────────────┴───────────────┐  ┌─────────────▼───────────────────┐
│         1. RunVault SDK           │  │          2. RunVault UI         │
│  (Library-First Python Core,      │  │  (@runvault/ui React Library    │
│   Local SQLite, PyTorch Hooks,    │  │   & Standalone Web Dashboard)   │
│   Direct-to-Storage Checkpoints)  │  │                                 │
└───────────────────┬───────────────┘  └─────────────────────────────────┘
                    │
┌───────────────────▼────────────────────────────────────────────────────┐
│                       Pluggable Storage Engine                         │
│    (Local Filesystem, Direct Google Drive API, Future S3 / GCS)        │
└────────────────────────────────────────────────────────────────────────┘
```

1. **RunVault SDK (`runvault`):**
   - Lightweight, typed Python library.
   - Single package distribution with optional extras for framework and storage integrations.
   - Direct-to-storage data plane: heavy checkpoint binaries stream directly to storage, never routing through a metadata API server.
   - Built-in embedded SQLite metadata engine for standalone offline execution.

2. **RunVault Storage:**
   - Pluggable storage abstraction for artifacts and model checkpoints.
   - Direct local filesystem storage and resumable Google Drive uploads, architected for future S3/GCS compatibility.

3. **RunVault UI:**
   - Modular frontend architecture split between `@runvault/ui` (reusable, headless/styled React components) and `apps/web` (standalone dashboard).
   - Designed for embedding in external applications, web dashboards, and JupyterLab extensions.

4. **RunVault Platform & Studio:**
   - Optional hosted multi-user control plane backed by PostgreSQL and FastAPI.
   - Managed notebook execution leveraging standard JupyterHub and JupyterLab container images (`DockerSpawner` / `KubeSpawner`).

---

## Development Status

| Stage | Focus Area | Status |
|---|---|---|
| **Phase 1** | Repository Foundation & Engineering Tooling | **Completed** |
| **Phase 1.1** | Foundation Verification & Architecture Corrections | **Completed** |
| **Phase 2** | Core SDK & Local Experiment Tracking | **Completed** |
| **Phase 3** | Checkpoint Engine & Recovery | **Completed** |
| **Phase 4.1** | Storage Provider Architecture & Local Backend | **Completed** |
| **Phase 4.2** | Google Drive Checkpoint Storage Integration | **Completed** |
| **Phase 4.3** | PyTorch Integration & Training Recovery | **Completed** |
| **Phase 4.4+** | UI Component Library (`@runvault/ui`) & Dashboard | Planned |
| **Phase 5** | Standalone Dashboard (`apps/web`) | Planned |
| **Phase 6** | Platform Control Plane & Hosted API | Planned |
| **Phase 7** | Studio & Managed JupyterLab Environments | Planned |

---

## Quick Start (Local Experiment Tracking)

RunVault provides a lightweight, offline-first experiment tracking workflow requiring zero setup or external services.

### Installation

```bash
# Clone the repository
git clone https://github.com/Black-fox17/run-vault.git
cd run-vault

# Install runvault in editable mode
pip install -e .
```

### Logging an Experiment Run

```python
import runvault

# 1. Create or retrieve an experiment
experiment = runvault.create_experiment(
    "cifar10-classifier",
    description="ResNet baseline comparison",
)

# 2. Start a run using the context manager
with experiment.start_run(name="resnet18-baseline", tags={"architecture": "resnet18"}) as run:
    # 3. Log configuration parameters (JSON-serializable scalar or structured values)
    run.log_params(
        {
            "learning_rate": 0.001,
            "batch_size": 64,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
        }
    )

    # 4. Log scalar metrics across training steps
    for epoch in range(5):
        train_loss = 0.8 / (epoch + 1)
        val_loss = 0.85 / (epoch + 1)
        val_acc = 0.65 + 0.06 * epoch

        run.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            step=epoch,
        )

    # 5. Set additional run tags
    run.set_tags({"status": "validated", "dataset": "cifar10"})

# The run automatically transitions to FINISHED upon exiting the context.
# If an unhandled exception occurs inside the block, the run is marked FAILED
# with the exception type and traceback recorded.
```

### Inspecting Previous Experiments and Runs

```python
import runvault

# List all experiments
experiments = runvault.list_experiments()
for exp in experiments:
    print(f"Experiment: {exp.name} ({exp.id})")

# Retrieve an experiment by name and list its runs
experiment = runvault.get_experiment_by_name("cifar10-classifier")
if experiment:
    runs = experiment.list_runs()
    for r in runs:
        print(f"Run: {r.name} | Status: {r.status.value} | Duration: {r.duration_seconds:.2f}s")

        # Inspect logged parameters and metric history
        params = runvault.get_params(r.id)
        loss_curve = runvault.get_metric_history(r.id, metric_name="val_loss")
        for point in loss_curve:
            print(f"  Step {point.step}: val_loss = {point.value:.4f}")
```

### Checkpoint Creation & Fault-Tolerant Recovery

RunVault features an atomic, integrity-verified checkpoint engine with automatic fallback recovery:

```python
import runvault
from pathlib import Path

# 1. Start a training run and persist atomic checkpoints
experiment = runvault.create_experiment("training-experiment")
with experiment.start_run(name="resnet-run") as run:
    # A framework-agnostic writer callback persists binary artifacts
    # (e.g. torch.save(model.state_dict(), dest), pickle, safetensors)
    def save_weights(dest: Path) -> None:
        dest.write_bytes(b"... serialized model parameters ...")

    # Save an immutable, atomic checkpoint
    ckpt = run.save_checkpoint(
        name="epoch-5",
        writer=save_weights,
        step=5,
        metadata={"val_loss": 0.28, "val_acc": 0.91},
    )
    print(f"Persisted checkpoint: {ckpt.name} ({ckpt.checksum_sha256[:8]}...)")

    # 2. Restore the latest valid checkpoint directly
    # If the latest checkpoint artifact was corrupted or truncated,
    # RunVault automatically falls back to the newest valid older checkpoint.
    checkpoint_file = run.restore_checkpoint()
    if run.last_restored_is_fallback:
        print(f"Warning: Fell back to older checkpoint {run.last_restored_checkpoint.name}!")
    else:
        print(f"Restored latest checkpoint: {checkpoint_file}")

# 3. Retrieve or restore checkpoints outside of an active run context
restored_ckpt, artifact_path, is_fallback = runvault.restore_checkpoint(run.id)
print(f"Restored: {restored_ckpt.name} from {artifact_path}")
```

### Durability Guarantees & Platform Support

- **Atomic No-Overwrite Publication:** Checkpoints are written to unique staging files on the same filesystem and exclusively linked to their final `.bin` destination via `os.link`. Existing checkpoints can never be clobbered or overwritten.
- **Payload Durability:** The checkpoint engine synchronizes file buffers to non-volatile storage prior to publication, prioritizing `os.fdatasync` where available (Linux/POSIX) and falling back cleanly to `os.fsync` on platforms where `fdatasync` is not exposed (such as Windows Python builds). If neither primitive is available or synchronization fails, checkpoint creation raises `CheckpointCreationError` rather than claiming durability.
- **Directory Metadata Durability:** On POSIX filesystems, directory entries are synchronized using `os.fsync` on the containing directory file descriptor. Failures trigger immediate published-artifact rollback. On Windows, directory synchronization relies on NTFS metadata journaling.
- **Fail-Safe Cleanup & Error Integrity:** Failure during any step (writer callback, synchronization, hash calculation, or SQLite commit) guarantees clean rollback without masking primary exceptions. Secondary cleanup failures are attached via standard Python exception notes.
- **Exception Surface:** Canonical exception classes include `CheckpointCreationError`, `CheckpointNotFoundError`, `CheckpointCorruptError` (with `CorruptCheckpointError` available as a backward-compatible alias), and `CheckpointRecoveryError`.

### Storage Provider Architecture

RunVault decouples artifact storage from checkpoint orchestration through an explicit `StorageProvider` contract:

- **`StorageProvider` Protocol:** Defines `stage_artifact`, `publish_artifact`, `exists`, `delete_artifact`, `verify_artifact`, `get_artifact_path`, `open_artifact`, and `read_artifact_bytes`.
- **`StorageCapabilities`:** Declares backend characteristics (`supports_exclusive_publication`, `supports_directory_sync`, `provides_local_path`, `can_verify_integrity`).
- **`LocalStorageProvider`:** Default local filesystem backend encapsulating POSIX/Windows durability, staging cleanup diagnostics, path containment, and symlink rejections.
- **Pluggable & Injected:** Custom providers can be configured on `runvault.init(storage_provider=...)`, `ActiveRun`, or passed directly to `save_checkpoint` / `restore_checkpoint`. Remote providers without local filesystem paths are fully supported via streaming (`recovery.open()`, `result.read_bytes()`, and `result.read_text()`).

### Google Drive Checkpoint Storage (`runvault[gdrive]`)

RunVault supports direct-to-storage checkpointing with Google Drive, allowing models trained in Google Colab or ephemeral environments to persist directly into the user's Drive storage without routing binary payloads through any server.

#### 1. Installation

Install the optional `[gdrive]` extra:

```bash
pip install "runvault[gdrive]"
```

The core `runvault` package remains lightweight and stdlib-only; Google API libraries are only imported when the Google Drive integration is explicitly used.

#### 2. Google Colab Workflow

In a Google Colab notebook, RunVault automatically detects the environment and requests access using Colab's native user authentication:

```python
import runvault
from runvault.integrations.gdrive import GoogleDriveStorageProvider

# 1. Initialize provider (auto-authenticates via google.colab.auth)
gdrive_storage = GoogleDriveStorageProvider(root_folder_name="RunVaultCheckpoints")

# 2. Attach provider to the active run
with runvault.init(experiment="colab-resnet", storage_provider=gdrive_storage) as run:
    # 3. Save checkpoints directly to Google Drive
    run.save_checkpoint(
        name="epoch-10",
        writer=lambda dest: torch.save(model.state_dict(), dest),
        step=10,
        metadata={"val_acc": 0.942},
    )

    # 4. Restore checkpoint directly from Drive with streaming verification
    recovery = run.restore_checkpoint()
    with recovery.open() as stream:
        state_dict = torch.load(stream)
```

#### 3. Local OAuth / Application Default Credentials

Outside of Google Colab, RunVault supports desktop OAuth with secure local token caching or explicit credentials:

```python
from runvault.integrations.gdrive import GoogleDriveStorageProvider

# Authenticate via local OAuth browser consent flow with cached token
provider = GoogleDriveStorageProvider.from_oauth(
    client_secrets_file="client_secret.json",
    token_cache_path="~/.config/runvault/gdrive_token.json",
)
```

- **Credential Isolation & Permissions:** The token cache file is created with owner-only `0o600` permissions inside an owner-only `0o700` directory using atomic temporary-file replacement. Custom `token_cache_path` paths are expanded via `Path.expanduser()` to cleanly isolate different user accounts.
- **Sanitized Logging:** Failed refresh attempts or credential errors log only exception class names (`type(err).__name__`), never leaking tokens or authorization headers to stdout or log collectors.

#### 4. Durability, Streaming & Security Invariants

- **Direct-to-Storage Plane:** Artifact binaries stream directly between client runtime and Google Drive using resumable chunked HTTP uploads (`MediaFileUpload`) and chunked downloads (`MediaIoBaseDownload`). Checkpoint binaries never proxy through an API server.
- **Upload Idempotency & Lost Response Recovery:** If a resumable upload completes on Google Drive but the HTTP response is dropped due to a client timeout or network disconnect, RunVault queries existing file metadata by `appProperties.runvault_checkpoint_id` and sha256 checksum before retrying, safely recovering the uploaded artifact without duplicating files.
- **Preflight & Post-Upload Collision Detection:** Immutability is strictly enforced. Preflight queries verify that neither the filename nor the checkpoint ID exists in the target run folder. In the event of a simultaneous race condition between workers, a post-upload verification query detects duplicates, cleans up the conflicting artifact via compensation deletion, and raises `CheckpointCreationError`.
- **Bounded-Memory Streaming:** Checkpoints are read via `GoogleDriveDownloadStream`, which enforces $O(\text{chunk\_size})$ memory usage by swapping chunk buffers during pagination rather than accumulating chunks. The stream satisfies the full standard Python binary I/O contract (`read`, `readinto`, `readline`, `readlines`, `seekable`, `readable`).
- **Transient Error Backoff:** Transient network failures and rate limits (HTTP 429, 500, 502, 503, 504, timeouts) are retried with exponential backoff and randomized jitter.
- **Accurate Error Classification:** Authentication failures (HTTP 401/403) raise `StorageUnavailableError` (halting corrupted fallbacks), whereas HTTP 404 and trashed files report non-existence for safe older-checkpoint fallback recovery. Programming errors (`TypeError`, `ValueError`) propagate immediately.
- **Self-Healing Folder Cache:** Root and run folder IDs are cached in-memory for performance, but automatically invalidated and re-queried if folders are modified, deleted, or moved externally.
- **Failure Compensation:** If a database metadata commit fails after an artifact has been uploaded to Google Drive, RunVault executes compensation cleanup to remove the orphan remote file.
- **Least-Privilege Scopes:** Uses `https://www.googleapis.com/auth/drive.file`, granting access exclusively to files and folders created by RunVault.
- **Zero Credential Persistence:** Secrets, tokens, and OAuth credentials are never stored in SQLite database tables or metadata files.
- **Shared Drive Notice:** Shared Drives (formerly Google Team Drives) require explicit multi-tenant permission flags (`supportsAllDrives=True`) and domain-level policy handling; they are unsupported in Phase 4.2.1 and standard user My Drive folders must be used.

### PyTorch Integration & Training Recovery (`runvault[torch]`)

RunVault provides first-class PyTorch training support designed for standard PyTorch loops without requiring heavyweight wrapper frameworks (such as Lightning or Accelerate).

#### 1. Installation

Install the optional `[torch]` extra:

```bash
pip install "runvault[torch]"
```

The core `runvault` package remains stdlib-only; PyTorch is never imported unless the PyTorch integration is explicitly invoked.

#### 2. Training Loop Integration

Use `TorchCheckpointManager` or `save_torch_checkpoint` to persist model weights, optimizer buffers, learning rate schedulers, AMP GradScalers, and RNG seeds:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import runvault
from runvault.integrations.torch import TorchCheckpointManager, save_torch_checkpoint

model = nn.Linear(128, 10)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

with runvault.init("resnet-cifar", name="train-baseline") as run:
    run.log_params({"lr": 1e-3, "batch_size": 64})
    manager = TorchCheckpointManager(
        run=run,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step_interval=500,  # Save checkpoint every 500 steps
        epoch_interval=1,  # Save checkpoint at each epoch boundary
    )

    for epoch in range(1, 11):
        for step, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Record step metrics & trigger step interval checkpoints
            manager.step_end(global_step, metrics={"train_loss": loss.item()})
            global_step += 1

        scheduler.step()
        # Record epoch metrics & persist epoch boundary checkpoint
        manager.epoch_end(
            epoch, step=global_step, metrics={"val_loss": val_loss}, best_metric=val_loss
        )
```

#### 3. Interrupted Training Resume & Recovery

Resume training seamlessly after unexpected termination, runtime preemption, or hardware failure:

```python
from runvault.integrations.torch import restore_torch_checkpoint

model = nn.Linear(128, 10)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Restore latest valid checkpoint into live model, optimizer, and RNG states
with runvault.init("resnet-cifar", name="resumed-run") as run:
    recovery = restore_torch_checkpoint(
        run_or_id="previous-run-id",
        model=model,
        optimizer=optimizer,
    )
    print(f"Resumed from checkpoint: {recovery.checkpoint.name}")
    print(f"Starting at Epoch: {recovery.epoch + 1}, Step: {recovery.step}")
    if recovery.is_fallback:
        print("Notice: Fell back to earlier valid checkpoint due to corruption in latest.")

    start_epoch = (recovery.epoch or 0) + 1
    global_step = recovery.step or 0

    # Resume training loop from restored epoch and step...
```

#### 4. Google Colab + Google Drive Workflow

In Google Colab, ephemeral runtimes can disconnect without warning. By combining `GoogleDriveStorageProvider` with `runvault[torch]`, training checkpoints stream directly into Google Drive storage and can be resumed across runtime instances:

```python
import runvault
from runvault.integrations.gdrive import GoogleDriveStorageProvider
from runvault.integrations.torch import TorchCheckpointManager, restore_torch_checkpoint

# 1. Direct-to-Drive remote storage (auto-authenticates in Colab)
gdrive = GoogleDriveStorageProvider(root_folder_name="ColabTrainingVault")

# 2. Check if a previous checkpoint exists to resume from
with runvault.init("colab-transformer", storage_provider=gdrive) as run:
    latest = run.get_latest_valid_checkpoint()
    if latest:
        rec = restore_torch_checkpoint(run, model, optimizer=optimizer, scheduler=scheduler)
        start_epoch = (rec.epoch or 0) + 1
    else:
        start_epoch = 1

    # 3. Training with automatic epoch checkpoints streamed directly to Drive
    manager = TorchCheckpointManager(run=run, model=model, optimizer=optimizer, scheduler=scheduler)
    for epoch in range(start_epoch, 50):
        # ... training step loop ...
        manager.epoch_end(epoch, metrics={"val_acc": val_acc})
```

#### 5. Serialization Security & Safe Loading

RunVault enforces **safe loading (`weights_only=True`) by default**:

- **Safe Deserialization (`safe_load=True`):** Checkpoints deserialize with PyTorch's `weights_only=True`. RunVault serializes RNG states (including NumPy) as clean Python primitives and tensors, allowing full state reconstruction without unpickling arbitrary Python classes.
- **Trusted Checkpoint Requirement (`safe_load=False`):** If custom objects require general unpickling, callers must explicitly pass `safe_load=False`. A prominent warning is logged.
- **Integrity vs Authenticity:** RunVault's SHA-256 verification guarantees that an artifact has not suffered bit rot, truncation, or network corruption. It does **not** verify author authenticity. Never load unverified third-party checkpoints with `safe_load=False`.

#### 6. RNG & Reproducibility Boundary

RunVault captures and restores:
- Python standard library `random` state
- NumPy `numpy.random` state (when installed)
- PyTorch CPU RNG state (`torch.get_rng_state()`)
- PyTorch CUDA RNG states for all available devices (`torch.cuda.get_rng_state_all()`)

**Limitations & Boundary:**
Exact mathematical reproducibility cannot be guaranteed purely through RNG state dicts. The reproducibility boundary is limited by:
- Multi-processing DataLoader workers: Worker processes initialized prior to RNG restoration maintain independent random states unless explicitly re-seeded in `worker_init_fn`.
- Asynchronous data prefetching and multi-threading.
- Non-deterministic CUDA kernels (e.g., atomic additions in backward passes). Strict algorithmic determinism requires setting `torch.use_deterministic_algorithms(True)` and `torch.backends.cudnn.deterministic = True`.

### Local Workspace Location

By default, RunVault stores metadata locally in an embedded SQLite database and checkpoint artifacts in:
```text
<current-working-directory>/.runvault/metadata.db
<current-working-directory>/.runvault/checkpoints/<run_id>/<checkpoint_id>.bin
```

The workspace directory can be customized via:
1. The `RUNVAULT_DIR` environment variable:
   ```bash
   export RUNVAULT_DIR=/path/to/custom/runvault_workspace
   ```
2. The explicit `workspace` argument on SDK functions:
   ```python
   exp = runvault.create_experiment("my-exp", workspace="/tmp/custom_workspace")
   ```

### Current Scope & Limitations

RunVault Phase 4.3 establishes the core Python SDK, local metadata engine, pluggable storage provider architecture, direct-to-storage Google Drive integration, and PyTorch training integration. The following capabilities are **not yet implemented** and belong to subsequent phases:
- UI Component Library (`@runvault/ui`) (Phase 4.4+).
- Standalone Dashboard (`apps/web`) (Phase 5).
- Additional cloud storage backends (AWS S3, Google Cloud Storage).
- Hosted metadata platform, FastAPI server, and PostgreSQL (Phase 6).
- Studio and managed JupyterLab environments (Phase 7).

---

## Development Setup

### Prerequisites
- **Python:** `3.11` or `3.12` (with `uv` recommended for package and environment management)
- **Node.js:** `>= 20.0.0` (v22 LTS recommended)
- **pnpm:** `>= 9.0.0` (v10 recommended)
- **Git:** Git 2.40+ configured with a `main`-based workflow

### Python Environment Setup

Using `uv` (recommended):
```bash
# Clone the repository
git clone https://github.com/your-org/run-vault.git
cd run-vault

# Create virtual environment and install in editable mode with development dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run Python checks:
```bash
# Linting & Formatting
ruff check .
ruff format --check .

# Strict Type Checking
pyright

# Unit & Verification Tests
pytest
```

### Frontend Workspace Setup

```bash
# Install workspace dependencies using pnpm
pnpm install

# Typecheck all packages
pnpm typecheck

# Run frontend tests
pnpm test

# Build all workspace packages and apps
pnpm build
```

---

## License

Licensed under the Apache License, Version 2.0 (the "License"). See `LICENSE` for details.
