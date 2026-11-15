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
| **Phase 3** | Framework Integrations & Checkpoint Engine | Planned |
| **Phase 4** | UI Component Library (`@runvault/ui`) | Planned |
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

### Local Workspace Location

By default, RunVault stores metadata locally in an embedded SQLite database at:
```text
<current-working-directory>/.runvault/metadata.db
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

RunVault Phase 2 establishes the core Python SDK and local metadata engine. The following capabilities are **not yet implemented** and belong to subsequent phases:
- Checkpoint serialization and recovery (Phase 3).
- Model artifact storage and Google Drive / S3 cloud storage providers (Phase 3).
- PyTorch framework hooks and training loop callbacks (Phase 3).
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
