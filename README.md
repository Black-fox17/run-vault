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
| **Phase 1** | Repository Foundation & Engineering Tooling | **In Progress** |
| **Phase 2** | Core SDK & Pluggable Storage Engine | Planned |
| **Phase 3** | PyTorch Integration & Checkpoint Recovery | Planned |
| **Phase 4** | UI Component Library (`@runvault/ui`) | Planned |
| **Phase 5** | Standalone Dashboard (`apps/web`) | Planned |
| **Phase 6** | Platform Control Plane & Hosted API | Planned |
| **Phase 7** | Studio & Managed JupyterLab Environments | Planned |

> **Note:** RunVault is currently in foundational scaffolding (Phase 1). SDK APIs, checkpoint engines, telemetry systems, storage drivers, and dashboard screens have not yet been implemented.

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
