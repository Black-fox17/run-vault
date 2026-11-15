# ADR 0009: JupyterHub and JupyterLab as Studio Foundation

## Status
Accepted

## Context
Data scientists and ML researchers spend significant time in interactive notebook environments (Jupyter, Google Colab). The RunVault Studio capability aims to provide managed, reproducible, persistent notebook workspaces with GPU access and direct integration with RunVault experiment tracking.

Building an interactive web-based notebook execution platform from scratch requires solving:
- Language server protocols, syntax highlighting, and rich cell outputs.
- Kernel lifecycle management (IPython / Jupyter kernels).
- Multi-user authentication, isolation, and session routing.
- Web terminal multiplexing and file tree navigation.
- GPU container orchestration and storage volume mounting.

Reinventing these established components would drain engineering bandwidth from RunVault's core value proposition.

## Decision
1. **Leverage Jupyter Ecosystem:** RunVault Studio will use **JupyterHub** as the multi-user orchestrator and **JupyterLab** as the workspace interface.
2. **Container Spawner Architecture:**
   - Single-node / on-premise deployments: `DockerSpawner` launching containerized user environments on the host Docker daemon.
   - Cloud / Kubernetes deployments: `KubeSpawner` scheduling pods on Kubernetes clusters with GPU resource quotas.
3. **Pre-Baked Environment Images:**
   - RunVault will maintain standardized container images containing Python, CUDA runtime, PyTorch, JupyterLab, and the `runvault` SDK pre-installed.
4. **Persistent Workspace Storage:**
   - User home directories (`/home/jovyan/work`) are mounted to persistent block or network volumes (host bind-mounts or Kubernetes Persistent Volume Claims) to survive container restarts.
5. **RunVault Extension:**
   - An extension embedding `@runvault/ui` components will be provided for JupyterLab, allowing researchers to monitor runs and inspect checkpoints directly within their notebook sidebar.

## Consequences
### Positive
- Avoids rebuilding years of notebook infrastructure.
- Zero learning curve: ML practitioners already know and use JupyterLab.
- Enterprise-grade authentication (OAuth2, GitHub, LDAP) out of the box via JupyterHub authenticators.

### Negative
- Requires maintaining Docker images and spawner configuration templates.
- Tighter coupling to the Jupyter architecture for managed notebook sessions.

## Alternatives Considered
- **Custom Web IDE / Execution Engine from Scratch:** Rejected as a massive, high-risk distraction from core ML platform goals.
- **VS Code Server (code-server):** Evaluated as a potential future workspace alternative, but JupyterLab remains the primary standard for interactive data science and experiment exploration.

## Open Implementation Details
- OAuth token exchange between JupyterHub user sessions and the RunVault Control Plane API.
- Automated GPU quota allocation and idle workspace shutdown policies.
