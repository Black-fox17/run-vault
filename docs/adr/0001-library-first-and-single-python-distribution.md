# ADR 0001: Library-First Architecture and Single Python Distribution

## Status
Accepted

## Context
RunVault aims to serve machine learning practitioners ranging from solo researchers training models on a single workstation or laptop, to distributed research teams operating on shared GPU clusters, to enterprises needing centralized dashboards and governance.

Many ML developer tools suffer from one of two failure modes:
1. **Server-dependent lock-in:** Forcing the user to start a local server daemon, docker container, or remote API before `import runvault` works creates high friction and fails in air-gapped, cluster batch (Slurm/PBS), or ephemeral environments.
2. **Micro-package sprawl:** Splitting capabilities into dozens of distinct packages (`runvault-core`, `runvault-api`, `runvault-client`, `runvault-storage`) creates release management overhead, version skew, and confusion for end users.

## Decision
1. **Library-First Philosophy:** The Python SDK is the foundational citizen of the RunVault ecosystem. It must function fully independently, offline, and without requiring any server, network connection, or daemon process.
2. **Single Python Distribution for SDK:** Client capabilities are published to PyPI as a single distribution package (`runvault`). Internal modularity is enforced via Python subpackages (`runvault.core`, `runvault.storage`, `runvault.integrations`). Hosted server infrastructure is maintained as a separate deployable service project, strictly excluded from the SDK package.
3. **Optional Dependency Groups (Extras):** Features requiring heavy framework or storage dependencies are declared as PEP 621 extras (e.g., `runvault[torch]`, `runvault[gdrive]`). The base installation `pip install runvault` installs only the lightweight core.

## Consequences
### Positive
- Zero friction for data scientists: `pip install runvault` works out of the box anywhere.
- Unbroken offline workflows: Experiments can be conducted in offline or restricted network environments.
- Simple distribution: One version number, one tag, one release pipeline on PyPI.

### Negative
- Requires strict internal discipline to ensure `runvault.core` never imports from optional submodules.
- Extra testing matrix combinations in CI to ensure base and extra configurations work cleanly.

## Alternatives Considered
- **Micro-repositories and multi-package publishing:** Rejected due to dependency hell, version drift, and cross-repo coordination overhead.
- **Server-mandatory agent model:** Rejected because it alienates data scientists working in HPC clusters and interactive notebooks.

## Open Implementation Details
- CLI entry points for local exploration tools vs server management.
- Dynamic plugin discovery mechanism for community-contributed storage backends.
