# ADR 0002: Lightweight Core and Optional Integration Dependencies

## Status
Accepted

## Context
Machine learning environments are notoriously fragile with respect to dependency resolution (CUDA driver variants, PyTorch C++ ABI bindings, NumPy versions). If the core experiment tracking library brings in heavy third-party dependencies (such as Pydantic, PyTorch, FastAPI, SQLAlchemy, or Google Cloud client libraries), it risks triggering dependency conflicts in user environments or inflating image sizes by hundreds of megabytes.

## Decision
1. **Lightweight Standard-Library Core:** The core `runvault` SDK will rely exclusively on the Python standard library wherever practical (`sqlite3`, `dataclasses`, `typing`, `json`, `pathlib`, `hashlib`, `concurrent.futures`, `urllib`).
2. **No Mandatory Pydantic in Core SDK:** Domain validation and data representations in `runvault.core` will use standard library `dataclasses` and explicit typing rather than requiring Pydantic at the base layer.
3. **Lazy and Isolated Integrations:** Integrations with training frameworks (`torch`) and cloud storage (`google-api-python-client`) must live in dedicated submodules and only be imported on demand.
4. **Actionable Import Errors:** When an optional submodule is accessed without its corresponding extra installed, RunVault must raise an explicit `ImportError` describing the missing extra and providing the exact `pip install runvault[<extra>]` command.

## Consequences
### Positive
- Negligible installation footprint and rapid install time (< 2 seconds).
- Minimal risk of diamond dependency conflicts with user ML project dependencies.
- Can run in constrained or legacy Python environments without wheel compilation issues.

### Negative
- Requires manual schema validation and serialization helpers in pure Python rather than leveraging Pydantic's automatic parsing in the core layer.

## Alternatives Considered
- **Pydantic v2 Core:** Rejected because while Pydantic v2 is fast (compiled Rust), it introduces a compiled C/Rust binary dependency and external dependency requirement into what should be a zero-friction base library.
- **Monolithic Dependencies:** Rejected due to dependency bloat.

## Open Implementation Details
- Exact helper function signature for lazy import gating.
- Protocol for handling optional storage backends without runtime overhead.
