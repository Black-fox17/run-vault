# RunVault — Agent Instructions and Engineering Rules

> Mandatory instructions and behavioral guardrails for all AI coding agents working on the RunVault codebase.

---

## 1. Architectural Integrity & Package Boundaries

- **Library-First Invariant:** The Python SDK (`runvault`) must operate independently, offline, and without requiring any server, daemon, or external service. Never introduce mandatory server dependencies to the SDK core.
- **Single Python Distribution for SDK:** Keep all client SDK capabilities within the single `runvault` distribution package using optional integration extras (such as `[torch]` and `[gdrive]`). Hosted server infrastructure belongs to a separate deployable project and must never be coupled into the SDK distribution package.
- **Lightweight Core:** The core `runvault` package must rely on the Python standard library wherever practical. Do not make heavy third-party packages (e.g., Pydantic, PyTorch, FastAPI, SQLAlchemy, Google API libraries) mandatory in the SDK core.
- **Direct-to-Storage Data Plane:** Model checkpoints and artifact binaries must stream directly between the client SDK and storage providers. Binaries must never pass through the metadata API server.
- **Frontend Modularity:** Keep reusable React UI components inside `packages/ui` (`@runvault/ui`). The dashboard (`apps/web`) is a consumer of `@runvault/ui`. Do not leak application-specific business logic or routes into `packages/ui`.
- **Preserve Clean Layers:** Keep domain business logic strictly decoupled from transport layers, presentation components, CLI endpoints, and API route handlers.

---

## 2. Typing & Code Standards

- **Strict Python Typing:**
  - All Python code must satisfy `pyright` in strict mode (`typeCheckingMode = "strict"`).
  - Explicit type annotations are required on all function arguments, return values, and module-level constants.
  - Do not use `# type: ignore` or Pyright suppressions unless accompanied by a clear, documented comment explaining why an external library's types are broken.
- **Strict TypeScript Standards:**
  - All TypeScript code must compile under `tsconfig.base.json` strict settings.
  - **Zero `any` Policy:** Do not use `any` or loose type assertions (`as unknown as ...`). Use explicit union types, generics, or `unknown` with runtime type guards.
- **Code Style & Formatting:**
  - Python: Enforce `ruff check` and `ruff format`.
  - Do not introduce custom formatting styles or disable linter rules without justification.

---

## 3. Dependency & Performance Discipline

- **Keep Optional Dependencies Optional:** Any code importing optional integrations (such as PyTorch, Google Drive, or FastAPI) must isolate imports within designated submodules and handle missing dependencies with actionable `ImportError` messages directing users to the appropriate installation extra (e.g., `pip install runvault[torch]`).
- **Non-Blocking Telemetry:** Never perform synchronous network or disk I/O inside machine learning training step loops. Telemetry must be collected in-memory and flushed asynchronously.
- **Atomic File Writes:** When persisting checkpoints or local metadata, use atomic write semantics (write to temporary file, `fdatasync`, atomic replace) to prevent data corruption during process interruption or crash.
- **Never Silently Swallow Errors:** Always handle exceptions explicitly. Never use bare `except:` or catch `Exception` and pass silently without logging or re-raising.

---

## 4. Engineering Workflow & Hygiene

- **Inspect Before Mutating:** Always inspect existing code, directory structures, and tests before making modifications.
- **No Unrelated Refactoring:** Focus strictly on the assigned task. Do not reformat, rename, or reorganize unrelated files or modules.
- **No Placeholder Business Logic:** Do not commit fake stub implementations or mock bypasses just to make a test pass. Implement the actual contract or do not claim it is implemented.
- **Test-Driven Verification:** Every new feature, bug fix, or abstraction must be accompanied by unit or integration tests (`pytest` for Python, `vitest` for TypeScript).
- **Run Mandatory Quality Checks:** Before declaring any task complete, execute the full suite of verification commands:
  - Python: `ruff check .`, `ruff format --check .`, `pyright`, `pytest`
  - Frontend: `pnpm typecheck`, `pnpm test`, `pnpm build`
