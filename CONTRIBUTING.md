# Contributing to RunVault

Thank you for your interest in contributing to RunVault. This document outlines our engineering standards, development workflows, and contribution guidelines.

---

## Git Workflow

- **Branching Model:** We use a `main`-based development workflow. All pull requests target the `main` branch.
- **Commit Messages:** Follow the Conventional Commits specification:
  - `feat: add local filesystem storage provider`
  - `fix: resolve race condition in telemetry queue flush`
  - `docs: update storage abstraction ADR`
  - `refactor: extract atomic write utility in storage engine`
- **Clean History:** Keep pull requests focused on a single change. Rebase against `main` before submitting.

---

## Python Development Standards

- **Python Versions:** We target Python 3.11 and 3.12 in CI. Code must remain compatible with `>=3.11`.
- **Package Management:** We recommend `uv` for managing dependencies and virtual environments.
- **Strict Typing:** All code must pass `pyright` in strict mode without errors.
- **Linting & Formatting:** We use `ruff`:
  ```bash
  ruff check .
  ruff format --check .
  ```
- **Testing:** We use `pytest`:
  ```bash
  pytest
  ```

---

## Frontend Development Standards

- **Package Manager:** `pnpm` (version 9+ or 10+) is required. Do not use npm or yarn.
- **TypeScript:** Strict mode is enforced across the workspace. Implicit `any` is strictly prohibited.
- **Monorepo Packages:**
  - `packages/ui`: Reusable component library. No application routing or backend API calls allowed here.
  - `apps/web`: Standalone dashboard application.
- **Verification Commands:**
  ```bash
  pnpm typecheck
  pnpm test
  pnpm build
  ```

---

## Pull Request Checklist

Before submitting a pull request, ensure:
1. `ruff check .` and `ruff format --check .` pass.
2. `pyright` reports 0 errors in strict mode.
3. `pytest` passes all tests.
4. `pnpm typecheck` reports 0 TypeScript errors.
5. `pnpm test` passes all frontend unit tests.
6. `pnpm build` builds all workspace packages successfully.
7. Any architectural changes are documented in `docs/adr/`.
