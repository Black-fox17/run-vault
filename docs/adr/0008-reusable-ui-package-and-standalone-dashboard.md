# ADR 0008: Reusable UI Component Package and Standalone Dashboard

## Status
Accepted

## Context
RunVault requires visualization and experiment management across multiple UI surfaces:
1. A standalone web dashboard for team experiment exploration, comparison, and platform administration.
2. Embedded views inside JupyterLab / Jupyter notebooks.
3. Potential future embedding inside VS Code extensions or third-party web portals.

If the frontend is built as a monolithic web application, components (loss charts, run tables, checkpoint inspectors) cannot be easily extracted, versioned, or embedded into notebook environments.

## Decision
1. **Frontend Monorepo Structure:** Use a `pnpm` workspace dividing the frontend into:
   - `packages/ui` (`@runvault/ui`): Reusable, themeable React component library. Contains metric charts, run status badges, comparison tables, and artifact trees. Exposes clean TypeScript interfaces and dual ESM/CJS exports.
   - `apps/web`: Standalone React dashboard application consuming `@runvault/ui`. Contains application routing, authentication, state management, and server communication.
2. **Strict Package Boundaries:**
   - `packages/ui` must not contain application routing or hardcoded API endpoints. All data must be passed via props or context providers.
   - `apps/web` handles orchestration and routes.
3. **Build Tooling:**
   - `packages/ui` uses `tsup` for rapid ESM/CJS bundling and `.d.ts` declaration generation.
   - `apps/web` uses `vite` for modern development and fast production builds.

## Consequences
### Positive
- High component reuse across the standalone dashboard and notebook/Studio extensions.
- Components can be independently developed, styled, and unit tested.
- Clear contract between visualization primitives and application state.

### Negative
- Monorepo tooling setup requires coordination via `pnpm-workspace.yaml`.
- Package publishing workflow needed if `@runvault/ui` is distributed on npm for third-party consumers.

## Alternatives Considered
- **Monolithic Next.js or Vite application:** Rejected because it makes embedding specific components into JupyterLab or other external web pages difficult without embedding full-page iframes.

## Open Implementation Details
- Data visualization engine selection for Phase 4: lightweight Canvas/WebGL (e.g., uPlot / Visx) to handle 100k+ step loss curves without DOM thrashing.
- Theme token integration (Tailwind vs CSS variables).
