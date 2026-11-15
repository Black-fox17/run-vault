/**
 * @runvault/ui - Reusable component library for RunVault dashboards and notebook extensions.
 */

export const UI_PACKAGE_VERSION = "0.1.0" as const;

export type RunStatus =
  | "initializing"
  | "running"
  | "completed"
  | "failed"
  | "aborted";

export interface StatusBadgeProps {
  status: RunStatus;
  label?: string;
}

export interface MetricPoint {
  readonly step: number;
  readonly value: number;
  readonly timestamp: number;
}

export const THEME_TOKENS = {
  colors: {
    status: {
      initializing: "#64748b",
      running: "#2563eb",
      completed: "#16a34a",
      failed: "#dc2626",
      aborted: "#d97706",
    },
  },
} as const;
