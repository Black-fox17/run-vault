import { describe, expect, it } from "vitest";
import { THEME_TOKENS, UI_PACKAGE_VERSION } from "./index";

describe("@runvault/ui package foundation", () => {
  it("exports a valid semver package version", () => {
    expect(UI_PACKAGE_VERSION).toBe("0.1.0");
  });

  it("exports theme tokens with defined status colors", () => {
    expect(THEME_TOKENS.colors.status.running).toBe("#2563eb");
    expect(THEME_TOKENS.colors.status.completed).toBe("#16a34a");
    expect(THEME_TOKENS.colors.status.failed).toBe("#dc2626");
  });
});
