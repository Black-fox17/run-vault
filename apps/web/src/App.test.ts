import { UI_PACKAGE_VERSION } from "@runvault/ui";
import { describe, expect, it } from "vitest";

describe("Web Dashboard application foundation", () => {
  it("imports successfully from @runvault/ui workspace package", () => {
    expect(UI_PACKAGE_VERSION).toBe("0.1.0");
  });
});
