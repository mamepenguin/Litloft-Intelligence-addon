/**
 * SPEC-ADDON-042: the addon registers its section in `search-results-top`.
 */

import { describe, it, expect } from "vitest";
import { readFileSync, realpathSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { slotComponents } from "./slots";

const here = dirname(realpathSync(fileURLToPath(import.meta.url)));
const manifest = JSON.parse(readFileSync(resolve(here, "../manifest.json"), "utf8")) as {
  slots: Record<string, Array<{ id: string; label: string; priority: number }>>;
};

describe("scene candidates slot registration (SPEC-ADDON-042)", () => {
  it("declares scene-segments in search-results-top with label and priority 10", () => {
    expect(manifest.slots["search-results-top"]).toEqual([
      expect.objectContaining({
        id: "scene-segments",
        label: "Scene candidates",
        priority: 10,
      }),
    ]);
  });

  it("maps the scene-segments id to a component", () => {
    expect(slotComponents["scene-segments"]).toBeDefined();
  });
});
