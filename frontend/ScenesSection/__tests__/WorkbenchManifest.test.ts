import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, it, expect } from "vitest";

interface SlotEntry {
  id: string;
  label: string;
  priority: number;
  i18n_key?: string;
}

interface Manifest {
  slots: Record<string, SlotEntry[]>;
}

const manifest = JSON.parse(
  readFileSync(
    resolve(process.cwd(), "..", "addons", "intelligence", "manifest.json"),
    "utf8",
  ),
) as Manifest;

const ids = (slot: string) => (manifest.slots[slot] ?? []).map((entry) => entry.id);

describe("SPEC-ADDON-099 the intelligence manifest's player-side panels", () => {
  it("lists transcript, scenes and detailed-summary in player-side at 10, 20 and 30, each with an i18n_key", () => {
    const panels = manifest.slots["player-side"] ?? [];
    const pick = (id: string) => panels.filter((entry) => entry.id === id);

    expect(pick("transcript")).toHaveLength(1);
    expect(pick("scenes")).toHaveLength(1);
    expect(pick("detailed-summary")).toHaveLength(1);
    expect(pick("transcript")[0].priority).toBe(10);
    expect(pick("scenes")[0].priority).toBe(20);
    expect(pick("detailed-summary")[0].priority).toBe(30);
    for (const id of ["scenes", "detailed-summary"]) {
      expect({ id, key: typeof pick(id)[0].i18n_key }).toEqual({ id, key: "string" });
    }
  });

  it("moves scenes out of file-detail-sections and keeps detailed-summary there for files without a player", () => {
    expect(ids("file-detail-sections")).not.toContain("scenes");
    expect(ids("file-detail-sections")).toContain("detailed-summary");
  });
});
