import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, it, expect } from "vitest";

interface SectionEntry {
  id: string;
  label: string;
  priority: number;
  i18n_key?: string;
}

interface Manifest {
  slots: Record<string, SectionEntry[]>;
}

const manifest = JSON.parse(
  readFileSync(
    resolve(process.cwd(), "..", "addons", "intelligence", "manifest.json"),
    "utf8",
  ),
) as Manifest;

const sections = manifest.slots["file-detail-sections"] ?? [];
const panels = manifest.slots["player-side"] ?? [];

describe("SPEC-ADDON-089 I5 — the intelligence manifest", () => {
  it("has exactly one scenes entry, in player-side, labelled Scenes at priority 20", () => {
    const inPlayer = panels.filter((entry) => entry.id === "scenes");

    expect(inPlayer).toHaveLength(1);
    expect(inPlayer[0]).toMatchObject({ label: "Scenes", priority: 20 });
    expect(typeof inPlayer[0].i18n_key).toBe("string");
    expect(sections.map((entry) => entry.id)).not.toContain("scenes");
  });

  it("has no visual-index or clip-frames entry left", () => {
    const ids = Object.values(manifest.slots).flatMap((entries) =>
      entries.map((entry) => entry.id),
    );
    expect(ids).not.toContain("visual-index");
    expect(ids).not.toContain("clip-frames");
  });
});
