import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, it, expect } from "vitest";

interface SectionEntry {
  id: string;
  label: string;
  priority: number;
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

const sections = manifest.slots["file-detail-sections"];

describe("SPEC-ADDON-089 I5 — the intelligence manifest", () => {
  it("has exactly one scenes entry, labelled Scenes at priority 20", () => {
    const scenes = sections.filter((entry) => entry.id === "scenes");

    expect(scenes).toHaveLength(1);
    expect(scenes[0]).toMatchObject({ label: "Scenes", priority: 20 });
  });

  it("has no visual-index or clip-frames entry left", () => {
    expect(sections.map((entry) => entry.id)).not.toContain("visual-index");
    expect(sections.map((entry) => entry.id)).not.toContain("clip-frames");
  });
});
