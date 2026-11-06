/**
 * SPEC-ADDON-064 entry points: the manifest registers 「ここに質問」 in the
 * folder ``…`` menu and on the collection page, and the slot registry exports
 * both components.
 */

import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

const MANIFEST = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "manifest.json"), "utf-8"),
) as { slots?: Record<string, { id: string; priority?: number }[]> };

describe("SPEC-ADDON-064 manifest and slot registration", () => {
  it("registers folder-ask as a folder bulk action before the AI rows", async () => {
    const entries = MANIFEST.slots?.["folder-bulk-actions-menu"] ?? [];
    const entry = entries.find((row) => row.id === "folder-ask");

    expect(entry).toBeDefined();
    expect(entry?.priority).toBe(5);
    expect(entry!.priority!).toBeLessThan(
      entries.find((row) => row.id === "folder-ai-actions")!.priority!,
    );
  });

  it("registers collection-ask in collection-actions", () => {
    const entries = MANIFEST.slots?.["collection-actions"] ?? [];
    expect(entries.some((row) => row.id === "collection-ask")).toBe(true);
  });

  it("exports both slot components", async () => {
    const { slotComponents } = await import("@/addons/intelligence/slots");

    expect(slotComponents["folder-ask"]).toBeDefined();
    expect(slotComponents["collection-ask"]).toBeDefined();
  });
});

