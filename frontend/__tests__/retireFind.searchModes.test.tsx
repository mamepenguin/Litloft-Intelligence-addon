/**
 * SPEC-ADDON-158 I2, the addon's side: what intelligence puts in the search
 * page's `…` menu through the `search-modes` slot.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, cleanup } from "@testing-library/react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import React from "react";

const ADDON_ROOT = [
  resolve(process.cwd(), "../addons/intelligence"),
  resolve(process.cwd(), "addons/intelligence"),
].find((candidate) => existsSync(resolve(candidate, "manifest.json")))!;

const MANIFEST = JSON.parse(
  readFileSync(resolve(ADDON_ROOT, "manifest.json"), "utf-8"),
) as { slots?: Record<string, { id: string; label: string; priority?: number }[]> };

vi.mock("@/addons/intelligence/api", async () => {
  const actual = await vi.importActual<typeof import("@/addons/intelligence/api")>(
    "@/addons/intelligence/api",
  );
  return {
    ...actual,
    getIntelligenceStatus: vi.fn(async () => ({
      status: "ok",
      features: { indexing: true, search: true, auto_tags: "false", summaries: "false", rag: true },
      llm: { provider: "ollama", model: "qwen", enabled: true, output_language: "auto" },
    })),
  };
});
vi.mock("@/lib/addons", async () => {
  const actual = await vi.importActual<typeof import("@/lib/addons")>("@/lib/addons");
  return {
    ...actual,
    getEnabledAddons: vi.fn(async () => ({
      intelligence: { label: "Intelligence", icon: "brain" },
    })),
  };
});

const catalogue = vi.hoisted(() => ({
  slots: {} as Record<string, unknown[]>,
}));
vi.mock("@/components/AddonSlotsProvider", () => ({
  useAddonSlots: () => ({
    addons: { intelligence: { label: "Intelligence", icon: "brain", scope: "both" } },
    catalogueDrive: "family",
    slots: catalogue.slots,
    loading: false,
    getSlotEntries: (id: string) => catalogue.slots[id] ?? [],
    hasSlot: (id: string) => (catalogue.slots[id] ?? []).length > 0,
  }),
  useAddonStatus: () => ({ features: {} }),
}));

import { AddonSlot } from "@/components/AddonSlot";
import { loadAddonSlotModule } from "@/lib/addons";

function intelligenceEntries(slot: string) {
  return (MANIFEST.slots?.[slot] ?? []).map((entry) => ({ ...entry, addonName: "intelligence" }));
}

beforeEach(async () => {
  await loadAddonSlotModule("intelligence");
});

afterEach(cleanup);

const QUERIES = ["SF 映画", "type:video 映画"];
const FILTERS = ["all", "passages"];

describe("SPEC-ADDON-158 I2 intelligence puts no Find row in the search page's … menu", () => {
  it("draws nothing for search-modes, from the manifest or from a catalogue that still names find-mode", async () => {
    const catalogues = [
      intelligenceEntries("search-modes"),
      [{ id: "find-mode", label: "Find", priority: 15, addonName: "intelligence" }],
    ];

    for (const entries of catalogues) {
      catalogue.slots = { "search-modes": entries };
      for (const query of QUERIES) {
        for (const filter of FILTERS) {
          const { container, unmount } = render(
            <div role="menu">
              <AddonSlot
                id="search-modes"
                layout="stack"
                props={{
                  query,
                  drive: "family",
                  filter,
                  closeMenu: () => {},
                  onSelect: () => {},
                }}
              />
            </div>,
          );
          // The Find row appeared only after its status probe answered.
          await act(() => new Promise((settle) => setTimeout(settle, 50)));
          expect(container.textContent).toBe("");
          expect(container.querySelectorAll("button, a, [role^='menuitem']")).toHaveLength(0);
          unmount();
        }
      }
    }
  });
});
