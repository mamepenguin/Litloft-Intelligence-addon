/**
 * SPEC-ADDON-169: what intelligence puts on the drive home through the
 * `drive-home-sections` slot.
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

const PICKUP_API = "/api/addons/intelligence/pickup";

const viewer = vi.hoisted(() => ({ nickname: null as string | null }));
vi.mock("@/components/ProfileProvider", () => ({
  useProfile: () => ({
    nickname: viewer.nickname,
    setNickname: vi.fn(),
    clearNickname: vi.fn(),
  }),
  ProfileProvider: ({ children }: { children: React.ReactNode }) => children,
}));

vi.mock("@/components/ClipboardProvider", () => ({
  useClipboard: () => ({
    clipboard: null,
    copy: vi.fn(),
    cut: vi.fn(),
    paste: vi.fn(),
    clear: vi.fn(),
    isCut: () => false,
  }),
}));

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

const VIEWERS: (string | null)[] = ["alice", null];

function requestUrl(input: unknown): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  return (input as Request).url;
}

const requested: string[] = [];

function videoFile(id: string) {
  return {
    id,
    filename: `${id}.mp4`,
    drive: "family",
    file_path: `${id}.mp4`,
    folder_path: "",
    mime_type: "video/mp4",
    size: 1,
    duration: 60,
    tags: [],
    created_at: "2026-11-01T00:00:00Z",
    modified_at: "2026-11-01T00:00:00Z",
  };
}

function respond(url: string): Response {
  if (url.includes(PICKUP_API)) {
    return new Response(JSON.stringify({ file_ids: ["a", "b"], total: 2 }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  if (url.includes("/api/files")) {
    return new Response(
      JSON.stringify(["a", "b"].map(videoFile)),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  }
  return new Response(JSON.stringify({}), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function intelligenceEntries(slot: string) {
  return (MANIFEST.slots?.[slot] ?? []).map((entry) => ({ ...entry, addonName: "intelligence" }));
}

async function renderDriveHomeSlot() {
  catalogue.slots = { "drive-home-sections": intelligenceEntries("drive-home-sections") };
  const view = render(<AddonSlot id="drive-home-sections" props={{ drive: "family" }} />);
  await act(() => new Promise((settle) => setTimeout(settle, 100)));
  return view;
}

beforeEach(async () => {
  requested.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: unknown) => {
      const url = requestUrl(input);
      requested.push(url);
      return respond(url);
    }),
  );
  await loadAddonSlotModule("intelligence");
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("SPEC-ADDON-169 the drive home has no Pickup row", () => {
  it("I1 I8: intelligence draws nothing in drive-home-sections, with or without a profile", async () => {
    for (const nickname of VIEWERS) {
      viewer.nickname = nickname;
      const { container, unmount } = await renderDriveHomeSlot();
      expect(container.innerHTML, `viewer ${nickname ?? "without a profile"}`).toBe("");
      unmount();
    }
  });

  it("I2: opening the drive home sends no request to the Pickup API", async () => {
    for (const nickname of VIEWERS) {
      viewer.nickname = nickname;
      const { unmount } = await renderDriveHomeSlot();
      unmount();
    }
    expect(requested.filter((url) => url.includes(PICKUP_API))).toEqual([]);
  });
});
