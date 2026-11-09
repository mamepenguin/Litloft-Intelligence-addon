import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, cleanup, fireEvent } from "@testing-library/react";
import React from "react";

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return { ...actual, getIntelligenceStatus: vi.fn() };
});
vi.mock("@/lib/addons", () => ({ getEnabledAddons: vi.fn() }));

import FindModeSlot from "./FindModeSlot";
import { getIntelligenceStatus } from "./api";
import { getEnabledAddons } from "@/lib/addons";

const enabledStatus = {
  status: "ok",
  features: { indexing: true, search: true, auto_tags: "false", summaries: "false", rag: true },
  llm: { provider: "ollama", model: "qwen", enabled: true, output_language: "auto" },
};

const PAGE_CTA = "Open Find page";

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getEnabledAddons).mockResolvedValue({
    intelligence: { label: "Intelligence", icon: "brain" },
  } as never);
});

afterEach(() => {
  cleanup();
});

describe("SPEC-ADDON-086 the Find entry is a row of the … menu", () => {
  it("is labelled with the page CTA and closes the menu before navigating with the trimmed query", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(enabledStatus as never);
    const log: string[] = [];

    render(
      <FindModeSlot
        query="  先週観た映画 "
        drive="家族"
        filter="all"
        closeMenu={() => {
          log.push("closeMenu");
        }}
        onSelect={(href: string) => {
          log.push(`onSelect ${href}`);
        }}
      />,
    );

    const label = await screen.findByText(PAGE_CTA);
    const row = label.closest("button, [role^='menuitem']") as HTMLElement;
    expect(row).not.toBeNull();
    fireEvent.click(row);

    expect(log).toEqual([
      "closeMenu",
      `onSelect /drive/${encodeURIComponent("家族")}/addons/intelligence/find?q=${encodeURIComponent("先週観た映画")}`,
    ]);
  });

  it("leaves no entry and nothing pending when the probe is rejected", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", onUnhandled);
    try {
      vi.mocked(getIntelligenceStatus).mockRejectedValue(new Error("network down"));

      const { container } = render(
        <FindModeSlot
          query="SF 映画"
          drive="family"
          filter="all"
          closeMenu={() => {}}
          onSelect={() => {}}
        />,
      );

      await waitFor(() => expect(getIntelligenceStatus).toHaveBeenCalled());
      await new Promise((resolve) => setTimeout(resolve, 50));

      expect(unhandled).toEqual([]);
      expect(container.textContent).toBe("");
      expect(screen.queryByText(PAGE_CTA)).toBeNull();
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });
});
