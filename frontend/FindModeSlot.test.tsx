/**
 * Tests for FindModeSlot — the search-modes row that hands off to the Find
 * page.
 *
 * Contract:
 *  - Emits a `…` menu row labelled with `find.pageCta` that closes the menu
 *    and then navigates to
 *    ``/drive/<drive>/addons/intelligence/find?q=<query>``.
 *  - Renders only when ``intelligence.features.rag === true && llm.enabled``
 *    (same gate as the Ask page — Find depends on Stage A + C LLM calls).
 *  - Hidden when query is empty (no point linking with no seed).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, cleanup, fireEvent } from "@testing-library/react";
import React from "react";

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    getIntelligenceStatus: vi.fn(),
  };
});

vi.mock("@/lib/addons", () => ({
  getEnabledAddons: vi.fn(),
}));

import FindModeSlot from "./FindModeSlot";
import { getIntelligenceStatus } from "./api";
import { getEnabledAddons } from "@/lib/addons";

const enabledStatus = {
  status: "ok",
  features: {
    indexing: true,
    search: true,
    auto_tags: "false",
    summaries: "false",
    rag: true,
  },
  llm: {
    provider: "ollama",
    model: "qwen",
    enabled: true,
    output_language: "auto",
  },
};

const disabledStatus = {
  ...enabledStatus,
  features: { ...enabledStatus.features, rag: false },
};

const PAGE_CTA = "Open Find page";

beforeEach(() => {
  vi.clearAllMocks();
  // Default: intelligence enabled for the drive. Individual tests
  // override this to verify the fallback behaviour.
  vi.mocked(getEnabledAddons).mockResolvedValue({
    intelligence: { label: "Intelligence", icon: "brain" },
  } as any);
});

afterEach(() => {
  cleanup();
});

describe("FindModeSlot", () => {
  it("draws a menu row labelled with the page CTA", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(enabledStatus as any);

    render(
      <FindModeSlot
        query="SF 映画"
        drive="family"
        filter="all"
        closeMenu={() => {}}
        onSelect={() => {}}
      />,
    );

    const row = await screen.findByRole("menuitem", { name: PAGE_CTA });
    expect(row).toBeInTheDocument();
    expect(screen.queryByRole("heading")).toBeNull();
  });

  it("closes the menu, then navigates to the Find page with the trimmed query", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(enabledStatus as any);
    const log: string[] = [];

    render(
      <FindModeSlot
        query="  先週観た映画 "
        drive="家族"
        filter="all"
        closeMenu={() => log.push("closeMenu")}
        onSelect={(href) => log.push(`onSelect ${href}`)}
      />,
    );

    fireEvent.click(await screen.findByRole("menuitem", { name: PAGE_CTA }));

    expect(log).toEqual([
      "closeMenu",
      `onSelect /drive/${encodeURIComponent("家族")}/addons/intelligence/find?q=${encodeURIComponent("先週観た映画")}`,
    ]);
  });

  it("renders nothing when intelligence.features.rag is false", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(disabledStatus as any);

    const { container } = render(
      <FindModeSlot
        query="anything"
        drive="family"
        filter="all"
        closeMenu={() => {}}
        onSelect={() => {}}
      />,
    );

    await waitFor(() => {
      expect(getIntelligenceStatus).toHaveBeenCalled();
    });
    expect(container.textContent).toBe("");
    expect(screen.queryByRole("menuitem", { name: PAGE_CTA })).toBeNull();
  });

  it("renders nothing when llm is disabled (Find needs Stage A + C LLM)", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue({
      ...enabledStatus,
      llm: { ...enabledStatus.llm, enabled: false },
    } as any);

    const { container } = render(
      <FindModeSlot
        query="x"
        drive="family"
        filter="all"
        closeMenu={() => {}}
        onSelect={() => {}}
      />,
    );

    await waitFor(() => expect(getIntelligenceStatus).toHaveBeenCalled());
    expect(container.textContent).toBe("");
  });

  it("renders nothing when the query is empty / whitespace", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(enabledStatus as any);

    const { container } = render(
      <FindModeSlot
        query="   "
        drive="family"
        filter="all"
        closeMenu={() => {}}
        onSelect={() => {}}
      />,
    );

    await waitFor(() => {
      expect(container.textContent).toBe("");
    });
  });

  it("leaves no row and nothing pending when the probe is rejected", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", onUnhandled);
    try {
      vi.mocked(getIntelligenceStatus).mockRejectedValue(new Error("network down"));

      const { container } = render(
        <FindModeSlot
          query="x"
          drive="family"
          filter="all"
          closeMenu={() => {}}
          onSelect={() => {}}
        />,
      );

      await waitFor(() => expect(getIntelligenceStatus).toHaveBeenCalled());
      await new Promise((resolve) => setTimeout(resolve, 25));

      expect(unhandled).toEqual([]);
      expect(container.textContent).toBe("");
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  it("falls back to the core addon registry when /status is unreachable", async () => {
    // Non-admin viewers cannot reach the addon's /status (it is admin-
    // gated for queue counters), so getIntelligenceStatus returns null.
    // The slot must still render based on the core's per-drive addon
    // registry — otherwise Find would be hidden from the very viewers
    // it is meant to serve.
    vi.mocked(getIntelligenceStatus).mockResolvedValue(null);
    vi.mocked(getEnabledAddons).mockResolvedValue({
      intelligence: { label: "Intelligence", icon: "brain" },
    } as any);

    render(
      <FindModeSlot
        query="SF 映画"
        drive="family"
        filter="all"
        closeMenu={() => {}}
        onSelect={() => {}}
      />,
    );

    expect(await screen.findByRole("menuitem", { name: PAGE_CTA })).toBeInTheDocument();
  });

  it("stays hidden when /status is null and intelligence is not enabled for the drive", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(null);
    vi.mocked(getEnabledAddons).mockResolvedValue({} as any);

    const { container } = render(
      <FindModeSlot
        query="SF 映画"
        drive="family"
        filter="all"
        closeMenu={() => {}}
        onSelect={() => {}}
      />,
    );

    await waitFor(() => {
      expect(getEnabledAddons).toHaveBeenCalled();
    });
    expect(container.textContent).toBe("");
  });
});
