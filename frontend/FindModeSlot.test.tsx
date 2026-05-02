/**
 * Tests for FindModeSlot — the entry point in the search-modes slot
 * that hands off to the dedicated Find page.
 *
 * Spec: ``2026-04-30-intelligence-find-mode.md`` §3.1 (UI / モード切替).
 *
 * Contract:
 *  - Mirrors the ``SemanticSearchSlot`` Ask handoff: emits a button that
 *    links to ``/drive/<drive>/addons/intelligence/find?q=<query>``.
 *  - Renders only when ``intelligence.features.rag === true && llm.enabled``
 *    (same gate as the Ask page — Find depends on Stage A + C LLM calls).
 *  - Hidden when query is empty (no point linking with no seed).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, cleanup } from "@testing-library/react";
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
  it("renders a Find handoff button when rag is enabled and query is non-empty", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(enabledStatus as any);

    render(
      <FindModeSlot
        query="SF 映画"
        drive="family"
        filter="all"
        onSelect={() => {}}
      />,
    );

    const btn = await screen.findByRole("button", { name: /find/i });
    expect(btn).toBeInTheDocument();
  });

  it("the Find button calls onSelect with /drive/<drive>/addons/intelligence/find?q=<query>", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(enabledStatus as any);
    const onSelect = vi.fn();

    render(
      <FindModeSlot
        query="先週観た映画"
        drive="家族"
        filter="all"
        onSelect={onSelect}
      />,
    );

    const btn = await screen.findByRole("button", { name: /find/i });
    btn.click();

    expect(onSelect).toHaveBeenCalledTimes(1);
    const href = onSelect.mock.calls[0][0] as string;
    expect(href).toContain("/drive/");
    expect(href).toContain("/addons/intelligence/find");
    expect(href).toContain(`q=${encodeURIComponent("先週観た映画")}`);
    // Drive segment is encoded as well.
    expect(href).toContain(encodeURIComponent("家族"));
  });

  it("renders nothing when intelligence.features.rag is false", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(disabledStatus as any);

    const { container } = render(
      <FindModeSlot
        query="anything"
        drive="family"
        filter="all"
        onSelect={() => {}}
      />,
    );

    // Wait for the status fetch to settle. The slot must remain empty.
    await waitFor(() => {
      expect(getIntelligenceStatus).toHaveBeenCalled();
    });
    expect(container.textContent).toBe("");
    expect(screen.queryByRole("button", { name: /find/i })).toBeNull();
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
        onSelect={() => {}}
      />,
    );

    // No button should ever appear — there's no seed query to hand off.
    await waitFor(() => {
      expect(container.textContent).toBe("");
    });
  });

  it("renders the popup layout (compact button) when context is undefined", async () => {
    // Backwards-compat default: undefined context === popup.
    vi.mocked(getIntelligenceStatus).mockResolvedValue(enabledStatus as any);

    render(
      <FindModeSlot
        query="SF 映画"
        drive="family"
        filter="all"
        onSelect={() => {}}
      />,
    );

    // Popup layout renders a single button labelled "Find: <query>" and
    // does NOT render a section heading.
    await screen.findByRole("button", { name: /find/i });
    expect(screen.queryByRole("heading")).toBeNull();
  });

  it("renders the popup layout when context is explicitly 'popup'", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(enabledStatus as any);

    render(
      <FindModeSlot
        query="SF 映画"
        drive="family"
        filter="all"
        onSelect={() => {}}
        context="popup"
      />,
    );

    await screen.findByRole("button", { name: /find/i });
    expect(screen.queryByRole("heading")).toBeNull();
  });

  it("renders the page layout (compact chip CTA) when context is 'page'", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(enabledStatus as any);

    render(
      <FindModeSlot
        query="SF 映画"
        drive="family"
        filter="all"
        onSelect={() => {}}
        context="page"
      />,
    );

    // Page layout is a right-aligned chip — Find is a handoff to
    // a different page, so it must not consume a heading slot on
    // the search results page.
    const cta = await screen.findByRole("button", { name: /find/i });
    expect(cta).toBeInTheDocument();
    expect(screen.queryByRole("heading")).toBeNull();
  });

  it("page layout CTA still calls onSelect with the find URL", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(enabledStatus as any);
    const onSelect = vi.fn();

    render(
      <FindModeSlot
        query="先週観た映画"
        drive="家族"
        filter="all"
        onSelect={onSelect}
        context="page"
      />,
    );

    const cta = await screen.findByRole("button", { name: /find/i });
    cta.click();

    expect(onSelect).toHaveBeenCalledTimes(1);
    const href = onSelect.mock.calls[0][0] as string;
    expect(href).toContain("/addons/intelligence/find");
    expect(href).toContain(`q=${encodeURIComponent("先週観た映画")}`);
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
        onSelect={() => {}}
      />,
    );

    const btn = await screen.findByRole("button", { name: /find/i });
    expect(btn).toBeInTheDocument();
  });

  it("stays hidden when /status is null and intelligence is not enabled for the drive", async () => {
    vi.mocked(getIntelligenceStatus).mockResolvedValue(null);
    vi.mocked(getEnabledAddons).mockResolvedValue({} as any);

    const { container } = render(
      <FindModeSlot
        query="SF 映画"
        drive="family"
        filter="all"
        onSelect={() => {}}
      />,
    );

    await waitFor(() => {
      expect(getEnabledAddons).toHaveBeenCalled();
    });
    expect(container.textContent).toBe("");
  });
});
