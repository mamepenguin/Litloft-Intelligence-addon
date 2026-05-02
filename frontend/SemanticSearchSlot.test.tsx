/**
 * Tests for SemanticSearchSlot — the search-modes slot that renders
 * intelligence-backed semantic results inside the search popup
 * (compact list) or the dedicated /drive/<name>/search page (full
 * grid).
 *
 * Spec: ``2026-05-01-search-ui-rich-redesign.md`` §intelligence
 * アドオン側の変更（後方互換維持）.
 *
 * Contract:
 *  - When ``context`` is undefined or "popup", the existing compact
 *    list layout MUST render unchanged (backward compat).
 *  - When ``context === "page"``, a section heading + grid layout
 *    suited to a full-page context renders instead.
 *  - In both layouts, the data-fetching pipeline is identical
 *    (``semanticSearch`` + the core's per-drive addon-registry probe).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, cleanup } from "@testing-library/react";
import React from "react";

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    semanticSearch: vi.fn(),
  };
});

vi.mock("@/lib/addons", () => ({
  getEnabledAddons: vi.fn(),
}));

// FileCard pulls clipboard context — the page layout uses the core
// FileCard for unified styling with filename-match results.
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

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: {
    children: React.ReactNode;
    href: string;
    [key: string]: unknown;
  }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

import SemanticSearchSlot from "./SemanticSearchSlot";
import { semanticSearch } from "./api";
import { getEnabledAddons } from "@/lib/addons";

const sampleResult = {
  file_id: "f-1",
  filename: "sample-movie.mp4",
  match_types: ["transcript"],
  segments: [
    {
      time_range: [12, 30] as [number, number],
      matches: [{ score: 0.9 }],
    },
  ],
} as any;

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getEnabledAddons).mockResolvedValue({
    intelligence: { label: "Intelligence", icon: "brain" },
  } as any);
  vi.mocked(semanticSearch).mockResolvedValue({
    results: [sampleResult],
  } as any);
});

afterEach(() => {
  cleanup();
});

describe("SemanticSearchSlot", () => {
  it("renders the popup (compact) layout by default — no section heading", async () => {
    render(
      <SemanticSearchSlot
        query="space"
        drive="family"
        filter="all"
        onSelect={() => {}}
      />,
    );

    // Wait for the debounced fetch to settle and a result to appear.
    await waitFor(() => {
      expect(screen.queryByText("sample-movie.mp4")).toBeInTheDocument();
    });

    // Popup layout has no level-2 section heading.
    expect(screen.queryByRole("heading", { level: 2 })).toBeNull();
    // Popup layout has no list role grouping.
    expect(screen.queryByRole("list")).toBeNull();
  });

  it("renders the popup layout when context is explicitly 'popup'", async () => {
    render(
      <SemanticSearchSlot
        query="space"
        drive="family"
        filter="all"
        onSelect={() => {}}
        context="popup"
      />,
    );

    await waitFor(() => {
      expect(screen.queryByText("sample-movie.mp4")).toBeInTheDocument();
    });

    expect(screen.queryByRole("heading", { level: 2 })).toBeNull();
  });

  it("renders nothing in page context — Phase 3 merges semantic into the host's unified list", async () => {
    // Phase 3 (`2026-05-02-search-results-unification-phase3.md`)
    // retired the slot's PageLayout. The host's `useFolderFiles`
    // merges filename-match and semantic hits into a single
    // `FolderContent` list, so the slot only contributes header
    // chips (the Find handoff lives in `FindModeSlot`). The popup
    // layout is unchanged. The slot also skips its availability
    // probe / fetch in page context to avoid duplicating the host's
    // request.
    const { container } = render(
      <SemanticSearchSlot
        query="space"
        drive="family"
        filter="all"
        onSelect={() => {}}
        context="page"
      />,
    );

    expect(container.textContent).toBe("");
    expect(getEnabledAddons).not.toHaveBeenCalled();
    expect(semanticSearch).not.toHaveBeenCalled();
  });

  it("renders nothing when intelligence is disabled for the drive in popup context", async () => {
    vi.mocked(getEnabledAddons).mockResolvedValue({} as any);

    const { container } = render(
      <SemanticSearchSlot
        query="space"
        drive="family"
        filter="all"
        onSelect={() => {}}
        context="popup"
      />,
    );

    await waitFor(() => {
      expect(getEnabledAddons).toHaveBeenCalled();
    });
    expect(container.textContent).toBe("");
  });
});
