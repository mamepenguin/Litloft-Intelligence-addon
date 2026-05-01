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
 *    (``semanticSearch`` + ``getSearchStatus`` calls).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, cleanup } from "@testing-library/react";
import React from "react";

vi.mock("./api", async () => {
  const actual = await vi.importActual<typeof import("./api")>("./api");
  return {
    ...actual,
    getSearchStatus: vi.fn(),
    semanticSearch: vi.fn(),
  };
});

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
import { getSearchStatus, semanticSearch } from "./api";

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
  vi.mocked(getSearchStatus).mockResolvedValue({ available: true } as any);
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

  it("renders the page layout when context is 'page' — section heading + grid", async () => {
    render(
      <SemanticSearchSlot
        query="space"
        drive="family"
        filter="all"
        onSelect={() => {}}
        context="page"
      />,
    );

    // Page layout exposes a level-2 heading.
    const heading = await screen.findByRole("heading", { level: 2 });
    expect(heading).toBeInTheDocument();

    // Result is still rendered.
    await waitFor(() => {
      expect(screen.queryByText("sample-movie.mp4")).toBeInTheDocument();
    });

    // Page layout uses a list role for the grid container.
    expect(screen.queryByRole("list")).toBeInTheDocument();
  });

  it("page layout card links to the file detail page", async () => {
    // Phase 2 unified the semantic page card with the core FileCard,
    // which uses Next ``<Link>`` to navigate. The slot-level onSelect
    // is now reserved for sub-targets inside the card (timestamp pills
    // in MatchOverlay, the Ask CTA in the section header).
    render(
      <SemanticSearchSlot
        query="space"
        drive="family"
        filter="all"
        onSelect={() => {}}
        context="page"
      />,
    );

    const titleNode = await screen.findByText("sample-movie.mp4");
    const link = titleNode.closest("a");
    expect(link).not.toBeNull();
    expect(link!.getAttribute("href")).toBe("/files/f-1");
  });

  it("renders nothing when intelligence search is unavailable", async () => {
    vi.mocked(getSearchStatus).mockResolvedValue({ available: false } as any);

    const { container } = render(
      <SemanticSearchSlot
        query="space"
        drive="family"
        filter="all"
        onSelect={() => {}}
        context="page"
      />,
    );

    await waitFor(() => {
      expect(getSearchStatus).toHaveBeenCalled();
    });
    expect(container.textContent).toBe("");
  });
});
