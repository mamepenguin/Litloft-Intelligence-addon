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
import { render, screen, waitFor, cleanup, fireEvent } from "@testing-library/react";
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

  /**
   * The row used to be one <button> wrapping the per-timestamp <button>s.
   * That is invalid HTML; React reported it on every render, and the noise
   * was filed as an act() warning and left alone for a release.
   *
   * Two assertions, because either alone can be talked out of: the DOM one
   * survives React changing its wording, and the console one catches the
   * other nestings (<a> in <a>, <button> in <a>) the selector below would
   * have to be extended for.
   */
  describe("the result row's click targets", () => {
    const renderRow = async () => {
      const nestingErrors: string[] = [];
      const spy = vi
        .spyOn(console, "error")
        .mockImplementation((...args: unknown[]) => {
          const text = args.map(String).join(" ");
          if (/descendant of/.test(text)) nestingErrors.push(text);
        });
      const { container } = render(
        <SemanticSearchSlot
          query="space"
          drive="family"
          filter="all"
          onSelect={onSelect}
        />,
      );
      await waitFor(() => {
        expect(screen.queryByText("sample-movie.mp4")).toBeInTheDocument();
      });
      spy.mockRestore();
      return { container, nestingErrors };
    };

    let onSelect: ReturnType<typeof vi.fn>;

    beforeEach(() => {
      onSelect = vi.fn();
    });

    it("does not nest one interactive element inside another", async () => {
      const { container, nestingErrors } = await renderRow();

      const nested = container.querySelectorAll(
        "button button, button a, a button, a a",
      );
      expect([...nested].map((el) => el.outerHTML)).toEqual([]);
      expect(nestingErrors).toEqual([]);
    });

    it("opens the file from the row and the moment from a timestamp", async () => {
      await renderRow();

      fireEvent.click(screen.getByRole("button", { name: "0:12" }));
      expect(onSelect).toHaveBeenCalledWith("/files/f-1?t=12");

      onSelect.mockClear();
      // The row's own target carries the filename as its accessible name;
      // it is the whole row minus the timestamps, so it has no text of
      // its own to be named by.
      fireEvent.click(screen.getByRole("button", { name: "sample-movie.mp4" }));
      expect(onSelect).toHaveBeenCalledWith("/files/f-1");
    });

    it("leaves no dead strip where the timestamps are", async () => {
      await renderRow();

      // jsdom has no layout, so this is the contract rather than a hit
      // test: the timestamp wrapper is a full-width block raised above
      // the row's stretched action, and its gaps and trailing space are
      // only clickable because it is transparent to the pointer and the
      // buttons inside take events back.
      const stamp = screen.getByRole("button", { name: "0:12" });
      const wrapper = stamp.parentElement!;
      expect(wrapper.className).toContain("pointer-events-none");
      expect(stamp.className).toContain("pointer-events-auto");
    });
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
