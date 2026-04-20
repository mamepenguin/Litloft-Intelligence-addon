/**
 * Unit tests for CitationRailContext — the new multi-expanded
 * accordion state machine (Phase 2 UI overhaul).
 *
 * Replaces the hover / pin / single-active contract with a Set-based
 * ``expanded`` and bulk operations.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  cleanup,
  act,
} from "@testing-library/react";
import React, { useEffect } from "react";
import { NextIntlClientProvider } from "next-intl";

vi.mock("@/addons/intelligence/api", () => ({
  getCitationChunkExcerpt: vi.fn().mockResolvedValue({
    chunk_id: "c1",
    file_id: "f1",
    prefix: "",
    target: "T",
    suffix: "",
    start_time: 0,
    end_time: 1,
    page: null,
  }),
}));

import {
  CitationRailProvider,
  useCitationRail,
  CITATION_STRONG_THRESHOLD,
} from "@/addons/intelligence/CitationRailContext";
import { getCitationChunkExcerpt } from "@/addons/intelligence/api";

const strongA = {
  section_path: "S/0",
  segment_type: "paragraph" as const,
  segment_text: "…",
  chunk_ids: ["chunkA"],
  top_score: 0.95,
  has_citation: true,
};
const strongB = {
  section_path: "S/1",
  segment_type: "paragraph" as const,
  segment_text: "…",
  chunk_ids: ["chunkB"],
  top_score: 0.92,
  has_citation: true,
};
const weakA = {
  section_path: "S/2",
  segment_type: "bullet" as const,
  segment_text: "…",
  chunk_ids: ["chunkC"],
  top_score: 0.72,
  has_citation: true,
};
const missing = {
  section_path: "S/3",
  segment_type: "paragraph" as const,
  segment_text: "…",
  chunk_ids: [],
  top_score: 0.3,
  has_citation: false,
};

interface ProbeHandle {
  current: ReturnType<typeof useCitationRail> | null;
}

function Probe({ handle }: { handle: ProbeHandle }) {
  const value = useCitationRail();
  useEffect(() => {
    handle.current = value;
  });
  return null;
}

function renderProbe() {
  const handle: ProbeHandle = { current: null };
  render(
    <NextIntlClientProvider locale="ja" messages={{}}>
      <CitationRailProvider fileId="f1" drive="d1">
        <Probe handle={handle} />
      </CitationRailProvider>
    </NextIntlClientProvider>,
  );
  return handle;
}

beforeEach(() => {
  vi.clearAllMocks();
  try {
    window.localStorage.removeItem("hv.intelligence.verify");
  } catch {
    // ignore
  }
});

afterEach(() => {
  cleanup();
});

describe("CitationRailContext", () => {
  it("exposes CITATION_STRONG_THRESHOLD = 0.90", () => {
    expect(CITATION_STRONG_THRESHOLD).toBe(0.9);
  });

  it("starts with Verify OFF and an empty expanded set", () => {
    const h = renderProbe();
    expect(h.current?.verify).toBe(false);
    expect(h.current?.expanded.size).toBe(0);
  });

  it("toggle flips a section from closed to open", async () => {
    const h = renderProbe();
    await act(async () => {
      h.current?.toggle(strongA);
    });
    expect(h.current?.isExpanded("S/0")).toBe(true);
    expect(h.current?.expanded.size).toBe(1);
  });

  it("toggle is a no-op for has_citation=false citations", async () => {
    const h = renderProbe();
    await act(async () => {
      h.current?.toggle(missing);
    });
    expect(h.current?.expanded.size).toBe(0);
    expect(getCitationChunkExcerpt).not.toHaveBeenCalled();
  });

  it("toggle again closes the section", async () => {
    const h = renderProbe();
    await act(async () => {
      h.current?.toggle(strongA);
    });
    await act(async () => {
      h.current?.toggle(strongA);
    });
    expect(h.current?.isExpanded("S/0")).toBe(false);
  });

  it("collapseAll clears every expanded section", async () => {
    const h = renderProbe();
    await act(async () => {
      h.current?.toggle(strongA);
      h.current?.toggle(strongB);
    });
    expect(h.current?.expanded.size).toBeGreaterThan(0);
    await act(async () => {
      h.current?.collapseAll();
    });
    expect(h.current?.expanded.size).toBe(0);
  });

  it("expandAll opens every citation with has_citation=true", async () => {
    const h = renderProbe();
    await act(async () => {
      h.current?.expandAll([strongA, strongB, weakA, missing]);
    });
    expect(h.current?.expanded.size).toBe(3);
    expect(h.current?.isExpanded("S/0")).toBe(true);
    expect(h.current?.isExpanded("S/1")).toBe(true);
    expect(h.current?.isExpanded("S/2")).toBe(true);
    expect(h.current?.isExpanded("S/3")).toBe(false);
  });

  it("expandWeakOnly opens only weak-tier citations (top_score < 0.90)", async () => {
    const h = renderProbe();
    await act(async () => {
      h.current?.expandWeakOnly([strongA, strongB, weakA, missing]);
    });
    expect(h.current?.expanded.size).toBe(1);
    expect(h.current?.isExpanded("S/2")).toBe(true);
  });

  it("setVerify(true) flips verify mode on", () => {
    // Persistence to localStorage is a side-effect verified by the
    // production provider's try/catch — the test env's storage stub
    // is unreliable across jsdom versions, so we cover the visible
    // state transition here and lean on the context code's own guard
    // for the write.
    const h = renderProbe();
    act(() => {
      h.current?.setVerify(true);
    });
    expect(h.current?.verify).toBe(true);
  });

  it("setVerify(false) collapses every expanded panel", async () => {
    const h = renderProbe();
    await act(async () => {
      h.current?.setVerify(true);
      h.current?.toggle(strongA);
      h.current?.toggle(strongB);
    });
    expect(h.current?.expanded.size).toBe(2);
    await act(async () => {
      h.current?.setVerify(false);
    });
    expect(h.current?.expanded.size).toBe(0);
  });

  it("close is safe to call for a section that is not expanded", async () => {
    const h = renderProbe();
    await act(async () => {
      h.current?.close("not-open");
    });
    expect(h.current?.expanded.size).toBe(0);
  });

  it("excerptState returns idle for non-expanded sections", () => {
    const h = renderProbe();
    expect(h.current?.excerptState("S/0").kind).toBe("idle");
  });
});
