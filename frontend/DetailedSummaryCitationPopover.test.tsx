/**
 * Tests for the detailed-summary citation marker (dot) and the
 * in-flow accordion panel.
 *
 * New interaction contract (Phase 2-4 UI overhaul):
 *   - The marker is a 14px SVG dot (solid teal for strong, dashed/half
 *     amber for weak). No hover semantics — click toggles the panel.
 *   - The panel opens in-flow directly below the citing segment;
 *     clicking the marker again (or Esc on the section) collapses it.
 *   - Verify OFF hides markers via ``visibility: hidden`` so the
 *     layout slot still anchors the preceding end-cap.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
  act,
} from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";
import React from "react";

vi.mock("@/addons/intelligence/api", () => ({
  getCitationChunkExcerpt: vi.fn(),
}));

import { DetailedSummaryCitationPopover } from "@/addons/intelligence/DetailedSummaryCitationPopover";
import { CitationInlinePanel } from "@/addons/intelligence/CitationInlinePanel";
import {
  CitationRailProvider,
  useCitationRail,
} from "@/addons/intelligence/CitationRailContext";
import { getCitationChunkExcerpt } from "@/addons/intelligence/api";

const linkedCitation = {
  section_path: "全体像/0",
  segment_type: "paragraph" as const,
  segment_text: "本作の概要を述べる段落。",
  chunk_ids: ["c1"],
  top_score: 0.95,
  has_citation: true,
};

const weakLinkedCitation = {
  section_path: "主要な章/場面/2",
  segment_type: "bullet" as const,
  segment_text: "確信度の低い bullet。",
  chunk_ids: ["c2"],
  top_score: 0.72,
  has_citation: true,
};

const unlinkedCitation = {
  section_path: "主要な章/場面/1",
  segment_type: "bullet" as const,
  segment_text: "根拠の弱い文。",
  chunk_ids: [],
  top_score: 0.3,
  has_citation: false,
};

function ForceVerifyOn() {
  const { setVerify } = useCitationRail();
  React.useEffect(() => {
    setVerify(true);
  }, [setVerify]);
  return null;
}

function renderMarkerWithPanel(props: {
  citation:
    | typeof linkedCitation
    | typeof weakLinkedCitation
    | typeof unlinkedCitation;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  verifyOn?: boolean;
}) {
  return render(
    <NextIntlClientProvider locale="ja" messages={{}}>
      <CitationRailProvider fileId="f1" drive="drive1">
        {props.verifyOn !== false && <ForceVerifyOn />}
        <DetailedSummaryCitationPopover citation={props.citation} />
        {props.citation.has_citation && (
          <CitationInlinePanel
            sectionPath={props.citation.section_path}
            citation={props.citation}
            segmentType="paragraph"
            videoRef={props.videoRef ?? null}
          />
        )}
      </CitationRailProvider>
    </NextIntlClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  try {
    window.localStorage.removeItem("hv.intelligence.verify");
  } catch {
    // ignore in test envs without storage
  }
});

afterEach(() => {
  cleanup();
});

describe("DetailedSummaryCitationPopover (dot marker)", () => {
  it("renders a strong dot marker when top_score >= 0.90", async () => {
    const { container } = renderMarkerWithPanel({ citation: linkedCitation });
    await waitFor(() => {
      expect(
        container.querySelector('[data-citation-marker="linked-strong"]'),
      ).not.toBeNull();
    });
  });

  it("renders a weak dot marker when has_citation is true but top_score < 0.90", async () => {
    const { container } = renderMarkerWithPanel({
      citation: weakLinkedCitation,
    });
    await waitFor(() => {
      const marker = container.querySelector(
        '[data-citation-marker="linked-weak"]',
      );
      expect(marker).not.toBeNull();
      expect(marker?.getAttribute("aria-label")).toMatch(/弱い|要確認|verify/i);
    });
  });

  it("treats top_score exactly 0.90 as strong (boundary inclusive)", async () => {
    const { container } = renderMarkerWithPanel({
      citation: { ...linkedCitation, top_score: 0.9 },
    });
    await waitFor(() => {
      expect(
        container.querySelector('[data-citation-marker="linked-strong"]'),
      ).not.toBeNull();
    });
  });

  it("treats top_score just below 0.90 as weak", async () => {
    const { container } = renderMarkerWithPanel({
      citation: { ...linkedCitation, top_score: 0.8999 },
    });
    await waitFor(() => {
      expect(
        container.querySelector('[data-citation-marker="linked-weak"]'),
      ).not.toBeNull();
    });
  });

  it("renders nothing for a no-citation (has_citation=false) segment", () => {
    const { container } = renderMarkerWithPanel({ citation: unlinkedCitation });
    expect(container.querySelector("[data-citation-marker]")).toBeNull();
    expect(container.querySelector("button")).toBeNull();
  });

  it("uses visibility:hidden (not display:none) when Verify is OFF so the slot remains", () => {
    const { container } = render(
      <NextIntlClientProvider locale="ja" messages={{}}>
        <CitationRailProvider fileId="f1" drive="drive1">
          {/* No ForceVerifyOn — Verify starts OFF */}
          <DetailedSummaryCitationPopover citation={linkedCitation} />
        </CitationRailProvider>
      </NextIntlClientProvider>,
    );
    const marker = container.querySelector<HTMLElement>(
      '[data-citation-marker="linked-strong"]',
    );
    expect(marker).not.toBeNull();
    // The slot button stays laid out; only the SVG inside is hidden.
    expect(marker?.style.visibility).toBe("hidden");
  });

  it("uses a 14px slot with verticalAlign -2px so markers line up with end punctuation", () => {
    const { container } = renderMarkerWithPanel({ citation: linkedCitation });
    const marker = container.querySelector<HTMLElement>(
      '[data-citation-marker="linked-strong"]',
    );
    expect(marker).not.toBeNull();
    expect(marker!.className).toMatch(/h-\[14px\]/);
    expect(marker!.className).toMatch(/w-\[14px\]/);
    expect(marker!.style.verticalAlign).toBe("-2px");
  });
});

describe("CitationInlinePanel (in-flow accordion)", () => {
  it("does not render the panel until the marker is clicked (Verify ON)", async () => {
    renderMarkerWithPanel({ citation: linkedCitation });
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /Strong|明確/ }),
      ).toBeInTheDocument();
    });
    expect(screen.queryByRole("region")).toBeNull();
  });

  it("click opens the panel and fetches the excerpt", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "前の文。 ",
        target: "これは抜粋テキストです。",
        suffix: " 次の文。",
        start_time: 42,
        end_time: 46,
        page: null,
      });

    renderMarkerWithPanel({ citation: linkedCitation });

    await act(async () => {
      fireEvent.click(
        await screen.findByRole("button", { name: /Strong|明確/ }),
      );
    });

    await waitFor(() => {
      expect(getCitationChunkExcerpt).toHaveBeenCalledWith("f1", "c1", "drive1");
    });
    expect(await screen.findByText(/抜粋テキスト/)).toBeInTheDocument();
  });

  it("second click collapses the panel (toggle)", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "",
        target: "ターゲットX",
        suffix: "",
        start_time: 0,
        end_time: 1,
        page: null,
      });

    renderMarkerWithPanel({ citation: linkedCitation });

    const marker = await screen.findByRole("button", { name: /Strong|明確/ });
    await act(async () => {
      fireEvent.click(marker);
    });
    await screen.findByTestId("citation-target");

    await act(async () => {
      fireEvent.click(marker);
    });
    await waitFor(() => {
      expect(screen.queryByTestId("citation-target")).toBeNull();
    });
  });

  it("renders the target text inside a highlighted <mark> with surrounding context", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "前の文脈。 ",
        target: "ここがマッチ箇所。",
        suffix: " 後ろの文脈。",
        start_time: 0,
        end_time: 5,
        page: null,
      });

    renderMarkerWithPanel({ citation: linkedCitation });
    await act(async () => {
      fireEvent.click(
        await screen.findByRole("button", { name: /Strong|明確/ }),
      );
    });

    const mark = await screen.findByTestId("citation-target");
    expect(mark.tagName).toBe("MARK");
    expect(mark.textContent).toBe("ここがマッチ箇所。");

    const paragraph = mark.parentElement!;
    expect(paragraph.textContent).toBe(
      "前の文脈。 ここがマッチ箇所。 後ろの文脈。",
    );
  });

  it("omits prefix / suffix elements when they are empty", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "",
        target: "最初のチャンク。",
        suffix: "",
        start_time: 0,
        end_time: 5,
        page: null,
      });

    renderMarkerWithPanel({ citation: linkedCitation });
    await act(async () => {
      fireEvent.click(
        await screen.findByRole("button", { name: /Strong|明確/ }),
      );
    });

    const mark = await screen.findByTestId("citation-target");
    const paragraph = mark.parentElement!;
    expect(paragraph.children.length).toBe(1);
    expect(paragraph.textContent).toBe("最初のチャンク。");
  });

  it("jump button seeks the video ref when the panel is open", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "前の文。 ",
        target: "これは抜粋テキストです。",
        suffix: " 次の文。",
        start_time: 42,
        end_time: 46,
        page: null,
      });

    const fakeVideo = {
      currentTime: 0,
      play: vi.fn().mockResolvedValue(undefined),
    } as unknown as HTMLVideoElement;
    const videoRef = {
      current: fakeVideo,
    } as React.RefObject<HTMLVideoElement | null>;

    renderMarkerWithPanel({ citation: linkedCitation, videoRef });

    await act(async () => {
      fireEvent.click(
        await screen.findByRole("button", { name: /Strong|明確/ }),
      );
    });
    await screen.findByText(/抜粋テキスト/);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /ジャンプ|Jump/ }));
    });

    expect(fakeVideo.currentTime).toBe(42);
  });

  it("renders a strong tier chip on the panel meta row", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "",
        target: "T",
        suffix: "",
        start_time: 0,
        end_time: 1,
        page: null,
      });

    const { container } = renderMarkerWithPanel({ citation: linkedCitation });
    await act(async () => {
      fireEvent.click(
        await screen.findByRole("button", { name: /Strong|明確/ }),
      );
    });

    await waitFor(() => {
      expect(
        container.querySelector('[data-citation-panel="strong"]'),
      ).not.toBeNull();
    });
  });

  it("renders a weak tier chip on the panel meta row", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c2",
        file_id: "f1",
        prefix: "",
        target: "T",
        suffix: "",
        start_time: 0,
        end_time: 1,
        page: null,
      });

    const { container } = renderMarkerWithPanel({
      citation: weakLinkedCitation,
    });
    await act(async () => {
      fireEvent.click(
        await screen.findByRole("button", { name: /Weak|要確認|弱い/ }),
      );
    });

    await waitFor(() => {
      expect(
        container.querySelector('[data-citation-panel="weak"]'),
      ).not.toBeNull();
    });
  });

  it("does not open a panel for a no-citation segment", async () => {
    renderMarkerWithPanel({ citation: unlinkedCitation });
    await new Promise((r) => setTimeout(r, 0));
    expect(getCitationChunkExcerpt).not.toHaveBeenCalled();
    expect(screen.queryByRole("region")).toBeNull();
  });
});
