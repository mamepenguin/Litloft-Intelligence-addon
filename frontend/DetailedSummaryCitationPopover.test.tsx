/**
 * Tests for the detailed-summary citation surface: marker (this file)
 * plus the absolute-overlay `CitationInlinePanel` directly beneath the
 * citing segment.
 *
 * Interaction contract under test:
 *   - Hover opens the panel; grace-period cancellation lets the cursor
 *     move from marker → panel without dismissing.
 *   - Click pins the panel so the reader can actually reach the Jump
 *     button without a cursor-race; re-click or close button dismiss.
 *   - Missing-citation markers don't open anything (title tooltip is
 *     the only affordance).
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
import { CitationRailProvider } from "@/addons/intelligence/CitationRailContext";
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

// Fixtures for adjacent-chunk highlight extension. top-1 is at
// transcript:10 throughout; top-2/3 idx choice exercises each path.
const citationForwardAdjacent = {
  section_path: "全体像/forward",
  segment_type: "paragraph" as const,
  segment_text: "前方に隣接 chunk が続く段落。",
  chunk_ids: ["transcript:10", "transcript:11"],
  top_score: 0.92,
  has_citation: true,
};

const citationBackwardAdjacent = {
  section_path: "全体像/backward",
  segment_type: "paragraph" as const,
  segment_text: "後方に隣接 chunk が先行する段落。",
  chunk_ids: ["transcript:10", "transcript:9"],
  top_score: 0.92,
  has_citation: true,
};

const citationBothAdjacent = {
  section_path: "全体像/both",
  segment_type: "paragraph" as const,
  segment_text: "前後に隣接 chunk がある段落。",
  chunk_ids: ["transcript:10", "transcript:11", "transcript:9"],
  top_score: 0.93,
  has_citation: true,
};

const citationGap2 = {
  section_path: "全体像/gap2",
  segment_type: "paragraph" as const,
  segment_text: "top-2 が 2 chunks 離れている段落 (拡張しない)。",
  chunk_ids: ["transcript:10", "transcript:12"],
  top_score: 0.90,
  has_citation: true,
};

const citationTop3Adjacent = {
  section_path: "全体像/top3adj",
  segment_type: "paragraph" as const,
  segment_text: "top-2 は離れているが top-3 が前方隣接。",
  chunk_ids: ["transcript:10", "transcript:20", "transcript:11"],
  top_score: 0.90,
  has_citation: true,
};

function renderMarkerWithPanel(props: {
  citation:
    | typeof linkedCitation
    | typeof weakLinkedCitation
    | typeof unlinkedCitation
    | typeof citationForwardAdjacent
    | typeof citationBackwardAdjacent
    | typeof citationBothAdjacent
    | typeof citationGap2
    | typeof citationTop3Adjacent;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
}) {
  return render(
    <NextIntlClientProvider locale="ja" messages={{}}>
      <CitationRailProvider fileId="f1" drive="drive1">
        <DetailedSummaryCitationPopover citation={props.citation} />
        <CitationInlinePanel
          sectionPath={props.citation.section_path}
          videoRef={props.videoRef ?? null}
        />
      </CitationRailProvider>
    </NextIntlClientProvider>,
  );
}

describe("DetailedSummaryCitationPopover + CitationInlinePanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders a strong link marker when top_score >= 0.90", () => {
    const { container } = renderMarkerWithPanel({ citation: linkedCitation });
    const marker = container.querySelector(
      '[data-citation-marker="linked-strong"]',
    );
    expect(marker).not.toBeNull();
  });

  it("renders a weak link marker when has_citation is true but top_score < 0.90", () => {
    const { container } = renderMarkerWithPanel({
      citation: weakLinkedCitation,
    });
    const marker = container.querySelector(
      '[data-citation-marker="linked-weak"]',
    );
    expect(marker).not.toBeNull();
    // Weak tier communicates "verify" via aria-label and title.
    expect(marker?.getAttribute("aria-label")).toMatch(/弱い|要確認/);
    expect(marker?.getAttribute("title")).toMatch(/弱い|要確認/);
  });

  it("treats top_score exactly 0.90 as strong (boundary inclusive)", () => {
    const { container } = renderMarkerWithPanel({
      citation: { ...linkedCitation, top_score: 0.9 },
    });
    expect(
      container.querySelector('[data-citation-marker="linked-strong"]'),
    ).not.toBeNull();
  });

  it("treats top_score just below 0.90 as weak", () => {
    const { container } = renderMarkerWithPanel({
      citation: { ...linkedCitation, top_score: 0.8999 },
    });
    expect(
      container.querySelector('[data-citation-marker="linked-weak"]'),
    ).not.toBeNull();
  });

  it("applies dashed stroke styling to the weak-tier Link2 icon", () => {
    const { container } = renderMarkerWithPanel({
      citation: weakLinkedCitation,
    });
    const marker = container.querySelector(
      '[data-citation-marker="linked-weak"]',
    );
    const icon = marker?.querySelector("svg");
    expect(icon?.className.baseVal ?? icon?.getAttribute("class") ?? "").toMatch(
      /stroke-dasharray/,
    );
  });

  it("does not apply dashed stroke styling to the strong-tier icon", () => {
    const { container } = renderMarkerWithPanel({ citation: linkedCitation });
    const marker = container.querySelector(
      '[data-citation-marker="linked-strong"]',
    );
    const icon = marker?.querySelector("svg");
    expect(
      icon?.className.baseVal ?? icon?.getAttribute("class") ?? "",
    ).not.toMatch(/stroke-dasharray/);
  });

  it("opens the inline panel on hover for weak-tier citations too", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c2",
        file_id: "f1",
        prefix: "",
        target: "弱いが関連する抜粋",
        suffix: "",
        start_time: 10,
        end_time: 15,
        page: null,
      });

    renderMarkerWithPanel({ citation: weakLinkedCitation });

    fireEvent.mouseEnter(screen.getByRole("button", { name: /弱い|要確認/ }));

    await waitFor(() => {
      expect(getCitationChunkExcerpt).toHaveBeenCalledWith(
        "f1",
        "c2",
        "drive1",
      );
    });
    expect(await screen.findByText(/弱いが関連する抜粋/)).toBeInTheDocument();
  });

  it("renders nothing when has_citation is false", () => {
    // "No strong single-source match" is a retrieval outcome, not a
    // hallucination warning — real fabrications keep high cosine and
    // land in the has_citation=true branch. The prior alert-triangle
    // was noise, so the component should render nothing here.
    const { container } = renderMarkerWithPanel({ citation: unlinkedCitation });
    expect(container.querySelector("[data-citation-marker]")).toBeNull();
    expect(container.querySelector("button")).toBeNull();
  });

  it("opens the inline panel on hover and fetches the top-1 chunk excerpt", async () => {
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

    fireEvent.mouseEnter(
      screen.getByRole("button", { name: /出典を表示/ }),
    );

    await waitFor(() => {
      expect(getCitationChunkExcerpt).toHaveBeenCalledWith("f1", "c1", "drive1");
    });
    expect(await screen.findByText(/抜粋テキスト/)).toBeInTheDocument();
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
    fireEvent.mouseEnter(
      screen.getByRole("button", { name: /出典を表示/ }),
    );

    const mark = await screen.findByTestId("citation-target");
    expect(mark.tagName).toBe("MARK");
    expect(mark.textContent).toBe("ここがマッチ箇所。");

    const paragraph = mark.parentElement!;
    expect(paragraph.textContent).toBe(
      "前の文脈。 ここがマッチ箇所。 後ろの文脈。",
    );
  });

  it("omits the prefix / suffix elements when they are empty", async () => {
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
    fireEvent.mouseEnter(
      screen.getByRole("button", { name: /出典を表示/ }),
    );

    const mark = await screen.findByTestId("citation-target");
    const paragraph = mark.parentElement!;
    expect(paragraph.children.length).toBe(1);
    expect(paragraph.textContent).toBe("最初のチャンク。");
  });

  it("cursor handoff from marker to panel keeps the panel open past the grace window", async () => {
    // Core UX invariant for the hover pattern: dragging the cursor off
    // the marker into the panel body must not race the close timer.
    // The panel's own mouseenter has to cancel the scheduled close.
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "",
        target: "抜粋",
        suffix: "",
        start_time: 0,
        end_time: 1,
        page: null,
      });

    renderMarkerWithPanel({ citation: linkedCitation });

    const marker = screen.getByRole("button", { name: /出典を表示/ });
    fireEvent.mouseEnter(marker);
    const panel = await screen.findByRole("region");

    fireEvent.mouseLeave(marker);
    fireEvent.mouseEnter(panel);

    await new Promise((r) => setTimeout(r, 250));

    expect(screen.queryByRole("region")).not.toBeNull();
  });

  it("closes on mouseleave once the grace window elapses", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "",
        target: "抜粋",
        suffix: "",
        start_time: 0,
        end_time: 1,
        page: null,
      });

    renderMarkerWithPanel({ citation: linkedCitation });

    const marker = screen.getByRole("button", { name: /出典を表示/ });
    fireEvent.mouseEnter(marker);
    await screen.findByRole("region");

    fireEvent.mouseLeave(marker);

    // Grace is 160 ms; wait comfortably past it inside an act() so the
    // setState triggered by the timer fires under React's test harness.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 250));
    });
    expect(screen.queryByRole("region")).toBeNull();
  });

  it("click pins the panel so mouseleave does not dismiss it", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "",
        target: "抜粋",
        suffix: "",
        start_time: 0,
        end_time: 1,
        page: null,
      });

    renderMarkerWithPanel({ citation: linkedCitation });

    const marker = screen.getByRole("button", { name: /出典を表示/ });
    fireEvent.mouseEnter(marker);
    fireEvent.click(marker);
    await screen.findByRole("region");

    // Even leaving the marker without a handoff must not close it —
    // pinning is the whole point of click support.
    fireEvent.mouseLeave(marker);
    await new Promise((r) => setTimeout(r, 300));
    expect(screen.queryByRole("region")).not.toBeNull();
  });

  it("jump button seeks the video ref after the panel is pinned", async () => {
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

    const marker = screen.getByRole("button", { name: /出典を表示/ });
    fireEvent.mouseEnter(marker);
    fireEvent.click(marker);
    await screen.findByText(/抜粋テキスト/);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /ジャンプ/ }));
    });

    expect(fakeVideo.currentTime).toBe(42);
  });

  it("re-clicking a pinned marker dismisses the panel", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "",
        target: "抜粋テキスト",
        suffix: "",
        start_time: 0,
        end_time: 1,
        page: null,
      });

    renderMarkerWithPanel({ citation: linkedCitation });

    const marker = screen.getByRole("button", { name: /出典を表示/ });
    fireEvent.mouseEnter(marker);
    fireEvent.click(marker);
    await screen.findByText(/抜粋テキスト/);

    fireEvent.click(marker);
    await waitFor(() => {
      expect(screen.queryByText(/抜粋テキスト/)).toBeNull();
    });
  });

  it("does not render a marker or panel for a no-citation segment", async () => {
    const { container } = renderMarkerWithPanel({ citation: unlinkedCitation });

    // No button to hover / click; nothing to fetch; no panel to open.
    expect(container.querySelector("button")).toBeNull();
    await new Promise((r) => setTimeout(r, 0));
    expect(getCitationChunkExcerpt).not.toHaveBeenCalled();
    expect(screen.queryByRole("region")).toBeNull();
  });

  it("panel close button clears a pinned activation", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        prefix: "",
        target: "抜粋テキスト",
        suffix: "",
        start_time: 0,
        end_time: 1,
        page: null,
      });

    renderMarkerWithPanel({ citation: linkedCitation });

    const marker = screen.getByRole("button", { name: /出典を表示/ });
    fireEvent.mouseEnter(marker);
    fireEvent.click(marker);
    await screen.findByText(/抜粋テキスト/);

    fireEvent.click(screen.getByRole("button", { name: /閉じる/ }));

    await waitFor(() => {
      expect(screen.queryByText(/抜粋テキスト/)).toBeNull();
    });
  });

  // --- Adjacent-chunk highlight extension (idx_gap == 1 only) -----------
  //
  // When top-2 or top-3 chunks are immediately adjacent to top-1 (gap=1),
  // the UI extends the highlight into the neighbour excerpt so ASR
  // chunk-boundary splits show up as one contiguous cited region. See
  // hako k0XoWYoUBAhtylHc94KTI and CITATION-PIPELINE.md Stage 6.

  async function openPanelWithExcerpt(citation: Parameters<typeof renderMarkerWithPanel>[0]["citation"]) {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: citation.chunk_ids[0],
        file_id: "f1",
        prefix: "前の文。 ",
        target: "ターゲット chunk。",
        suffix: " 次の文。",
        start_time: 10,
        end_time: 15,
        page: null,
      });
    renderMarkerWithPanel({ citation });
    const marker = screen.getByRole("button");
    fireEvent.mouseEnter(marker);
    await screen.findByTestId("citation-target");
  }

  it("does not extend when only top-1 chunk is present", async () => {
    await openPanelWithExcerpt(linkedCitation);
    expect(screen.queryByTestId("citation-extended-prefix")).toBeNull();
    expect(screen.queryByTestId("citation-extended-suffix")).toBeNull();
  });

  it("marks suffix as extended when top-2 is the forward neighbour", async () => {
    await openPanelWithExcerpt(citationForwardAdjacent);
    const suffixMark = screen.getByTestId("citation-extended-suffix");
    expect(suffixMark.tagName).toBe("MARK");
    expect(suffixMark.textContent).toContain("次の文");
    expect(screen.queryByTestId("citation-extended-prefix")).toBeNull();
  });

  it("marks prefix as extended when top-2 is the backward neighbour", async () => {
    await openPanelWithExcerpt(citationBackwardAdjacent);
    const prefixMark = screen.getByTestId("citation-extended-prefix");
    expect(prefixMark.tagName).toBe("MARK");
    expect(prefixMark.textContent).toContain("前の文");
    expect(screen.queryByTestId("citation-extended-suffix")).toBeNull();
  });

  it("extends only one side when both top-2 and top-3 are adjacent on opposite sides", async () => {
    // citationBothAdjacent: chunk_ids = [t:10, t:11 (forward), t:9 (backward)]
    // Rule: top-2 wins over top-3, so forward extends and backward is
    // dropped. Prevents the full-excerpt highlight case that reads as
    // noise.
    await openPanelWithExcerpt(citationBothAdjacent);
    expect(screen.getByTestId("citation-extended-suffix").tagName).toBe("MARK");
    expect(screen.queryByTestId("citation-extended-prefix")).toBeNull();
  });

  it("prefers top-2's side when top-2 is backward and top-3 is forward", async () => {
    const citation = {
      section_path: "全体像/top2-backward",
      segment_type: "paragraph" as const,
      segment_text: "top-2 が後方隣接、top-3 が前方隣接。",
      chunk_ids: ["transcript:10", "transcript:9", "transcript:11"],
      top_score: 0.92,
      has_citation: true,
    };
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "transcript:10",
        file_id: "f1",
        prefix: "前の文。 ",
        target: "ターゲット。",
        suffix: " 次の文。",
        start_time: 10,
        end_time: 15,
        page: null,
      });
    renderMarkerWithPanel({ citation });
    fireEvent.mouseEnter(screen.getByRole("button"));
    await screen.findByTestId("citation-target");
    expect(screen.getByTestId("citation-extended-prefix").tagName).toBe("MARK");
    expect(screen.queryByTestId("citation-extended-suffix")).toBeNull();
  });

  it("does not extend when top-2 is 2 chunks away (idx_gap == 2)", async () => {
    await openPanelWithExcerpt(citationGap2);
    expect(screen.queryByTestId("citation-extended-prefix")).toBeNull();
    expect(screen.queryByTestId("citation-extended-suffix")).toBeNull();
  });

  it("extends when top-3 (not top-2) is the adjacent neighbour", async () => {
    await openPanelWithExcerpt(citationTop3Adjacent);
    expect(screen.getByTestId("citation-extended-suffix").tagName).toBe("MARK");
    expect(screen.queryByTestId("citation-extended-prefix")).toBeNull();
  });
});
