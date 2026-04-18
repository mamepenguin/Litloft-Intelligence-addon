/**
 * Tests for DetailedSummaryCitationPopover — hover/focus-driven
 * disclosure, excerpt fetch, and jump action for video/audio chunks.
 *
 * We render the popover in isolation to decouple it from the
 * DetailedSummarySection plumbing; that way we can assert the
 * behaviour the parent component relies on (lazy fetch,
 * videoRef.currentTime seek) without re-building the whole file
 * detail page.
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
import { getCitationChunkExcerpt } from "@/addons/intelligence/api";

const linkedCitation = {
  section_path: "全体像/0",
  segment_type: "paragraph" as const,
  segment_text: "本作の概要を述べる段落。",
  chunk_ids: ["c1"],
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

function renderPopover(props: {
  citation: typeof linkedCitation | typeof unlinkedCitation;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
}) {
  return render(
    <NextIntlClientProvider locale="ja" messages={{}}>
      <DetailedSummaryCitationPopover
        fileId="f1"
        drive="drive1"
        citation={props.citation}
        videoRef={props.videoRef ?? null}
      />
    </NextIntlClientProvider>,
  );
}

describe("DetailedSummaryCitationPopover", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders a link marker when has_citation is true", () => {
    const { container } = renderPopover({ citation: linkedCitation });
    const marker = container.querySelector('[data-citation-marker="linked"]');
    expect(marker).not.toBeNull();
  });

  it("renders a missing marker with an alert tooltip when has_citation is false", () => {
    const { container } = renderPopover({ citation: unlinkedCitation });
    const marker = container.querySelector(
      '[data-citation-marker="missing"]',
    );
    expect(marker).not.toBeNull();
    expect(marker?.getAttribute("title")).toMatch(/強い根拠/);
  });

  it("opens on hover and fetches the top-1 chunk excerpt", async () => {
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

    renderPopover({ citation: linkedCitation });

    fireEvent.mouseEnter(
      screen.getByRole("button", { name: /出典を表示/ }),
    );

    await waitFor(() => {
      expect(getCitationChunkExcerpt).toHaveBeenCalledWith("f1", "c1", "drive1");
    });
    expect(await screen.findByText(/抜粋テキスト/)).toBeInTheDocument();
  });

  it("renders the target text inside a highlighted <mark> with surrounding context", async () => {
    // The popover's whole UX premise is that the user can see at a
    // glance which substring inside the excerpt is the actual match
    // vs. neighbour context. We assert the DOM shape here so a
    // refactor that accidentally flattens everything back into a
    // single span (the pre-case-A bug that prompted this split)
    // fails loudly instead of silently regressing the UI.
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

    renderPopover({ citation: linkedCitation });
    fireEvent.mouseEnter(
      screen.getByRole("button", { name: /出典を表示/ }),
    );

    const mark = await screen.findByTestId("citation-target");
    expect(mark.tagName).toBe("MARK");
    expect(mark.textContent).toBe("ここがマッチ箇所。");

    // Prefix / suffix sit as sibling spans beside the <mark>; they
    // must render as plain text (no extra highlighting) so the mark
    // remains the visual anchor.
    const paragraph = mark.parentElement!;
    expect(paragraph.textContent).toBe(
      "前の文脈。 ここがマッチ箇所。 後ろの文脈。",
    );
  });

  it("omits the prefix / suffix elements when they are empty", async () => {
    // Edge chunks (first/last in a transcript) return empty prefix or
    // suffix. Rendering empty ``<span>``s would still inflate the DOM
    // and add phantom whitespace around the mark, so the component
    // should skip them entirely.
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

    renderPopover({ citation: linkedCitation });
    fireEvent.mouseEnter(
      screen.getByRole("button", { name: /出典を表示/ }),
    );

    const mark = await screen.findByTestId("citation-target");
    const paragraph = mark.parentElement!;
    // Only the <mark> should sit inside the paragraph when both
    // neighbours are empty.
    expect(paragraph.children.length).toBe(1);
    expect(paragraph.textContent).toBe("最初のチャンク。");
  });

  it("jump button seeks the video ref when start_time is present", async () => {
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

    renderPopover({ citation: linkedCitation, videoRef });

    fireEvent.mouseEnter(
      screen.getByRole("button", { name: /出典を表示/ }),
    );

    // Wait for excerpt to load so the Jump button enables.
    await screen.findByText(/抜粋テキスト/);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /ジャンプ/ }));
    });

    expect(fakeVideo.currentTime).toBe(42);
  });

  it("stays open when the pointer moves from the trigger into the popover", async () => {
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

    renderPopover({ citation: linkedCitation });

    const trigger = screen.getByRole("button", { name: /出典を表示/ });
    fireEvent.mouseEnter(trigger);
    const dialog = await screen.findByRole("dialog");

    // User drifts off the trigger — schedules a close — then reaches
    // the popover within the grace period, which must cancel it.
    fireEvent.mouseLeave(trigger);
    fireEvent.mouseEnter(dialog);

    // Wait past the grace period. Without the cancel, the popover
    // would have closed; with it, the dialog must still be present.
    await new Promise((r) => setTimeout(r, 300));

    expect(screen.queryByRole("dialog")).not.toBeNull();
  });

  it("closes on outside pointer click", async () => {
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

    renderPopover({ citation: linkedCitation });

    fireEvent.mouseEnter(
      screen.getByRole("button", { name: /出典を表示/ }),
    );

    await screen.findByRole("dialog");

    // The popover's Escape handler is wired via window.addEventListener.
    // Clicking outside the trigger+popover fires the same dismiss path
    // (mousedown handler), which is simpler to drive from jsdom.
    act(() => {
      document.body.dispatchEvent(
        new MouseEvent("mousedown", { bubbles: true }),
      );
    });

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });
});
