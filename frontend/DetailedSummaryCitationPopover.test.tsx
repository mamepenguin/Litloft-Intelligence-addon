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
        text: "これは抜粋テキストです。",
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

  it("jump button seeks the video ref when start_time is present", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        text: "これは抜粋テキストです。",
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

  it("closes on outside pointer click", async () => {
    (getCitationChunkExcerpt as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValue({
        chunk_id: "c1",
        file_id: "f1",
        text: "抜粋",
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
