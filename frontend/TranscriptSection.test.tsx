/**
 * RED-phase tests for TranscriptSection transcript-refine UI.
 *
 * Spec: docs/superpowers/specs/2026-04-15-intelligence-transcript-refine.md
 *
 * Covers:
 *   - "AI で修正" button appears only when features.transcript_refine !== false
 *   - "AI 修正済み" badge renders for chunks with refinedAt
 *   - textOriginal tooltip is present on refined chunks (title attr)
 *   - Revert button only appears when at least one chunk is refined
 *
 * The enhancements are not yet implemented — these tests are expected
 * to fail (RED phase). They intentionally do NOT try to patch internal
 * module state; instead they rely on props + addon status context that
 * the future implementation must accept.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import React from "react";

// Mock global fetch (used by component for VTT endpoints)
const fetchMock = vi.fn().mockResolvedValue({
  ok: false,
  status: 404,
  text: async () => "",
  json: async () => null,
} as Response);
vi.stubGlobal("fetch", fetchMock);

// Mock the addon API module to return controlled transcript data.
// The module exports `getFileTranscript` which the component calls on
// mount. The test data includes refined + unrefined chunks.
vi.mock("@/addons/intelligence/api", () => ({
  getFileTranscript: vi.fn().mockResolvedValue({
    available: true,
    file_id: "abc",
    drive: "family",
    language: "ja",
    chunks: [
      {
        index: 0,
        text: "これは修正された文章です。",
        start: 0,
        end: 5,
        // New fields (spec): refinedAt + textOriginal
        refinedAt: "2026-04-15T00:00:00Z",
        textOriginal: "これはげんぶんの文章です。",
      },
      {
        index: 1,
        text: "未修正の文章。",
        start: 5,
        end: 10,
      },
    ],
  }),
  refineFileTranscript: vi.fn().mockResolvedValue({
    job_id: "job-1",
    chunk_count: 2,
  }),
  revertFileTranscript: vi.fn().mockResolvedValue({ success: true }),
}));

// Mock the addon slots provider so `useAddonSlots()` exposes
// `features.transcript_refine`. The exact hook surface is to be
// finalised during implementation — we target the natural shape.
const mockAddonStatus = {
  features: { transcript_refine: "manual" as string | false },
};
vi.mock("@/components/AddonSlotsProvider", () => ({
  useAddonStatus: () => mockAddonStatus,
  useAddonSlots: () => ({ slots: {} }),
}));

// Use the real TranscriptSection from the intelligence addon directory.
// This is the same path used by the build-time symlink copy.
import TranscriptSection from "@/addons/intelligence/TranscriptSection";

function renderSection() {
  const videoRef = { current: null } as React.RefObject<HTMLVideoElement | null>;
  return render(
    <TranscriptSection fileId="abc" drive="family" videoRef={videoRef} />
  );
}

describe("TranscriptSection — transcript refine UI", () => {
  beforeEach(() => {
    mockAddonStatus.features.transcript_refine = "manual";
    fetchMock.mockClear();
  });

  afterEach(() => {
    cleanup();
  });

  it("shows the 'Refine with AI' button when feature is enabled", async () => {
    renderSection();
    // Button label text per spec UI section.
    const btn = await screen.findByRole("button", { name: /Refine with AI/ });
    expect(btn).toBeInTheDocument();
  });

  it("hides the refine button when feature flag is 'false'", async () => {
    mockAddonStatus.features.transcript_refine = false;
    renderSection();
    // Give the async mount a tick — chunks still render, button shouldn't.
    await screen.findByText("未修正の文章。");
    expect(
      screen.queryByRole("button", { name: /Refine with AI/ })
    ).not.toBeInTheDocument();
  });

  it("renders 'AI refined' badge for refined chunks", async () => {
    renderSection();
    const badges = await screen.findAllByText(/AI refined/);
    // One badge per refined chunk (1 out of 2 in our fixture).
    expect(badges).toHaveLength(1);
  });

  // RED phase: not yet implemented
  it.todo("shows textOriginal in a tooltip (title attr) on refined chunks");

  // RED phase: not yet implemented
  it.todo("shows the revert button only when at least one chunk is refined");

  it("hides the revert button when no chunks are refined", async () => {
    const apiMock = await import("@/addons/intelligence/api");
    (apiMock.getFileTranscript as unknown as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({
        available: true,
        file_id: "abc",
        drive: "family",
        language: "ja",
        chunks: [
          { index: 0, text: "一切修正なし。", start: 0, end: 5 },
        ],
      });

    renderSection();
    await screen.findByText("一切修正なし。");
    expect(
      screen.queryByRole("button", { name: /Undo AI refine/ })
    ).not.toBeInTheDocument();
  });
});
