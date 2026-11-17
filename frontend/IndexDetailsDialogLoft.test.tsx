/**
 * SPEC-ADDON-117 item 1.8, I12: the Index details dialog offers the frame OCR row
 * for a `.loft` exactly when `loft_source` is "youtube", under a native video's
 * mode rule.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, render, screen, fireEvent } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  getIndexDetails: vi.fn(),
  reindexFile: vi.fn(),
}));

import IndexDetailsMenuItem from "@/addons/intelligence/IndexDetailsMenuItem";
import { ShortcutsProvider } from "@/components/ShortcutsProvider";
import { getIndexDetails } from "@/addons/intelligence/api";

const mockedGet = getIndexDetails as unknown as ReturnType<typeof vi.fn>;

const LOFT_MIME = "application/vnd.litloft.loft+json";
const ENTRY_LABEL = /index details|インデックス詳細|semanticSearch\.indexDetails\.title/i;
const FRAME_OCR_LABEL = /^Text in video$|semanticSearch\.tasks\.frame_ocr\.label/;
const METADATA_LABEL = /^Metadata extraction$|semanticSearch\.tasks\.metadata/;
const CLIP_LABEL = /^Image analysis \(CLIP\)$|semanticSearch\.tasks\.clip\.label/;

function details(overrides: Record<string, unknown> = {}) {
  return {
    file_id: "abc12345",
    drive: "drive1",
    filename: "cooking.loft",
    video_frame_ocr: "on_index",
    status: {
      metadata: true,
      clip: true,
      whisper: true,
      text: false,
      frame_ocr: false,
      frame_scan: true,
    },
    indexed_at: "2026-11-16T10:00:00Z",
    embeddings: {},
    provider_stats: {},
    loft_source: null,
    ...overrides,
  };
}

async function offersFrameOcr(mimeType: string, payload: Record<string, unknown>) {
  cleanup();
  mockedGet.mockResolvedValue(details(payload));
  render(
    <ShortcutsProvider>
      <IndexDetailsMenuItem
        fileId="abc12345"
        drive="drive1"
        mimeType={mimeType}
        fileType="video"
      />
    </ShortcutsProvider>,
  );
  fireEvent.click(screen.getByRole("menuitem", { name: ENTRY_LABEL }));
  await screen.findByText(METADATA_LABEL);
  await screen.findByText(CLIP_LABEL);
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
  return screen.queryByText(FRAME_OCR_LABEL) !== null;
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("IndexDetailsDialog frame OCR row for a .loft (SPEC-ADDON-117 I12)", () => {
  it("SPEC-ADDON-117 I12: offers it for a YouTube .loft as for a native video", async () => {
    expect(await offersFrameOcr("video/mp4", {})).toBe(true);
    expect(await offersFrameOcr(LOFT_MIME, { loft_source: "youtube" })).toBe(true);
  });

  it.each([
    ["another provider", "other"],
    ["an unclassified .loft", null],
  ])("SPEC-ADDON-117 I12: never offers it for %s", async (_name, source) => {
    expect(await offersFrameOcr(LOFT_MIME, { loft_source: "youtube" })).toBe(true);
    expect(await offersFrameOcr(LOFT_MIME, { loft_source: source })).toBe(false);
  });

  it("SPEC-ADDON-117 I12: follows the native mode rule for a YouTube .loft", async () => {
    const native = await offersFrameOcr("video/mp4", { video_frame_ocr: "false" });
    const loft = await offersFrameOcr(LOFT_MIME, {
      loft_source: "youtube",
      video_frame_ocr: "false",
    });
    const loftOn = await offersFrameOcr(LOFT_MIME, { loft_source: "youtube" });

    expect(loft).toBe(native);
    expect(loftOn).toBe(true);
  });
});
