/**
 * SPEC-ADDON-193 I11: the Index details dialog offers the frame OCR row for a
 * `.loft` whose `loft_source` is "youtube" or "web", under a native video's
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
    filename: "reference.loft",
    video_frame_ocr: "on_index",
    status: {
      metadata: true,
      clip: true,
      whisper: true,
      text: false,
      frame_ocr: false,
      frame_scan: true,
    },
    indexed_at: "2026-11-28T10:00:00Z",
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

describe("IndexDetailsDialog frame OCR row for a web reference (SPEC-ADDON-193 I11)", () => {
  it("offers it for a web .loft as for a YouTube .loft", async () => {
    expect(await offersFrameOcr(LOFT_MIME, { loft_source: "youtube" })).toBe(true);
    expect(await offersFrameOcr(LOFT_MIME, { loft_source: "web" })).toBe(true);
  });

  it.each([
    ["a .loft of source other", "other"],
    ["an unclassified .loft", null],
  ])("never offers it for %s", async (_name, source) => {
    expect(await offersFrameOcr(LOFT_MIME, { loft_source: "web" })).toBe(true);
    expect(await offersFrameOcr(LOFT_MIME, { loft_source: source })).toBe(false);
  });

  it("follows the native mode rule for a web .loft", async () => {
    const native = await offersFrameOcr("video/mp4", { video_frame_ocr: "false" });
    const web = await offersFrameOcr(LOFT_MIME, {
      loft_source: "web",
      video_frame_ocr: "false",
    });
    const webOn = await offersFrameOcr(LOFT_MIME, { loft_source: "web" });

    expect(web).toBe(native);
    expect(webOn).toBe(true);
  });
});
