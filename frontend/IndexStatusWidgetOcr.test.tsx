/**
 * SPEC-ADDON-050 I7: the dashboard counts OCR like the other indexing work — a
 * progress bar over pictures and a queue card for the picture being read.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  getSearchStatus: vi.fn(),
  searchQueuePause: vi.fn(),
  searchQueueResume: vi.fn(),
  getFailedJobs: vi.fn(),
}));

import IndexStatusWidget from "@/addons/intelligence/IndexStatusWidget";
import { getSearchStatus } from "@/addons/intelligence/api";

const mockedStatus = getSearchStatus as unknown as ReturnType<typeof vi.fn>;

const OCR_LABEL = /^Text in image$|semanticSearch\.tasks\.ocr\.label/;

function statusPayload(overrides: Record<string, unknown> = {}) {
  return {
    available: true,
    indexed: { total: 40, metadata: 40, clip: 40, whisper: 0, text: 0, ocr: 3 },
    pending: { total: 2, metadata: 0, clip: 0, whisper: 0, text: 0, ocr: 2 },
    queue: { processing: 0, waiting: 0, paused: false, tasks: {} },
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockedStatus.mockResolvedValue(statusPayload());
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("IndexStatusWidget OCR (SPEC-ADDON-050 I7)", () => {
  it("draws a progress bar over pictures, done over done-plus-pending", async () => {
    render(<IndexStatusWidget />);

    expect(
      await screen.findByText(/Text in image.*3 \/ 5|semanticSearch\.ocrFiles/),
    ).toBeInTheDocument();
  });

  it("draws the OCR queue card with the picture being read", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({
        queue: {
          processing: 1,
          waiting: 0,
          paused: false,
          tasks: {
            ocr: {
              waiting: 0,
              processing: [{ file_id: "pic-file", filename: "slide.png" }],
            },
          },
        },
      }),
    );
    render(<IndexStatusWidget />);

    const label = await screen.findByText(OCR_LABEL);
    const row = label.closest("div.rounded-lg") as HTMLElement;
    expect(row).not.toBeNull();
    expect(row.querySelector(".lucide-scan-text")).not.toBeNull();
    expect(await screen.findByText("slide.png")).toBeInTheDocument();
  });
});
