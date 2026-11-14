/**
 * SPEC-ADDON-113 item 6: the frame scan is counted like the other indexing work —
 * a progress bar over native videos and a queue row for the video being scanned.
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

function statusPayload(overrides: Record<string, unknown> = {}) {
  return {
    available: true,
    indexed: { total: 40, metadata: 40, clip: 40, whisper: 0, text: 0, frame_scan: 3 },
    pending: { total: 2, metadata: 0, clip: 0, whisper: 0, text: 0, frame_scan: 2 },
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

describe("IndexStatusWidget frame scan (SPEC-ADDON-113)", () => {
  it("SPEC-ADDON-113: draws a Scene change scan bar, done over done-plus-pending", async () => {
    render(<IndexStatusWidget />);

    expect(await screen.findByText(/Scene change scan.*3 \/ 5/)).toBeInTheDocument();
  });

  it("SPEC-ADDON-113: draws the frame_scan queue row with the video being scanned", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({
        queue: {
          processing: 1,
          waiting: 0,
          paused: false,
          tasks: {
            frame_scan: {
              waiting: 0,
              processing: [{ file_id: "vid-file", filename: "talk.mp4" }],
            },
          },
        },
      }),
    );
    render(<IndexStatusWidget />);

    const label = await screen.findByText(/^Scene change scan$/);
    const row = label.closest("div.rounded-lg") as HTMLElement;
    expect(row).not.toBeNull();
    expect(row.querySelector(".lucide-scan-line")).not.toBeNull();
    expect(await screen.findByText("talk.mp4")).toBeInTheDocument();
  });
});
