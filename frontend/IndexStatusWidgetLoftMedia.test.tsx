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
import en from "@/addons/intelligence/messages/en.json";

const mockedStatus = getSearchStatus as unknown as ReturnType<typeof vi.fn>;

function statusPayload(tasks: Record<string, unknown>) {
  return {
    available: true,
    indexed: { total: 40, metadata: 40, clip: 40, whisper: 0, text: 0, frame_scan: 3 },
    pending: { total: 2, metadata: 0, clip: 0, whisper: 0, text: 0, frame_scan: 2 },
    queue: { processing: 2, waiting: 1, paused: false, tasks },
  };
}

function rowOf(label: string): HTMLElement {
  const element = screen.getByText(label, { exact: true });
  const row = element.closest("div.rounded-lg") as HTMLElement | null;
  expect(row).not.toBeNull();
  return row as HTMLElement;
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("IndexStatusWidget loft_media row", () => {
  it("SPEC-ADDON-117 SPEC-ADDON-178: draws the loft_media row after the scene change scan with its file", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({
        frame_scan: {
          waiting: 0,
          processing: [{ file_id: "vid-file", filename: "talk.mp4" }],
        },
        loft_media: {
          waiting: 1,
          processing: [{ file_id: "yt-file", filename: "cooking.loft" }],
        },
      }),
    );
    render(<IndexStatusWidget />);

    await screen.findByText("cooking.loft");
    const fetchRow = rowOf(en.semanticSearch.tasks.loft_media.label);
    const scanRow = rowOf(en.semanticSearch.tasks.frame_scan.label);

    expect(fetchRow.querySelector(".lucide-download")).not.toBeNull();
    expect(fetchRow).toHaveTextContent("cooking.loft");
    expect(
      scanRow.compareDocumentPosition(fetchRow) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    const rows = Array.from(document.querySelectorAll("div.rounded-lg"));
    expect(rows.indexOf(fetchRow)).toBe(rows.indexOf(scanRow) + 1);
  });
});
