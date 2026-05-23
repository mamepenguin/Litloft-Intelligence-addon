/**
 * Tests for IndexStatusWidget after spec
 * 2026-05-24-intelligence-reindex-controls §3.1.
 *
 * Covers:
 *  1. The "Reindex" button (and its ConfirmDialog) are gone — re-rendering
 *     the widget must not let an operator nuke every flag in one click.
 *  2. The failed-jobs summary row is rendered. When zero, it shows the
 *     "no failed jobs" hint. When > 0, it surfaces a clickable amber row.
 *  3. Clicking the failed-jobs row opens the FailedJobsModal.
 *  4. The Pause / Resume button still works (regression guard).
 *  5. No emoji anywhere (UI rule `feedback_no_emoji_in_ui`).
 *
 * RED-phase: the widget still renders the Reindex button today; these
 * tests fail on the current code base. They turn green only after the
 * spec §3.1 refactor lands.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
} from "@testing-library/react";

// Mock the entire addon api module so we can drive both the status
// payload and the failed-jobs count without touching the network.
vi.mock("@/addons/intelligence/api", () => ({
  getSearchStatus: vi.fn(),
  searchQueuePause: vi.fn(),
  searchQueueResume: vi.fn(),
  getFailedJobs: vi.fn(),
}));

// Mock the FailedJobsModal sibling component so we can detect "did
// the widget open it" without rendering the full modal here.
vi.mock("@/addons/intelligence/FailedJobsModal", () => ({
  default: ({ onClose }: { onClose: () => void }) => (
    <div data-testid="failed-jobs-modal-stub">
      <button onClick={onClose}>close-modal</button>
    </div>
  ),
}));

import IndexStatusWidget from "@/addons/intelligence/IndexStatusWidget";
import {
  getSearchStatus,
  searchQueuePause,
  searchQueueResume,
  getFailedJobs,
} from "@/addons/intelligence/api";

const mockedStatus = getSearchStatus as unknown as ReturnType<typeof vi.fn>;
const mockedPause = searchQueuePause as unknown as ReturnType<typeof vi.fn>;
const mockedResume = searchQueueResume as unknown as ReturnType<typeof vi.fn>;
const mockedFailedJobs = getFailedJobs as unknown as ReturnType<typeof vi.fn>;

function statusPayload(overrides: Record<string, unknown> = {}) {
  return {
    available: true,
    indexed: { total: 100, metadata: 100, clip: 80, whisper: 50, text: 90 },
    pending: { total: 0, metadata: 0, clip: 0, whisper: 0, text: 0 },
    queue: { processing: 0, waiting: 0, paused: false, tasks: {} },
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockedStatus.mockResolvedValue(statusPayload());
  mockedFailedJobs.mockResolvedValue({
    items: [],
    total: 0,
    limit: 50,
    offset: 0,
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("IndexStatusWidget — reindex button removal (spec §3.1)", () => {
  it("does NOT render a Reindex button", async () => {
    render(<IndexStatusWidget />);

    // Wait for the widget to settle (initial render is the skeleton).
    await waitFor(() => expect(mockedStatus).toHaveBeenCalled());

    // Locale-tolerant: the en.json key may or may not have been merged
    // yet, so accept the raw key path as well.
    const reindexRe = /^(reindex|semanticSearch\.reindex)$/i;
    const matches = screen.queryAllByText(reindexRe);
    expect(matches).toEqual([]);
  });

  it("does NOT render the confirm-reindex dialog text anywhere", async () => {
    render(<IndexStatusWidget />);
    await waitFor(() => expect(mockedStatus).toHaveBeenCalled());

    const confirmRe = /reindex all files|confirmReindex|semanticSearch\.confirmReindex/i;
    expect(screen.queryByText(confirmRe)).toBeNull();
  });
});

describe("IndexStatusWidget — failed-jobs summary (spec §3.1)", () => {
  it("renders a no-failed-jobs hint when count is 0", async () => {
    mockedFailedJobs.mockResolvedValue({
      items: [],
      total: 0,
      limit: 50,
      offset: 0,
    });
    render(<IndexStatusWidget />);

    // Either the localised "no failed jobs" copy, or the raw i18n key
    // — accept both so the test isn't gated on the merge script.
    await waitFor(() => {
      const noneRe = /no failed jobs|failed jobs.*0|semanticSearch\.failedJobs\.none/i;
      expect(screen.getByText(noneRe)).toBeInTheDocument();
    });
  });

  it("renders the failed-jobs count when > 0", async () => {
    mockedFailedJobs.mockResolvedValue({
      items: [
        { file_id: "abc", filename: "a.mp4", drive: "d1", job_kind: "transcription", provider: "whisper_local", error_class: "FatalError", error_message_excerpt: "err", attempted_at: "2026-05-23T00:00:00Z", attempts: 1 },
      ],
      total: 3,
      limit: 50,
      offset: 0,
    });
    render(<IndexStatusWidget />);

    // Either a rendered string containing "3" or a count-bound i18n
    // value — anything visible that names the number.
    await waitFor(() => {
      expect(
        screen.getByText(/3.*failed|failed.*3|semanticSearch\.failedJobs/i),
      ).toBeInTheDocument();
    });
  });

  it("opens the FailedJobsModal when the summary is clicked", async () => {
    mockedFailedJobs.mockResolvedValue({
      items: [
        { file_id: "abc", filename: "a.mp4", drive: "d1", job_kind: "transcription", provider: "whisper_local", error_class: "FatalError", error_message_excerpt: "err", attempted_at: "2026-05-23T00:00:00Z", attempts: 1 },
      ],
      total: 1,
      limit: 50,
      offset: 0,
    });
    render(<IndexStatusWidget />);

    // The widget renders the row as a button so the operator can
    // open it with keyboard navigation too. The accessible name
    // covers either the localised label or the raw i18n key.
    const summary = await screen.findByRole("button", {
      name: /failed jobs|semanticSearch\.failedJobs/i,
    });
    fireEvent.click(summary);

    await waitFor(() => {
      expect(screen.getByTestId("failed-jobs-modal-stub")).toBeInTheDocument();
    });
  });
});

describe("IndexStatusWidget — pause / resume regression guard", () => {
  it("still exposes the Pause button when running", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({ queue: { processing: 0, waiting: 0, paused: false, tasks: {} } }),
    );
    render(<IndexStatusWidget />);

    const pauseBtn = await screen.findByRole("button", {
      name: /pause|semanticSearch\.pause/i,
    });
    fireEvent.click(pauseBtn);
    await waitFor(() => expect(mockedPause).toHaveBeenCalled());
  });

  it("renders Resume when paused", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({ queue: { processing: 0, waiting: 0, paused: true, tasks: {} } }),
    );
    render(<IndexStatusWidget />);

    const resumeBtn = await screen.findByRole("button", {
      name: /resume|semanticSearch\.resume/i,
    });
    fireEvent.click(resumeBtn);
    await waitFor(() => expect(mockedResume).toHaveBeenCalled());
  });
});

describe("IndexStatusWidget — UI rules", () => {
  it("renders no emoji characters", async () => {
    mockedFailedJobs.mockResolvedValue({
      items: [],
      total: 5,
      limit: 50,
      offset: 0,
    });
    render(<IndexStatusWidget />);
    await waitFor(() => expect(mockedStatus).toHaveBeenCalled());

    expect(document.body.textContent ?? "").not.toMatch(
      /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u,
    );
  });
});
