/**
 * Tests for IndexStatusWidget after spec
 * 2026-05-24-intelligence-reindex-controls §3.1.
 *
 * Covers:
 *  1. The "Reindex" button (and its ConfirmDialog) are gone — re-rendering
 *     the widget must not let an operator nuke every flag in one click.
 *  2. The failed-jobs warning is *not* here — it moved to the
 *     `dashboard-alerts` slot (see FailedJobsAlert.test.tsx).
 *  3. The Pause / Resume button still works (regression guard).
 *  4. No emoji anywhere (UI rule `feedback_no_emoji_in_ui`).
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

describe("IndexStatusWidget — the failed-jobs warning is not here", () => {
  it("does not poll for failed jobs at all", async () => {
    // The warning moved to `dashboard-alerts` (FailedJobsAlert), above
    // the drive cards. Two pollers would say the same thing twice and
    // ask the server twice for it.
    render(<IndexStatusWidget />);
    await waitFor(() => expect(mockedStatus).toHaveBeenCalled());

    expect(mockedFailedJobs).not.toHaveBeenCalled();
    expect(screen.queryByTestId("failed-jobs-modal-stub")).toBeNull();
    expect(
      screen.queryByText(/failed jobs|semanticSearch\.failedJobs/i),
    ).toBeNull();
  });
});

describe("IndexStatusWidget — pause / resume regression guard", () => {
  it("renders chapter suggestion queue activity", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({
        queue: {
          processing: 1,
          waiting: 0,
          paused: false,
          tasks: {
            chapter_suggestions: {
              waiting: 0,
              processing: [{ file_id: "chapter-file", filename: "talk.mp4" }],
            },
          },
        },
      }),
    );
    render(<IndexStatusWidget />);

    expect(
      await screen.findByText(
        /AI chapter candidates|semanticSearch\.tasks\.chapter_suggestions\.label/,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("talk.mp4")).toBeInTheDocument();
  });

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

describe("IndexStatusWidget — idle queues are collapsed (ADM-1)", () => {
  /**
   * The eleven task kinds the backend reports, with the label each one
   * renders. Matching the rendered label rather than the message key is
   * deliberate: next-intl resolves against the real catalogue in this
   * suite, so a key-shaped matcher matches nothing and a "row is absent"
   * assertion passes without ever having looked at a row.
   */
  const TASK_LABEL: Record<string, string> = {
    metadata: "Metadata extraction",
    clip: "Image analysis (CLIP)",
    whisper: "Transcription (Whisper)",
    text_content: "Text extraction",
    auto_tags: "AI tag candidates",
    summaries: "AI summaries",
    vision_describe: "AI image descriptions",
    transcript_refine: "AI transcript cleanup",
    retrieval_keywords: "AI search keyword expansion",
    chapter_suggestions: "AI chapter candidates",
    video_visual: "Visual index",
  };
  const ELEVEN_KINDS = Object.keys(TASK_LABEL);

  type Breakdown = { waiting: number; processing: { file_id: string; filename: string }[] };

  /** Every task kind reporting state, with the named ones actually busy. */
  function tasksPayload(busy: Record<string, Breakdown> = {}) {
    return Object.fromEntries(
      ELEVEN_KINDS.map((kind) => [kind, busy[kind] ?? { waiting: 0, processing: [] }]),
    );
  }

  const label = (kind: string) => TASK_LABEL[kind];
  const disclosure = () =>
    screen.findByRole("button", { name: /idle queue/i });

  it("draws no task row when every queue is stopped", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({
        queue: { processing: 0, waiting: 0, paused: false, tasks: tasksPayload() },
      }),
    );
    render(<IndexStatusWidget />);
    await waitFor(() => expect(mockedStatus).toHaveBeenCalled());
    await disclosure();

    for (const kind of ELEVEN_KINDS) {
      expect(screen.queryByText(label(kind))).toBeNull();
    }
    // The "Current activity" heading goes with them.
    expect(screen.queryByText("Current activity")).toBeNull();
  });

  it("keeps the queue total line when every queue is stopped", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({
        queue: { processing: 0, waiting: 0, paused: false, tasks: tasksPayload() },
      }),
    );
    render(<IndexStatusWidget />);

    expect(await screen.findByText("Task queue")).toBeInTheDocument();
    expect(screen.getByText("0 processing / 0 waiting")).toBeInTheDocument();
  });

  it("draws a row only for the queues that are actually moving", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({
        queue: {
          processing: 1,
          waiting: 3,
          paused: false,
          tasks: tasksPayload({
            whisper: { waiting: 0, processing: [{ file_id: "a", filename: "talk.mp4" }] },
            summaries: { waiting: 3, processing: [] },
          }),
        },
      }),
    );
    render(<IndexStatusWidget />);
    await waitFor(() => expect(mockedStatus).toHaveBeenCalled());
    await disclosure();

    expect(screen.getByText(label("whisper"))).toBeInTheDocument();
    expect(screen.getByText(label("summaries"))).toBeInTheDocument();
    expect(screen.getByText("Current activity")).toBeInTheDocument();
    for (const kind of ELEVEN_KINDS.filter((k) => k !== "whisper" && k !== "summaries")) {
      expect(screen.queryByText(label(kind))).toBeNull();
    }
  });

  it("reaches every stopped queue through the disclosure", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({
        queue: {
          processing: 1,
          waiting: 0,
          paused: false,
          tasks: tasksPayload({
            whisper: { waiting: 0, processing: [{ file_id: "a", filename: "talk.mp4" }] },
          }),
        },
      }),
    );
    render(<IndexStatusWidget />);

    const toggle = await disclosure();
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    // Ten are idle; the count has to say so, or the row is a mystery box.
    expect(toggle.textContent).toContain("10");
    fireEvent.click(toggle);

    // All eleven are now on screen: the busy one above, the ten below.
    for (const kind of ELEVEN_KINDS) {
      expect(screen.getByText(label(kind))).toBeInTheDocument();
    }
    expect(
      screen.getByRole("button", { name: /idle queue/i }),
    ).toHaveAttribute("aria-expanded", "true");
  });

  it("still offers the disclosure when nothing is running", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({
        queue: { processing: 0, waiting: 0, paused: false, tasks: tasksPayload() },
      }),
    );
    render(<IndexStatusWidget />);

    const toggle = await disclosure();
    expect(toggle.textContent).toContain("11");
    fireEvent.click(toggle);
    for (const kind of ELEVEN_KINDS) {
      expect(screen.getByText(label(kind))).toBeInTheDocument();
    }
  });

  it("offers no disclosure when the backend reports no task state at all", async () => {
    mockedStatus.mockResolvedValue(
      statusPayload({
        queue: { processing: 0, waiting: 0, paused: false, tasks: {} },
      }),
    );
    render(<IndexStatusWidget />);
    await screen.findByText("Task queue");

    expect(screen.queryByRole("button", { name: /idle queue/i })).toBeNull();
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
