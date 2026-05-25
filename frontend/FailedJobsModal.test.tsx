/**
 * Tests for FailedJobsModal — spec
 * 2026-05-24-intelligence-reindex-controls §3.2.
 *
 * Covers:
 *  1. Initial fetch of /admin/failed-jobs and rendering of every row's
 *     filename / drive / job_kind / provider / error_class.
 *  2. Per-row retry button: clicking it calls
 *     reindexFile(file_id, [<task derived from job_kind>], drive).
 *  3. Per-row "details" deep link: rendered as an SPA anchor (a real
 *     <a href>) targeting `/drive/{drive}/file/{file_id}`. The spec
 *     forbids `window.location.href` here — `<Link>` or `router.push`.
 *  4. Multi-select scaffolding: each row has a leading whitespace
 *     (24px) so the future checkbox column can land without reflowing
 *     the table.
 *  5. No emoji.
 *
 * RED-phase: the modal component does not exist yet. Vitest reports
 * the failing dynamic import as the RED signal.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  within,
} from "@testing-library/react";

vi.mock("@/addons/intelligence/api", () => ({
  getFailedJobs: vi.fn(),
  reindexFile: vi.fn(),
  resolveFailedJob: vi.fn(),
}));

// Capture next/link <Link> usage so the SPA-navigation rule is checked.
vi.mock("next/link", () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string;
    children: React.ReactNode;
    [key: string]: unknown;
  }) => (
    <a data-testid="next-link" href={href} {...rest}>
      {children}
    </a>
  ),
}));

import FailedJobsModal from "@/addons/intelligence/FailedJobsModal";
import {
  getFailedJobs,
  reindexFile,
  resolveFailedJob,
} from "@/addons/intelligence/api";

const mockedFailedJobs = getFailedJobs as unknown as ReturnType<typeof vi.fn>;
const mockedReindex = reindexFile as unknown as ReturnType<typeof vi.fn>;
const mockedResolve = resolveFailedJob as unknown as ReturnType<typeof vi.fn>;

function payload(overrides: Record<string, unknown> = {}) {
  return {
    items: [
      {
        file_id: "abc12345",
        filename: "movie.mp4",
        drive: "drive1",
        job_kind: "transcription",
        provider: "whisper_local",
        error_class: "FatalError",
        error_message_excerpt: "ffmpeg returned 1: ...",
        attempted_at: "2026-05-23T12:34:56Z",
        attempts: 3,
      },
      {
        file_id: "def98765",
        filename: "image.png",
        drive: "drive2",
        job_kind: "clip",
        provider: null,
        error_class: "TransientError",
        error_message_excerpt: "OOM",
        attempted_at: "2026-05-23T09:00:00Z",
        attempts: 1,
      },
    ],
    total: 2,
    limit: 50,
    offset: 0,
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockedFailedJobs.mockResolvedValue(payload());
  mockedReindex.mockResolvedValue({ status: "accepted" });
  mockedResolve.mockResolvedValue({ status: "resolved" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("FailedJobsModal — initial render", () => {
  it("fetches failed jobs on mount and renders one row per item", async () => {
    render(<FailedJobsModal open onClose={() => {}} />);

    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalledTimes(1));

    expect(await screen.findByText("movie.mp4")).toBeInTheDocument();
    expect(screen.getByText("image.png")).toBeInTheDocument();
    // Drive labels appear so the operator can disambiguate
    // same-named files.
    expect(screen.getByText("drive1")).toBeInTheDocument();
    expect(screen.getByText("drive2")).toBeInTheDocument();
    // job_kind + provider + error_class all surface so retry-vs-investigate
    // is decidable at a glance.
    expect(screen.getByText(/transcription/i)).toBeInTheDocument();
    expect(screen.getByText(/whisper_local/i)).toBeInTheDocument();
    expect(screen.getByText(/FatalError/)).toBeInTheDocument();
    expect(screen.getByText(/TransientError/)).toBeInTheDocument();
  });

  it("renders the attempts counter for each row", async () => {
    render(<FailedJobsModal open onClose={() => {}} />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    // Locate each row by filename and assert the attempts value lives
    // somewhere in that row. Using row-scoped queries avoids collisions
    // with other digits in the fixture (drive1, "ffmpeg returned 1:" excerpt).
    const movieRow = (await screen.findByText("movie.mp4")).closest(
      "tr, li, [role='row']",
    );
    expect(movieRow).not.toBeNull();
    expect(within(movieRow as HTMLElement).getByText("3")).toBeInTheDocument();

    const imageRow = (await screen.findByText("image.png")).closest(
      "tr, li, [role='row']",
    );
    expect(imageRow).not.toBeNull();
    expect(within(imageRow as HTMLElement).getByText("1")).toBeInTheDocument();
  });

  it("does not fetch when open=false", async () => {
    render(<FailedJobsModal open={false} onClose={() => {}} />);
    // Allow microtasks to drain
    await new Promise((r) => setTimeout(r, 10));
    expect(mockedFailedJobs).not.toHaveBeenCalled();
  });
});

describe("FailedJobsModal — retry button", () => {
  it("calls reindexFile with the row's file_id and derived task on click", async () => {
    render(<FailedJobsModal open onClose={() => {}} />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    // Locate the row for movie.mp4 (transcription / whisper_local).
    const row = screen.getByText("movie.mp4").closest("tr, li, [role='row'], div");
    expect(row).not.toBeNull();
    const retryBtn = within(row as HTMLElement).getByRole("button", {
      name: /retry|再試行|semanticSearch\.failedJobs\.retry/i,
    });
    fireEvent.click(retryBtn);

    // Spec §3.2 — retry hits the per-file reindex helper with the
    // canonical task name. job_kind="transcription" maps to "whisper",
    // job_kind="clip" maps to "clip".
    await waitFor(() => expect(mockedReindex).toHaveBeenCalled());
    const [calledFileId, calledTasks, calledDrive] = mockedReindex.mock.calls[0];
    expect(calledFileId).toBe("abc12345");
    expect(calledTasks).toEqual(["whisper"]);
    expect(calledDrive).toBe("drive1");
  });

  it("derives task=clip for job_kind=clip rows", async () => {
    render(<FailedJobsModal open onClose={() => {}} />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    const row = screen.getByText("image.png").closest("tr, li, [role='row'], div");
    const retryBtn = within(row as HTMLElement).getByRole("button", {
      name: /retry|再試行|semanticSearch\.failedJobs\.retry/i,
    });
    fireEvent.click(retryBtn);

    await waitFor(() => expect(mockedReindex).toHaveBeenCalled());
    const [calledFileId, calledTasks, calledDrive] = mockedReindex.mock.calls[0];
    expect(calledFileId).toBe("def98765");
    expect(calledTasks).toEqual(["clip"]);
    expect(calledDrive).toBe("drive2");
  });
});

describe("FailedJobsModal — deep link (SPA navigation rule)", () => {
  it("renders the details link as a next/link <a> (no window.location)", async () => {
    render(<FailedJobsModal open onClose={() => {}} />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    // Find a Link instance pointing at the file detail route.
    const links = screen.getAllByTestId("next-link");
    const movieLink = links.find(
      (el) =>
        (el.getAttribute("href") ?? "").includes("drive1") &&
        (el.getAttribute("href") ?? "").includes("abc12345"),
    );
    expect(movieLink).toBeDefined();

    // SPA navigation rule (`project_spa_navigation`): the href is a
    // relative path and the click is intercepted by next/link — we
    // proxy that by asserting an <a> with the right href is present.
    const href = movieLink!.getAttribute("href") ?? "";
    expect(href).toMatch(/\/drive\/drive1\/.+abc12345/);
    // Disallow `javascript:` / fully-qualified URLs that would force a
    // full reload.
    expect(href.startsWith("http")).toBe(false);
    expect(href.startsWith("javascript:")).toBe(false);
  });
});

describe("FailedJobsModal — resolve button", () => {
  it("calls resolveFailedJob with the row identity and removes the row", async () => {
    render(<FailedJobsModal open onClose={() => {}} />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    const row = screen.getByText("movie.mp4").closest("tr, li, [role='row'], div");
    expect(row).not.toBeNull();
    const resolveBtn = within(row as HTMLElement).getByRole("button", {
      name: /exclude|対象外|semanticSearch\.failedJobs\.resolve/i,
    });
    fireEvent.click(resolveBtn);

    await waitFor(() => expect(mockedResolve).toHaveBeenCalled());
    expect(mockedResolve).toHaveBeenCalledWith({
      file_id: "abc12345",
      job_kind: "transcription",
      provider: "whisper_local",
    });
    await waitFor(() =>
      expect(screen.queryByText("movie.mp4")).not.toBeInTheDocument(),
    );
  });
});

describe("FailedJobsModal — multi-select scaffolding (spec §3.2)", () => {
  it("reserves a 24px column at the row head for future checkboxes", async () => {
    render(<FailedJobsModal open onClose={() => {}} />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    // The spec only requires the DOM structure permit a checkbox
    // column without reflow; the simplest signal is a dedicated cell
    // / spacer per row with a width hint. We look for any element
    // carrying the `data-checkbox-slot` marker the implementation is
    // expected to use, or fall back to a literal width style.
    const slots = document.querySelectorAll(
      "[data-checkbox-slot], [style*='24px'], [class*='checkbox-slot']",
    );
    expect(slots.length).toBeGreaterThan(0);
  });
});

describe("FailedJobsModal — empty state and close", () => {
  it("renders an empty hint when total is 0", async () => {
    mockedFailedJobs.mockResolvedValue({
      items: [],
      total: 0,
      limit: 50,
      offset: 0,
    });
    render(<FailedJobsModal open onClose={() => {}} />);

    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());
    expect(
      screen.getByText(
        /no failed jobs|semanticSearch\.failedJobs\.none|失敗.*ありません/i,
      ),
    ).toBeInTheDocument();
  });

  it("calls onClose when the close button fires", async () => {
    const onClose = vi.fn();
    render(<FailedJobsModal open onClose={onClose} />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    // The modal must expose a closeable affordance — by role or by
    // localised name — so the dashboard widget can dismiss it.
    const closeBtn = screen.getByRole("button", {
      name: /close|閉じる|semanticSearch\.failedJobs\.close/i,
    });
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalled();
  });
});

describe("FailedJobsModal — UI rules", () => {
  it("renders no emoji", async () => {
    render(<FailedJobsModal open onClose={() => {}} />);
    await waitFor(() => expect(mockedFailedJobs).toHaveBeenCalled());

    expect(document.body.textContent ?? "").not.toMatch(
      /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u,
    );
  });
});
