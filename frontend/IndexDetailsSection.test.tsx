/**
 * Tests for IndexDetailsSection — spec
 * 2026-05-24-intelligence-reindex-controls §3.3.
 *
 * Covers:
 *  1. Initial render fetches /files/{id}/index-details and renders
 *     one row per task (metadata / clip / whisper / text).
 *  2. Each row carries a "Regenerate" button. Clicking it confirms
 *     via the ConfirmDialog and then POSTs to /files/{id}/reindex
 *     with body { tasks: [<that task>] }.
 *  3. mime_type-driven display: clip only renders for image/video
 *     files; whisper only renders for transcribable types
 *     (audio/video).
 *  4. Cancelling the confirm dialog does NOT issue the POST.
 *  5. No emoji.
 *
 * RED-phase: the component does not exist yet. The dynamic import in
 * each test fails, which is the RED signal for this TDD step.
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
  getIndexDetails: vi.fn(),
  reindexFile: vi.fn(),
}));

// The host's ConfirmDialog ships a known shape — keep the stub small so
// the tests can drive Confirm / Cancel directly.
vi.mock("@/components/ConfirmDialog", () => ({
  ConfirmDialog: ({
    open,
    onConfirm,
    onCancel,
  }: {
    open: boolean;
    title?: string;
    message?: string;
    onConfirm: () => void;
    onCancel: () => void;
  }) =>
    open ? (
      <div role="dialog" data-testid="confirm-dialog">
        <button onClick={onConfirm}>dialog-confirm</button>
        <button onClick={onCancel}>dialog-cancel</button>
      </div>
    ) : null,
}));

import IndexDetailsSection from "@/addons/intelligence/IndexDetailsSection";
import { getIndexDetails, reindexFile } from "@/addons/intelligence/api";

const mockedGet = getIndexDetails as unknown as ReturnType<typeof vi.fn>;
const mockedReindex = reindexFile as unknown as ReturnType<typeof vi.fn>;

function defaultDetails(overrides: Record<string, unknown> = {}) {
  return {
    file_id: "abc12345",
    drive: "drive1",
    filename: "movie.mp4",
    status: {
      metadata: true,
      clip: true,
      whisper: true,
      text: false,
    },
    indexed_at: "2026-05-23T10:00:00Z",
    embeddings: {},
    provider_stats: {},
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockedGet.mockResolvedValue(defaultDetails());
  mockedReindex.mockResolvedValue({ status: "accepted" });
});

afterEach(() => {
  vi.restoreAllMocks();
});

function renderSection(
  props: { fileId?: string; drive?: string; mimeType?: string; fileType?: string } = {},
) {
  return render(
    <IndexDetailsSection
      fileId={props.fileId ?? "abc12345"}
      drive={props.drive ?? "drive1"}
      mimeType={props.mimeType ?? "video/mp4"}
      fileType={props.fileType ?? "video"}
    />,
  );
}

describe("IndexDetailsSection — render", () => {
  it("fetches /files/{id}/index-details on mount", async () => {
    renderSection();
    await waitFor(() => expect(mockedGet).toHaveBeenCalledTimes(1));
    expect(mockedGet).toHaveBeenCalledWith("abc12345", "drive1");
  });

  it("renders one row per applicable task for a video file", async () => {
    renderSection({ mimeType: "video/mp4", fileType: "video" });
    await waitFor(() => expect(mockedGet).toHaveBeenCalled());

    // Video supports every task. Locale-tolerant matching against
    // either the localised label or the raw next-intl key path.
    await waitFor(() => {
      expect(
        screen.getByText(
          /metadata|Metadata extraction|semanticSearch\.tasks\.metadata/i,
        ),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/CLIP|Image analysis|semanticSearch\.tasks\.clip/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Whisper|Transcription|semanticSearch\.tasks\.whisper/i,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Text extraction|semanticSearch\.tasks\.text/i),
    ).toBeInTheDocument();
  });

  it("hides whisper for image-only files (no audio track)", async () => {
    renderSection({ mimeType: "image/png", fileType: "image" });
    await waitFor(() => expect(mockedGet).toHaveBeenCalled());

    // Image rows: metadata + clip exist, whisper should be hidden
    // because the file has no audio.
    expect(
      screen.queryByText(
        /Whisper|Transcription|semanticSearch\.tasks\.whisper/i,
      ),
    ).toBeNull();
  });

  it("hides clip for plain text files (no visual content)", async () => {
    renderSection({ mimeType: "text/markdown", fileType: "text" });
    await waitFor(() => expect(mockedGet).toHaveBeenCalled());

    expect(
      screen.queryByText(/CLIP|Image analysis|semanticSearch\.tasks\.clip/i),
    ).toBeNull();
  });
});

describe("IndexDetailsSection — regenerate button", () => {
  async function openTaskDialog(taskRegex: RegExp) {
    renderSection();
    await waitFor(() => expect(mockedGet).toHaveBeenCalled());

    // The "Regenerate" button is per-task. Find the row by task name
    // first, then locate the button within it.
    const taskLabel = await screen.findByText(taskRegex);
    const row =
      taskLabel.closest("li, tr, [role='row'], [data-task-row]") ??
      (taskLabel.parentElement as HTMLElement);
    expect(row).not.toBeNull();
    const regenBtn = within(row as HTMLElement).getByRole("button", {
      name: /regenerate|再生成|semanticSearch\.indexDetails\.regenerate/i,
    });
    fireEvent.click(regenBtn);
    return await screen.findByRole("dialog");
  }

  it("opens a confirm dialog for whisper and POSTs reindexFile on confirm", async () => {
    const dialog = await openTaskDialog(
      /Whisper|Transcription|semanticSearch\.tasks\.whisper/i,
    );
    fireEvent.click(within(dialog).getByText("dialog-confirm"));

    await waitFor(() => expect(mockedReindex).toHaveBeenCalled());
    const [fileId, tasks] = mockedReindex.mock.calls[0];
    expect(fileId).toBe("abc12345");
    expect(tasks).toEqual(["whisper"]);
  });

  it("sends task=clip when the CLIP row is regenerated", async () => {
    const dialog = await openTaskDialog(
      /CLIP|Image analysis|semanticSearch\.tasks\.clip/i,
    );
    fireEvent.click(within(dialog).getByText("dialog-confirm"));

    await waitFor(() => expect(mockedReindex).toHaveBeenCalled());
    const [, tasks] = mockedReindex.mock.calls[0];
    expect(tasks).toEqual(["clip"]);
  });

  it("does NOT call reindexFile when the dialog is cancelled", async () => {
    const dialog = await openTaskDialog(
      /Whisper|Transcription|semanticSearch\.tasks\.whisper/i,
    );
    fireEvent.click(within(dialog).getByText("dialog-cancel"));

    // Allow microtasks to drain.
    await new Promise((r) => setTimeout(r, 10));
    expect(mockedReindex).not.toHaveBeenCalled();
  });
});

describe("IndexDetailsSection — UI rules", () => {
  it("renders no emoji", async () => {
    renderSection();
    await waitFor(() => expect(mockedGet).toHaveBeenCalled());

    expect(document.body.textContent ?? "").not.toMatch(
      /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u,
    );
  });
});
