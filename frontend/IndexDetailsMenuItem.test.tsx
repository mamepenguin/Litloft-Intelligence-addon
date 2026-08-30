/**
 * Tests for the Index Details menu entry and its dialog.
 *
 * Originally `IndexDetailsSection.test.tsx` (spec
 * 2026-05-24-intelligence-reindex-controls §3.3); the surface moved into
 * the core `[...]` menu's `file-actions-menu` slot per spec
 * 2026-08-30-file-actions-menu-addon-slot, and these cases moved with it.
 *
 * Covers:
 *  1. The entry renders without fetching; opening it fetches
 *     /files/{id}/index-details and renders one row per task
 *     (metadata / clip / whisper / text).
 *  2. Each row carries a "Regenerate" button. Clicking it confirms
 *     via the ConfirmDialog and then POSTs to /files/{id}/reindex
 *     with body { tasks: [<that task>] }.
 *  3. mime_type-driven display: clip only renders for image/video
 *     files; whisper only renders for transcribable types
 *     (audio/video).
 *  4. Cancelling the confirm dialog does NOT issue the POST.
 *  5. The host-slot contract: the menu is left open while the dialog is
 *     up, and asked to close only once it is dismissed.
 *  6. No emoji.
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
      <div data-testid="confirm-dialog">
        <button onClick={onConfirm}>dialog-confirm</button>
        <button onClick={onCancel}>dialog-cancel</button>
      </div>
    ) : null,
}));

import IndexDetailsMenuItem from "@/addons/intelligence/IndexDetailsMenuItem";
import { ShortcutsProvider } from "@/components/ShortcutsProvider";
import { getIndexDetails, reindexFile } from "@/addons/intelligence/api";

const mockedGet = getIndexDetails as unknown as ReturnType<typeof vi.fn>;
const mockedReindex = reindexFile as unknown as ReturnType<typeof vi.fn>;

const ENTRY_LABEL =
  /index details|インデックス詳細|semanticSearch\.indexDetails\.title/i;

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

interface RenderOpts {
  fileId?: string;
  drive?: string;
  mimeType?: string;
  fileType?: string;
  onRequestClose?: () => void;
  onDialogOpenChange?: (open: boolean) => void;
}

// Wrapped in the real ShortcutsProvider: the dialog registers Escape
// through the shortcut stack rather than binding window directly, so a
// bare render would exercise the no-op default context instead of the
// dispatch path that actually ships.
function renderEntry(props: RenderOpts = {}) {
  return render(
    <ShortcutsProvider>
      <IndexDetailsMenuItem
        fileId={props.fileId ?? "abc12345"}
        drive={props.drive ?? "drive1"}
        mimeType={props.mimeType ?? "video/mp4"}
        fileType={props.fileType ?? "video"}
        onRequestClose={props.onRequestClose}
        onDialogOpenChange={props.onDialogOpenChange}
      />
    </ShortcutsProvider>,
  );
}

/** Render the entry and open its dialog, as a user would. */
async function openDialog(props: RenderOpts = {}) {
  const result = renderEntry(props);
  fireEvent.click(screen.getByRole("menuitem", { name: ENTRY_LABEL }));
  await waitFor(() => expect(mockedGet).toHaveBeenCalled());
  return result;
}

describe("IndexDetailsMenuItem — the entry", () => {
  it("renders a menu row and fetches nothing until it is opened", () => {
    renderEntry();

    expect(
      screen.getByRole("menuitem", { name: ENTRY_LABEL }),
    ).toBeInTheDocument();
    // Every file detail page mounts this entry; only the ones where
    // someone actually asks should cost a request.
    expect(mockedGet).not.toHaveBeenCalled();
  });

  it("fetches /files/{id}/index-details when opened", async () => {
    await openDialog();

    expect(mockedGet).toHaveBeenCalledTimes(1);
    expect(mockedGet).toHaveBeenCalledWith("abc12345", "drive1");
  });

  it("tells the host a dialog is open, and asks it to close only on dismiss", async () => {
    // The host must not close its menu while the dialog is up: that
    // would unmount this component and take the dialog with it.
    const onRequestClose = vi.fn();
    const onDialogOpenChange = vi.fn();
    await openDialog({ onRequestClose, onDialogOpenChange });

    expect(onDialogOpenChange).toHaveBeenCalledWith(true);
    expect(onRequestClose).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /close|閉じる/i }));

    expect(onDialogOpenChange).toHaveBeenLastCalledWith(false);
    expect(onRequestClose).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes on Escape", async () => {
    const onRequestClose = vi.fn();
    await openDialog({ onRequestClose });

    fireEvent.keyDown(document, { key: "Escape" });

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(onRequestClose).toHaveBeenCalledTimes(1);
  });

  it("reports the dialog closed on the Escape path too", async () => {
    const onDialogOpenChange = vi.fn();
    await openDialog({ onDialogOpenChange });

    fireEvent.keyDown(document, { key: "Escape" });

    // Skipping this would leave the host's guard latched on.
    expect(onDialogOpenChange).toHaveBeenLastCalledWith(false);
  });

  it("closes on a backdrop click", async () => {
    const onRequestClose = vi.fn();
    const { container } = await openDialog({ onRequestClose });

    // The backdrop lives in the portal, not in the render container.
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    const backdrop = document.body.querySelector(".fixed.inset-0 > .absolute");
    fireEvent.click(backdrop as Element);

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(onRequestClose).toHaveBeenCalledTimes(1);
  });

  it("refetches on reopen and ignores the response the first open left in flight", async () => {
    // Closing invalidates whatever is in flight. Without that, a slow
    // first response lands after the second and silently overwrites it —
    // and would undo an optimistic Regenerate flip made in between.
    let resolveFirst!: (v: unknown) => void;
    mockedGet
      .mockImplementationOnce(
        () => new Promise((resolve) => {
          resolveFirst = resolve;
        }),
      )
      .mockResolvedValueOnce(
        defaultDetails({ status: { metadata: false, clip: false, whisper: false, text: false } }),
      );

    renderEntry();
    const entry = screen.getByRole("menuitem", { name: ENTRY_LABEL });

    fireEvent.click(entry);
    await waitFor(() => expect(mockedGet).toHaveBeenCalledTimes(1));
    fireEvent.keyDown(document, { key: "Escape" });

    fireEvent.click(entry);
    await waitFor(() => expect(mockedGet).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(screen.getAllByText(/Pending|未処理|statusPending/i).length).toBeGreaterThan(0),
    );

    // The first open's response arrives last and must be discarded.
    resolveFirst(defaultDetails());
    await new Promise((r) => setTimeout(r, 10));

    expect(screen.queryByText(/Indexed|インデックス済|statusDone/i)).toBeNull();
  });

  it("leaves Escape to the confirmation while it is up", async () => {
    await openDialog();
    const row = await findTaskRow(
      /Whisper|Transcription|semanticSearch\.tasks\.whisper/i,
    );
    fireEvent.click(
      within(row).getByRole("button", {
        name: /regenerate|再生成|semanticSearch\.indexDetails\.regenerate/i,
      }),
    );

    fireEvent.keyDown(document, { key: "Escape" });

    // One press must not dismiss both layers.
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });
});

describe("IndexDetailsMenuItem — rows", () => {
  it("renders one row per applicable task for a video file", async () => {
    await openDialog({ mimeType: "video/mp4", fileType: "video" });

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
    await openDialog({ mimeType: "image/png", fileType: "image" });

    // Image rows: metadata + clip exist, whisper should be hidden
    // because the file has no audio.
    expect(
      screen.queryByText(
        /Whisper|Transcription|semanticSearch\.tasks\.whisper/i,
      ),
    ).toBeNull();
  });

  it("hides clip for plain text files (no visual content)", async () => {
    await openDialog({ mimeType: "text/markdown", fileType: "text" });

    expect(
      screen.queryByText(/CLIP|Image analysis|semanticSearch\.tasks\.clip/i),
    ).toBeNull();
  });

  it("hides whisper and text for .loft files (remote URL wrappers)", async () => {
    await openDialog({
      mimeType: "application/vnd.litloft.loft+json",
      fileType: "video",
    });

    expect(
      screen.queryByText(
        /Whisper|Transcription|semanticSearch\.tasks\.whisper/i,
      ),
    ).toBeNull();
    expect(
      screen.queryByText(/Text extraction|semanticSearch\.tasks\.text/i),
    ).toBeNull();
  });

  it("still shows metadata and clip for .loft files", async () => {
    await openDialog({
      mimeType: "application/vnd.litloft.loft+json",
      fileType: "video",
    });

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
  });

  it("shows metadata and text for an HTML file", async () => {
    // text/html is in the indexer's TEXT_MIMES (spec
    // 2026-05-12-html-indexing), so these rows describe real state.
    await openDialog({ mimeType: "text/html", fileType: "text" });

    await waitFor(() => {
      expect(
        screen.getByText(
          /metadata|Metadata extraction|semanticSearch\.tasks\.metadata/i,
        ),
      ).toBeInTheDocument();
    });
    expect(
      screen.getByText(/Text extraction|semanticSearch\.tasks\.text/i),
    ).toBeInTheDocument();
  });
});

/** Locate a task's row by its label. */
async function findTaskRow(taskRegex: RegExp): Promise<HTMLElement> {
  const taskLabel = await screen.findByText(taskRegex);
  const row =
    taskLabel.closest("li, tr, [role='row'], [data-task-row]") ??
    (taskLabel.parentElement as HTMLElement);
  expect(row).not.toBeNull();
  return row as HTMLElement;
}

describe("IndexDetailsMenuItem — regenerate button", () => {
  async function openTaskConfirm(taskRegex: RegExp) {
    await openDialog();

    // The "Regenerate" button is per-task. Find the row by task name
    // first, then locate the button within it.
    const row = await findTaskRow(taskRegex);
    fireEvent.click(
      within(row).getByRole("button", {
        name: /regenerate|再生成|semanticSearch\.indexDetails\.regenerate/i,
      }),
    );
    return await screen.findByTestId("confirm-dialog");
  }

  it("opens a confirm dialog for whisper and POSTs reindexFile on confirm", async () => {
    const confirm = await openTaskConfirm(
      /Whisper|Transcription|semanticSearch\.tasks\.whisper/i,
    );
    fireEvent.click(within(confirm).getByText("dialog-confirm"));

    await waitFor(() => expect(mockedReindex).toHaveBeenCalled());
    const [fileId, tasks] = mockedReindex.mock.calls[0];
    expect(fileId).toBe("abc12345");
    expect(tasks).toEqual(["whisper"]);
  });

  it("sends task=clip when the CLIP row is regenerated", async () => {
    const confirm = await openTaskConfirm(
      /CLIP|Image analysis|semanticSearch\.tasks\.clip/i,
    );
    fireEvent.click(within(confirm).getByText("dialog-confirm"));

    await waitFor(() => expect(mockedReindex).toHaveBeenCalled());
    const [, tasks] = mockedReindex.mock.calls[0];
    expect(tasks).toEqual(["clip"]);
  });

  it("does NOT call reindexFile when the dialog is cancelled", async () => {
    const confirm = await openTaskConfirm(
      /Whisper|Transcription|semanticSearch\.tasks\.whisper/i,
    );
    fireEvent.click(within(confirm).getByText("dialog-cancel"));

    // Allow microtasks to drain.
    await new Promise((r) => setTimeout(r, 10));
    expect(mockedReindex).not.toHaveBeenCalled();
  });
});

describe("IndexDetailsMenuItem — UI rules", () => {
  it("renders no emoji", async () => {
    await openDialog();

    expect(document.body.textContent ?? "").not.toMatch(
      /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u,
    );
  });
});
