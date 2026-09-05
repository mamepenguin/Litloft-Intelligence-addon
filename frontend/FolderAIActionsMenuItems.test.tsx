/**
 * The folder AI actions, as rows of the `Add` menu.
 *
 * 案 2 leaves the folder toolbar four exposed controls plus a conditional
 * `Play`; an addon's own dropdown would be a fifth, and
 * `folder-actions-menu` is the contract for putting the rows inside the
 * host's menu instead.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

const { toast, api } = vi.hoisted(() => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  api: {
    batchSuggestedTags: vi.fn(),
    batchSummaries: vi.fn(),
    generateFolderVisualDescription: vi.fn(),
  },
}));
vi.mock("@/components/ToastProvider", () => ({ useToast: () => toast }));
vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));
vi.mock("@/addons/intelligence/api", () => api);

import FolderAIActionsMenuItems from "@/addons/intelligence/FolderAIActionsMenuItems";

const PROPS = { fileIds: ["a", "b"], drive: "family" };

beforeEach(() => {
  vi.clearAllMocks();
  api.batchSuggestedTags.mockResolvedValue({ queued: 2, skipped: 0 });
  api.batchSummaries.mockResolvedValue({ queued: 2, skipped: 0 });
  api.generateFolderVisualDescription.mockResolvedValue({ queued: 2 });
});
afterEach(cleanup);

describe("the folder AI actions in the Add menu", () => {
  it("draws menu rows, and nothing that is not one", () => {
    // The whole point of the move. Asserted as "every button here is a
    // row", not as "there is no [aria-haspopup] and no role=menu": a plain
    // `<button>` added beside the rows satisfies both of those and puts a
    // fifth control back on the toolbar, which is the exact mutation this
    // test is named for.
    const { container } = render(<FolderAIActionsMenuItems {...PROPS} />);
    const buttons = [...container.querySelectorAll("button")];
    expect(buttons.map((b) => b.getAttribute("role"))).toEqual([
      "menuitem",
      "menuitem",
      "menuitem",
    ]);
    expect(buttons.map((b) => b.textContent)).toEqual([
      "generateFolderTags",
      "generateFolderSummaries",
      "visionFolderButton",
    ]);
    expect(container.querySelectorAll('[role="menu"]')).toHaveLength(0);
  });

  it("renders nothing when the folder has no files", () => {
    // `AddButton`'s separator is `empty:hidden`, so drawing nothing is
    // what takes the rule away with the rows.
    const { container } = render(
      <FolderAIActionsMenuItems {...PROPS} fileIds={[]} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it.each([
    ["generateFolderTags"],
    ["generateFolderSummaries"],
    ["visionFolderButton"],
  ])("asks the host to close its menu from %s", async (row) => {
    // Every row, not one of three. The host owns the menu, and none of
    // these opens a dialog that would unmount with it.
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const onRequestClose = vi.fn();
    render(<FolderAIActionsMenuItems {...PROPS} onRequestClose={onRequestClose} />);
    fireEvent.click(screen.getByRole("menuitem", { name: row }));
    await waitFor(() => expect(onRequestClose).toHaveBeenCalledTimes(1));
  });

  it("queues a batch once, even though pressing a row unmounts the rows", async () => {
    // The host closes its menu on click, so anything holding the guard in
    // this component's state is gone before the request returns and a
    // fresh mount starts clean. Two full open/click cycles, one in-flight
    // batch: the second must not queue a second job.
    let resolve!: (v: unknown) => void;
    api.batchSuggestedTags.mockReturnValue(new Promise((r) => { resolve = r; }));

    const first = render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: "generateFolderTags" }));
    expect(api.batchSuggestedTags).toHaveBeenCalledTimes(1);
    first.unmount();

    const second = render(<FolderAIActionsMenuItems {...PROPS} />);
    const row = screen.getByRole("menuitem", { name: "generateFolderTags" });
    // And it says so, rather than silently doing nothing.
    expect(row).toBeDisabled();
    fireEvent.click(row);
    expect(api.batchSuggestedTags).toHaveBeenCalledTimes(1);

    resolve({ queued: 2, skipped: 0 });
    await waitFor(() => expect(toast.success).toHaveBeenCalled());
    second.unmount();

    // And the guard is released, so the next one goes through.
    render(<FolderAIActionsMenuItems {...PROPS} />);
    expect(screen.getByRole("menuitem", { name: "generateFolderTags" })).not.toBeDisabled();
  });

  it("refuses a second claim even from a copy that has not re-rendered", async () => {
    // Not hypothetical: `FolderToolbar` draws its left group once per
    // breakpoint, so two live `AddButton`s — and two live copies of these
    // rows — are in the DOM at the same time. Only the copy that was
    // pressed re-renders, so the other one's row is still enabled, and the
    // `disabled` attribute cannot be what stops it.
    let resolve!: (v: unknown) => void;
    api.batchSuggestedTags.mockReturnValue(new Promise((r) => { resolve = r; }));

    render(
      <>
        <div data-copy="a"><FolderAIActionsMenuItems {...PROPS} /></div>
        <div data-copy="b"><FolderAIActionsMenuItems {...PROPS} /></div>
      </>,
    );
    const rowIn = (copy: string) =>
      document.querySelector(
        `[data-copy="${copy}"] [role="menuitem"]`,
      ) as HTMLElement;

    fireEvent.click(rowIn("a"));
    expect(api.batchSuggestedTags).toHaveBeenCalledTimes(1);

    // The untouched copy still looks pressable, and pressing it must not
    // queue a second batch.
    expect(rowIn("b")).not.toBeDisabled();
    fireEvent.click(rowIn("b"));
    expect(api.batchSuggestedTags).toHaveBeenCalledTimes(1);

    resolve({ queued: 1, skipped: 0 });
    await waitFor(() => expect(toast.success).toHaveBeenCalled());
  });

  it("guards each action and drive separately", async () => {
    let resolve!: (v: unknown) => void;
    api.batchSuggestedTags.mockReturnValue(new Promise((r) => { resolve = r; }));
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: "generateFolderTags" }));
    // A different action on the same drive is untouched...
    expect(screen.getByRole("menuitem", { name: "generateFolderSummaries" })).not.toBeDisabled();
    cleanup();
    // ...and so is the same action on another drive.
    render(<FolderAIActionsMenuItems {...PROPS} drive="work" />);
    expect(screen.getByRole("menuitem", { name: "generateFolderTags" })).not.toBeDisabled();
    resolve({ queued: 1, skipped: 0 });
    await waitFor(() => expect(toast.success).toHaveBeenCalled());
  });

  it.each([
    ["generateFolderTags", "batchSuggestedTags"],
    ["generateFolderSummaries", "batchSummaries"],
  ] as const)("reports what %s queued", async (row, call) => {
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: row }));
    await waitFor(() => expect(api[call]).toHaveBeenCalled());
    expect(toast.success).toHaveBeenCalledTimes(1);
    expect(toast.info).not.toHaveBeenCalled();
  });

  it.each([
    ["generateFolderTags", "batchSuggestedTags"],
    ["generateFolderSummaries", "batchSummaries"],
  ] as const)("says so when %s had nothing to do", async (row, call) => {
    // Queued nothing and skipped some: a different message and a
    // different kind, because it is not a failure.
    api[call].mockResolvedValue({ queued: 0, skipped: 3 });
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: row }));
    await waitFor(() => expect(toast.info).toHaveBeenCalledTimes(1));
    expect(toast.success).not.toHaveBeenCalled();
  });

  it.each([
    ["nothing queued and nothing skipped", { queued: 0, skipped: 0 }],
    ["something queued", { queued: 2, skipped: 1 }],
  ])("counts %s as a result, not as an empty run", async (_case, result) => {
    // `queued: 0, skipped: 0` is not the empty case: the empty message is
    // for a run that had things to do and skipped them all.
    api.batchSuggestedTags.mockResolvedValue(result);
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: "generateFolderTags" }));
    await waitFor(() => expect(toast.success).toHaveBeenCalledTimes(1));
    expect(toast.info).not.toHaveBeenCalled();
  });

  it.each([
    ["batchSuggestedTags", "generateFolderTags"],
    ["batchSummaries", "generateFolderSummaries"],
  ] as const)("reports a failure of %s rather than swallowing it", async (call, row) => {
    api[call].mockRejectedValue(new Error("boom"));
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: row }));
    await waitFor(() => expect(toast.error).toHaveBeenCalledTimes(1));
  });

  it("reports a queued vision batch as a success, not as information", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: "visionFolderButton" }));
    await waitFor(() => expect(toast.success).toHaveBeenCalledTimes(1));
    expect(toast.info).not.toHaveBeenCalled();
    expect(toast.error).not.toHaveBeenCalled();
  });

  it("confirms before spending on vision, and does nothing if declined", () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: "visionFolderButton" }));
    expect(confirm).toHaveBeenCalledTimes(1);
    expect(api.generateFolderVisualDescription).not.toHaveBeenCalled();
    confirm.mockRestore();
  });

  it.each([
    ["too many files", { info: { kind: "too_many_files", max: 50, requested: 900 } }],
    ["anything else", new Error("boom")],
  ])("reports a vision failure as an error toast: %s", async (_case, thrown) => {
    // These two used to be a plain message held for eight seconds rather
    // than five. `toast.error` carries that distinction by kind, which
    // reaches a reader who has already looked away from the menu.
    vi.spyOn(window, "confirm").mockReturnValue(true);
    api.generateFolderVisualDescription.mockRejectedValue(thrown);
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: "visionFolderButton" }));
    await waitFor(() => expect(toast.error).toHaveBeenCalledTimes(1));
    expect(toast.success).not.toHaveBeenCalled();
    // The message, not just the count: collapsing the two branches into
    // one generic error kept the count at one.
    expect(toast.error).toHaveBeenCalledWith(
      thrown instanceof Error ? "visionFolderError" : "visionFolderTooMany",
    );
  });
});
