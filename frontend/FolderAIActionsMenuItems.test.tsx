/**
 * The folder AI actions, as rows of the `Add` menu.
 *
 * They were a labelled `AI ▾` button on the folder toolbar with a dropdown
 * of its own. 案 2 leaves that bar four exposed controls plus a conditional
 * `Play`; an addon's own dropdown is a fifth, and `folder-actions-menu` is
 * the contract for putting it inside the host's menu instead.
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
  it("draws menu rows, not a control with a menu of its own", () => {
    // The whole point of the move. A trigger here would put a fifth
    // control on the toolbar and open a second menu inside the host's.
    const { container } = render(<FolderAIActionsMenuItems {...PROPS} />);
    const rows = [...container.querySelectorAll('[role="menuitem"]')];
    expect(rows.map((r) => r.textContent)).toEqual([
      "generateFolderTags",
      "generateFolderSummaries",
      "visionFolderButton",
    ]);
    expect(container.querySelectorAll("[aria-haspopup]")).toHaveLength(0);
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

  it("asks the host to close its menu, and does not close it itself", async () => {
    // The host owns the menu. None of these three opens a dialog that
    // would unmount with it, so the close is immediate — unlike
    // `IndexDetailsMenuItem`, which defers it until its dialog is gone.
    const onRequestClose = vi.fn();
    render(<FolderAIActionsMenuItems {...PROPS} onRequestClose={onRequestClose} />);
    fireEvent.click(screen.getByRole("menuitem", { name: "generateFolderTags" }));
    expect(onRequestClose).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(api.batchSuggestedTags).toHaveBeenCalledWith(["a", "b"], "family"));
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
  });
});
