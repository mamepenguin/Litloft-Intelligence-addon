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
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const { toast, api } = vi.hoisted(() => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  api: {
    batchSuggestedTags: vi.fn(),
    batchSummaries: vi.fn(),
    generateFolderVisualDescription: vi.fn(),
  },
}));
vi.mock("@/components/ToastProvider", () => ({ useToast: () => toast }));
// Values as well as the key: the confirmations below are about the number
// they carry, and a stub that drops it cannot tell 2 files from 619.
vi.mock("next-intl", () => ({
  useTranslations:
    () =>
    (key: string, values?: Record<string, unknown>) =>
      values ? `${key} ${JSON.stringify(values)}` : key,
}));
vi.mock("@/addons/intelligence/api", () => api);

import FolderAIActionsMenuItems from "@/addons/intelligence/FolderAIActionsMenuItems";

const PROPS = { fileIds: ["a", "b"], drive: "family" };

beforeEach(() => {
  vi.clearAllMocks();
  // Every row asks before it spends, so the tests about what gets queued
  // have to answer. `restoreMocks` is off in this project, so a spy
  // installed in one test stays installed for the next; stating the
  // default here is what stops one test arming another.
  vi.spyOn(window, "confirm").mockReturnValue(true);
  api.batchSuggestedTags.mockResolvedValue({ queued: 2, skipped: 0 });
  api.batchSummaries.mockResolvedValue({ queued: 2, skipped: 0 });
  api.generateFolderVisualDescription.mockResolvedValue({ queued: 2 });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

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

  /**
   * F-9. All three of these queue batch LLM jobs, so all three ask first.
   * The question carries the number, because that is the part the reader
   * cannot see: the rows act
   * on what the folder has loaded, which on a long folder is not the
   * folder.
   */
  const ROWS = [
    ["generateFolderTags", "tagsFolderConfirm", () => api.batchSuggestedTags],
    ["generateFolderSummaries", "summariesFolderConfirm", () => api.batchSummaries],
    ["visionFolderButton", "visionFolderConfirm", () => api.generateFolderVisualDescription],
  ] as const;

  it("asks about every row, not two of three", () => {
    // The population, stated separately: three rows are drawn and three
    // are listed here, so "they all ask" cannot be true of a short list.
    const { container } = render(<FolderAIActionsMenuItems {...PROPS} />);
    expect(container.querySelectorAll('[role="menuitem"]')).toHaveLength(ROWS.length);
    expect(ROWS.map(([row]) => row)).toEqual(
      [...container.querySelectorAll('[role="menuitem"]')].map((b) => b.textContent),
    );
  });

  it.each(ROWS)("%s asks first, and says how many files", async (row, key) => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<FolderAIActionsMenuItems {...PROPS} fileIds={["a", "b", "c"]} />);
    fireEvent.click(screen.getByRole("menuitem", { name: row }));

    expect(confirm).toHaveBeenCalledTimes(1);
    const asked = confirm.mock.calls[0][0] as string;
    expect(asked).toContain(key);
    // Three files were handed in, so the question says three.
    expect(asked).toContain('"count":3');
  });

  it.each(ROWS)("%s sends nothing when the reader declines", async (row, _key, api_) => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: row }));
    expect(api_()).not.toHaveBeenCalled();

    // ...and it holds nothing on the way out. A guard claimed before the
    // question would leave this row dead for the rest of the session.
    vi.mocked(window.confirm).mockReturnValue(true);
    cleanup();
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: row }));
    await waitFor(() => expect(api_()).toHaveBeenCalledTimes(1));
  });

  /**
   * The wording is a claim about the catalogue, not about the render: the
   * stub above echoes keys, and the keys still say `Folder` because
   * renaming them moves no text a reader sees. Both locales, because a
   * label fixed in one language and not the other is the failure worth
   * catching.
   */
  describe("the shipped labels", () => {
    const catalogue = (locale: "ja" | "en") =>
      JSON.parse(
        readFileSync(
          resolve(__dirname, `messages/${locale}.json`),
          "utf8",
        ),
      ).file as Record<string, string>;

    // Derived from `ROWS`, not copied: `ROWS` is pinned to the component
    // by the assertions above, and a second hand-written list is pinned to
    // nothing — a row moving to another key would leave these assertions
    // being made about a key nothing draws.
    const LABEL_KEYS = ROWS.map(([label]) => label);
    const CONFIRM_KEYS = ROWS.map(([, confirmKey]) => confirmKey);

    it.each(["ja", "en"] as const)("%s: one row, one question", (locale) => {
      const file = catalogue(locale);
      // Rule (7): every key named here resolves, so "none of them says
      // folder" is not true of a set of missing keys.
      for (const key of [...LABEL_KEYS, ...CONFIRM_KEYS]) {
        expect(typeof file[key]).toBe("string");
      }

      for (const key of LABEL_KEYS) {
        // The rows act on the rows the folder has loaded, so naming the
        // folder claims more than they do...
        expect(file[key]).not.toMatch(/folder|フォルダ/i);
        // ...and the number is not here, where it would change under the
        // reader as they scroll.
        expect(file[key]).not.toMatch(/\d/);
        // The trailing ellipsis is what tells the reader a question is
        // coming, which is the other half of moving the count into it.
        expect(file[key]).toMatch(/…$/);
      }
      for (const key of CONFIRM_KEYS) {
        // It is here instead, where it is read once and acted on.
        expect(file[key]).toContain("{count}");
      }
    });
  });

  it.each([
    ["too many files", { info: { kind: "too_many_files", max: 50, requested: 900 } }],
    ["anything else", new Error("boom")],
  ])("reports a vision failure as an error toast: %s", async (_case, thrown) => {
    // These two used to be a plain message held for eight seconds rather
    // than five. `toast.error` carries that distinction by kind, which
    // reaches a reader who has already looked away from the menu.
    api.generateFolderVisualDescription.mockRejectedValue(thrown);
    render(<FolderAIActionsMenuItems {...PROPS} />);
    fireEvent.click(screen.getByRole("menuitem", { name: "visionFolderButton" }));
    await waitFor(() => expect(toast.error).toHaveBeenCalledTimes(1));
    expect(toast.success).not.toHaveBeenCalled();
    // The message, not just the count: collapsing the two branches into
    // one generic error kept the count at one.
    // `stringContaining`, because the stub appends the values a message
    // was given; the two keys are distinct, which is the distinction this
    // line exists for.
    expect(toast.error).toHaveBeenCalledWith(
      expect.stringContaining(
        thrown instanceof Error ? "visionFolderError" : "visionFolderTooMany",
      ),
    );
  });
});
