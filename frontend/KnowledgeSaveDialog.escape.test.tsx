/**
 * Escape closes the save dialog, from inside its own field.
 *
 * It used to bind `window.addEventListener("keydown", …)` directly,
 * which answered every press for as long as the dialog was open — the
 * capture basket underneath it (`knowledge/CaptureBasket.tsx:118`)
 * carries a comment naming this dialog as the reason it has to leave
 * the shortcut stack entirely while the dialog is up. Now both are on
 * the stack and the later push wins, which is the mechanism that
 * comment was working around.
 *
 * The press comes from a focused input on purpose: `ShortcutsProvider`
 * counts that as "editing", and a shortcut registered without
 * `editingOnly: false` does not fire there. Pressing against
 * `document.body` would pass with the flag missing and prove nothing.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

import { ShortcutsProvider } from "@/components/ShortcutsProvider";

vi.mock("@/addons/intelligence/knowledgeBridge", () => ({
  distillToKnowledge: vi.fn(async () => ({ noteFileId: "n1", notePath: "n.md" })),
  getNotesBySourceFile: vi.fn(async () => []),
}));

vi.mock("@/lib/api", () => ({
  getFolders: vi.fn(async () => []),
  getFolderTree: vi.fn(async () => []),
}));

import { KnowledgeSaveDialog } from "@/addons/intelligence/KnowledgeSaveDialog";

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(cleanup);

function withStack(ui: React.ReactElement) {
  return render(<ShortcutsProvider>{ui}</ShortcutsProvider>);
}

describe("KnowledgeSaveDialog", () => {
  it("closes on Escape pressed inside its filename field", async () => {
    const onClose = vi.fn();
    withStack(
      <KnowledgeSaveDialog
        open
        drive="notes"
        fileId="f1"
        content="# Body"
        sourceFilename="talk.mp4"
        onClose={onClose}
        onSaved={vi.fn()}
      />,
    );

    const field = await screen.findByRole("textbox");
    act(() => field.focus());
    expect(document.activeElement).toBe(field);

    fireEvent.keyDown(field, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("does not answer Escape while it is closed", () => {
    const onClose = vi.fn();
    withStack(
      <KnowledgeSaveDialog
        open={false}
        drive="notes"
        fileId="f1"
        content="# Body"
        sourceFilename="talk.mp4"
        onClose={onClose}
        onSaved={vi.fn()}
      />,
    );
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(onClose).not.toHaveBeenCalled();
  });
});

/**
 * One disabled treatment per row (DESIGN.md §6, UI redesign Phase 3, C2b).
 *
 * §6 names the defect precisely: two buttons in one row, driven by the same
 * flag, wearing different disabled treatments — so the moment that flag turns
 * they show two ways of being unavailable. This row is exactly that shape.
 * Both buttons take `submitting`, and before C2b the cancel button faded with
 * `disabled:opacity-50` while the submit button changed colour with
 * `disabled:bg-sand`.
 *
 * Asserted as agreement between the two rather than as a fixed recipe: the
 * defect is the difference, and pinning one spelling would go red on a change
 * to `Button` that kept them consistent.
 */
describe("KnowledgeSaveDialog — the confirm row", () => {
  const disabledTokens = (el: HTMLElement) =>
    el.className
      .split(/\s+/)
      .filter((t) => t.startsWith("disabled:"))
      .sort();

  it("gives both buttons the same disabled treatment", async () => {
    withStack(
      <KnowledgeSaveDialog
        open
        drive="notes"
        fileId="f1"
        content="# Body"
        sourceFilename="talk.mp4"
        onClose={vi.fn()}
        onSaved={vi.fn()}
      />,
    );

    await screen.findByRole("textbox");
    // Exact accessible names, not a regular expression over the text. The
    // dialog also holds a folder picker whose label begins "Save to:", and a
    // loose `/save/i` matched that one first — a button with no disabled
    // treatment at all, which made the comparison below pass against the
    // wrong pair.
    const cancel = screen.getByRole("button", { name: "Cancel" });
    const submit = screen.getByRole("button", { name: "Save" });

    // Non-empty, so the assertion cannot pass by both carrying nothing.
    expect(disabledTokens(submit).length).toBeGreaterThan(0);
    expect(disabledTokens(cancel)).toEqual(disabledTokens(submit));
  });
});
