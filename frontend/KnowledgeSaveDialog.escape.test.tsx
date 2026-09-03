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
