/**
 * Tests for the "AI" entry in the file detail action row and the
 * registry behind it.
 *
 * The contract the sections rely on: an offer is listed while it is
 * active, disappears the moment its section withdraws it, and the menu
 * itself vanishes when nothing is left to offer — so the row is
 * untouched on a file with nothing to generate, and on an install with
 * no intelligence at all.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { act, render, screen, fireEvent } from "@testing-library/react";

import FileAIActionsButton from "@/addons/intelligence/FileAIActionsButton";
import { ShortcutsProvider } from "@/components/ShortcutsProvider";
import {
  resetFileAiActions,
  useOfferFileAiAction,
  type FileAiActionKind,
} from "@/addons/intelligence/fileAiActions";

/** Stands in for a section: offers one action while `active` holds. */
function Offering({
  fileId,
  kind,
  labelKey,
  active,
  busy,
  onRun,
}: {
  fileId: string;
  kind: FileAiActionKind;
  labelKey: string;
  active: boolean;
  busy?: boolean;
  onRun?: () => void;
}) {
  useOfferFileAiAction({
    fileId,
    kind,
    labelKey,
    active,
    busy,
    run: () => onRun?.(),
  });
  return null;
}

beforeEach(() => {
  resetFileAiActions();
});

/**
 * Escape reaches the menu through the shortcut stack, and `AppShell`
 * mounts the provider around every route — a bare render is a tree the
 * browser never has.
 */
function renderWithStack(ui: React.ReactElement) {
  return render(<ShortcutsProvider>{ui}</ShortcutsProvider>);
}

describe("FileAIActionsButton", () => {
  it("renders nothing when no section is offering anything", () => {
    const { container } = render(<FileAIActionsButton fileId="f1" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("appears once a section offers, and lists what it offered", () => {
    renderWithStack(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );

    fireEvent.click(screen.getByRole("button", { name: "AI" }));
    expect(
      screen.getByRole("menuitem", { name: /Create AI summary/ }),
    ).toBeInTheDocument();
  });

  it("runs the offering section's own callback", () => {
    let ran = 0;
    renderWithStack(
      <>
        <Offering
          fileId="f1"
          kind="tags"
          labelKey="generateTags"
          active
          onRun={() => {
            ran += 1;
          }}
        />
        <FileAIActionsButton fileId="f1" />
      </>,
    );

    fireEvent.click(screen.getByRole("button", { name: "AI" }));
    fireEvent.click(screen.getByRole("menuitem", { name: /Create AI tag candidates/ }));
    expect(ran).toBe(1);
    // Running an action closes the menu; the section takes over from here.
    expect(screen.queryByRole("menuitem")).toBeNull();
  });

  it("keeps a fixed order regardless of which section registers first", () => {
    renderWithStack(
      <>
        <Offering fileId="f1" kind="chapters" labelKey="generateChapters" active />
        <Offering fileId="f1" kind="tags" labelKey="generateTags" active />
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );

    fireEvent.click(screen.getByRole("button", { name: "AI" }));
    const labels = screen.getAllByRole("menuitem").map((el) => el.textContent);
    expect(labels).toEqual([
      "Create AI tag candidates",
      "Create AI summary",
      "Create AI chapter candidates",
    ]);
  });

  it("drops an entry when its section withdraws the offer", () => {
    const { rerender } = render(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <Offering fileId="f1" kind="tags" labelKey="generateTags" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    fireEvent.click(screen.getByRole("button", { name: "AI" }));
    expect(screen.getAllByRole("menuitem")).toHaveLength(2);

    // The summary landed: that section now heads its own content.
    rerender(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active={false} />
        <Offering fileId="f1" kind="tags" labelKey="generateTags" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    expect(screen.getAllByRole("menuitem")).toHaveLength(1);
    expect(screen.getByRole("menuitem").textContent).toBe("Create AI tag candidates");
  });

  it("disappears when the last offer is withdrawn", () => {
    const { rerender } = render(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    expect(screen.getByRole("button", { name: "AI" })).toBeInTheDocument();

    rerender(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active={false} />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    expect(screen.queryByRole("button", { name: "AI" })).toBeNull();
  });

  it("survives the same section being mounted twice for one file", () => {
    // The file detail page builds the inspector and the mobile bottom
    // sheet from the same subtree and keeps them exclusive by
    // convention. If that convention ever slips, one unmount must not
    // take an offer away from the copy still on screen.
    const { rerender } = render(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    fireEvent.click(screen.getByRole("button", { name: "AI" }));
    // One row for one action, however many components offered it.
    expect(screen.getAllByRole("menuitem")).toHaveLength(1);

    rerender(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    expect(screen.getByRole("button", { name: "AI" })).toBeInTheDocument();
  });

  it("keeps one file's offers out of another file's menu", () => {
    renderWithStack(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f2" />
      </>,
    );
    expect(screen.queryByRole("button", { name: "AI" })).toBeNull();
  });

  it("disables an entry whose run is already in flight", () => {
    renderWithStack(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active busy />
        <FileAIActionsButton fileId="f1" />
      </>,
    );

    fireEvent.click(screen.getByRole("button", { name: "AI" }));
    expect(screen.getByRole("menuitem")).toBeDisabled();
  });

  it("closes on Escape", () => {
    renderWithStack(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );

    const trigger = screen.getByRole("button", { name: "AI" });
    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
  });

  it("uses no emoji", () => {
    renderWithStack(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    fireEvent.click(screen.getByRole("button", { name: "AI" }));
    expect(document.body.textContent ?? "").not.toMatch(
      /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u,
    );
  });
});

/**
 * The touch floor (UI redesign Phase 3, C2b).
 *
 * 00-basis's mobile sizing rule puts the floor at 44px, and this trigger
 * renders 32px tall (`py-1.5` around `text-sm`). It reaches the floor by
 * growing its own box on a coarse pointer rather than by hanging a
 * pseudo-element over its neighbours — `Button`'s recipe for that is written
 * for icon-only controls, where the 32px square is deliberate and the
 * arithmetic depends on it. This one carries a label.
 */
describe("FileAIActionsButton — touch target", () => {
  it("reaches the 44px floor on a coarse pointer", () => {
    // The trigger only exists once a section offers something, so the
    // offering is the setup rather than decoration — without it this would
    // measure an empty container.
    renderWithStack(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    const trigger = screen.getByRole("button", { name: "AI" });
    const tokens = trigger.className.split(/\s+/);
    // `min-h-11` is 2.75rem = 44px, and it is gated on `pointer-coarse` so a
    // mouse still gets the 32px control. Both halves matter: the bare
    // `min-h-11` would grow the button everywhere, and the bare
    // `pointer-coarse:` prefix on something else would not grow it at all.
    expect(tokens).toContain("pointer-coarse:min-h-11");
    expect(tokens).not.toContain("min-h-11");
  });
});
