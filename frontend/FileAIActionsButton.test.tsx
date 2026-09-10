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

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
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

afterEach(() => {
  vi.restoreAllMocks();
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

  /**
   * `docs/ADDON-DEVELOPMENT.md` puts the coarse-pointer floor on the entry
   * rather than on the row it lands in: `.file-action-row-touch > *` grows
   * the wrapper, and a wrapper at 44 around a button at 32 is still a 32px
   * target.
   *
   * **This pins a class, not a geometry.** jsdom lays nothing out, so no
   * test in this repository can assert the 44px outcome; the arithmetic
   * that says the class is the right one lives beside it in the component.
   * `min-h`, not `h-11`, because the trigger carries a label — and no
   * `min-w`, because padding and two icons put the width past the floor
   * before the label is drawn.
   */
  it("clears the touch floor in the axis that needs it", () => {
    renderWithStack(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );

    const trigger = screen.getByRole("button", { name: "AI" });
    expect(trigger).toHaveClass("pointer-coarse:min-h-11");
    // Not a fixed box: `h-11` would clip the label in a locale whose word
    // is longer than "AI".
    expect(trigger.className).not.toMatch(/pointer-coarse:h-11\b/);
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

  it("closes on a press outside it, without pressing what is under it", () => {
    // Core's `DismissScrim` rather than a hand-written scrim: the press
    // outside closes the menu and the `click` that press produces is
    // swallowed before the page sees it. This file was the last popup in
    // the tree still dismissing on its own scrim's click, which only
    // works while that scrim is the box a tap reaches.
    //
    // jsdom hit-tests nothing, so this is a claim about event order, not
    // about which element a real tap lands on.
    const page = document.createElement("button");
    let pressed = 0;
    page.addEventListener("click", () => {
      pressed += 1;
    });
    document.body.appendChild(page);

    renderWithStack(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    const trigger = screen.getByRole("button", { name: "AI" });
    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");

    fireEvent.pointerDown(page);
    fireEvent.click(page);

    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(pressed).toBe(0);
    page.remove();
  });

  /**
   * Boxes for the direction decision, which jsdom lays out as zeros.
   *
   * The wrapper is the one element with `relative` in this tree, and the
   * menu is the one with `role="menu"`; everything else keeps jsdom's
   * own answer, which is what makes the ancestor walk fall through to
   * the viewport — the resting strip's case, since the strip is `fixed`
   * and nothing above it clips.
   */
  function withBoxes(
    trigger: { top: number; bottom: number; left?: number; right?: number },
    menuHeight: number,
    viewportHeight: number,
  ) {
    const original = Element.prototype.getBoundingClientRect;
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function (this: Element) {
        if (this.classList.contains("relative")) {
          // `left`/`right` default to a trigger hard against the left
          // edge, which is where every case written before the
          // horizontal axis existed put it. A case that cares states
          // them.
          return {
            ...trigger,
            left: trigger.left ?? 0,
            right: trigger.right ?? 100,
            height: trigger.bottom - trigger.top,
          } as DOMRect;
        }
        if (this.getAttribute("role") === "menu") {
          return { height: menuHeight, top: 0, bottom: menuHeight } as DOMRect;
        }
        // An ancestor states its own box, so a case that puts one in the
        // chain does not also have to be handed to the mock separately.
        const box = this.getAttribute("data-box");
        if (box) return JSON.parse(box) as DOMRect;
        return original.call(this);
      },
    );
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      value: viewportHeight,
    });
  }

  function openMenuWith(
    trigger: { top: number; bottom: number; left?: number; right?: number },
    menuHeight: number,
    viewportHeight: number,
  ): HTMLElement {
    withBoxes(trigger, menuHeight, viewportHeight);
    renderWithStack(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    fireEvent.click(screen.getByRole("button", { name: "AI" }));
    return screen.getByRole("menu");
  }

  /**
   * The horizontal axis, which is the one the row's own placement decides.
   *
   * `FileDetailContainer` builds the sheet's resting strip as the file's
   * name with `flex-1` followed by a `flex-shrink-0` action row, so this
   * button is drawn near the *right* edge at every phone width — its
   * distance from that edge is set by the row's furniture, not by the
   * viewport. A menu that always hung rightward from the trigger's left
   * edge therefore ran off the screen by the same amount whatever the
   * phone: measured in a real browser at 393x727, 134px of a 240px menu.
   *
   * jsdom lays nothing out, so the boxes are stated and what is asserted
   * is the decision they produce. The geometry is core's
   * `e2e-components`.
   */
  it("hangs leftward from a trigger drawn at the right of the row", () => {
    // 393px phone, the trigger where the strip's furniture puts it.
    // 328 − 240 = 88, which clears the frame's left edge, so the menu
    // takes the room to its left and stays on screen.
    const menu = openMenuWith(
      { top: 681, bottom: 718, left: 290, right: 328 },
      82,
      727,
    );

    expect(menu.className).toContain("right-0");
    expect(menu.className).not.toContain("left-0");
  });

  it("hangs rightward when there is not room to its left", () => {
    // The other side of the same rule, and the case the old constant was
    // right about: a trigger near the left edge has nowhere to hang
    // leftward, so it goes the other way. 100 − 240 = −140, past the
    // frame's left edge.
    const menu = openMenuWith(
      { top: 200, bottom: 236, left: 8, right: 100 },
      82,
      727,
    );

    expect(menu.className).toContain("left-0");
    expect(menu.className).not.toContain("right-0");
  });

  it("decides the two axes independently", () => {
    // The pair the strip actually produces: no room below *and* no room
    // to the right. A single flag would have to pick one of them.
    const menu = openMenuWith(
      { top: 681, bottom: 718, left: 290, right: 328 },
      82,
      727,
    );

    expect(menu.className).toContain("bottom-full");
    expect(menu.className).toContain("right-0");
  });

  it("counts the gap when it asks whether the menu fits below", () => {
    // `MENU_GAP_PX` is added to the height before the comparison, so a
    // menu that fits below by less than the gap has to flip. Without the
    // term the boxes below keep the menu downward and it is drawn into
    // the 4px that separates it from its trigger.
    //
    // 82 + 4 = 86 against 84 below and 500 above: down is short by two
    // pixels, and only the gap says so.
    const menu = openMenuWith({ top: 500, bottom: 536 }, 82, 620);

    expect(menu.className).toContain("bottom-full");
  });

  it("keeps an absolute detour out of the frame", () => {
    // The walk skips an `overflow` ancestor while it is outside the
    // menu's containing-block chain — the stretch an `absolute` box takes
    // it out of — and resumes at the next positioned one. Core's
    // `FileActions` carries the same guard and this is the case that
    // separates the two behaviours: without it the scroller below is
    // taken as the frame, which has no room, and the menu flips.
    //
    // The scroller states its own box through `data-box`, and the
    // `absolute` box between it and the wrapper is what the guard is
    // about. Its numbers are chosen so the two answers differ: inside
    // that scroller the trigger has 4px below it and 100 above, so a walk
    // that stopped there would flip; the viewport has 764 below, so the
    // walk that skips it does not. A scroller with more room below than
    // above would give the same answer either way and measure nothing.
    withBoxes({ top: 100, bottom: 136 }, 82, 900);
    render(
      <ShortcutsProvider>
        {/* The scroller the detour has to skip. It is short and sits
            above the trigger, so a walk that stopped here would find no
            room below and flip. */}
        <div style={{ overflowY: "auto" }} data-box='{"top":0,"bottom":140,"left":0,"right":300}'>
          {/* The `absolute` box that takes the chain out of the
              scroller's containing block. */}
          <div style={{ position: "absolute" }}>
            <Offering
              fileId="f1"
              kind="summary"
              labelKey="summaryGenerate"
              active
            />
            <FileAIActionsButton fileId="f1" />
          </div>
        </div>
      </ShortcutsProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "AI" }));

    // The frame falls through to the viewport, which has 764 below the
    // trigger, so the menu keeps its downward direction.
    expect(screen.getByRole("menu").className).toContain("top-full");
    expect(screen.getByRole("menu").className).not.toContain("bottom-full");
  });

  it("hangs upward when the row it is in sits on the bottom edge", () => {
    // The state the file detail is in when it opens: the sheet is
    // collapsed and this row is drawn in its 56px resting strip, which
    // core positions `fixed bottom-0`. There is nothing below the trigger
    // there — measured on the running stack at 500x639, the trigger's
    // bottom was 629.5 against a 639 viewport, and a menu hanging down
    // showed 5.5px of 82.
    //
    // jsdom lays nothing out, so the boxes are stated: what is asserted
    // is the decision the numbers produce, not a geometry this
    // environment could measure. Core's `e2e-components` draws the same
    // strip in a real browser.
    const menu = openMenuWith({ top: 593.5, bottom: 629.5 }, 82, 639);

    expect(menu.className).toContain("bottom-full");
    expect(menu.className).not.toContain("top-full");
  });

  it("hangs downward when there is room below it", () => {
    // The other side, and the direction the menu reads as everywhere
    // else: the expanded sheet and the desktop rail both leave room.
    const menu = openMenuWith({ top: 200, bottom: 240 }, 82, 800);

    expect(menu.className).toContain("top-full");
    expect(menu.className).not.toContain("bottom-full");
  });

  it("keeps hanging downward when neither side has room", () => {
    // A trigger with nothing either way keeps the direction it has
    // everywhere else rather than flipping into an equally bad one —
    // `spaceAbove > spaceBelow` is what makes the flip an improvement
    // rather than a coin toss.
    const menu = openMenuWith({ top: 10, bottom: 50 }, 400, 60);

    expect(menu.className).toContain("top-full");
  });

  it("stops the walk at the strip, not at the scroller the strip is drawn over", () => {
    // The `fixed` stop, which is the half of the walk the boxes above do
    // not reach: with no ancestors between the wrapper and the document,
    // the frame falls through to the viewport either way.
    //
    // Here the strip is drawn over a scroller — the page behind it — and
    // that scroller has room below the trigger. A walk that did not stop
    // at the strip would take the scroller's box, conclude there is room,
    // and hang the menu off the bottom of the screen again. A `fixed` box
    // is laid out against the viewport, so nothing above it clips what is
    // inside it.
    withBoxes({ top: 593.5, bottom: 629.5 }, 82, 639);
    render(
      <ShortcutsProvider>
        <div
          // `overflowY` rather than the `overflow` shorthand: jsdom does
          // not expand the shorthand into the longhands the walk reads.
          style={{ overflowY: "auto" }}
          // Room below the trigger, which is what makes this case
          // discriminate: the frame the walk picks decides the answer.
          data-box='{"top":0,"bottom":2000}'
        >
          <div style={{ position: "fixed", bottom: "0px" }}>
            <Offering
              fileId="f1"
              kind="summary"
              labelKey="summaryGenerate"
              active
            />
            <FileAIActionsButton fileId="f1" />
          </div>
        </div>
      </ShortcutsProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "AI" }));

    expect(screen.getByRole("menu").className).toContain("bottom-full");
  });

  it("anchors the menu to the trigger rather than to the screen", () => {
    // The Bottom Sheet is why. This row is drawn inside `Drawer.Content`,
    // which carries a transform, so a `fixed` box there resolves against
    // the drawer instead of the viewport — and the drawer hangs below the
    // fold by however far vaul has translated it. The menu used to be
    // `fixed inset-x-2 bottom-4` below `sm` and landed off the bottom of
    // the screen; `absolute` resolves against the wrapper, which is on
    // screen wherever the sheet is.
    //
    // A spelling check, and it says so: jsdom lays nothing out, so what
    // is asserted is the class list. The geometry is measured in core's
    // `e2e-components/popup-dismiss.spec.ts`, in a real browser, inside a
    // real sheet.
    renderWithStack(
      <>
        <Offering fileId="f1" kind="summary" labelKey="summaryGenerate" active />
        <FileAIActionsButton fileId="f1" />
      </>,
    );
    fireEvent.click(screen.getByRole("button", { name: "AI" }));

    const menu = screen.getByRole("menu");
    expect(menu.className.split(/\s+/)).toContain("absolute");
    expect(menu.className).not.toMatch(/(^|\s|:)fixed(\s|$)/);
    expect(menu.className).not.toMatch(/bottom-4/);
    // One direction or the other, never both and never neither: the two
    // cases above say which, this says the class list can only be in one
    // of the two states they describe.
    expect(
      [menu.className.includes("top-full"), menu.className.includes("bottom-full")],
    ).toEqual([true, false]);
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

