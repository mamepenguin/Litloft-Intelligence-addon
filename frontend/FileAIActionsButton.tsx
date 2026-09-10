"use client";

/**
 * "AI" in the file detail action row — the single entry point for
 * everything intelligence can generate about this file.
 *
 * The folder toolbar already had this shape (`FolderAIActionsButton`),
 * and matching it is the point: one verb, one icon, one menu, whether
 * the thing being asked about is a folder or a file.
 *
 * It lists only what is not there yet. A file that already has a
 * summary shows the summary section instead, with its own regenerate
 * control, and the menu drops that entry — so the menu is always an
 * answer to "what is missing", never a duplicate of what is on screen.
 * With nothing missing there is no button at all.
 */

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { useTranslations } from "next-intl";
import {
  BookOpen,
  ChevronDown,
  FileText,
  Image as ImageIcon,
  ListVideo,
  Sparkles,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { useFileAiActions, type FileAiActionKind } from "./fileAiActions";
import { DismissScrim } from "@/components/DismissScrim";
import { useShortcuts } from "@/hooks/useShortcuts";
import { OVERLAY_PRIORITY } from "@/lib/shortcuts";

/** `mt-1` / `mb-1`: the same 4px whichever way the menu hangs. */
const MENU_GAP_PX = 4;

/**
 * The menu's width, and it has to be written here because `min-w-[240px]`
 * on the box below is what sets it: the decision is made before the box
 * has laid out in the direction being decided, so it cannot be read off
 * the element the way the height is.
 *
 * Core's `FileActions` carries the same pair for the same reason with its
 * own number (160). When unit J extracts the walk, the width is the
 * caller's and the rule is the hook's.
 */
const MENU_WIDTH_PX = 240;

/** Same icon the section itself uses, so the menu previews the result. */
const ACTION_ICON: Record<FileAiActionKind, LucideIcon> = {
  tags: Sparkles,
  summary: BookOpen,
  detailedSummary: FileText,
  chapters: ListVideo,
  visualDescription: ImageIcon,
};

interface FileAIActionsButtonProps {
  fileId: string;
}

export default function FileAIActionsButton({ fileId }: FileAIActionsButtonProps) {
  const t = useTranslations("file");
  const actions = useFileAiActions(fileId);
  const [open, setOpen] = useState(false);
  /**
   * Which way the menu hangs, measured rather than declared.
   *
   * `false` until the box is read, which is the direction the menu takes
   * everywhere it has room — and the one a trigger with room for neither
   * keeps.
   */
  const [openUp, setOpenUp] = useState(false);
  const [alignLeft, setAlignLeft] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  // Closing on Escape rather than only on the scrim: the row also lives
  // in a narrow inspector column where the scrim covers the whole page.
  //
  // On the shortcut stack, not on `document`. The old listener called
  // `stopPropagation` to keep the press to itself, which cannot work
  // from a listener on the same node the provider listens on — the
  // intent was already failing. Push order is what actually delivers
  // it: the menu opens last, so it answers first, and whatever opens
  // over the menu answers before the menu does.
  useShortcuts(
    "intelligence-file-ai-actions",
    "Dialog",
    [
      {
        key: "escape",
        label: "Close",
        editingOnly: false,
        hidden: true,
        handler: () => {
          setOpen(false);
          triggerRef.current?.focus();
        },
      },
    ],
    open,
    OVERLAY_PRIORITY,
  );

  // The offer disappears the moment its section has content, which can
  // happen while the menu is open — leave nothing hanging.
  useEffect(() => {
    if (actions.length === 0) setOpen(false);
  }, [actions.length]);

  const handleRun = useCallback((run: () => void) => {
    setOpen(false);
    run();
  }, []);

  useLayoutEffect(() => {
    if (!open) return;

    const measure = () => {
      const wrapper = wrapperRef.current;
      const menu = menuRef.current;
      if (!wrapper || !menu) return;

      const trigger = wrapper.getBoundingClientRect();
      // The rect, and it is the layout box because this menu no longer
      // animates — see the class list below. While it carried
      // `animate-fade-in-scale`, this read happened mid-animation and an
      // 82px menu measured 77.9, so the `+ MENU_GAP_PX` the line below
      // adds was cancelled by a scale nobody was accounting for.
      const menuHeight = menu.getBoundingClientRect().height;

      // The first ancestor that clips this menu, or the viewport.
      //
      // The walk stops at a `fixed` ancestor because such a box is laid
      // out against the viewport, so nothing above it clips what is
      // inside — and that is the case this exists for: at rest the file
      // detail draws this row in the sheet's 56px resting strip, which
      // core's `MobileInspectorSheet` positions `fixed bottom-0`. There
      // is nothing below the trigger there, so a menu that could only
      // hang downward hung off the bottom of the screen.
      //
      // **The same walk core's `FileActions` runs**, line for line —
      // the `[...]` beside this button in the same row. It was a subset
      // of that walk before: it read only the vertical edges and left out
      // the `absolute` detour, on the argument that the detour cannot
      // arise from here. Whether it can or not, the two being one walk is
      // what lets unit J lift it into a hook without first having to
      // decide which version was right. When it does, this is a caller
      // and the two constants above are the caller's.
      let bounds: { left: number; top: number; bottom: number } | null = null;
      let inAbsoluteDetour = false;
      for (let el = wrapper.parentElement; el; el = el.parentElement) {
        const { overflowX, overflowY, position } = getComputedStyle(el);
        const positioned =
          position === "relative" ||
          position === "absolute" ||
          position === "fixed" ||
          position === "sticky";
        if (
          (positioned || !inAbsoluteDetour) &&
          /auto|scroll|hidden/.test(overflowX + overflowY)
        ) {
          const box = el.getBoundingClientRect();
          bounds = { left: box.left, top: box.top, bottom: box.bottom };
          break;
        }
        if (position === "fixed") break;
        if (positioned) inAbsoluteDetour = position === "absolute";
      }

      // `visualViewport` rather than `innerHeight`: an on-screen keyboard
      // moves what can be seen without moving the layout viewport.
      const frame = bounds ?? {
        left: 0,
        top: 0,
        bottom: window.visualViewport?.height ?? window.innerHeight,
      };

      // The horizontal axis, and the same rule core's `FileActions` uses:
      // hang leftward from the trigger's right edge unless that would
      // cross the frame's left edge, in which case hang rightward from
      // its left edge instead.
      //
      // The default matters here rather than being a coin toss. This row
      // is drawn at the *right* of the sheet's strip — the file's name
      // takes the width with `flex-1` and the action row is
      // `flex-shrink-0` after it — so a menu that always hung rightward
      // ran off the screen by more than a third of itself at every phone
      // width, which is what this replaces.
      setAlignLeft(trigger.right - MENU_WIDTH_PX < frame.left);

      // The gap is the same 4px in both directions, so it decides only
      // whether the menu fits below at all and cancels out of the
      // comparison. Flips only when above is the better of the two, so a
      // trigger with room for neither keeps the direction the menu reads
      // as everywhere else.
      const spaceBelow = frame.bottom - trigger.bottom;
      const spaceAbove = trigger.top - frame.top;
      setOpenUp(menuHeight + MENU_GAP_PX > spaceBelow && spaceAbove > spaceBelow);
    };

    measure();

    // The list can lose an entry while the menu is open — a section that
    // finishes generating withdraws its offer — and that changes the
    // height the decision was made on. Flipping cannot re-enter this:
    // `bottom-full mb-1` and `top-full mt-1` move the box, they do not
    // resize it.
    const menu = menuRef.current;
    if (!menu || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(menu);
    return () => observer.disconnect();
  }, [open, actions.length]);

  if (actions.length === 0) return null;

  const busy = actions.some((action) => action.busy);
  const label = t("aiFileActions", { defaultMessage: "AI" });

  return (
    <div ref={wrapperRef} className="relative flex items-center">
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((s) => !s)}
        // The floor is the entry's own, not the row's. `.file-action-row-touch
        // > *` grows the *wrapper* this button sits in and centres it, so
        // the wrapper reaches 44 while the button inside stays at its own
        // height — `py-1.5` plus a 20px line box, which is 32.
        //
        // Height only. Width is already past the floor without any help:
        // `px-3` and the two lucide icons come to 66px before the label,
        // so a `min-w-11` would be a class that can never bind — and
        // nothing here could tell, since jsdom lays nothing out and the
        // test below can only ever pin the class string.
        //
        // Core's `globals.css` records this trigger at 44 and its siblings
        // at 32, which is the opposite of what was measured here. The two
        // are reconcilable — that number is the wrapper's, this one is the
        // button's — but neither is reproducible from this repository, so
        // it is recorded rather than asserted.
        className="flex items-center gap-1.5 rounded-full bg-bg-card px-3 py-1.5 text-sm text-text-primary transition-colors hover:bg-bg-elevated pointer-coarse:min-h-11"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
      >
        <Sparkles size={16} className={busy ? "animate-pulse" : ""} />
        <span>{label}</span>
        <ChevronDown size={14} className="text-text-muted" />
      </button>

      {open && (
        <DismissScrim
          onDismiss={() => setOpen(false)}
          // No tint, and the tier is all this box is: core's primitive
          // dismisses on a press outside the menu and swallows the click
          // that press produces, so nothing here depends on the scrim
          // being the element a tap reaches. Which matters twice over in
          // the Bottom Sheet, where `fixed inset-0` resolves against
          // vaul's transform and covers the drawer rather than the page.
          className="fixed inset-0 z-30"
        >
          <div
            ref={menuRef}
            role="menu"
            // Anchored to the trigger at every width, like `FileActions`
            // beside it in the same row — and, like it, on **both axes**
            // in the direction the box says there is room for. The
            // vertical half alone left the menu hanging a third of its
            // width off the right edge of every phone, because the row
            // this button sits in is drawn at the right of the strip.
            //
            // On a phone this row is drawn *inside* the Bottom Sheet, and
            // `Drawer.Content` carries a transform: a `fixed` box there
            // resolves against the drawer, not the viewport. The drawer
            // hangs below the fold by however far vaul has translated it,
            // so a menu pinned to "the bottom of the screen" landed below
            // the screen. Measured before this change, at 500x819: the
            // menu's top was 1008 — 189px past the bottom edge.
            //
            // Anchoring removes the question rather than answering it:
            // `absolute` resolves against the wrapper above, which is on
            // screen wherever the sheet is.
            // **No entry animation**, which is a change and not an
            // omission. `DESIGN.md` §Animation files `animate-fade-in-scale`
            // under modals and dialogs; core's `FileActions` — the only
            // other menu that measures its own height to pick a direction
            // — carries none. A menu that scales from 0.95 while being
            // measured reports a height it never has, and this is the one
            // place in the tree where that lands in a decision rather than
            // only on the eye.
            className={`absolute z-30 max-h-[60vh] min-w-[240px] overflow-y-auto rounded-2xl border border-bg-border bg-bg-primary py-1 shadow-lg ${
              openUp ? "bottom-full mb-1" : "top-full mt-1"
            } ${alignLeft ? "left-0" : "right-0"}`}
          >
            {actions.map((action) => {
              const Icon = ACTION_ICON[action.kind];
              return (
                <button
                  key={action.kind}
                  type="button"
                  role="menuitem"
                  disabled={action.busy}
                  onClick={() => handleRun(action.run)}
                  className="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm text-text-primary transition-colors hover:bg-bg-elevated disabled:opacity-50"
                >
                  <Icon size={16} className="flex-shrink-0 text-text-muted" />
                  <span className="flex-1">{t(action.labelKey)}</span>
                </button>
              );
            })}
          </div>
        </DismissScrim>
      )}
    </div>
  );
}
