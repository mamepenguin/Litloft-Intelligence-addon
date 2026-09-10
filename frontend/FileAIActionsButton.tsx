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
      // Core's `FileActions` — the `[...]` beside this button in the same
      // row — carries the full version of this walk, including the case
      // where an `absolute` ancestor takes the chain out of the DOM
      // parentage for a stretch. That case cannot arise from here: every
      // ancestor between this wrapper and the strip is in flow. When core
      // extracts the walk into a hook of its own, this is its first
      // caller.
      let frame: { top: number; bottom: number } | null = null;
      for (let el = wrapper.parentElement; el; el = el.parentElement) {
        const { overflowX, overflowY, position } = getComputedStyle(el);
        if (/auto|scroll|hidden/.test(overflowX + overflowY)) {
          const box = el.getBoundingClientRect();
          frame = { top: box.top, bottom: box.bottom };
          break;
        }
        if (position === "fixed") break;
      }

      // `visualViewport` rather than `innerHeight`: an on-screen keyboard
      // moves what can be seen without moving the layout viewport.
      const bounds = frame ?? {
        top: 0,
        bottom: window.visualViewport?.height ?? window.innerHeight,
      };

      // The gap is the same 4px in both directions, so it decides only
      // whether the menu fits below at all and cancels out of the
      // comparison. Flips only when above is the better of the two, so a
      // trigger with room for neither keeps the direction the menu reads
      // as everywhere else.
      const spaceBelow = bounds.bottom - trigger.bottom;
      const spaceAbove = trigger.top - bounds.top;
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
            // beside it in the same row — and, like it, in the direction
            // the box says there is room for.
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
            className={`absolute left-0 z-30 max-h-[60vh] min-w-[240px] overflow-y-auto rounded-2xl border border-bg-border bg-bg-primary py-1 shadow-lg animate-fade-in-scale ${
              openUp ? "bottom-full mb-1 origin-bottom-left" : "top-full mt-1 origin-top-left"
            }`}
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
