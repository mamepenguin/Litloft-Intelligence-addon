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

import { useCallback, useEffect, useRef, useState } from "react";
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

  if (actions.length === 0) return null;

  const busy = actions.some((action) => action.busy);
  const label = t("aiFileActions", { defaultMessage: "AI" });

  return (
    <div className="relative flex items-center">
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
            role="menu"
            // Anchored to the trigger at every width, like `FileActions`
            // beside it in the same row — not a `fixed … bottom-4` sheet
            // below `sm`.
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
            className="absolute left-0 top-full z-30 mt-1 max-h-[60vh] min-w-[240px] overflow-y-auto rounded-2xl border border-bg-border bg-bg-primary py-1 shadow-lg animate-fade-in-scale origin-top-left"
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
