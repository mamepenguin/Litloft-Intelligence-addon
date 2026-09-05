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
        // `pointer-coarse:min-h-11` is the 44px touch floor (00-basis, mobile
        // sizing). The box itself, not an overhanging pseudo-element: that
        // recipe is `Button`'s and is for icon-only controls, where growing
        // the box would change a deliberate 32px square. This one carries a
        // label, so it can simply be tall enough.
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
        <>
          <div
            className="fixed inset-0 z-30 bg-black/30 sm:bg-transparent"
            aria-hidden="true"
            onClick={() => setOpen(false)}
          />
          <div
            role="menu"
            className="fixed inset-x-2 bottom-4 z-40 max-h-[60vh] overflow-y-auto rounded-2xl border border-bg-border bg-bg-primary py-1 shadow-lg animate-fade-in-scale sm:absolute sm:inset-x-auto sm:bottom-auto sm:left-0 sm:top-full sm:mt-1 sm:max-h-none sm:min-w-[240px] sm:overflow-visible sm:origin-top-left"
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
        </>
      )}
    </div>
  );
}
