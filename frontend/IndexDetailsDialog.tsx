"use client";

/**
 * Per-file indexing state, with a Regenerate button per task.
 *
 * This is operator-facing rather than reader-facing, so it lives behind
 * the core `[...]` menu's `file-actions-menu` slot rather than occupying
 * a section on the file detail page. `IndexDetailsMenuItem` owns the
 * entry that opens it.
 *
 * Portalled to `document.body`: the menu that hosts the entry is `z-30`
 * and clips its overflow, so a dialog rendered in place would be cut off.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useTranslations } from "next-intl";
import {
  Check,
  CircleDashed,
  FileText,
  Image as ImageIcon,
  Loader2,
  Mic,
  RefreshCw,
  ScrollText,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { getIndexDetails, reindexFile } from "./api";
import type { IndexDetailsResponse, ReindexTask } from "./api";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { useShortcuts } from "@/hooks/useShortcuts";

export interface IndexDetailsDialogProps {
  open: boolean;
  fileId: string;
  drive: string;
  mimeType?: string;
  fileType?: string;
  onClose: () => void;
}

interface TaskSpec {
  task: ReindexTask;
  // Existing ``semanticSearch.tasks.*.label`` keys live under
  // ``metadata`` / ``clip`` / ``whisper`` / ``text_content``. The
  // reindex API speaks ``text`` instead of ``text_content`` (it lines
  // up with the IndexedFile ``text_indexed`` flag) so we map here
  // rather than rename the long-lived translation key.
  i18nKey: "metadata" | "clip" | "whisper" | "text_content";
  icon: LucideIcon;
}

const TASK_SPECS: TaskSpec[] = [
  { task: "metadata", i18nKey: "metadata", icon: ScrollText },
  { task: "clip", i18nKey: "clip", icon: ImageIcon },
  { task: "whisper", i18nKey: "whisper", icon: Mic },
  { task: "text", i18nKey: "text_content", icon: FileText },
];

// .loft files are remote-URL wrappers (YouTube / Vimeo / Soundcloud).
// Transcription uses an adjacent .vtt managed by media_import (not
// manually retriggerable), and text extraction is auto-skipped by the
// indexer because the mime is not in TEXT_MIMES. Neither task is
// meaningful to show the user.
const LOFT_MIME = "application/vnd.litloft.loft+json";

/**
 * Decide whether a given task is applicable to the file's mime/type.
 *
 * Mirrors backend pipeline gating so the user never sees a "Regenerate"
 * button that the worker would immediately mark UnsupportedMimeType.
 *
 * - ``metadata`` is universal (every file has metadata).
 * - ``clip`` covers image + video (CLIP can embed video frames).
 * - ``whisper`` covers audio + video (transcribable media).
 *   .loft transcription is driven by media_import's adjacent .vtt,
 *   not a user-facing reindex action — hidden here.
 * - ``text`` covers text/* mime types and the legacy ``text`` file_type
 *   (loft markdown, plain text, etc.). PDFs surface as
 *   ``application/pdf`` which the text pipeline also handles.
 *   .loft is auto-skipped (mime not in TEXT_MIMES) — hidden here.
 */
function isTaskApplicable(
  task: ReindexTask,
  mimeType: string | undefined,
  fileType: string | undefined,
): boolean {
  const mt = (mimeType ?? "").toLowerCase();
  const ft = (fileType ?? "").toLowerCase();
  switch (task) {
    case "metadata":
      return true;
    case "clip":
      // Image or video carries visual frames CLIP can embed.
      return (
        mt.startsWith("image/") ||
        mt.startsWith("video/") ||
        ft === "image" ||
        ft === "video"
      );
    case "whisper":
      // .loft transcription is via adjacent .vtt (media_import), not
      // a manually triggerable Whisper job.
      if (mt === LOFT_MIME) return false;
      // Audio or video carries a transcribable track.
      return (
        mt.startsWith("audio/") ||
        mt.startsWith("video/") ||
        ft === "audio" ||
        ft === "video"
      );
    case "text":
      // .loft mime is not in TEXT_MIMES — the indexer auto-marks it
      // done without extracting anything. Nothing to show the user.
      if (mt === LOFT_MIME) return false;
      // text/* mime, application/pdf, video (subtitle / sidecar), audio
      // (transcript text), or the legacy "text" / "loft" file_type.
      // Hidden only for pure image files where there is no text
      // content to extract.
      return !(mt.startsWith("image/") || ft === "image");
    default:
      return false;
  }
}

export default function IndexDetailsDialog({
  open,
  fileId,
  drive,
  mimeType,
  fileType,
  onClose,
}: IndexDetailsDialogProps) {
  const t = useTranslations("semanticSearch");
  const tDetails = useTranslations("semanticSearch.indexDetails");
  const tc = useTranslations("common");
  const [details, setDetails] = useState<IndexDetailsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [pendingTask, setPendingTask] = useState<ReindexTask | null>(null);
  const [regenerating, setRegenerating] = useState<ReindexTask | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  // Generation counter, not an is-mounted flag. Closing the dialog bumps
  // it, so a response that lands afterwards is discarded rather than
  // written into state a later open would show — and, worse, rather than
  // undoing the optimistic flip a Regenerate made in between.
  const requestIdRef = useRef(0);

  const fetchDetails = useCallback(async () => {
    const requestId = ++requestIdRef.current;
    setLoading(true);
    try {
      const res = await getIndexDetails(fileId, drive);
      if (requestId !== requestIdRef.current) return;
      setDetails(res);
    } catch {
      if (requestId !== requestIdRef.current) return;
      setDetails(null);
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, [fileId, drive]);

  // Gated on `open` so merely having the menu entry in the DOM costs no
  // request — the details are only fetched once someone asks for them.
  useEffect(() => {
    if (!open) {
      // Invalidate anything still in flight from the previous open.
      requestIdRef.current++;
      return;
    }
    fetchDetails();
  }, [open, fetchDetails]);

  // The menu entry behind this dialog still holds focus, so move it here
  // or the dialog is never announced and Tab walks the page behind it.
  useEffect(() => {
    if (open) panelRef.current?.focus();
  }, [open]);

  // Registered rather than bound to window so the shortcut stack decides
  // precedence: the confirmation pushes later and wins, and the cheat
  // sheet suppresses us entirely. Disabled while the confirmation is up
  // so one press cannot dismiss both layers.
  useShortcuts(
    "intelligence-index-details",
    "Index details",
    [{ key: "escape", label: "Close", handler: onClose, hidden: true }],
    open && pendingTask === null,
  );

  const handleRequestRegenerate = useCallback((task: ReindexTask) => {
    setPendingTask(task);
  }, []);

  const handleCancel = useCallback(() => {
    setPendingTask(null);
  }, []);

  const handleConfirm = useCallback(async () => {
    if (!pendingTask) return;
    const task = pendingTask;
    setPendingTask(null);
    setRegenerating(task);
    // Closing the dialog mid-flight invalidates the write, the same way
    // it invalidates a fetch.
    const requestId = requestIdRef.current;
    try {
      await reindexFile(fileId, [task], drive);
      if (requestId !== requestIdRef.current) return;
      // Optimistic flip — the *_indexed flag is False until the worker
      // produces a fresh embedding. The next poll / fetch will catch
      // up; we update the local state so the row immediately shows
      // "queued" instead of leaving the stale "done" badge in place.
      setDetails((prev) =>
        prev
          ? { ...prev, status: { ...prev.status, [task]: false } }
          : prev,
      );
    } catch {
      // ignore — leave the existing badge in place
    } finally {
      if (requestId === requestIdRef.current) {
        setRegenerating(null);
      }
    }
  }, [pendingTask, fileId, drive]);

  if (!open) return null;

  // Only render rows whose task applies to this file's mime/type.
  const visibleTasks = TASK_SPECS.filter((spec) =>
    isTaskApplicable(spec.task, mimeType, fileType),
  );

  return createPortal(
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <div
          className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
          onClick={onClose}
        />
        <div
          ref={panelRef}
          role="dialog"
          aria-modal="true"
          aria-label={tDetails("title")}
          tabIndex={-1}
          className="relative mx-4 w-full max-w-md rounded-2xl bg-bg-card p-6 shadow-lg outline-none animate-fade-in-scale"
        >
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-text-primary">
              {tDetails("title")}
            </h2>
            <button
              onClick={onClose}
              className="rounded-xl p-1 text-text-muted hover:text-text-primary"
              aria-label={tc("close")}
            >
              <X size={18} />
            </button>
          </div>

          {loading && !details ? (
            <div className="h-5 w-32 animate-pulse rounded-lg bg-bg-elevated" />
          ) : (
            <ul className="space-y-2" role="list">
              {visibleTasks.map((spec) => {
                const Icon = spec.icon;
                const done = details?.status?.[spec.task] ?? false;
                const isRegenerating = regenerating === spec.task;
                return (
                  <li
                    key={spec.task}
                    data-task-row={spec.task}
                    className="flex items-center justify-between gap-3 rounded-lg border border-bg-border/50 bg-bg-elevated/40 px-3 py-2"
                  >
                    <div className="flex items-center gap-2 text-xs">
                      <Icon
                        size={14}
                        className={done ? "text-accent-teal" : "text-text-muted"}
                      />
                      <span className="font-medium text-text-primary">
                        {t(`tasks.${spec.i18nKey}.label`)}
                      </span>
                      <span className="inline-flex items-center gap-1 text-text-muted">
                        {done ? (
                          <>
                            <Check size={12} className="text-accent-teal" />
                            {tDetails("statusDone")}
                          </>
                        ) : (
                          <>
                            <CircleDashed size={12} />
                            {tDetails("statusPending")}
                          </>
                        )}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleRequestRegenerate(spec.task)}
                      disabled={isRegenerating}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-bg-elevated px-2.5 py-1 text-xs font-medium text-text-primary transition-colors hover:bg-bg-border disabled:opacity-50"
                      aria-label={tDetails("regenerate")}
                    >
                      {isRegenerating ? (
                        <Loader2 size={12} className="animate-spin" />
                      ) : (
                        <RefreshCw size={12} />
                      )}
                      {tDetails("regenerate")}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={pendingTask !== null}
        title={tDetails("regenerate")}
        message={
          pendingTask
            ? tDetails("confirmRegenerate", {
                task: t(
                  `tasks.${
                    pendingTask === "text" ? "text_content" : pendingTask
                  }.label`,
                ),
              })
            : ""
        }
        confirmLabel={tDetails("regenerate")}
        onConfirm={handleConfirm}
        onCancel={handleCancel}
      />
    </>,
    document.body,
  );
}
