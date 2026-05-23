"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { getIndexDetails, reindexFile } from "./api";
import type { IndexDetailsResponse, ReindexTask } from "./api";
import { ConfirmDialog } from "@/components/ConfirmDialog";

interface IndexDetailsSectionProps {
  fileId: string;
  drive: string;
  mimeType?: string;
  fileType?: string;
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

/**
 * Decide whether a given task is applicable to the file's mime/type.
 *
 * Mirrors backend pipeline gating so the user never sees a "Regenerate"
 * button that the worker would immediately mark UnsupportedMimeType.
 *
 * - ``metadata`` is universal (every file has metadata).
 * - ``clip`` covers image + video (CLIP can embed video frames).
 * - ``whisper`` covers audio + video (transcribable media).
 * - ``text`` covers text/* mime types and the legacy ``text`` file_type
 *   (loft markdown, plain text, etc.). PDFs surface as
 *   ``application/pdf`` which the text pipeline also handles.
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
      // Audio or video carries a transcribable track.
      return (
        mt.startsWith("audio/") ||
        mt.startsWith("video/") ||
        ft === "audio" ||
        ft === "video"
      );
    case "text":
      // text/* mime, application/pdf, video (subtitle / sidecar), audio
      // (transcript text), or the legacy "text" / "loft" file_type.
      // Hidden only for pure image files where there is no text
      // content to extract.
      return !(mt.startsWith("image/") || ft === "image");
    default:
      return false;
  }
}

export default function IndexDetailsSection({
  fileId,
  drive,
  mimeType,
  fileType,
}: IndexDetailsSectionProps) {
  const t = useTranslations("semanticSearch");
  const tDetails = useTranslations("semanticSearch.indexDetails");
  const [details, setDetails] = useState<IndexDetailsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [pendingTask, setPendingTask] = useState<ReindexTask | null>(null);
  const [regenerating, setRegenerating] = useState<ReindexTask | null>(null);
  const mountedRef = useRef<boolean>(true);

  const fetchDetails = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getIndexDetails(fileId, drive);
      if (!mountedRef.current) return;
      setDetails(res);
    } catch {
      if (!mountedRef.current) return;
      setDetails(null);
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [fileId, drive]);

  useEffect(() => {
    mountedRef.current = true;
    fetchDetails();
    return () => {
      mountedRef.current = false;
    };
  }, [fetchDetails]);

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
    try {
      await reindexFile(fileId, [task], drive);
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
      setRegenerating(null);
    }
  }, [pendingTask, fileId, drive]);

  // Only render rows whose task applies to this file's mime/type.
  const visibleTasks = TASK_SPECS.filter((spec) =>
    isTaskApplicable(spec.task, mimeType, fileType),
  );

  if (loading && !details) {
    return (
      <section className="rounded-xl border border-bg-border bg-bg-card p-4">
        <div className="h-5 w-32 animate-pulse rounded-lg bg-bg-elevated" />
      </section>
    );
  }

  return (
    <section className="rounded-xl border border-bg-border bg-bg-card p-4">
      <h3 className="mb-3 text-sm font-semibold text-text-primary">
        {tDetails("title")}
      </h3>
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
    </section>
  );
}
