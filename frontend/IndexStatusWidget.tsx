"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import {
  Brain,
  FileText,
  Image as ImageIcon,
  Loader2,
  MessageSquareText,
  Mic,
  Pause,
  Play,
  RefreshCw,
  ScrollText,
  SearchX,
  Sparkles,
  Tags,
  WandSparkles,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import {
  getSearchStatus,
  searchQueuePause,
  searchQueueReindex,
  searchQueueResume,
} from "./api";
import type {
  QueueProcessingFile,
  QueueTaskBreakdown,
  QueueTaskKind,
  SearchServiceStatus,
} from "./api";
import { ConfirmDialog } from "@/components/ConfirmDialog";

const POLL_INTERVAL = 10_000;

// Display order on the dashboard. Indexing pipeline first (metadata →
// clip → whisper → text), then LLM-driven tasks. Items without backend
// state are filtered out at render time.
const TASK_ORDER: QueueTaskKind[] = [
  "metadata",
  "clip",
  "whisper",
  "text_content",
  "auto_tags",
  "summaries",
  "vision_describe",
  "transcript_refine",
];

const TASK_ICON: Record<QueueTaskKind, LucideIcon> = {
  metadata: ScrollText,
  clip: ImageIcon,
  whisper: Mic,
  text_content: FileText,
  auto_tags: Tags,
  summaries: MessageSquareText,
  vision_describe: Sparkles,
  transcript_refine: WandSparkles,
};

function ProgressBar({
  done,
  total,
  label,
}: {
  done: number;
  total: number;
  label: string;
}) {
  const percent = total > 0 ? (done / total) * 100 : 0;

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-text-muted">{label}</span>
        <span className="text-text-primary">{Math.floor(percent)}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-bg-elevated">
        <div
          className="h-full rounded-full bg-accent transition-all"
          style={{ width: `${Math.min(percent, 100)}%` }}
        />
      </div>
    </div>
  );
}

function describeProcessingFile(file: QueueProcessingFile): string {
  // Prefer the human-readable filename; fall back to a short id so the
  // dashboard never renders an empty active row when a file was purged
  // mid-flight.
  if (file.filename) return file.filename;
  return `#${file.file_id.slice(0, 8)}`;
}

function TaskRow({
  kind,
  breakdown,
}: {
  kind: QueueTaskKind;
  breakdown: QueueTaskBreakdown;
}) {
  const t = useTranslations("semanticSearch");
  const Icon = TASK_ICON[kind];
  const processingCount = breakdown.processing.length;
  const waiting = breakdown.waiting;
  const isActive = processingCount > 0 || waiting > 0;

  return (
    <div className="rounded-lg border border-bg-border bg-bg-elevated/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs font-medium text-text-primary">
          <Icon size={14} className={isActive ? "text-accent" : "text-text-muted"} />
          <span>{t(`tasks.${kind}.label`)}</span>
        </div>
        <span className="text-xs text-text-muted">
          {t("tasks.queueCounts", {
            processing: processingCount,
            waiting,
          })}
        </span>
      </div>

      {processingCount > 0 ? (
        <ul className="mt-2 space-y-1 text-xs text-text-primary">
          {breakdown.processing.map((file) => (
            <li
              key={file.file_id}
              className="flex items-center gap-1.5 truncate"
              title={file.filename ?? file.file_id}
            >
              <Loader2 size={10} className="animate-spin text-accent shrink-0" />
              <span className="truncate">{describeProcessingFile(file)}</span>
            </li>
          ))}
        </ul>
      ) : waiting > 0 ? (
        <p className="mt-2 text-xs text-text-muted">
          {t("tasks.waitingHint", { waiting })}
        </p>
      ) : (
        <p className="mt-2 text-xs text-text-muted">{t("tasks.idle")}</p>
      )}
    </div>
  );
}

function StatusContent({ status, drive }: { status: SearchServiceStatus; drive?: string }) {
  const t = useTranslations("semanticSearch");
  const [confirmReindex, setConfirmReindex] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const handlePauseResume = useCallback(async () => {
    const isPaused = status.queue?.paused;
    setActionLoading(isPaused ? "resume" : "pause");
    try {
      if (isPaused) {
        await searchQueueResume(drive);
      } else {
        await searchQueuePause(drive);
      }
    } catch {
      // Silently fail - next poll will show actual state
    }
    setActionLoading(null);
  }, [status.queue?.paused, drive]);

  const handleReindex = useCallback(async () => {
    setConfirmReindex(false);
    setActionLoading("reindex");
    try {
      await searchQueueReindex(drive);
    } catch {
      // Silently fail
    }
    setActionLoading(null);
  }, [drive]);

  const indexed = status.indexed;
  const pending = status.pending;
  const queue = status.queue;
  const totalFiles = indexed?.total ?? 0;
  const whisperTotal = indexed
    ? indexed.whisper + (pending?.whisper ?? 0)
    : 0;
  const clipTotal = indexed
    ? indexed.clip + (pending?.clip ?? 0)
    : 0;
  const textTotal = indexed
    ? (indexed.text ?? 0) + (pending?.text ?? 0)
    : 0;

  const taskMap = queue?.tasks ?? {};
  const taskRows = TASK_ORDER.map((kind) => ({ kind, breakdown: taskMap[kind] }))
    .filter(
      (entry): entry is { kind: QueueTaskKind; breakdown: QueueTaskBreakdown } =>
        entry.breakdown !== undefined,
    );

  return (
    <>
      <div className="mb-3 flex items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
          {t("statusRunning")}
        </span>
        {queue?.paused && (
          <span className="inline-flex items-center gap-1.5 rounded-md bg-yellow-500/10 px-2 py-0.5 text-xs font-medium text-yellow-400">
            <Pause size={10} />
            {t("paused")}
          </span>
        )}
      </div>

      <div className="space-y-3">
        {indexed && (
          <ProgressBar
            done={indexed.metadata}
            total={totalFiles}
            label={t("indexedFiles", {
              indexed: indexed.metadata.toLocaleString(),
              total: totalFiles.toLocaleString(),
            })}
          />
        )}

        {indexed && whisperTotal > 0 && (
          <ProgressBar
            done={indexed.whisper}
            total={whisperTotal}
            label={t("whisperFiles", {
              done: indexed.whisper.toLocaleString(),
              total: whisperTotal.toLocaleString(),
            })}
          />
        )}

        {indexed && clipTotal > 0 && (
          <ProgressBar
            done={indexed.clip}
            total={clipTotal}
            label={t("clipFiles", {
              done: indexed.clip.toLocaleString(),
              total: clipTotal.toLocaleString(),
            })}
          />
        )}

        {indexed && textTotal > 0 && (
          <ProgressBar
            done={indexed.text ?? 0}
            total={textTotal}
            label={t("textFiles", {
              done: (indexed.text ?? 0).toLocaleString(),
              total: textTotal.toLocaleString(),
            })}
          />
        )}

        {queue && (
          <div className="flex items-center justify-between text-xs">
            <span className="text-text-muted">{t("queueLabel")}</span>
            <span className="text-text-primary">
              {t("queueStatus", {
                processing: queue.processing,
                waiting: queue.waiting,
              })}
            </span>
          </div>
        )}
      </div>

      {taskRows.length > 0 && (
        <div className="mt-4 space-y-2">
          <div className="text-xs font-semibold text-text-muted">
            {t("tasks.heading")}
          </div>
          <div className="space-y-2">
            {taskRows.map(({ kind, breakdown }) => (
              <TaskRow key={kind} kind={kind} breakdown={breakdown} />
            ))}
          </div>
        </div>
      )}

      <div className="mt-4 flex gap-2">
        <button
          onClick={() => setConfirmReindex(true)}
          disabled={actionLoading === "reindex"}
          className="inline-flex items-center gap-1.5 rounded-lg bg-bg-elevated px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-bg-border disabled:opacity-50"
        >
          {actionLoading === "reindex" ? (
            <Loader2 size={12} className="animate-spin" />
          ) : (
            <RefreshCw size={12} />
          )}
          {t("reindex")}
        </button>
        <button
          onClick={handlePauseResume}
          disabled={actionLoading === "pause" || actionLoading === "resume"}
          className="inline-flex items-center gap-1.5 rounded-lg bg-bg-elevated px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-bg-border disabled:opacity-50"
        >
          {actionLoading === "pause" || actionLoading === "resume" ? (
            <Loader2 size={12} className="animate-spin" />
          ) : queue?.paused ? (
            <Play size={12} />
          ) : (
            <Pause size={12} />
          )}
          {queue?.paused ? t("resume") : t("pause")}
        </button>
      </div>

      <ConfirmDialog
        open={confirmReindex}
        title={t("reindex")}
        message={t("confirmReindex")}
        onConfirm={handleReindex}
        onCancel={() => setConfirmReindex(false)}
      />
    </>
  );
}

interface IndexStatusWidgetProps {
  // Optional: when omitted the widget renders the global admin view
  // (process-wide queue + total indexed counts), suitable for /admin's
  // dashboard-widgets slot. When passed, it currently renders the same
  // global counters but flagged with the drive context — a future
  // change can split per-drive vs global counters here.
  drive?: string;
}

export default function IndexStatusWidget({ drive }: IndexStatusWidgetProps) {
  const t = useTranslations("semanticSearch");
  const [status, setStatus] = useState<SearchServiceStatus | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchStatus = useCallback(async () => {
    const result = await getSearchStatus(drive);
    setStatus(result);
  }, [drive]);

  useEffect(() => {
    fetchStatus();
    intervalRef.current = setInterval(fetchStatus, POLL_INTERVAL);
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchStatus]);

  if (status === null) {
    return (
      <div className="rounded-xl border border-bg-border bg-bg-card p-5 animate-pulse">
        <div className="mb-4 h-5 w-40 rounded bg-bg-elevated" />
        <div className="space-y-3">
          <div className="h-3 w-full rounded bg-bg-elevated" />
          <div className="h-3 w-3/4 rounded bg-bg-elevated" />
          <div className="h-3 w-1/2 rounded bg-bg-elevated" />
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-bg-border bg-bg-card p-5">
      <div className="mb-4 flex items-center gap-2">
        <Brain size={18} className="text-accent" />
        <h3 className="text-sm font-semibold text-text-primary">
          {t("title")}
        </h3>
      </div>

      {status.available ? (
        <StatusContent status={status} drive={drive} />
      ) : (
        <div className="flex items-center gap-2 text-sm text-text-muted">
          <SearchX size={16} />
          {t("unavailable")}
        </div>
      )}
    </div>
  );
}
