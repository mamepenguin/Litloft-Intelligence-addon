"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import {
  AlertCircle,
  CheckCheck,
  ListVideo,
  RefreshCw,
  Sparkles,
  X,
} from "lucide-react";

import { formatDuration } from "@/lib/format";
import {
  FILE_CHAPTERS_UPDATED_EVENT,
  type FileChaptersUpdatedDetail,
} from "@/lib/addonEvents";
import { usePolicy } from "@/hooks/usePolicy";
import { useWebSocket } from "@/hooks/useWebSocket";
import type { FileType } from "@/types";
import {
  approveSuggestedChapters,
  dismissSuggestedChapters,
  generateSuggestedChapters,
  getSuggestedChapters,
} from "./api";
import type { SuggestedChaptersResponse } from "./api";

interface SuggestedChaptersSectionProps {
  fileId: string;
  drive: string;
  fileType: FileType;
}

type Operation = "approve" | "dismiss" | "generate" | null;

const CHAPTER_SUGGESTIONS_READY_EVENT =
  "intelligence.chapter_suggestions.ready";
const CHAPTER_SUGGESTIONS_FAILED_EVENT =
  "intelligence.chapter_suggestions.failed";

export default function SuggestedChaptersSection({
  fileId,
  drive,
  fileType,
}: SuggestedChaptersSectionProps) {
  const t = useTranslations("file");
  const [data, setData] = useState<SuggestedChaptersResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [operation, setOperation] = useState<Operation>(null);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);
  const isMedia = fileType === "video" || fileType === "audio";
  const policy = usePolicy(drive, "intelligence", "chapter_suggestions");
  const readyEvent = useWebSocket(CHAPTER_SUGGESTIONS_READY_EVENT);
  const failedEvent = useWebSocket(CHAPTER_SUGGESTIONS_FAILED_EVENT);

  const load = useCallback(async () => {
    if (!isMedia || policy.isLoading || !policy.enabled) {
      setLoaded(!policy.isLoading);
      return;
    }
    try {
      const result = await getSuggestedChapters(fileId, drive);
      if (!mountedRef.current) return;
      setData(result);
      setError(null);
    } catch {
      if (!mountedRef.current) return;
      setData(null);
      setError(t("chapterCandidatesLoadError", {
        defaultMessage: "Could not load chapter candidates. Try again.",
      }));
    } finally {
      if (mountedRef.current) setLoaded(true);
    }
  }, [drive, fileId, isMedia, policy.enabled, policy.isLoading, t]);

  useEffect(() => {
    mountedRef.current = true;
    setData(null);
    setLoaded(false);
    setOperation(null);
    setError(null);
    void load();
    return () => {
      mountedRef.current = false;
    };
  }, [load]);

  useEffect(() => {
    if (!readyEvent) return;
    if ((readyEvent.data as { file_id?: string }).file_id !== fileId) return;
    void load().finally(() => {
      if (mountedRef.current) setOperation(null);
    });
  }, [fileId, load, readyEvent]);

  useEffect(() => {
    if (!failedEvent || operation !== "generate") return;
    const failure = failedEvent.data as { file_id?: string; reason?: string };
    if (failure.file_id !== fileId) return;
    setOperation(null);
    // Retrying is the right advice only when a retry could work. A model
    // that spent its output budget thinking will do it again.
    setError(
      failure.reason === "model_token_budget"
        ? t("chapterCandidatesTokenBudget", {
            defaultMessage:
              "The model used its whole output budget on thinking, so no "
              + "chapters came back. Set llm.reasoning to disabled, or "
              + "choose a model that does not think.",
          })
        : t("chapterCandidatesGenerationFailed", {
            defaultMessage: "Chapter generation failed. Try creating them again.",
          }),
    );
  }, [failedEvent, fileId, operation, t]);

  const handleGenerate = useCallback(async () => {
    setOperation("generate");
    setError(null);
    try {
      await generateSuggestedChapters(fileId, drive);
    } catch {
      if (mountedRef.current) setError(t("chapterCandidatesActionError", {
        defaultMessage: "The chapter action failed. Try again.",
      }));
      if (mountedRef.current) setOperation(null);
    }
  }, [drive, fileId, t]);

  const handleApprove = useCallback(async () => {
    setOperation("approve");
    setError(null);
    try {
      await approveSuggestedChapters(fileId, drive);
      if (!mountedRef.current) return;
      setData((current) => current ? { ...current, status: "accepted" } : current);
      window.dispatchEvent(
        new CustomEvent<FileChaptersUpdatedDetail>(FILE_CHAPTERS_UPDATED_EVENT, {
          detail: { fileId },
        }),
      );
    } catch {
      if (mountedRef.current) setError(t("chapterCandidatesActionError", {
        defaultMessage: "The chapter action failed. Try again.",
      }));
    } finally {
      if (mountedRef.current) setOperation(null);
    }
  }, [drive, fileId, t]);

  const handleDismiss = useCallback(async () => {
    setOperation("dismiss");
    setError(null);
    try {
      await dismissSuggestedChapters(fileId, drive);
      if (!mountedRef.current) return;
      setData((current) => current ? { ...current, status: "dismissed" } : current);
    } catch {
      if (mountedRef.current) setError(t("chapterCandidatesActionError", {
        defaultMessage: "The chapter action failed. Try again.",
      }));
    } finally {
      if (mountedRef.current) setOperation(null);
    }
  }, [drive, fileId, t]);

  if (!loaded) return null;

  if (!isMedia || policy.isLoading || !policy.enabled) return null;

  if (data?.enabled === false) return null;

  if (!data) {
    return error ? <ErrorMessage message={error} /> : null;
  }

  const status = data.status;
  const chapters = data.chapters ?? [];
  const hasPending = data.enabled && data.available && status === "pending" && chapters.length > 0;

  if (!hasPending) {
    const statusMessage = status === "accepted"
      ? t("chapterCandidatesAccepted", {
          defaultMessage: "Chapter candidates approved",
        })
      : status === "dismissed"
        ? t("chapterCandidatesDismissed", {
            defaultMessage: "Chapter candidates dismissed",
          })
        : null;

    return (
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <Sparkles size={14} className="text-text-muted" aria-hidden="true" />
          {statusMessage && (
            <span className="text-xs text-text-muted">{statusMessage}</span>
          )}
          <button
            type="button"
            onClick={handleGenerate}
            disabled={operation !== null}
            className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
          >
            <RefreshCw
              size={11}
              className={operation === "generate" ? "animate-spin" : ""}
              aria-hidden="true"
            />
            {operation === "generate"
              ? t("generatingChapters", { defaultMessage: "Creating chapters..." })
              : statusMessage
                ? t("regenerateChapters", { defaultMessage: "Create again" })
                : t("generateChapters", {
                    defaultMessage: "Create AI chapter candidates",
                  })}
          </button>
        </div>
        {error && <ErrorMessage message={error} />}
      </div>
    );
  }

  return (
    <section aria-labelledby={`suggested-chapters-${fileId}`}>
      <div className="mb-2 flex items-center gap-2">
        <ListVideo size={14} className="text-accent-amber" aria-hidden="true" />
        <h2
          id={`suggested-chapters-${fileId}`}
          className="text-sm font-semibold text-text-muted"
        >
          {t("suggestedChapters", { defaultMessage: "AI chapter candidates" })}
        </h2>
        {data.model && (
          <span className="text-[10px] text-text-muted">{data.model}</span>
        )}
      </div>

      <ol className="overflow-hidden rounded-lg border border-dashed border-accent-amber/40 bg-accent-amber/8">
        {chapters.map((chapter, index) => (
          <li
            key={`${chapter.start_time}-${chapter.title}-${index}`}
            className="flex items-baseline gap-3 border-b border-bg-border px-3 py-2 last:border-b-0"
          >
            <time className="shrink-0 text-xs font-medium tabular-nums text-accent-amber">
              {formatDuration(chapter.start_time)}
            </time>
            <span className="min-w-0 text-sm text-text-primary">{chapter.title}</span>
          </li>
        ))}
      </ol>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={handleApprove}
          disabled={operation !== null}
          className="flex items-center gap-1 rounded-lg bg-accent-teal/15 px-2 py-1 text-xs font-medium text-accent-teal transition-colors hover:bg-accent-teal/20 disabled:opacity-50"
        >
          {operation === "approve" ? (
            <RefreshCw size={11} className="animate-spin" aria-hidden="true" />
          ) : (
            <CheckCheck size={11} aria-hidden="true" />
          )}
          {operation === "approve"
            ? t("approvingChapters", { defaultMessage: "Approving..." })
            : t("approveChapters", { defaultMessage: "Approve all" })}
        </button>
        <button
          type="button"
          onClick={handleDismiss}
          disabled={operation !== null}
          className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          {operation === "dismiss" ? (
            <RefreshCw size={11} className="animate-spin" aria-hidden="true" />
          ) : (
            <X size={11} aria-hidden="true" />
          )}
          {operation === "dismiss"
            ? t("dismissingChapters", { defaultMessage: "Dismissing..." })
            : t("dismissChapters", { defaultMessage: "Dismiss" })}
        </button>
        <button
          type="button"
          onClick={handleGenerate}
          disabled={operation !== null}
          className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          <RefreshCw
            size={11}
            className={operation === "generate" ? "animate-spin" : ""}
            aria-hidden="true"
          />
          {operation === "generate"
            ? t("generatingChapters", { defaultMessage: "Creating chapters..." })
            : t("regenerateChapters", { defaultMessage: "Create again" })}
        </button>
      </div>
      {error && <ErrorMessage message={error} />}
    </section>
  );
}

function ErrorMessage({ message }: { message: string }) {
  return (
    <p
      role="alert"
      className="mt-2 flex items-center gap-1.5 rounded-lg bg-danger-bg px-2 py-1.5 text-xs text-danger"
    >
      <AlertCircle size={13} className="shrink-0" aria-hidden="true" />
      {message}
    </p>
  );
}
