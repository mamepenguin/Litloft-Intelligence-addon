"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";
import {
  AlertTriangle,
  Clock,
  ExternalLink,
  Loader2,
  RefreshCw,
  X,
} from "lucide-react";

import { getFailedJobs, reindexFile } from "./api";
import type { FailedJobItem, ReindexTask } from "./api";

const PAGE_LIMIT = 50;

interface FailedJobsModalProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Map the backend ``job_kind`` to the canonical reindex ``task`` name.
 *
 * Spec ``2026-05-24-intelligence-reindex-controls.md`` §3.2 — the
 * worker side labels jobs by their pipeline stage (``transcription``,
 * ``clip``, ``text``, ``metadata``) while the reindex API speaks the
 * IndexedFile flag suffixes (``whisper``, ``clip``, ``text``,
 * ``metadata``). Unknown kinds fall back to metadata so the retry
 * action never silently dispatches an empty payload.
 */
function jobKindToTask(jobKind: string): ReindexTask {
  switch (jobKind) {
    case "transcription":
      return "whisper";
    case "clip":
      return "clip";
    case "text":
    case "text_content":
      return "text";
    case "metadata":
    default:
      return "metadata";
  }
}

function formatAttempted(iso: string): string {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

export default function FailedJobsModal({ open, onClose }: FailedJobsModalProps) {
  const t = useTranslations("semanticSearch.failedJobs");
  const tc = useTranslations("common");
  const [items, setItems] = useState<FailedJobItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [retrying, setRetrying] = useState<string | null>(null);
  const mountedRef = useRef<boolean>(true);

  const load = useCallback(
    async (nextOffset: number) => {
      setLoading(true);
      try {
        const res = await getFailedJobs(PAGE_LIMIT, nextOffset);
        if (!mountedRef.current) return;
        setItems(res.items ?? []);
        setTotal(res.total ?? 0);
      } catch {
        if (!mountedRef.current) return;
        setItems([]);
        setTotal(0);
      } finally {
        if (mountedRef.current) {
          setLoading(false);
        }
      }
    },
    [],
  );

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    setOffset(0);
    load(0);
  }, [open, load]);

  const handleRetry = useCallback(
    async (row: FailedJobItem) => {
      const task = jobKindToTask(row.job_kind);
      const key = `${row.file_id}:${task}`;
      setRetrying(key);
      try {
        await reindexFile(row.file_id, [task], row.drive);
        // Optimistically drop the row from the visible list so the
        // operator can see they actually retried it. The 10 s poll on
        // IndexStatusWidget refreshes the count summary in the
        // background.
        setItems((prev) =>
          prev.filter(
            (it) =>
              !(it.file_id === row.file_id && it.job_kind === row.job_kind),
          ),
        );
      } catch {
        // Leave the row in place — next poll will re-fetch.
      } finally {
        setRetrying(null);
      }
    },
    [],
  );

  const handlePrev = useCallback(() => {
    if (offset <= 0) return;
    const next = Math.max(0, offset - PAGE_LIMIT);
    setOffset(next);
    load(next);
  }, [offset, load]);

  const handleNext = useCallback(() => {
    if (offset + PAGE_LIMIT >= total) return;
    const next = offset + PAGE_LIMIT;
    setOffset(next);
    load(next);
  }, [offset, total, load]);

  if (!open) return null;

  const hasItems = items.length > 0;
  const showingFrom = total > 0 ? offset + 1 : 0;
  const showingTo = Math.min(offset + items.length, total);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />
      <div className="relative mx-4 flex w-full max-w-4xl flex-col rounded-2xl bg-bg-card p-6 shadow-lg animate-fade-in-scale max-h-[85vh]">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle size={18} className="text-accent-amber" />
            <h2 className="text-lg font-semibold text-text-primary">
              {t("title")}
            </h2>
            {total > 0 && (
              <span className="rounded-lg bg-accent-amber/10 px-2 py-0.5 text-xs font-medium text-accent-amber">
                {t("countBadge", { count: total })}
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="rounded-xl p-1 text-text-muted hover:text-text-primary"
            aria-label={tc("close")}
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {loading && !hasItems ? (
            <div className="flex items-center justify-center py-12 text-text-muted">
              <Loader2 size={20} className="animate-spin" />
            </div>
          ) : !hasItems ? (
            <div className="py-12 text-center text-sm text-text-muted">
              {t("none")}
            </div>
          ) : (
            <table className="w-full text-xs" role="table">
              <thead>
                <tr className="border-b border-bg-border text-left text-text-muted">
                  <th
                    className="w-6 px-1 py-2"
                    aria-hidden
                    data-checkbox-slot
                    style={{ width: "24px" }}
                  />
                  <th className="px-2 py-2 font-medium">{t("col.file")}</th>
                  <th className="px-2 py-2 font-medium">{t("col.drive")}</th>
                  <th className="px-2 py-2 font-medium">{t("col.task")}</th>
                  <th className="px-2 py-2 font-medium">{t("col.provider")}</th>
                  <th className="px-2 py-2 font-medium">{t("col.error")}</th>
                  <th className="px-2 py-2 font-medium">{t("col.attempts")}</th>
                  <th className="px-2 py-2 font-medium">{t("col.attemptedAt")}</th>
                  <th className="px-2 py-2 font-medium">{t("col.actions")}</th>
                </tr>
              </thead>
              <tbody>
                {items.map((row) => {
                  const task = jobKindToTask(row.job_kind);
                  const retryKey = `${row.file_id}:${task}`;
                  const isRetrying = retrying === retryKey;
                  return (
                    <tr
                      key={`${row.file_id}:${row.job_kind}:${row.provider ?? "none"}`}
                      className="border-b border-bg-border/50 align-top text-text-primary"
                      role="row"
                    >
                      <td
                        className="px-1 py-2"
                        data-checkbox-slot
                        style={{ width: "24px" }}
                      />
                      <td className="px-2 py-2">
                        <span
                          className="block max-w-[220px] truncate font-medium"
                          title={row.filename}
                        >
                          {row.filename}
                        </span>
                      </td>
                      <td className="px-2 py-2 text-text-muted">{row.drive}</td>
                      <td className="px-2 py-2">{row.job_kind}</td>
                      <td className="px-2 py-2 text-text-muted">
                        {row.provider ?? "-"}
                      </td>
                      <td className="px-2 py-2">
                        <span className="block font-medium text-accent-amber">
                          {row.error_class ?? "-"}
                        </span>
                        <span
                          className="block max-w-[260px] truncate text-text-muted"
                          title={row.error_message_excerpt ?? ""}
                        >
                          {row.error_message_excerpt ?? ""}
                        </span>
                      </td>
                      <td className="px-2 py-2 text-text-muted">
                        {row.attempts}
                      </td>
                      <td
                        className="px-2 py-2 text-text-muted whitespace-nowrap"
                        title={row.attempted_at}
                      >
                        <span className="inline-flex items-center gap-1">
                          <Clock size={12} aria-hidden />
                          {formatAttempted(row.attempted_at)}
                        </span>
                      </td>
                      <td className="px-2 py-2">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleRetry(row)}
                            disabled={isRetrying}
                            className="inline-flex items-center gap-1 rounded-lg bg-bg-elevated px-2 py-1 text-xs font-medium text-text-primary transition-colors hover:bg-bg-border disabled:opacity-50"
                            aria-label={t("retry")}
                          >
                            {isRetrying ? (
                              <Loader2 size={12} className="animate-spin" />
                            ) : (
                              <RefreshCw size={12} />
                            )}
                            {t("retry")}
                          </button>
                          <Link
                            href={`/drive/${encodeURIComponent(row.drive)}/file/${row.file_id}`}
                            className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-accent hover:underline"
                            aria-label={t("details")}
                          >
                            <ExternalLink size={12} />
                            {t("details")}
                          </Link>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        {total > PAGE_LIMIT && (
          <div className="mt-3 flex items-center justify-between border-t border-bg-border pt-3 text-xs text-text-muted">
            <span>
              {t("pagination", { from: showingFrom, to: showingTo, total })}
            </span>
            <div className="flex gap-2">
              <button
                onClick={handlePrev}
                disabled={offset === 0 || loading}
                className="rounded-lg bg-bg-elevated px-3 py-1 font-medium text-text-primary hover:bg-bg-border disabled:opacity-50"
              >
                {t("prev")}
              </button>
              <button
                onClick={handleNext}
                disabled={offset + PAGE_LIMIT >= total || loading}
                className="rounded-lg bg-bg-elevated px-3 py-1 font-medium text-text-primary hover:bg-bg-border disabled:opacity-50"
              >
                {t("next")}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
