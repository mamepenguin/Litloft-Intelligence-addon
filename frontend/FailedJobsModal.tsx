"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  ExternalLink,
  Loader2,
  RefreshCw,
  X,
} from "lucide-react";

import { getFailedJobs, reindexFile, resolveFailedJob } from "./api";
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

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

function formatAttempted(iso: string): { date: string; time: string | null } {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return { date: iso, time: null };
    return {
      date: `${d.getFullYear()}/${pad2(d.getMonth() + 1)}/${pad2(d.getDate())}`,
      time: `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`,
    };
  } catch {
    return { date: iso, time: null };
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
  const [resolving, setResolving] = useState<string | null>(null);
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

  const removeVisibleGroup = useCallback((row: FailedJobItem) => {
    setItems((prev) =>
      prev.filter(
        (it) =>
          !(
            it.file_id === row.file_id &&
            it.job_kind === row.job_kind &&
            it.provider === row.provider
          ),
      ),
    );
    setTotal((prev) => Math.max(0, prev - 1));
  }, []);

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
        removeVisibleGroup(row);
      } catch {
        // Leave the row in place — next poll will re-fetch.
      } finally {
        setRetrying(null);
      }
    },
    [removeVisibleGroup],
  );

  const handleResolve = useCallback(
    async (row: FailedJobItem) => {
      const key = `${row.file_id}:${row.job_kind}:${row.provider ?? "none"}`;
      setResolving(key);
      try {
        await resolveFailedJob({
          file_id: row.file_id,
          job_kind: row.job_kind,
          provider: row.provider,
        });
        removeVisibleGroup(row);
      } catch {
        // Leave the row in place — next poll will re-fetch.
      } finally {
        setResolving(null);
      }
    },
    [removeVisibleGroup],
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
  const pageSummary =
    total > 0
      ? t("pagination", { from: showingFrom, to: showingTo, total })
      : null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center px-3 py-4 sm:px-6"
      role="dialog"
      aria-modal
      aria-labelledby="failed-jobs-modal-title"
    >
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />
      <div className="relative flex max-h-[min(86vh,760px)] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-bg-border bg-bg-card shadow-lg animate-fade-in-scale">
        <div className="flex items-start justify-between gap-4 border-b border-bg-border bg-bg-elevated/70 px-4 py-4 sm:px-5">
          <div className="flex min-w-0 items-start gap-3">
            <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-danger/10 text-danger">
              <AlertTriangle size={18} aria-hidden />
            </div>
            <div className="min-w-0">
              <div className="flex min-w-0 flex-wrap items-center gap-2">
                <h2
                  id="failed-jobs-modal-title"
                  className="truncate text-base font-semibold text-text-primary sm:text-lg"
                >
                  {t("title")}
                </h2>
                {total > 0 && (
                  <span className="inline-flex h-6 items-center rounded-full bg-danger/10 px-2.5 text-xs font-semibold tabular-nums text-danger">
                    {t("countBadge", { count: total })}
                  </span>
                )}
              </div>
              {pageSummary && (
                <p className="mt-1 text-xs text-text-muted">{pageSummary}</p>
              )}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl p-2 text-text-muted transition-colors hover:bg-bg-card hover:text-text-primary"
            aria-label={tc("close")}
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto overflow-x-hidden bg-bg-primary/60">
          {loading && !hasItems ? (
            <div className="flex min-h-64 items-center justify-center text-text-muted">
              <Loader2 size={22} className="animate-spin" aria-hidden />
            </div>
          ) : !hasItems ? (
            <div className="flex min-h-64 flex-col items-center justify-center px-6 text-center">
              <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-bg-elevated text-text-muted">
                <AlertTriangle size={22} aria-hidden />
              </div>
              <p className="text-sm font-medium text-text-primary">{t("none")}</p>
            </div>
          ) : (
            <div className="p-2 sm:p-3">
              <div
                className="mb-2 hidden grid-cols-[24px_minmax(112px,1.2fr)_minmax(66px,0.62fr)_minmax(78px,0.74fr)_minmax(92px,0.78fr)_minmax(136px,1.08fr)_minmax(46px,0.42fr)_minmax(108px,0.82fr)_minmax(104px,0.78fr)] gap-2 px-2 text-[11px] font-semibold uppercase text-text-muted lg:grid"
              >
                <span
                  aria-hidden
                  data-checkbox-slot
                  style={{ width: "24px" }}
                />
                <span>{t("col.file")}</span>
                <span>{t("col.drive")}</span>
                <span>{t("col.task")}</span>
                <span>{t("col.provider")}</span>
                <span>{t("col.error")}</span>
                <span>{t("col.attempts")}</span>
                <span>{t("col.attemptedAt")}</span>
                <span>{t("col.actions")}</span>
              </div>
              <ul className={`space-y-2 ${loading ? "opacity-60" : ""}`}>
                {items.map((row) => {
                  const task = jobKindToTask(row.job_kind);
                  const retryKey = `${row.file_id}:${task}`;
                  const resolveKey = `${row.file_id}:${row.job_kind}:${row.provider ?? "none"}`;
                  const isRetrying = retrying === retryKey;
                  const isResolving = resolving === resolveKey;
                  const attempted = formatAttempted(row.attempted_at);
                  return (
                    <li
                      key={`${row.file_id}:${row.job_kind}:${row.provider ?? "none"}`}
                      className="grid grid-cols-[24px_minmax(0,1fr)] gap-x-3 gap-y-3 rounded-xl border border-bg-border bg-bg-card p-3 text-sm text-text-primary transition-colors hover:bg-bg-elevated/60 lg:grid-cols-[24px_minmax(112px,1.2fr)_minmax(66px,0.62fr)_minmax(78px,0.74fr)_minmax(92px,0.78fr)_minmax(136px,1.08fr)_minmax(46px,0.42fr)_minmax(108px,0.82fr)_minmax(104px,0.78fr)] lg:items-start lg:gap-x-2 lg:p-2 lg:text-xs"
                    >
                      <span
                        className="h-full"
                        data-checkbox-slot
                        style={{ width: "24px" }}
                      />
                      <span
                        className="block min-w-0 truncate font-semibold text-text-primary"
                        title={row.filename}
                      >
                        {row.filename}
                      </span>
                      <div className="col-start-2 min-w-0 text-text-muted lg:col-auto">
                        <span className="inline-flex min-w-0 max-w-full rounded-lg bg-bg-elevated px-2 py-0.5 text-[11px] lg:block lg:bg-transparent lg:px-0 lg:py-0 lg:text-xs">
                          <span className="block truncate">{row.drive}</span>
                        </span>
                      </div>
                      <div className="col-start-2 min-w-0 font-medium lg:col-auto">
                        <span className="inline-flex min-w-0 max-w-full rounded-lg bg-bg-elevated px-2 py-0.5 text-[11px] lg:block lg:bg-transparent lg:px-0 lg:py-0 lg:text-xs">
                          <span className="block truncate">{row.job_kind}</span>
                        </span>
                      </div>
                      <div className="col-start-2 min-w-0 text-text-muted lg:col-auto">
                        <span className="inline-flex min-w-0 max-w-full rounded-lg bg-bg-elevated px-2 py-0.5 text-[11px] lg:block lg:bg-transparent lg:px-0 lg:py-0 lg:text-xs">
                          <span className="block truncate">
                            {row.provider ?? "-"}
                          </span>
                        </span>
                      </div>
                      <div className="col-start-2 min-w-0 lg:col-auto">
                        <span className="inline-flex max-w-full items-center rounded-lg bg-danger/10 px-2 py-0.5 text-xs font-semibold text-danger">
                          {row.error_class ?? "-"}
                        </span>
                        <span
                          className="mt-1 block max-w-full truncate text-xs text-text-muted"
                          title={row.error_message_excerpt ?? ""}
                        >
                          {row.error_message_excerpt ?? ""}
                        </span>
                      </div>
                      <div className="col-start-2 flex items-center gap-2 text-xs text-text-muted lg:col-auto lg:block">
                        <span className="lg:hidden">{t("col.attempts")}</span>
                        <span className="font-medium tabular-nums text-text-primary">
                          {row.attempts}
                        </span>
                      </div>
                      <div
                        className="col-start-2 min-w-0 text-xs text-text-muted lg:col-auto"
                        title={row.attempted_at}
                      >
                        <span className="inline-flex max-w-full items-start gap-1.5">
                          <Clock
                            size={13}
                            className="mt-0.5 shrink-0"
                            aria-hidden
                          />
                          <span className="leading-tight tabular-nums">
                            <span className="block whitespace-nowrap">
                              {attempted.date}
                            </span>
                            {attempted.time && (
                              <span className="block whitespace-nowrap">
                                {attempted.time}
                              </span>
                            )}
                          </span>
                        </span>
                      </div>
                      <div className="col-start-2 flex flex-wrap items-center gap-2 lg:col-auto lg:flex-col lg:items-start lg:gap-1.5">
                        <button
                          type="button"
                          onClick={() => handleRetry(row)}
                          disabled={isRetrying || isResolving}
                          className="inline-flex h-7 items-center gap-1.5 rounded-xl bg-sand px-2.5 text-xs font-semibold text-text-primary transition-colors hover:bg-sand-hover disabled:opacity-50"
                          aria-label={t("retry")}
                        >
                          {isRetrying ? (
                            <Loader2
                              size={13}
                              className="animate-spin"
                              aria-hidden
                            />
                          ) : (
                            <RefreshCw size={13} aria-hidden />
                          )}
                          {t("retry")}
                        </button>
                        <button
                          type="button"
                          onClick={() => handleResolve(row)}
                          disabled={isRetrying || isResolving}
                          className="inline-flex h-7 items-center gap-1 whitespace-nowrap rounded-xl px-2 text-[11px] font-semibold text-accent-teal transition-colors hover:bg-accent-teal/10 disabled:bg-sand disabled:text-warm-silver disabled:cursor-not-allowed"
                          aria-label={t("resolve")}
                        >
                          {isResolving ? (
                            <Loader2
                              size={13}
                              className="animate-spin"
                              aria-hidden
                            />
                          ) : (
                            <CheckCircle2 size={13} aria-hidden />
                          )}
                          {t("resolve")}
                        </button>
                        <Link
                          href={`/drive/${encodeURIComponent(row.drive)}/file/${row.file_id}`}
                          className="inline-flex h-7 items-center gap-1.5 rounded-xl px-2.5 text-xs font-semibold text-accent transition-colors hover:bg-accent/10"
                          aria-label={t("details")}
                        >
                          <ExternalLink size={13} aria-hidden />
                          {t("details")}
                        </Link>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>

        {total > PAGE_LIMIT && (
          <div className="flex items-center justify-between gap-3 border-t border-bg-border bg-bg-card px-4 py-3 text-xs text-text-muted sm:px-5">
            <span>{pageSummary}</span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handlePrev}
                disabled={offset === 0 || loading}
                className="rounded-xl bg-sand px-3 py-1.5 font-semibold text-text-primary transition-colors hover:bg-sand-hover disabled:opacity-50"
              >
                {t("prev")}
              </button>
              <button
                type="button"
                onClick={handleNext}
                disabled={offset + PAGE_LIMIT >= total || loading}
                className="rounded-xl bg-sand px-3 py-1.5 font-semibold text-text-primary transition-colors hover:bg-sand-hover disabled:opacity-50"
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
