"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import {
  ChevronDown,
  ChevronRight,
  Download,
  FileText,
  RefreshCw,
} from "lucide-react";
import { MarkdownPreview } from "@/components/MarkdownPreview";
import {
  deleteDetailedSummary,
  downloadDetailedSummary,
  getDetailedSummary,
  startDetailedSummary,
} from "./api";
import type { DetailedSummaryResponse } from "./api";

interface DetailedSummarySectionProps {
  fileId: string;
  drive: string;
}

// Detailed summary generation is expensive: tens of seconds to a few
// minutes on local ollama. Poll every 3 seconds up to 10 minutes so
// the UI can surface progress without hammering the backend.
const POLL_INTERVAL_MS = 3000;
const POLL_MAX_ATTEMPTS = 200; // 200 * 3s = 10 minutes

export default function DetailedSummarySection({
  fileId,
  drive,
}: DetailedSummarySectionProps) {
  const t = useTranslations("file");
  const [data, setData] = useState<DetailedSummaryResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [working, setWorking] = useState(false);
  const [downloading, setDownloading] = useState(false);
  // Default to collapsed: detailed summaries are long (thousands of
  // characters) and would crowd the file-detail page if auto-expanded.
  const [collapsed, setCollapsed] = useState(true);

  // Track the active polling loop so stale timers from a previous file
  // don't clobber the current one. Incrementing ``pollToken`` cancels
  // any in-flight poll that was started before the switch.
  const pollTokenRef = useRef(0);

  const fetchData = useCallback(async () => {
    const result = await getDetailedSummary(fileId, drive);
    setData(result);
    setLoaded(true);
    return result;
  }, [fileId, drive]);

  useEffect(() => {
    setData(null);
    setLoaded(false);
    setWorking(false);
    setDownloading(false);
    setCollapsed(true);
    pollTokenRef.current += 1;
    void fetchData().then((result) => {
      // If we land on a ``generating`` row (e.g. user navigated away
      // and came back while the background task is still running),
      // resume polling so the UI finishes the handoff.
      if (result.status === "generating") {
        void pollUntilDone();
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileId, drive]);

  const pollUntilDone = useCallback(async () => {
    const token = ++pollTokenRef.current;
    setWorking(true);
    try {
      for (let i = 0; i < POLL_MAX_ATTEMPTS; i++) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        if (token !== pollTokenRef.current) return; // file switched
        const result = await getDetailedSummary(fileId, drive);
        if (token !== pollTokenRef.current) return;
        setData(result);
        if (
          result.status === "generated"
          || result.status === "failed"
          || !result.status
        ) {
          return;
        }
      }
    } finally {
      if (token === pollTokenRef.current) setWorking(false);
    }
  }, [fileId, drive]);

  const handleGenerate = useCallback(async () => {
    setWorking(true);
    setCollapsed(false);
    try {
      // If a prior ``generated`` / ``failed`` row still exists the
      // server returns 409 — clear it first so the new generation
      // starts from a clean slate. ``not_generated`` paths have
      // nothing to delete; the DELETE 404 is expected and ignored.
      try {
        await deleteDetailedSummary(fileId, drive);
      } catch {
        // No row to delete — proceed with generation.
      }
      await startDetailedSummary(fileId, drive);
      await pollUntilDone();
    } catch {
      // Surface as a status refresh so the `failed` error message,
      // if any, appears in the UI.
      await fetchData();
    } finally {
      setWorking(false);
    }
  }, [fileId, drive, pollUntilDone, fetchData]);

  const handleDownload = useCallback(async () => {
    setDownloading(true);
    try {
      await downloadDetailedSummary(fileId, drive);
    } catch {
      // silently fail — user can retry
    } finally {
      setDownloading(false);
    }
  }, [fileId, drive]);

  const handleToggleCollapsed = useCallback(() => {
    setCollapsed((prev) => !prev);
  }, []);

  if (!loaded) return null;

  const reason = data?.reason;
  const status = data?.status;

  // Hide entirely for unsupported types — no affordance exists that
  // would help the user.
  if (!data?.available && reason === "unsupported_type") return null;

  // Also hide when the feature is disabled (``available=false`` with
  // no ``reason`` and no ``status`` means the router returned the
  // feature-disabled shortcut). The section should stay dormant so
  // OFF drives present no UI for this capability.
  if (
    !data?.available
    && !status
    && !reason
  ) {
    return null;
  }

  if (!data?.available && reason === "insufficient_content") {
    return (
      <div className="flex items-center gap-2">
        <FileText size={14} className="text-text-muted/50" />
        <span className="text-xs text-text-muted/70">
          {t("detailedSummaryInsufficientContent", {
            defaultMessage:
              "Not enough content for a detailed summary",
          })}
        </span>
      </div>
    );
  }

  if (!data?.available && status === "generating") {
    return (
      <div className="flex items-center gap-2">
        <FileText size={14} className="text-text-muted" />
        <span className="flex items-center gap-1 text-xs text-text-muted">
          <RefreshCw size={11} className="animate-spin" />
          {t("detailedSummaryGenerating", {
            defaultMessage: "Generating detailed summary…",
          })}
        </span>
      </div>
    );
  }

  if (!data?.available && status === "failed") {
    return (
      <div>
        <div className="flex items-center gap-2">
          <FileText size={14} className="text-accent-red/70" />
          <span className="text-xs text-accent-red/80">
            {t("detailedSummaryFailed", {
              defaultMessage: "Detailed summary generation failed",
            })}
          </span>
        </div>
        {data?.error && (
          <p className="mt-1 pl-6 text-[11px] text-text-muted/80">
            {data.error}
          </p>
        )}
        <button
          onClick={handleGenerate}
          disabled={working}
          className="mt-2 flex items-center gap-1 rounded px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          <RefreshCw size={11} className={working ? "animate-spin" : ""} />
          {t("detailedSummaryRetry", { defaultMessage: "Retry" })}
        </button>
      </div>
    );
  }

  if (!data?.available) {
    // reason === "not_generated" (or null) — present the generate CTA.
    return (
      <div>
        <div className="flex items-center gap-2">
          <FileText size={14} className="text-text-muted" />
          <button
            onClick={handleGenerate}
            disabled={working}
            className="flex items-center gap-1 rounded px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
          >
            <RefreshCw size={11} className={working ? "animate-spin" : ""} />
            {working
              ? t("detailedSummaryGenerating", {
                  defaultMessage: "Generating detailed summary…",
                })
              : t("detailedSummaryGenerate", {
                  defaultMessage: "Generate detailed summary",
                })}
          </button>
        </div>
      </div>
    );
  }

  // available=true, status="generated" — render the Markdown body.
  return (
    <div>
      <div className={`flex items-center gap-2 ${collapsed ? "" : "mb-2"}`}>
        <button
          onClick={handleToggleCollapsed}
          aria-expanded={!collapsed}
          aria-label={
            collapsed
              ? t("detailedSummaryShow", { defaultMessage: "Expand" })
              : t("detailedSummaryHide", { defaultMessage: "Collapse" })
          }
          className="flex items-center gap-2 rounded text-text-muted transition-colors hover:text-text-primary"
        >
          {collapsed ? (
            <ChevronRight size={14} className="text-text-muted" />
          ) : (
            <ChevronDown size={14} className="text-text-muted" />
          )}
          <FileText size={14} className="text-accent-teal" />
          <h2 className="text-sm font-semibold">
            {t("detailedSummaryTitle", {
              defaultMessage: "AI Detailed Summary",
            })}
          </h2>
        </button>
        {!collapsed && data.model && (
          <span className="text-[10px] text-text-muted/50">{data.model}</span>
        )}
        {!collapsed && data.was_truncated && (
          <span className="text-[10px] text-text-muted/70">
            {t("detailedSummaryTruncatedNote", {
              defaultMessage: "(excerpts from long content)",
            })}
          </span>
        )}
      </div>

      {!collapsed && (
        <>
          {data.detailed_summary && (
            <MarkdownPreview
              source={data.detailed_summary}
              chrome={false}
              mermaid={false}
            />
          )}
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              onClick={handleDownload}
              disabled={downloading}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <Download size={11} />
              {t("detailedSummaryDownload", {
                defaultMessage: "Download as Markdown",
              })}
            </button>
            <button
              onClick={handleGenerate}
              disabled={working}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <RefreshCw size={11} className={working ? "animate-spin" : ""} />
              {working
                ? t("detailedSummaryGenerating", {
                    defaultMessage: "Generating detailed summary…",
                  })
                : t("detailedSummaryRegenerate", {
                    defaultMessage: "Regenerate",
                  })}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
