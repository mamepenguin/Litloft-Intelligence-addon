"use client";

import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { BookOpen, EyeOff, RefreshCw } from "lucide-react";
import { getSummary, hideSummary, regenerateSummary } from "./api";
import type { SummaryResponse } from "./api";

interface SummarySectionProps {
  fileId: string;
  drive: string;
}

// Maximum polling attempts after a regenerate call. Each attempt waits
// POLL_INTERVAL_MS — LLM generation for long transcripts can take a while.
const POLL_MAX_ATTEMPTS = 20;
const POLL_INTERVAL_MS = 2000;

export default function SummarySection({ fileId, drive }: SummarySectionProps) {
  const t = useTranslations("file");
  const [data, setData] = useState<SummaryResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [hiding, setHiding] = useState(false);
  const [hidden, setHidden] = useState(false);

  const fetchData = useCallback(async () => {
    const result = await getSummary(fileId, drive);
    setData(result);
    setLoaded(true);
  }, [fileId, drive]);

  useEffect(() => {
    setData(null);
    setLoaded(false);
    setRegenerating(false);
    setHiding(false);
    setHidden(false);
    fetchData();
  }, [fileId, fetchData]);

  const handleRegenerate = useCallback(async () => {
    setRegenerating(true);
    setHidden(false);
    try {
      await regenerateSummary(fileId, drive);
      // Poll for results — LLM processing takes a few seconds.
      for (let i = 0; i < POLL_MAX_ATTEMPTS; i++) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        const result = await getSummary(fileId, drive);
        if (result.available && result.long_summary) {
          setData(result);
          break;
        }
      }
    } catch {
      // silently fail — user can retry
    } finally {
      setRegenerating(false);
    }
  }, [fileId, drive]);

  const handleHide = useCallback(async () => {
    setHiding(true);
    try {
      await hideSummary(fileId, drive);
      setHidden(true);
    } catch {
      // silently fail
    } finally {
      setHiding(false);
    }
  }, [fileId, drive]);

  if (!loaded) return null;

  // Section stays hidden after a user actively hides it
  if (hidden) return null;

  if (!data?.available) {
    // File type is not summarizable (image, archive, etc.) — hide the
    // section entirely so the UI doesn't offer a useless button.
    if (data?.reason === "unsupported_type") return null;

    // File exists but the transcript / extracted text is below the
    // min_context_chars threshold. Offering a generate button would
    // just lead to a silent skip, so show an informative note instead.
    if (data?.reason === "insufficient_content") {
      return (
        <div className="flex items-center gap-2">
          <BookOpen size={14} className="text-text-muted/50" />
          <span className="text-xs text-text-muted/70">
            {t("summaryInsufficientContent", {
              defaultMessage: "Not enough content to summarize",
            })}
          </span>
        </div>
      );
    }

    // Ready to generate — show the button. Covers reason="not_generated"
    // as well as feature-disabled/null-reason legacy paths.
    return (
      <div>
        <div className="flex items-center gap-2">
          <BookOpen size={14} className="text-text-muted" />
          <button
            onClick={handleRegenerate}
            disabled={regenerating}
            className="flex items-center gap-1 rounded px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
          >
            <RefreshCw size={11} className={regenerating ? "animate-spin" : ""} />
            {regenerating
              ? t("summaryGenerating", { defaultMessage: "Generating summary..." })
              : t("summaryGenerate", { defaultMessage: "Generate AI summary" })}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <BookOpen size={14} className="text-accent-teal" />
        <h2 className="text-sm font-semibold text-text-muted">
          {t("summaryTitle", { defaultMessage: "AI Summary" })}
        </h2>
        {data.model && (
          <span className="text-[10px] text-text-muted/50">{data.model}</span>
        )}
        {data.was_truncated && (
          <span className="text-[10px] text-text-muted/70">
            {t("summaryTruncatedNote", {
              defaultMessage: "(excerpts from long content)",
            })}
          </span>
        )}
      </div>

      {data.short_summary && (
        <p className="mb-2 text-sm font-medium text-text-primary">
          {data.short_summary}
        </p>
      )}

      {data.long_summary && (
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-text-muted">
          {data.long_summary}
        </p>
      )}

      <div className="mt-3 flex items-center gap-2">
        <button
          onClick={handleRegenerate}
          disabled={regenerating}
          className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          <RefreshCw size={11} className={regenerating ? "animate-spin" : ""} />
          {regenerating
            ? t("summaryGenerating", { defaultMessage: "Generating summary..." })
            : t("summaryRegenerate", { defaultMessage: "Regenerate" })}
        </button>
        <button
          onClick={handleHide}
          disabled={hiding}
          className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          {hiding ? (
            <RefreshCw size={11} className="animate-spin" />
          ) : (
            <EyeOff size={11} />
          )}
          {t("summaryHide", { defaultMessage: "Hide" })}
        </button>
      </div>
    </div>
  );
}
