"use client";

import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import {
  BookOpen,
  ChevronDown,
  ChevronRight,
  Pencil,
  RefreshCw,
  RotateCcw,
  Save,
  X,
} from "lucide-react";
import {
  editSummary,
  getSummary,
  regenerateSummary,
  revertSummary,
} from "./api";
import type { SummaryResponse } from "./api";
import { useOfferFileAiAction } from "./fileAiActions";

interface SummarySectionProps {
  fileId: string;
  drive: string;
}

// Maximum polling attempts after a regenerate call. Each attempt waits
// POLL_INTERVAL_MS — LLM generation for long transcripts can take a while.
const POLL_MAX_ATTEMPTS = 20;
const POLL_INTERVAL_MS = 2000;

// Matches SummaryEditRequest validation on the backend.
const SHORT_MAX = 200;
const LONG_MAX = 4000;

export default function SummarySection({ fileId, drive }: SummarySectionProps) {
  const t = useTranslations("file");
  const [data, setData] = useState<SummaryResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  // Collapse is a client-only toggle — the chevron in the header flips
  // between expanded (body visible) and collapsed (header only, body
  // hidden). No API call, no persistence: re-visiting the file starts
  // expanded again. The header stays visible in both states so there's
  // always an anchor to re-open the section.
  const [collapsed, setCollapsed] = useState(false);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [reverting, setReverting] = useState(false);
  const [draftShort, setDraftShort] = useState("");
  const [draftLong, setDraftLong] = useState("");

  const fetchData = useCallback(async () => {
    const result = await getSummary(fileId, drive);
    setData(result);
    setLoaded(true);
  }, [fileId, drive]);

  useEffect(() => {
    setData(null);
    setLoaded(false);
    setRegenerating(false);
    setCollapsed(false);
    setEditing(false);
    setSaving(false);
    setReverting(false);
    fetchData();
  }, [fileId, fetchData]);

  const handleRegenerate = useCallback(async () => {
    setRegenerating(true);
    setCollapsed(false);
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

  const handleToggleCollapsed = useCallback(() => {
    setCollapsed((prev) => !prev);
    // Leaving edit mode on collapse keeps the invariant "body hidden =
    // nothing interactive below the header"; re-opening the section
    // brings the user back to the read-only view.
    setEditing(false);
  }, []);

  const handleStartEdit = useCallback(() => {
    setDraftShort(data?.short_summary ?? "");
    setDraftLong(data?.long_summary ?? "");
    setEditing(true);
  }, [data]);

  const handleCancelEdit = useCallback(() => {
    setEditing(false);
    setDraftShort("");
    setDraftLong("");
  }, []);

  const handleSaveEdit = useCallback(async () => {
    const short = draftShort.trim();
    const long = draftLong.trim();
    if (!short || !long) return;
    if (short.length > SHORT_MAX || long.length > LONG_MAX) return;
    setSaving(true);
    try {
      const result = await editSummary(fileId, drive, {
        short_summary: short,
        long_summary: long,
      });
      setData(result);
      setEditing(false);
    } catch {
      // silently fail — leave the edit mode open so the user can retry.
    } finally {
      setSaving(false);
    }
  }, [fileId, drive, draftShort, draftLong]);

  const handleRevert = useCallback(async () => {
    setReverting(true);
    try {
      const result = await revertSummary(fileId, drive);
      setData(result);
    } catch {
      // silently fail
    } finally {
      setReverting(false);
    }
  }, [fileId, drive]);

  // The generate offer moves to the action row's "AI" menu; what stays
  // here are the states that report something real — too little text to
  // work with, a run in flight, a failure. "You could make one" is not
  // one of those, and it was the only thing most files ever showed.
  useOfferFileAiAction({
    fileId,
    kind: "summary",
    labelKey: "summaryGenerate",
    active: loaded
      && !data?.available
      && data?.reason !== "unsupported_type"
      && data?.reason !== "insufficient_content",
    busy: regenerating,
    run: handleRegenerate,
  });

  if (!loaded) return null;

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

    // Ready to generate — the offer above carries it. Covers
    // reason="not_generated" as well as feature-disabled/null-reason
    // legacy paths.
    return null;
  }

  const shortInvalid =
    draftShort.trim().length === 0 || draftShort.length > SHORT_MAX;
  const longInvalid =
    draftLong.trim().length === 0 || draftLong.length > LONG_MAX;

  return (
    <div>
      <div className={`flex items-center gap-2 ${collapsed ? "" : "mb-2"}`}>
        <button
          onClick={handleToggleCollapsed}
          aria-expanded={!collapsed}
          aria-label={
            collapsed
              ? t("summaryShow", { defaultMessage: "Show summary" })
              : t("summaryHide", { defaultMessage: "Hide summary" })
          }
          className="flex items-center gap-2 rounded-lg text-text-muted transition-colors hover:text-text-primary"
        >
          {collapsed ? (
            <ChevronRight size={14} className="text-text-muted" />
          ) : (
            <ChevronDown size={14} className="text-text-muted" />
          )}
          <BookOpen size={14} className="text-accent-teal" />
          <h2 className="text-sm font-semibold">
            {t("summaryTitle", { defaultMessage: "AI Summary" })}
          </h2>
        </button>
        {!collapsed && data.model && !data.edited_at && (
          <span className="text-[10px] text-text-muted/50">{data.model}</span>
        )}
        {!collapsed && data.was_truncated && !data.edited_at && (
          <span className="text-[10px] text-text-muted/70">
            {t("summaryTruncatedNote", {
              defaultMessage: "(excerpts from long content)",
            })}
          </span>
        )}
        {data.edited_at && (
          <span className="rounded-lg bg-bg-elevated px-1.5 py-0.5 text-[10px] text-text-muted/80">
            {t("summaryEditedBadge", { defaultMessage: "Edited" })}
          </span>
        )}
      </div>

      {collapsed ? null : editing ? (
        <>
          {/* Styled to mirror the read-only <p> tags exactly: same font
              size, weight, color, spacing. `field-sizing: content` makes
              the textarea auto-grow so the layout stays stable between
              read and edit modes. Border / padding / background /
              default textarea min-height are stripped. */}
          <textarea
            id="summary-edit-short"
            aria-label={t("summaryShortLabel", { defaultMessage: "Short summary" })}
            value={draftShort}
            onChange={(e) => setDraftShort(e.target.value)}
            maxLength={SHORT_MAX}
            rows={1}
            className="mb-2 block w-full resize-none border-0 bg-transparent p-0 text-sm font-medium text-text-primary outline-none ring-0 focus:ring-0 [field-sizing:content]"
          />
          <textarea
            id="summary-edit-long"
            aria-label={t("summaryLongLabel", { defaultMessage: "Detailed summary" })}
            value={draftLong}
            onChange={(e) => setDraftLong(e.target.value)}
            maxLength={LONG_MAX}
            rows={1}
            className="block w-full resize-none whitespace-pre-wrap border-0 bg-transparent p-0 text-sm leading-relaxed text-text-muted outline-none ring-0 focus:ring-0 [field-sizing:content]"
          />
          <div className="mt-3 flex items-center gap-2">
            <button
              onClick={handleSaveEdit}
              disabled={saving || shortInvalid || longInvalid}
              className="flex items-center gap-1 rounded-lg bg-accent-teal px-2 py-1 text-[11px] text-white transition-colors hover:bg-accent-teal/90 disabled:opacity-50"
            >
              {saving ? (
                <RefreshCw size={11} className="animate-spin" />
              ) : (
                <Save size={11} />
              )}
              {t("summarySave", { defaultMessage: "Save" })}
            </button>
            <button
              onClick={handleCancelEdit}
              disabled={saving}
              className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <X size={11} />
              {t("summaryCancel", { defaultMessage: "Cancel" })}
            </button>
          </div>
        </>
      ) : (
        <>
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

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              onClick={handleStartEdit}
              className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
            >
              <Pencil size={11} />
              {t("summaryEdit", { defaultMessage: "Edit" })}
            </button>
            {data.has_original && (
              <button
                onClick={handleRevert}
                disabled={reverting}
                className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
              >
                {reverting ? (
                  <RefreshCw size={11} className="animate-spin" />
                ) : (
                  <RotateCcw size={11} />
                )}
                {t("summaryRevert", {
                  defaultMessage: "Revert to AI version",
                })}
              </button>
            )}
            <button
              onClick={handleRegenerate}
              disabled={regenerating}
              className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <RefreshCw size={11} className={regenerating ? "animate-spin" : ""} />
              {regenerating
                ? t("summaryGenerating", { defaultMessage: "Generating summary..." })
                : t("summaryRegenerate", { defaultMessage: "Regenerate" })}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
