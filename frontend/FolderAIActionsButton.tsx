"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { BookOpen, ChevronDown, ImageIcon, Sparkles } from "lucide-react";
import {
  batchSuggestedTags,
  batchSummaries,
  generateFolderVisualDescription,
} from "./api";
import type { FolderVisualDescriptionTooManyError } from "./api";

interface FolderAIActionsButtonProps {
  fileIds: string[];
  drive: string;
  // ``path`` is still passed by the slot host (FolderToolbar) for
  // legacy reasons but is no longer used — backend selects by id.
  path?: string;
}

type Pending = "tags" | "summaries" | "vision" | null;

export default function FolderAIActionsButton({
  fileIds,
  drive,
}: FolderAIActionsButtonProps) {
  const t = useTranslations("file");
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState<Pending>(null);
  const [resultMessage, setResultMessage] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  const showResult = useCallback((msg: string, ms = 5000) => {
    setResultMessage(msg);
    setTimeout(() => setResultMessage(null), ms);
  }, []);

  const handleTags = useCallback(async () => {
    if (fileIds.length === 0 || pending) return;
    setOpen(false);
    setPending("tags");
    try {
      const result = await batchSuggestedTags(fileIds, drive);
      if (result.queued === 0 && result.skipped > 0) {
        showResult(t("tagsBatchEmpty"));
      } else {
        showResult(
          t("tagsBatchQueued", { queued: result.queued, skipped: result.skipped })
        );
      }
    } catch {
      // non-critical
    } finally {
      setPending(null);
    }
  }, [fileIds, drive, pending, t, showResult]);

  const handleSummaries = useCallback(async () => {
    if (fileIds.length === 0 || pending) return;
    setOpen(false);
    setPending("summaries");
    try {
      const result = await batchSummaries(fileIds, drive);
      if (result.queued === 0 && result.skipped > 0) {
        showResult(t("summariesBatchEmpty"));
      } else {
        showResult(
          t("summariesBatchQueued", {
            queued: result.queued,
            skipped: result.skipped,
          })
        );
      }
    } catch {
      // non-critical
    } finally {
      setPending(null);
    }
  }, [fileIds, drive, pending, t, showResult]);

  const handleVision = useCallback(async () => {
    if (fileIds.length === 0 || pending) return;
    setOpen(false);
    const confirmed = window.confirm(
      t("visionFolderConfirm", {
        defaultMessage:
          "Generate AI descriptions for every image in this folder? This may incur LLM costs.",
      }),
    );
    if (!confirmed) return;
    setPending("vision");
    try {
      const result = await generateFolderVisualDescription(drive, fileIds);
      showResult(
        t("visionFolderQueued", {
          queued: result.queued,
          defaultMessage: "{queued} images queued",
        }),
      );
    } catch (e) {
      const info = (e as { info?: FolderVisualDescriptionTooManyError }).info;
      if (info?.kind === "too_many_files") {
        showResult(
          t("visionFolderTooMany", {
            max: info.max,
            requested: info.requested,
            defaultMessage:
              "Too many files ({requested}). Max per batch: {max}.",
          }),
          8000,
        );
      } else {
        showResult(
          t("visionFolderError", {
            defaultMessage: "Failed to queue folder — please retry.",
          }),
          8000,
        );
      }
    } finally {
      setPending(null);
    }
  }, [drive, fileIds, pending, t, showResult]);

  if (fileIds.length === 0) return null;

  const busy = pending !== null;

  return (
    <div ref={ref} className="relative flex items-center gap-2">
      <button
        type="button"
        onClick={() => setOpen((s) => !s)}
        disabled={busy}
        className="flex items-center gap-1.5 rounded-2xl border border-bg-border bg-bg-card px-3 py-2 text-sm font-medium text-text-primary transition-colors hover:bg-bg-elevated disabled:opacity-50"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t("aiFolderActions", { defaultMessage: "AI" })}
      >
        <Sparkles size={16} className={busy ? "animate-pulse" : ""} />
        <span>{t("aiFolderActions", { defaultMessage: "AI" })}</span>
        <ChevronDown size={14} className="text-text-muted" />
      </button>

      {open && (
        <div
          role="menu"
          className="fixed inset-x-2 bottom-4 z-40 max-h-[60vh] overflow-y-auto rounded-2xl border border-bg-border bg-bg-primary py-1 shadow-xl animate-fade-in-scale sm:absolute sm:inset-x-auto sm:bottom-auto sm:left-0 sm:top-full sm:mt-1 sm:max-h-none sm:min-w-[240px] sm:overflow-visible sm:origin-top-left"
        >
          <button
            type="button"
            role="menuitem"
            onClick={handleTags}
            className="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm text-text-primary transition-colors hover:bg-bg-elevated"
          >
            <Sparkles size={16} className="flex-shrink-0 text-text-muted" />
            <span className="flex-1">{t("generateFolderTags")}</span>
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={handleSummaries}
            className="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm text-text-primary transition-colors hover:bg-bg-elevated"
          >
            <BookOpen size={16} className="flex-shrink-0 text-text-muted" />
            <span className="flex-1">{t("generateFolderSummaries")}</span>
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={handleVision}
            className="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm text-text-primary transition-colors hover:bg-bg-elevated"
          >
            <ImageIcon size={16} className="flex-shrink-0 text-text-muted" />
            <span className="flex-1">
              {t("visionFolderButton", {
                defaultMessage: "Generate AI descriptions for folder images",
              })}
            </span>
          </button>
        </div>
      )}

      {resultMessage && (
        <span className="text-xs text-text-muted animate-fade-in">
          {resultMessage}
        </span>
      )}
    </div>
  );
}
