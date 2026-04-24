"use client";

import { useCallback, useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { Check, CheckCheck, RefreshCw, Sparkles, X } from "lucide-react";
import { getSuggestedTags, dismissSuggestedTags, regenerateSuggestedTags } from "./api";
import type { SuggestedTagsResponse } from "./api";
import { fetchJSON } from "@/lib/api";
import { saveFileTags } from "@/lib/tags";
import type { FileItem } from "@/types";

interface SuggestedTagsSectionProps {
  fileId: string;
  drive: string;
}

/**
 * Append ``newTags`` to the file's current tags and persist. Routes
 * through ``saveFileTags`` so ``.md`` files write their frontmatter
 * (spec ``docs/superpowers/specs/2026-04-24-knowledge-tag-unification.md``
 * §D3/D9). Without this dispatch, approving a suggested tag on a ``.md``
 * would write ``File.tags`` directly and get overwritten on the next
 * scanner pass.
 */
async function mergeAndSaveTags(
  file: FileItem,
  newTags: string[],
): Promise<void> {
  const merged = [...new Set([...file.tags, ...newTags])];
  await saveFileTags(file, merged);
}

async function getFileData(fileId: string): Promise<FileItem> {
  return fetchJSON<FileItem>(`/api/files/${fileId}`);
}

export default function SuggestedTagsSection({ fileId, drive }: SuggestedTagsSectionProps) {
  const t = useTranslations("file");
  const [data, setData] = useState<SuggestedTagsResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [accepting, setAccepting] = useState<Set<string>>(new Set());
  const [acceptedTags, setAcceptedTags] = useState<Set<string>>(new Set());
  const [dismissing, setDismissing] = useState(false);
  const [regenerating, setRegenerating] = useState(false);
  const [acceptingAll, setAcceptingAll] = useState(false);
  const [hidden, setHidden] = useState(false);

  const fetchData = useCallback(async () => {
    const result = await getSuggestedTags(fileId, drive);
    setData(result);
    setLoaded(true);
  }, [fileId, drive]);

  useEffect(() => {
    setData(null);
    setLoaded(false);
    setAccepting(new Set());
    setAcceptedTags(new Set());
    setHidden(false);
    setRegenerating(false);
    fetchData();
  }, [fileId, fetchData]);

  const handleAcceptTag = useCallback(async (tag: string) => {
    setAccepting((prev) => new Set([...prev, tag]));
    try {
      const file = await getFileData(fileId);
      await mergeAndSaveTags(file, [tag]);
      setAcceptedTags((prev) => new Set([...prev, tag]));
    } catch {
      // silently fail
    } finally {
      setAccepting((prev) => {
        const next = new Set(prev);
        next.delete(tag);
        return next;
      });
    }
  }, [fileId]);

  const handleAcceptAll = useCallback(async () => {
    if (!data?.tags) return;
    const pendingTags = data.tags.filter((tag) => !acceptedTags.has(tag));
    if (pendingTags.length === 0) return;

    setAcceptingAll(true);
    try {
      const file = await getFileData(fileId);
      await mergeAndSaveTags(file, pendingTags);
      setAcceptedTags((prev) => new Set([...prev, ...pendingTags]));
    } catch {
      // silently fail
    } finally {
      setAcceptingAll(false);
    }
  }, [fileId, data, acceptedTags]);

  const handleDismiss = useCallback(async () => {
    setDismissing(true);
    try {
      await dismissSuggestedTags(fileId, drive);
      setHidden(true);
    } catch {
      // silently fail
    } finally {
      setDismissing(false);
    }
  }, [fileId, drive]);

  const handleRegenerate = useCallback(async () => {
    setRegenerating(true);
    setAcceptedTags(new Set());
    setHidden(false);
    try {
      await regenerateSuggestedTags(fileId, drive);
      // Poll for results — LLM processing takes a few seconds
      const maxAttempts = 15;
      for (let i = 0; i < maxAttempts; i++) {
        await new Promise((r) => setTimeout(r, 2000));
        const result = await getSuggestedTags(fileId, drive);
        if (result.available && result.tags && result.tags.length > 0) {
          setData(result);
          break;
        }
      }
    } catch {
      // silently fail
    } finally {
      setRegenerating(false);
    }
  }, [fileId, drive]);

  if (!loaded) return null;

  // Show compact regenerate-only UI when no pending tags to display
  const hasPendingTags = data?.available
    && data.tags
    && data.tags.length > 0
    && data.status !== "accepted"
    && !hidden
    && data.tags.some((tag) => !acceptedTags.has(tag));

  if (!hasPendingTags) {
    // Show regenerate button for dismissed, accepted, or not-yet-generated files
    return (
      <div>
        <div className="flex items-center gap-2">
          <Sparkles size={14} className="text-text-muted" />
          <button
            onClick={handleRegenerate}
            disabled={regenerating}
            className="flex items-center gap-1 rounded px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
          >
            <RefreshCw size={11} className={regenerating ? "animate-spin" : ""} />
            {regenerating
              ? t("regeneratingTags", { defaultMessage: "Generating..." })
              : t("generateTags", { defaultMessage: "Generate AI tags" })}
          </button>
        </div>
      </div>
    );
  }

  const pendingTags = (data?.tags ?? []).filter((tag) => !acceptedTags.has(tag));

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <Sparkles size={14} className="text-accent-amber" />
        <h2 className="text-sm font-semibold text-text-muted">
          {t("suggestedTags", { defaultMessage: "AI Suggested Tags" })}
        </h2>
        {data.model && (
          <span className="text-[10px] text-text-muted/50">{data.model}</span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {(data?.tags ?? []).map((tag) => {
          const isAccepted = acceptedTags.has(tag);
          const isAccepting = accepting.has(tag);

          if (isAccepted) {
            return (
              <span
                key={tag}
                className="flex items-center gap-1 rounded-full bg-accent-teal/15 px-2.5 py-1 text-xs font-medium text-accent-teal"
              >
                <Check size={12} />
                {tag}
              </span>
            );
          }

          return (
            <span
              key={tag}
              className="group flex items-center gap-1 rounded-full border border-dashed border-accent-amber/40 bg-accent-amber/8 px-2.5 py-1 text-xs font-medium text-accent-amber"
            >
              {tag}
              <button
                onClick={() => handleAcceptTag(tag)}
                disabled={isAccepting}
                className="rounded-full p-0.5 text-accent-amber/70 transition-colors hover:bg-accent-amber/20 hover:text-accent-amber disabled:opacity-50"
                aria-label={t("acceptTag", { defaultMessage: "Add tag {tag}", tag })}
              >
                {isAccepting ? (
                  <RefreshCw size={12} className="animate-spin" />
                ) : (
                  <Check size={12} />
                )}
              </button>
            </span>
          );
        })}
      </div>

      <div className="mt-2 flex items-center gap-2">
        <button
          onClick={handleAcceptAll}
          disabled={acceptingAll}
          className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          {acceptingAll ? (
            <RefreshCw size={11} className="animate-spin" />
          ) : (
            <CheckCheck size={11} />
          )}
          {t("acceptAllTags", { defaultMessage: "Accept all" })}
        </button>
        <button
          onClick={handleDismiss}
          disabled={dismissing}
          className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          {dismissing ? (
            <RefreshCw size={11} className="animate-spin" />
          ) : (
            <X size={11} />
          )}
          {t("dismissTags", { defaultMessage: "Dismiss" })}
        </button>
        <button
          onClick={handleRegenerate}
          disabled={regenerating}
          className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          <RefreshCw size={11} className={regenerating ? "animate-spin" : ""} />
          {t("regenerateTags", { defaultMessage: "Regenerate" })}
        </button>
      </div>
    </div>
  );
}
