"use client";

import { useCallback, useState } from "react";
import { useTranslations } from "next-intl";
import { BookOpen } from "lucide-react";
import { batchSummaries } from "./api";

interface FolderSummariesButtonProps {
  fileIds: string[];
  drive: string;
}

export default function FolderSummariesButton({ fileIds, drive }: FolderSummariesButtonProps) {
  const t = useTranslations("file");
  const [loading, setLoading] = useState(false);
  const [resultMessage, setResultMessage] = useState<string | null>(null);

  const handleClick = useCallback(async () => {
    if (fileIds.length === 0 || loading) return;
    setLoading(true);
    setResultMessage(null);
    try {
      const result = await batchSummaries(fileIds, drive);
      if (result.queued === 0 && result.skipped > 0) {
        setResultMessage(t("summariesBatchEmpty"));
      } else {
        setResultMessage(
          t("summariesBatchQueued", {
            queued: result.queued,
            skipped: result.skipped,
          })
        );
      }
      setTimeout(() => setResultMessage(null), 5000);
    } catch {
      // error handled silently — addon API failure is non-critical
    } finally {
      setLoading(false);
    }
  }, [fileIds, drive, loading, t]);

  if (fileIds.length === 0) return null;

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleClick}
        disabled={loading}
        className="flex items-center gap-1.5 rounded-lg border border-bg-border bg-bg-card px-3 py-2 text-sm font-medium text-text-primary transition-colors hover:bg-bg-elevated disabled:opacity-50"
        aria-label={t("generateFolderSummaries")}
      >
        <BookOpen size={16} className={loading ? "animate-pulse" : ""} />
        <span className="hidden sm:inline">{t("generateFolderSummaries")}</span>
      </button>
      {resultMessage && (
        <span className="text-xs text-text-muted animate-fade-in">
          {resultMessage}
        </span>
      )}
    </div>
  );
}
