"use client";

import { useCallback, useState } from "react";
import { useTranslations } from "next-intl";
import { Sparkles } from "lucide-react";
import { refineFolderTranscripts } from "./api";

interface FolderRefineButtonProps {
  drive: string;
  // Still received from the slot host (FolderToolbar) but unused —
  // the backend selects files by id now, not path-prefix.
  path?: string;
  fileIds: string[];
}

export default function FolderRefineButton({
  drive,
  fileIds,
}: FolderRefineButtonProps) {
  const t = useTranslations("file");
  const [loading, setLoading] = useState(false);
  const [resultMessage, setResultMessage] = useState<string | null>(null);
  const [skippedMessage, setSkippedMessage] = useState<string | null>(null);

  const handleClick = useCallback(async () => {
    if (fileIds.length === 0 || loading) return;
    setLoading(true);
    setResultMessage(null);
    setSkippedMessage(null);
    try {
      const result = await refineFolderTranscripts(drive, fileIds);
      const queued = (result as { queued?: number } | null)?.queued ?? 0;
      const skipped =
        (result as { skipped_corrected?: string[] } | null)?.skipped_corrected
          ?.length ?? 0;
      setResultMessage(t("refineBatchQueued", { queued }));
      if (skipped > 0) {
        setSkippedMessage(
          t("refineBatchSkippedCorrected", { count: skipped }),
        );
      }
      setTimeout(() => {
        setResultMessage(null);
        setSkippedMessage(null);
      }, 5000);
    } catch {
      // non-critical — user can retry
    } finally {
      setLoading(false);
    }
  }, [drive, fileIds, loading, t]);

  if (fileIds.length === 0) return null;

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleClick}
        disabled={loading}
        className="flex items-center gap-1.5 rounded-lg border border-bg-border bg-bg-card px-3 py-2 text-sm font-medium text-text-primary transition-colors hover:bg-bg-elevated disabled:opacity-50"
        aria-label={t("generateFolderTranscriptRefine")}
      >
        <Sparkles size={16} className={loading ? "animate-pulse" : ""} />
        <span className="hidden sm:inline">
          {t("generateFolderTranscriptRefine")}
        </span>
      </button>
      {resultMessage && (
        <span className="text-xs text-text-muted animate-fade-in">
          {resultMessage}
        </span>
      )}
      {skippedMessage && (
        <span className="text-xs text-text-muted animate-fade-in">
          {skippedMessage}
        </span>
      )}
    </div>
  );
}
