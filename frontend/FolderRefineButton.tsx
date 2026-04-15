"use client";

import { useCallback, useState } from "react";
import { useTranslations } from "next-intl";
import { Sparkles } from "lucide-react";
import { refineFolderTranscripts } from "./api";

interface FolderRefineButtonProps {
  drive: string;
  path: string;
  fileIds: string[];
}

export default function FolderRefineButton({
  drive,
  path,
  fileIds,
}: FolderRefineButtonProps) {
  const t = useTranslations("file");
  const [loading, setLoading] = useState(false);
  const [resultMessage, setResultMessage] = useState<string | null>(null);

  const handleClick = useCallback(async () => {
    if (fileIds.length === 0 || loading) return;
    setLoading(true);
    setResultMessage(null);
    try {
      const result = await refineFolderTranscripts(drive, path);
      const queued = (result as { queued?: number } | null)?.queued ?? 0;
      setResultMessage(
        t("refineBatchQueued", { queued }),
      );
      setTimeout(() => setResultMessage(null), 5000);
    } catch {
      // non-critical — user can retry
    } finally {
      setLoading(false);
    }
  }, [drive, path, fileIds.length, loading, t]);

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
    </div>
  );
}
