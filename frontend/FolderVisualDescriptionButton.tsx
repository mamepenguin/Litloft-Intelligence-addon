"use client";

import { useCallback, useState } from "react";
import { useTranslations } from "next-intl";
import { ImageIcon } from "lucide-react";
import { generateFolderVisualDescription } from "./api";
import type { FolderVisualDescriptionTooManyError } from "./api";

interface FolderVisualDescriptionButtonProps {
  drive: string;
  path: string;
  fileIds: string[];
}

export default function FolderVisualDescriptionButton({
  drive,
  path,
  fileIds,
}: FolderVisualDescriptionButtonProps) {
  const t = useTranslations("file");
  const [loading, setLoading] = useState(false);
  const [resultMessage, setResultMessage] = useState<string | null>(null);

  const handleClick = useCallback(async () => {
    if (fileIds.length === 0 || loading) return;
    const confirmed = window.confirm(
      t("visionFolderConfirm", {
        defaultMessage:
          "Generate AI descriptions for every image in this folder? This may incur LLM costs.",
      }),
    );
    if (!confirmed) return;
    setLoading(true);
    setResultMessage(null);
    try {
      const result = await generateFolderVisualDescription(drive, path);
      setResultMessage(
        t("visionFolderQueued", {
          queued: result.queued,
          defaultMessage: "{queued} images queued",
        }),
      );
      setTimeout(() => setResultMessage(null), 5000);
    } catch (e) {
      // Surface the 413 cap clearly so the operator can shrink the
      // selection rather than silently doing nothing.
      const info = (e as { info?: FolderVisualDescriptionTooManyError }).info;
      if (info?.kind === "too_many_files") {
        setResultMessage(
          t("visionFolderTooMany", {
            max: info.max,
            requested: info.requested,
            defaultMessage:
              "Too many files ({requested}). Max per batch: {max}.",
          }),
        );
      } else {
        setResultMessage(
          t("visionFolderError", {
            defaultMessage: "Failed to queue folder — please retry.",
          }),
        );
      }
      setTimeout(() => setResultMessage(null), 8000);
    } finally {
      setLoading(false);
    }
  }, [drive, path, fileIds.length, loading, t]);

  // Hide the button when the folder is empty — matches the FolderSummariesButton
  // pattern. A drive-level policy gate happens server-side (404), so showing
  // the button for folders with no images is a minor UX noise we accept.
  if (fileIds.length === 0) return null;

  return (
    <div className="flex items-center gap-2">
      <button
        onClick={handleClick}
        disabled={loading}
        className="flex items-center gap-1.5 rounded-lg border border-bg-border bg-bg-card px-3 py-2 text-sm font-medium text-text-primary transition-colors hover:bg-bg-elevated disabled:opacity-50"
        aria-label={t("visionFolderButton", {
          defaultMessage: "Generate AI descriptions for folder images",
        })}
      >
        <ImageIcon size={16} className={loading ? "animate-pulse" : ""} />
        <span className="hidden sm:inline">
          {t("visionFolderButton", {
            defaultMessage: "Generate AI descriptions for folder images",
          })}
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
