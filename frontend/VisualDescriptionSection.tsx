"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import {
  AlertTriangle,
  ImageIcon,
  Loader2,
  RefreshCw,
  Settings,
} from "lucide-react";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { getFile } from "@/lib/api";
import type { FileItem } from "@/types";
import {
  deleteVisualDescription,
  generateVisualDescription,
  getVisualDescription,
} from "./api";
import type {
  VisualDescriptionResponse,
  VisualDescriptionStatus,
} from "./api";

interface VisualDescriptionSectionProps {
  fileId: string;
  drive: string;
}

const POLL_MAX_ATTEMPTS = 30;
const POLL_INTERVAL_MS = 2000;

function isImageFile(file: FileItem | null): boolean {
  if (!file) return false;
  if (file.file_type === "image") return true;
  const mime = file.mime_type;
  return typeof mime === "string" && mime.toLowerCase().startsWith("image/");
}

export default function VisualDescriptionSection({
  fileId,
  drive,
}: VisualDescriptionSectionProps) {
  const t = useTranslations("file");
  const [loaded, setLoaded] = useState(false);
  const [file, setFile] = useState<FileItem | null>(null);
  const [data, setData] = useState<VisualDescriptionResponse | null>(null);
  const [available, setAvailable] = useState<boolean>(true);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmRegenerateOpen, setConfirmRegenerateOpen] = useState(false);

  // Guard against state updates from requests that race a file-id
  // change — when the parent navigates to a new file the in-flight
  // fetch shouldn't clobber the fresh state.
  const requestIdRef = useRef(0);

  const refetch = useCallback(
    async (requestId: number) => {
      const result = await getVisualDescription(fileId, drive);
      if (requestId !== requestIdRef.current) return;
      if (result === null) {
        setAvailable(false);
        setData(null);
      } else {
        setAvailable(true);
        setData(result);
      }
    },
    [fileId, drive],
  );

  useEffect(() => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setLoaded(false);
    setFile(null);
    setData(null);
    setAvailable(true);
    setError(null);
    setWorking(false);
    setConfirmRegenerateOpen(false);
    (async () => {
      try {
        const [fileResult] = await Promise.all([
          getFile(fileId).catch(() => null),
          refetch(requestId),
        ]);
        if (requestId !== requestIdRef.current) return;
        setFile(fileResult);
      } finally {
        if (requestId === requestIdRef.current) setLoaded(true);
      }
    })();
  }, [fileId, refetch]);

  const handleGenerate = useCallback(async () => {
    const requestId = requestIdRef.current;
    setWorking(true);
    setError(null);
    try {
      await generateVisualDescription(fileId, drive);
      // Optimistic pending while the backend queues the work.
      setData((prev) =>
        prev
          ? { ...prev, status: "pending" }
          : {
              file_id: fileId,
              visual_description: null,
              status: "pending",
              model: null,
              generated_at: null,
            },
      );
      for (let i = 0; i < POLL_MAX_ATTEMPTS; i++) {
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        if (requestId !== requestIdRef.current) return;
        const result = await getVisualDescription(fileId, drive);
        if (requestId !== requestIdRef.current) return;
        if (result === null) {
          setAvailable(false);
          return;
        }
        setData(result);
        if (result.status && result.status !== "pending") break;
      }
    } catch (e) {
      if (requestId !== requestIdRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (requestId === requestIdRef.current) setWorking(false);
    }
  }, [fileId, drive]);

  const handleRegenerate = useCallback(() => {
    setConfirmRegenerateOpen(true);
  }, []);

  const handleConfirmRegenerate = useCallback(async () => {
    setConfirmRegenerateOpen(false);
    const requestId = requestIdRef.current;
    setWorking(true);
    setError(null);
    try {
      // Best-effort delete first so the embedding row is cleared before
      // the worker inserts the new one. 404 is fine (nothing to clear).
      try {
        await deleteVisualDescription(fileId, drive);
      } catch {
        // non-critical
      }
      if (requestId !== requestIdRef.current) return;
      await handleGenerate();
    } catch (e) {
      if (requestId !== requestIdRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
      setWorking(false);
    }
  }, [fileId, drive, handleGenerate]);

  if (!loaded) return null;

  // Feature gate — when GET returned 404 the feature is unreachable
  // (disabled globally, per-drive policy OFF, or file not found).
  // Rendering nothing keeps the file detail page clean when the addon
  // is simply off.
  if (!available) return null;

  // Visual description is image-only in Phase 1. Non-image files get
  // nothing rendered so the section doesn't clutter video/audio pages.
  if (!isImageFile(file)) return null;

  const status: VisualDescriptionStatus | string | null =
    data?.status ?? null;

  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <ImageIcon size={14} className="text-accent-teal" />
        <h2 className="text-sm font-semibold text-text-primary">
          {t("visionTitle", { defaultMessage: "AI Visual Description" })}
        </h2>
        {data?.model && status === "success" && (
          <span className="text-[10px] text-text-muted/50">{data.model}</span>
        )}
      </div>

      {status === "unsupported" && (
        <div className="flex items-start gap-2 rounded-lg border border-bg-border bg-bg-elevated/50 px-3 py-2 text-xs text-text-muted">
          <Settings size={14} className="mt-0.5 flex-shrink-0" />
          <span>
            {t("visionUnsupported", {
              defaultMessage:
                "No vision-capable LLM is configured. Set `llm.vision_model` in search-config.yml to enable this feature.",
            })}
          </span>
        </div>
      )}

      {status === "pending" && (
        <div className="flex items-center gap-2 text-xs text-text-muted">
          <Loader2 size={12} className="animate-spin" />
          <span>
            {t("visionGenerating", {
              defaultMessage: "Generating description…",
            })}
          </span>
        </div>
      )}

      {status === "failed" && (
        <div className="space-y-2">
          <div className="flex items-start gap-2 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-xs text-danger">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            <span>
              {t("visionFailed", {
                defaultMessage:
                  "Description generation failed. Try again or check the intelligence logs.",
              })}
            </span>
          </div>
          <button
            onClick={handleGenerate}
            disabled={working}
            className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
          >
            <RefreshCw size={11} className={working ? "animate-spin" : ""} />
            {t("visionRetry", { defaultMessage: "Retry" })}
          </button>
        </div>
      )}

      {status === "success" && data?.visual_description && (
        <>
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-text-muted">
            {data.visual_description}
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              onClick={handleRegenerate}
              disabled={working}
              className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
            >
              <RefreshCw size={11} className={working ? "animate-spin" : ""} />
              {working
                ? t("visionGenerating", {
                    defaultMessage: "Generating description…",
                  })
                : t("visionRegenerate", { defaultMessage: "Regenerate" })}
            </button>
          </div>
        </>
      )}

      {/* Never generated yet — offer a manual trigger. Covers both
          features=manual and features=on_index (pre-index). */}
      {!status && (
        <button
          onClick={handleGenerate}
          disabled={working}
          className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
        >
          <RefreshCw size={11} className={working ? "animate-spin" : ""} />
          {working
            ? t("visionGenerating", {
                defaultMessage: "Generating description…",
              })
            : t("visionGenerate", {
                defaultMessage: "Generate AI description",
              })}
        </button>
      )}

      {error && (
        <p className="mt-2 text-[11px] text-danger/80">{error}</p>
      )}

      <ConfirmDialog
        open={confirmRegenerateOpen}
        title={t("visionRegenerate", { defaultMessage: "Regenerate" })}
        message={t("visionRegenerateConfirm", {
          defaultMessage:
            "Regenerate the AI description? The current description will be overwritten.",
        })}
        confirmLabel={t("visionRegenerate", { defaultMessage: "Regenerate" })}
        onConfirm={handleConfirmRegenerate}
        onCancel={() => setConfirmRegenerateOpen(false)}
      />
    </div>
  );
}
