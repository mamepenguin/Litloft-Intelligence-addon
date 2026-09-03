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
import { useOfferFileAiAction } from "./fileAiActions";
import { GeneratingRow } from "./GeneratingRow";

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
      // Optimistic pending while the backend queues the work. The
      // previous attempt's reason goes with its status — leaving it
      // would let the old explanation be read against the new attempt.
      setData((prev) =>
        prev
          ? { ...prev, status: "pending", reason: null }
          : {
              file_id: fileId,
              visual_description: null,
              status: "pending",
              reason: null,
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
      // A 409 means the worker declined the file outright. Say so
      // rather than leaving the user watching a spinner that will
      // never resolve.
      const declined = (e as { info?: { kind?: string } } | undefined)?.info;
      setError(
        declined?.kind === "not_queued"
          ? t("visionNotQueued", {
              defaultMessage:
                "The request could not be queued. Check the drive's addon "
                + "policy and the intelligence logs.",
            })
          : t("visionActionError", {
              defaultMessage: "Could not start description generation.",
            }),
      );
    } finally {
      if (requestId === requestIdRef.current) setWorking(false);
    }
  }, [fileId, drive, t]);

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
      setError(
        t("visionActionError", {
          defaultMessage: "Could not start description generation.",
        }),
      );
      setWorking(false);
    }
  }, [fileId, drive, handleGenerate, t]);

  const status: VisualDescriptionStatus | string | null =
    data?.status ?? null;

  // Never attempted: the heading and its button were the whole section,
  // on every image in the drive. The offer moves to the action row's
  // "AI" menu and the heading waits for a description to head.
  useOfferFileAiAction({
    fileId,
    kind: "visualDescription",
    labelKey: "visionGenerate",
    active: loaded && available && isImageFile(file) && !status,
    busy: working,
    run: handleGenerate,
  });

  if (!loaded) return null;

  // Feature gate — when GET returned 404 the feature is unreachable
  // (disabled globally, per-drive policy OFF, or file not found).
  // Rendering nothing keeps the file detail page clean when the addon
  // is simply off.
  if (!available) return null;

  // Visual description is image-only in Phase 1. Non-image files get
  // nothing rendered so the section doesn't clutter video/audio pages.
  if (!isImageFile(file)) return null;

  // Nothing has been attempted, so there is nothing to head — unless
  // the attempt this component just made failed before the backend
  // recorded a status. That failure is the one thing the "AI" menu
  // cannot say for us: it fires and forgets, and a run that dies here
  // would otherwise be silent and endlessly repeatable.
  if (!status) {
    if (working) {
      return (
        <GeneratingRow
          label={t("visionGenerating", { defaultMessage: "Generating description…" })}
        />
      );
    }
    return error ? (
      <p className="text-[11px] text-danger/80">{error}</p>
    ) : null;
  }
  const reason = data?.reason ?? null;
  const model = data?.model ?? null;

  // Only "no vision model is configured" is beyond the user's reach —
  // there is nothing to run, so no button is offered. Every other
  // verdict came from a real attempt against a real model and can be
  // asked again, including the ones recorded before the backend kept a
  // reason at all: those were the guesses this feature stopped making.
  const notConfigured =
    status === "unsupported" && reason === "not_configured";

  const failureMessage = () => {
    switch (reason) {
      case "model_missing":
        return t("visionModelMissing", {
          model: model ?? "",
          defaultMessage:
            "The model {model} was not found on the LLM provider. Pull it "
            + "on the provider side.",
        });
      case "image_rejected":
        return t("visionImageRejected", {
          defaultMessage: "The model could not read this image.",
        });
      case "token_budget":
        return t("visionTokenBudget", {
          defaultMessage:
            "The description was cut off by the token limit. Raise "
            + "llm.vision_max_tokens and try again.",
        });
      default:
        return t("visionFailed", {
          defaultMessage:
            "Description generation failed. Try again or check the "
            + "intelligence logs.",
        });
    }
  };

  const retryButton = (
    <button
      onClick={handleGenerate}
      disabled={working}
      className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
    >
      <RefreshCw size={11} className={working ? "animate-spin" : ""} />
      {t("visionRetry", { defaultMessage: "Retry" })}
    </button>
  );

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

      {notConfigured && (
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

      {status === "unsupported" && !notConfigured && (
        <div className="space-y-2">
          <div className="flex items-start gap-2 rounded-lg border border-bg-border bg-bg-elevated/50 px-3 py-2 text-xs text-text-muted">
            <Settings size={14} className="mt-0.5 flex-shrink-0" />
            <span>
              {reason
                ? t("visionModelCannotSee", {
                    model: model ?? "",
                    defaultMessage:
                      "The configured model {model} does not accept images. "
                      + "Set a vision-capable llm.vision_model, then try again.",
                  })
                : /* No reason recorded means this verdict predates the
                     backend keeping one, which is to say it came from
                     the guessing that produced the bug. Repeating its
                     conclusion here would be making the same guess a
                     second time. */
                  t("visionUnknownVerdict", {
                    defaultMessage:
                      "An earlier attempt recorded that this could not be "
                      + "described, without saying why. Try again.",
                  })}
            </span>
          </div>
          {retryButton}
        </div>
      )}

      {status === "pending" && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs text-text-muted">
            <Loader2 size={12} className="animate-spin" />
            <span>
              {t("visionGenerating", {
                defaultMessage: "Generating description…",
              })}
            </span>
          </div>
          {/* A run this component started is genuinely in flight, so the
              spinner is the whole story. Arriving at a pending row we
              did not start is different: the worker that claimed it may
              be long gone (a restart mid-flight leaves the row exactly
              like this), and a spinner with no way out is the dead end
              this section had elsewhere. Pressing it while the work is
              genuinely running costs nothing — the worker recognises
              the file as already queued and answers "already_queued"
              rather than buying a second LLM call. */}
          {!working && retryButton}
        </div>
      )}

      {status === "failed" && (
        <div className="space-y-2">
          <div className="flex items-start gap-2 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-xs text-danger">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            <span>{failureMessage()}</span>
          </div>
          {retryButton}
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
