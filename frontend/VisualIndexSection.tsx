"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import {
  AlertTriangle,
  Film,
  Loader2,
  RefreshCw,
} from "lucide-react";

import { usePolicy } from "@/hooks/usePolicy";
import { useWebSocket } from "@/hooks/useWebSocket";
import { formatDuration } from "@/lib/format";
import type { MediaController } from "@/lib/mediaController";
import {
  generateVideoVisualIndex,
  getFrameUrl,
  getVideoVisualIndex,
  retryVideoVisualIndex,
} from "./api";
import type {
  VideoVisualIndexResponse,
  VideoVisualSceneItem,
} from "./api";

interface VisualIndexSectionProps {
  fileId: string;
  drive: string;
  // Synchronously decides eligibility so the header never flashes for
  // files that cannot have a visual index. Optional only because the
  // slot prop wiring is untyped; FileDetailContent always supplies it.
  fileType?: string;
  mimeType?: string;
  mediaController?: MediaController | null;
}

const PAGE_SIZE = 12;

// .loft files report file_type='video' but have no locally probeable
// video bytes — excluded from the visual index pipeline in Phase 1
// (design doc §2.3).
const LOFT_MIME_TYPE = "application/vnd.litloft.loft+json";

const EVENT_PROGRESS = "intelligence.video_visual.progress";
const EVENT_SUCCEEDED = "intelligence.video_visual.succeeded";
const EVENT_PARTIAL = "intelligence.video_visual.partial";
const EVENT_FAILED = "intelligence.video_visual.failed";

export default function VisualIndexSection({
  fileId,
  drive,
  fileType,
  mimeType,
  mediaController,
}: VisualIndexSectionProps) {
  const t = useTranslations("file");
  const isEligibleType = fileType === "video" && mimeType !== LOFT_MIME_TYPE;
  const policy = usePolicy(drive, "intelligence", "video_visual_index");

  const [data, setData] = useState<VideoVisualIndexResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [showCount, setShowCount] = useState(PAGE_SIZE);
  const [working, setWorking] = useState<"generate" | "retry" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const requestIdRef = useRef(0);
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const stripRef = useRef<HTMLDivElement | null>(null);

  const load = useCallback(
    async (requestId: number) => {
      if (!isEligibleType || policy.isLoading || !policy.enabled) {
        if (requestId === requestIdRef.current) setLoaded(!policy.isLoading);
        return;
      }
      const result = await getVideoVisualIndex(fileId, drive);
      if (requestId !== requestIdRef.current) return;
      setData(result);
      setLoaded(true);
    },
    [drive, fileId, isEligibleType, policy.enabled, policy.isLoading],
  );

  useEffect(() => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setData(null);
    setLoaded(false);
    setExpanded(false);
    setShowCount(PAGE_SIZE);
    setWorking(null);
    setError(null);
    void load(requestId);
  }, [fileId, load]);

  const progressEvent = useWebSocket(EVENT_PROGRESS);
  const succeededEvent = useWebSocket(EVENT_SUCCEEDED);
  const partialEvent = useWebSocket(EVENT_PARTIAL);
  const failedEvent = useWebSocket(EVENT_FAILED);

  useEffect(() => {
    const candidates = [progressEvent, succeededEvent, partialEvent, failedEvent];
    const matches = candidates.some(
      (evt) => evt && (evt.data as { file_id?: string }).file_id === fileId,
    );
    if (!matches) return;
    void load(requestIdRef.current);
  }, [progressEvent, succeededEvent, partialEvent, failedEvent, fileId, load]);

  const handleGenerate = useCallback(async () => {
    setWorking("generate");
    setError(null);
    try {
      await generateVideoVisualIndex(fileId, drive);
      await load(requestIdRef.current);
    } catch (e) {
      // Every refusal used to read as "wait for scene indexing". The
      // reason arrives as data now, so only the refusal that means
      // that says it, and anything else falls back rather than
      // asserting a cause it does not know.
      const declined = (e as { info?: { kind?: string; reason?: string } })
        ?.info;
      if (declined?.kind === "not_queued" && declined.reason === "waiting_clip") {
        setError(
          t("visualIndexWaitingPrerequisite", {
            defaultMessage:
              "Waiting on scene indexing to finish before this can start.",
          }),
        );
      } else {
        setError(
          t("visualIndexActionError", {
            defaultMessage: "Could not start visual index generation.",
          }),
        );
      }
    } finally {
      if (requestIdRef.current) setWorking(null);
    }
  }, [drive, fileId, load, t]);

  const handleRetry = useCallback(async () => {
    setWorking("retry");
    setError(null);
    try {
      await retryVideoVisualIndex(fileId, drive);
      await load(requestIdRef.current);
    } catch {
      setError(
        t("visualIndexActionError", {
          defaultMessage: "Could not start visual index generation.",
        }),
      );
    } finally {
      setWorking(null);
    }
  }, [drive, fileId, load, t]);

  const handleToggle = useCallback(() => {
    setExpanded((v) => !v);
  }, []);

  const seekTo = useCallback(
    (time: number) => {
      if (!mediaController) return;
      mediaController.seek(time);
      mediaController.play();
    },
    [mediaController],
  );

  useEffect(() => {
    if (!expanded || !data || data.scenes.length === 0) return;
    if (showCount >= data.scenes.length) return;
    const node = sentinelRef.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setShowCount((c) => Math.min(c + PAGE_SIZE, data.scenes.length));
        }
      },
      { root: stripRef.current },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [expanded, data, showCount]);

  if (!loaded) return null;
  if (!isEligibleType || policy.isLoading || !policy.enabled) return null;
  if (!data || !data.eligible || !data.available) return null;

  const activeRun = data.active_run;
  const stagedRun = data.staged_run;
  const isStagedInFlight =
    !!stagedRun && (stagedRun.status === "queued" || stagedRun.status === "running");

  let headerLabel = "";
  if (isStagedInFlight) {
    const completed = stagedRun!.completed_count;
    const selected = stagedRun!.selected_count;
    const progress = selected > 0 ? `${completed}/${selected}` : "…";
    headerLabel = activeRun
      ? t("visualIndexUpdating", { defaultMessage: "Updating {progress}", progress })
      : t("visualIndexProcessing", { defaultMessage: "Processing {progress}", progress });
  } else if (activeRun && activeRun.status === "partial") {
    headerLabel = t("visualIndexPartial", { defaultMessage: "Partial" });
  } else if (activeRun) {
    headerLabel = t("visualIndexSceneCount", {
      defaultMessage: "{count} scenes",
      count: data.scenes.length,
    });
  }

  const retryableCount =
    stagedRun && (stagedRun.status === "partial" || stagedRun.status === "failed")
      ? stagedRun.failed_count
      : activeRun?.status === "partial"
        ? activeRun.failed_count
        : 0;
  const canRetry = retryableCount > 0 && !isStagedInFlight && working === null;
  const canGenerate = !isStagedInFlight && working === null;

  const visibleScenes = data.scenes.slice(0, showCount);

  return (
    <div>
      <button
        type="button"
        onClick={handleToggle}
        aria-expanded={expanded}
        aria-controls={`visual-index-${fileId}`}
        className="flex w-full cursor-pointer items-center gap-2 text-sm text-text-muted transition-colors hover:text-text-primary"
      >
        <Film size={14} />
        <span>
          {t("visualIndexTitle", { defaultMessage: "Visual index" })}
          {headerLabel ? ` · ${headerLabel}` : ""}
        </span>
        {isStagedInFlight && (
          <Loader2 size={12} className="animate-spin text-text-muted" aria-hidden="true" />
        )}
      </button>

      {expanded && (
        <div id={`visual-index-${fileId}`} className="mt-2 space-y-3">
          {data.stale && (
            <p className="text-[11px] text-accent-amber">
              {t("visualIndexStale", {
                defaultMessage:
                  "The source scenes have changed since this index was built.",
              })}
            </p>
          )}

          {data.scenes.length === 0 && !isStagedInFlight && (
            <p className="text-xs text-text-muted">
              {t("visualIndexEmpty", {
                defaultMessage: "No visual index has been generated yet.",
              })}
            </p>
          )}

          {data.scenes.length > 0 && (
            <div
              ref={stripRef}
              className="scrollbar-hover flex gap-3 overflow-x-auto p-1"
            >
              {visibleScenes.map((scene) => (
                <SceneCard
                  key={scene.ordering}
                  scene={scene}
                  fileId={fileId}
                  onSeek={seekTo}
                  t={t}
                />
              ))}
              {showCount < data.scenes.length && (
                <div ref={sentinelRef} aria-hidden className="w-1 shrink-0" />
              )}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-2">
            {canGenerate && (
              <button
                type="button"
                onClick={handleGenerate}
                disabled={working !== null}
                className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
              >
                <RefreshCw
                  size={11}
                  className={working === "generate" ? "animate-spin" : ""}
                  aria-hidden="true"
                />
                {working === "generate"
                  ? t("visualIndexGenerating", { defaultMessage: "Starting…" })
                  : activeRun
                    ? t("visualIndexGenerateAgain", { defaultMessage: "Generate again" })
                    : t("visualIndexGenerate", { defaultMessage: "Generate" })}
              </button>
            )}
            {canRetry && (
              <button
                type="button"
                onClick={handleRetry}
                disabled={working !== null}
                className="flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary disabled:opacity-50"
              >
                <RefreshCw
                  size={11}
                  className={working === "retry" ? "animate-spin" : ""}
                  aria-hidden="true"
                />
                {working === "retry"
                  ? t("visualIndexGenerating", { defaultMessage: "Starting…" })
                  : t("visualIndexRetryFailed", {
                      defaultMessage: "Retry failed scenes",
                    })}
              </button>
            )}
          </div>

          {error && <p className="text-[11px] text-danger/80">{error}</p>}
        </div>
      )}
    </div>
  );
}

function SceneCard({
  scene,
  fileId,
  onSeek,
  t,
}: {
  scene: VideoVisualSceneItem;
  fileId: string;
  onSeek: (time: number) => void;
  t: ReturnType<typeof useTranslations>;
}) {
  return (
    <div className="w-64 shrink-0 overflow-hidden rounded-lg bg-bg-card">
      <button
        type="button"
        onClick={() => onSeek(scene.start_time)}
        className="group block w-full cursor-pointer overflow-hidden rounded-t-lg transition-colors hover:ring-2 hover:ring-accent"
      >
        <img
          src={getFrameUrl(fileId, scene.start_time)}
          alt={scene.scene_label ?? ""}
          loading="lazy"
          className="aspect-video w-full object-cover"
        />
        <div className="flex items-center justify-between px-1.5 py-1 text-xs text-text-muted group-hover:text-accent">
          <span>{formatDuration(scene.start_time)}</span>
          {scene.status === "failed" && (
            <AlertTriangle
              size={12}
              className="text-danger"
              aria-label={t("visualIndexSceneFailed", { defaultMessage: "Failed" })}
            />
          )}
        </div>
      </button>
      <div className="space-y-1 px-2 pb-2">
        {scene.scene_label && (
          <p className="line-clamp-2 text-xs font-medium leading-snug text-text-primary">
            {scene.scene_label}
          </p>
        )}
        {scene.visible_text && (
          <p className="line-clamp-2 text-[11px] italic text-text-muted">
            {scene.visible_text}
          </p>
        )}
        {scene.transcript_excerpt && (
          <details className="text-[11px] text-text-muted">
            <summary className="cursor-pointer select-none">
              {t("visualIndexTranscriptExcerpt", { defaultMessage: "Transcript" })}
            </summary>
            <p className="mt-1 whitespace-pre-wrap">{scene.transcript_excerpt}</p>
          </details>
        )}
      </div>
    </div>
  );
}
