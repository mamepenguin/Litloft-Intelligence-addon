"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Film } from "lucide-react";
import { useTranslations } from "next-intl";

import { getClipTimestamps, getFrameUrl, getSceneTileUrl } from "./api";
import type { ClipTimestampItem } from "./api";
import { formatDuration } from "@/lib/format";
import type { MediaController } from "@/lib/mediaController";

interface ClipFramesSectionProps {
  fileId: string;
  drive: string;
  // Synchronously decides eligibility so the header never flashes for
  // files that cannot have scene CLIP frames. Optional only because the
  // slot prop wiring is untyped; FileDetailContent always supplies it.
  fileType?: string;
  mimeType?: string;
  // Legacy: still accepted for callers that pass a native video ref.
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  // Preferred: works for native video AND LoftRef (YouTube) embeds.
  mediaController?: MediaController | null;
}

const PAGE_SIZE = 20;
const NEAR_END_ROOT_MARGIN_PX = 400;

// .loft files report file_type='video' but are stored as
// clip_thumbnail only (hako tOG7qDce-f1403dh6dkij) — they never have
// scene timestamps.
const LOFT_MIME_TYPE = "application/vnd.litloft.loft+json";

export default function ClipFramesSection({
  fileId,
  drive,
  fileType,
  mimeType,
  videoRef,
  mediaController,
}: ClipFramesSectionProps) {
  const t = useTranslations("searchIndex");

  // Tri-state. `null` means "we haven't fetched timestamps yet" — used to
  // hide the section entirely when the file has no CLIP index, without
  // ever firing a network request on mount.
  const isLoft = mimeType === LOFT_MIME_TYPE;

  const [timestamps, setTimestamps] = useState<ClipTimestampItem[] | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showCount, setShowCount] = useState(PAGE_SIZE);
  const [failedTiles, setFailedTiles] = useState<Set<number>>(() => new Set());

  const stripRef = useRef<HTMLDivElement | null>(null);
  const sentinelRef = useRef<HTMLDivElement | null>(null);

  // Reset visible count when the file changes so a previous file's
  // scroll position doesn't pre-load frames for a new video.
  useEffect(() => {
    setTimestamps(null);
    setExpanded(false);
    setShowCount(PAGE_SIZE);
    setFailedTiles(new Set());
  }, [fileId, isLoft]);

  // A `.loft` has no frame route worth calling: its rows come from the
  // storyboard the index cut, so they are read on mount instead of on expand.
  useEffect(() => {
    if (!isLoft) return;
    let current = true;
    getClipTimestamps(fileId, drive)
      .then((res) => {
        if (current) setTimestamps(res?.timestamps ?? []);
      })
      .catch(() => {
        if (current) setTimestamps([]);
      });
    return () => {
      current = false;
    };
  }, [isLoft, fileId, drive]);

  const handleToggle = useCallback(async () => {
    if (isLoft) {
      setExpanded((value) => !value);
      return;
    }
    if (expanded) {
      setExpanded(false);
      return;
    }
    setExpanded(true);
    if (timestamps !== null || loading) return;
    setLoading(true);
    try {
      const res = await getClipTimestamps(fileId, drive);
      setTimestamps(res?.timestamps ?? []);
    } finally {
      setLoading(false);
    }
  }, [expanded, fileId, drive, timestamps, loading, isLoft]);

  const markTileFailed = useCallback((start: number) => {
    setFailedTiles((current) => {
      const next = new Set(current);
      next.add(start);
      return next;
    });
  }, []);

  const seekTo = useCallback(
    (time: number) => {
      // Prefer the unified controller (covers LoftRef/YouTube) and fall
      // back to the legacy native video ref for callers that haven't
      // migrated yet.
      if (mediaController) {
        mediaController.seek(time);
        mediaController.play();
        return;
      }
      const video = videoRef?.current;
      if (video) {
        video.currentTime = time;
      }
    },
    [videoRef, mediaController]
  );

  // IntersectionObserver-driven infinite scroll. Only attach once the
  // strip is mounted (i.e. expanded with timestamps) so we don't hold a
  // dangling observer.
  useEffect(() => {
    if (!expanded || !timestamps || timestamps.length === 0) return;
    if (showCount >= timestamps.length) return;
    const node = sentinelRef.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setShowCount((c) => Math.min(c + PAGE_SIZE, timestamps.length));
        }
      },
      { root: stripRef.current, rootMargin: `0px ${NEAR_END_ROOT_MARGIN_PX}px 0px 0px` }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [expanded, timestamps, showCount]);

  const totalLabel = timestamps ? ` (${timestamps.length})` : "";
  const visible = timestamps ? timestamps.slice(0, showCount) : [];
  // Built once per row set, so a failed tile's own re-render asks for
  // nothing else.
  const pictureUrls = useMemo(() => {
    if (!timestamps) return new Map<number, string>();
    return new Map(
      timestamps.map((ts) => [
        ts.start,
        isLoft
          ? getSceneTileUrl(fileId, ts.start)
          : getFrameUrl(fileId, ts.start),
      ])
    );
  }, [isLoft, fileId, timestamps]);

  // Scene CLIP frames only exist for video files. Bail synchronously for
  // text / image / audio so the header never flashes for them.
  if (fileType !== "video") {
    return null;
  }

  if (isLoft) {
    // Its rows are fetched on mount, so nothing is drawn until they arrive
    // and there is something to open.
    if (timestamps === null || timestamps.length === 0) {
      return null;
    }
  } else if (timestamps !== null && timestamps.length === 0) {
    // A native video hides the whole section once the first expand has
    // confirmed it has no CLIP frames yet — e.g. still queued for indexing.
    return null;
  }

  return (
    <div>
      <button
        type="button"
        onClick={handleToggle}
        aria-expanded={expanded}
        aria-controls={`clip-frames-${fileId}`}
        className="flex w-full cursor-pointer items-center gap-2 text-sm text-text-muted transition-colors hover:text-text-primary"
      >
        <Film size={14} />
        <span>{t("clipTitle")}{totalLabel}</span>
        <span className="ml-auto" aria-hidden>
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </span>
      </button>

      {expanded && (
        <div id={`clip-frames-${fileId}`} className="mt-2">
          {loading && (
            <div className="text-xs text-text-muted">…</div>
          )}
          {timestamps && timestamps.length > 0 && (
            <div
              ref={stripRef}
              // ``p-1`` reserves 4px on every side so each card's
              // ``hover:ring-2`` outline isn't clipped by the scroll
              // container, which clips both axes.
              className="scrollbar-hover flex flex-wrap gap-2 p-1 mouse:max-h-[17.5rem] mouse:overflow-y-auto pointer-coarse:flex-nowrap pointer-coarse:overflow-x-auto"
            >
              {visible.map((ts) => (
                <button
                  key={ts.start}
                  onClick={() => seekTo(ts.start)}
                  className="group w-48 shrink-0 cursor-pointer overflow-hidden rounded-lg bg-bg-card transition-colors hover:ring-2 hover:ring-accent"
                >
                  {isLoft && failedTiles.has(ts.start) ? (
                    <div
                      aria-hidden
                      className="aspect-video w-full bg-bg-elevated"
                    />
                  ) : (
                    <img
                      src={pictureUrls.get(ts.start)}
                      alt={ts.content_preview}
                      loading="lazy"
                      onError={
                        isLoft ? () => markTileFailed(ts.start) : undefined
                      }
                      className="aspect-video w-full object-cover"
                    />
                  )}
                  <div className="px-1.5 py-1 text-center text-xs text-text-muted group-hover:text-accent">
                    {formatDuration(ts.start)}
                  </div>
                </button>
              ))}
              {showCount < timestamps.length && (
                <div
                  ref={sentinelRef}
                  aria-hidden
                  className="w-1 shrink-0"
                />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
