"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp, Film } from "lucide-react";
import { useTranslations } from "next-intl";

import { getClipTimestamps, getFrameUrl } from "./api";
import type { ClipTimestampItem } from "./api";
import { formatDuration } from "@/lib/format";
import type { MediaController } from "@/lib/mediaController";

interface ClipFramesSectionProps {
  fileId: string;
  drive: string;
  // Legacy: still accepted for callers that pass a native video ref.
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  // Preferred: works for native video AND LoftRef (YouTube) embeds.
  mediaController?: MediaController | null;
}

const PAGE_SIZE = 20;
const ARROW_SCROLL_PX_PER_FRAME = 8;
const NEAR_END_ROOT_MARGIN_PX = 400;

export default function ClipFramesSection({ fileId, drive, videoRef, mediaController }: ClipFramesSectionProps) {
  const t = useTranslations("searchIndex");

  // Tri-state. `null` means "we haven't fetched timestamps yet" — used to
  // hide the section entirely when the file has no CLIP index, without
  // ever firing a network request on mount.
  const [timestamps, setTimestamps] = useState<ClipTimestampItem[] | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showCount, setShowCount] = useState(PAGE_SIZE);

  const stripRef = useRef<HTMLDivElement | null>(null);
  const sentinelRef = useRef<HTMLDivElement | null>(null);

  // Detect a fine pointer (mouse) at runtime. Tailwind's arbitrary
  // ``[@media(hover:hover)]:`` variant is finicky to combine with
  // ``hidden`` reliably across the project's Tailwind v4 setup, and the
  // arrows are decorative anyway — touch users keep native scroll.
  const [hasFinePointer, setHasFinePointer] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(pointer: fine)");
    setHasFinePointer(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setHasFinePointer(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  // Reset visible count when the file changes so a previous file's
  // scroll position doesn't pre-load frames for a new video.
  useEffect(() => {
    setTimestamps(null);
    setExpanded(false);
    setShowCount(PAGE_SIZE);
  }, [fileId]);

  const handleToggle = useCallback(async () => {
    if (expanded) {
      setExpanded(false);
      return;
    }
    setExpanded(true);
    if (timestamps !== null || loading) return;
    setLoading(true);
    try {
      const res = await getClipTimestamps(fileId, drive);
      setTimestamps(res.available && res.timestamps ? res.timestamps : []);
    } finally {
      setLoading(false);
    }
  }, [expanded, fileId, drive, timestamps, loading]);

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

  // Press-and-hold arrow scroll. requestAnimationFrame increments
  // scrollLeft until pointerup / pointercancel / pointerleave, with
  // pointer capture so the gesture survives a small drift off the
  // button.
  const scrollDirRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const stopArrowScroll = useCallback(() => {
    scrollDirRef.current = 0;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);
  const startArrowScroll = useCallback(
    (direction: -1 | 1, evt: ReactPointerEvent<HTMLButtonElement>) => {
      evt.preventDefault();
      try {
        evt.currentTarget.setPointerCapture(evt.pointerId);
      } catch {
        // ignore — browsers without pointer capture still get tick scroll
      }
      scrollDirRef.current = direction;
      const tick = () => {
        const strip = stripRef.current;
        if (!strip || scrollDirRef.current === 0) {
          rafRef.current = null;
          return;
        }
        strip.scrollLeft += direction * ARROW_SCROLL_PX_PER_FRAME;
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    },
    []
  );
  useEffect(() => () => stopArrowScroll(), [stopArrowScroll]);

  const totalLabel = timestamps ? ` (${timestamps.length})` : "";
  const visible = timestamps ? timestamps.slice(0, showCount) : [];

  // Hide the whole section once we've confirmed the file has no CLIP
  // frames. Until then (closed default) the header is always visible
  // because we don't yet know whether frames exist — the API call only
  // fires on first expand, which is the whole point of this redesign.
  if (timestamps !== null && timestamps.length === 0) {
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
        <div id={`clip-frames-${fileId}`} className="relative mt-2">
          {loading && (
            <div className="text-xs text-text-muted">…</div>
          )}
          {timestamps && timestamps.length > 0 && (
            <>
              <div
                ref={stripRef}
                className="scrollbar-hover flex gap-2 overflow-x-auto"
              >
                {visible.map((ts) => (
                  <button
                    key={ts.start}
                    onClick={() => seekTo(ts.start)}
                    className="group w-48 shrink-0 cursor-pointer overflow-hidden rounded-lg bg-bg-card transition-colors hover:ring-2 hover:ring-accent"
                  >
                    <img
                      src={getFrameUrl(fileId, ts.start)}
                      alt={ts.content_preview}
                      loading="lazy"
                      className="aspect-video w-full object-cover"
                    />
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

              {/* Press-and-hold arrow overlays for mouse users. Hidden
                  on touch devices because the native horizontal swipe
                  is the better gesture there. Pointer-fine detection is
                  done in JS (matchMedia) for reliability across the
                  project's Tailwind v4 setup. */}
              {hasFinePointer && (
                <>
                  <button
                    type="button"
                    aria-label="Scroll left"
                    onPointerDown={(e) => startArrowScroll(-1, e)}
                    onPointerUp={stopArrowScroll}
                    onPointerCancel={stopArrowScroll}
                    onPointerLeave={stopArrowScroll}
                    className="absolute left-2 top-[calc(3.375rem-1.25rem)] z-10 flex h-10 w-10 items-center justify-center rounded-full bg-bg-primary/90 text-text-primary shadow-md ring-1 ring-bg-border backdrop-blur-sm transition-colors hover:bg-bg-primary"
                  >
                    <ChevronLeft size={22} />
                  </button>
                  <button
                    type="button"
                    aria-label="Scroll right"
                    onPointerDown={(e) => startArrowScroll(1, e)}
                    onPointerUp={stopArrowScroll}
                    onPointerCancel={stopArrowScroll}
                    onPointerLeave={stopArrowScroll}
                    className="absolute right-2 top-[calc(3.375rem-1.25rem)] z-10 flex h-10 w-10 items-center justify-center rounded-full bg-bg-primary/90 text-text-primary shadow-md ring-1 ring-bg-border backdrop-blur-sm transition-colors hover:bg-bg-primary"
                  >
                    <ChevronRight size={22} />
                  </button>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
