"use client";

/**
 * Popover that surfaces the top-1 chunk excerpt behind a detailed-summary
 * segment's citation marker. Appears on hover / focus of the 🔗 icon and
 * closes on Escape, pointer-leave, or blur. "ジャンプ" jumps the
 * associated video/audio to the chunk's timestamp — for document chunks
 * we surface the page number as read-only context (in-preview jump is
 * left to the existing TextPreview scroll machinery; wiring it in here
 * would require a global event bus the project does not yet expose, so
 * documents get the excerpt but not a jump button to avoid a half-working
 * affordance).
 */

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

// Grace period between leaving the trigger / popover and auto-dismissal.
// Long enough to tolerate the sub-pixel vertical travel between the 🔗
// icon and the popover body (and the user's hand-off hesitation); short
// enough that an actual intent-to-leave feels responsive.
const CLOSE_GRACE_MS = 160;
import { useTranslations } from "next-intl";
import { Link2, AlertTriangle, PlayCircle } from "lucide-react";

import { getCitationChunkExcerpt } from "./api";
import type { CitationChunkExcerpt, DetailedSummaryCitation } from "./api";

interface DetailedSummaryCitationPopoverProps {
  fileId: string;
  drive: string;
  citation: DetailedSummaryCitation;
  videoRef?: React.RefObject<HTMLVideoElement | null> | null;
  // Called with the chunk's timestamp when the user clicks "ジャンプ".
  // Prefer the videoRef path, but the parent can override for drives
  // that wire up a different media element (e.g. future audio-only
  // surface). Returning false means the handler did not handle the
  // jump and the popover falls back to the built-in videoRef seek.
  onJump?: (excerpt: CitationChunkExcerpt) => boolean | void;
}

type PopoverState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready"; excerpt: CitationChunkExcerpt | null }
  | { kind: "error" };

export function DetailedSummaryCitationPopover({
  fileId,
  drive,
  citation,
  videoRef,
  onJump,
}: DetailedSummaryCitationPopoverProps) {
  const t = useTranslations("detailedSummary");
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<PopoverState>({ kind: "idle" });
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const cancelClose = useCallback(() => {
    if (closeTimerRef.current != null) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, []);

  useEffect(() => () => cancelClose(), [cancelClose]);

  const hasCitation = citation.has_citation;
  const topChunkId = citation.chunk_ids[0] ?? null;

  // Fetch the chunk excerpt lazily on first open. The citation itself
  // is tiny (<1 KB per file) but excerpts could balloon to hundreds of
  // KB across a long summary — defer until the user actually opens the
  // popover. Keep the cached result in state so repeated hovers don't
  // refetch.
  const loadExcerpt = useCallback(async () => {
    if (!topChunkId) {
      setState({ kind: "ready", excerpt: null });
      return;
    }
    setState({ kind: "loading" });
    try {
      const excerpt = await getCitationChunkExcerpt(fileId, topChunkId, drive);
      setState({ kind: "ready", excerpt });
    } catch {
      setState({ kind: "error" });
    }
  }, [fileId, drive, topChunkId]);

  const handleOpen = useCallback(() => {
    cancelClose();
    setOpen(true);
    if (state.kind === "idle" || state.kind === "error") {
      void loadExcerpt();
    }
  }, [state.kind, loadExcerpt, cancelClose]);

  const handleClose = useCallback(() => {
    cancelClose();
    setOpen(false);
  }, [cancelClose]);

  // Schedule a deferred close. Either re-entering the trigger or the
  // popover body cancels the timer, so the user can freely traverse the
  // sub-pixel gap between them without the popover dismissing.
  const scheduleClose = useCallback(() => {
    cancelClose();
    closeTimerRef.current = setTimeout(() => {
      closeTimerRef.current = null;
      setOpen(false);
    }, CLOSE_GRACE_MS);
  }, [cancelClose]);

  // Outside-click + Escape dismissal.
  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        handleClose();
        triggerRef.current?.focus();
      }
    };
    const handlePointer = (e: MouseEvent) => {
      const target = e.target as Node | null;
      if (!target) return;
      if (
        popoverRef.current?.contains(target) ||
        triggerRef.current?.contains(target)
      ) {
        return;
      }
      handleClose();
    };
    window.addEventListener("keydown", handleKey);
    window.addEventListener("mousedown", handlePointer);
    return () => {
      window.removeEventListener("keydown", handleKey);
      window.removeEventListener("mousedown", handlePointer);
    };
  }, [open, handleClose]);

  // Ensure the popover never escapes the viewport by flipping side if
  // the default (below) would overflow. Cheap layout read — measurement
  // happens after paint and before the browser commits the frame.
  const [placement, setPlacement] = useState<"below" | "above">("below");
  useLayoutEffect(() => {
    if (!open || !popoverRef.current || !triggerRef.current) return;
    const triggerRect = triggerRef.current.getBoundingClientRect();
    const popoverRect = popoverRef.current.getBoundingClientRect();
    const spaceBelow = window.innerHeight - triggerRect.bottom;
    if (spaceBelow < popoverRect.height + 8) {
      setPlacement("above");
    } else {
      setPlacement("below");
    }
  }, [open, state]);

  const handleJump = useCallback(() => {
    if (state.kind !== "ready" || !state.excerpt) return;
    if (onJump) {
      const handled = onJump(state.excerpt);
      if (handled) {
        handleClose();
        return;
      }
    }
    const video = videoRef?.current;
    if (video && state.excerpt.start_time != null) {
      video.currentTime = state.excerpt.start_time;
      // Resume playback if paused — clicking "ジャンプ" signals intent
      // to consume the source, not just reposition. Ignore promise
      // rejection (autoplay blocked, etc.) — the user can press play.
      void video.play?.().catch(() => {});
    }
    handleClose();
  }, [state, onJump, videoRef, handleClose]);

  const jumpDisabled = (() => {
    if (state.kind !== "ready" || !state.excerpt) return true;
    // Video/audio jump only when we have a seekable timestamp and a
    // video ref to drive. Document jump is currently unavailable — we
    // render the excerpt but the button stays disabled.
    if (state.excerpt.start_time != null && videoRef?.current) return false;
    if (onJump && state.excerpt) return false;
    return true;
  })();

  return (
    <span className="relative inline-flex">
      <button
        ref={triggerRef}
        type="button"
        onMouseEnter={handleOpen}
        onFocus={handleOpen}
        onMouseLeave={scheduleClose}
        onClick={() => (open ? handleClose() : handleOpen())}
        className={`mx-1 inline-flex h-4 w-4 items-center justify-center rounded align-middle transition-colors ${
          hasCitation
            ? "text-accent-teal/60 hover:text-accent-teal"
            : "text-accent-amber hover:text-accent-amber/80"
        }`}
        aria-label={
          hasCitation
            ? t("citations.linkLabel", { defaultMessage: "Show citation" })
            : t("citations.noCitation", {
                defaultMessage:
                  "No strong source match found. This may be inaccurate.",
              })
        }
        title={
          hasCitation
            ? undefined
            : t("citations.noCitation", {
                defaultMessage:
                  "No strong source match found. This may be inaccurate.",
              })
        }
        aria-haspopup="dialog"
        aria-expanded={open}
        data-citation-marker={hasCitation ? "linked" : "missing"}
      >
        {hasCitation ? (
          <Link2 size={11} aria-hidden />
        ) : (
          <AlertTriangle size={11} aria-hidden />
        )}
      </button>

      {open && (
        <div
          ref={popoverRef}
          role="dialog"
          aria-label={t("citations.popoverLabel", {
            defaultMessage: "Citation preview",
          })}
          onMouseEnter={cancelClose}
          onMouseLeave={scheduleClose}
          className={`absolute z-40 w-80 rounded-lg border border-bg-border bg-bg-card p-3 text-xs shadow-lg ${
            placement === "above" ? "bottom-full" : "top-full"
          } left-0`}
        >
          {state.kind === "loading" && (
            <p className="text-text-muted">
              {t("citations.loading", { defaultMessage: "Loading…" })}
            </p>
          )}
          {state.kind === "error" && (
            <p className="text-accent-red">
              {t("citations.error", {
                defaultMessage: "Failed to load excerpt",
              })}
            </p>
          )}
          {state.kind === "ready" && !state.excerpt && (
            <p className="text-text-muted">
              {t("citations.noExcerpt", {
                defaultMessage: "No excerpt available",
              })}
            </p>
          )}
          {state.kind === "ready" && state.excerpt && (
            <>
              <p className="mb-2 whitespace-pre-wrap leading-relaxed text-text-primary">
                {state.excerpt.text}
              </p>
              <div className="flex items-center justify-between gap-2 text-[11px] text-text-muted">
                <span>
                  {state.excerpt.start_time != null
                    ? formatTimestamp(state.excerpt.start_time)
                    : state.excerpt.page != null
                    ? t("citations.pageLabel", {
                        defaultMessage: "Page {page}",
                        page: state.excerpt.page,
                      })
                    : ""}
                </span>
                <button
                  type="button"
                  onClick={handleJump}
                  disabled={jumpDisabled}
                  className="inline-flex items-center gap-1 rounded bg-accent-teal px-2 py-1 text-[11px] text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <PlayCircle size={11} />
                  {t("citations.jump", { defaultMessage: "Jump" })}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </span>
  );
}

function formatTimestamp(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  const s = Math.floor(seconds % 60);
  const m = Math.floor(seconds / 60) % 60;
  const h = Math.floor(seconds / 3600);
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}
