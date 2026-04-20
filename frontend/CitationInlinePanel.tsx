"use client";

/**
 * Inline overlay panel that renders the excerpt for the currently
 * active detailed-summary citation.
 *
 * Position strategy:
 *   - Default (`overlay={true}`): absolutely positioned at `top-full`
 *     of the containing segment, floating above the subsequent content
 *     without shifting layout. Used for paragraphs and bullets — the
 *     parent `<div>` / `<li>` gets `position: relative` so the panel
 *     anchors to the citing line.
 *   - Table rows (`overlay={false}`): the panel is rendered in-flow
 *     inside a spanning `<tr>` beneath the cited row because table
 *     layout semantics don't tolerate an absolute child inside a
 *     `<td>` gracefully.
 *
 * Dismissal:
 *   - Hover: `scheduleClose` fires on mouseleave; re-entering either
 *     the marker or the panel cancels it. The marker handles its own
 *     enter/leave; the panel mirrors the pattern so moving cursor
 *     trigger → panel doesn't race the grace timer.
 *   - Pinned (click-opened): `scheduleClose` is a no-op. The panel
 *     only closes when the user clicks the X, re-clicks the marker,
 *     clicks outside, or presses Escape.
 */

import { useEffect, useRef } from "react";
import { useTranslations } from "next-intl";
import { PlayCircle, X } from "lucide-react";

import { useCitationRail } from "./CitationRailContext";
import type { CitationFetchState } from "./CitationRailContext";
import type { CitationChunkExcerpt } from "./api";

// Parse "transcript:42" / "document:3" to 42 / 3. Returns null for
// malformed ids.
function parseChunkIdx(chunkId: string): number | null {
  const colon = chunkId.indexOf(":");
  if (colon < 0) return null;
  const n = Number.parseInt(chunkId.slice(colon + 1), 10);
  return Number.isFinite(n) ? n : null;
}

// Adjacent-chunk highlight extension: when top-2 or top-3 is the
// immediate forward / backward neighbour of top-1 (idx_gap == 1),
// extend the highlight to cover the matching prefix/suffix portion of
// the excerpt. Only idx_gap==1 qualifies — idx_gap==2 has too low a
// precision against curated GT (~25%) to be worth the false positives.
//
// At most one direction extends, even when both top-2 and top-3 are
// adjacent on opposite sides. Simultaneous bilateral extension
// produces a fully-highlighted excerpt that reads as noise — and in
// the observed cases where both sides fire, the score_gap tends to be
// large (both are low-confidence adjacencies). Preferring the
// earlier-ranked chunk (top-2 wins over top-3) uses the retrieval
// rank as a proxy for the smaller score_gap.
//
// Data: hako k0XoWYoUBAhtylHc94KTI, CITATION-PIPELINE.md Stage 6.
function computeExtension(
  chunkIds: readonly string[],
): { extendForward: boolean; extendBackward: boolean } {
  if (chunkIds.length < 2) {
    return { extendForward: false, extendBackward: false };
  }
  const top1 = parseChunkIdx(chunkIds[0]);
  if (top1 === null) {
    return { extendForward: false, extendBackward: false };
  }
  for (const id of chunkIds.slice(1, 3)) {
    const idx = parseChunkIdx(id);
    if (idx === null) continue;
    if (idx === top1 + 1) {
      return { extendForward: true, extendBackward: false };
    }
    if (idx === top1 - 1) {
      return { extendForward: false, extendBackward: true };
    }
  }
  return { extendForward: false, extendBackward: false };
}

interface CitationInlinePanelProps {
  sectionPath: string;
  videoRef?: React.RefObject<HTMLVideoElement | null> | null;
  onJump?: (excerpt: CitationChunkExcerpt) => boolean | void;
  // Absolute-overlay (default) vs. in-flow push-down. Table cells set
  // this to false because an absolute child inside `<td>` breaks table
  // layout math.
  overlay?: boolean;
}

export function CitationInlinePanel({
  sectionPath,
  videoRef,
  onJump,
  overlay = true,
}: CitationInlinePanelProps) {
  const t = useTranslations("detailedSummary");
  const { active, state, clearActive, scheduleClose, cancelClose } =
    useCitationRail();
  const panelRef = useRef<HTMLElement | null>(null);

  const isThisActive = active?.citation.section_path === sectionPath;
  const pinned = isThisActive && active?.pinned === true;

  // When pinned, clicks outside both the panel and any citation marker
  // dismiss it. Escape always dismisses. Non-pinned closes happen via
  // the hover grace timer.
  useEffect(() => {
    if (!isThisActive || !pinned) return;
    const handleMousedown = (e: MouseEvent) => {
      const target = e.target as Node | null;
      if (!target) return;
      if (panelRef.current?.contains(target)) return;
      if ((target as Element).closest?.("[data-citation-marker]")) return;
      clearActive();
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        clearActive();
      }
    };
    window.addEventListener("mousedown", handleMousedown);
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("mousedown", handleMousedown);
      window.removeEventListener("keydown", handleKey);
    };
  }, [isThisActive, pinned, clearActive]);

  if (!isThisActive) return null;

  const hasCitation = active.citation.has_citation;
  const { extendForward, extendBackward } = computeExtension(
    active.citation.chunk_ids,
  );

  // Overlay mode puts the panel above subsequent content with
  // box-shadow, solid background and a z-index so the content beneath
  // is fully obscured (not blurred or semi-transparent) while the
  // reader is focused on verifying the source. In-flow mode inherits
  // the surrounding table cell's layout so the expansion row feels
  // natively part of the table.
  const positionClass = overlay
    ? "absolute left-0 right-0 top-full z-20 shadow-lg"
    : "relative";
  const containerClass = overlay ? "" : "mt-2";

  return (
    <aside
      ref={panelRef}
      role="region"
      aria-label={t("inline.label", { defaultMessage: "Citation excerpt" })}
      onMouseEnter={cancelClose}
      onMouseLeave={scheduleClose}
      className={`${positionClass} ${containerClass} mt-2 mb-3 rounded-md border border-bg-border border-l-2 border-l-accent-teal bg-bg-card p-3 pr-8 text-xs`}
    >
      <button
        type="button"
        onClick={clearActive}
        aria-label={t("inline.close", { defaultMessage: "Close" })}
        className="absolute top-1.5 right-1.5 rounded p-1 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
      >
        <X size={12} />
      </button>
      {hasCitation ? (
        <InlineExcerptBody
          state={state}
          videoRef={videoRef}
          onJump={onJump}
          onAfterJump={clearActive}
          extendForward={extendForward}
          extendBackward={extendBackward}
        />
      ) : (
        <p className="text-accent-amber/90">
          {t("citations.noCitation", {
            defaultMessage:
              "No strong source match found. This may be inaccurate.",
          })}
        </p>
      )}
    </aside>
  );
}

function InlineExcerptBody({
  state,
  videoRef,
  onJump,
  onAfterJump,
  extendForward = false,
  extendBackward = false,
}: {
  state: CitationFetchState;
  videoRef?: React.RefObject<HTMLVideoElement | null> | null;
  onJump?: (excerpt: CitationChunkExcerpt) => boolean | void;
  onAfterJump?: () => void;
  // When the citation has an adjacent top-2/3 chunk at idx±1, mark the
  // corresponding prefix/suffix portion of the excerpt as target-like
  // so the user sees one contiguous highlight across the ASR chunk
  // boundary. See computeExtension() above and CITATION-PIPELINE.md
  // Stage 6.
  extendForward?: boolean;
  extendBackward?: boolean;
}) {
  const t = useTranslations("detailedSummary");

  if (state.kind === "loading") {
    return (
      <p className="text-text-muted">
        {t("citations.loading", { defaultMessage: "Loading…" })}
      </p>
    );
  }
  if (state.kind === "error") {
    return (
      <p className="text-accent-red">
        {t("citations.error", { defaultMessage: "Failed to load excerpt" })}
      </p>
    );
  }
  if (state.kind === "idle") return null;

  const excerpt = state.excerpt;
  if (!excerpt) {
    return (
      <p className="text-text-muted">
        {t("citations.noExcerpt", { defaultMessage: "No excerpt available" })}
      </p>
    );
  }

  const jumpDisabled = (() => {
    if (excerpt.start_time != null && videoRef?.current) return false;
    if (onJump && excerpt) return false;
    return true;
  })();

  const handleJump = () => {
    if (onJump) {
      const handled = onJump(excerpt);
      if (handled) {
        onAfterJump?.();
        return;
      }
    }
    const video = videoRef?.current;
    if (video && excerpt.start_time != null) {
      video.currentTime = excerpt.start_time;
      // Resume playback if paused — clicking "Jump" signals intent to
      // consume the source, not just reposition. Ignore promise
      // rejection (autoplay blocked, etc.) — the user can press play.
      void video.play?.().catch(() => {});
    }
    onAfterJump?.();
  };

  return (
    <div className="space-y-2">
      <p className="whitespace-pre-wrap leading-relaxed">
        {excerpt.prefix && (
          extendBackward ? (
            <mark
              data-testid="citation-extended-prefix"
              // Extended highlight: text stays at full primary colour,
              // background is a faded mix of the target highlight so
              // the reader sees "adjacent chunk also cited" without
              // mistaking it for the primary target.
              className="rounded px-0.5 text-text-primary"
              style={{
                backgroundColor:
                  "color-mix(in srgb, var(--highlight-bg) 45%, transparent)",
              }}
            >
              {excerpt.prefix}
            </mark>
          ) : (
            <span className="text-text-muted/80">{excerpt.prefix}</span>
          )
        )}
        <mark
          data-testid="citation-target"
          // Reuse the app-wide ``--highlight-bg`` token (the same colour
          // MarkdownPreview uses for <mark>) so citation highlights
          // match the rest of the product. The earlier ~22 % teal tint
          // rendered so faintly against bg-card that the match was hard
          // to spot.
          className="rounded px-0.5 text-text-primary"
          style={{ backgroundColor: "var(--highlight-bg)" }}
        >
          {excerpt.target}
        </mark>
        {excerpt.suffix && (
          extendForward ? (
            <mark
              data-testid="citation-extended-suffix"
              className="rounded px-0.5 text-text-primary"
              style={{
                backgroundColor:
                  "color-mix(in srgb, var(--highlight-bg) 45%, transparent)",
              }}
            >
              {excerpt.suffix}
            </mark>
          ) : (
            <span className="text-text-muted/80">{excerpt.suffix}</span>
          )
        )}
      </p>
      <div className="flex items-center justify-between gap-2 text-[11px] text-text-muted">
        <span>
          {excerpt.start_time != null
            ? formatTimestamp(excerpt.start_time)
            : excerpt.page != null
            ? t("citations.pageLabel", {
                defaultMessage: "Page {page}",
                page: excerpt.page,
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
    </div>
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
