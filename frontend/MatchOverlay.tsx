"use client";

/**
 * MatchOverlay — match metadata rendered inside a core ``FileCard`` on
 * the search results page.
 *
 * Spec ``2026-05-01-search-ui-rich-redesign.md`` Phase 2:
 * SemanticResultCard was retired and intelligence semantic results now
 * use the same ``FileCard`` as filename matches. This overlay carries
 * the bits unique to semantic results (why a file matched and where in
 * the media the match lives) without duplicating the card's structural
 * elements.
 *
 * Color usage follows DESIGN.md §2.2 (warm palette, no cool blues):
 *   transcript / transcript_keyword → accent-teal (audio = nature)
 *   clip                            → accent-amber (visual = focus)
 *   metadata / content / text_..._keyword → sand / warm-light (neutral)
 *
 * Click handlers are scoped to the overlay so card-level navigation
 * (the parent FileCard's <Link> wrapper) is preempted by per-pill
 * timestamp deep-links via ``stopPropagation``.
 */

import type { SemanticSearchResult, SemanticSearchSegment } from "./api";
import { formatDuration } from "@/lib/format";

const MATCH_TYPE_STYLES: Record<string, string> = {
  transcript: "bg-accent-teal/15 text-accent-teal",
  transcript_keyword: "bg-accent-teal/10 text-accent-teal",
  clip: "bg-accent-amber/15 text-accent-amber",
  metadata: "bg-sand text-text-primary",
  content: "bg-warm-light text-text-primary",
  text_content_keyword: "bg-warm-light/60 text-text-primary",
};

function MatchBadge({ type, label }: { type: string; label: string }) {
  const style = MATCH_TYPE_STYLES[type] ?? "bg-sand text-text-primary";
  return (
    <span
      className={`inline-flex rounded-md px-1.5 py-0.5 text-[10px] font-medium ${style}`}
    >
      {label}
    </span>
  );
}

function TimestampLink({
  seconds,
  fileId,
  onClick,
}: {
  seconds: number;
  fileId: string;
  onClick: (url: string) => void;
}) {
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        e.preventDefault();
        onClick(`/files/${fileId}?t=${Math.floor(seconds)}`);
      }}
      className="rounded-md px-1.5 py-0.5 text-[10px] font-medium text-accent transition-colors hover:bg-accent/10"
    >
      {formatDuration(seconds)}
    </button>
  );
}

export function MatchOverlay({
  result,
  onSelect,
  matchLabels,
  matchedPagesLabel,
}: {
  result: SemanticSearchResult;
  onSelect: (url: string) => void;
  matchLabels: Record<string, string>;
  /** Localised "p.1, 3, 5" prefix label. */
  matchedPagesLabel: (pages: number[]) => string;
}) {
  const matchedPages = [
    ...new Set(
      result.segments
        .flatMap((seg) => seg.matches)
        .filter((m) => m.page != null)
        .map((m) => m.page as number),
    ),
  ].sort((a, b) => a - b);

  const timestamps = result.segments
    .filter(
      (seg): seg is SemanticSearchSegment & { time_range: [number, number] } =>
        seg.time_range != null && seg.time_range[0] > 0,
    )
    .slice(0, 3);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-1">
        {result.match_types.map((type) => (
          <MatchBadge
            key={type}
            type={type}
            label={matchLabels[type] ?? type}
          />
        ))}
      </div>
      {timestamps.length > 0 && (
        <div className="flex flex-wrap gap-0.5">
          {timestamps.map((seg) => (
            <TimestampLink
              key={seg.time_range[0]}
              seconds={seg.time_range[0]}
              fileId={result.file_id}
              onClick={onSelect}
            />
          ))}
        </div>
      )}
      {matchedPages.length > 0 && (
        <p className="text-[11px] text-text-muted">
          {matchedPagesLabel(matchedPages)}
        </p>
      )}
    </div>
  );
}
