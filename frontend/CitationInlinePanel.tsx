"use client";

/**
 * In-flow accordion panel that renders the excerpt for an expanded
 * citation, sitting directly below the cited segment.
 *
 * Design shift (Phase 4 UI overhaul): the old implementation used an
 * absolute-positioned overlay so the surrounding layout didn't shift
 * when opening / closing. The new model is in-flow — the segment
 * pushes the following content down while expanded. Tradeoff: minor
 * layout movement on toggle; payoff: the relationship between cited
 * line and excerpt is unambiguous, the panel never escapes container
 * clipping, and scroll-into-view keeps the reader anchored.
 *
 * Also dropped with this shift:
 *   - adjacent-chunk highlight extension (Stage 6). The signal was
 *     too noisy against curated GT (~25% precision at idx_gap=1); the
 *     evals didn't reproduce Stage 6's claimed lift on real queries
 *     once the UI was redesigned. Keep the pipeline's excerpt slice
 *     as-is; don't visually grow the highlight into neighbours.
 *   - the close ✕ button. Clicking the marker again collapses the
 *     panel; Esc on the section also collapses.
 */

import { useTranslations } from "next-intl";
import { PlayCircle, ExternalLink, Copy } from "lucide-react";

import { useCitationRail, CITATION_STRONG_THRESHOLD } from "./CitationRailContext";
import type { CitationFetchState } from "./CitationRailContext";
import type { CitationChunkExcerpt, DetailedSummaryCitation } from "./api";

interface CitationInlinePanelProps {
  sectionPath: string;
  /**
   * The citation bound to this segment. Needed so the panel can
   * render tier chips, locator metadata, and pick the excerpt slice
   * without having to reach back into the provider.
   */
  citation: DetailedSummaryCitation;
  /**
   * Segment type context — shapes the outer frame.
   *   - paragraph / bullet: rounded dashed (weak) or solid (strong) card.
   *   - table: no outer card; the cited row already carries the
   *     border-left accent per DESIGN.md table H3 convention.
   */
  segmentType: "paragraph" | "bullet" | "table";
  videoRef?: React.RefObject<HTMLVideoElement | null> | null;
  onJump?: (excerpt: CitationChunkExcerpt) => boolean | void;
}

export function CitationInlinePanel({
  sectionPath,
  citation,
  segmentType,
  videoRef,
  onJump,
}: CitationInlinePanelProps) {
  const t = useTranslations("detailedSummary");
  const { isExpanded, excerptState } = useCitationRail();

  if (!isExpanded(sectionPath)) return null;

  const state = excerptState(sectionPath);
  const tier: "strong" | "weak" =
    citation.top_score >= CITATION_STRONG_THRESHOLD ? "strong" : "weak";

  const containerClass =
    segmentType === "table"
      ? "mt-2"
      : tier === "strong"
        ? "mt-2 rounded-md border border-bg-border bg-bg-card p-3 text-[13px]"
        : "mt-2 rounded-md border border-dashed bg-bg-card p-3 text-[13px]";

  const containerStyle: React.CSSProperties =
    segmentType === "table"
      ? {}
      : tier === "weak"
        ? {
            borderColor:
              "color-mix(in srgb, var(--accent-amber) 55%, var(--bg-border))",
          }
        : {};

  return (
    <aside
      role="region"
      aria-label={t("inline.label", { defaultMessage: "Citation excerpt" })}
      className={containerClass}
      style={containerStyle}
      data-citation-panel={tier}
    >
      <PanelMeta tier={tier} citation={citation} />
      <InlineExcerptBody
        state={state}
        videoRef={videoRef}
        onJump={onJump}
      />
    </aside>
  );
}

function PanelMeta({
  tier,
  citation,
}: {
  tier: "strong" | "weak";
  citation: DetailedSummaryCitation;
}) {
  const t = useTranslations("detailedSummary");
  const tierLabel =
    tier === "strong"
      ? t("citations.tierStrong", { defaultMessage: "strong" })
      : t("citations.tierWeak", { defaultMessage: "weak" });

  // Source-type hint is derived from chunk_id prefix. Falls back to
  // "excerpt" for rows with no prefix (rare).
  const sourceType = (() => {
    const id = citation.chunk_ids[0] ?? "";
    if (id.startsWith("transcript:")) {
      return t("citations.sourceTranscript", { defaultMessage: "transcript" });
    }
    if (id.startsWith("document:")) {
      return t("citations.sourceDocument", { defaultMessage: "document" });
    }
    if (id.startsWith("table:")) {
      return t("citations.sourceTable", { defaultMessage: "table" });
    }
    return t("citations.sourceExcerpt", { defaultMessage: "excerpt" });
  })();

  const chipClass =
    tier === "strong"
      ? "inline-flex items-center rounded border px-1.5 py-0.5 text-[11px]"
      : "inline-flex items-center rounded border border-dashed px-1.5 py-0.5 text-[11px]";
  const chipStyle: React.CSSProperties =
    tier === "strong"
      ? {
          backgroundColor:
            "color-mix(in srgb, var(--accent-teal) 18%, transparent)",
          color: "var(--accent-teal)",
          borderColor:
            "color-mix(in srgb, var(--accent-teal) 40%, transparent)",
        }
      : {
          backgroundColor:
            "color-mix(in srgb, var(--accent-amber) 12%, transparent)",
          color: "var(--accent-amber)",
          borderColor:
            "color-mix(in srgb, var(--accent-amber) 55%, transparent)",
        };

  return (
    <div className="mb-2 flex items-center gap-2 text-[11px] text-text-muted">
      <span className={chipClass} style={chipStyle}>
        {tierLabel}
      </span>
      <span>{sourceType}</span>
    </div>
  );
}

function InlineExcerptBody({
  state,
  videoRef,
  onJump,
}: {
  state: CitationFetchState;
  videoRef?: React.RefObject<HTMLVideoElement | null> | null;
  onJump?: (excerpt: CitationChunkExcerpt) => boolean | void;
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
      if (handled) return;
    }
    const video = videoRef?.current;
    if (video && excerpt.start_time != null) {
      video.currentTime = excerpt.start_time;
      void video.play?.().catch(() => {});
    }
  };

  const handleCopy = () => {
    const text = [excerpt.prefix, excerpt.target, excerpt.suffix]
      .filter(Boolean)
      .join("");
    try {
      void navigator.clipboard?.writeText(text);
    } catch {
      // Clipboard unavailable — silent fail is fine for a convenience
      // shortcut.
    }
  };

  const locator = (() => {
    if (excerpt.start_time != null) return formatTimestamp(excerpt.start_time);
    if (excerpt.page != null) {
      return t("citations.pageLabel", {
        defaultMessage: "Page {page}",
        page: excerpt.page,
      });
    }
    return "";
  })();

  return (
    <div className="space-y-2">
      <p
        className="whitespace-pre-wrap leading-relaxed cursor-pointer"
        onClick={jumpDisabled ? undefined : handleJump}
      >
        {excerpt.prefix && (
          <span className="text-text-muted/80">{excerpt.prefix}</span>
        )}
        <mark
          data-testid="citation-target"
          className="rounded px-0.5 text-text-primary"
          style={{ backgroundColor: "var(--highlight-bg)" }}
        >
          {excerpt.target}
        </mark>
        {excerpt.suffix && (
          <span className="text-text-muted/80">{excerpt.suffix}</span>
        )}
      </p>
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-text-muted">
        <span>{locator}</span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted hover:bg-bg-elevated hover:text-text-primary"
          >
            <Copy size={11} />
            {t("citations.copyExcerpt", { defaultMessage: "Copy excerpt" })}
          </button>
          <button
            type="button"
            onClick={handleJump}
            disabled={jumpDisabled}
            className="inline-flex items-center gap-1 rounded bg-accent-teal px-2 py-1 text-[11px] text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <PlayCircle size={11} />
            {t("citations.jump", { defaultMessage: "Jump" })}
          </button>
          {excerpt.file_id && (
            <a
              href={`/files/${excerpt.file_id}`}
              className="inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] text-text-muted hover:bg-bg-elevated hover:text-text-primary"
              target="_blank"
              rel="noopener noreferrer"
            >
              <ExternalLink size={11} />
              {t("citations.openFile", { defaultMessage: "Open file" })}
            </a>
          )}
        </div>
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
