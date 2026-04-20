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
import { PlayCircle, Copy } from "lucide-react";

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

  // DESIGN.md §5 / mockup rev2: cards use rounded-xl (12px).
  //
  //   paragraph / bullet (default accordion):
  //     - strong: 1px solid bg-border, no left-edge accent
  //     - weak:   1px dashed amber (all four sides) per mockup rule
  //               ``.segment[data-tier="weak"] > .accordion``
  //
  //   table (H3 variant, chosen per user confirm):
  //     - strong: 1px solid bg-border (no amber accent)
  //     - weak:   1px solid bg-border + border-left 3px solid amber
  //               mirroring the cited row's own first-cell accent so
  //               the expansion row reads as part of the table frame
  //
  // Both layouts sit on var(--bg-elevated) with 14px/16px padding so
  // they match the mockup's ``.accordion`` / ``.accordion-h3`` specs.
  // Vertical margins sit OUTSIDE the elevated surface. mt-3 (12px)
  // on top gives comfortable separation between the segment text and
  // the accordion; mb-1 (4px) on bottom pairs with the segment's own
  // 4px padding-bottom for a total 8px gap under the accordion, so
  // the next li / paragraph isn't crushed against the elevated
  // surface. Inner padding p-5 (20px) gives the meta row + excerpt
  // card breathing room inside the coloured surface itself.
  const paragraphBulletClass =
    tier === "strong"
      ? "mt-3 mb-1 rounded-xl border border-bg-border bg-bg-elevated p-5 text-[13px] animate-fade-in"
      : "mt-3 mb-1 rounded-xl border border-dashed bg-bg-elevated p-5 text-[13px] animate-fade-in";

  // Table rows drop the outer margin — a <td colspan> containing the
  // accordion already sits on its own table-row so neighbouring content
  // is positioned by the table-layout, not margin.
  const tableAccordionClass =
    "my-2 rounded-xl border border-bg-border bg-bg-elevated p-5 text-[13px] animate-fade-in";

  const containerClass =
    segmentType === "table" ? tableAccordionClass : paragraphBulletClass;

  const containerStyle: React.CSSProperties = (() => {
    if (segmentType === "table") {
      return tier === "weak"
        ? { borderLeft: "3px solid var(--accent-amber)" }
        : {};
    }
    // paragraph / bullet
    if (tier === "weak") {
      return {
        borderColor:
          "color-mix(in srgb, var(--accent-amber) 55%, var(--bg-border))",
      };
    }
    return {};
  })();

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

  // Chip — rounded-2xl pill per mockup `.chip` (no bg fill; colour +
  // border carries the tier). font-weight 600 is the spec.
  const chipClass =
    tier === "strong"
      ? "inline-flex items-center gap-1 rounded-2xl border bg-bg-card px-2.5 py-0.5 text-[11px] font-semibold"
      : "inline-flex items-center gap-1 rounded-2xl border border-dashed bg-bg-card px-2.5 py-0.5 text-[11px] font-semibold";
  const chipStyle: React.CSSProperties =
    tier === "strong"
      ? {
          color: "var(--accent-teal)",
          borderColor:
            "color-mix(in srgb, var(--accent-teal) 30%, transparent)",
        }
      : {
          color: "var(--accent-amber)",
          borderColor:
            "color-mix(in srgb, var(--accent-amber) 55%, transparent)",
        };

  return (
    <div className="mb-2 flex items-center justify-between gap-2.5 text-xs text-text-muted">
      <span className="inline-flex items-center gap-2">
        <span className={chipClass} style={chipStyle}>
          {tierLabel}
        </span>
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
  // An all-empty excerpt payload is functionally "no excerpt" even
  // though the server returned a row. This happens when a weak-tier
  // citation points at a chunk whose prefix/target/suffix are all
  // blank (e.g. an intro-applause snap fallback) — without this guard
  // the panel would paint a blank excerpt card when the user hits
  // "Needs check" and every weak citation landed on such a row.
  const hasContent =
    !!excerpt &&
    Boolean(
      (excerpt.prefix && excerpt.prefix.trim()) ||
        (excerpt.target && excerpt.target.trim()) ||
        (excerpt.suffix && excerpt.suffix.trim()),
    );
  if (!excerpt || !hasContent) {
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

  // Mockup `.acc-excerpt`: the whole excerpt card is the jump target.
  // Head row carries locator (tabular-nums, weight 600) on the left and
  // a "ジャンプ ▶" affordance on the right (accent colour, weight 650).
  // Body is 14px / line-height 1.7, prefix+suffix muted, target on
  // highlight-bg. `.acc-actions` below hosts Copy / Open-file as
  // `rounded-2xl` ghost buttons — Jump is NOT duplicated here, the
  // whole card already handles that.
  return (
    <div>
      <button
        type="button"
        onClick={jumpDisabled ? undefined : handleJump}
        disabled={jumpDisabled}
        className="block w-full cursor-pointer rounded-xl border border-bg-border bg-bg-card px-3.5 py-3 text-left transition-colors hover:border-accent disabled:cursor-not-allowed disabled:opacity-60"
      >
        <div className="mb-1.5 flex items-center justify-between gap-2 text-xs text-text-muted">
          <span className="font-semibold tabular-nums text-text-primary">
            {locator}
          </span>
          {!jumpDisabled && (
            <span
              className="inline-flex items-center gap-1 font-[650]"
              style={{ color: "var(--accent)" }}
            >
              {t("citations.jump", { defaultMessage: "Jump" })}
              <PlayCircle size={12} />
            </span>
          )}
        </div>
        <p className="whitespace-pre-wrap text-[14px] leading-[1.7] text-text-primary">
          {excerpt.prefix && (
            <span className="text-text-muted">{excerpt.prefix}</span>
          )}
          <mark
            data-testid="citation-target"
            className="rounded-[3px] px-0.5 font-medium text-text-primary"
            style={{ backgroundColor: "var(--highlight-bg)" }}
          >
            {excerpt.target}
          </mark>
          {excerpt.suffix && (
            <span className="text-text-muted">{excerpt.suffix}</span>
          )}
        </p>
      </button>
      <div className="mt-2 flex items-center gap-2.5">
        <button
          type="button"
          onClick={handleCopy}
          className="inline-flex items-center gap-1 rounded-2xl border border-bg-border bg-transparent px-3 py-1 text-xs text-text-muted transition-colors hover:border-accent hover:text-text-primary"
        >
          <Copy size={11} />
          {t("citations.copyExcerpt", { defaultMessage: "Copy excerpt" })}
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
