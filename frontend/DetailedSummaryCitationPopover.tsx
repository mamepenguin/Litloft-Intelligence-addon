"use client";

/**
 * Citation marker for a detailed-summary segment.
 *
 * The filename still says "Popover" for backwards import stability,
 * but the UI is now a tiny SVG dot that flips a per-section accordion
 * in `CitationRailContext`. No hover, no pin, no overlay —
 * click (or Enter on the containing segment) opens the in-flow
 * excerpt panel beneath the citing line.
 *
 * Visual vocabulary:
 *   - strong (top_score >= 0.90): solid teal circle.
 *   - weak   (top_score <  0.90): half-filled amber circle with a
 *     dashed ring — communicates "verify me" without being alarming.
 *   - missing (has_citation = false): no marker rendered at all (it's
 *     a retrieval outcome, not a hallucination warning — see hako
 *     commit 8d38f88 rationale).
 *
 * Verify OFF hides the dot with ``visibility: hidden`` — never
 * ``display: none`` — because the 14px slot anchors the end-cap of
 * the preceding text. Collapsing the slot would reflow the sentence
 * and double-punctuate (the mockup wraps markers alongside the
 * period/comma by design). The slot also renders for has_citation=false
 * only when Verify is ON is an intentional non-goal: we hide missing
 * markers completely in both states.
 */

import { useCallback, useId } from "react";
import { useTranslations } from "next-intl";

import { useCitationRail, CITATION_STRONG_THRESHOLD } from "./CitationRailContext";
import type { DetailedSummaryCitation } from "./api";

type CitationTier = "strong" | "weak";

function deriveTier(citation: DetailedSummaryCitation): CitationTier {
  return citation.top_score >= CITATION_STRONG_THRESHOLD ? "strong" : "weak";
}

interface DetailedSummaryCitationPopoverProps {
  citation: DetailedSummaryCitation;
}

export function DetailedSummaryCitationPopover({
  citation,
}: DetailedSummaryCitationPopoverProps) {
  const t = useTranslations("detailedSummary");
  const { verify, toggle, isExpanded } = useCitationRail();
  // Unique id for the weak tier's half-moon clipPath. Required because
  // ``clip-path: url(#hc-…)`` refs are global to the document; two
  // weak markers sharing the same id would both clip to whichever was
  // defined last (typically a no-op on the second). useId returns a
  // stable, unique value per component instance.
  const reactId = useId();
  const clipId = `hv-half-${reactId.replace(/:/g, "")}`;

  const isActive = isExpanded(citation.section_path);

  const handleClick = useCallback(() => {
    toggle(citation);
  }, [toggle, citation]);

  // Missing-citation segments render nothing. This keeps the marker
  // set honest with the mockup (no empty slots for no-citation prose)
  // and the old `citation.has_citation === false` path stays silent.
  if (!citation.has_citation) return null;

  const tier = deriveTier(citation);

  // When Verify is OFF we still render the `<button>` so the focus /
  // toggle API stays stable, but the SVG inside is hidden via
  // visibility so the layout slot anchors the end-cap of the
  // preceding text without the dot glyph drawing.
  const visibilityStyle: React.CSSProperties = verify
    ? {}
    : { visibility: "hidden" };

  const ariaLabel =
    tier === "strong"
      ? t("citations.markerStrong", { defaultMessage: "Strong source citation" })
      : t("citations.markerWeak", {
          defaultMessage: "Weak source citation — verify",
        });

  // The 14×14 slot is reserved via fixed width/height on the wrapper
  // so Verify ON/OFF and missing/strong/weak all occupy the same
  // horizontal space — preventing table column widths and paragraph
  // text-wrap from shifting when Verify is toggled.
  return (
    <button
      type="button"
      onClick={handleClick}
      aria-pressed={isActive}
      aria-label={ariaLabel}
      title={tier === "weak" ? ariaLabel : undefined}
      className="ml-1.5 inline-flex h-[14px] w-[14px] items-center justify-center align-middle transition-[filter] hover:brightness-110"
      style={{ verticalAlign: "-2px", ...visibilityStyle }}
      data-citation-marker={`linked-${tier}`}
      data-citation-tier={tier}
    >
      {tier === "strong" ? (
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          aria-hidden
          focusable="false"
        >
          <circle cx="12" cy="12" r="5" fill="var(--accent-teal)" />
        </svg>
      ) : (
        // Weak = amber half-moon per docs/citation-ui-mockup.html §1.
        // A dashed ring around the outside + a solid left-half fill,
        // delivered via SVG clipPath on the inner circle. Shape itself
        // (● vs ◐) carries the tier so the UI still discriminates for
        // viewers with reduced colour sensitivity.
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          aria-hidden
          focusable="false"
        >
          <defs>
            <clipPath id={clipId}>
              <rect x="0" y="0" width="12" height="24" />
            </clipPath>
          </defs>
          <circle
            cx="12"
            cy="12"
            r="5"
            fill="none"
            stroke="var(--accent-amber)"
            strokeWidth="1.2"
            strokeDasharray="2 1.2"
          />
          <circle
            cx="12"
            cy="12"
            r="5"
            fill="var(--accent-amber)"
            clipPath={`url(#${clipId})`}
          />
        </svg>
      )}
    </button>
  );
}
