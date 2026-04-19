"use client";

/**
 * Inline citation marker for a detailed-summary segment.
 *
 * Despite the file name (kept for import stability), this is no longer
 * a self-contained popover: excerpt display lives in
 * `CitationInlinePanel`, which now opens as an absolutely-positioned
 * overlay directly beneath the citing segment (so the surrounding
 * layout doesn't shift when the panel opens or closes).
 *
 * Interaction model:
 *   - Hover on the marker opens the overlay; moving the cursor into
 *     the panel body keeps it open (the panel handles mouseenter too).
 *     mouseleave schedules a 160 ms grace close that can be cancelled
 *     by re-entering either the marker or the panel.
 *   - Click pins the overlay: `scheduleClose` becomes a no-op until
 *     the user re-clicks the marker, clicks outside, or hits Escape.
 *     Pinning is the mechanism that lets the reader actually reach
 *     the "Jump" button without a cursor-race.
 *   - Focus via keyboard mirrors hover (onFocus opens, onBlur
 *     schedules close), so tab-navigating the summary still surfaces
 *     the excerpt.
 */

import { useCallback } from "react";
import { useTranslations } from "next-intl";
import { Link2, AlertTriangle } from "lucide-react";

import { useCitationRail } from "./CitationRailContext";
import type { DetailedSummaryCitation } from "./api";

// Confidence tier derived from top_score. Calibrated against citation
// eval baseline (ruri-v3-30m, N=69, 2026-04-19): top_score ≥ 0.90 hits
// location offset 0 at 86% vs ~68% for [0.80, 0.90). See
// docs/CITATION-PIPELINE.md Stage 5 and hako Uxs06_pOPfbkGtvwIK_Vq.
const CITATION_STRONG_THRESHOLD = 0.9;

type CitationTier = "strong" | "weak" | "missing";

function deriveTier(citation: DetailedSummaryCitation): CitationTier {
  if (!citation.has_citation) return "missing";
  return citation.top_score >= CITATION_STRONG_THRESHOLD ? "strong" : "weak";
}

interface DetailedSummaryCitationPopoverProps {
  citation: DetailedSummaryCitation;
}

export function DetailedSummaryCitationPopover({
  citation,
}: DetailedSummaryCitationPopoverProps) {
  const t = useTranslations("detailedSummary");
  const { setActive, clearActive, scheduleClose, cancelClose, active } =
    useCitationRail();

  const hasCitation = citation.has_citation;
  const tier = deriveTier(citation);
  const isActive =
    active?.citation.section_path === citation.section_path && hasCitation;
  const isPinned = isActive && active?.pinned === true;

  const handlePointerEnter = useCallback(() => {
    if (!hasCitation) return;
    cancelClose();
    // Hover-open is transient (pin=false). If the citation is already
    // pinned we still call setActive to re-sync, but the existing pin
    // is preserved via the early-return on same-citation below —
    // actually simpler to just cancel the close and leave state alone
    // when the same citation is already active.
    if (isActive) return;
    setActive(citation);
  }, [hasCitation, isActive, cancelClose, setActive, citation]);

  const handlePointerLeave = useCallback(() => {
    if (!hasCitation) return;
    scheduleClose();
  }, [hasCitation, scheduleClose]);

  const handleClick = useCallback(() => {
    if (!hasCitation) return;
    if (isPinned) {
      // Toggle off a pinned activation.
      clearActive();
      return;
    }
    // Promote a hover-open (or a fresh activation) to pinned.
    setActive(citation, { pin: true });
  }, [hasCitation, isPinned, setActive, clearActive, citation]);

  return (
    <button
      type="button"
      onMouseEnter={handlePointerEnter}
      onMouseLeave={handlePointerLeave}
      onFocus={handlePointerEnter}
      onBlur={handlePointerLeave}
      onClick={handleClick}
      aria-pressed={isActive}
      className={`mx-1 inline-flex h-4 w-4 items-center justify-center rounded align-middle transition-colors ${
        hasCitation
          ? isActive
            ? "text-accent-teal"
            : "text-accent-teal/60 hover:text-accent-teal"
          : "text-accent-amber hover:text-accent-amber/80 cursor-default"
      }`}
      aria-label={
        tier === "strong"
          ? t("citations.linkLabel", { defaultMessage: "Show citation" })
          : tier === "weak"
            ? t("citations.weakLinkLabel", {
                defaultMessage: "Weak source match — verify",
              })
            : t("citations.noCitation", {
                defaultMessage:
                  "No strong source match found. This may be inaccurate.",
              })
      }
      title={
        tier === "strong"
          ? undefined
          : tier === "weak"
            ? t("citations.weakLinkLabel", {
                defaultMessage: "Weak source match — verify",
              })
            : t("citations.noCitation", {
                defaultMessage:
                  "No strong source match found. This may be inaccurate.",
              })
      }
      data-citation-marker={
        tier === "missing" ? "missing" : `linked-${tier}`
      }
    >
      {hasCitation ? (
        <Link2
          size={11}
          aria-hidden
          className={
            tier === "weak" ? "[stroke-dasharray:2_1.5]" : undefined
          }
        />
      ) : (
        <AlertTriangle size={11} aria-hidden />
      )}
    </button>
  );
}
