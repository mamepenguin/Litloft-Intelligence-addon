"use client";

import { RefreshCw } from "lucide-react";

/**
 * "This is being made right now."
 *
 * The five generators no longer head an empty box, so a run started
 * from the action row's "AI" menu had nothing to show for itself: the
 * menu closes on selection, and a 16px pulse on the button is not an
 * answer to "did that work?". A run is live state, which is exactly the
 * kind of thing a section is still for — and saying it the same way in
 * all five keeps one concept looking like one concept.
 */
export function GeneratingRow({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2">
      <RefreshCw size={11} className="animate-spin text-text-muted" aria-hidden="true" />
      <span className="text-xs text-text-muted">{label}</span>
    </div>
  );
}
