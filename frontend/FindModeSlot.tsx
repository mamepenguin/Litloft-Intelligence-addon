"use client";

/**
 * FindModeSlot — search-modes entry that hands off to the Find page.
 *
 * Spec: ``2026-04-30-intelligence-find-mode.md`` §3.1 (UI / モード切替).
 *
 * Mirrors the existing Ask handoff inside ``SemanticSearchSlot`` but
 * dedicated to Find. Renders only when:
 *  - ``intelligence.features.rag === true`` (Find depends on Stage A
 *    + C LLM calls, gated by the same flag as Ask), and
 *  - ``llm.enabled === true``, and
 *  - the user has typed a non-empty query (no point handing off with
 *    no seed).
 *
 * Page layout is a single right-aligned chip — Find is a side door
 * (handoff to a different page), so it stays visually subordinate to
 * the actual results on the search page.
 */

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { ListFilter } from "lucide-react";

import { getIntelligenceStatus } from "./api";
import { getEnabledAddons } from "@/lib/addons";

type SlotContext = "popup" | "page";

interface FindModeSlotProps {
  query: string;
  drive: string;
  filter: string;
  onSelect: (url: string) => void;
  /** Layout mode. "popup" (default) = compact list row for the search
   *  modal, "page" = prominent section heading + CTA card for the
   *  /drive/<name>/search page. */
  context?: SlotContext;
}

export default function FindModeSlot({
  query,
  drive,
  onSelect,
  context = "popup",
}: FindModeSlotProps) {
  const t = useTranslations("find");
  // ``ragGate`` is ``null`` when we have not finished probing yet,
  // ``true`` when Find should render, ``false`` when we have a
  // definitive signal that it should stay hidden. The probe combines
  // two sources because the addon's own ``/status`` is admin-gated
  // and returns ``null`` for any viewer that has not unlocked every
  // protected drive — which would otherwise hide Find from the very
  // viewers it is meant to serve.
  const [ragGate, setRagGate] = useState<boolean | null>(null);

  useEffect(() => {
    if (!drive) {
      setRagGate(false);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    Promise.all([
      getIntelligenceStatus(drive, controller.signal),
      getEnabledAddons(drive),
    ]).then(([status, addons]) => {
      if (cancelled) return;
      if (status === null) {
        // /status unreachable (most often the admin gate). Fall back
        // to "is intelligence enabled for this drive?" — Find then
        // surfaces and any real LLM-not-configured error appears on
        // the dedicated find page instead of in this slot.
        setRagGate(Boolean(addons["intelligence"]));
        return;
      }
      const ragEnabled = status.features?.rag === true;
      const llmEnabled = status.llm?.enabled === true;
      setRagGate(ragEnabled && llmEnabled);
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [drive]);

  if (ragGate !== true) return null;

  const trimmed = query.trim();
  if (trimmed.length === 0) return null;

  const href = `/drive/${encodeURIComponent(drive)}/addons/intelligence/find?q=${encodeURIComponent(trimmed)}`;

  if (context === "page") {
    return (
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => onSelect(href)}
          aria-label={`Find: ${trimmed}`}
          title={t("pageDescription", { query: trimmed })}
          className="inline-flex items-center gap-1.5 rounded-full bg-sand px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-sand-hover"
        >
          <ListFilter size={14} className="flex-shrink-0 text-accent-teal" />
          <span className="truncate">{t("pageCta")}</span>
        </button>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={() => onSelect(href)}
      aria-label={`Find: ${trimmed}`}
      className="flex w-full items-center gap-2 border-t border-bg-border px-4 py-2.5 text-left text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
    >
      <ListFilter size={12} className="flex-shrink-0 text-accent-teal" />
      <span className="truncate">{t("button", { query: trimmed })}</span>
    </button>
  );
}
