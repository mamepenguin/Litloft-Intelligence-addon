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
import type { IntelligenceStatus } from "./api";

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
  const [status, setStatus] = useState<IntelligenceStatus | null>(null);
  const [statusReady, setStatusReady] = useState(false);

  useEffect(() => {
    if (!drive) {
      setStatus(null);
      setStatusReady(true);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    getIntelligenceStatus(drive, controller.signal).then((res) => {
      if (cancelled) return;
      setStatus(res);
      setStatusReady(true);
    });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [drive]);

  if (!statusReady) return null;
  const ragEnabled = status?.features?.rag === true;
  const llmEnabled = status?.llm?.enabled === true;
  if (!ragEnabled || !llmEnabled) return null;

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
