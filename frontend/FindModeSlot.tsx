"use client";

/**
 * FindModeSlot — search-modes entry that hands off to the Find page.
 *
 * Spec: ``2026-04-30-intelligence-find-mode.md`` §3.1 (UI / モード切替).
 *
 * Renders only when:
 *  - ``intelligence.features.rag === true`` (Find depends on Stage A
 *    + C LLM calls, gated by the same flag as Ask), and
 *  - ``llm.enabled === true``, and
 *  - the user has typed a non-empty query (no point handing off with
 *    no seed).
 *
 * Find is a side door (handoff to a different page), so it takes the last
 * row of the results bar's `…` menu and asks the menu to close first.
 */

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";
import { ListFilter } from "lucide-react";

import { ActionMenuItem } from "@/components/ActionMenuItem";
import { getIntelligenceStatus } from "./api";
import { getEnabledAddons } from "@/lib/addons";

interface FindModeSlotProps {
  query: string;
  drive: string;
  filter: string;
  onSelect: (url: string) => void;
  closeMenu: () => void;
}

export default function FindModeSlot({
  query,
  drive,
  onSelect,
  closeMenu,
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
    ])
      .then(([status, addons]) => {
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
      })
      .catch(() => {
        // A failed probe is an answer too: no row, and no gate that stays
        // pending for the life of the open menu.
        if (!cancelled) setRagGate(false);
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

  return (
    <ActionMenuItem
      icon={ListFilter}
      label={t("pageCta")}
      onClick={() => {
        closeMenu();
        onSelect(href);
      }}
    />
  );
}
