"use client";

/**
 * FindChip — a single removable chip surface for the Find page.
 *
 * Spec: ``2026-04-30-intelligence-find-mode.md`` §3.1 (chip の意味と
 * 編集セマンティクス). Each chip carries a label + an × button. The
 * × invokes ``onRemove`` so the parent page can rebuild ``overrides``
 * and re-POST. ``data-slot`` lets tests + style hooks identify which
 * decomposed slot the chip represents without parsing the label.
 */

import { X } from "lucide-react";

export type FindChipSlot =
  | "time_range"
  | "personal_scope"
  | "file_type_hint"
  | "semantic_query";

interface FindChipProps {
  label: string;
  slot: FindChipSlot;
  onRemove: () => void;
}

export default function FindChip({ label, slot, onRemove }: FindChipProps) {
  return (
    <span
      data-slot={slot}
      className="inline-flex items-center gap-1 rounded-full bg-bg-elevated px-2 py-0.5 text-xs text-text-primary"
    >
      <span className="break-anywhere">{label}</span>
      <button
        type="button"
        aria-label={`Remove ${label}`}
        onClick={onRemove}
        className="inline-flex h-4 w-4 items-center justify-center rounded-full text-text-muted transition-colors hover:bg-bg-card hover:text-text-primary"
      >
        <X size={11} aria-hidden="true" />
      </button>
    </span>
  );
}
