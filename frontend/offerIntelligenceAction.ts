"use client";

import { useTranslations } from "next-intl";
import {
  BookOpen,
  FileText,
  Image as ImageIcon,
  ListVideo,
  Sparkles,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { useOfferFileAiAction } from "@/lib/fileAiActions";

export type IntelligenceAiActionKind =
  | "tags"
  | "summary"
  | "detailedSummary"
  | "chapters"
  | "visualDescription";

/** Same icon the section itself uses, so the menu previews the result. */
const ACTIONS: Record<
  IntelligenceAiActionKind,
  { id: string; order: number; icon: LucideIcon }
> = {
  tags: { id: "intelligence.tags", order: 10, icon: Sparkles },
  summary: { id: "intelligence.summary", order: 20, icon: BookOpen },
  detailedSummary: {
    id: "intelligence.detailedSummary",
    order: 30,
    icon: FileText,
  },
  chapters: { id: "intelligence.chapters", order: 40, icon: ListVideo },
  visualDescription: {
    id: "intelligence.visualDescription",
    order: 50,
    icon: ImageIcon,
  },
};

/** Offer one of intelligence's generators to the file's "AI" menu. */
export function useOfferIntelligenceAction(params: {
  fileId: string;
  kind: IntelligenceAiActionKind;
  /** Message key in the `file` namespace. */
  labelKey: string;
  active: boolean;
  busy?: boolean;
  run: () => void;
}): void {
  const { fileId, kind, labelKey, active, busy, run } = params;
  const t = useTranslations("file");
  const { id, order, icon } = ACTIONS[kind];
  useOfferFileAiAction({
    fileId,
    id,
    label: t(labelKey),
    icon,
    order,
    active,
    busy,
    run,
  });
}
