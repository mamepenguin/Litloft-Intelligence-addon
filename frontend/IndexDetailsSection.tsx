"use client";

import { useEffect, useState } from "react";
import { ChevronRight, Database } from "lucide-react";
import { useTranslations } from "next-intl";

import { getFileIndexDetails } from "./api";
import type { IndexDetailsResponse, IndexDetailType } from "./api";
import { formatDuration } from "@/lib/format";

interface IndexDetailsSectionProps {
  fileId: string;
  drive: string;
}

const TYPE_LABELS: Record<string, string> = {
  metadata: "Metadata",
  clip: "CLIP",
  whisper: "Whisper",
  text_content: "Text",
  blip_caption: "BLIP Caption",
};

// Category indicator dots. The visual cue is "AI-generated embedding"
// (accent-teal) vs "filesystem-side metadata" (text-muted/warm) —
// per-type hues collapse into the warm palette per DESIGN.md §2.2.
const TYPE_COLORS: Record<string, string> = {
  metadata: "bg-text-muted",
  clip: "bg-accent-teal",
  whisper: "bg-accent-teal",
  text_content: "bg-accent-teal",
  blip_caption: "bg-accent-amber",
};

export default function IndexDetailsSection({ fileId, drive }: IndexDetailsSectionProps) {
  const t = useTranslations("searchIndex");
  const [data, setData] = useState<IndexDetailsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedType, setExpandedType] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    getFileIndexDetails(fileId, drive).then((res) => {
      setData(res.available ? res : null);
      setLoading(false);
    });
  }, [fileId, drive]);

  if (loading || !data) return null;

  const status = data.status!;
  const embeddings = data.embeddings!;

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-sm text-text-muted">
        <Database size={14} />
        <span>{t("indexTitle")}</span>
      </div>

      <div className="rounded-lg bg-bg-card p-3">
        {/* BLIP caption (if available) */}
        {embeddings.blip_caption && embeddings.blip_caption.count > 0 && (
          <div className="mb-3 rounded-lg bg-accent-amber/5 border border-accent-amber/20 px-3 py-2">
            <p className="text-xs text-accent-amber/80 italic">
              &quot;{embeddings.blip_caption.items[0]?.content_preview}&quot;
            </p>
          </div>
        )}

        {/* Status badges */}
        <div className="mb-3 flex flex-wrap gap-2">
          {(Object.entries(status) as [string, boolean][]).map(([key, done]) => (
            <span
              key={key}
              className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${
                done
                  ? "bg-accent-teal/10 text-accent-teal"
                  : "bg-accent-amber/10 text-accent-amber"
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${done ? "bg-accent-teal" : "bg-accent-amber"}`} />
              {TYPE_LABELS[key] || key}
            </span>
          ))}
        </div>

        {/* Embedding types */}
        <div className="space-y-1">
          {Object.entries(embeddings).map(([type, detail]: [string, IndexDetailType]) => {
            if (detail.count === 0) return null;
            const isExpanded = expandedType === type;
            return (
              <div key={type}>
                <button
                  onClick={() => setExpandedType(isExpanded ? null : type)}
                  className="flex w-full cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm transition-colors hover:bg-bg-primary"
                >
                  <ChevronRight
                    size={14}
                    className={`shrink-0 text-text-muted transition-transform ${isExpanded ? "rotate-90" : ""}`}
                  />
                  <span className={`h-2 w-2 shrink-0 rounded-full ${TYPE_COLORS[type] || "bg-text-muted"}`} />
                  <span className="text-text-primary">
                    {TYPE_LABELS[type] || type}
                  </span>
                  <span className="text-xs text-text-muted">
                    {detail.count}
                  </span>
                </button>
                {isExpanded && (
                  <div className="ml-8 mt-1 max-h-48 space-y-0.5 overflow-y-auto">
                    {detail.items.map((item, i) => (
                      <div
                        key={i}
                        className="flex gap-2 rounded-lg px-2 py-1 text-xs text-text-muted"
                      >
                        {item.start != null && (
                          <span className="shrink-0 font-mono">
                            {formatDuration(item.start)}
                          </span>
                        )}
                        <span className="min-w-0 flex-1 truncate">
                          {item.content_preview}
                        </span>
                      </div>
                    ))}
                    {detail.count > detail.items.length && (
                      <div className="px-2 py-1 text-xs text-text-muted italic">
                        +{detail.count - detail.items.length} {t("more")}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
