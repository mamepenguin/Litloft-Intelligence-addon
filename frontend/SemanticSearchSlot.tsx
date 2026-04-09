"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";

import { semanticSearch, getSearchStatus } from "./api";
import type { SemanticSearchResult, SemanticSearchSegment } from "./api";
import { formatDuration } from "@/lib/format";

interface SemanticSearchSlotProps {
  query: string;
  drive: string;
  filter: string;
  onSelect: (url: string) => void;
}

const MATCH_TYPE_STYLES: Record<string, string> = {
  transcript: "bg-blue-500/15 text-blue-400",
  transcript_keyword: "bg-cyan-500/15 text-cyan-400",
  clip: "bg-emerald-500/15 text-emerald-400",
  metadata: "bg-zinc-500/15 text-zinc-400",
  content: "bg-purple-500/15 text-purple-400",
  text_content_keyword: "bg-violet-500/15 text-violet-400",
};

function MatchBadge({ type, label }: { type: string; label: string }) {
  const style = MATCH_TYPE_STYLES[type] ?? "bg-zinc-500/15 text-zinc-400";
  return (
    <span className={`inline-flex rounded px-1.5 py-0.5 text-[10px] font-medium ${style}`}>
      {label}
    </span>
  );
}

function TimestampLink({
  seconds,
  fileId,
  onClick,
}: {
  seconds: number;
  fileId: string;
  onClick: (url: string) => void;
}) {
  const label = formatDuration(seconds);
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick(`/files/${fileId}?t=${Math.floor(seconds)}`);
      }}
      className="rounded px-1 py-0.5 text-[10px] font-medium text-accent hover:bg-accent/10 transition-colors"
    >
      {label}
    </button>
  );
}

function SemanticResultItem({
  result,
  onSelect,
  t,
}: {
  result: SemanticSearchResult;
  onSelect: (url: string) => void;
  t: (key: string, values?: Record<string, string | number>) => string;
}) {
  const matchLabels: Record<string, string> = {
    transcript: t("matchTranscript"),
    transcript_keyword: t("matchTranscriptKeyword"),
    clip: t("matchClip"),
    metadata: t("matchMetadata"),
    content: t("matchContent"),
    text_content_keyword: t("matchTextContentKeyword"),
  };

  const matchedPages = [
    ...new Set(
      result.segments
        .flatMap((seg) => seg.matches)
        .filter((m) => m.page != null)
        .map((m) => m.page as number)
    ),
  ].sort((a, b) => a - b);

  const timestamps = result.segments
    .filter((seg): seg is SemanticSearchSegment & { time_range: [number, number] } =>
      seg.time_range != null && seg.time_range[0] > 0,
    )
    .slice(0, 5);

  return (
    <button
      onClick={() => onSelect(`/files/${result.file_id}`)}
      className="flex w-full items-start gap-3 px-4 py-2.5 text-left transition-colors hover:bg-bg-elevated"
    >
      <img
        src={`/api/files/${result.file_id}/thumbnail`}
        alt=""
        className="h-10 w-16 flex-shrink-0 rounded bg-bg-elevated object-cover"
        onError={(e) => { e.currentTarget.style.display = "none"; }}
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-text-primary">{result.filename}</p>
        <div className="mt-0.5 flex flex-wrap items-center gap-1">
          {result.match_types.map((type) => (
            <MatchBadge
              key={type}
              type={type}
              label={matchLabels[type] ?? type}
            />
          ))}
        </div>
        {timestamps.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-0.5">
            {timestamps.map((seg) => (
              <TimestampLink
                key={seg.time_range[0]}
                seconds={seg.time_range[0]}
                fileId={result.file_id}
                onClick={onSelect}
              />
            ))}
          </div>
        )}
        {matchedPages.length > 0 && (
          <p className="mt-1 text-[11px] text-text-tertiary">
            {t("matchedPages", { pages: matchedPages.join(", ") })}
          </p>
        )}
      </div>
    </button>
  );
}

export default function SemanticSearchSlot({
  query,
  drive,
  filter,
  onSelect,
}: SemanticSearchSlotProps) {
  const t = useTranslations("search");
  const [available, setAvailable] = useState<boolean | null>(null);
  const [results, setResults] = useState<SemanticSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Check availability on mount
  useEffect(() => {
    let cancelled = false;
    getSearchStatus().then((res) => {
      if (!cancelled) {
        setAvailable(res.available);
      }
    });
    return () => { cancelled = true; };
  }, []);

  // Debounced search
  useEffect(() => {
    if (!available || !query.trim() || !drive) {
      setResults([]);
      return;
    }

    if (debounceRef.current) clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(async () => {
      const filterType = filter === "all" ? undefined : (filter as "video" | "image" | "audio" | "document");
      setLoading(true);
      try {
        const res = await semanticSearch(query.trim(), {
          limit: 20,
          type: filterType,
          drive,
        });
        setResults(res.results);
      } catch {
        setResults([]);
      }
      setLoading(false);
    }, 300);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query, drive, filter, available]);

  // Not available or not checked yet - render nothing
  if (available === null || !available) return null;

  // No query - nothing to show
  if (!query.trim()) return null;

  // Loading with no results yet
  if (loading && results.length === 0) {
    return (
      <div className="flex items-center justify-center py-4">
        <div className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
      </div>
    );
  }

  if (results.length === 0) return null;

  return (
    <>
      {results.map((result) => (
        <SemanticResultItem
          key={result.file_id}
          result={result}
          onSelect={onSelect}
          t={t}
        />
      ))}
      {loading && (
        <div className="flex items-center justify-center border-t border-bg-border py-3">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
        </div>
      )}
    </>
  );
}
