"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { Sparkles } from "lucide-react";

import { semanticSearch } from "./api";
import type { SemanticSearchResult, SemanticSearchSegment } from "./api";
import { getEnabledAddons } from "@/lib/addons";
import { formatDuration } from "@/lib/format";

type SlotContext = "popup" | "page";

interface SemanticSearchSlotProps {
  query: string;
  drive: string;
  filter: string;
  onSelect: (url: string) => void;
  /** Layout mode. "popup" (default) = compact list for the search modal,
   *  "page" = full-page grid for /drive/<name>/search. */
  context?: SlotContext;
}

// Popup-only badge palette. Phase 3 removed the page-context layout
// from this slot; the host's MatchOverlay (warm palette per
// DESIGN.md §2.2) is reused for the unified search list. The popup
// view sits inside the search modal with its own visual context, so
// the compact badge style is kept here.
// Semantic matches (vector similarity) get the accent-teal "positive
// signal" treatment; keyword and metadata matches use the neutral
// warm-light surface so the badge palette stays inside DESIGN.md §2.2.
const POPUP_MATCH_TYPE_STYLES: Record<string, string> = {
  transcript: "bg-accent-teal/15 text-accent-teal",
  transcript_keyword: "bg-warm-light text-text-muted",
  clip: "bg-accent-teal/15 text-accent-teal",
  metadata: "bg-warm-light text-text-muted",
  content: "bg-accent-teal/15 text-accent-teal",
  text_content_keyword: "bg-warm-light text-text-muted",
};

function PopupMatchBadge({ type, label }: { type: string; label: string }) {
  const style =
    POPUP_MATCH_TYPE_STYLES[type] ?? "bg-warm-light text-text-muted";
  return (
    <span
      className={`inline-flex rounded-lg px-1.5 py-0.5 text-[10px] font-medium ${style}`}
    >
      {label}
    </span>
  );
}

function PopupTimestampLink({
  seconds,
  fileId,
  onClick,
}: {
  seconds: number;
  fileId: string;
  onClick: (url: string) => void;
}) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick(`/files/${fileId}?t=${Math.floor(seconds)}`);
      }}
      className="rounded-lg px-1 py-0.5 text-[10px] font-medium text-accent transition-colors hover:bg-accent/10"
    >
      {formatDuration(seconds)}
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
        .map((m) => m.page as number),
    ),
  ].sort((a, b) => a - b);

  const timestamps = result.segments
    .filter(
      (seg): seg is SemanticSearchSegment & { time_range: [number, number] } =>
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
        className="h-10 w-16 flex-shrink-0 rounded-lg bg-bg-elevated object-cover"
        onError={(e) => {
          e.currentTarget.style.display = "none";
        }}
      />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-text-primary">{result.filename}</p>
        <div className="mt-0.5 flex flex-wrap items-center gap-1">
          {result.match_types.map((type) => (
            <PopupMatchBadge
              key={type}
              type={type}
              label={matchLabels[type] ?? type}
            />
          ))}
        </div>
        {timestamps.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-0.5">
            {timestamps.map((seg) => (
              <PopupTimestampLink
                key={seg.time_range[0]}
                seconds={seg.time_range[0]}
                fileId={result.file_id}
                onClick={onSelect}
              />
            ))}
          </div>
        )}
        {matchedPages.length > 0 && (
          <p className="mt-1 text-[11px] text-text-muted">
            {t("matchedPages", { pages: matchedPages.join(", ") })}
          </p>
        )}
      </div>
    </button>
  );
}

function PopupLayout({
  results,
  loading,
  query,
  drive,
  onSelect,
  t,
  askT,
}: {
  results: SemanticSearchResult[];
  loading: boolean;
  query: string;
  drive: string;
  onSelect: (url: string) => void;
  t: (key: string, values?: Record<string, string | number>) => string;
  askT: (key: string, values?: Record<string, string | number>) => string;
}) {
  const askHref = `/drive/${encodeURIComponent(drive)}/addons/intelligence?q=${encodeURIComponent(query)}`;

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
      <button
        type="button"
        onClick={() => onSelect(askHref)}
        className="flex w-full items-center gap-2 border-t border-bg-border px-4 py-2.5 text-left text-xs text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
      >
        <Sparkles size={12} className="flex-shrink-0 text-accent-teal" />
        <span className="truncate">{askT("button", { query })}</span>
      </button>
    </>
  );
}

export default function SemanticSearchSlot({
  query,
  drive,
  filter,
  onSelect,
  context = "popup",
}: SemanticSearchSlotProps) {
  const t = useTranslations("search");
  // Separate namespace for the Ask link label: reuses the already-
  // translated "「{query}」について AI で質問応答" string from the
  // askSearch namespace (formerly owned by the deleted AskSearchMode
  // slot) so we don't add a duplicate translation just for a button.
  const askT = useTranslations("askSearch");
  const [available, setAvailable] = useState<boolean | null>(null);
  const [results, setResults] = useState<SemanticSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Phase 3 retired the slot's page layout; the host now owns the
  // unified search-results list. Skip the availability probe and the
  // debounced fetch in page context — they would only burn cycles
  // duplicating what `useFolderFiles` already does.
  const isPageContext = context === "page";

  // Check availability on mount (rechecked when the active drive
  // changes). The probe asks the core's addon registry whether
  // intelligence is enabled for this drive — the addon's own /status
  // is admin-gated and would 403 for any viewer who has not unlocked
  // every protected drive, hiding semantic search even on drives the
  // viewer can fully access.
  useEffect(() => {
    if (isPageContext) return;
    if (!drive) {
      setAvailable(false);
      return;
    }
    let cancelled = false;
    getEnabledAddons(drive).then((addons) => {
      if (!cancelled) {
        setAvailable(Boolean(addons["intelligence"]));
      }
    });
    return () => {
      cancelled = true;
    };
  }, [drive, isPageContext]);

  // Debounced search
  useEffect(() => {
    if (isPageContext) return;
    if (!available || !query.trim() || !drive) {
      setResults([]);
      return;
    }

    if (debounceRef.current) clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(async () => {
      const filterType =
        filter === "all"
          ? undefined
          : (filter as "video" | "image" | "audio" | "document");
      setLoading(true);
      try {
        const res = await semanticSearch(query.trim(), drive, {
          limit: 20,
          type: filterType,
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
  }, [query, drive, filter, available, isPageContext]);

  // Page context renders nothing — see comment at the top of
  // `isPageContext` above. The merged list lives in core.
  if (isPageContext) return null;

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

  const trimmedQuery = query.trim();

  return (
    <PopupLayout
      results={results}
      loading={loading}
      query={trimmedQuery}
      drive={drive}
      onSelect={onSelect}
      t={t}
      askT={askT}
    />
  );
}
